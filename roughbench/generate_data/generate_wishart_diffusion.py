"""
Generate synthetic Wishart diffusion data.

CLI usage (defaults to configs/synthetic_diffusions/wishart_diffusion.toml):
    uv run python -m roughbench.generate_data.generate_wishart_diffusion --config <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from stochastax.manifolds.spd import SPDManifold
from roughbench.manifold_rde.synthetic_wishart_diffusion import (
    make_wishart_parameters,
    simulate_wishart_diffusion,
)
from roughbench.generate_data.utils import (
    resolve_output_dirs,
    save_plot,
    plotting_context,
    create_figure,
    decorate_axes,
    finalize_plot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Wishart diffusion trajectories.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic_diffusions/wishart_diffusion.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional override seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override batch size.")
    parser.add_argument("--timesteps", type=int, default=None, help="Optional override timesteps.")
    parser.add_argument("--T", type=float, default=None, help="Optional override horizon.")
    parser.add_argument("--subdir", type=str, default=None, help="Optional override data subdir.")
    parser.add_argument("--output-dir", type=str, default="", help="Optional override base data dir.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Generate in smaller chunks to avoid OOM (e.g. 256).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable diagnostic plotting.")
    return parser.parse_args()


def _load_config(path: str) -> dict[str, object]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _section(config: dict[str, object], key: str) -> dict[str, object]:
    section = config.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a table.")
    return section


def _get_int(section: dict[str, object], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be an int.")
    if not isinstance(value, int):
        raise ValueError(f"Config value '{key}' must be an int.")
    return int(value)


def _get_float(section: dict[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be a float.")
    if not isinstance(value, int | float):
        raise ValueError(f"Config value '{key}' must be a float.")
    return float(value)


def _get_str(section: dict[str, object], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Config value '{key}' must be a string.")
    return value


def _get_bool(section: dict[str, object], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be a boolean.")
    return bool(value)


def _maybe_identity_corr_matrix(value: object, dim: int, atol: float = 1e-6) -> jax.Array | None:
    if value is None:
        return None
    dense = np.asarray(value, dtype=np.float32)
    num_paths = dim * (dim + 1) // 2
    if dense.shape != (num_paths, num_paths):
        raise ValueError(f"corr_matrix must have shape ({num_paths}, {num_paths}). Got {dense.shape}.")
    if np.allclose(dense, np.eye(num_paths, dtype=np.float32), atol=atol, rtol=0.0):
        return None
    return jnp.asarray(dense, dtype=jnp.float32)


def _to_array(value: object, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    return jnp.asarray(value, dtype=jnp.float32)


def _plot_trace(
    *,
    ts: jax.Array,
    X_paths: jax.Array,
    subdir: str,
    output_dir: Path | None,
    max_paths: int = 64,
) -> None:
    # X_paths is vech(X): (..., 6) for d=3. Diagonal entries are [X00, X11, X22] at vech indices [0, 2, 5].
    if X_paths.ndim != 3:
        raise ValueError(f"Expected X_paths shaped (B,T,C), got {X_paths.shape}.")
    if int(X_paths.shape[-1]) < 3:
        raise ValueError(f"Expected at least 3 diagonal entries in vech, got {X_paths.shape[-1]}.")
    diag_idx = jnp.asarray([0, 2, 5], dtype=jnp.int32) if int(X_paths.shape[-1]) == 6 else None
    if diag_idx is None:
        # Fallback: reconstruct matrices if not 3x3 vech.
        mats = SPDManifold.unvech(X_paths)  # (B,T,d,d)
        traces = jnp.trace(mats, axis1=-2, axis2=-1)
    else:
        traces = jnp.sum(jnp.take(X_paths, diag_idx, axis=-1), axis=-1)
    traces_np = np.asarray(traces)
    ts_np = np.asarray(ts)

    num_paths = traces_np.shape[0]
    k = min(max_paths, num_paths)
    idx = np.linspace(0, num_paths - 1, k, dtype=int)

    with plotting_context(font_scale=1.1):
        _, ax = create_figure(figsize=(10.0, 6.0))
        for i in idx:
            ax.plot(ts_np, traces_np[i], color="tab:blue", alpha=0.3, linewidth=0.8)
        decorate_axes(ax, title="Wishart diffusion trace", xlabel="Time", ylabel="Trace", legend=False)
        finalize_plot(tight_layout=True)
    save_plot(filename="wishart_diffusion_trace.png", subdir=subdir, data_dir=output_dir, dpi=200)


def _plot_eigenvalue_trajectories(
    *,
    ts: jax.Array,
    X_paths: jax.Array,
    subdir: str,
    output_dir: Path | None,
    max_paths: int | None = 256,
) -> None:
    if X_paths.ndim != 3:
        raise ValueError(f"Expected X_paths shaped (B,T,C), got {X_paths.shape}.")

    num_paths = int(X_paths.shape[0])
    if num_paths <= 0:
        raise ValueError("X_paths must contain at least one path.")

    if max_paths is None or int(max_paths) <= 0:
        idx = np.arange(num_paths, dtype=int)
    else:
        k = min(int(max_paths), num_paths)
        idx = np.linspace(0, num_paths - 1, k, dtype=int)

    mats = SPDManifold.unvech(X_paths[idx])  # (K,T,d,d)
    evals = jnp.linalg.eigvalsh(mats)  # (K,T,d), sorted ascending

    ts_np = np.asarray(ts)
    evals_np = np.asarray(evals)
    dim = int(evals_np.shape[-1])
    k = int(evals_np.shape[0])

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    with plotting_context(font_scale=1.1):
        _, ax = create_figure(figsize=(10.0, 6.0))
        alpha = 0.15 if k > 8 else 0.6
        lw = 0.8 if k > 8 else 1.3
        for j in range(dim):
            # Plot all selected paths for eigenvalue j with same color.
            for p in range(k):
                ax.plot(
                    ts_np,
                    evals_np[p, :, j],
                    color=colors[j % len(colors)],
                    alpha=alpha,
                    linewidth=lw,
                    label=(rf"$\lambda_{{{j + 1}}}(t)$" if p == 0 else None),
                )
        decorate_axes(
            ax,
            title="Wishart diffusion eigenvalue trajectories",
            xlabel="Time",
            ylabel="Eigenvalue",
            legend=True,
            legend_loc="best",
        )
        ax.set_ylim(0.0, 8.0)
        finalize_plot(tight_layout=True)
    save_plot(filename="wishart_diffusion_eigenvalues.png", subdir=subdir, data_dir=output_dir, dpi=200)


def generate_wishart_diffusion_batch(
    *,
    key: jax.Array,
    batch_size: int,
    timesteps: int,
    T: float,
    X0: jax.Array,
    Sigma: jax.Array,
    gamma: float,
    A: jax.Array,
    eps: float,
    corr_matrix: jax.Array | None,
    tol: float,
    noise_scale: float,
) -> dict[str, jax.Array]:
    params = make_wishart_parameters(Sigma=Sigma, gamma=gamma, A=A, eps=eps)
    b = params["b"]
    H = params["H"]
    ts = jnp.linspace(0.0, float(T), timesteps)

    keys = jax.random.split(key, batch_size)

    def _simulate(single_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = simulate_wishart_diffusion(
            key=single_key,
            timesteps=timesteps,
            T=T,
            X0=X0,
            b=b,
            H=H,
            Sigma=Sigma,
            corr_matrix=corr_matrix,
            tol=tol,
            eps=eps,
            noise_scale=noise_scale,
        )
        X = result["X_path"]  # (T,d,d)
        qv_vech = result["quadratic_variation"]  # (T-1,m,m) per-step increments
        t = int(X.shape[0])
        d = int(X.shape[-1])

        # vech ordering must match SPDManifold.vech
        X_flat = X.reshape((t,) + (d, d))
        vech = SPDManifold.vech(X_flat)  # (T,m)

        return vech, qv_vech

    vech_paths, qv_vech_paths = jax.vmap(_simulate)(keys)
    return {"ts": ts, "solution": vech_paths, "quadratic_variation": qv_vech_paths}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _write_final_npz_from_npy(
    *,
    target: Path,
    ts_npy: Path,
    solution_npy: Path,
    quadratic_variation_npy: Path,
) -> None:
    # An .npz is just a zip of .npy members. Zip without compression for speed and streaming.
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(ts_npy, arcname="ts.npy")
        zf.write(solution_npy, arcname="solution.npy")
        zf.write(quadratic_variation_npy, arcname="quadratic_variation.npy")


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)

    sim = _section(config, "simulation")
    params = _section(config, "parameters")
    corr = _section(config, "correlation")
    output = _section(config, "output")

    Sigma = _to_array(params.get("Sigma"), "parameters.Sigma")
    dim = _get_int(sim, "d", int(Sigma.shape[0]))
    if Sigma.shape != (dim, dim):
        raise ValueError(f"Sigma must have shape ({dim}, {dim}). Got {Sigma.shape}.")

    A = _to_array(params.get("A", jnp.zeros((dim, dim), dtype=jnp.float32)), "parameters.A")
    if A.shape != (dim, dim):
        raise ValueError(f"A must have shape ({dim}, {dim}). Got {A.shape}.")

    gamma = _get_float(params, "gamma", 1.0)
    eps = _get_float(params, "eps", 1e-6)
    timesteps = _get_int(sim, "timesteps", 1024)
    T = _get_float(sim, "T", 1.0)
    batch_size = _get_int(sim, "batch_size", 256)
    tol = _get_float(sim, "tol", 1e-3)
    chunk_size = _get_int(sim, "chunk_size", 256)
    noise_scale = _get_float(sim, "noise_scale", 1.0)

    seed = _get_int(sim, "seed", 0)
    if args.seed is not None:
        seed = int(args.seed)
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    if args.timesteps is not None:
        timesteps = int(args.timesteps)
    if args.T is not None:
        T = float(args.T)
    if args.chunk_size is not None:
        chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    corr_matrix = _maybe_identity_corr_matrix(corr.get("corr_matrix", None), dim)

    X0_value = params.get("X0", None)
    X0 = jnp.eye(dim, dtype=jnp.float32) if X0_value is None else _to_array(X0_value, "parameters.X0")
    if X0.shape != (dim, dim):
        raise ValueError(f"X0 must have shape ({dim}, {dim}). Got {X0.shape}.")

    subdir = _get_str(output, "subdir", "synthetic_diffusions")
    filename = _get_str(output, "filename", "wishart_diffusion_data.npz")
    if args.subdir is not None:
        subdir = str(args.subdir)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    key = jax.random.PRNGKey(seed)

    data_path, _ = resolve_output_dirs(subdir=subdir, data_dir=output_dir)
    target = data_path / filename

    ts_npy = data_path / f"{target.stem}__ts.npy"
    solution_npy = data_path / f"{target.stem}__solution.npy"
    qv_npy = data_path / f"{target.stem}__quadratic_variation.npy"

    m = dim * (dim + 1) // 2
    solution_mm = np.lib.format.open_memmap(
        solution_npy, mode="w+", dtype=np.float32, shape=(batch_size, timesteps, m)
    )
    qv_mm = np.lib.format.open_memmap(
        qv_npy, mode="w+", dtype=np.float32, shape=(batch_size, timesteps - 1, m, m)
    )

    num_chunks = _ceil_div(batch_size, chunk_size)
    ts_np: np.ndarray | None = None

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        stop = min(start + chunk_size, batch_size)
        this_chunk = stop - start

        chunk_key = jax.random.fold_in(key, chunk_idx)
        payload = generate_wishart_diffusion_batch(
            key=chunk_key,
            batch_size=this_chunk,
            timesteps=timesteps,
            T=T,
            X0=X0,
            Sigma=Sigma,
            gamma=gamma,
            A=A,
            eps=eps,
            corr_matrix=corr_matrix,
            tol=tol,
            noise_scale=noise_scale,
        )

        if ts_np is None:
            ts_np = np.asarray(payload["ts"], dtype=np.float32)

        solution_mm[start:stop] = np.asarray(payload["solution"], dtype=np.float32)
        qv_mm[start:stop] = np.asarray(payload["quadratic_variation"], dtype=np.float32)
        solution_mm.flush()
        qv_mm.flush()

        print(f"Wrote chunk {chunk_idx + 1}/{num_chunks}: rows {start}:{stop}")

    if ts_np is None:
        raise RuntimeError("No data generated.")

    np.save(ts_npy, ts_np)
    _write_final_npz_from_npy(
        target=target,
        ts_npy=ts_npy,
        solution_npy=solution_npy,
        quadratic_variation_npy=qv_npy,
    )
    print(f"Saved dataset to {target}")

    print("")
    print("Generated arrays:")
    print(f"  ts: shape={ts_np.shape}, dtype={ts_np.dtype}")
    print(f"  solution: shape={solution_mm.shape}, dtype={solution_mm.dtype}")
    print(f"  quadratic_variation: shape={qv_mm.shape}, dtype={qv_mm.dtype}")

    if not bool(args.no_plot) and _get_bool(output, "plot", True):
        k = min(256, batch_size)
        _plot_trace(
            ts=jnp.asarray(ts_np),
            X_paths=jnp.asarray(solution_mm[:k]),
            subdir=subdir,
            output_dir=output_dir,
        )
        _plot_eigenvalue_trajectories(
            ts=jnp.asarray(ts_np),
            X_paths=jnp.asarray(solution_mm[:k]),
            subdir=subdir,
            output_dir=output_dir,
            max_paths=256,
        )


if __name__ == "__main__":
    main()
