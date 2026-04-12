from __future__ import annotations

import argparse
from math import gamma
from pathlib import Path

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.scipy.stats
from stochastax.controls.drivers import (
    bm_driver,
    correlate_bm_driver_against_reference,
    riemann_liouville_driver,
)

from roughbench.generate_data.utils import (
    config_section,
    create_figure,
    decorate_axes,
    finalize_plot,
    load_config,
    plotting_context,
    save_npz,
    save_plot,
)
from roughbench.rde.rough_volatility import (
    BonesiniModelSpec,
    get_bonesini_noise_drivers,
    make_bergomi_model_spec,
    make_black_scholes_model_spec,
    make_classical_local_stochastic_volatility_model_spec,
    make_heston_model_spec,
    make_quadratic_rough_heston_model_spec,
    make_rough_bergomi_model_spec,
    make_rough_heston_model_spec,
    simulate_quadratic_rough_heston_variance_paths,
    simulate_rough_heston_variance_paths,
    solve_quadratic_rough_heston_from_variance_path,
    solve_rough_heston_from_variance_path,
    solve_wong_zakai,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rough-volatility Monte Carlo paths from a TOML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rough_volatility/rBergomi.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional override for random seed."
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=None,
        help="Optional override for number of Monte Carlo paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional override base data dir.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable diagnostic plotting."
    )
    return parser.parse_args()


def _normalize_model_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _infer_model_name(config_path: str, config: dict[str, object]) -> str:
    model_section = config_section(config, "model")
    general = config_section(config, "general")
    family = model_section.get("family", general.get("model"))
    if family is None:
        family = Path(config_path).stem
    return _normalize_model_name(str(family))


def _simulate_internal_model(
    *,
    model_spec: BonesiniModelSpec,
    num_paths: int,
    noise_timesteps: int,
    rde_timesteps: int,
    s_0: float,
    seed: int,
) -> tuple[dfx.Solution, jax.Array, jax.Array]:
    keys = jax.random.split(jax.random.PRNGKey(seed), num_paths)
    y0s, Xs, Ws = jax.vmap(
        lambda key: get_bonesini_noise_drivers(
            key, noise_timesteps, model_spec, s_0=s_0
        )
    )(keys)
    solutions = jax.vmap(
        lambda y0, X, W: solve_wong_zakai(
            y0, X, W, model_spec, noise_timesteps, rde_timesteps
        )
    )(y0s, Xs, Ws)
    return solutions, Xs, Ws


def _build_external_variance_and_price_paths(
    *,
    num_paths: int,
    seed: int,
    noise_timesteps: int,
    v_0: float,
    hurst: float,
    rho: float | None,
    external_cfg: dict[str, object],
) -> tuple[jax.Array, jax.Array]:
    mode = _normalize_model_name(
        str(external_cfg.get("mode", "rough_lognormal_surrogate"))
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), num_paths)
    ts = jnp.linspace(0.0, 1.0, noise_timesteps + 1)

    def _brownian_price_path(key: jax.Array) -> jax.Array:
        return jnp.squeeze(bm_driver(key, noise_timesteps, 1).path)

    if mode == "constant":
        W_keys = jax.random.split(jax.random.PRNGKey(seed + 17), num_paths)
        Ws = jax.vmap(_brownian_price_path)(W_keys)
        Xs = jnp.full((num_paths, noise_timesteps + 1), v_0, dtype=Ws.dtype)
        return Xs, Ws

    if mode == "linear":
        slope = float(external_cfg.get("linear_slope", 0.02))
        W_keys = jax.random.split(jax.random.PRNGKey(seed + 17), num_paths)
        Ws = jax.vmap(_brownian_price_path)(W_keys)
        base = v_0 + slope * ts
        Xs = jnp.broadcast_to(base[None, :], (num_paths, noise_timesteps + 1))
        return Xs, Ws

    if mode == "sinusoidal":
        amplitude = float(external_cfg.get("amplitude", 0.01))
        W_keys = jax.random.split(jax.random.PRNGKey(seed + 17), num_paths)
        Ws = jax.vmap(_brownian_price_path)(W_keys)
        base = v_0 + amplitude * jnp.sin(jnp.pi * ts)
        Xs = jnp.broadcast_to(base[None, :], (num_paths, noise_timesteps + 1))
        return Xs, Ws

    if mode != "rough_lognormal_surrogate":
        raise ValueError(f"Unknown external variance mode: {mode}")

    surrogate_scale = float(
        external_cfg.get("surrogate_scale", external_cfg.get("nu", 0.5))
    )
    floor = float(external_cfg.get("variance_floor", 1e-6))
    rho_eff = float(rho if rho is not None else 0.0)
    gamma_h = gamma(hurst + 0.5)
    variance_norm = (ts ** (2.0 * hurst)) / (gamma_h**2)

    def _single(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key_w, key_b, key_v = jax.random.split(key, 3)
        W_path = bm_driver(key_w, noise_timesteps, 1)
        W = jnp.squeeze(W_path.path)
        B_path = bm_driver(key_b, noise_timesteps, 1)
        correlated = correlate_bm_driver_against_reference(W_path, B_path, rho_eff)
        G = jnp.squeeze(
            riemann_liouville_driver(key_v, noise_timesteps, hurst, correlated).path
        ) / gamma_h
        variance = v_0 * jnp.exp(
            surrogate_scale * G - 0.5 * (surrogate_scale**2) * variance_norm
        )
        return jnp.maximum(variance, floor), W

    Xs, Ws = jax.vmap(_single)(keys)
    return Xs, Ws


def _simulate_external_variance_model(
    *,
    model_name: str,
    num_paths: int,
    seed: int,
    noise_timesteps: int,
    rde_timesteps: int,
    s_0: float,
    params: dict[str, object],
    external_cfg: dict[str, object],
) -> tuple[dfx.Solution, jax.Array, jax.Array, BonesiniModelSpec]:
    v_0 = float(params["v_0"])
    hurst = float(params["hurst"])
    mode = _normalize_model_name(
        str(external_cfg.get("mode", "rough_lognormal_surrogate"))
    )
    if mode == "volterra_simulated":
        if model_name == "rough_heston":
            Xs, Ws = simulate_rough_heston_variance_paths(
                jax.random.PRNGKey(seed),
                num_paths=num_paths,
                noise_timesteps=noise_timesteps,
                v_0=v_0,
                hurst=hurst,
                nu=float(params["nu"]),
                rho=float(params["rho"]),
                lambda_=float(params["lambda_"]),
                v_bar=float(params["v_bar"]),
                truncation_eps=float(external_cfg.get("truncation_eps", 1e-8)),
            )
        elif model_name == "quadratic_rough_heston":
            Xs, Ws = simulate_quadratic_rough_heston_variance_paths(
                jax.random.PRNGKey(seed),
                num_paths=num_paths,
                noise_timesteps=noise_timesteps,
                v_0=v_0,
                hurst=hurst,
                a=float(params["a"]),
                b=float(params["b"]),
                c=float(params["c"]),
                lambda_=float(params["lambda_"]),
                eta=float(params["eta"]),
                theta_level=float(external_cfg["theta_level"])
                if "theta_level" in external_cfg
                else None,
                truncation_eps=float(external_cfg.get("truncation_eps", 1e-8)),
            )
        else:
            raise ValueError(f"Unsupported external-variance model: {model_name}")
    else:
        Xs, Ws = _build_external_variance_and_price_paths(
            num_paths=num_paths,
            seed=seed,
            noise_timesteps=noise_timesteps,
            v_0=v_0,
            hurst=hurst,
            rho=float(params["rho"]) if "rho" in params else None,
            external_cfg=external_cfg,
        )

    if model_name == "rough_heston":
        spec = make_rough_heston_model_spec(
            v_0=v_0,
            hurst=hurst,
            nu=float(params["nu"]),
            rho=float(params["rho"]),
            lambda_=float(params["lambda_"]),
            v_bar=float(params["v_bar"]),
        )
        solve_vmap = jax.vmap(
            lambda X, W: solve_rough_heston_from_variance_path(
                s_0=s_0,
                variance_path=X,
                price_brownian_path=W,
                v_0=v_0,
                hurst=hurst,
                nu=float(params["nu"]),
                rho=float(params["rho"]),
                lambda_=float(params["lambda_"]),
                v_bar=float(params["v_bar"]),
                noise_timesteps=noise_timesteps,
                rde_timesteps=rde_timesteps,
            )
        )
    elif model_name == "quadratic_rough_heston":
        spec = make_quadratic_rough_heston_model_spec(
            v_0=v_0,
            hurst=hurst,
            a=float(params["a"]),
            b=float(params["b"]),
            c=float(params["c"]),
            lambda_=float(params["lambda_"]),
            eta=float(params["eta"]),
        )
        solve_vmap = jax.vmap(
            lambda X, W: solve_quadratic_rough_heston_from_variance_path(
                s_0=s_0,
                variance_path=X,
                price_brownian_path=W,
                v_0=v_0,
                hurst=hurst,
                a=float(params["a"]),
                b=float(params["b"]),
                c=float(params["c"]),
                lambda_=float(params["lambda_"]),
                eta=float(params["eta"]),
                noise_timesteps=noise_timesteps,
                rde_timesteps=rde_timesteps,
            )
        )
    else:
        raise ValueError(f"Unsupported external-variance model: {model_name}")

    solutions = solve_vmap(Xs, Ws)
    return solutions, Xs, Ws, spec


def _build_model_spec(
    model_name: str, params: dict[str, object]
) -> tuple[BonesiniModelSpec, bool]:
    if model_name == "black_scholes":
        return make_black_scholes_model_spec(v_0=float(params["v_0"])), False
    if model_name == "bergomi":
        return make_bergomi_model_spec(
            v_0=float(params["v_0"]), rho=float(params["rho"])
        ), False
    if model_name in {"rbergomi", "rough_bergomi"}:
        return make_rough_bergomi_model_spec(
            v_0=float(params["v_0"]),
            nu=float(params["nu"]),
            hurst=float(params["hurst"]),
            rho=float(params["rho"]),
        ), False
    if model_name == "heston":
        return make_heston_model_spec(
            v_0=float(params["v_0"]),
            rho=float(params["rho"]),
            nu=float(params["nu"]),
            lambda_=float(params["lambda_"]),
            v_bar=float(params["v_bar"]),
        ), False
    if model_name == "classical_local_stochastic_volatility":
        xi_0 = float(params["xi_0"])
        xi_s = float(params.get("xi_s", 0.0))
        xi_v = float(params.get("xi_v", 0.0))
        f2_0 = float(params["f2_0"])
        f2_1 = float(params.get("f2_1", 0.0))
        lambda_ = float(params["lambda_"])
        v_bar = float(params["v_bar"])
        return make_classical_local_stochastic_volatility_model_spec(
            v_0=float(params["v_0"]),
            rho=float(params["rho"]),
            xi=lambda t, s, v: xi_0 + xi_s * s + xi_v * v,
            f_1=lambda t, v: -lambda_ * (v - v_bar),
            f_2=lambda v: f2_0 + f2_1 * v,
        ), False
    if model_name == "rough_heston":
        return make_rough_heston_model_spec(
            v_0=float(params["v_0"]),
            hurst=float(params["hurst"]),
            nu=float(params["nu"]),
            rho=float(params["rho"]),
            lambda_=float(params["lambda_"]),
            v_bar=float(params["v_bar"]),
        ), True
    if model_name == "quadratic_rough_heston":
        return make_quadratic_rough_heston_model_spec(
            v_0=float(params["v_0"]),
            hurst=float(params["hurst"]),
            a=float(params["a"]),
            b=float(params["b"]),
            c=float(params["c"]),
            lambda_=float(params["lambda_"]),
            eta=float(params["eta"]),
        ), True
    raise ValueError(f"Unknown rough-volatility model: {model_name}")


def plot_bonesini_monte_carlo(
    solution: dfx.Solution,
    model_spec: BonesiniModelSpec,
    X_drivers: jax.Array | None = None,
    W_drivers: jax.Array | None = None,
    plot_variance: bool = False,
    use_log_price: bool = True,
    output_dir: Path | None = None,
) -> None:
    ax_main = None
    ax_marginal = None
    with plotting_context(font_scale=1.1) as plt:
        _, axs = create_figure(
            nrows=1,
            ncols=2,
            figsize=(12.0, 6.0),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        try:
            ax_main, ax_marginal = axs  # type: ignore
        except Exception:
            try:
                ax_main = axs[0]  # type: ignore
                ax_marginal = axs[1]  # type: ignore
            except Exception:
                ax_main = axs  # type: ignore

    ts_paths = jnp.asarray(solution.ts)
    ys_paths = jnp.asarray(solution.ys)
    final_values = []
    initial_values = []
    ax_var = None

    for i in range(ts_paths.shape[0]):
        ts = ts_paths[i]
        ys = ys_paths[i]
        S = ys if ys.ndim == 1 else ys[:, 0]
        price_data = jnp.log(S) if use_log_price else S
        if ax_main is not None:
            ax_main.plot(ts, price_data, color="gray", alpha=0.6)
        final_values.append(price_data[-1])
        initial_values.append(price_data[0])

        if ax_main is not None and plot_variance and ys.ndim > 1 and ys.shape[1] > 1:
            if ax_var is None:
                ax_var = ax_main.twinx()
                ax_var.set_ylabel("Volatility State", color="tab:green")
                ax_var.tick_params(axis="y", labelcolor="tab:green")
            ax_var.plot(ts, ys[:, 1], color="tab:green", alpha=0.3)

    mean_initial = float(jnp.mean(jnp.array(initial_values)))
    mean_final = float(jnp.mean(jnp.array(final_values)))

    if ax_main is not None:
        price_label = "log" if use_log_price else "price"
        ax_main.axhline(
            y=mean_initial,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"t=0 Mean ({price_label}): {mean_initial:.4f}",
        )
        ax_main.axhline(
            y=mean_final,
            color="blue",
            linestyle="--",
            alpha=0.8,
            label=f"t=1 Mean ({price_label}): {mean_final:.4f}",
        )
        decorate_axes(
            ax_main,
            title=f"{model_spec.name} Monte Carlo",
            xlabel="Time",
            ylabel=("Log-Price" if use_log_price else "Price"),
            legend=True,
        )

    if ax_marginal is not None and final_values:
        ax_marginal.hist(
            final_values,
            bins=30,
            orientation="horizontal",
            color="gray",
            alpha=0.7,
            density=True,
        )
        final_array = jnp.array(final_values)
        mean_val = jnp.mean(final_array)
        std_val = jnp.std(final_array)
        y_range = jnp.linspace(mean_val - 3 * std_val, mean_val + 3 * std_val, 100)
        normal_pdf = jax.scipy.stats.norm.pdf(y_range, mean_val, std_val)
        ax_marginal.plot(
            normal_pdf,
            y_range,
            color="red",
            linestyle="--",
            alpha=0.8,
            label="Normal",
        )
        decorate_axes(
            ax_marginal,
            title=f"t=1 Marginal ({'log' if use_log_price else 'price'})",
            xlabel="Density",
            ylabel=("Log-Price" if use_log_price else "Price"),
            legend=True,
        )
        if ax_main is not None:
            y_min, y_max = ax_main.get_ylim()
            ax_marginal.set_ylim(y_min, y_max)

    finalize_plot(tight_layout=True)
    save_plot(
        filename=f"{model_spec.name.lower().replace(' ', '_')}_monte_carlo.png",
        subdir="rough_volatility",
        data_dir=output_dir,
        dpi=200,
    )

    drivers = (
        jnp.stack([X_drivers, W_drivers], axis=-1)
        if X_drivers is not None and W_drivers is not None
        else jnp.zeros((ys_paths.shape[0], 1, 2))
    )
    save_npz(
        filename=f"{model_spec.name.lower().replace(' ', '_')}_data.npz",
        subdir="rough_volatility",
        data_dir=output_dir,
        solution=ys_paths,
        driver=drivers,
    )


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    general = config_section(config, "general")
    params = config_section(config, "parameters")
    external_cfg = config_section(config, "external_variance")

    model_name = _infer_model_name(args.config, config)
    num_paths = int(args.num_paths or general.get("num_paths", 128))
    seed = int(args.seed if args.seed is not None else general.get("seed", 42))
    noise_timesteps = int(params.get("noise_timesteps", 128))
    rde_timesteps = int(params.get("rde_timesteps", 256))
    s_0 = float(params.get("s_0", 1.0))
    use_log_price = bool(general.get("log_price", True))
    plot_variance = bool(general.get("plot_variance", True))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    model_spec, requires_external_variance = _build_model_spec(model_name, params)

    print(f"Generating {model_spec.name} Monte Carlo...")
    if requires_external_variance:
        if external_cfg.get("mode") is None:
            print(
                "Using built-in surrogate external variance path. "
                "This runs end-to-end but is not a law-exact Volterra variance simulator."
            )
        solutions, Xs, Ws, model_spec = _simulate_external_variance_model(
            model_name=model_name,
            num_paths=num_paths,
            seed=seed,
            noise_timesteps=noise_timesteps,
            rde_timesteps=rde_timesteps,
            s_0=s_0,
            params=params,
            external_cfg=external_cfg,
        )
    else:
        solutions, Xs, Ws = _simulate_internal_model(
            model_spec=model_spec,
            num_paths=num_paths,
            noise_timesteps=noise_timesteps,
            rde_timesteps=rde_timesteps,
            s_0=s_0,
            seed=seed,
        )

    if not bool(args.no_plot):
        plot_bonesini_monte_carlo(
            solutions,
            model_spec,
            X_drivers=Xs,
            W_drivers=Ws,
            plot_variance=plot_variance,
            use_log_price=use_log_price,
            output_dir=output_dir,
        )
    else:
        drivers = jnp.stack([Xs, Ws], axis=-1)
        save_npz(
            filename=f"{model_spec.name.lower().replace(' ', '_')}_data.npz",
            subdir="rough_volatility",
            data_dir=output_dir,
            solution=jnp.asarray(solutions.ys),
            driver=drivers,
        )

    print("")
    print(
        "All rough-volatility outputs saved under data/rough_volatility and mirrored plots under docs/rde_bench/rough_volatility"
    )


if __name__ == "__main__":
    main()
 
