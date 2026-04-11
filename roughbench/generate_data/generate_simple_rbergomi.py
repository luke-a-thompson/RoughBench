import argparse
from pathlib import Path

import numpy as np

from roughbench.rde.simple_rbergomi import rBergomi
from utils import (
    save_plot,
    save_npz,
    plotting_context,
    create_figure,
    decorate_axes,
    finalize_plot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simple rBergomi Monte Carlo data and save as .npz."
    )
    parser.add_argument("--n", type=int, default=512, help="Steps per year.")
    parser.add_argument("--T", type=float, default=1.0, help="Maturity / time horizon.")
    parser.add_argument(
        "--N", type=int, default=32768, help="Number of Monte Carlo paths."
    )

    parser.add_argument(
        "--a",
        type=float,
        default=-0.4,
        help="Alpha parameter (H = a + 0.5). Requires 2*a+1>0.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=-0.848,
        help="Correlation between variance and price Brownian motions.",
    )
    parser.add_argument(
        "--eta", type=float, default=1.991, help="Vol-of-vol parameter eta."
    )
    parser.add_argument(
        "--xi", type=float, default=0.04, help="Forward variance level xi (e.g. v0)."
    )
    parser.add_argument("--S0", type=float, default=1.0, help="Initial price.")

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for NumPy RNG."
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="rough_volatility",
        help="Subdir under data/ to save into.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable saving diagnostic plots."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional override base data dir (defaults to <repo>/data/<subdir>).",
    )
    return parser.parse_args()


def generate_simple_rbergomi_data(
    *,
    n: int,
    T: float,
    N: int,
    a: float,
    rho: float,
    eta: float,
    xi: float,
    S0: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Generate simple rBergomi price paths and instantaneous variance.

    Returns a dict with keys:
      - dt: scalar timestep size
      - price: S_t on the simulator grid, shape (N, s)
      - log_price: log(S_t), shape (N, s)
      - variance: V_t (instantaneous variance), shape (N, s)
      - driver: price Brownian driver B_t, shape (N, s, 1)
    """
    np.random.seed(seed)
    rb = rBergomi(n=n, N=N, T=T, a=a, rho=rho, eta=eta, xi=xi)

    dW1 = rb.dW1()
    dW2 = rb.dW2()
    dB = rb.dB(dW1, dW2)  # uses rb.rho by default

    Y = rb.Y(dW1)
    V = rb.V(Y)  # V_t = xi * exp(eta Y_t - 0.5 eta^2 t^(2a+1))
    S = rb.S(V, dB, S0=S0)
    log_S = np.log(S)

    # Price Brownian path B_t, with B_0 = 0 and increments dB on each step.
    B = np.zeros_like(S, dtype=np.float64)
    B[:, 1:] = np.cumsum(dB, axis=1)

    return {
        "dt": np.asarray(rb.dt, dtype=np.float32),
        "price": np.asarray(S, dtype=np.float32),
        "log_price": np.asarray(log_S, dtype=np.float32),
        "variance": np.asarray(V, dtype=np.float32),
        "driver": np.asarray(B[:, :, np.newaxis], dtype=np.float32),
    }


def _save_plots(
    *,
    payload: dict[str, np.ndarray],
    subdir: str,
    output_dir: Path | None,
    max_paths: int = 128,
) -> None:
    price = payload["price"]
    log_price = payload["log_price"]
    variance = payload["variance"]

    num_paths = int(price.shape[0])
    s = int(price.shape[1])
    ts = payload["dt"].item() * np.arange(s, dtype=np.float32)

    k = min(max_paths, num_paths)
    idx = np.linspace(0, num_paths - 1, k, dtype=int)

    with plotting_context(font_scale=1.1):
        _, ax = create_figure(figsize=(10.0, 6.0))
        for i in idx:
            ax.plot(ts, log_price[i], color="gray", alpha=0.5, linewidth=0.8)
        decorate_axes(
            ax,
            title="Simple rBergomi Monte Carlo (log-price)",
            xlabel="Time",
            ylabel="Log-Price",
            legend=False,
        )
        finalize_plot(tight_layout=True)
    save_plot(
        filename="simple_rbergomi_log_price_monte_carlo.png",
        subdir=subdir,
        data_dir=output_dir,
        dpi=200,
    )

    with plotting_context(font_scale=1.1):
        _, ax = create_figure(figsize=(10.0, 6.0))
        for i in idx:
            ax.plot(ts, variance[i], color="tab:green", alpha=0.25, linewidth=0.8)
        decorate_axes(
            ax,
            title="Simple rBergomi Monte Carlo (variance)",
            xlabel="Time",
            ylabel="Instantaneous variance",
            legend=False,
        )
        finalize_plot(tight_layout=True)
    save_plot(
        filename="simple_rbergomi_variance_monte_carlo.png",
        subdir=subdir,
        data_dir=output_dir,
        dpi=200,
    )


if __name__ == "__main__":
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    filename = "simple_rbergomi_data.npz"

    payload = generate_simple_rbergomi_data(
        n=args.n,
        T=args.T,
        N=args.N,
        a=args.a,
        rho=args.rho,
        eta=args.eta,
        xi=args.xi,
        S0=args.S0,
        seed=args.seed,
    )

    save_npz(
        filename,
        subdir=args.subdir,
        data_dir=output_dir,
        dt=payload["dt"],
        price=payload["price"],
        log_price=payload["log_price"],
        variance=payload["variance"],
        driver=payload["driver"],
    )

    print("")
    print("Generated arrays:")
    print(f"  dt: shape={payload['dt'].shape}, dtype={payload['dt'].dtype}")
    print(f"  price: shape={payload['price'].shape}, dtype={payload['price'].dtype}")
    print(
        f"  log_price: shape={payload['log_price'].shape}, dtype={payload['log_price'].dtype}"
    )
    print(
        f"  variance: shape={payload['variance'].shape}, dtype={payload['variance'].dtype}"
    )
    print(f"  driver: shape={payload['driver'].shape}, dtype={payload['driver'].dtype}")

    if not bool(args.no_plot):
        _save_plots(payload=payload, subdir=args.subdir, output_dir=output_dir)
