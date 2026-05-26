from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from roughbench.drivers import bm_driver
from roughbench.generate_data.utils import (
    draw_sde_paths,
    finalize_plot,
    plotting_context,
    save_npz_compressed,
    save_plot,
)

from roughbench.rde.ou_process import ou_process


def generate_ou_data(
    batch_size: int,
    timesteps: int,
    dim: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate Ornstein-Uhlenbeck paths and BM drivers.

    Returns a dict with keys:
      - solution: OU paths, shape (batch_size, timesteps+1, dim)
      - driver: BM driver paths, shape (batch_size, timesteps+1, dim)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, batch_size)

    batched_bm_drivers = jax.vmap(bm_driver, in_axes=(0, None, None))(
        keys, timesteps, dim
    )
    batched_ou_paths = jax.vmap(
        ou_process, in_axes=(0, None, None, None, None, None, None)
    )(keys, timesteps, dim, theta, mu, sigma, x0)

    return {
        "solution": jax.device_get(batched_ou_paths),
        "driver": jax.device_get(batched_bm_drivers),
    }


def plot_ou_monte_carlo(
    solution: np.ndarray,
    *,
    batch_size: int,
    timesteps: int,
    theta: float,
    mu: float,
    sigma: float,
    output_dir: Path | None = None,
) -> None:
    """Plot OU process Monte Carlo paths."""
    with plotting_context(font_scale=1.1, style="sde"):
        ts = np.linspace(0.0, 1.0, timesteps + 1, dtype=np.float32)
        title = (
            f"Ornstein-Uhlenbeck "
            f"(theta={theta}, mu={mu}, sigma={sigma}, N={timesteps}, paths={batch_size})"
        )
        draw_sde_paths(
            times=ts,
            paths=solution[:, :, 0],
            suptitle=title,
            ylabel="$X(t)$",
            expectation=np.mean(solution[:, :, 0], axis=0),
            marginal=True,
            figsize=(12.0, 7.0),
        )
        finalize_plot(tight_layout=True)

    save_plot(
        filename="ou_process_monte_carlo.png",
        subdir="ou_processes",
        data_dir=output_dir,
        dpi=200,
    )


if __name__ == "__main__":
    print("Generating OU process Monte Carlo simulation...")
    batch_size = 5000
    timesteps = 8192
    dim = 3
    theta = 0.5
    mu = 0.0
    sigma = 0.3
    x0 = 1.0
    seed = 42

    data = generate_ou_data(
        batch_size=batch_size,
        timesteps=timesteps,
        dim=dim,
        theta=theta,
        mu=mu,
        sigma=sigma,
        x0=x0,
        seed=seed,
    )

    save_npz_compressed(
        solution=data["solution"],
        driver=data["driver"],
        filename="ou_process_data.npz",
        subdir="ou_processes",
    )

    plot_ou_monte_carlo(
        data["solution"],
        batch_size=batch_size,
        timesteps=timesteps,
        theta=theta,
        mu=mu,
        sigma=sigma,
    )

    # E[X_t] = μ + (X_0 - μ)e^(-θt) at t=1.0
    T = 1.0
    expected_mean = mu + (x0 - mu) * jnp.exp(-theta * T)
    expected_var = (sigma**2) / (2.0 * theta)

    mean_final = float(data["solution"][:, -1, 0].mean())
    std_final = float(data["solution"][:, -1, 0].std())
    print("")
    print(
        f"Mean of final values: {mean_final:.4f} (expected ≈ {float(expected_mean):.4f})"
    )
    print(
        f"Std of final values: {std_final:.4f} (stationary σ ≈ {float(jnp.sqrt(expected_var)):.4f})"
    )
