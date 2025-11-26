import jax
import jax.numpy as jnp
from pathlib import Path
from roughbench.rde.ou_process import ou_process
from quicksig.drivers.drivers import bm_driver
from utils import (
    save_plot,
    save_npz_compressed,
    plotting_context,
    create_figure,
    decorate_axes,
    finalize_plot,
)


def plot_ou_monte_carlo(
    batch_size: int,
    timesteps: int,
    dim: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    seed: int = 42,
    output_dir: Path | None = None,
) -> None:
    """
    Generate and plot Ornstein-Uhlenbeck process paths using Monte Carlo simulation.

    Args:
        batch_size: Number of paths to generate
        timesteps: Number of time steps
        dim: Dimension of the OU process
        theta: Rate of mean reversion
        mu: Long-term mean
        sigma: Volatility
        x0: Initial value
        seed: Random seed
        output_dir: Directory to save the plot (defaults to docs/rde_bench/)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, batch_size)

    # Generate Brownian motion drivers (same as used internally by ou_process)
    batched_bm_drivers = jax.vmap(bm_driver, in_axes=(0, None, None))(keys, timesteps, dim)

    # Vectorize over multiple paths
    batched_ou_paths = jax.vmap(ou_process, in_axes=(0, None, None, None, None, None, None))(
        keys, timesteps, dim, theta, mu, sigma, x0
    )

    ou_paths_np = jax.device_get(batched_ou_paths.path)
    bm_drivers_np = jax.device_get(batched_bm_drivers.path)

    # Plot with shared style
    with plotting_context(font_scale=1.1) as plt:
        _, ax = create_figure(figsize=(10.0, 6.0))
        for i in range(batch_size):
            ax.plot(ou_paths_np[i, :, 0], linewidth=0.5, alpha=0.15, color="tab:orange")
        ax.axhline(y=mu, color="red", linestyle="--", linewidth=2, label=f"Mean μ={mu}")
        title = f"Ornstein-Uhlenbeck (θ={theta}, μ={mu}, σ={sigma}, N={timesteps}, batch={batch_size})"
        decorate_axes(ax, title=title, xlabel="Time step", ylabel="Value", legend=True)
        finalize_plot(tight_layout=True)

    filename = "ou_process_monte_carlo.png"
    save_plot(filename=filename, subdir="ou_processes", data_dir=output_dir, dpi=200)

    # Save solution and driver as compressed .npz
    save_npz_compressed(
        solution=ou_paths_np,
        driver=bm_drivers_np,
        filename="ou_process_data.npz",
        subdir="ou_processes",
        data_dir=output_dir,
    )

    # E[X_t] = μ + (X_0 - μ)e^(-θt) at t=1.0
    T = 1.0
    expected_mean = mu + (x0 - mu) * jnp.exp(-theta * T)
    expected_var = (sigma**2) / (2.0 * theta)

    mean_final = float(ou_paths_np[:, -1, 0].mean())
    std_final = float(ou_paths_np[:, -1, 0].std())
    print("")
    print(f"Mean of final values: {mean_final:.4f} (expected ≈ {float(expected_mean):.4f})")
    print(f"Std of final values: {std_final:.4f} (stationary σ ≈ {float(jnp.sqrt(expected_var)):.4f})")


if __name__ == "__main__":
    print("Generating OU process Monte Carlo simulation...")
    plot_ou_monte_carlo(
        batch_size=5000,
        timesteps=8192,
        dim=3,
        theta=0.5,
        mu=0.0,
        sigma=0.3,
        x0=1.0,
        seed=42,
    )
