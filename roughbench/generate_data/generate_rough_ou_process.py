import jax
import jax.numpy as jnp
from pathlib import Path
from roughbench.rde.rough_ou_process import rough_ou_process
from quicksig.drivers.drivers import fractional_bm_driver
from utils import save_plot, save_npz_compressed


def plot_rough_ou_monte_carlo(
    batch_size: int = 1000,
    timesteps: int = 512,
    dim: int = 1,
    theta: float = 0.5,
    mu: float = 0.0,
    sigma: float = 0.3,
    hurst: float = 0.7,
    x0: float = 1.0,
    seed: int = 42,
    output_dir: Path | None = None,
) -> None:
    """
    Generate and plot rough Ornstein-Uhlenbeck process paths using Monte Carlo simulation.
    
    Args:
        batch_size: Number of paths to generate
        timesteps: Number of time steps
        dim: Dimension of the rough OU process
        theta: Rate of mean reversion
        mu: Long-term mean
        sigma: Volatility
        hurst: Hurst parameter (controls memory)
        x0: Initial value
        seed: Random seed
        output_dir: Directory to save the plot (defaults to docs/rde_bench/)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, batch_size)
    
    # Generate fractional Brownian motion drivers (same as used internally by rough_ou_process)
    batched_fbm_drivers = jax.vmap(
        fractional_bm_driver,
        in_axes=(0, None, None, None)
    )(keys, timesteps, dim, hurst)
    
    # Vectorize over multiple paths
    batched_rough_ou_paths = jax.vmap(
        rough_ou_process,
        in_axes=(0, None, None, None, None, None, None, None)
    )(keys, timesteps, dim, theta, mu, sigma, hurst, x0)
    
    rough_ou_paths_np = jax.device_get(batched_rough_ou_paths.path)
    fbm_drivers_np = jax.device_get(batched_fbm_drivers.path)
    
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(10, 6))
        for i in range(batch_size):
            plt.plot(rough_ou_paths_np[i, :, 0], linewidth=0.5, alpha=0.15, color="tab:blue")
        plt.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Mean μ={mu}')
        plt.title(f"Rough Ornstein-Uhlenbeck Process (θ={theta}, μ={mu}, σ={sigma}, H={hurst}, N={timesteps}, batch={batch_size})")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
    except Exception:
        plt = None  # type: ignore
    
    filename = f"rough_ou_process_H{hurst:.2f}_monte_carlo.png"
    save_plot(filename=filename, subdir="rough_ou_processes", data_dir=output_dir, dpi=150)
    
    # Save solution and driver as compressed .npz
    save_npz_compressed(
        solution=rough_ou_paths_np,
        driver=fbm_drivers_np,
        filename=f"rough_ou_data_H{hurst:.2f}.npz",
        subdir="rough_ou_processes",
        data_dir=output_dir
    )

    # E[X_t] = μ + (X_0 - μ)e^(-θt) at t=1.0 (same as regular OU)
    T = 1.0
    expected_mean = mu + (x0 - mu) * jnp.exp(-theta * T)
    
    # Note: For rough OU with H ≠ 0.5, the stationary variance is more complex
    # and depends on the Hurst parameter. For H=0.5 (standard BM), it reduces to σ²/(2θ)
    expected_var_h05 = (sigma**2) / (2.0 * theta)
    
    mean_final = float(rough_ou_paths_np[:, -1, 0].mean())
    std_final = float(rough_ou_paths_np[:, -1, 0].std())
    print("")
    print(f"Mean of final values: {mean_final:.4f} (expected ≈ {float(expected_mean):.4f})")
    print(f"Std of final values: {std_final:.4f} (H=0.5 stationary σ ≈ {float(jnp.sqrt(expected_var_h05)):.4f})")


if __name__ == "__main__":
    print("Generating rough OU process Monte Carlo simulation...")
    plot_rough_ou_monte_carlo(
        batch_size=1000,
        timesteps=512,
        theta=0.5,
        mu=0.0,
        sigma=0.3,
        hurst=0.7,
        x0=1.0,
        seed=42,
    )

