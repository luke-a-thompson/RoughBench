import jax
import jax.numpy as jnp
from pathlib import Path
from roughbench.rde.ou_process import ou_process
from utils import save_plot, save_npy


def plot_ou_monte_carlo(
    batch_size: int = 1000,
    timesteps: int = 512,
    dim: int = 1,
    theta: float = 0.5,
    mu: float = 0.0,
    sigma: float = 0.3,
    x0: float = 1.0,
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
    
    # Vectorize over multiple paths
    batched_ou_paths = jax.vmap(
        ou_process,
        in_axes=(0, None, None, None, None, None, None)
    )(keys, timesteps, dim, theta, mu, sigma, x0)
    
    ou_paths_np = jax.device_get(batched_ou_paths.path)
    
    # Plot only if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(10, 6))
        for i in range(batch_size):
            plt.plot(ou_paths_np[i, :, 0], linewidth=0.5, alpha=0.15, color="tab:orange")
        plt.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Mean μ={mu}')
        plt.title(f"Ornstein-Uhlenbeck Process (θ={theta}, μ={mu}, σ={sigma}, N={timesteps}, batch={batch_size})")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
    except Exception:
        plt = None  # type: ignore
    
    filename = "ou_process_monte_carlo.png"
    save_plot(filename=filename, subdir="ou_processes", data_dir=output_dir, dpi=150)
    
    # Save arrays under data/ou_processes only (before printing stats)
    save_npy(ou_paths_np, filename="ou_process_paths.npy", subdir="ou_processes", data_dir=output_dir)

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
        batch_size=1000,
        timesteps=512,
        theta=0.5,
        mu=0.0,
        sigma=0.3,
        x0=1.0,
        seed=42,
    )

