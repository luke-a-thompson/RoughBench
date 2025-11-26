import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.scipy.stats
from roughbench.rde.rough_volatility import BonesiniModelSpec
from pathlib import Path
from utils import (
    save_plot,
    save_npz_compressed,
    plotting_context,
    create_figure,
    decorate_axes,
    finalize_plot,
)


def plot_bonesini_rde(
    solution: dfx.Solution | list[dfx.Solution],
    model_spec: BonesiniModelSpec | list[BonesiniModelSpec],
    output_dir: Path | None = None,
) -> None:
    if isinstance(model_spec, BonesiniModelSpec):
        model_spec = [model_spec]

    if isinstance(solution, dfx.Solution):
        solution = [solution]

    if len(solution) != len(model_spec):
        raise ValueError("Number of solutions and model specifications must match.")

    with plotting_context(font_scale=1.1) as plt:
        _, ax = create_figure(figsize=(10.0, 6.0))
        for sol, spec in zip(solution, model_spec):
            ys = jnp.asarray(sol.ys)
            if jnp.isnan(ys).any():
                print(f"Warning: {spec.name} contains NaN")
                continue
            ts = jnp.asarray(sol.ts)
            S = ys if ys.ndim == 1 else ys[:, 0]
            log_S = jnp.log(S)
            ax.plot(ts, log_S, label=spec.name)
        decorate_axes(
            ax,
            title="Log-Price paths",
            xlabel="Time",
            ylabel="Log-Price",
            legend=True,
        )
        finalize_plot(tight_layout=True)

    filename = "price_comparison.png"
    save_plot(filename=filename, subdir="rough_volatility", data_dir=output_dir, dpi=200)


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

    if jnp.isnan(ys_paths).any():
        print(f"Warning: {model_spec.name} contains NaN")

    num_paths = ts_paths.shape[0]

    final_values = []
    initial_values = []
    ax_var = None
    for i in range(num_paths):
        ts = ts_paths[i]
        ys = ys_paths[i]
        S = ys if ys.ndim == 1 else ys[:, 0]
        price_data = jnp.log(S) if use_log_price else S
        if ax_main is not None:
            ax_main.plot(ts, price_data, color="gray", alpha=0.6)
        final_values.append(price_data[-1])
        initial_values.append(price_data[0])

        # Optionally plot variance from second dimension on a twin y-axis
        if ax_main is not None and plot_variance and ys.ndim > 1 and ys.shape[1] > 1:
            if ax_var is None:
                ax_var = ax_main.twinx()
                ax_var.set_ylabel("Variance", color="tab:green")
                ax_var.tick_params(axis="y", labelcolor="tab:green")
            V = ys[:, 1]
            ax_var.plot(ts, V, color="tab:green", alpha=0.3)

    # Calculate means
    mean_initial = float(jnp.mean(jnp.array(initial_values)))
    mean_final = float(jnp.mean(jnp.array(final_values)))

    # Plot mean lines
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

        # Simple fat-tailedness check: compare to normal distribution
        final_array = jnp.array(final_values)
        mean_val = jnp.mean(final_array)
        std_val = jnp.std(final_array)

        # Generate normal distribution with same mean/std
        y_range = jnp.linspace(mean_val - 3 * std_val, mean_val + 3 * std_val, 100)
        normal_pdf = jnp.exp(-0.5 * ((y_range - mean_val) / std_val) ** 2) / (std_val * jnp.sqrt(2 * jnp.pi))

        # Plot normal distribution for comparison
        if ax_marginal is not None:
            ax_marginal.plot(
                normal_pdf,
                y_range,
                color="red",
                linestyle="--",
                alpha=0.8,
                label="Normal",
            )

        # Simple fat-tailedness indicator: compare tail probabilities
        tail_threshold = 2 * std_val
        empirical_tail_prob = jnp.mean(jnp.abs(final_array - mean_val) > tail_threshold)
        normal_tail_prob = 2 * (1 - jax.scipy.stats.norm.cdf(tail_threshold, 0, std_val))

        if ax_marginal is not None and ax_main is not None:
            price_label = "log" if use_log_price else "price"
            ax_marginal.set_xlabel("Density")
            ax_marginal.set_ylabel("Log-Price" if use_log_price else "Price")
            title = f"t=1 Marginal ({price_label})"
            decorate_axes(
                ax_marginal,
                title=title,
                xlabel="Density",
                ylabel=("Log-Price" if use_log_price else "Price"),
                legend=True,
            )
            y_min, y_max = ax_main.get_ylim()
            ax_marginal.set_ylim(y_min, y_max)

    finalize_plot(tight_layout=True)

    filename = f"{model_spec.name.lower().replace(' ', '_')}_monte_carlo.png"
    save_plot(filename=filename, subdir="rough_volatility", data_dir=output_dir, dpi=200)

    # Save solution and drivers as compressed .npz
    ys_paths = jnp.asarray(solution.ys)

    # Stack X and W drivers into a single array (batch, timesteps+1, 2) if available
    if X_drivers is not None and W_drivers is not None:
        # Stack along last axis: [:, :, 0] is X, [:, :, 1] is W
        drivers = jnp.stack([X_drivers, W_drivers], axis=-1)
    else:
        # No drivers provided, create a dummy array
        drivers = jnp.zeros((ys_paths.shape[0], 1, 2))

    save_npz_compressed(
        solution=ys_paths,
        driver=drivers,
        filename=f"{model_spec.name.lower().replace(' ', '_')}_data.npz",
        subdir="rough_volatility",
        data_dir=output_dir,
    )


if __name__ == "__main__":
    from roughbench.rde.rough_volatility import (
        make_black_scholes_model_spec,
        make_bergomi_model_spec,
        make_rough_bergomi_model_spec,
        get_bonesini_noise_drivers,
        solve_bonesini_rde_from_drivers,
    )

    noise_timesteps = 1000
    rde_timesteps = 10_000
    num_paths = 5000

    output_dir = None

    # BLACK-SCHOLES
    print("Generating Black-Scholes Monte Carlo...")
    black_scholes_model_spec = make_black_scholes_model_spec(v_0=0.04)
    keys_bs = jax.random.split(jax.random.PRNGKey(42), num_paths)
    y0_bs, X_bs, W_bs = jax.vmap(
        lambda key: get_bonesini_noise_drivers(key, noise_timesteps, black_scholes_model_spec, s_0=1.0)
    )(keys_bs)
    solve_vmap_bs = jax.vmap(
        lambda y0, X, W: solve_bonesini_rde_from_drivers(
            y0, X, W, black_scholes_model_spec, noise_timesteps, rde_timesteps
        )
    )
    solutions_bs = solve_vmap_bs(y0_bs, X_bs, W_bs)
    plot_bonesini_monte_carlo(
        solutions_bs,
        black_scholes_model_spec,
        X_drivers=X_bs,
        W_drivers=W_bs,
        output_dir=output_dir,
    )

    # BERGOMI
    print("Generating Bergomi Monte Carlo...")
    bergomi_model_spec = make_bergomi_model_spec(v_0=0.0, rho=-0.848)
    keys_b = jax.random.split(jax.random.PRNGKey(42), num_paths)
    y0_b, X_b, W_b = jax.vmap(
        lambda key: get_bonesini_noise_drivers(key, noise_timesteps, bergomi_model_spec, s_0=1.0)
    )(keys_b)
    solve_vmap_b = jax.vmap(
        lambda y0, X, W: solve_bonesini_rde_from_drivers(y0, X, W, bergomi_model_spec, noise_timesteps, rde_timesteps)
    )
    solutions_b = solve_vmap_b(y0_b, X_b, W_b)
    plot_bonesini_monte_carlo(
        solutions_b,
        bergomi_model_spec,
        X_drivers=X_b,
        W_drivers=W_b,
        output_dir=output_dir,
    )

    # ROUGH BERGOMI
    print("Generating Rough Bergomi Monte Carlo...")
    rough_bergomi_model_spec = make_rough_bergomi_model_spec(v_0=0.04, nu=1.991, hurst=0.25, rho=-0.848)
    keys_rb = jax.random.split(jax.random.PRNGKey(42), num_paths)
    y0_rb, X_rb, W_rb = jax.vmap(
        lambda key: get_bonesini_noise_drivers(key, noise_timesteps, rough_bergomi_model_spec, s_0=100.0)
    )(keys_rb)
    solve_vmap_rb = jax.vmap(
        lambda y0, X, W: solve_bonesini_rde_from_drivers(
            y0, X, W, rough_bergomi_model_spec, noise_timesteps, rde_timesteps
        )
    )
    solutions_rb = solve_vmap_rb(y0_rb, X_rb, W_rb)
    plot_bonesini_monte_carlo(
        solutions_rb,
        rough_bergomi_model_spec,
        X_drivers=X_rb,
        W_drivers=W_rb,
        use_log_price=True,
        output_dir=output_dir,
    )

    print("\nAll rough volatility plots saved to generate_data/rough_volatility and docs/rde_bench/rough_volatility")
