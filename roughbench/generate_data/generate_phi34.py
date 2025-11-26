"""Generate Phi^4_3 SPDE spacetime data."""

import jax
import jax.numpy as jnp
from roughbench.spde.phi34 import SimParams, precompute, simulate, structure_factor, two_point_correlation
from utils import save_npy


def main() -> None:
    """
    Run a default Phi^4_3 simulation and save spacetime snapshots.
    """
    N: int = 48
    L: float = 0.1
    dx: float = L / float(N)
    dt: float = 0.01 * dx * dx
    # Keep physical horizons fixed; recompute steps if dt changes
    base_dt_coeff: float = 0.01  # reference coefficient for physical-time baselines
    total_time: float = 2048 * base_dt_coeff * dx * dx
    burnin_time: float = 64 * base_dt_coeff * dx * dx
    sim_steps: int = int(jnp.ceil(total_time / dt))
    burnin_steps: int = int(jnp.ceil(burnin_time / dt))

    params: SimParams = SimParams(
        N=N,
        L=L,
        dx=dx,
        dt=dt,
        steps=sim_steps + burnin_steps,
        dtype=jnp.float32,
        seed=0,
        use_bandlimited_noise=False,
    )

    print("Generating Phi^4_3 spacetime data...")
    pre = precompute(params)
    phi_final, snaps = simulate(params, pre, phi0=None, snapshot_every=1, burnin=burnin_steps)

    S_q = structure_factor(phi_final, L)
    C_x = two_point_correlation(phi_final)

    print("phi_final:", phi_final.shape)
    print("snaps:", None if snaps is None else snaps.shape)
    print("S_q:", S_q.shape, "C_x:", C_x.shape)

    # Save spacetime rollout (snaps) as NPY
    if snaps is not None:
        snaps_np: object
        if params.dtype == jnp.float64:
            snaps_np = jnp.asarray(jax.device_get(snaps), dtype=jnp.float64)
        else:
            snaps_np = jnp.asarray(jax.device_get(snaps), dtype=jnp.float32)

        save_npy(snaps_np, "phi34_snaps.npy", subdir="phi34")

        # Also save TCXYZ (add channel dim C=1) for visualization tools
        snaps_tcxyz_np: object = jnp.expand_dims(snaps_np, axis=1)
        save_npy(snaps_tcxyz_np, "phi34_snaps_tcxyz.npy", subdir="phi34")

    print("\nPhi^4_3 data generation complete.")


if __name__ == "__main__":
    main()
