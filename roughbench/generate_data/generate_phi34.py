"""Generate Phi^4_3 SPDE spacetime data."""

import jax
import jax.numpy as jnp
from roughbench.spde.phi4_3_zhu_zhu_corrected import (
    SimParams,
    precompute,
    simulate,
    structure_factor,
    to_tcxyz,
    two_point_correlation,
)
from utils import save_npy


def main() -> None:
    """
    Run a default Zhu-Zhu Phi^4_3 lattice simulation and save diagnostics.
    """
    N: int = 32
    sim_steps: int = 2048
    burnin_steps: int = 32
    dt_coeff: float = 0.01

    # The corrected solver follows Zhu-Zhu's conventions:
    # cutoff N, lattice size M = 2N + 1, spacing eps = 2 / M on [-1, 1)^3.
    M: int = 2 * N + 1
    eps: float = 2.0 / float(M)
    dt: float = dt_coeff * eps * eps

    params: SimParams = SimParams(
        N=N,
        dt=dt,
        steps=sim_steps + burnin_steps,
        seed=0,
    )

    print("Generating Phi^4_3 spacetime data...")
    pre = precompute(params)
    phi_final, snaps = simulate(params, pre, phi0=None, snapshot_every=1, burnin=burnin_steps)

    S_q = structure_factor(phi_final, params)
    C_x = two_point_correlation(phi_final)

    print("cutoff N:", params.N, "lattice M:", params.M, "eps:", params.eps, "dt:", params.dt)
    print("renorm constants:", {"C0": pre.C0, "C11": pre.C11, "C12": pre.C12, "C1": pre.C1, "Cmass": pre.Cmass})
    print("phi_final:", phi_final.shape)
    print("snaps:", None if snaps is None else snaps.shape)
    print("S_q:", S_q.shape, "C_x:", C_x.shape)

    save_npy(jax.device_get(phi_final), "phi34_final.npy", subdir="phi34")
    save_npy(jax.device_get(S_q), "phi34_structure_factor.npy", subdir="phi34")
    save_npy(jax.device_get(C_x), "phi34_two_point_correlation.npy", subdir="phi34")
    save_npy(
        jnp.asarray([pre.C0, pre.C11, pre.C12, pre.C1, pre.Cmass], dtype=jnp.float64),
        "phi34_renorm_constants.npy",
        subdir="phi34",
    )

    # Save spacetime rollout (snaps) as NPY.
    if snaps is not None:
        snaps_np: object
        if params.dtype == jnp.float64:
            snaps_np = jnp.asarray(jax.device_get(snaps), dtype=jnp.float64)
        else:
            snaps_np = jnp.asarray(jax.device_get(snaps), dtype=jnp.float32)

        save_npy(snaps_np, "phi34_snaps.npy", subdir="phi34")

        # Also save TCXYZ for visualization tools.
        snaps_tcxyz_np: object = to_tcxyz(snaps_np)
        save_npy(snaps_tcxyz_np, "phi34_snaps_tcxyz.npy", subdir="phi34")

    print("\nPhi^4_3 data generation complete.")


if __name__ == "__main__":
    main()

