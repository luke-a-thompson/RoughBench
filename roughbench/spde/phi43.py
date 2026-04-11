from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import jax
from jax import config as jax_config
import jax.numpy as jnp


jax_config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SimParams:
    """Paper-aligned parameters for the Zhu-Zhu lattice approximation.

    Conventions follow Zhu-Zhu (2015):
    - the spatial domain is the torus [-1, 1)^3;
    - N is the Fourier cutoff from the paper;
    - the lattice has M = 2*N + 1 sites per axis;
    - the lattice spacing is eps = 2 / (2*N + 1).

    Notes
    -----
    The renormalisation constants are computed from the paper's finite-lattice
    formulas using the exact lattice kernels in tau, with the tau-integrals
    approximated by a trapezoidal rule. This is faithful to the paper's finite
    lattice objects while remaining straightforward to implement.
    """

    N: int
    dt: float
    steps: int
    dtype: jnp.dtype = jnp.float64
    seed: int = 0
    num_tau: int = 256
    tau_max_multiplier: float = 20.0
    include_c12: bool = True

    @property
    def M(self) -> int:
        return 2 * self.N + 1

    @property
    def L(self) -> float:
        return 2.0

    @property
    def eps(self) -> float:
        return 2.0 / float(self.M)


@dataclass(frozen=True)
class Precomp:
    mode_numbers_fft: jax.Array
    mode_numbers_rfft: jax.Array
    lam_rfft: jax.Array
    solver_denom: jax.Array
    C0: float
    C11: float
    C12: float
    C1: float
    Cmass: float


def _fft_mode_numbers(M: int) -> np.ndarray:
    """Integer Fourier mode numbers in FFT ordering for an odd grid of length M."""
    return np.fft.fftfreq(M, d=1.0 / float(M)).astype(np.int64)


def _rfft_mode_numbers(M: int) -> np.ndarray:
    """Nonnegative integer Fourier mode numbers in rFFT ordering."""
    return np.arange(M // 2 + 1, dtype=np.int64)


def _centered_mode_numbers(N: int) -> np.ndarray:
    """Integer Fourier mode numbers in centered order: [-N, ..., N]."""
    return np.arange(-N, N + 1, dtype=np.int64)


def _laplacian_symbol_from_mode_numbers_np(
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Positive symbol lambda(k) = |k|^2 f(eps k) from Zhu-Zhu.

    Since eps = 2 / (2N + 1), this equals
        lambda(k) = 4 / eps^2 * sum_j sin^2(pi * eps * k_j / 2).
    """
    sx = np.sin(0.5 * math.pi * eps * kx) ** 2
    sy = np.sin(0.5 * math.pi * eps * ky) ** 2
    sz = np.sin(0.5 * math.pi * eps * kz) ** 2
    return (4.0 / (eps * eps)) * (
        sx[:, None, None] + sy[None, :, None] + sz[None, None, :]
    )


def _laplacian_symbol_from_mode_numbers_jnp(
    kx: jax.Array,
    ky: jax.Array,
    kz: jax.Array,
    eps: float,
) -> jax.Array:
    sx = jnp.sin(0.5 * math.pi * float(eps) * kx) ** 2
    sy = jnp.sin(0.5 * math.pi * float(eps) * ky) ** 2
    sz = jnp.sin(0.5 * math.pi * float(eps) * kz) ** 2
    return (4.0 / (float(eps) ** 2)) * (
        sx[:, None, None] + sy[None, :, None] + sz[None, None, :]
    )


def _linear_convolution_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact linear convolution over the mode box via zero-padded FFT."""
    out_shape = tuple(np.array(a.shape) + np.array(b.shape) - 1)
    fa = np.fft.fftn(a, s=out_shape)
    fb = np.fft.fftn(b, s=out_shape)
    return np.fft.ifftn(fa * fb).real


def _paper_renorm_geometry(N: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute main-box and sideband alias geometry on [-2N, 2N]^3."""
    M = 2 * N + 1
    full_modes = np.arange(-2 * N, 2 * N + 1, dtype=np.int64)
    mx, my, mz = np.meshgrid(full_modes, full_modes, full_modes, indexing="ij")

    main_mask = (np.abs(mx) <= N) & (np.abs(my) <= N) & (np.abs(mz) <= N)

    shift_x = np.where(mx > N, 1, np.where(mx < -N, -1, 0))
    shift_y = np.where(my > N, 1, np.where(my < -N, -1, 0))
    shift_z = np.where(mz > N, 1, np.where(mz < -N, -1, 0))

    alias_x = mx - M * shift_x
    alias_y = my - M * shift_y
    alias_z = mz - M * shift_z

    centered_modes = _centered_mode_numbers(N)
    lam_box = _laplacian_symbol_from_mode_numbers_np(
        centered_modes, centered_modes, centered_modes, eps
    )
    alias_lam = lam_box[alias_x + N, alias_y + N, alias_z + N]

    return main_mask, alias_lam


def compute_C0_C1(params: SimParams) -> tuple[float, float, float, float, float]:
    """Compute (C0, C11, C12, C1, Cmass) for the Zhu-Zhu lattice model.

    The constants are computed from the paper's finite-lattice kernels:
        C0  = 2^-3 sum_k 1 / (2 lambda(k)),
        C11 = 2^-5 ∫_0^∞ sum_{k1,k2} V_tau(k1) V_tau(k2) P_tau(k1+k2) d tau,
        C12 = 2^-5 ∫_0^∞ sum_{k1,k2} V_tau(k1) V_tau(k2) P_tau(alias(k1+k2)) d tau
              over the sideband region only.

    Here
        P_tau(k) = exp(-tau lambda(k)),
        V_tau(k) = P_tau(k) / (2 lambda(k))
    with V_tau(0) set to 0.

    The tau-integrals are approximated by a trapezoidal rule. The inner mode sums
    are evaluated exactly on the finite lattice using zero-padded linear convolutions,
    so there is no circular-aliasing contamination between the C11 and C12 parts.
    """
    N = params.N
    M = params.M
    eps = params.eps

    centered_modes = _centered_mode_numbers(N)
    lam_box = _laplacian_symbol_from_mode_numbers_np(
        centered_modes,
        centered_modes,
        centered_modes,
        eps,
    )

    zero_index = (N, N, N)
    lam_safe = lam_box.copy()
    lam_safe[zero_index] = np.inf

    c0 = float((2.0**-3) * np.sum(0.5 / lam_safe))

    positive_lam = lam_box[lam_box > 0.0]
    lam_min_pos = float(np.min(positive_lam))
    tau_max = float(params.tau_max_multiplier) / lam_min_pos
    num_tau = max(int(params.num_tau), 2)
    taus = np.linspace(0.0, tau_max, num_tau, dtype=np.float64)

    main_mask, alias_lam_full = _paper_renorm_geometry(N, eps)
    main_slice = slice(N, 3 * N + 1)

    integrand11 = np.empty((num_tau,), dtype=np.float64)
    integrand12 = np.empty((num_tau,), dtype=np.float64)

    for i, tau in enumerate(taus):
        P_box = np.exp(-tau * lam_box)
        V_box = np.zeros_like(lam_box)
        positive_mask = lam_box > 0.0
        V_box[positive_mask] = P_box[positive_mask] / (2.0 * lam_box[positive_mask])

        conv_VV = _linear_convolution_3d(V_box, V_box)

        P_main_full = np.zeros_like(conv_VV)
        P_main_full[main_slice, main_slice, main_slice] = P_box
        integrand11[i] = float(np.sum(P_main_full * conv_VV))

        if params.include_c12:
            P_alias_full = np.where(main_mask, 0.0, np.exp(-tau * alias_lam_full))
            integrand12[i] = float(np.sum(P_alias_full * conv_VV))
        else:
            integrand12[i] = 0.0

    c11 = float((2.0**-5) * np.trapezoid(integrand11, taus))
    c12 = float((2.0**-5) * np.trapezoid(integrand12, taus))
    c1 = c11 + c12
    cmass = 3.0 * c0 - 9.0 * c1
    return c0, c11, c12, c1, cmass


def precompute(params: SimParams) -> Precomp:
    fft_modes = _fft_mode_numbers(params.M)
    rfft_modes = _rfft_mode_numbers(params.M)

    lam_rfft = _laplacian_symbol_from_mode_numbers_jnp(
        jnp.asarray(fft_modes),
        jnp.asarray(fft_modes),
        jnp.asarray(rfft_modes),
        params.eps,
    ).astype(params.dtype)

    solver_denom = (1.0 + float(params.dt) * lam_rfft).astype(params.dtype)

    C0, C11, C12, C1, Cmass = compute_C0_C1(params)

    return Precomp(
        mode_numbers_fft=jnp.asarray(fft_modes),
        mode_numbers_rfft=jnp.asarray(rfft_modes),
        lam_rfft=lam_rfft,
        solver_denom=solver_denom,
        C0=C0,
        C11=C11,
        C12=C12,
        C1=C1,
        Cmass=Cmass,
    )


def _noise_real_space(key: jax.Array, params: SimParams) -> jax.Array:
    """Space-time white-noise increment on the lattice.

    At each lattice site, the Euler increment is sqrt(dt) * eps^(-3/2) * Normal(0,1),
    matching the independent Brownian drivers in the finite-dimensional lattice SDE.
    """
    shape = (params.M, params.M, params.M)
    scale = (float(params.dt) ** 0.5) / (float(params.eps) ** 1.5)
    return jax.random.normal(key, shape, dtype=params.dtype) * scale


def semi_implicit_step(
    phi: jax.Array,
    key: jax.Array,
    params: SimParams,
    pre: Precomp,
) -> tuple[jax.Array, jax.Array]:
    """One semi-implicit Euler step.

    We treat the discrete Laplacian implicitly and the cubic plus renormalisation
    drift explicitly:
        (I - dt Delta_eps) phi^{n+1}
            = phi^n + dt (- (phi^n)^3 + Cmass phi^n) + dW^n.

    This is a numerical integrator for the Zhu-Zhu finite-dimensional lattice SDE,
    not a claim from the paper itself.
    """
    drift = -(phi**3) + float(pre.Cmass) * phi
    rhs = phi + float(params.dt) * drift

    key, subkey = jax.random.split(key)
    rhs = rhs + _noise_real_space(subkey, params)

    rhs_hat = jnp.fft.rfftn(rhs, axes=(0, 1, 2))
    phi_next_hat = rhs_hat / pre.solver_denom
    phi_next = jnp.fft.irfftn(
        phi_next_hat,
        s=(params.M, params.M, params.M),
        axes=(0, 1, 2),
    )
    return phi_next.astype(params.dtype), key


def simulate(
    params: SimParams,
    pre: Precomp,
    phi0: Optional[jax.Array] = None,
    snapshot_every: int = 0,
    burnin: int = 0,
) -> tuple[jax.Array, Optional[jax.Array]]:
    """Run the semi-implicit lattice simulation.

    Returns
    -------
    phi_final:
        Final field on the M x M x M lattice.
    snapshots:
        Optional stack of snapshots with shape (T_snap, M, M, M).
    """
    if phi0 is None:
        phi = jnp.zeros((params.M, params.M, params.M), dtype=params.dtype)
    else:
        phi = phi0.astype(params.dtype)
        if phi.shape != (params.M, params.M, params.M):
            raise ValueError(
                f"phi0 must have shape {(params.M, params.M, params.M)}, got {phi.shape}."
            )

    key = jax.random.PRNGKey(params.seed)

    def one_step(
        carry: tuple[jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        state, key_in = carry
        state_next, key_out = semi_implicit_step(state, key_in, params, pre)
        return (state_next, key_out), state_next

    (phi_final, _), traj = jax.lax.scan(
        one_step, (phi, key), xs=None, length=params.steps
    )

    snapshots: Optional[jax.Array] = None
    if snapshot_every > 0:
        start = max(int(burnin), 0)
        traj_post = traj[start:]
        if traj_post.shape[0] == 0:
            snapshots = traj_post
        else:
            idx = jnp.arange(traj_post.shape[0])
            mask = (idx + 1) % int(snapshot_every) == 0
            snapshots = traj_post[mask]

    return phi_final, snapshots


def structure_factor(phi: jax.Array, params: SimParams) -> jax.Array:
    """Return |phi_hat|^2 / |T^3| on the rFFT grid."""
    volume = float(params.L) ** 3
    hat_phi = jnp.fft.rfftn(phi, axes=(0, 1, 2))
    return (hat_phi * jnp.conj(hat_phi)).real / volume


def two_point_correlation(phi: jax.Array) -> jax.Array:
    """Equal-time two-point correlation from a single snapshot."""
    M = phi.shape[0]
    power = jnp.abs(jnp.fft.fftn(phi, axes=(0, 1, 2))) ** 2
    return jnp.fft.ifftn(power, axes=(0, 1, 2)).real / float(M**3)


def to_tcxyz(snaps: jax.Array) -> jax.Array:
    """Convert (T, X, Y, Z) snapshots to (T, C, X, Y, Z) with C = 1."""
    return snaps[:, None, ...]
