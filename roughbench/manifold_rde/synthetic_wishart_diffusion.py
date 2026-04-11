from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
import diffrax as dfx
from typing import Literal


def sym(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def spd_sqrt(matrix: Array, eps: float = 1e-6) -> Array:
    matrix = sym(matrix)
    evals, evecs = jnp.linalg.eigh(matrix)
    evals = jnp.maximum(evals, eps)
    return (evecs * jnp.sqrt(evals)) @ evecs.T


def spd_log(matrix: Array, eps: float = 1e-6) -> Array:
    matrix = sym(matrix)
    evals, evecs = jnp.linalg.eigh(matrix)
    evals = jnp.maximum(evals, eps)
    return (evecs * jnp.log(evals)) @ evecs.T


def spd_exp(sym_matrix: Array, exp_floor: float | None = None) -> Array:
    sym_matrix = sym(sym_matrix)
    evals, evecs = jnp.linalg.eigh(sym_matrix)
    if exp_floor is None:
        exp_floor = float(jnp.finfo(evals.dtype).tiny)
    min_log = jnp.log(jnp.asarray(exp_floor, dtype=evals.dtype))
    evals = jnp.maximum(evals, min_log)
    return (evecs * jnp.exp(evals)) @ evecs.T


def _is_tracer(value: object) -> bool:
    return hasattr(value, "aval")


def _assert_square(matrix: Array, dim: int, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != dim or matrix.shape[1] != dim:
        raise ValueError(f"{name} must have shape ({dim}, {dim}). Got {matrix.shape}.")


def _check_spd(
    matrix: Array, name: str, eps: float, allow_semidef: bool = False
) -> None:
    if _is_tracer(matrix):
        return
    dense = np.asarray(jax.device_get(matrix))
    if not np.allclose(dense, dense.T, atol=1e-6):
        raise ValueError(f"{name} must be symmetric.")
    evals = np.linalg.eigvalsh(dense)
    bound = -float(eps) if allow_semidef else float(eps)
    if np.min(evals) < bound:
        label = "PSD" if allow_semidef else "SPD"
        raise ValueError(f"{name} must be {label} with min eigenvalue >= {bound:.3e}.")


def _validate_params(
    *,
    X0: Array,
    b: Array,
    H: Array,
    Sigma: Array,
    eps: float,
) -> None:
    dim = int(X0.shape[0])
    _assert_square(X0, dim, "X0")
    _assert_square(b, dim, "b")
    _assert_square(H, dim, "H")
    _assert_square(Sigma, dim, "Sigma")
    _check_spd(X0, "X0", eps=eps, allow_semidef=False)
    _check_spd(b, "b", eps=0.0, allow_semidef=True)


def make_wishart_parameters(
    *,
    Sigma: Array,
    gamma: float,
    A: Array,
    eps: float,
) -> dict[str, Array]:
    dim = int(Sigma.shape[0])
    _assert_square(Sigma, dim, "Sigma")
    _assert_square(A, dim, "A")
    Q = Sigma.T @ Sigma
    b = (dim - 1) * Q + eps * jnp.eye(dim, dtype=Sigma.dtype)
    H = -float(gamma) * jnp.eye(dim, dtype=Sigma.dtype) + A
    return {"b": b, "H": H, "Q": Q}


def _correlation_factor(
    corr_matrix: Array | None, num_paths: int, dtype: jnp.dtype
) -> Array:
    if corr_matrix is None:
        return jnp.eye(num_paths, dtype=dtype)
    if corr_matrix.shape != (num_paths, num_paths):
        raise ValueError(
            f"corr_matrix must have shape ({num_paths}, {num_paths}). Got {corr_matrix.shape}."
        )
    return jnp.linalg.cholesky(corr_matrix)


def _is_identity_corr(
    corr_matrix: Array | None, num_paths: int, eps: float = 1e-6
) -> bool:
    if corr_matrix is None:
        return True
    if _is_tracer(corr_matrix):
        return False
    dense = np.asarray(jax.device_get(corr_matrix))
    if dense.shape != (num_paths, num_paths):
        return False
    return bool(np.allclose(dense, np.eye(num_paths), atol=eps, rtol=0.0))


def _quadratic_variation_closed_form(X: Array, Q: Array) -> Array:
    dim = int(X.shape[0])
    term1 = jnp.einsum("im,jn->ijmn", X, Q)
    term2 = jnp.einsum("in,jm->ijmn", X, Q)
    term3 = jnp.einsum("jm,in->ijmn", X, Q)
    term4 = jnp.einsum("jn,im->ijmn", X, Q)
    qv = term1 + term2 + term3 + term4
    return qv.reshape(dim * dim, dim * dim)


def _quadratic_variation_matrix(
    *,
    X: Array,
    Sigma: Array,
    chol: Array,
    eps: float,
    use_closed_form: bool,
) -> Array:
    dim = int(X.shape[0])
    if use_closed_form:
        Q = Sigma.T @ Sigma
        return _quadratic_variation_closed_form(sym(X), Q)

    num_paths = dim * dim
    X_sym = sym(X)
    sqrtX = spd_sqrt(X_sym, eps=eps)

    def mv(dB_flat: Array) -> Array:
        dW_flat = chol @ dB_flat
        dW = dW_flat.reshape(dim, dim)
        dX = sym(sqrtX @ dW @ Sigma + Sigma.T @ dW.T @ sqrtX)
        return dX.reshape(num_paths)

    basis = jnp.eye(num_paths, dtype=X.dtype)
    cols = jax.vmap(mv)(basis)
    return cols.T @ cols


def _matrix_sqrt_and_inv_sqrt(X: Array, eps: float) -> tuple[Array, Array]:
    X_sym = sym(X)
    evals, evecs = jnp.linalg.eigh(X_sym)
    evals = jnp.maximum(evals, eps)
    sqrt_evals = jnp.sqrt(evals)
    inv_sqrt_evals = 1.0 / sqrt_evals
    X_sqrt = (evecs * sqrt_evals) @ evecs.T
    X_inv_sqrt = (evecs * inv_sqrt_evals) @ evecs.T
    return X_sqrt, X_inv_sqrt


def _affine_log(X: Array, M: Array, eps: float) -> Array:
    X_sqrt, X_inv_sqrt = _matrix_sqrt_and_inv_sqrt(X, eps=eps)
    mid = sym(X_inv_sqrt @ M @ X_inv_sqrt)
    log_mid = spd_log(mid, eps=eps)
    return sym(X_sqrt @ log_mid @ X_sqrt)


def _affine_exp(X: Array, V: Array, eps: float) -> Array:
    X_sqrt, X_inv_sqrt = _matrix_sqrt_and_inv_sqrt(X, eps=eps)
    mid = sym(X_inv_sqrt @ V @ X_inv_sqrt)
    exp_mid = spd_exp(mid)
    return sym(X_sqrt @ exp_mid @ X_sqrt)


def _sym_from_vec(v: Array, *, dim: int) -> Array:
    # v has length m = d(d+1)/2, ordered by np.tril_indices (row-major).
    i, j = np.tril_indices(dim)
    m = i.shape[0]
    if int(v.shape[0]) != m:
        raise ValueError(f"Expected length {m} for symmetric vector, got {v.shape[0]}.")
    out = jnp.zeros((dim, dim), dtype=v.dtype)
    scale = 1.0 / jnp.sqrt(2.0)
    off = i != j
    out = out.at[i, j].set(jnp.where(off, v * scale, v))
    out = out.at[j, i].set(jnp.where(off, v * scale, v))
    return out


def _qv_vec_affine(
    *,
    X_sqrt: Array,
    chol: Array,
    noise_scale: float,
    dim: int,
) -> Array:
    num_paths = dim * dim
    m = dim * (dim + 1) // 2

    def mv(dB_flat: Array) -> Array:
        dW_flat = chol @ dB_flat
        dW_sym = _sym_from_vec(dW_flat, dim=dim)
        dX = sym(X_sqrt @ dW_sym @ X_sqrt) * noise_scale
        return dX.reshape(num_paths)

    basis = jnp.eye(m, dtype=X_sqrt.dtype)
    cols = jax.vmap(mv)(basis)
    return cols.T @ cols


def simulate_wishart_diffusion(
    *,
    key: Array,
    timesteps: int,
    T: float,
    X0: Array,
    b: Array,
    H: Array,
    Sigma: Array,
    corr_matrix: Array | None = None,
    tol: float = 1e-3,
    eps: float = 1e-6,
    noise_scale: float = 1.0,
    drift_mode: Literal["isotropic", "anisotropic"] = "isotropic",
) -> dict[str, Array]:
    """
    Simulate a mean-reverting SPD diffusion under affine-invariant geometry.

    Returns a dict with keys:
      - ts: time grid, shape (timesteps,)
      - X_path: SPD state path, shape (timesteps, d, d)
      - quadratic_variation: per-step increments Δ⟨vech(X)⟩, shape (timesteps-1, m, m)
    """
    if timesteps <= 1:
        raise ValueError("timesteps must be at least 2.")
    if T <= 0.0:
        raise ValueError("T must be positive.")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative.")
    if drift_mode not in ("isotropic", "anisotropic"):
        raise ValueError("drift_mode must be 'isotropic' or 'anisotropic'.")

    _validate_params(X0=X0, b=b, H=H, Sigma=Sigma, eps=eps)

    dim = int(X0.shape[0])
    ts = jnp.linspace(0.0, float(T), timesteps)
    m = dim * (dim + 1) // 2
    bm = dfx.VirtualBrownianTree(
        t0=0.0, t1=float(T), tol=float(tol), shape=(m,), key=key
    )
    chol = _correlation_factor(corr_matrix, m, X0.dtype)

    # Precompute vech selection indices (consistent with SPDManifold.vech: row-major + tril_indices)
    i, j = np.tril_indices(dim)
    lin = np.ravel_multi_index((i, j), (dim, dim)).astype(np.int32)
    lin_j = jnp.asarray(lin, dtype=jnp.int32)

    # Mean target is b (SPD). Build a PSD stiffness K from -H (trace-safe).
    target = sym(b)
    K = sym(-H)
    evals, evecs = jnp.linalg.eigh(K)
    evals = jnp.maximum(evals, 0.0)
    K = sym((evecs * evals) @ evecs.T)
    rate = jnp.trace(K) / jnp.asarray(dim, dtype=K.dtype)

    def _brownian_increment(t0: Array, t1: Array) -> Array:
        inc = bm.evaluate(t0, t1)
        W = inc.W if hasattr(inc, "W") else inc
        return jnp.asarray(W)

    dB = jax.vmap(_brownian_increment)(ts[:-1], ts[1:])  # (T-1, m)
    dt = float(T) / float(timesteps - 1)

    def _step(X: Array, dB_k: Array) -> tuple[Array, tuple[Array, Array]]:
        X_sym = sym(X)
        X_sqrt, X_inv_sqrt = _matrix_sqrt_and_inv_sqrt(X_sym, eps=eps)

        if drift_mode == "isotropic":
            # Log_X(target) points from X to target under AIRM.
            drift = jnp.asarray(rate, dtype=X.dtype) * _affine_log(
                X_sym, target, eps=eps
            )
        else:
            # Affine-invariant log at X pointing to target:
            mid = sym(X_inv_sqrt @ target @ X_inv_sqrt)
            mid_log = spd_log(mid, eps=eps)
            # Apply anisotropic linear operator in the "middle" coordinates.
            mid_drift = sym(K @ mid_log + mid_log @ K) * 0.5
            drift = sym(X_sqrt @ mid_drift @ X_sqrt)
        dW_flat = chol @ dB_k
        dW_sym = _sym_from_vec(dW_flat, dim=dim)
        noise = sym(X_sqrt @ dW_sym @ X_sqrt) * noise_scale

        V = drift * dt + noise
        X_next = _affine_exp(X_sym, V, eps=eps)

        qv_vec = _qv_vec_affine(
            X_sqrt=X_sqrt, chol=chol, noise_scale=noise_scale, dim=dim
        )
        # Convert density to per-step increment by multiplying by dt.
        qv_vech_inc = jnp.take(jnp.take(qv_vec, lin_j, axis=0), lin_j, axis=1) * dt
        return X_next, (X_sym, qv_vech_inc)

    X_last, (X_prefix, qv_prefix) = jax.lax.scan(_step, sym(X0), dB)

    X_path = jnp.concatenate([X_prefix, X_last[None, :, :]], axis=0)
    # qv_prefix already corresponds to per-step increments (T-1, m, m)
    return {"ts": ts, "X_path": X_path, "quadratic_variation": qv_prefix}
