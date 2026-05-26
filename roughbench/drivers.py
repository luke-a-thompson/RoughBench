from __future__ import annotations

import jax
import jax.numpy as jnp


def bm_driver(key: jax.Array, timesteps: int, dim: int) -> jax.Array:
    """Generate a Brownian path on [0, 1] with shape (timesteps + 1, dim)."""
    dt = 1.0 / float(timesteps)
    increments = jax.random.normal(key, (timesteps, dim)) * jnp.sqrt(dt)
    return jnp.concatenate(
        [jnp.zeros((1, dim), dtype=increments.dtype), jnp.cumsum(increments, axis=0)],
        axis=0,
    )


def correlate_bm_driver_against_reference(
    reference_path: jax.Array, independent_path: jax.Array, rho: float
) -> jax.Array:
    rho_bar = jnp.sqrt(1.0 - rho**2)
    return rho * reference_path + rho_bar * independent_path


def _davies_harte_from_normals(
    normals: jax.Array, timesteps: int, hurst: float
) -> jax.Array:
    n = int(timesteps)
    k = jnp.arange(n, dtype=normals.dtype)
    gamma_k = 0.5 * (
        jnp.abs(k - 1.0) ** (2.0 * hurst)
        - 2.0 * (k ** (2.0 * hurst))
        + (k + 1.0) ** (2.0 * hurst)
    )
    circulant = jnp.concatenate(
        [
            gamma_k,
            jnp.zeros((1,), dtype=gamma_k.dtype),
            gamma_k[1:][::-1],
        ]
    )
    eigenvalues = jnp.maximum(jnp.real(jnp.fft.fft(circulant)), 0.0)
    increments = jnp.real(
        jnp.fft.ifft(
            jnp.sqrt(eigenvalues)[:, None] * jnp.fft.fft(normals, axis=0),
            axis=0,
        )
    )[:n]
    return increments * ((1.0 / float(timesteps)) ** hurst)


def fractional_bm_driver(
    key: jax.Array, timesteps: int, dim: int, hurst: float
) -> jax.Array:
    """Generate fractional Brownian motion using a JAX Davies-Harte embedding."""
    if not (0.0 < float(hurst) < 1.0):
        raise ValueError(f"hurst must be in (0, 1). Got {hurst}")

    normals = jax.random.normal(key, (2 * timesteps, dim), dtype=jnp.float32)
    increments = _davies_harte_from_normals(normals, timesteps, hurst)
    return jnp.concatenate(
        [jnp.zeros((1, dim), dtype=increments.dtype), jnp.cumsum(increments, axis=0)],
        axis=0,
    )


def riemann_liouville_driver(
    key: jax.Array, timesteps: int, hurst: float, brownian_driver: jax.Array
) -> jax.Array:
    del key
    brownian = jnp.asarray(brownian_driver)
    if brownian.ndim == 1:
        brownian = brownian[:, None]
    dt = 1.0 / float(timesteps)
    source_normals = jnp.diff(brownian, axis=0) / jnp.sqrt(dt)
    k = jnp.arange(timesteps, dtype=source_normals.dtype)
    gamma_k = 0.5 * (
        jnp.abs(k - 1.0) ** (2.0 * hurst)
        - 2.0 * (k ** (2.0 * hurst))
        + (k + 1.0) ** (2.0 * hurst)
    )
    toeplitz_index = jnp.abs(
        jnp.arange(timesteps)[:, None] - jnp.arange(timesteps)[None, :]
    )
    covariance = gamma_k[toeplitz_index]
    covariance = covariance + 1e-6 * jnp.eye(timesteps, dtype=source_normals.dtype)
    chol = jnp.linalg.cholesky(covariance)
    increments = (chol @ source_normals) * (dt**hurst)
    path = jnp.cumsum(increments, axis=0)
    return jnp.concatenate(
        [jnp.zeros((1, path.shape[1]), dtype=path.dtype), path], axis=0
    )
