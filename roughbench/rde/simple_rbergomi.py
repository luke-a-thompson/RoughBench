import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import (
    ControlTerm,
    Euler,
    LinearInterpolation,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)


def _tbss_kernel(x: jnp.ndarray, a: float) -> jnp.ndarray:
    return x**a


def _optimal_node(k: jnp.ndarray, a: float) -> jnp.ndarray:
    """Optimal TBSS discretisation node minimising hybrid scheme error."""
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def _hybrid_covariance(a: float, n: int) -> jnp.ndarray:
    """Covariance matrix for hybrid scheme, kappa=1."""
    off = 1.0 / ((a + 1) * n ** (a + 1))
    return jnp.array([[1.0 / n, off], [off, 1.0 / ((2 * a + 1) * n ** (2 * a + 1))]])


class rBergomi(eqx.Module):
    """
    Rough Bergomi model via hybrid scheme + diffrax SDE solver.

    The Volterra variance process is non-Markovian, so it is computed via the
    hybrid scheme using JAX. The log-price SDE is then expressed as a diffrax
    MultiTerm (ODETerm drift + ControlTerm diffusion) and solved with
    diffeqsolve, using the precomputed variance path as a LinearInterpolation
    control.

    VirtualBrownianTree provides the underlying noise for both processes.
    """

    n: int = eqx.field(static=True)
    s: int = eqx.field(static=True)
    T: float
    a: float
    rho: float
    eta: float
    xi: float

    def __init__(
        self,
        n: int = 100,
        T: float = 1.0,
        a: float = -0.4,
        rho: float = -0.7,
        eta: float = 1.5,
        xi: float = 0.235**2,
    ) -> None:
        if 2 * a + 1 <= 0:
            raise ValueError(f"Need 2*a+1 > 0 for sqrt(2*a+1); got a={a}.")
        self.n = n
        self.s = int(round(n * T)) + 1
        if self.s < 2:
            raise ValueError(f"Need at least 2 grid points; got s={self.s}.")
        self.T = T
        self.a = a
        self.rho = rho
        self.eta = eta
        self.xi = xi

    def _variance_and_bm(
        self, key_var: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Sample the variance path V and hybrid BM increments dW via the hybrid
        scheme, using VirtualBrownianTree as the underlying noise source.

        Returns (V, dW, t_grid) where:
          V     : variance process, shape (s,)
          dW    : hybrid-covariance increments, shape (s-1, 2)
          t_grid: time grid, shape (s,)
        """
        dt = self.T / self.n
        t_grid = jnp.arange(self.s) * dt

        # 2D VirtualBrownianTree: increments are N(0, dt * I_2)
        bm_var = VirtualBrownianTree(
            t0=0.0, t1=float(self.T), tol=dt * 0.1, shape=(2,), key=key_var
        )
        raw_inc = jax.vmap(lambda t: bm_var.evaluate(t, t + dt))(t_grid[:-1])

        # Whiten then re-colour with hybrid covariance C so each row ~ N(0, C)
        L = jnp.linalg.cholesky(_hybrid_covariance(self.a, self.n))
        dW = (raw_inc / jnp.sqrt(dt)) @ L.T  # (s-1, 2)

        # Volterra process via hybrid scheme
        Y1 = jnp.zeros(self.s).at[1:].set(dW[:, 1])  # exact integrals, kappa=1

        k = jnp.arange(2, self.s)
        G = jnp.zeros(self.s + 1).at[2 : self.s].set(
            _tbss_kernel(_optimal_node(k, self.a) / self.n, self.a)
        )
        Y2 = jnp.convolve(G, dW[:, 0])[: self.s]

        Y = jnp.sqrt(2 * self.a + 1) * (Y1 + Y2)

        V = self.xi * jnp.exp(
            self.eta * Y - 0.5 * self.eta**2 * t_grid ** (2 * self.a + 1)
        )
        return V, dW, t_grid

    def _price_sde(
        self,
        V: jnp.ndarray,
        dB: jnp.ndarray,
        t_grid: jnp.ndarray,
        S0: float = 1.0,
    ) -> jnp.ndarray:
        """
        Solve the log-price SDE via diffeqsolve.

          d(log S) = -V(t)/2 dt + sqrt(V(t)) dB

        V is supplied as a LinearInterpolation; dB is the pre-built correlated
        BM path passed as a ControlTerm control.
        """
        dt = t_grid[1] - t_grid[0]
        var_interp = LinearInterpolation(ts=t_grid, ys=V)
        B_path = LinearInterpolation(
            ts=t_grid,
            ys=jnp.concatenate([jnp.zeros(1), jnp.cumsum(dB)]),
        )

        def drift(t, log_s, args):
            return -0.5 * var_interp.evaluate(t)

        def diffusion(t, log_s, args):
            return jnp.sqrt(var_interp.evaluate(t))

        sol = diffeqsolve(
            MultiTerm(ODETerm(drift), ControlTerm(diffusion, B_path)),
            solver=Euler(),
            t0=t_grid[0],
            t1=t_grid[-1],
            dt0=dt,
            y0=jnp.log(jnp.array(S0)),
            saveat=SaveAt(ts=t_grid),
            max_steps=self.s,
        )
        return jnp.exp(sol.ys)

    def simulate_single(self, key: jax.Array, S0: float = 1.0) -> jnp.ndarray:
        """Simulate a single path. Returns S of shape (s,)."""
        key_var, key_price = jax.random.split(key)

        V, dW, t_grid = self._variance_and_bm(key_var)

        bm_price = VirtualBrownianTree(
            t0=0.0, t1=float(self.T), tol=(self.T / self.n) * 0.1, shape=(), key=key_price
        )
        dW2 = jax.vmap(lambda t: bm_price.evaluate(t, t + self.T / self.n))(t_grid[:-1])
        dB = self.rho * dW[:, 0] + jnp.sqrt(1 - self.rho**2) * dW2

        return self._price_sde(V, dB, t_grid, S0)

    def simulate_single_partial(
        self, key: jax.Array, rho: float, S0: float = 1.0
    ) -> jnp.ndarray:
        """Price process driven by the variance Brownian component only."""
        key_var, _ = jax.random.split(key)
        V, dW, t_grid = self._variance_and_bm(key_var)
        dB = rho * dW[:, 0]
        return self._price_sde(V, dB, t_grid, S0)

    def simulate(self, N: int, key: jax.Array, S0: float = 1.0) -> jnp.ndarray:
        """Simulate N paths. Returns array of shape (N, s)."""
        return jax.vmap(lambda k: self.simulate_single(k, S0))(jax.random.split(key, N))


if __name__ == "__main__":
    model = rBergomi()
    paths = model.simulate(100, jax.random.PRNGKey(0))
    print(paths)
