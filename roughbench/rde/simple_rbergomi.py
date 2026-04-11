import numpy as np


def _tbss_kernel(x: float, a: float) -> float:
    return x**a


def _optimal_node(k: int, a: float) -> float:
    """Optimal TBSS discretisation node minimising hybrid scheme error."""
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def _hybrid_covariance(a: float, n: int) -> np.ndarray:
    """Covariance matrix for hybrid scheme, kappa=1."""
    off = 1.0 / ((a + 1) * n ** (a + 1))
    return np.array([[1.0 / n, off], [off, 1.0 / ((2 * a + 1) * n ** (2 * a + 1))]])


class rBergomi:
    """Generates paths of the rough Bergomi model via the hybrid scheme."""

    def __init__(
        self,
        n: int = 100,
        N: int = 1000,
        T: float = 1.0,
        a: float = -0.4,
        rho: float = -0.7,
        eta: float = 1.5,
        xi: float = 0.235**2,
    ) -> None:
        self.T = T
        self.n = n
        self.dt = 1.0 / n
        self.s = int(round(n * T)) + 1
        self.t = (np.arange(self.s) * self.dt)[np.newaxis, :]
        self.a = a
        self.N = N
        self.rho = rho
        self.eta = eta
        self.xi = xi

        if self.s < 2:
            raise ValueError(f"Need at least 2 grid points; got s={self.s}.")
        if 2 * a + 1 <= 0:
            raise ValueError(f"Need 2*a + 1 > 0 for sqrt(2*a+1); got a={a}.")

        self.c = _hybrid_covariance(a, n)

    def generate_variance_increments(self) -> np.ndarray:
        """Correlated 2d Brownian increments for the variance process."""
        return np.random.multivariate_normal(np.zeros(2), self.c, (self.N, self.s - 1))

    def generate_price_increments(self) -> np.ndarray:
        """Independent Brownian increments for the price process."""
        return np.random.randn(self.N, self.s - 1) * np.sqrt(self.dt)

    def volterra_process(self, dW: np.ndarray) -> np.ndarray:
        """Volterra process via hybrid scheme from correlated 2d Brownian increments."""
        Y1 = np.zeros((self.N, self.s))
        Y1[:, 1:] = dW[:, :, 1]  # Exact integrals, kappa=1

        G = np.zeros(1 + self.s)
        k = np.arange(2, self.s)
        G[2 : self.s] = _tbss_kernel(_optimal_node(k, self.a) / self.n, self.a)

        X = dW[:, :, 0]
        Y2 = np.array([np.convolve(G, X[i])[: self.s] for i in range(self.N)])

        return np.sqrt(2 * self.a + 1) * (Y1 + Y2)

    def correlate_increments(self, dW1: np.ndarray, dW2: np.ndarray) -> np.ndarray:
        """Correlated price Brownian increments from dW1 and orthogonal dW2."""
        return self.rho * dW1[:, :, 0] + np.sqrt(1 - self.rho**2) * dW2

    def variance_process(self, Y: np.ndarray) -> np.ndarray:
        """rBergomi variance process."""
        return self.xi * np.exp(
            self.eta * Y - 0.5 * self.eta**2 * self.t ** (2 * self.a + 1)
        )

    def price_process(
        self, V: np.ndarray, dB: np.ndarray, S0: float = 1.0
    ) -> np.ndarray:
        """rBergomi price process."""
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * self.dt
        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))
        return S

    def partial_price_process(
        self, V: np.ndarray, dW1: np.ndarray, rho: float, S0: float = 1.0
    ) -> np.ndarray:
        """Price process driven by the variance Brownian component only."""
        increments = (
            rho * np.sqrt(V[:, :-1]) * dW1[:, :, 0] - 0.5 * rho**2 * V[:, :-1] * self.dt
        )
        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))
        return S


if __name__ == "__main__":
    rb = rBergomi()
    dW1 = rb.generate_variance_increments()
    dW2 = rb.generate_price_increments()
    dB = rb.correlate_increments(dW1, dW2)
    Y = rb.volterra_process(dW1)
    V = rb.variance_process(Y)
    S = rb.price_process(V, dB)
    print(S)
