import numpy as np


def g(x: float, a: float) -> float:
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a


def b(k: int, a: float) -> float:
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def cov(a: float, n: int) -> np.ndarray:
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.0, 0.0], [0.0, 0.0]])
    cov[0, 0] = 1.0 / n
    cov[0, 1] = 1.0 / ((1.0 * a + 1) * n ** (1.0 * a + 1))
    cov[1, 1] = 1.0 / ((2.0 * a + 1) * n ** (2.0 * a + 1))
    cov[1, 0] = cov[0, 1]
    return cov


class rBergomi(object):
    """
    Class for generating paths of the rBergomi model.
    """

    def __init__(
        self,
        n: int = 100,
        N: int = 1000,
        T: float = 1.00,
        a: float = -0.4,
        rho: float = -0.7,
        eta: float = 1.5,
        xi: float = 0.235**2,
    ) -> None:
        """
        Constructor for class.
        """
        # Basic assignments
        self.params = [xi, eta, rho, a + 1 / 2]
        self.T = T  # Maturity
        self.n = n  # Granularity (steps per year)
        # Use a consistent grid: m steps of size dt, and (m+1) grid points.
        self.dt = 1.0 / self.n  # Step size
        self.s = int(round(self.n * self.T)) + 1  # Number of grid points
        self.t = (np.arange(self.s) * self.dt)[np.newaxis, :]  # Time grid
        self.a = a  # Alpha
        self.N = N  # Paths
        self.rho = rho
        self.eta = eta
        self.xi = xi

        if self.s < 2:
            raise ValueError(f"Need at least 2 grid points; got s={self.s}.")
        if 2 * self.a + 1 <= 0:
            raise ValueError(
                f"Need 2*a + 1 > 0 for sqrt(2*a+1); got a={self.a}."
            )

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0, 0])
        self.c = cov(self.a, self.n)

    def dW1(self) -> np.ndarray:
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s - 1))

    def Y(self, dW: np.ndarray) -> np.ndarray:
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((self.N, self.s))  # Exact integrals
        # Y2 = np.zeros((self.N, 1 + self.s))  # Riemann sums

        # Construct Y1 through exact integral
        for i in range(1, self.s):
            Y1[:, i] = dW[:, i - 1, 1]  # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.s)  # Gamma
        for k in range(2, self.s):
            G[k] = g(b(k, self.a) / self.n, self.a)

        X = dW[:, :, 0]  # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.N, len(X[0, :]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(self.N):
            GX[i, :] = np.convolve(G, X[i, :])

        # Extract appropriate part of convolution
        Y2 = GX[:, : self.s]

        # Finally contruct and return full process
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def dW2(self) -> np.ndarray:
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s - 1) * np.sqrt(self.dt)

    def dB(self, dW1: np.ndarray, dW2: np.ndarray, rho: float | None = None) -> np.ndarray:
        """
        Constructs correlated price Brownian increments, dB.
        """
        rho_ = self.rho if rho is None else rho
        self.rho = rho_
        dB = rho_ * dW1[:, :, 0] + np.sqrt(1 - rho_**2) * dW2
        return dB

    def V(self, Y: np.ndarray, xi: float | None = None, eta: float | None = None) -> np.ndarray:
        """
        rBergomi variance process.
        """
        xi_ = self.xi if xi is None else xi
        eta_ = self.eta if eta is None else eta
        self.xi = xi_
        self.eta = eta_
        a = self.a
        t = self.t
        V = xi_ * np.exp(eta_ * Y - 0.5 * eta_**2 * t ** (2 * a + 1))
        return V

    def S(self, V: np.ndarray, dB: np.ndarray, S0: float = 1.0) -> np.ndarray:
        """
        rBergomi price process.
        """
        self.S0 = S0
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

    def S1(self, V: np.ndarray, dW1: np.ndarray, rho: float, S0: float = 1.0) -> np.ndarray:
        """
        rBergomi parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = rho * np.sqrt(V[:, :-1]) * dW1[:, :, 0] - 0.5 * rho**2 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

if __name__ == "__main__":
    rb = rBergomi()
    dW1 = rb.dW1()
    dW2 = rb.dW2()
    dB = rb.dB(dW1, dW2)
    Y = rb.Y(dW1)
    V = rb.V(Y)
    S = rb.S(V, dB)
    print(S)