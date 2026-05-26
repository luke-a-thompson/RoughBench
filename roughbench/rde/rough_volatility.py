from dataclasses import dataclass
from enum import StrEnum
from math import gamma, log
from typing import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
from jax import Array
from roughbench.drivers import (
    bm_driver,
    correlate_bm_driver_against_reference,
    riemann_liouville_driver,
)


class ModelFamily(StrEnum):
    BLACK_SCHOLES = "Black-Scholes"
    HESTON = "Heston"
    BERGOMI = "Bergomi"
    CLASSICAL_LOCAL_STOCHASTIC_VOLATILITY = "Classical Local Stochastic Volatility"
    ROUGH_BERGOMI = "Rough Bergomi"
    ROUGH_HESTON = "Rough Heston"
    QUADRATIC_ROUGH_HESTON = "Quadratic Rough Heston"


class NoiseFamily(StrEnum):
    ZERO = "Zero"
    INDEPENDENT_BROWNIAN = "Independent Brownian"
    PAPER_ROUGH_BERGOMI = "Paper Rough Bergomi"
    EXTERNAL = "External"


@dataclass(frozen=True, slots=True)
class BonesiniModelSpec:
    """
    Specification for the Bonesini RDE.

    Args:
        name: Model family.
        hurst: Hurst parameter.
        v_0: Forward volatility.
        nu: Vol-of-vol parameter.
        rho: Correlation between price and volatility Brownian motions.
        sigma: Multiplies dW_t in dS.
        g: dt drift in dS.
        tau: Multiplies dX_t in dV.
        varsigma: Multiplies dW_t in dV.
        h: dt drift in dV.

    Raises:
        ValueError: If the parameters are invalid.
    """

    name: ModelFamily

    hurst: float
    v_0: float
    nu: float | None
    rho: float | None

    sigma: Callable[[Array, Array, float], Array] | None  # Multiplies dW_t in dS
    g: Callable[[Array, Array, float], Array] | None  # dt drift in dS

    tau: Callable[[Array, Array, float], Array] | None  # Multiplies dX_t in dV
    varsigma: Callable[[Array, Array, float], Array] | None  # Multiplies dW_t in dV
    h: Callable[[Array, Array, float], Array] | None  # dt drift in dV
    initial_vol_state: float = 0.0
    noise_family: NoiseFamily = NoiseFamily.ZERO
    extra_params: dict[str, object] | None = None

    def __post_init__(self):
        # Skip validation when inside JAX transformations (parameters are tracers).
        params = [self.hurst, self.v_0, self.nu, self.rho, self.initial_vol_state]
        if any(isinstance(x, jax.Array) for x in params if x is not None):
            return

        if not (0.0 < float(self.hurst) < 1.0):
            raise ValueError(f"Hurst must be between 0 and 1. Got {self.hurst}")
        if float(self.v_0) < 0.0:
            raise ValueError(f"v_0 must be positive. Got {self.v_0}")
        if self.nu is not None and float(self.nu) < 0.0:
            raise ValueError(f"nu must be non-negative. Got {self.nu}")
        if self.rho is not None and not (-1.0 <= float(self.rho) <= 1.0):
            raise ValueError(f"rho must be between -1 and 1. Got {self.rho}")


def make_lead_lag_control(
    ts: jax.Array, X: jax.Array, W: jax.Array
) -> dfx.LinearInterpolation:
    """
    Delayed piecewise-linear control for the 2D path Z = (X_lag, W).

    This matches the paper's Wong-Zakai approximation:
      - W^Delta is the standard piecewise-linear interpolation of W.
      - X_lag^Delta is the one-step delayed piecewise-linear interpolation of X.

    If ts = (t_k) and X_0 = 0, then X_lag^Delta is represented on the grid by
    the samples (0, X_{t_0}, X_{t_1}, ..., X_{t_{N-1}}), so that on each
    interval [t_k, t_{k+1}] for k >= 1 it interpolates from X_{t_{k-1}} to
    X_{t_k}, and it is identically zero on [t_0, t_1].

    Inputs:
      ts: shape (N+1,)
      X, W: shape (N+1,) or (N+1, d) - here 1D per process is fine
    Returns:
      LinearInterpolation of Z on the original grid ts.
    """
    ts = jnp.asarray(ts)
    X = jnp.asarray(X).reshape(-1)
    W = jnp.asarray(W).reshape(-1)
    X_lag = jnp.concatenate([jnp.zeros((1,), dtype=X.dtype), X[:-1]], axis=0)
    Z = jnp.stack([X_lag, W], axis=1)
    return dfx.LinearInterpolation(ts=ts, ys=Z)


def build_terms_with_leadlag(
    model_spec: BonesiniModelSpec, Z_control: dfx.LinearInterpolation
):
    def f0(t: float, y: jax.Array, args: tuple[float, float, float]) -> jax.Array:
        """
        ODE terms integrated against time (dt).
        """
        s, v = y[0], y[1]
        price_dt = model_spec.g(s, v, t) if model_spec.g else 0.0
        vol_dt = model_spec.h(s, v, t) if model_spec.h else 0.0
        # Row 0 corresponds to price equation, row 1 to volatility equation.
        # Row 0 receives price drift, row 1 receives volatility drift.
        return jnp.array([price_dt, vol_dt])

    def vf_Z(t: float, y: jax.Array, args: tuple[float, float, float]) -> jax.Array:
        """
        Control terms integrated against the lead-lag control Z.
        """
        s, v = y[0], y[1]
        # columns: [X, W]
        col_X = jnp.array(
            [
                0.0,
                (model_spec.tau(s, v, t) if model_spec.tau else 0.0),
            ]
        )
        col_W = jnp.array(
            [
                (model_spec.sigma(s, v, t) if model_spec.sigma else 0.0),
                (model_spec.varsigma(s, v, t) if model_spec.varsigma else 0.0),
            ]
        )
        # Control matrix [0, sigma; tau, varsigma]. Row 0 corresponds to price equation, row 1 to volatility equation.
        # Price receives 0 * dX, sigma * dW
        # Volatility receives tau * dX, varsigma * dW
        return jnp.stack([col_X, col_W], axis=1)  # (state dim, control dim)

    return dfx.MultiTerm(dfx.ODETerm(f0), dfx.ControlTerm(vf_Z, control=Z_control))


def _build_noise_drivers(
    key: jax.Array,
    noise_timesteps: int,
    model_spec: BonesiniModelSpec,
) -> tuple[jax.Array, jax.Array]:
    """Build noise drivers (X, W) for the given model family."""
    key_W, key_B, key_V = jax.random.split(key, 3)

    W_path = bm_driver(key_W, noise_timesteps, 1)
    W = jnp.squeeze(W_path)

    if model_spec.noise_family == NoiseFamily.ZERO:
        X = jnp.zeros_like(W)
    elif model_spec.noise_family == NoiseFamily.INDEPENDENT_BROWNIAN:
        B_path = bm_driver(key_B, noise_timesteps, 1)
        X = jnp.squeeze(B_path)
    elif model_spec.noise_family == NoiseFamily.PAPER_ROUGH_BERGOMI:
        B_path = bm_driver(key_B, noise_timesteps, 1)
        W1_corr = correlate_bm_driver_against_reference(W_path, B_path, model_spec.rho)
        gamma_h = gamma(float(model_spec.hurst) + 0.5)
        X = jnp.squeeze(
            riemann_liouville_driver(
                key_V, noise_timesteps, model_spec.hurst, W1_corr
            )
        ) / gamma_h
    elif model_spec.noise_family == NoiseFamily.EXTERNAL:
        raise NotImplementedError(
            f"{model_spec.name} requires an externally supplied volatility path X. "
            "Use solve_wong_zakai(...) with explicit X and W drivers."
        )
    else:
        raise ValueError(f"Unknown noise family: {model_spec.noise_family}")

    return X, W


def _make_initial_state(s_0: float, model_spec: BonesiniModelSpec) -> jax.Array:
    return jnp.array([s_0, model_spec.initial_vol_state])


def get_bonesini_rde_params(
    key: jax.Array,
    noise_timesteps: int,
    model_spec: BonesiniModelSpec,
    s_0: float,
) -> tuple[jax.Array, dfx.MultiTerm]:
    """Generates a Bonesini RDE path."""
    X, W = _build_noise_drivers(key, noise_timesteps, model_spec)
    ts_noise = jnp.linspace(0.0, 1.0, noise_timesteps + 1)
    Z = make_lead_lag_control(ts_noise, X=X, W=W)
    terms = build_terms_with_leadlag(model_spec, Z_control=Z)
    return _make_initial_state(s_0, model_spec), terms


def get_bonesini_noise_drivers(
    key: jax.Array,
    noise_timesteps: int,
    model_spec: BonesiniModelSpec,
    s_0: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate initial condition and noise drivers (X, W) for vmapping over paths."""
    X, W = _build_noise_drivers(key, noise_timesteps, model_spec)
    return _make_initial_state(s_0, model_spec), X, W


def solve_wong_zakai(
    y_0: jax.Array,
    X: jax.Array,
    W: jax.Array,
    model_spec: BonesiniModelSpec,
    noise_timesteps: int,
    rde_timesteps: int,
) -> dfx.Solution:
    if noise_timesteps > rde_timesteps:
        raise ValueError("Noise timesteps must be less than or equal to RDE timesteps.")

    ts_noise = jnp.linspace(0.0, 1.0, noise_timesteps + 1)
    Z = make_lead_lag_control(ts_noise, X=X, W=W)
    terms = build_terms_with_leadlag(model_spec, Z_control=Z)

    return dfx.diffeqsolve(
        terms=terms,
        solver=dfx.Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1.0 / rde_timesteps,
        y0=y_0,
        saveat=dfx.SaveAt(ts=ts_noise),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=None,
    )


def simulate_rough_heston_variance_paths(
    key: jax.Array,
    *,
    num_paths: int,
    noise_timesteps: int,
    v_0: float,
    hurst: float,
    nu: float,
    rho: float,
    lambda_: float,
    v_bar: float,
    truncation_eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """
    O(N^2) reference discretisation of the rough Heston variance equation.

    Returns:
      variance_paths: shape (num_paths, noise_timesteps + 1)
      price_brownian_paths: shape (num_paths, noise_timesteps + 1)
    """
    dt = 1.0 / noise_timesteps
    kernel_weights = (
        (jnp.arange(1, noise_timesteps + 1, dtype=jnp.float32) * dt)
        ** (hurst - 0.5)
    ) / gamma(hurst + 0.5)
    rho_bar = jnp.sqrt(1.0 - rho**2)

    keys = jax.random.split(key, 2 * num_paths)
    W_keys = keys[:num_paths]
    B_keys = keys[num_paths:]
    W_paths = jax.vmap(lambda k: jnp.squeeze(bm_driver(k, noise_timesteps, 1)))(W_keys)
    B_paths = jax.vmap(lambda k: jnp.squeeze(bm_driver(k, noise_timesteps, 1)))(B_keys)

    dW = jnp.diff(W_paths, axis=1)
    dB = jnp.diff(B_paths, axis=1)
    dZ = rho * dW + rho_bar * dB

    V = jnp.full((num_paths, noise_timesteps + 1), v_0, dtype=W_paths.dtype)
    for k in range(1, noise_timesteps + 1):
        V_past = V[:, :k]
        drift = -lambda_ * (V_past - v_bar) * dt
        diffusion = nu * jnp.sqrt(jnp.maximum(V_past, truncation_eps)) * dZ[:, :k]
        weights = kernel_weights[:k][::-1]
        V_k = v_0 + jnp.sum((drift + diffusion) * weights[None, :], axis=1)
        V = V.at[:, k].set(jnp.maximum(V_k, truncation_eps))

    return V, W_paths


def simulate_quadratic_rough_heston_variance_paths(
    key: jax.Array,
    *,
    num_paths: int,
    noise_timesteps: int,
    v_0: float,
    hurst: float,
    a: float,
    b: float,
    c: float,
    lambda_: float,
    eta: float,
    theta_level: float | None = None,
    truncation_eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """
    O(N^2) reference discretisation of the quadratic rough Heston variance equation.

    Returns:
      variance_paths: shape (num_paths, noise_timesteps + 1)
      price_brownian_paths: shape (num_paths, noise_timesteps + 1)
    """
    dt = 1.0 / noise_timesteps
    kernel_weights = (
        (jnp.arange(1, noise_timesteps + 1, dtype=jnp.float32) * dt)
        ** (hurst - 0.5)
    ) / gamma(hurst + 0.5)
    theta = v_0 if theta_level is None else theta_level

    W_keys = jax.random.split(key, num_paths)
    W_paths = jax.vmap(lambda k: jnp.squeeze(bm_driver(k, noise_timesteps, 1)))(W_keys)
    dW = jnp.diff(W_paths, axis=1)

    V = jnp.full((num_paths, noise_timesteps + 1), v_0, dtype=W_paths.dtype)
    for k in range(1, noise_timesteps + 1):
        V_past = V[:, :k]
        quadratic = a * ((V_past - b) ** 2) + c
        drift = (theta - V_past) * dt
        diffusion = eta * jnp.sqrt(jnp.maximum(quadratic, truncation_eps)) * dW[:, :k]
        weights = kernel_weights[:k][::-1]
        V_k = v_0 + lambda_ * jnp.sum((drift + diffusion) * weights[None, :], axis=1)
        V = V.at[:, k].set(jnp.maximum(V_k, truncation_eps))

    return V, W_paths


def _solve_from_variance_path(
    s_0: float,
    variance_path: jax.Array,
    price_brownian_path: jax.Array,
    model_spec: BonesiniModelSpec,
    noise_timesteps: int,
    rde_timesteps: int,
) -> dfx.Solution:
    variance_path = jnp.asarray(variance_path).reshape(-1)
    price_brownian_path = jnp.asarray(price_brownian_path).reshape(-1)

    if variance_path.shape[0] != noise_timesteps + 1:
        raise ValueError(
            "variance_path must have length noise_timesteps + 1. "
            f"Got {variance_path.shape[0]} and noise_timesteps={noise_timesteps}."
        )
    if price_brownian_path.shape[0] != noise_timesteps + 1:
        raise ValueError(
            "price_brownian_path must have length noise_timesteps + 1. "
            f"Got {price_brownian_path.shape[0]} and noise_timesteps={noise_timesteps}."
        )

    centered_variance = variance_path - variance_path[0]
    # The Wong-Zakai control uses the delayed interpolation X_lag. Shift the
    # supplied variance path forward by one step so the saved second state
    # aligns with the user-supplied variance levels on the original grid.
    centered_variance_driver = jnp.concatenate(
        [centered_variance[1:], centered_variance[-1:]], axis=0
    )
    y_0 = jnp.array([s_0, variance_path[0]])
    return solve_wong_zakai(
        y_0=y_0,
        X=centered_variance_driver,
        W=price_brownian_path,
        model_spec=model_spec,
        noise_timesteps=noise_timesteps,
        rde_timesteps=rde_timesteps,
    )


def solve_rough_heston_from_variance_path(
    s_0: float,
    variance_path: jax.Array,
    price_brownian_path: jax.Array,
    *,
    v_0: float,
    hurst: float,
    nu: float,
    rho: float,
    lambda_: float,
    v_bar: float,
    noise_timesteps: int,
    rde_timesteps: int,
) -> dfx.Solution:
    model_spec = make_rough_heston_model_spec(
        v_0=v_0, hurst=hurst, nu=nu, rho=rho, lambda_=lambda_, v_bar=v_bar
    )
    return _solve_from_variance_path(
        s_0=s_0,
        variance_path=variance_path,
        price_brownian_path=price_brownian_path,
        model_spec=model_spec,
        noise_timesteps=noise_timesteps,
        rde_timesteps=rde_timesteps,
    )


def solve_quadratic_rough_heston_from_variance_path(
    s_0: float,
    variance_path: jax.Array,
    price_brownian_path: jax.Array,
    *,
    v_0: float,
    hurst: float,
    a: float,
    b: float,
    c: float,
    lambda_: float,
    eta: float,
    noise_timesteps: int,
    rde_timesteps: int,
    theta: Callable[[float], Array] | None = None,
) -> dfx.Solution:
    model_spec = make_quadratic_rough_heston_model_spec(
        v_0=v_0,
        hurst=hurst,
        a=a,
        b=b,
        c=c,
        lambda_=lambda_,
        eta=eta,
        theta=theta,
    )
    return _solve_from_variance_path(
        s_0=s_0,
        variance_path=variance_path,
        price_brownian_path=price_brownian_path,
        model_spec=model_spec,
        noise_timesteps=noise_timesteps,
        rde_timesteps=rde_timesteps,
    )


def make_black_scholes_model_spec(v_0: float) -> BonesiniModelSpec:
    return BonesiniModelSpec(
        name=ModelFamily.BLACK_SCHOLES,
        hurst=0.5,
        v_0=v_0,
        nu=0.0,
        rho=0.0,
        sigma=lambda s, v, t: s * jnp.sqrt(v_0),
        g=lambda s, v, t: -0.5 * s * v_0,
        tau=None,
        varsigma=None,
        h=None,
        initial_vol_state=0.0,
        noise_family=NoiseFamily.ZERO,
    )


def make_bergomi_model_spec(v_0: float, rho: float) -> BonesiniModelSpec:
    rho_bar = jnp.sqrt(1.0 - rho**2)
    # V is the log-variance state: stock vol = exp(V).
    # To match the convention that v_0 is the initial instantaneous variance
    # (as in Black-Scholes and rough Bergomi), we set V_0 = 0.5*log(v_0)
    # so that exp(V_0) = sqrt(v_0) and exp(V_0)^2 = v_0.
    v_state_0 = 0.5 * log(v_0)
    return BonesiniModelSpec(
        name=ModelFamily.BERGOMI,
        hurst=0.5,
        v_0=v_0,
        nu=None,
        rho=rho,
        sigma=lambda s, v, t: s * jnp.exp(v),
        g=lambda s, v, t: -0.5 * s * (jnp.exp(2.0 * v) + rho * v * jnp.exp(v)),
        tau=lambda s, v, t: rho_bar * v,
        varsigma=lambda s, v, t: rho * v,
        h=lambda s, v, t: -0.5 * v,
        initial_vol_state=v_state_0,
        noise_family=NoiseFamily.INDEPENDENT_BROWNIAN,
    )


def make_rough_bergomi_model_spec(
    v_0: float, nu: float, hurst: float, rho: float
) -> BonesiniModelSpec:
    gamma_h = gamma(hurst + 0.5)
    return BonesiniModelSpec(
        name=ModelFamily.ROUGH_BERGOMI,
        hurst=hurst,
        v_0=v_0,
        nu=nu,
        rho=rho,
        # The RL driver is normalised to Var[X_t] = t^{2H}; dividing by
        # Gamma(H + 1/2) matches the paper convention used here.
        # Relative to simple_rbergomi.py, the same stock-volatility law is
        # obtained after reparameterising eta_simple = 2 * nu / Gamma(H + 1/2).
        sigma=lambda s, v, t: (
            s
            * jnp.sqrt(v_0)
            * jnp.exp(
                nu * v - 0.5 * (nu**2) * (t ** (2.0 * hurst)) / (gamma_h**2)
            )
        ),
        g=lambda s, v, t: (
            -0.5
            * s
            * v_0
            * jnp.exp(
                2.0 * nu * v - (nu**2) * (t ** (2.0 * hurst)) / (gamma_h**2)
            )
        ),
        tau=lambda s, v, t: 1.0,  # dV = dX
        varsigma=None,
        h=None,
        initial_vol_state=0.0,
        noise_family=NoiseFamily.PAPER_ROUGH_BERGOMI,
    )


def make_heston_model_spec(
    v_0: float, rho: float, nu: float, lambda_: float, v_bar: float
) -> BonesiniModelSpec:
    rho_bar = jnp.sqrt(1.0 - rho**2)
    return BonesiniModelSpec(
        name=ModelFamily.HESTON,
        hurst=0.5,
        v_0=v_0,
        nu=nu,
        rho=rho,
        sigma=lambda s, v, t: s * jnp.sqrt(jnp.maximum(v, 0.0)),
        g=lambda s, v, t: -0.5 * s * (v + 0.5 * rho * nu),
        tau=lambda s, v, t: rho_bar * nu * jnp.sqrt(jnp.maximum(v, 0.0)),
        varsigma=lambda s, v, t: rho * nu * jnp.sqrt(jnp.maximum(v, 0.0)),
        h=lambda s, v, t: -lambda_ * (v - v_bar) - 0.25 * (nu**2),
        initial_vol_state=v_0,
        noise_family=NoiseFamily.INDEPENDENT_BROWNIAN,
        extra_params={"lambda": lambda_, "v_bar": v_bar},
    )


def make_classical_local_stochastic_volatility_model_spec(
    v_0: float,
    rho: float,
    xi: Callable[[float, Array, Array], Array],
    f_1: Callable[[float, Array], Array],
    f_2: Callable[[Array], Array],
) -> BonesiniModelSpec:
    rho_bar = jnp.sqrt(1.0 - rho**2)

    def sigma(s: Array, v: Array, t: float) -> Array:
        return s * xi(t, s, v)

    def g(s: Array, v: Array, t: float) -> Array:
        sigma_s = jax.grad(lambda s_: sigma(s_, v, t))(s)
        sigma_v = jax.grad(lambda v_: sigma(s, v_, t))(v)
        return -0.5 * (sigma(s, v, t) * sigma_s + rho * f_2(v) * sigma_v)

    return BonesiniModelSpec(
        name=ModelFamily.CLASSICAL_LOCAL_STOCHASTIC_VOLATILITY,
        hurst=0.5,
        v_0=v_0,
        nu=None,
        rho=rho,
        sigma=sigma,
        g=g,
        tau=lambda s, v, t: rho_bar * f_2(v),
        varsigma=lambda s, v, t: rho * f_2(v),
        h=lambda s, v, t: f_1(t, v) - 0.5 * jax.grad(f_2)(v) * f_2(v),
        initial_vol_state=v_0,
        noise_family=NoiseFamily.INDEPENDENT_BROWNIAN,
    )


def make_rough_heston_model_spec(
    v_0: float, hurst: float, nu: float, rho: float, lambda_: float, v_bar: float
) -> BonesiniModelSpec:
    return BonesiniModelSpec(
        name=ModelFamily.ROUGH_HESTON,
        hurst=hurst,
        v_0=v_0,
        nu=nu,
        rho=rho,
        sigma=lambda s, v, t: s * jnp.sqrt(jnp.maximum(v, 0.0)),
        g=lambda s, v, t: -0.5 * s * v,
        tau=lambda s, v, t: 1.0,
        varsigma=None,
        h=lambda s, v, t: 0.0,
        initial_vol_state=v_0,
        noise_family=NoiseFamily.EXTERNAL,
        extra_params={"lambda": lambda_, "v_bar": v_bar},
    )


def make_quadratic_rough_heston_model_spec(
    v_0: float,
    hurst: float,
    a: float,
    b: float,
    c: float,
    lambda_: float,
    eta: float,
    theta: Callable[[float], Array] | None = None,
) -> BonesiniModelSpec:
    def quad(v: Array) -> Array:
        return a * ((v - b) ** 2) + c

    return BonesiniModelSpec(
        name=ModelFamily.QUADRATIC_ROUGH_HESTON,
        hurst=hurst,
        v_0=v_0,
        nu=eta,
        rho=None,
        sigma=lambda s, v, t: s * jnp.sqrt(quad(v)),
        g=lambda s, v, t: -s * quad(v),
        tau=lambda s, v, t: 1.0,
        varsigma=None,
        h=lambda s, v, t: 0.0,
        initial_vol_state=v_0,
        noise_family=NoiseFamily.EXTERNAL,
        extra_params={
            "a": a,
            "b": b,
            "c": c,
            "lambda": lambda_,
            "theta": theta,
        },
    )
