import jax
import jax.numpy as jnp
import pytest
from math import gamma

from roughbench.rde.rough_volatility import (
    make_classical_local_stochastic_volatility_model_spec,
    make_black_scholes_model_spec,
    make_bergomi_model_spec,
    make_heston_model_spec,
    make_quadratic_rough_heston_model_spec,
    make_rough_bergomi_model_spec,
    make_rough_heston_model_spec,
    get_bonesini_noise_drivers,
    simulate_quadratic_rough_heston_variance_paths,
    simulate_rough_heston_variance_paths,
    solve_quadratic_rough_heston_from_variance_path,
    solve_rough_heston_from_variance_path,
    solve_wong_zakai,
)


# --- Common utilities ---


def _simulate_model(
    model_spec,
    *,
    num_paths: int,
    noise_timesteps: int,
    rde_timesteps: int,
    s0: float,
    seed: int,
):
    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    y0s, Xs, Ws = jax.vmap(
        lambda k: get_bonesini_noise_drivers(
            k, noise_timesteps=noise_timesteps, model_spec=model_spec, s_0=s0
        )
    )(keys)

    sols = jax.vmap(
        lambda y0, X, W: solve_wong_zakai(
            y0,
            X,
            W,
            model_spec=model_spec,
            noise_timesteps=noise_timesteps,
            rde_timesteps=rde_timesteps,
        )
    )(y0s, Xs, Ws)

    Ys = sols.ys
    assert Ys is not None
    return Ys[..., 0], Ys[..., 1], Xs, Ws


def _assert_martingale(S_paths: jax.Array, s0: float, atol: float = 1e-2) -> None:
    ST = S_paths[:, -1]
    mean_ST = jnp.mean(ST)
    se = jnp.std(ST, ddof=1) / jnp.sqrt(S_paths.shape[0])
    assert jnp.abs(mean_ST - s0) <= 4.0 * se + atol


def _assert_no_static_arbitrage(
    S_paths: jax.Array,
    *,
    s0: float,
    maturities: tuple[float, float] = (0.5, 1.0),
    strike_multipliers: tuple[float, float, float] = (0.8, 1.0, 1.2),
) -> None:
    T = S_paths.shape[1] - 1
    time_indices = [max(1, int(T * tau)) for tau in maturities]
    strikes = jnp.array(strike_multipliers) * s0

    prices = []
    ses = []
    for idx in time_indices:
        ST = S_paths[:, idx]
        payoffs = jnp.maximum(ST[:, None] - strikes[None, :], 0.0)
        prices.append(jnp.mean(payoffs, axis=0))
        ses.append(jnp.std(payoffs, axis=0, ddof=1) / jnp.sqrt(S_paths.shape[0]))
    prices = jnp.stack(prices, axis=0)
    ses = jnp.stack(ses, axis=0)

    # Monotone decreasing in strike.
    mono_lhs = prices[:, :-1] - prices[:, 1:]
    mono_tol = 3.0 * jnp.sqrt(ses[:, :-1] ** 2 + ses[:, 1:] ** 2)
    assert jnp.all(mono_lhs >= -mono_tol)

    # Convex in strike for equally spaced strike grid.
    convex_lhs = prices[:, :-2] - 2.0 * prices[:, 1:-1] + prices[:, 2:]
    convex_tol = 3.0 * jnp.sqrt(
        ses[:, :-2] ** 2 + 4.0 * ses[:, 1:-1] ** 2 + ses[:, 2:] ** 2
    )
    assert jnp.all(convex_lhs >= -convex_tol)

    # Calendar monotonicity in maturity.
    cal_lhs = prices[1:, :] - prices[:-1, :]
    cal_tol = 3.0 * jnp.sqrt(ses[1:, :] ** 2 + ses[:-1, :] ** 2)
    assert jnp.all(cal_lhs >= -cal_tol)


# --- Fixtures ---


@pytest.fixture(scope="module")
def bs_mc():
    v0 = 0.04
    s0 = 1.0
    spec = make_black_scholes_model_spec(v_0=v0)
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec, num_paths=384, noise_timesteps=256, rde_timesteps=512, s0=s0, seed=2024
    )
    return {"S": S_paths, "V": V_paths, "X": Xs, "W": Ws, "s0": s0, "v0": v0}


@pytest.fixture(scope="module")
def bergomi_mc():
    rho = -0.7
    v0 = 0.2
    s0 = 1.0
    spec = make_bergomi_model_spec(v_0=v0, rho=rho)
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec, num_paths=384, noise_timesteps=256, rde_timesteps=512, s0=s0, seed=7
    )
    return {"S": S_paths, "V": V_paths, "X": Xs, "W": Ws, "s0": s0, "rho": rho}


@pytest.fixture(scope="module")
def rough_bergomi_mc():
    v0 = 0.04
    nu = 1.991
    H = 0.25
    rho = -0.7
    s0 = 1.0
    spec = make_rough_bergomi_model_spec(v_0=v0, nu=nu, hurst=H, rho=rho)
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec, num_paths=320, noise_timesteps=256, rde_timesteps=512, s0=s0, seed=123
    )
    return {"S": S_paths, "V": V_paths, "X": Xs, "W": Ws, "s0": s0, "v0": v0, "nu": nu, "H": H}


@pytest.fixture(scope="module")
def heston_mc():
    v0 = 0.08
    s0 = 1.0
    spec = make_heston_model_spec(v_0=v0, rho=-0.6, nu=0.2, lambda_=3.0, v_bar=0.04)
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec, num_paths=192, noise_timesteps=128, rde_timesteps=256, s0=s0, seed=99
    )
    return {"S": S_paths, "V": V_paths, "X": Xs, "W": Ws, "s0": s0, "v0": v0}


@pytest.fixture(scope="module")
def clsv_mc():
    v0 = 0.08
    s0 = 1.0
    spec = make_classical_local_stochastic_volatility_model_spec(
        v_0=v0,
        rho=-0.3,
        xi=lambda t, s, v: 0.15 + 0.01 * s + 0.02 * v,
        f_1=lambda t, v: -2.0 * (v - 0.04),
        f_2=lambda v: 0.03 + 0.01 * v,
    )
    S_paths, V_paths, Xs, Ws = _simulate_model(
        spec, num_paths=128, noise_timesteps=96, rde_timesteps=192, s0=s0, seed=2718
    )
    return {"S": S_paths, "V": V_paths, "X": Xs, "W": Ws, "s0": s0, "v0": v0}


@pytest.fixture(scope="module")
def rough_heston_mc():
    v0 = 0.05
    v_bar = 0.04
    s0 = 1.0
    N, M = 64, 128
    V_paths, W_paths = simulate_rough_heston_variance_paths(
        jax.random.key(404),
        num_paths=96,
        noise_timesteps=N,
        v_0=v0,
        hurst=0.1,
        nu=0.1,
        rho=-0.7,
        lambda_=1.5,
        v_bar=v_bar,
    )
    sols = jax.vmap(
        lambda X, W: solve_rough_heston_from_variance_path(
            s_0=s0, variance_path=X, price_brownian_path=W,
            v_0=v0, hurst=0.1, nu=0.1, rho=-0.7, lambda_=1.5, v_bar=v_bar,
            noise_timesteps=N, rde_timesteps=M,
        )
    )(V_paths, W_paths)
    return {"S": sols.ys[..., 0], "V": V_paths, "W": W_paths, "s0": s0, "v0": v0}


@pytest.fixture(scope="module")
def quadratic_rough_heston_mc():
    v0 = 0.05
    s0 = 1.0
    N, M = 64, 128
    V_paths, W_paths = simulate_quadratic_rough_heston_variance_paths(
        jax.random.key(505),
        num_paths=96,
        noise_timesteps=N,
        v_0=v0,
        hurst=0.1,
        a=2.0,
        b=0.1,
        c=0.02,
        lambda_=1.0,
        eta=0.1,
        theta_level=0.04,
    )
    sols = jax.vmap(
        lambda X, W: solve_quadratic_rough_heston_from_variance_path(
            s_0=s0, variance_path=X, price_brownian_path=W,
            v_0=v0, hurst=0.1, a=2.0, b=0.1, c=0.02, lambda_=1.0, eta=0.1,
            noise_timesteps=N, rde_timesteps=M,
        )
    )(V_paths, W_paths)
    return {"S": sols.ys[..., 0], "V": V_paths, "W": W_paths, "s0": s0, "v0": v0}


# --- Parameterized: initial variance + nontrivial paths ---


@pytest.mark.parametrize("fixture_name", ["heston_mc"])
def test_initial_variance_and_nontrivial_paths(request, fixture_name) -> None:
    mc = request.getfixturevalue(fixture_name)
    V = mc["V"]
    assert jnp.allclose(V[:, 0], float(mc["v0"]))
    assert jnp.std(V[:, -1], ddof=1) > 1e-3


# --- Parameterized: variance sensible and mean reverting ---


@pytest.mark.parametrize("fixture_name,max_neg_frac", [
    ("heston_mc", 0.0),
    ("clsv_mc", 0.0),
    ("rough_heston_mc", 0.0),
    ("quadratic_rough_heston_mc", 0.0),
])
def test_variance_sensible_and_mean_reverting(request, fixture_name, max_neg_frac) -> None:
    V = request.getfixturevalue(fixture_name)["V"]
    assert jnp.isfinite(V).all()
    assert jnp.mean(V < 0.0) <= max_neg_frac
    assert jnp.mean(V[:, -1]) < jnp.mean(V[:, 0])


# --- Parameterized: requires external X driver ---


@pytest.mark.parametrize("spec", [
    make_rough_heston_model_spec(v_0=0.04, hurst=0.1, nu=0.5, rho=-0.7, lambda_=1.5, v_bar=0.04),
    make_quadratic_rough_heston_model_spec(v_0=0.04, hurst=0.1, a=2.0, b=0.1, c=0.02, lambda_=1.0, eta=0.3),
])
def test_requires_external_x(spec) -> None:
    with pytest.raises(NotImplementedError):
        get_bonesini_noise_drivers(jax.random.key(0), noise_timesteps=32, model_spec=spec, s_0=1.0)


# --- Parameterized: external driver smoke ---


@pytest.mark.parametrize("solve_fn,X_fn,extra_kwargs,seed", [
    (
        solve_rough_heston_from_variance_path,
        lambda N: 0.04 + 0.02 * jnp.linspace(0.0, 1.0, N + 1),
        dict(v_0=0.04, hurst=0.1, nu=0.5, rho=-0.7, lambda_=1.5, v_bar=0.04),
        7,
    ),
    (
        solve_quadratic_rough_heston_from_variance_path,
        lambda N: 0.04 + 0.01 * jnp.sin(jnp.linspace(0.0, jnp.pi, N + 1)),
        dict(v_0=0.04, hurst=0.1, a=2.0, b=0.1, c=0.02, lambda_=1.0, eta=0.3),
        11,
    ),
])
def test_external_driver_smoke(solve_fn, X_fn, extra_kwargs, seed) -> None:
    N = 64
    W = jnp.concatenate(
        [jnp.zeros((1,)), jnp.cumsum(jax.random.normal(jax.random.key(seed), (N,))) / N**0.5]
    )
    X = X_fn(N)
    sol = solve_fn(
        s_0=1.0, variance_path=X, price_brownian_path=W,
        noise_timesteps=N, rde_timesteps=2 * N, **extra_kwargs,
    )
    assert sol.ys is not None
    assert jnp.isfinite(sol.ys).all()
    assert jnp.allclose(sol.ys[:, 1], X, atol=1e-5)


# --- Parameterized: simulate variance paths smoke ---


@pytest.mark.parametrize("simulate_fn,kwargs,seed", [
    (
        simulate_rough_heston_variance_paths,
        dict(v_0=0.04, hurst=0.1, nu=0.5, rho=-0.7, lambda_=1.5, v_bar=0.04),
        21,
    ),
    (
        simulate_quadratic_rough_heston_variance_paths,
        dict(v_0=0.04, hurst=0.1, a=2.0, b=0.1, c=0.02, lambda_=1.0, eta=0.3, theta_level=0.04),
        22,
    ),
])
def test_simulate_variance_paths_smoke(simulate_fn, kwargs, seed) -> None:
    V, W = simulate_fn(jax.random.key(seed), num_paths=8, noise_timesteps=32, **kwargs)
    assert V.shape == (8, 33)
    assert W.shape == (8, 33)
    assert jnp.isfinite(V).all() and jnp.isfinite(W).all()
    assert jnp.allclose(V[:, 0], 0.04)
    assert jnp.all(V > 0.0)
    assert jnp.std(V[:, -1], ddof=1) > 1e-4


# --- Parameterized: martingale ---


@pytest.mark.parametrize("fixture_name,atol", [
    ("bs_mc", 1e-2),
    ("bergomi_mc", 1e-2),
    ("rough_bergomi_mc", 1e-2),
    ("heston_mc", 1.5e-2),
    ("clsv_mc", 2e-2),
    ("rough_heston_mc", 2.5e-2),
    ("quadratic_rough_heston_mc", 2.5e-2),
])
def test_martingale(request, fixture_name, atol) -> None:
    mc = request.getfixturevalue(fixture_name)
    _assert_martingale(mc["S"], s0=float(mc["s0"]), atol=atol)


# --- Parameterized: no static arbitrage ---


@pytest.mark.parametrize("fixture_name", [
    "heston_mc",
    "clsv_mc",
    "rough_heston_mc",
    "quadratic_rough_heston_mc",
])
def test_call_surface_no_static_arbitrage(request, fixture_name) -> None:
    mc = request.getfixturevalue(fixture_name)
    _assert_no_static_arbitrage(mc["S"], s0=float(mc["s0"]))


# --- Black–Scholes ---


def test_black_scholes_log_return_variance(bs_mc) -> None:
    S = bs_mc["S"]
    v0, s0 = float(bs_mc["v0"]), float(bs_mc["s0"])
    log_ret = jnp.log(S[:, -1] / s0)
    assert jnp.isclose(jnp.var(log_ret, ddof=1), v0, rtol=0.15)


def test_black_scholes_log_normality(bs_mc) -> None:
    log_ret = jnp.log(bs_mc["S"][:, -1] / float(bs_mc["s0"]))
    z = (log_ret - jnp.mean(log_ret)) / jnp.std(log_ret, ddof=1)
    assert jnp.abs(jnp.mean(z**3)) < 0.2       # skew ~ 0
    assert jnp.abs(jnp.mean(z**4) - 3.0) < 0.3  # kurtosis ~ 3


# --- Bergomi ---


def test_bergomi_driver_correlation(bergomi_mc) -> None:
    dX = jnp.diff(bergomi_mc["X"], axis=1).flatten()
    dW = jnp.diff(bergomi_mc["W"], axis=1).flatten()
    assert jnp.abs(jnp.corrcoef(dX, dW)[0, 1]) < 0.12


def test_bergomi_leverage_effect(bergomi_mc) -> None:
    S, V = bergomi_mc["S"], bergomi_mc["V"]
    rho = float(bergomi_mc["rho"])
    corr = jnp.corrcoef(jnp.diff(jnp.log(S), axis=1).flatten(), jnp.diff(V, axis=1).flatten())[0, 1]
    assert jnp.sign(corr) == jnp.sign(rho)
    assert jnp.abs(corr) > 0.1


# --- Rough Bergomi ---


def test_rough_bergomi_vol_roughness_scaling(rough_bergomi_mc) -> None:
    V = rough_bergomi_mc["V"]
    H = float(rough_bergomi_mc["H"])
    lags = jnp.array([1, 2, 4])
    vars_per_lag = jnp.array([
        jnp.var((V[:, int(L):] - V[:, :-int(L)]).flatten(), ddof=1)
        for L in lags
    ])
    slope, _ = jnp.polyfit(jnp.log(lags.astype(float)), jnp.log(vars_per_lag), 1)
    assert jnp.isclose(slope, 2.0 * H, atol=0.25)


def test_rough_bergomi_exp_normalization(rough_bergomi_mc) -> None:
    V = rough_bergomi_mc["V"]
    nu = float(rough_bergomi_mc["nu"])
    H = float(rough_bergomi_mc["H"])
    gamma_h = gamma(H + 0.5)
    T = V.shape[1] - 1
    for k in [int(T * 0.25), int(T * 0.5), int(T * 0.75), T]:
        t = k / T
        m = jnp.mean(jnp.exp(nu * V[:, k] - 0.5 * nu**2 * t**(2.0 * H) / gamma_h**2))
        assert jnp.isclose(m, 1.0, atol=0.12)


# --- Model-specific tests ---


def test_heston_leverage_effect(heston_mc) -> None:
    S, V = heston_mc["S"], heston_mc["V"]
    corr = jnp.corrcoef(jnp.diff(jnp.log(S), axis=1).flatten(), jnp.diff(V, axis=1).flatten())[0, 1]
    assert corr < -0.05


def test_classical_local_stochastic_volatility_smoke() -> None:
    spec = make_classical_local_stochastic_volatility_model_spec(
        v_0=0.04,
        rho=-0.3,
        xi=lambda t, s, v: 0.2 + 0.1 * v + 0.05 * s,
        f_1=lambda t, v: -1.0 * (v - 0.04),
        f_2=lambda v: 0.08 + 0.05 * v,
    )
    S_paths, V_paths, _, _ = _simulate_model(
        spec, num_paths=64, noise_timesteps=64, rde_timesteps=128, s0=1.0, seed=2718
    )
    assert jnp.isfinite(S_paths).all()
    assert jnp.isfinite(V_paths).all()
    assert jnp.allclose(V_paths[:, 0], 0.04)
