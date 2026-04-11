from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roughbench.utils.load_tsf import convert_tsf_to_dataframe


def _to_numeric_series(raw_series: Iterable[int | float | str]) -> pd.Series:
    values = pd.Series(list(raw_series))
    values = values.replace("NaN", np.nan)
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(numeric, index=values.index)


def remove_deterministic_seasonality(values: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be a positive integer.")
    seasonal_mean = values.groupby(values.index % period).transform("mean")
    return values - seasonal_mean


def _build_spd_covariance(series_list: list[pd.Series], jitter: float) -> np.ndarray:
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative.")
    if len(series_list) == 0:
        raise ValueError("series_list must contain at least one series.")

    frame = pd.concat(series_list, axis=1)
    frame = frame.dropna(axis=0, how="any")
    if frame.shape[0] < 2:
        raise ValueError("Not enough aligned observations after dropping NaNs.")

    data = frame.to_numpy(dtype=float)
    covariance = np.cov(data, rowvar=False, bias=False)
    covariance += jitter * np.eye(covariance.shape[0])
    return covariance


def covariance_from_tsf(
    file_path: str,
    num_series: int,
    period: int,
    jitter: float = 1e-6,
) -> np.ndarray:
    if num_series <= 0:
        raise ValueError("num_series must be a positive integer.")

    tsf_data = convert_tsf_to_dataframe(file_path)
    series_values = tsf_data["data"]["series_value"]
    if num_series > len(series_values):
        raise ValueError("num_series exceeds available series in the dataset.")

    deseasonalized: list[pd.Series] = []
    for raw_series in series_values.iloc[:num_series]:
        numeric_series = _to_numeric_series(raw_series)
        deseasonalized.append(remove_deterministic_seasonality(numeric_series, period))

    return _build_spd_covariance(deseasonalized, jitter=jitter)


def plot_covariance_heatmap(
    covariance: np.ndarray,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    heatmap = ax.imshow(covariance, cmap="viridis", aspect="auto")
    ax.set_title("Deseasonalized Covariance")
    ax.set_xlabel("Series")
    ax.set_ylabel("Series")
    fig.colorbar(heatmap, ax=ax, shrink=0.85)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def residualise(
    power: np.ndarray,
    period: int,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be a positive integer.")
    if power.ndim != 2:
        raise ValueError("power must have shape (T, d).")

    timesteps, panels = power.shape
    phases = np.arange(timesteps) % period
    seasonal_mean = np.zeros((period, panels), dtype=float)
    for phase in range(period):
        seasonal_mean[phase] = power[phases == phase].mean(axis=0)
    residuals = power - seasonal_mean[phases]

    if scale is not None:
        if scale.shape != (panels,):
            raise ValueError("scale must have shape (d,).")
        safe_scale = np.where(scale == 0.0, 1.0, scale)
        residuals = residuals / safe_scale

    return residuals


def window_drivers(
    residuals: np.ndarray,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    if window_size <= 1:
        raise ValueError("window_size must be at least 2.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (T, d).")

    timesteps = residuals.shape[0]
    drivers: list[np.ndarray] = []
    for start in range(0, timesteps - window_size + 1, stride):
        window = residuals[start : start + window_size]
        drivers.append(window)
    return drivers


def project_to_spd_correlation(matrix: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.maximum(eigenvalues, eps)
    projected = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    diag = np.sqrt(np.clip(np.diag(projected), eps, None))
    projected = projected / np.outer(diag, diag)
    np.fill_diagonal(projected, 1.0)
    return projected


def _covariance_to_correlation(covariance: np.ndarray, eps: float) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(covariance), eps, None))
    corr = covariance / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return corr


def window_targets(
    residuals: np.ndarray,
    window_size: int,
    stride: int,
    alpha: float,
    eps: float,
) -> list[np.ndarray]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (T, d).")

    timesteps, dim = residuals.shape
    targets: list[np.ndarray] = []
    identity = np.eye(dim)
    for start in range(0, timesteps - window_size + 1, stride):
        window = residuals[start : start + window_size]
        increments = window[1:] - window[:-1]
        realized = increments.T @ increments
        mean_trace = np.trace(realized) / dim
        covariance = (1.0 - alpha) * realized + alpha * mean_trace * identity
        corr = _covariance_to_correlation(covariance, eps=eps)
        targets.append(project_to_spd_correlation(corr, eps=eps))
    return targets


def window_covariances(
    residuals: np.ndarray,
    window_size: int,
    stride: int,
    alpha: float,
) -> list[np.ndarray]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (T, d).")

    timesteps, dim = residuals.shape
    covariances: list[np.ndarray] = []
    identity = np.eye(dim)
    for start in range(0, timesteps - window_size + 1, stride):
        window = residuals[start : start + window_size]
        increments = window[1:] - window[:-1]
        realized = increments.T @ increments
        mean_trace = np.trace(realized) / dim
        covariance = (1.0 - alpha) * realized + alpha * mean_trace * identity
        covariances.append(covariance)
    return covariances


def plot_window_covariance_heatmaps(
    covariances: list[np.ndarray],
    columns: int = 5,
    max_windows: int | None = None,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    if len(covariances) == 0:
        raise ValueError("covariances must contain at least one matrix.")
    if columns <= 0:
        raise ValueError("columns must be a positive integer.")

    matrices = covariances
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive.")
        matrices = covariances[:max_windows]

    count = len(matrices)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.0 * rows))
    if rows == 1 and columns == 1:
        axes_list = [axes]
    else:
        axes_list = list(np.ravel(axes))

    vmin = min(np.min(matrix) for matrix in matrices)
    vmax = max(np.max(matrix) for matrix in matrices)
    heatmap = None
    for idx, ax in enumerate(axes_list):
        if idx >= count:
            ax.axis("off")
            continue
        heatmap = ax.imshow(
            matrices[idx], cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto"
        )
        ax.set_title(f"Window {idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    if heatmap is not None:
        fig.colorbar(heatmap, ax=axes_list, shrink=0.75)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def branched_lifts(
    windows: list[np.ndarray],
    depth: int,
    lift_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> list[np.ndarray]:
    if depth <= 0:
        raise ValueError("depth must be a positive integer.")
    if lift_fn is None:
        return windows
    return [lift_fn(window, depth) for window in windows]


def make_dataset(
    correlations: list[np.ndarray],
    lifts: list[np.ndarray],
    lookback_windows: int,
    horizon: int,
) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
    if lookback_windows <= 0:
        raise ValueError("lookback_windows must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if len(correlations) != len(lifts):
        raise ValueError("correlations and lifts must have the same length.")

    inputs: list[dict[str, np.ndarray]] = []
    targets: list[np.ndarray] = []
    for idx in range(lookback_windows - 1, len(correlations) - horizon):
        state = np.stack(correlations[idx - lookback_windows + 1 : idx + 1], axis=0)
        inputs.append({"state": state, "driver": lifts[idx]})
        targets.append(correlations[idx + horizon])
    return inputs, targets


if __name__ == "__main__":
    file_path = "raw_data/monash_timeseries/solar_10_minutes_dataset.tsf"
    num_series = 10
    period = 144
    window_size = period
    stride = period
    alpha = 0.05

    covariance = covariance_from_tsf(file_path, num_series=num_series, period=period)
    plot_covariance_heatmap(covariance, output_path="data/solar_covariance_heatmap.png")

    tsf_data = convert_tsf_to_dataframe(file_path)
    series_values = tsf_data["data"]["series_value"]
    numeric_series = []
    for raw_series in series_values.iloc[:num_series]:
        numeric = _to_numeric_series(raw_series).to_numpy(dtype=float)
        numeric_series.append(numeric)
    min_length = min(len(series) for series in numeric_series)
    power = np.stack([series[:min_length] for series in numeric_series], axis=1)
    residuals = residualise(power, period=period)
    covariances = window_covariances(
        residuals,
        window_size=window_size,
        stride=stride,
        alpha=alpha,
    )
    plot_window_covariance_heatmaps(
        covariances,
        columns=6,
        max_windows=24,
        output_path="data/solar_covariance_windows.png",
    )
    print(covariance.shape)
