from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roughbench.manifold_rde.solar_covariancy import (
    _to_numeric_series,
    project_to_spd_correlation,
    residualise,
    window_covariances,
)
from roughbench.utils.load_tsf import convert_tsf_to_dataframe
from utils import resolve_output_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SPD covariance trajectory from the Monash Solar TSF dataset and save as .npz."
    )
    parser.add_argument(
        "--tsf-path",
        type=str,
        default="raw_data/monash_timeseries/solar_10_minutes_dataset.tsf",
        help="Path to the TSF file.",
    )
    parser.add_argument("--num-series", type=int, default=10, help="Number of series (panels) to use.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=5,
        help="Size of each panel group. Panels are randomly grouped, remainder dropped.",
    )
    parser.add_argument("--period", type=int, default=144, help="Seasonality period to remove (10-min data: 144/day).")
    parser.add_argument(
        "--window-size",
        type=int,
        default=144,
        help="Window size (in timesteps) for realized covariance computation.",
    )
    parser.add_argument("--stride", type=int, default=144, help="Stride (in timesteps) between windows.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Shrinkage toward identity: covariance = (1-alpha)*realized + alpha*mean_trace*I.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=1e-6,
        help="Diagonal jitter added to each covariance to encourage strict PD.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Epsilon for correlation normalization and SPD projection.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="If set, divide residuals by per-series standard deviation before windowing.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used only for reproducible subsampling (if added later)."
    )
    parser.add_argument("--subdir", type=str, default="solar_covariancy", help="Subdir under data/ to save into.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional override base data dir (defaults to <repo>/data/<subdir>).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="solar_spd_covariance_trajectory.npz",
        help="Output .npz filename.",
    )
    return parser.parse_args()


def _load_power_matrix(tsf_path: str, num_series: int | None = None) -> np.ndarray:
    tsf_data = convert_tsf_to_dataframe(tsf_path)
    series_values = tsf_data["data"]["series_value"]
    total_available = len(series_values)

    if num_series is None:
        num_series = total_available
    elif num_series <= 0:
        raise ValueError("num_series must be a positive integer.")
    elif num_series > total_available:
        raise ValueError(f"num_series ({num_series}) exceeds available series in the dataset ({total_available}).")

    numeric_series: list[np.ndarray] = []
    for raw_series in series_values.iloc[:num_series]:
        numeric = _to_numeric_series(raw_series).to_numpy(dtype=float)
        numeric_series.append(numeric)

    min_length = min(len(series) for series in numeric_series)
    if min_length < 2:
        raise ValueError("Not enough observations after alignment.")
    power = np.stack([series[:min_length] for series in numeric_series], axis=1)
    return power


def _covariance_to_correlation(covariance: np.ndarray, eps: float) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(covariance), eps, None))
    corr = covariance / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return corr


def generate_solar_spd_covariance_trajectory(
    *,
    tsf_path: str,
    num_series: int | None,
    period: int,
    window_size: int,
    stride: int,
    alpha: float,
    jitter: float,
    eps: float,
    scale: bool,
    group_size: int | None = None,
    seed: int = 0,
    target_windows: int = 365,
) -> dict[str, np.ndarray]:
    """
    Build a windowed SPD covariance trajectory from a multivariate power time series.

    If group_size is provided, randomly groups panels into groups of that size,
    drops remainder, and returns grouped covariances shaped (num_groups, target_windows, group_size, group_size).

    Returns a dict with keys:
      - power: raw aligned power array, shape (T, d) or (T, total_panels) if not grouped
      - residuals: deseasonalized (and optionally scaled) residuals, shape (T, d) or (T, total_panels) if not grouped
      - covariances: SPD covariance trajectory, shape (W, d, d) or (num_groups, target_windows, group_size, group_size) if grouped
      - correlations: SPD correlation trajectory, shape (W, d, d) or (num_groups, target_windows, group_size, group_size) if grouped
    """
    # Load all available series if num_series is None
    power = _load_power_matrix(tsf_path, num_series=num_series)
    total_panels = power.shape[1]

    if group_size is not None:
        if group_size <= 0:
            raise ValueError("group_size must be a positive integer.")
        if group_size > total_panels:
            raise ValueError(f"group_size ({group_size}) exceeds total panels ({total_panels}).")

        # Randomly group panels
        rng = np.random.RandomState(seed=seed)
        panel_indices = np.arange(total_panels)
        rng.shuffle(panel_indices)

        num_groups = total_panels // group_size
        num_used_panels = num_groups * group_size
        remainder = total_panels - num_used_panels

        if remainder > 0:
            print(f"Dropping {remainder} remainder panel(s) (using {num_used_panels} of {total_panels} panels).")

        used_indices = panel_indices[:num_used_panels]
        grouped_power = power[:, used_indices]  # (T, num_used_panels)

        # Reshape to (T, num_groups, group_size) then process each group
        grouped_power_reshaped = grouped_power.reshape(power.shape[0], num_groups, group_size)

        scale_vec = None
        if bool(scale):
            std = np.nanstd(grouped_power, axis=0)  # (num_used_panels,)
            scale_vec = np.asarray(std, dtype=float)

        # Process each group separately
        group_covariances: list[np.ndarray] = []
        group_correlations: list[np.ndarray] = []

        for g in range(num_groups):
            group_power = grouped_power_reshaped[:, g, :]  # (T, group_size)

            group_scale = None
            if scale_vec is not None:
                # scale_vec indices correspond to columns in grouped_power
                group_scale = scale_vec[g * group_size : (g + 1) * group_size]

            group_residuals = residualise(group_power, period=period, scale=group_scale)
            cov_list = window_covariances(group_residuals, window_size=window_size, stride=stride, alpha=alpha)

            if len(cov_list) == 0:
                raise ValueError(
                    f"No windows were produced for group {g}; check window_size and stride vs series length."
                )

            # Truncate or pad to exactly target_windows
            if len(cov_list) > target_windows:
                cov_list = cov_list[:target_windows]
            elif len(cov_list) < target_windows:
                # Pad with last covariance if needed
                last_cov = cov_list[-1]
                cov_list.extend([last_cov] * (target_windows - len(cov_list)))

            group_cov = np.stack([cov + float(jitter) * np.eye(cov.shape[0]) for cov in cov_list], axis=0)
            group_corr = np.stack(
                [project_to_spd_correlation(_covariance_to_correlation(cov, eps=eps), eps=eps) for cov in group_cov],
                axis=0,
            )

            group_covariances.append(group_cov)
            group_correlations.append(group_corr)

        # Stack groups: (num_groups, target_windows, group_size, group_size)
        covariances = np.stack(group_covariances, axis=0)
        correlations = np.stack(group_correlations, axis=0)

        return {
            "power": np.asarray(grouped_power, dtype=np.float32),
            "residuals": np.asarray(
                residualise(grouped_power, period=period, scale=scale_vec)
                if scale_vec is not None
                else residualise(grouped_power, period=period),
                dtype=np.float32,
            ),
            "covariances": np.asarray(covariances, dtype=np.float32),
            "correlations": np.asarray(correlations, dtype=np.float32),
        }
    else:
        # Original non-grouped behavior
        scale_vec = None
        if bool(scale):
            std = np.nanstd(power, axis=0)
            scale_vec = np.asarray(std, dtype=float)

        residuals = residualise(power, period=period, scale=scale_vec)
        cov_list = window_covariances(residuals, window_size=window_size, stride=stride, alpha=alpha)
        if len(cov_list) == 0:
            raise ValueError("No windows were produced; check window_size and stride vs series length.")

        covariances = np.stack([cov + float(jitter) * np.eye(cov.shape[0]) for cov in cov_list], axis=0)
        correlations = np.stack(
            [project_to_spd_correlation(_covariance_to_correlation(cov, eps=eps), eps=eps) for cov in covariances],
            axis=0,
        )

        return {
            "power": np.asarray(power, dtype=np.float32),
            "residuals": np.asarray(residuals, dtype=np.float32),
            "covariances": np.asarray(covariances, dtype=np.float32),
            "correlations": np.asarray(correlations, dtype=np.float32),
        }


def save_solar_spd_covariance_npz(
    *,
    payload: dict[str, np.ndarray],
    filename: str,
    subdir: str,
    data_dir: Path | None,
    meta: dict[str, object],
) -> Path:
    data_path, _ = resolve_output_dirs(subdir=subdir, data_dir=data_dir)
    target = data_path / filename
    np.savez_compressed(
        target,
        power=payload["power"],
        residuals=payload["residuals"],
        covariances=payload["covariances"],
        correlations=payload["correlations"],
        meta=np.asarray(meta, dtype=object),
    )
    print(f"Saved compressed data to {target}")
    return target


if __name__ == "__main__":
    args = _parse_args()
    np.random.seed(int(args.seed))

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    # Determine num_series: use None to load all if group_size is provided, otherwise use args.num_series
    num_series_arg = None if args.group_size is not None else int(args.num_series)
    group_size_arg = int(args.group_size) if args.group_size is not None else None

    payload = generate_solar_spd_covariance_trajectory(
        tsf_path=str(args.tsf_path),
        num_series=num_series_arg,
        period=int(args.period),
        window_size=int(args.window_size),
        stride=int(args.stride),
        alpha=float(args.alpha),
        jitter=float(args.jitter),
        eps=float(args.eps),
        scale=bool(args.scale),
        group_size=group_size_arg,
        seed=int(args.seed),
        target_windows=365,
    )

    # Calculate metadata about grouping
    total_available = None
    num_groups = None
    remainder_dropped = None
    if group_size_arg is not None:
        tsf_data = convert_tsf_to_dataframe(str(args.tsf_path))
        total_available = len(tsf_data["data"]["series_value"])
        num_groups = total_available // group_size_arg
        remainder_dropped = total_available % group_size_arg

    meta: dict[str, object] = {
        "tsf_path": str(args.tsf_path),
        "num_series": num_series_arg if num_series_arg is not None else "all",
        "total_available": total_available,
        "group_size": group_size_arg,
        "num_groups": num_groups,
        "remainder_dropped": remainder_dropped,
        "period": int(args.period),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "alpha": float(args.alpha),
        "jitter": float(args.jitter),
        "eps": float(args.eps),
        "scale": bool(args.scale),
        "seed": int(args.seed),
        "target_windows": 365,
    }

    save_solar_spd_covariance_npz(
        payload=payload,
        filename=str(args.filename),
        subdir=str(args.subdir),
        data_dir=output_dir,
        meta=meta,
    )

    print("")
    print("Generated arrays:")
    for key, value in payload.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
