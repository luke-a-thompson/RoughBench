from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
import contextlib
import tomllib
from typing import Literal

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.pyplot as plt

import numpy as np

SDE_COLORMAP = "viridis"

_SDE_PASTEL_COLORS = [
    "#0079ff",
    "#ffb84c",
    "#00dfa2",
    "#f266ab",
    "#a459d1",
    "#5f264a",
    "#d4adfc",
    "#7f7f7f",
    "#b3e5be",
    "#97deff",
]


def _repo_root(this_file: Path) -> Path:
    """Ascend from this file to the repository root.

    Layout assumption: this file lives in roughbench/roughbench/generate_data/.
    Repo root is two levels above the package directory.
    """
    # .../roughbench/roughbench/generate_data/utils.py -> repo root at parents[2]
    return this_file.resolve().parents[2]


def load_config(path: str) -> dict[str, object]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def config_section(config: dict[str, object], key: str) -> dict[str, object]:
    section = config.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a table.")
    return section


def resolve_output_dirs(subdir: str, data_dir: Path | None = None) -> tuple[Path, Path]:
    """Return (data_dir, docs_dir) for a given subdir.

    - data_dir defaults to <repo>/data/<subdir>
    - docs_dir is always <repo>/docs/rde_bench/<subdir>
    """
    this_file = Path(__file__).resolve()
    repo_root = _repo_root(this_file)

    default_data_dir = repo_root / "data" / subdir
    data_path = data_dir if data_dir is not None else default_data_dir

    docs_path = repo_root / "docs" / "rde_bench" / subdir

    data_path.mkdir(parents=True, exist_ok=True)
    docs_path.mkdir(parents=True, exist_ok=True)

    return data_path, docs_path


def save_plot(
    filename: str,
    subdir: str,
    data_dir: Path | None = None,
    dpi: int = 150,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Save the current matplotlib figure to both images and docs mirrors.

    Returns (image_path, docs_path) to the saved files.
    If matplotlib is unavailable, no files are written; intended paths are returned.
    """
    data_path, docs_dir = resolve_output_dirs(subdir=subdir, data_dir=data_dir)
    image_path = data_path / filename
    docs_path = docs_dir / filename

    plt.savefig(image_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(docs_path, dpi=dpi, bbox_inches="tight")
    if verbose:
        print("")
        print(f"Saved plot to {image_path} and mirrored to {docs_path}")
    plt.close()
    return image_path, docs_path


def save_npy(
    array: object,
    filename: str,
    subdir: str,
    data_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Save a single array as .npy under <repo>/data/<subdir>/filename.

    This does not write anything to docs by design.
    """
    data_path, _ = resolve_output_dirs(subdir=subdir, data_dir=data_dir)
    target = data_path / filename
    np.save(target, np.asarray(array))
    if verbose:
        print(f"Saved data to {target}")
    return target


def save_npz_compressed(
    solution: object,
    driver: object,
    filename: str,
    subdir: str,
    data_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Save solution and driver arrays as compressed .npz under <repo>/data/<subdir>/filename.

    Args:
        solution: Solution array (trajectories)
        driver: Driver array (noise/control)
        filename: Output filename (should end in .npz)
        subdir: Subdirectory under data/
        data_dir: Optional override for data directory
        verbose: Print save confirmation

    Returns:
        Path to saved file

    This does not write anything to docs by design.
    """
    data_path, _ = resolve_output_dirs(subdir=subdir, data_dir=data_dir)
    target = data_path / filename
    np.savez_compressed(
        target, solution=np.asarray(solution), driver=np.asarray(driver)
    )
    if verbose:
        print(f"Saved compressed data (solution + driver) to {target}")
    return target


def save_npz(
    filename: str,
    subdir: str,
    data_dir: Path | None = None,
    verbose: bool = True,
    **arrays: object,
) -> Path:
    """Save named arrays as compressed .npz under <repo>/data/<subdir>/filename.

    Args:
        filename: Output filename (should end in .npz)
        subdir: Subdirectory under data/
        data_dir: Optional override for data directory
        verbose: Print save confirmation
        **arrays: Named arrays to save

    Returns:
        Path to saved file

    This does not write anything to docs by design.
    """
    data_path, _ = resolve_output_dirs(subdir=subdir, data_dir=data_dir)
    target = data_path / filename
    np.savez_compressed(target, **{k: np.asarray(v) for k, v in arrays.items()})
    if verbose:
        print(f"Saved compressed data to {target}")
    return target


def _roughbench_rcparams(font_scale: float = 1.0) -> dict[str, object]:
    """Return a consistent matplotlib rcParams dictionary used for plotting."""
    base = 10.0 * float(font_scale)
    return {
        # Figure and axes
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.transparent": False,
        # Typography
        "font.size": base,
        "axes.titlesize": base * 1.25,
        "axes.labelsize": base * 1.1,
        "legend.fontsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        # Lines and grid
        "lines.linewidth": 1.5,
        "grid.linestyle": "--",
        "grid.color": "#b0b0b0",
        "grid.alpha": 0.4,
        "axes.grid": True,
        "axes.axisbelow": True,
    }


def _sde_rcparams(font_scale: float = 1.0) -> dict[str, object]:
    """Aleatory-inspired style for SDE/RDE path plots."""
    rc = _roughbench_rcparams(font_scale=font_scale)
    rc.update(
        {
            "figure.dpi": 200,
            "figure.frameon": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.grid.axis": "both",
            "axes.axisbelow": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=_SDE_PASTEL_COLORS),
            "font.family": "serif",
            "font.serif": [
                "New Century Schoolbook",
                "Century Schoolbook L",
                "DejaVu Serif",
                "serif",
            ],
            "text.usetex": False,
            "lines.linewidth": 1.0,
            "grid.color": (0.76, 0.78, 0.83),
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.5,
            "legend.frameon": True,
            "legend.edgecolor": "0.8",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    return rc


@contextlib.contextmanager
def plotting_context(
    font_scale: float = 1.0,
    style: Literal["roughbench", "sde"] = "roughbench",
) -> Iterator[object]:
    """Context manager applying RoughBench plot style.

    Usage:
        with plotting_context():
            fig, ax = create_figure()
            ...

    Yields the imported pyplot module.
    """
    rc = (
        _sde_rcparams(font_scale=font_scale)
        if style == "sde"
        else _roughbench_rcparams(font_scale=font_scale)
    )
    with matplotlib.rc_context(rc=rc):
        yield plt


def path_color(index: int, total: int, colormap: str = SDE_COLORMAP) -> object:
    """Return a stable color for an ensemble path."""
    if total <= 1:
        return plt.get_cmap(colormap)(0.55)
    return plt.get_cmap(colormap)(index / float(total - 1))


def _final_value_colors(paths: np.ndarray, colormap: str) -> tuple[list[object], int]:
    final_values = np.asarray(paths)[:, -1]
    n_bins = max(1, int(np.sqrt(len(final_values))))
    cm = plt.colormaps[colormap]
    color_positions = np.linspace(0.0, 1.0, n_bins, endpoint=True)
    if n_bins == 1:
        return [cm(color_positions[0]) for _ in final_values], n_bins

    _, bins = np.histogram(final_values, n_bins)
    indices = np.digitize(final_values, bins[1:-1], right=False)
    return [cm(color_positions[int(index)]) for index in indices], n_bins


def draw_sde_paths(
    *,
    times: object,
    paths: object,
    title: str | None = None,
    suptitle: str | None = None,
    xlabel: str = "$t$",
    ylabel: str = "$X(t)$",
    expectation: object | None = None,
    marginal: bool = True,
    colormap: str = SDE_COLORMAP,
    figsize: tuple[float, float] = (12.0, 7.0),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
    """Draw SDE paths using Aleatory's horizontal draw layout."""
    times_np = np.asarray(times)
    paths_np = np.asarray(paths)
    if paths_np.ndim != 2:
        raise ValueError(f"Expected paths with shape (N, T), got {paths_np.shape}.")

    colors, n_bins = _final_value_colors(paths_np, colormap)

    if marginal:
        fig = plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        ax_paths = fig.add_subplot(gs[:4])
        ax_marginal = fig.add_subplot(gs[4:], sharey=ax_paths)

        final_values = paths_np[:, -1]
        _, _, patches = ax_marginal.hist(
            final_values, n_bins, orientation="horizontal", density=True
        )
        cm = plt.colormaps[colormap]
        color_positions = np.linspace(0.0, 1.0, n_bins, endpoint=True)
        for color_position, patch in zip(color_positions, patches):
            plt.setp(patch, "facecolor", cm(color_position))
        plt.setp(ax_marginal.get_yticklabels(), visible=False)
        ax_marginal.set_title(
            "$X_T$ Marginal"
            if xlabel == "$t$" and ylabel == "$X(t)$"
            else "Final Marginal"
        )
        plt.subplots_adjust(wspace=0.025, hspace=0.025)
    else:
        fig, ax_paths = plt.subplots(figsize=figsize)
        ax_marginal = None

    for path, color in zip(paths_np, colors):
        ax_paths.plot(times_np, path, "-", color=color, lw=1.0)

    if expectation is not None:
        ax_paths.plot(
            times_np,
            np.asarray(expectation),
            "--",
            lw=1.75,
            label="Marginal Expectations",
        )
        ax_paths.legend()

    if suptitle is not None:
        fig.suptitle(suptitle)
    ax_paths.set_title(
        title
        if title is not None
        else "Monte Carlo Simulated Paths $\\{X_t, t \\in [t_0, T]\\}$"
    )
    ax_paths.set_xlabel(xlabel)
    ax_paths.set_ylabel(ylabel)
    return fig, ax_paths, ax_marginal


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (10.0, 6.0),
    gridspec_kw: dict[str, object] | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a figure and axes with consistent defaults.

    Returns (fig, ax).
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, gridspec_kw=gridspec_kw)
    return fig, ax


def decorate_axes(
    ax: matplotlib.axes.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    legend_loc: str = "best",
    legend_frame: bool = False,
) -> None:
    """Apply consistent decorations to a single Axes object."""

    # Minor grid and clean spines
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.15)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc=legend_loc, frameon=legend_frame)


def finalize_plot(tight_layout: bool = True) -> None:
    if tight_layout:
        plt.tight_layout()
