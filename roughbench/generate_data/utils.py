from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
import contextlib

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

import numpy as np

try:
    import matplotlib as _mpl  # type: ignore[import]
    import matplotlib.pyplot as _plt  # type: ignore[import]
except Exception:
    _mpl = None  # type: ignore[assignment]
    _plt = None  # type: ignore[assignment]


def _repo_root(this_file: Path) -> Path:
    """Ascend from this file to the repository root.

    Layout assumption: this file lives in roughbench/roughbench/generate_data/.
    Repo root is two levels above the package directory.
    """
    # .../roughbench/roughbench/generate_data/utils.py -> repo root at parents[2]
    return this_file.resolve().parents[2]


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
    try:
        plt.close()
    except Exception:
        pass
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
    np.savez_compressed(target, solution=np.asarray(solution), driver=np.asarray(driver))
    if verbose:
        print(f"Saved compressed data (solution + driver) to {target}")
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


@contextlib.contextmanager
def plotting_context(font_scale: float = 1.0) -> Iterator[object]:
    """Context manager applying RoughBench plot style.

    Usage:
        with plotting_context():
            fig, ax = create_figure()
            ...

    Yields the imported pyplot module.
    """
    rc = _roughbench_rcparams(font_scale=font_scale)
    with matplotlib.rc_context(rc=rc):
        yield plt


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
) -> None:
    """Apply consistent decorations to a single Axes object."""

    # Minor grid and clean spines
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.15)
    for side in ["top", "right"]:
        if side in ax.spines:
            ax.spines[side].set_visible(False)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        try:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc=legend_loc, frameon=False)
        except Exception:
            pass


def finalize_plot(tight_layout: bool = True) -> None:
    """Finalize the current figure (e.g., tight_layout)."""
    if tight_layout:
        try:
            plt.tight_layout()
        except Exception:
            pass
