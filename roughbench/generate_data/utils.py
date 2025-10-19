from __future__ import annotations

from pathlib import Path
import numpy as np


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


def save_plot(filename: str, subdir: str, data_dir: Path | None = None, dpi: int = 150, verbose: bool = True) -> tuple[Path, Path]:
    """Save the current matplotlib figure to both images and docs mirrors.

    Returns (image_path, docs_path) to the saved files.
    If matplotlib is unavailable, no files are written; intended paths are returned.
    """
    data_path, docs_dir = resolve_output_dirs(subdir=subdir, data_dir=data_dir)
    image_path = data_path / filename
    docs_path = docs_dir / filename

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        # matplotlib not available; skip image saving but return intended paths
        if verbose:
            print(f"(skip) Matplotlib unavailable; intended to save plot to {image_path} and mirror to {docs_path}")
        return image_path, docs_path

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

def save_npy(array: object, filename: str, subdir: str, data_dir: Path | None = None, verbose: bool = True) -> Path:
    """Save a single array as .npy under <repo>/data/<subdir>/filename.

    This does not write anything to docs by design.
    """
    data_path, _ = resolve_output_dirs(subdir=subdir, data_dir=data_dir)
    target = data_path / filename
    np.save(target, np.asarray(array))
    if verbose:
        print(f"Saved data to {target}")
    return target


