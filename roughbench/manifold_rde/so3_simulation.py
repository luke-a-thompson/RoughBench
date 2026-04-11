from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial.transform import Rotation


def _reorder_quat(quat_xyzw: np.ndarray, quat_order: str) -> np.ndarray:
    """
    Ensure quaternion last-dimension is (x, y, z, w) as expected by SciPy.

    Parameters
    ----------
    quat_xyzw:
        Quaternion array with shape (..., 4) in either xyzw or wxyz order.
    quat_order:
        Either "xyzw" or "wxyz" describing the input ordering.
    """
    if quat_order == "xyzw":
        return quat_xyzw
    if quat_order == "wxyz":
        # (w, x, y, z) -> (x, y, z, w)
        return quat_xyzw[..., [1, 2, 3, 0]]
    raise ValueError(f"quat_order must be 'xyzw' or 'wxyz', got: {quat_order!r}")


def convert_quat_npz_to_rotmats_by_damping(
    input_npz: str,
    output_dir: str,
    *,
    chunk_trajectories: int = 8,
    quat_order: str = "xyzw",
    output_npz_filename: str = "so3_simulation_rotmats_by_damping.npz",
    cleanup_tmp: bool = True,
) -> None:
    """
    Read a large SO(3) quaternion simulation `.npz` and save rotation matrices grouped by damping label.

    Expects `input_npz` to contain:
      - `quat`: (N, T, 4) float32
      - `distribution_indices`: (N,) int labels (damping groups)
      - `dt`: scalar float32

    Writes a single `.npz` into `output_dir` with arrays:
      - `R_sim_damped{label}`: shape (N_label, T, 3, 3), dtype float32 (one per unique label)
      - `dt`: scalar float32
      - `labels`: unique labels (sorted)
      - `distribution_indices`: copy of input labels
    """
    if chunk_trajectories <= 0:
        raise ValueError(
            f"chunk_trajectories must be positive, got: {chunk_trajectories}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_npz, mmap_mode="r") as z:
        quat = z["quat"]
        distribution_indices = z["distribution_indices"]
        dt = z["dt"]

        if quat.ndim != 3 or quat.shape[-1] != 4:
            raise ValueError(f"Expected quat shape (N, T, 4), got: {quat.shape}")
        if (
            distribution_indices.ndim != 1
            or distribution_indices.shape[0] != quat.shape[0]
        ):
            raise ValueError(
                "Expected distribution_indices shape (N,) matching quat.shape[0]; "
                f"got {distribution_indices.shape} vs N={quat.shape[0]}"
            )

        labels = np.unique(distribution_indices)
        labels_sorted = np.sort(labels)

        n_traj, n_steps, _ = quat.shape

        output_npz_path = out_dir / output_npz_filename
        tmp_root = out_dir / "_tmp_so3_rotmats"

        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        tmp_paths_by_label: dict[int, Path] = {}

        try:
            # 1) Create per-label .npy memmaps (RAM-safe) in a temp folder.
            for label in labels_sorted:
                label_int = int(label)
                traj_idx = np.flatnonzero(distribution_indices == label)
                n_label = int(traj_idx.size)

                tmp_path = tmp_root / f"R_sim_damped{label_int}.npy"
                tmp_paths_by_label[label_int] = tmp_path

                out = open_memmap(
                    tmp_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_label, n_steps, 3, 3),
                )

                print(
                    f"[so3_simulation] label={label_int}  trajectories={n_label}  steps={n_steps}  tmp={tmp_path}"
                )

                write_row = 0
                for start in range(0, n_label, chunk_trajectories):
                    batch_idx = traj_idx[start : start + chunk_trajectories]

                    q = quat[batch_idx, :, :]
                    q = _reorder_quat(q, quat_order=quat_order)

                    q_flat = np.asarray(q, dtype=np.float32).reshape(-1, 4)
                    r_flat = Rotation.from_quat(q_flat).as_matrix()
                    r = r_flat.reshape(q.shape[0], n_steps, 3, 3).astype(
                        np.float32, copy=False
                    )

                    out[write_row : write_row + q.shape[0], :, :, :] = r
                    write_row += q.shape[0]

                    if (start // chunk_trajectories) % 25 == 0:
                        print(
                            f"[so3_simulation] label={label_int}  {write_row}/{n_label} trajectories done"
                        )

                out.flush()

            # 2) Pack into a single .npz with multiple arrays (like the input file).
            # Use mmap_mode='r' so packing doesn't pull full arrays into memory.
            savez_kwargs: dict[str, np.ndarray] = {
                "dt": np.asarray(dt, dtype=np.float32),
                "labels": labels_sorted,
                "distribution_indices": np.asarray(distribution_indices),
            }
            for label_int, tmp_path in tmp_paths_by_label.items():
                savez_kwargs[f"R_sim_damped{label_int}"] = np.load(
                    tmp_path, mmap_mode="r"
                )

            print(f"[so3_simulation] packing into {output_npz_path}")
            # Some type stubs for numpy define `savez(file, *args, allow_pickle=..., **kwds)`.
            # If a kwarg name collides with a formal parameter name in stubs, type-checkers can
            # get confused. We avoid any ambiguity by using an explicit dict cast.
            np.savez(output_npz_path, **dict(savez_kwargs))  # type: ignore[arg-type]
            print(f"[so3_simulation] wrote {output_npz_path}")
        finally:
            if cleanup_tmp and tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)

        print(
            "[so3_simulation] done. Wrote arrays: "
            + ", ".join([f"R_sim_damped{int(x)}" for x in labels_sorted])
            + f" into {output_npz_path}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SO(3) quaternions to rotation matrices grouped by damping."
    )
    p.add_argument(
        "--input-npz",
        default="/home/luke/roughbench/raw_data/sg_so3_simulation/rigid_body_FREE_ROTATION.npz",
        help="Path to rigid_body_FREE_ROTATION.npz",
    )
    p.add_argument(
        "--output-dir",
        default="/home/luke/roughbench/data/so3_simulation",
        help="Directory to write the output .npz into",
    )
    p.add_argument(
        "--output-npz-filename",
        default="so3_simulation_rotmats_by_damping.npz",
        help="Filename (within output-dir) for the packed npz.",
    )
    p.add_argument(
        "--chunk-trajectories",
        type=int,
        default=64,
        help="Number of trajectories to convert per chunk (lower = less RAM, slower).",
    )
    p.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion component order in the file. SciPy expects xyzw.",
    )
    p.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary per-label .npy files used for packing (debug).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Quick existence checks with clear errors.
    if not os.path.exists(args.input_npz):
        raise FileNotFoundError(args.input_npz)

    convert_quat_npz_to_rotmats_by_damping(
        args.input_npz,
        args.output_dir,
        chunk_trajectories=args.chunk_trajectories,
        quat_order=args.quat_order,
        output_npz_filename=args.output_npz_filename,
        cleanup_tmp=not args.keep_tmp,
    )


if __name__ == "__main__":
    main()
