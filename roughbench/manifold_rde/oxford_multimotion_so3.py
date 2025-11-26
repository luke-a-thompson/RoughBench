"""
Oxford Multimotion SO(3) Dataset. We follow SG-NCDE in casting this as a manifold-constrained CDE problem over SO(3).
"""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "time_sec",
    "time_nsec",
    "object",
    "rotation_quaternion_x",
    "rotation_quaternion_y",
    "rotation_quaternion_z",
    "rotation_quaternion_w",
]


def _compute_downsample_indices(t: np.ndarray, sample_hz: float) -> np.ndarray:
    """
    Compute indices to downsample a monotone time array to approximately sample_hz.

    Args:
        t: (T,) array of monotonically increasing times (float64).
        sample_hz: Target sampling frequency in Hz. If <= 0, no downsampling is applied.

    Returns:
        (K,) int64 array of indices into t with K <= T.
    """
    if sample_hz <= 0.0:
        return np.arange(t.shape[0], dtype=np.int64)

    target_dt = 1.0 / sample_hz
    indices: list[int] = [0]
    last_t = float(t[0])

    for i in range(1, t.shape[0]):
        ti = float(t[i])
        if ti >= last_t + target_dt:
            indices.append(i)
            last_t = ti

    if indices[-1] != t.shape[0] - 1:
        indices.append(t.shape[0] - 1)

    return np.asarray(indices, dtype=np.int64)


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Convert an array of quaternions to rotation matrices.

    Args:
        quat: (T, 4) array with columns [x, y, z, w], assumed normalised.

    Returns:
        (T, 3, 3) array with each R[k] an SO(3) rotation matrix.
    """
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    T = quat.shape[0]
    R = np.empty((T, 3, 3), dtype=np.float32)

    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)

    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)

    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R


def load_vicon_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a vicon.csv file and perform basic validation.

    Args:
        path: Path to the CSV file.

    Returns:
        Cleaned DataFrame with required columns.
    """
    df: pd.DataFrame = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing object or quaternion components
    df = df.dropna(subset=REQUIRED_COLUMNS)
    mask = df["object"].astype(str).str.strip() != ""
    df = df[mask].copy()  # type: ignore[assignment]

    return df


def convert_vicon_csv_to_npz(
    csv_path: str | Path,
    output_dir: str | Path,
    sample_hz: float = 40.0,
) -> None:
    """
    Convert a vicon.csv file to a single .npz file containing all objects.

    Args:
        csv_path: Path to the vicon.csv file.
        output_dir: Directory to save the .npz files.
        sample_hz: Target sampling frequency in Hz for downsampling. If <= 0,
            no downsampling is applied.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_name = csv_path.stem.replace("_vicon", "")
    df = load_vicon_csv(csv_path)
    obj_names = sorted(df["object"].unique())

    arrays: dict[str, np.ndarray] = {}
    t_ref_full: np.ndarray | None = None
    keep_idx: np.ndarray | None = None

    for obj in obj_names:
        df_obj = df[df["object"] == obj].copy()  # type: ignore[assignment]
        df_obj = df_obj.sort_values(by=["time_sec", "time_nsec"])  # type: ignore[arg-type]
        df_obj = df_obj.reset_index(drop=True)

        # Use float64 for time to avoid precision issues with large epoch seconds.
        t_sec: np.ndarray = np.asarray(df_obj["time_sec"], dtype=np.float64)
        t_nsec: np.ndarray = np.asarray(df_obj["time_nsec"], dtype=np.float64)
        t = t_sec + t_nsec * 1e-9

        dt = np.diff(t)
        bad_idx = np.where(dt <= 0)[0]
        if len(bad_idx) > 0:
            idx = bad_idx[0]
            raise ValueError(
                f"Time not monotonically increasing for {csv_path.name}, object '{obj}' "
                f"at frame {idx}: t[{idx}]={t[idx]:.6f}, t[{idx + 1}]={t[idx + 1]:.6f}, dt={dt[idx]:.9f}"
            )

        if t_ref_full is None:
            t_ref_full = t
            keep_idx = _compute_downsample_indices(t, sample_hz)
        else:
            if not np.allclose(t_ref_full, t):
                raise ValueError(
                    f"Time stamps differ between objects in {csv_path.name} "
                    f"(first object and '{obj}' have different t)."
                )

        qx: np.ndarray = np.asarray(df_obj["rotation_quaternion_x"], dtype=np.float32)
        qy: np.ndarray = np.asarray(df_obj["rotation_quaternion_y"], dtype=np.float32)
        qz: np.ndarray = np.asarray(df_obj["rotation_quaternion_z"], dtype=np.float32)
        qw: np.ndarray = np.asarray(df_obj["rotation_quaternion_w"], dtype=np.float32)
        quat = np.stack([qx, qy, qz, qw], axis=1)

        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        mask = norms[:, 0] > 0
        quat[mask] = quat[mask] / norms[mask]

        R = quat_to_rotmat(quat)

        if keep_idx is not None:
            quat = quat[keep_idx]
            R = R[keep_idx]

        arrays[f"quat_{obj}"] = quat
        arrays[f"R_{obj}"] = R

    if t_ref_full is None or keep_idx is None:
        raise ValueError(f"No objects found in {csv_path}")

    arrays["t"] = t_ref_full[keep_idx]

    out_path = output_dir / f"{segment_name}.npz"
    np.savez_compressed(out_path, **arrays)  # type: ignore[call-arg]


def make_triad_video(
    npz_path: str | Path,
    object_name: str,
    out_path: str | Path,
    fps: float = 24.0,
) -> None:
    """
    Render a video of the rotating body frame for a single object as three axes (X, Y, Z).

    Args:
        npz_path: Path to a .npz file produced by convert_vicon_csv_to_npz.
        object_name: Name of the object, e.g. "box1" or "sensor_payload".
        out_path: Output video path, e.g. "swinging_4_static_box1.mp4".
        fps: Frames per second for the video.
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)

    data = np.load(npz_path)
    key_R = f"R_{object_name}"
    if key_R not in data:
        raise KeyError(f"{key_R} not found in {npz_path.name}")

    R: np.ndarray = np.asarray(data[key_R], dtype=np.float32)
    t: np.ndarray = np.asarray(data["t"], dtype=np.float64)

    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"{key_R} has shape {R.shape}, expected (T, 3, 3)")
    if t.ndim != 1 or t.shape[0] != R.shape[0]:
        raise ValueError(f"t shape {t.shape} does not match R shape {R.shape}")

    # Lazily import matplotlib so core functionality does not require it.
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    basis: np.ndarray = np.eye(3, dtype=np.float32)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{npz_path.stem} – {object_name}")

    colors: list[str] = ["r", "g", "b"]
    lines: list[object] = []
    for i in range(3):
        (line,) = ax.plot(
            [0.0, float(basis[i, 0])],
            [0.0, float(basis[i, 1])],
            [0.0, float(basis[i, 2])],
            color=colors[i],
            linewidth=2.0,
        )
        lines.append(line)

    def update(frame: int) -> list[object]:  # type: ignore[type-arg]
        Rk: np.ndarray = R[frame]
        # Columns of Rk are the rotated basis vectors.
        for i, line in enumerate(lines):
            v = Rk[:, i]
            line.set_data([0.0, float(v[0])], [0.0, float(v[1])])  # type: ignore[union-attr]
            line.set_3d_properties([0.0, float(v[2])])  # type: ignore[union-attr]
        return lines

    interval_ms = 1000.0 / fps if fps > 0.0 else 0.0
    anim = FuncAnimation(fig, update, frames=R.shape[0], interval=interval_ms, blit=False)  # type: ignore[arg-type]
    anim.save(out_path, fps=int(fps))
    plt.close(fig)


def make_triad_video_all_objects(
    npz_path: str | Path,
    out_path: str | Path,
    fps: float = 24.0,
) -> None:
    """
    Render a video showing the rotating body frames for all objects in one common 3D frame.

    Args:
        npz_path: Path to a .npz file produced by convert_vicon_csv_to_npz.
        out_path: Output video path, e.g. "swinging_4_static_all.mp4".
        fps: Frames per second for the video.
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)

    data = np.load(npz_path)
    object_names = sorted(k[2:] for k in data.files if k.startswith("R_"))
    if not object_names:
        raise ValueError(f"No objects found in {npz_path}")

    R_list: list[np.ndarray] = []
    for name in object_names:
        key_R = f"R_{name}"
        R_arr = np.asarray(data[key_R], dtype=np.float32)
        R_list.append(R_arr)

    t: np.ndarray = np.asarray(data["t"], dtype=np.float64)
    T = t.shape[0]
    for R in R_list:
        if R.ndim != 3 or R.shape[1:] != (3, 3):
            raise ValueError(f"R array has shape {R.shape}, expected (T, 3, 3)")
        if R.shape[0] != T:
            raise ValueError(f"Mismatch between t shape {t.shape} and R shape {R.shape}")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    basis: np.ndarray = np.eye(3, dtype=np.float32)

    # Lay objects out on a grid in the XY plane so their frames do not overlap.
    n_obj = len(object_names)
    n_cols = int(np.ceil(np.sqrt(float(n_obj))))
    n_rows = int(np.ceil(n_obj / n_cols))
    spacing = 3.0

    offsets: dict[str, np.ndarray] = {}
    xs: list[float] = []
    ys: list[float] = []
    for idx, name in enumerate(object_names):
        row = idx // n_cols
        col = idx % n_cols
        x = (col - (n_cols - 1) / 2.0) * spacing
        y = ((n_rows - 1) / 2.0 - row) * spacing
        offset = np.asarray([x, y, 0.0], dtype=np.float32)
        offsets[name] = offset
        xs.append(float(x))
        ys.append(float(y))

    x_min = min(xs) - 1.5
    x_max = max(xs) + 1.5
    y_min = min(ys) - 1.5
    y_max = max(ys) + 1.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{npz_path.stem} – all objects")

    axis_colors: list[str] = ["r", "g", "b"]
    lines: dict[str, list[object]] = {}

    for name in object_names:
        obj_lines: list[object] = []
        for axis_idx in range(3):
            offset = offsets[name]
            (line,) = ax.plot(
                [float(offset[0]), float(offset[0] + basis[axis_idx, 0])],
                [float(offset[1]), float(offset[1] + basis[axis_idx, 1])],
                [float(offset[2]), float(offset[2] + basis[axis_idx, 2])],
                color=axis_colors[axis_idx],
                linewidth=1.5,
                alpha=0.5,
            )
            obj_lines.append(line)
        lines[name] = obj_lines

    def update(frame: int) -> list[object]:  # type: ignore[type-arg]
        for name, R_arr in zip(object_names, R_list):
            Rk: np.ndarray = R_arr[frame]
            for axis_idx, line in enumerate(lines[name]):
                offset = offsets[name]
                v = Rk[:, axis_idx]
                end = offset + v
                line.set_data([float(offset[0]), float(end[0])], [float(offset[1]), float(end[1])])  # type: ignore[union-attr]
                line.set_3d_properties([float(offset[2]), float(end[2])])  # type: ignore[union-attr]
        all_lines: list[object] = []
        for obj_lines in lines.values():
            all_lines.extend(obj_lines)
        return all_lines

    interval_ms = 1000.0 / fps if fps > 0.0 else 0.0
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)  # type: ignore[arg-type]
    anim.save(out_path, fps=int(fps))
    plt.close(fig)


if __name__ == "__main__":
    raw_data_dir = Path(__file__).parent.parent.parent / "raw_data" / "oxford_multimotion"
    output_dir = Path(__file__).parent.parent.parent / "data" / "oxford_multimotion"

    sample_hz = 40.0

    for csv_file in sorted(raw_data_dir.glob("*_vicon.csv")):
        convert_vicon_csv_to_npz(csv_file, output_dir, sample_hz=sample_hz)

        segment_name = csv_file.stem.replace("_vicon", "")
        npz_path = output_dir / f"{segment_name}.npz"
        video_path = output_dir / f"{segment_name}_all.mp4"
        make_triad_video_all_objects(npz_path, video_path, fps=sample_hz)
