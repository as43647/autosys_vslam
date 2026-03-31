from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)
DEFAULT_SEQUENCES = tuple(range(11, 22))
STEP_SIZE = 10
PATH_STEP_SIZE = 3


@dataclass
class SequenceError:
    first_frame: int
    r_err: float
    t_err: float
    length: float
    speed: float


def load_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    if not path.exists():
        return poses

    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            values = np.fromstring(stripped, sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(f"{path} line {line_num} does not contain 12 values")
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(pose)
    return poses


def trajectory_distances(poses: list[np.ndarray]) -> list[float]:
    distances = [0.0]
    for i in range(1, len(poses)):
        delta = poses[i - 1][:3, 3] - poses[i][:3, 3]
        distances.append(distances[-1] + float(np.linalg.norm(delta)))
    return distances


def last_frame_from_segment_length(distances: list[float], first_frame: int, length: float) -> int:
    start_distance = distances[first_frame]
    for idx in range(first_frame, len(distances)):
        if distances[idx] > start_distance + length:
            return idx
    return -1


def rotation_error(pose_error: np.ndarray) -> float:
    trace = float(np.trace(pose_error[:3, :3]))
    value = 0.5 * (trace - 1.0)
    value = max(min(value, 1.0), -1.0)
    return math.acos(value)


def translation_error(pose_error: np.ndarray) -> float:
    return float(np.linalg.norm(pose_error[:3, 3]))


def calc_sequence_errors(poses_gt: list[np.ndarray], poses_result: list[np.ndarray]) -> list[SequenceError]:
    errors: list[SequenceError] = []
    distances = trajectory_distances(poses_gt)

    for first_frame in range(0, len(poses_gt), STEP_SIZE):
        for length in LENGTHS:
            last_frame = last_frame_from_segment_length(distances, first_frame, length)
            if last_frame == -1:
                continue

            pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) @ poses_gt[last_frame]
            pose_delta_result = np.linalg.inv(poses_result[first_frame]) @ poses_result[last_frame]
            pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt

            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            num_frames = float(last_frame - first_frame + 1)
            speed = length / (0.1 * num_frames)
            errors.append(SequenceError(first_frame, r_err / length, t_err / length, float(length), speed))

    return errors


def save_sequence_errors(errors: Iterable[SequenceError], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for err in errors:
            handle.write(
                f"{err.first_frame} {err.r_err:.6f} {err.t_err:.6f} {err.length:.6f} {err.speed:.6f}\n"
            )


def save_path_plot_data(poses_gt: list[np.ndarray], poses_result: list[np.ndarray], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(0, len(poses_gt), PATH_STEP_SIZE):
            handle.write(
                f"{poses_gt[idx][0, 3]:.6f} {poses_gt[idx][2, 3]:.6f} "
                f"{poses_result[idx][0, 3]:.6f} {poses_result[idx][2, 3]:.6f}\n"
            )


def compute_roi(poses_gt: list[np.ndarray], poses_result: list[np.ndarray]) -> tuple[float, float, float, float]:
    coords = np.array(
        [[pose[0, 3], pose[2, 3]] for pose in poses_gt] + [[pose[0, 3], pose[2, 3]] for pose in poses_result],
        dtype=np.float64,
    )
    x_min, z_min = coords.min(axis=0)
    x_max, z_max = coords.max(axis=0)
    dx = 1.1 * (x_max - x_min)
    dz = 1.1 * (z_max - z_min)
    mx = 0.5 * (x_max + x_min)
    mz = 0.5 * (z_max + z_min)
    radius = 0.5 * max(dx, dz, 1.0)
    return mx - radius, mx + radius, mz - radius, mz + radius


def save_error_plot_data(errors: list[SequenceError], plot_error_dir: Path, prefix: str) -> None:
    file_tl = plot_error_dir / f"{prefix}_tl.txt"
    file_rl = plot_error_dir / f"{prefix}_rl.txt"
    file_ts = plot_error_dir / f"{prefix}_ts.txt"
    file_rs = plot_error_dir / f"{prefix}_rs.txt"

    with file_tl.open("w", encoding="utf-8") as fp_tl, \
        file_rl.open("w", encoding="utf-8") as fp_rl, \
        file_ts.open("w", encoding="utf-8") as fp_ts, \
        file_rs.open("w", encoding="utf-8") as fp_rs:
        for length in LENGTHS:
            selected = [err for err in errors if abs(err.length - length) < 1.0]
            if len(selected) > 2:
                t_avg = sum(err.t_err for err in selected) / len(selected)
                r_avg = sum(err.r_err for err in selected) / len(selected)
                fp_tl.write(f"{length:.6f} {t_avg:.6f}\n")
                fp_rl.write(f"{length:.6f} {r_avg:.6f}\n")

        speed = 2.0
        while speed < 25.0:
            selected = [err for err in errors if abs(err.speed - speed) < 2.0]
            if len(selected) > 2:
                t_avg = sum(err.t_err for err in selected) / len(selected)
                r_avg = sum(err.r_err for err in selected) / len(selected)
                fp_ts.write(f"{speed:.6f} {t_avg:.6f}\n")
                fp_rs.write(f"{speed:.6f} {r_avg:.6f}\n")
            speed += 2.0


def load_xy_pairs(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 2), dtype=np.float64)
    data = np.loadtxt(path, dtype=np.float64)
    data = np.atleast_2d(data)
    return data


def maybe_plot_path(plot_path_dir: Path, seq_name: str, roi: tuple[float, float, float, float]) -> None:
    if plt is None:
        return

    data = load_xy_pairs(plot_path_dir / f"{seq_name}.txt")
    if data.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(data[:, 0], data[:, 1], color="#FF0000", label="Ground Truth")
    ax.plot(data[:, 2], data[:, 3], color="#0000FF", label="Visual Odometry")
    ax.scatter(data[0, 0], data[0, 1], color="#000000", label="Sequence Start", s=30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(roi[0], roi[1])
    ax.set_ylim(roi[2], roi[3])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path_dir / f"{seq_name}.png", dpi=150)
    plt.close(fig)


def maybe_plot_error_curves(plot_error_dir: Path, prefix: str) -> None:
    if plt is None:
        return

    specs = {
        "tl": ("Path Length [m]", "Translation Error [%]", lambda x, y: (x, y * 100.0)),
        "rl": ("Path Length [m]", "Rotation Error [deg/m]", lambda x, y: (x, y * 57.3)),
        "ts": ("Speed [km/h]", "Translation Error [%]", lambda x, y: (x * 3.6, y * 100.0)),
        "rs": ("Speed [km/h]", "Rotation Error [deg/m]", lambda x, y: (x * 3.6, y * 57.3)),
    }

    for suffix, (xlabel, ylabel, transform) in specs.items():
        data = load_xy_pairs(plot_error_dir / f"{prefix}_{suffix}.txt")
        if data.size == 0:
            continue

        x_vals, y_vals = transform(data[:, 0], data[:, 1])
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x_vals, y_vals, color="#0000FF", marker="o")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(plot_error_dir / f"{prefix}_{suffix}.png", dpi=150)
        plt.close(fig)


def save_stats(errors: list[SequenceError], result_dir: Path) -> None:
    if not errors:
        return
    t_avg = sum(err.t_err for err in errors) / len(errors)
    r_avg = sum(err.r_err for err in errors) / len(errors)
    with (result_dir / "stats.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{t_avg:.6f} {r_avg:.6f}\n")


def evaluate_pair(
    gt_path: Path,
    est_path: Path,
    output_dir: Path,
    prefix: str,
    skip_plots: bool,
) -> list[SequenceError]:
    poses_gt = load_poses(gt_path)
    poses_result = load_poses(est_path)

    print(f"Processing: {prefix}, poses: {len(poses_result)}/{len(poses_gt)}")
    if not poses_gt or len(poses_result) != len(poses_gt):
        raise ValueError(f"Couldn't read (all) poses of: {prefix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir = output_dir / "errors"
    plot_path_dir = output_dir / "plot_path"
    plot_error_dir = output_dir / "plot_error"
    error_dir.mkdir(parents=True, exist_ok=True)
    plot_path_dir.mkdir(parents=True, exist_ok=True)
    plot_error_dir.mkdir(parents=True, exist_ok=True)

    seq_errors = calc_sequence_errors(poses_gt, poses_result)
    save_sequence_errors(seq_errors, error_dir / f"{prefix}.txt")
    save_error_plot_data(seq_errors, plot_error_dir, prefix)
    save_stats(seq_errors, output_dir)

    if not skip_plots and plt is not None:
        save_path_plot_data(poses_gt, poses_result, plot_path_dir / f"{prefix}.txt")
        roi = compute_roi(poses_gt, poses_result)
        maybe_plot_path(plot_path_dir, prefix, roi)
        maybe_plot_error_curves(plot_error_dir, prefix)

    return seq_errors


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    devkit_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Python port of KITTI evaluate_odometry.cpp")
    parser.add_argument("result_sha", nargs="?", help="Result folder name under results/")
    parser.add_argument("user_sha", nargs="?", help="Ignored compatibility argument")
    parser.add_argument("email", nargs="?", help="Ignored compatibility argument")
    parser.add_argument("--gt_dir", default=str(devkit_root / "data" / "odometry" / "poses"), help="Ground-truth pose directory")
    parser.add_argument("--result_root", default=str(devkit_root / "results"), help="Root directory containing result folders")
    parser.add_argument("--gt_file", help="Evaluate a single KITTI GT pose file directly")
    parser.add_argument("--est_file", help="Evaluate a single estimated KITTI pose file directly")
    parser.add_argument("--output_dir", help="Output directory for single-file mode")
    parser.add_argument("--prefix", default="single", help="Output file prefix for single-file mode")
    parser.add_argument(
        "--sequences",
        nargs="*",
        type=int,
        default=list(DEFAULT_SEQUENCES),
        help="Sequence ids to evaluate, e.g. --sequences 11 12 13",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip PNG plot generation even if matplotlib is available",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    single_file_mode = bool(args.gt_file or args.est_file)
    if single_file_mode:
        if not (args.gt_file and args.est_file):
            print("ERROR: --gt_file and --est_file must be provided together.")
            return 1
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.est_file).resolve().parent / "eval_output"
        seq_errors = evaluate_pair(
            Path(args.gt_file),
            Path(args.est_file),
            output_dir,
            args.prefix,
            args.skip_plots,
        )
        t_avg = sum(err.t_err for err in seq_errors) / len(seq_errors)
        r_avg = sum(err.r_err for err in seq_errors) / len(seq_errors)
        print("Average odometry error:")
        print(f"  translation: {t_avg * 100.0:.4f} %")
        print(f"  rotation   : {r_avg * 57.3:.4f} deg/m")
        print(f"Saved results to: {output_dir}")
        return 0

    if not args.result_sha:
        print("ERROR: result_sha is required unless --gt_file and --est_file are used.")
        return 1

    gt_dir = Path(args.gt_dir)
    result_dir = Path(args.result_root) / args.result_sha
    result_data_dir = result_dir / "data"
    error_dir = result_dir / "errors"
    plot_path_dir = result_dir / "plot_path"
    plot_error_dir = result_dir / "plot_error"

    error_dir.mkdir(parents=True, exist_ok=True)
    plot_path_dir.mkdir(parents=True, exist_ok=True)
    plot_error_dir.mkdir(parents=True, exist_ok=True)

    total_errors: list[SequenceError] = []

    print("Thank you for participating in our evaluation!")
    if plt is None and not args.skip_plots:
        print("[Warning] matplotlib is not available; PNG plot generation will be skipped.")

    for seq in args.sequences:
        seq_name = f"{seq:02d}"
        gt_path = gt_dir / f"{seq_name}.txt"
        est_path = result_data_dir / f"{seq_name}.txt"

        poses_gt = load_poses(gt_path)
        poses_result = load_poses(est_path)

        print(f"Processing: {seq_name}.txt, poses: {len(poses_result)}/{len(poses_gt)}")
        if not poses_gt or len(poses_result) != len(poses_gt):
            print(f"ERROR: Couldn't read (all) poses of: {seq_name}.txt")
            return 1

        seq_errors = calc_sequence_errors(poses_gt, poses_result)
        save_sequence_errors(seq_errors, error_dir / f"{seq_name}.txt")
        total_errors.extend(seq_errors)

        if seq <= 15:
            save_path_plot_data(poses_gt, poses_result, plot_path_dir / f"{seq_name}.txt")
            save_error_plot_data(seq_errors, plot_error_dir, seq_name)
            if not args.skip_plots and plt is not None:
                roi = compute_roi(poses_gt, poses_result)
                maybe_plot_path(plot_path_dir, seq_name, roi)
                maybe_plot_error_curves(plot_error_dir, seq_name)

    if total_errors:
        save_error_plot_data(total_errors, plot_error_dir, "avg")
        if not args.skip_plots and plt is not None:
            maybe_plot_error_curves(plot_error_dir, "avg")
        save_stats(total_errors, result_dir)

        t_avg = sum(err.t_err for err in total_errors) / len(total_errors)
        r_avg = sum(err.r_err for err in total_errors) / len(total_errors)
        print("Average odometry error:")
        print(f"  translation: {t_avg * 100.0:.4f} %")
        print(f"  rotation   : {r_avg * 57.3:.4f} deg/m")

    print(f"Saved results to: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
