import argparse

import numpy as np


def load_kitti_poses(path):
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            values = np.fromstring(stripped, sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(f"{path} 第 {line_num} 行不是 12 個數值")

            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(pose)

    if not poses:
        raise ValueError(f"{path} 沒有可用的 pose")

    return np.stack(poses, axis=0)


def align_positions_horn(est_xyz, gt_xyz):
    est_center = est_xyz.mean(axis=0)
    gt_center = gt_xyz.mean(axis=0)

    est_zero = est_xyz - est_center
    gt_zero = gt_xyz - gt_center

    w = est_zero.T @ gt_zero
    u, _, vh = np.linalg.svd(w)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        s[2, 2] = -1

    rot = u @ s @ vh
    trans = gt_center - rot @ est_center
    return rot, trans


def apply_alignment(poses, rot, trans):
    aligned = poses.copy()
    aligned[:, :3, :3] = rot @ aligned[:, :3, :3]
    aligned[:, :3, 3] = (rot @ aligned[:, :3, 3].T).T + trans
    return aligned


def compute_ate(gt_poses, est_poses, align):
    gt_xyz = gt_poses[:, :3, 3]
    est_xyz = est_poses[:, :3, 3]

    if align:
        rot, trans = align_positions_horn(est_xyz, gt_xyz)
        est_xyz = (rot @ est_xyz.T).T + trans

    errors = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "count": int(errors.shape[0]),
    }


def relative_transform(a, b):
    return np.linalg.inv(a) @ b


def rotation_error_deg(transform):
    rot = transform[:3, :3]
    cos_theta = (np.trace(rot) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_rpe(gt_poses, est_poses, delta):
    if delta <= 0:
        raise ValueError("delta 必須大於 0")
    if len(gt_poses) <= delta or len(est_poses) <= delta:
        raise ValueError("軌跡太短，無法計算指定 delta 的 RPE")

    trans_errors = []
    rot_errors = []

    for i in range(len(gt_poses) - delta):
        gt_rel = relative_transform(gt_poses[i], gt_poses[i + delta])
        est_rel = relative_transform(est_poses[i], est_poses[i + delta])
        err = np.linalg.inv(gt_rel) @ est_rel

        trans_errors.append(np.linalg.norm(err[:3, 3]))
        rot_errors.append(rotation_error_deg(err))

    trans_errors = np.asarray(trans_errors)
    rot_errors = np.asarray(rot_errors)

    return {
        "delta": delta,
        "count": int(trans_errors.shape[0]),
        "trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "trans_mean": float(np.mean(trans_errors)),
        "trans_median": float(np.median(trans_errors)),
        "trans_std": float(np.std(trans_errors)),
        "trans_min": float(np.min(trans_errors)),
        "trans_max": float(np.max(trans_errors)),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot_errors ** 2))),
        "rot_mean_deg": float(np.mean(rot_errors)),
        "rot_median_deg": float(np.median(rot_errors)),
        "rot_std_deg": float(np.std(rot_errors)),
        "rot_min_deg": float(np.min(rot_errors)),
        "rot_max_deg": float(np.max(rot_errors)),
    }


def print_ate(result):
    print("ATE Results")
    print(f"frames   : {result['count']}")
    print(f"rmse     : {result['rmse']:.6f} m")
    print(f"mean     : {result['mean']:.6f} m")
    print(f"median   : {result['median']:.6f} m")
    print(f"std      : {result['std']:.6f} m")
    print(f"min      : {result['min']:.6f} m")
    print(f"max      : {result['max']:.6f} m")


def print_rpe(result):
    print("RPE Results")
    print(f"delta    : {result['delta']} frames")
    print(f"pairs    : {result['count']}")
    print(f"trans rmse     : {result['trans_rmse']:.6f} m")
    print(f"trans mean     : {result['trans_mean']:.6f} m")
    print(f"trans median   : {result['trans_median']:.6f} m")
    print(f"trans std      : {result['trans_std']:.6f} m")
    print(f"trans min      : {result['trans_min']:.6f} m")
    print(f"trans max      : {result['trans_max']:.6f} m")
    print(f"rot rmse       : {result['rot_rmse_deg']:.6f} deg")
    print(f"rot mean       : {result['rot_mean_deg']:.6f} deg")
    print(f"rot median     : {result['rot_median_deg']:.6f} deg")
    print(f"rot std        : {result['rot_std_deg']:.6f} deg")
    print(f"rot min        : {result['rot_min_deg']:.6f} deg")
    print(f"rot max        : {result['rot_max_deg']:.6f} deg")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate KITTI-format trajectories.")
    parser.add_argument("--gt", required=True, help="KITTI ground truth pose file")
    parser.add_argument("--est", required=True, help="Estimated KITTI pose file")
    parser.add_argument("--metric", choices=["ate", "rpe", "both"], default="both")
    parser.add_argument("--delta", type=int, default=1, help="Frame interval for RPE")
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply Horn alignment before ATE evaluation",
    )
    return parser


def main():
    args = build_parser().parse_args()

    gt_poses = load_kitti_poses(args.gt)
    est_poses = load_kitti_poses(args.est)

    n = min(len(gt_poses), len(est_poses))
    if len(gt_poses) != len(est_poses):
        print(f"[警告] GT 幀數={len(gt_poses)}，EST 幀數={len(est_poses)}，將截斷為前 {n} 幀")

    gt_poses = gt_poses[:n]
    est_poses = est_poses[:n]

    if args.metric in {"ate", "both"}:
        ate_result = compute_ate(gt_poses, est_poses, align=args.align)
        print_ate(ate_result)
        if args.metric == "both":
            print()

    if args.metric in {"rpe", "both"}:
        rpe_result = compute_rpe(gt_poses, est_poses, delta=args.delta)
        print_rpe(rpe_result)


if __name__ == "__main__":
    main()
