import time
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .depth import DepthInference
from .features import FeatureExtractor
from .segment import YoloMaskTracker


@dataclass
class Keyframe:
    id: int
    pose: np.ndarray
    depth: np.ndarray
    features: dict


class BaseRGBDSLAM:
    def __init__(self, args):
        self.args = args
        self.loader = self.build_loader()
        self.mask_tracker = YoloMaskTracker(
            args.seg_path,
            interval=args.yolo_interval,
            imgsz=args.yolo_imgsz,
            geo_proj_step=args.geo_proj_step,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            persistence=args.mask_persistence,
            dilate_kernel=args.mask_dilate_kernel,
        )
        self.depth_estimator = None
        use_depth_model = not (
            (args.input_mode == "tum" and args.use_gt_depth)
            or (args.input_mode == "kitti" and getattr(args, "kitti_depth_source", "model") == "stereo")
        )
        if use_depth_model:
            self.depth_estimator = DepthInference(
                args.model_path,
                input_size=(args.depth_input_width, args.depth_input_height),
            )
        self.frontend = FeatureExtractor(
            max_kpts=args.max_kpts,
            match_threshold=args.match_threshold,
        )

        self.K = None
        self.focal_px = None
        self.init_intrinsics_default()

        self.keyframes = []
        self.current_pose = np.eye(4)
        self.traj_log = []
        self.pose_history = []

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Trajectory", 1024, 768)
        self.traj_geo = o3d.geometry.LineSet()
        self.vis.add_geometry(self.traj_geo)
        self.print_runtime_summary()

    def build_loader(self):
        raise NotImplementedError

    def init_intrinsics_default(self):
        raise NotImplementedError

    def maybe_init_runtime_intrinsics(self, img):
        del img

    def get_depth_mm(self, img, gt_depth):
        raise NotImplementedError

    def cleanup(self):
        pass

    def print_runtime_summary(self):
        if self.depth_estimator is None:
            depth_info = {
                "model_loaded": False,
                "requested_backend": "disabled",
                "active_providers": [],
                "using_gpu": False,
                "warmup_seconds": None,
            }
        else:
            depth_info = self.depth_estimator.get_runtime_info()
        feat_info = self.frontend.get_runtime_info()
        yolo_info = self.mask_tracker.get_runtime_info()

        print("\n" + "=" * 60)
        print(f"{'RUNTIME DEVICE SUMMARY':^60}")
        print("=" * 60)
        print(f"Input mode          : {self.args.input_mode}")
        print(f"PyTorch CUDA        : {feat_info['torch_cuda_available']}")
        if feat_info["torch_cuda_available"]:
            print(f"PyTorch GPU         : {feat_info['torch_device_name']}")
            print(f"PyTorch device idx  : {feat_info['torch_current_device']}")
        print(f"XFeat model device  : {feat_info['model_device']}")
        print(f"YOLO model device   : {yolo_info['model_device']}")
        print(f"ONNX providers      : {depth_info['active_providers']}")
        print(f"Depth using GPU     : {depth_info['using_gpu']}")
        if depth_info.get("warmup_seconds") is not None:
            print(f"Depth warm-up       : {depth_info['warmup_seconds']:.2f}s")
        print(f"YOLO imgsz          : {yolo_info['imgsz']}")
        print(f"YOLO interval       : {yolo_info['interval']}")
        print(f"Mask persistence    : {yolo_info['persistence']} frames")
        print(f"Mask dilate kernel  : {yolo_info['dilate_kernel']}")
        print("=" * 60)

        if not feat_info["torch_cuda_available"]:
            print("[警告] PyTorch 沒有偵測到 CUDA，XFeat/YOLO 很可能只能走 CPU。")
        if depth_info["model_loaded"] and not depth_info["using_gpu"]:
            print("[警告] ONNX Runtime 沒有使用 CUDAExecutionProvider，深度模型目前走 CPU fallback。")

    def process(self):
        fps_list, latency_list = [], []
        mask_times, feat_times, pnp_times = [], [], []

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = None
        rel_pose = None

        print(
            f"[系統] 開始處理，input_mode={self.args.input_mode}，"
            f"每 {self.mask_tracker.interval} 幀執行一次 YOLO..."
        )
        start_wall_time = time.time()

        i = 0
        while True:
            t_frame_start = time.time()
            img, gt_depth, ts = self.loader.get_frame(i)
            if img is None:
                break

            self.maybe_init_runtime_intrinsics(img)

            depth_mm = self.get_depth_mm(img, gt_depth)
            if depth_mm is None:
                print("[警告] 無法取得深度，結束處理")
                break

            mask, mask_mode, dt_mask = self.mask_tracker.get_static_mask(
                img,
                i,
                depth_mm,
                self.K,
                rel_pose,
            )
            mask_times.append(dt_mask)

            t_f_s = time.time()
            feats = self.frontend.extract(img)

            kpts = feats["keypoints"].cpu().numpy()
            v = kpts[:, 1].astype(int)
            u = kpts[:, 0].astype(int)
            valid_kpts = (
                (u >= 0)
                & (u < img.shape[1])
                & (v >= 0)
                & (v < img.shape[0])
                & (mask[v, u] > 0)
            )

            disp_img = img.copy()
            for idx in range(len(kpts)):
                color = (0, 255, 0) if valid_kpts[idx] else (0, 0, 255)
                cv2.circle(disp_img, (int(kpts[idx, 0]), int(kpts[idx, 1])), 2, color, -1)

            feats["keypoints"] = feats["keypoints"][valid_kpts]
            feats["descriptors"] = feats["descriptors"][valid_kpts]
            feat_times.append((time.time() - t_f_s) * 1000)

            t_p_s = time.time()
            if i == 0:
                self.current_pose = np.eye(4)
                self.keyframes.append(Keyframe(i, self.current_pose.copy(), depth_mm, feats))
            else:
                ref = self.keyframes[-1]
                idx_ref, idx_cur = self.frontend.match(ref.features["descriptors"], feats["descriptors"])

                if len(idx_ref) >= self.args.min_match_count:
                    kp_ref = ref.features["keypoints"][idx_ref].cpu().numpy()
                    kp_cur = feats["keypoints"][idx_cur].cpu().numpy()

                    pts_3d = []
                    pts_2d = []

                    for (ur, vr), (uc, vc) in zip(kp_ref.astype(int), kp_cur):
                        if vr < 0 or vr >= ref.depth.shape[0] or ur < 0 or ur >= ref.depth.shape[1]:
                            continue

                        z = ref.depth[vr, ur] / 1000.0
                        if not np.isfinite(z) or z <= self.args.min_depth_m or z >= self.args.max_depth_m:
                            continue

                        x = (ur - self.K[0, 2]) * z / self.K[0, 0]
                        y = (vr - self.K[1, 2]) * z / self.K[1, 1]
                        pts_3d.append([x, y, z])
                        pts_2d.append([uc, vc])

                    pts_3d = np.array(pts_3d, dtype=np.float32)
                    pts_2d = np.array(pts_2d, dtype=np.float32)

                    if len(pts_3d) >= 6:
                        ok, rvec, tvec, _ = cv2.solvePnPRansac(pts_3d, pts_2d, self.K, None)
                        if ok:
                            r_mat, _ = cv2.Rodrigues(rvec)
                            transform = np.eye(4)
                            transform[:3, :3] = r_mat
                            transform[:3, 3] = tvec.squeeze()

                            rel_pose = transform
                            self.current_pose = self.current_pose @ np.linalg.inv(transform)
                            self.keyframes.append(Keyframe(i, self.current_pose.copy(), depth_mm, feats))

                            q = R.from_matrix(self.current_pose[:3, :3]).as_quat()
                            t = self.current_pose[:3, 3]
                            self.traj_log.append(
                                f"{ts} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}"
                            )

            self.pose_history.append((ts, self.current_pose.copy()))

            pnp_times.append((time.time() - t_p_s) * 1000)

            dt_total = (time.time() - t_frame_start) * 1000
            latency_list.append(dt_total)
            fps_list.append(1000.0 / max(dt_total, 1e-6))

            mask_overlay = cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)
            final_view = cv2.addWeighted(disp_img, 0.8, mask_overlay, 0.2, 0)
            cv2.putText(
                final_view,
                f"FPS: {fps_list[-1]:.1f} | {mask_mode} | mode={self.args.input_mode}",
                (10, 30),
                1,
                1.2,
                (0, 255, 255),
                2,
            )

            if out_video is None:
                out_video = cv2.VideoWriter(
                    self.get_video_output_path(),
                    fourcc,
                    20.0,
                    (img.shape[1], img.shape[0]),
                )
            out_video.write(final_view)

            cv2.imshow("Hybrid SLAM", final_view)
            self.update_vis()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            i += 1

        self._print_report(latency_list, mask_times, feat_times, pnp_times, start_wall_time, time.time())
        self.save_results()

        if out_video:
            out_video.release()

        self.cleanup()
        cv2.destroyAllWindows()
        self.vis.destroy_window()

    def _print_report(self, latency_list, mask_times, feat_times, pnp_times, start_wall_time, end_wall_time):
        total_frames = len(latency_list)
        total_elapsed_sec = end_wall_time - start_wall_time
        if total_frames == 0:
            return

        avg_system_fps = total_frames / max(total_elapsed_sec, 1e-6)
        latencies = np.array(latency_list)
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        p95_lat = np.percentile(latencies, 95)

        avg_mask = np.mean(mask_times) if mask_times else 0
        avg_feat = np.mean(feat_times) if feat_times else 0
        avg_pnp = np.mean(pnp_times) if pnp_times else 0
        sum_components = max(avg_mask + avg_feat + avg_pnp, 1e-6)

        print("\n" + "=" * 60)
        print(f"{'SLAM SYSTEM PERFORMANCE REPORT':^60}")
        print("=" * 60)
        print(f"總處理幀數   : {total_frames:>10} frames")
        print(f"系統吞吐量   : {avg_system_fps:>10.2f} FPS")
        print("-" * 60)
        print("延遲統計 (Latency Statistics):")
        print(f" - 平均延遲 (Mean) : {mean_lat:>10.2f} ms")
        print(f" - 標準差   (Std)  : {std_lat:>10.2f} ms")
        print(f" - P95 穩定指標    : {p95_lat:>10.2f} ms")
        print("-" * 60)
        print("耗時分項分析 (Component Breakdown):")
        print(f" - Mask (Hybrid)   : {avg_mask:>8.2f} ms ({avg_mask / sum_components * 100:>5.1f}%)")
        print(f" - Feat (XFeat)    : {avg_feat:>8.2f} ms ({avg_feat / sum_components * 100:>5.1f}%)")
        print(f" - Pose (PnP)      : {avg_pnp:>8.2f} ms ({avg_pnp / sum_components * 100:>5.1f}%)")
        print("=" * 60)

    def update_vis(self):
        if len(self.keyframes) < 2:
            return

        pts = [kf.pose[:3, 3] for kf in self.keyframes]
        self.traj_geo.points = o3d.utility.Vector3dVector(pts)
        self.traj_geo.lines = o3d.utility.Vector2iVector([[j, j + 1] for j in range(len(pts) - 1)])
        self.vis.update_geometry(self.traj_geo)
        self.vis.poll_events()
        self.vis.update_renderer()

    def save_results(self):
        output_path = self.get_trajectory_output_path()
        lines = self.format_trajectory_lines()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def get_trajectory_output_path(self):
        if self.args.trajectory_output:
            return self.args.trajectory_output
        return "camera_trajectories_walking_testH.txt"

    def get_video_output_path(self):
        if getattr(self.args, "video_output", None):
            return self.args.video_output
        return "testH.mp4"

    def format_trajectory_lines(self):
        if self.pose_history:
            lines = []
            for ts, pose in self.pose_history:
                q = R.from_matrix(pose[:3, :3]).as_quat()
                t = pose[:3, 3]
                lines.append(f"{ts} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}")
            return lines
        return self.traj_log
