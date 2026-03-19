import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YoloMaskTracker:
    def __init__(
        self,
        model_path,
        interval=None,
        classes=None,
        imgsz=320,
        geo_proj_step=None,
        min_depth_m=0.1,
        max_depth_m=8.0,
    ):
        self.seg_model = YOLO(model_path)
        self.interval = interval
        self.classes = classes if classes is not None else [0, 1, 2, 3] # 預設為人、腳踏車、汽車、摩托車
        self.imgsz = imgsz
        self.geo_proj_step = geo_proj_step
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.last_mask = None
        self.last_depth = None

    def get_static_mask(self, img_bgr, frame_idx, depth_mm, intrinsics, rel_pose=None):
        h, w = img_bgr.shape[:2]
        t_start = time.time()

        if frame_idx % self.interval == 0 or self.last_mask is None or rel_pose is None:
            mask = self._run_yolo(img_bgr, w, h)
            self.last_mask = mask.copy()
            self.last_depth = depth_mm.copy()
            mode = "YOLO-SEG"
        else:
            mask = self._project_cached_mask(w, h, intrinsics, rel_pose)
            mode = "GEO-PROJ"

        dt = (time.time() - t_start) * 1000
        return mask, mode, dt

    def _run_yolo(self, img_bgr, width, height):
        results = self.seg_model(img_bgr, classes=self.classes, verbose=False, imgsz=self.imgsz)
        mask = np.ones((height, width), dtype=np.uint8) * 255

        if results[0].masks is not None:
            for seg in results[0].masks.data:
                resized = cv2.resize(seg.cpu().numpy(), (width, height), interpolation=cv2.INTER_NEAREST)
                dilated = cv2.dilate((resized > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8))
                mask[dilated > 0] = 0

        return mask

    def _project_cached_mask(self, width, height, intrinsics, rel_pose):
        mask = np.ones((height, width), dtype=np.uint8) * 255
        ys, xs = np.where(self.last_mask == 0)

        if len(xs) == 0:
            return mask

        xs, ys = xs[:: self.geo_proj_step], ys[:: self.geo_proj_step]
        z = self.last_depth[ys, xs] / 1000.0
        valid = (z > self.min_depth_m) & (z < self.max_depth_m)
        xs, ys, z = xs[valid], ys[valid], z[valid]

        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        pts_3d = np.vstack(
            [
                (xs - cx) * z / fx,
                (ys - cy) * z / fy,
                z,
            ]
        )

        r_rel = rel_pose[:3, :3]
        t_rel = rel_pose[:3, 3].reshape(3, 1)
        pts_new = r_rel @ pts_3d + t_rel

        valid_z = pts_new[2] > 1e-6
        pts_new = pts_new[:, valid_z]

        u_new = (pts_new[0] / pts_new[2] * fx + cx).astype(int)
        v_new = (pts_new[1] / pts_new[2] * fy + cy).astype(int)

        in_view = (u_new >= 0) & (u_new < width) & (v_new >= 0) & (v_new < height)
        mask[v_new[in_view], u_new[in_view]] = 0
        mask = cv2.dilate(255 - mask, np.ones((11, 11), np.uint8))
        return 255 - mask

    def get_runtime_info(self):
        model_device = "unknown"
        try:
            model_device = str(next(self.seg_model.model.parameters()).device)
        except Exception:
            pass

        return {
            "torch_cuda_available": torch.cuda.is_available(),
            "model_device": model_device,
            "imgsz": self.imgsz,
            "interval": self.interval,
        }
