import cv2
import numpy as np

from .core import BaseRGBDSLAM
from .loaders import KITTILoader


class KITTIRGBDSLAM(BaseRGBDSLAM):
    def __init__(self, args):
        self.stereo_matcher = None
        super().__init__(args)

    def build_loader(self):
        if not self.args.dataset_path:
            raise ValueError("kitti mode requires --dataset_path")
        return KITTILoader(
            self.args.dataset_path,
            image_folder=self.args.kitti_image_folder,
            right_image_folder=self.args.kitti_right_image_folder,
            pose_path=self.args.kitti_pose_path,
        )

    def init_intrinsics_default(self):
        self.K = self.loader.K.copy()
        self.focal_px = float(self.K[0, 0])
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5 * 5,
            P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        print("[Intrinsics] loaded KITTI intrinsics from calib.txt")
        print(self.K)
        print(f"[KITTI] depth source: {self.args.kitti_depth_source}")

    def get_depth_mm(self, img, gt_depth):
        if self.args.kitti_depth_source == "stereo":
            if gt_depth is None:
                raise RuntimeError(
                    f"Stereo depth requested, but right images were not found in '{self.args.kitti_right_image_folder}'."
                )
            return self._stereo_depth_mm(img, gt_depth)

        disp = self.depth_estimator.process_disparity(img)
        if disp is None:
            return None
        return self.depth_estimator.disparity_to_depth_mm(disp, self.focal_px, self.args.baseline)

    def _stereo_depth_mm(self, left_bgr, right_bgr):
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity[disparity <= 0.0] = 0.0

        depth_mm = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.0
        depth_mm[valid] = (self.focal_px * self.args.baseline) / disparity[valid] * 1000.0
        return depth_mm

    def get_trajectory_output_path(self):
        if self.args.trajectory_output:
            return self.args.trajectory_output
        return "kitti_trajectory.txt"

    def format_trajectory_lines(self):
        lines = []
        for _, pose in self.pose_history:
            pose_3x4 = pose[:3, :4].reshape(-1)
            lines.append(" ".join(str(v) for v in pose_3x4))
        return lines
