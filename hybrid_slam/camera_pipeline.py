import numpy as np

from .core import BaseRGBDSLAM
from .loaders import CameraLoader


class CameraRGBDSLAM(BaseRGBDSLAM):
    def build_loader(self):
        return CameraLoader(self.args.camera_id, self.args.cam_width, self.args.cam_height)

    def init_intrinsics_default(self):
        self.K = np.eye(3, dtype=np.float64)
        self.focal_px = None

    def maybe_init_runtime_intrinsics(self, img):
        if self.focal_px is not None:
            return

        h, w = img.shape[:2]
        fx = self.args.fx if self.args.fx is not None else 0.9 * w
        fy = self.args.fy if self.args.fy is not None else 0.9 * w
        cx = self.args.cx if self.args.cx is not None else w / 2.0
        cy = self.args.cy if self.args.cy is not None else h / 2.0

        self.K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.focal_px = fx

        print("[Intrinsics] Camera 模式內參初始化完成")
        print(self.K)
        if self.args.fx is None or self.args.fy is None:
            print("[警告] 目前 fx/fy 使用近似值，只適合 demo，不適合正式量測")

    def get_depth_mm(self, img, gt_depth):
        del gt_depth
        disp = self.depth_estimator.process_disparity(img)
        if disp is None:
            return None

        return self.depth_estimator.disparity_to_depth_mm(disp, self.focal_px, self.args.baseline)

    def cleanup(self):
        self.loader.release()
