import numpy as np

from .core import BaseRGBDSLAM
from .loaders import TUMDatasetLoader


class TUMRGBDSLAM(BaseRGBDSLAM):
    def build_loader(self):
        if not self.args.dataset_path:
            raise ValueError("tum 模式下必須提供 --dataset_path")
        return TUMDatasetLoader(self.args.dataset_path)

    def init_intrinsics_default(self):
        self.K = np.array(
            [
                [517.3, 0, 318.6],
                [0, 516.5, 255.3],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.focal_px = 517.3
        print("[Intrinsics] 使用 TUM 預設內參")

    def get_depth_mm(self, img, gt_depth):
        if self.args.use_gt_depth:
            if gt_depth is None:
                return None
            return gt_depth.astype(np.float32) / 5.0

        disp = self.depth_estimator.process_disparity(img)
        if disp is None:
            return None

        return self.depth_estimator.disparity_to_depth_mm(disp, self.focal_px, self.args.baseline)
