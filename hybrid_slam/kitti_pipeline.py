from .core import BaseRGBDSLAM
from .loaders import KITTILoader


class KITTIRGBDSLAM(BaseRGBDSLAM):
    def build_loader(self):
        if not self.args.dataset_path:
            raise ValueError("kitti 模式下必須提供 --dataset_path")
        return KITTILoader(
            self.args.dataset_path,
            image_folder=self.args.kitti_image_folder,
            pose_path=self.args.kitti_pose_path,
        )

    def init_intrinsics_default(self):
        self.K = self.loader.K.copy()
        self.focal_px = float(self.K[0, 0])
        print("[Intrinsics] 使用 KITTI calib.txt 內參")
        print(self.K)

    def get_depth_mm(self, img, gt_depth):
        del gt_depth
        disp = self.depth_estimator.process_disparity(img)
        if disp is None:
            return None
        return self.depth_estimator.disparity_to_depth_mm(disp, self.focal_px, self.args.baseline)

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
