from .camera_pipeline import CameraRGBDSLAM
from .kitti_pipeline import KITTIRGBDSLAM
from .tum_pipeline import TUMRGBDSLAM


def build_slam(args):
    if args.input_mode == "camera":
        return CameraRGBDSLAM(args)
    if args.input_mode == "tum":
        return TUMRGBDSLAM(args)
    if args.input_mode == "kitti":
        return KITTIRGBDSLAM(args)
    raise ValueError("input_mode 必須是 tum、camera 或 kitti")
