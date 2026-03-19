from .camera_pipeline import CameraRGBDSLAM
from .tum_pipeline import TUMRGBDSLAM


def build_slam(args):
    if args.input_mode == "camera":
        return CameraRGBDSLAM(args)
    if args.input_mode == "tum":
        return TUMRGBDSLAM(args)
    raise ValueError("input_mode 必須是 tum 或 camera")
