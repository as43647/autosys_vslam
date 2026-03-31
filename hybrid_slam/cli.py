import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_mode", type=str, choices=["tum", "camera", "kitti"], default="camera")

    parser.add_argument("--dataset_path", type=str, default="D:\\dataset\\tum\\rgbd_dataset_freiburg3_walking_xyz")
    parser.add_argument("--use_gt_depth", action="store_true")

    parser.add_argument("--kitti_image_folder", type=str, default="image_2")
    parser.add_argument("--kitti_right_image_folder", type=str, default="image_3")
    parser.add_argument("--kitti_pose_path", type=str, default=None)
    parser.add_argument("--kitti_depth_source", type=str, choices=["model", "stereo"], default="model")
    parser.add_argument("--trajectory_output", type=str, default="trajectory.txt")
    parser.add_argument("--video_output", type=str, default="trajectory.mp4")

    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)

    parser.add_argument("--model_path", type=str, default="D:\\autosys_vslam\\autosys_vslam\\depth_model\\autosys_mdepth_optm.onnx")
    parser.add_argument("--seg_path", type=str, default="D:\\autosys_vslam\\autosys_vslam\\yolo\\yolo11n-seg.pt")
    parser.add_argument("--max_kpts", type=int, default=500)
    parser.add_argument("--yolo_interval", type=int, default=1)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--geo_proj_step", type=int, default=1)
    parser.add_argument("--mask_persistence", type=int, default=3)
    parser.add_argument("--mask_dilate_kernel", type=int, default=15)

    parser.add_argument("--match_threshold", type=float, default=0.82)
    parser.add_argument("--min_match_count", type=int, default=25)

    parser.add_argument("--depth_input_width", type=int, default=256)
    parser.add_argument("--depth_input_height", type=int, default=256)
    parser.add_argument("--min_depth_m", type=float, default=0.1)
    parser.add_argument("--max_depth_m", type=float, default=30.0) # 8.0
    parser.add_argument("--baseline", type=float, default=0.54) # 0.1

    # TUM 1
    # parser.add_argument("--fx", type=float, default=517.3)
    # parser.add_argument("--fy", type=float, default=516.5)
    # parser.add_argument("--cx", type=float, default=318.6)
    # parser.add_argument("--cy", type=float, default=255.3)

    # # TUM 3
    # parser.add_argument("--fx", type=float, default=535.4)
    # parser.add_argument("--fy", type=float, default=539.2)
    # parser.add_argument("--cx", type=float, default=320.1)
    # parser.add_argument("--cy", type=float, default=247.6)

    # 相機內參
    # parser.add_argument("--fx", type=float, default=843.3496)
    # parser.add_argument("--fy", type=float, default=845.6298)
    # parser.add_argument("--cx", type=float, default=334.5379)
    # parser.add_argument("--cy", type=float, default=245.5660)

    # KITTI
    parser.add_argument("--fx", type=float, default=718.856)
    parser.add_argument("--fy", type=float, default=718.856)
    parser.add_argument("--cx", type=float, default=607.1928)
    parser.add_argument("--cy", type=float, default=185.2157)

    return parser
