import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    # 輸入模式: TUM數據集還是相機
    parser.add_argument("--input_mode", type=str, choices=["tum", "camera", "kitti"], default="camera")

    parser.add_argument("--dataset_path", type=str, default="C:\\Users\\as436\\OneDrive\\文件\\xfeat\\00") #rgbd_dataset_freiburg3_walking_xyz
    parser.add_argument("--use_gt_depth", action="store_true")
    parser.add_argument("--kitti_image_folder", type=str, default="image_2")
    parser.add_argument("--kitti_pose_path", type=str, default=None)

    # 相機參數
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--cam_width", type=int, default=640)
    parser.add_argument("--cam_height", type=int, default=480)

    parser.add_argument("--model_path", type=str, default="C:\\Users\\as436\\OneDrive\\文件\\xfeat\\depth_model\\autosys_mdepth_optm.onnx") # 深度估計模型路徑
    parser.add_argument("--seg_path", type=str, default="C:\\Users\\as436\\OneDrive\\文件\\xfeat\\yolo\\yolo11n-seg.pt") # 分割模型路徑
    parser.add_argument("--max_kpts", type=int, default=500) # 特徵點數量
    parser.add_argument("--yolo_interval", type=int, default=3) # 每隔幾幀執行一次YOLO
    parser.add_argument("--yolo_imgsz", type=int, default=320) # YOLO輸入尺寸
    parser.add_argument("--geo_proj_step", type=int, default=2) # 每隔幾幀執行一次幾何投影 (計算3D位置)

    # 匹配參數 (不更改)
    parser.add_argument("--match_threshold", type=float, default=0.82)
    parser.add_argument("--min_match_count", type=int, default=25)
    
    # 深度參數 (不更改)
    parser.add_argument("--depth_input_width", type=int, default=256)
    parser.add_argument("--depth_input_height", type=int, default=256)
    parser.add_argument("--min_depth_m", type=float, default=0.1)
    parser.add_argument("--max_depth_m", type=float, default=50.0) # 8.0
    parser.add_argument("--baseline", type=float, default=0.1)

    # 相機內參
    # parser.add_argument("--fx", type=float, default=823.0409)
    # parser.add_argument("--fy", type=float, default=825.8716)
    # parser.add_argument("--cx", type=float, default=330.1168)
    # parser.add_argument("--cy", type=float, default=231.5212)
    parser.add_argument("--fx", type=float, default=718.856)
    parser.add_argument("--fy", type=float, default=718.856)
    parser.add_argument("--cx", type=float, default=607.1928)
    parser.add_argument("--cy", type=float, default=185.2157)
    parser.add_argument("--trajectory_output", type=str, default=None)

    return parser
