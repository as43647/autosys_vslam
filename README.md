# AutoSys Hybrid SLAM

這個專案是一套以 RGB 影像為主、深度資訊為輔的 Hybrid SLAM 系統，整合了：

- YOLO segmentation：排除動態物體區域
- XFeat：特徵點與描述子提取
- PnP + RANSAC：相機位姿估計
- ONNX depth model / dataset depth / stereo depth：深度來源
- Open3D：即時軌跡視覺化

目前入口支援 3 種模式：

- `camera`：即時攝影機
- `tum`：TUM RGB-D dataset
- `kitti`：KITTI odometry sequence

## Project Structure

```text
.
|-- main.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- hybrid_slam/
|   |-- __init__.py
|   |-- cli.py
|   |-- factory.py
|   |-- core.py
|   |-- camera_pipeline.py
|   |-- tum_pipeline.py
|   |-- kitti_pipeline.py
|   |-- loaders.py
|   |-- depth.py
|   |-- features.py
|   `-- segment.py
|-- modules/
|   `-- xfeat.py
|-- camera_calibration/
|   |-- camera_calibration.py
|   `-- checkerboard.jpg
|-- depth_model/
|   `-- autosys_mdepth_optm.onnx
|-- yolo/
|   `-- yolo11n-seg.pt
|-- kitti/
|   |-- kitti_eval.py
|   |-- kitti_video.py
|   `-- pose/
|       `-- ...
|-- kitti_evaluation/
|   |-- evaluate_odometry.cpp
|   |-- evaluate_odometry.py
|   |-- matrix.cpp
|   |-- matrix.h
|   `-- mail.h
|-- tum_evaluation/
|   |-- associate.py
|   |-- ate.py
|   `-- rpe.py
|-- trajectory/
`-- video/
```

## Pipeline Overview

每張影像的主要流程如下：

1. 讀取 RGB 影像與對應深度來源
2. 用 YOLO segmentation 取得動態遮罩
3. 在靜態區域上抽取 XFeat 特徵
4. 使用前一個 keyframe 的 3D 點與當前 2D 點做 PnP
5. 更新相機姿態並輸出 trajectory 與影片

不同模式的深度來源：

- `camera`：ONNX depth model
- `tum --use_gt_depth`：TUM ground-truth depth
- `tum`：ONNX depth model
- `kitti --kitti_depth_source model`：ONNX depth model
- `kitti --kitti_depth_source stereo`：StereoSGBM 雙目深度

## Environment Setup

建議使用虛擬環境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果要使用 GPU：

- YOLO segmentation 目前要求 `PyTorch CUDA` 可用
- ONNX Runtime 會優先嘗試 `CUDAExecutionProvider`
- `requirements.txt` 內也有註解提醒如何安裝 CUDA 版 PyTorch

`FeatureExtractor` 會優先使用本地的 `modules.xfeat`；若匯入失敗，才退回 `torch.hub` 載入 XFeat。

## Required Files

請先確認以下模型檔存在：

- `depth_model/autosys_mdepth_optm.onnx`
- `yolo/yolo11n-seg.pt`

資料需求：

- `tum`：資料夾內需要 `associations.txt`
- `kitti`：sequence 資料夾內需要 `calib.txt` 和左影像資料夾
- `kitti --kitti_depth_source stereo`：還需要右影像資料夾，預設為 `image_3`

## Run

入口：

```powershell
python .\main.py
```

### Camera Mode

```powershell
python .\main.py --input_mode camera
```

指定解析度：

```powershell
python .\main.py --input_mode camera --cam_width 640 --cam_height 480
```

指定相機內參：

```powershell
python .\main.py --input_mode camera --fx 823.0409 --fy 825.8716 --cx 330.1168 --cy 231.5212
```

### TUM Mode

使用 ground-truth depth：

```powershell
python .\main.py --input_mode tum --dataset_path D:\dataset\tum\rgbd_dataset_freiburg3_walking_xyz --use_gt_depth
```

使用模型深度：

```powershell
python .\main.py --input_mode tum --dataset_path D:\dataset\tum\rgbd_dataset_freiburg3_walking_xyz
```

### KITTI Mode

使用模型深度：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --kitti_depth_source model
```

使用雙目深度：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --kitti_depth_source stereo
```

帶入 ground-truth pose：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --kitti_pose_path D:\dataset\kitti\poses\00.txt
```

指定輸出：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --trajectory_output .\trajectory\00_model.txt --video_output .\video\00_model.mp4
```

## Important CLI Arguments

### Input

- `--input_mode`：`camera`、`tum`、`kitti`
- `--dataset_path`：TUM 或 KITTI 資料路徑
- `--use_gt_depth`：TUM 模式下使用 ground-truth depth

### KITTI

- `--kitti_image_folder`：左影像資料夾，預設 `image_2`
- `--kitti_right_image_folder`：右影像資料夾，預設 `image_3`
- `--kitti_pose_path`：ground-truth pose 檔案
- `--kitti_depth_source`：`model` 或 `stereo`

### Output

- `--trajectory_output`
- `--video_output`

### Camera

- `--camera_id`
- `--cam_width`
- `--cam_height`
- `--fx`
- `--fy`
- `--cx`
- `--cy`

### YOLO / Segmentation

- `--seg_path`
- `--yolo_interval`
- `--yolo_imgsz`
- `--geo_proj_step`
- `--mask_persistence`
- `--mask_dilate_kernel`

### Feature Matching

- `--max_kpts`
- `--match_threshold`
- `--min_match_count`

### Depth

- `--model_path`
- `--depth_input_width`
- `--depth_input_height`
- `--min_depth_m`
- `--max_depth_m`
- `--baseline`

## Recommended Settings

平衡速度與穩定性：

```powershell
python .\main.py --input_mode camera --yolo_interval 2 --geo_proj_step 1
```

偏向即時性：

```powershell
python .\main.py --input_mode camera --yolo_interval 3 --geo_proj_step 2 --yolo_imgsz 320
```

偏向低負載：

```powershell
python .\main.py --input_mode camera --yolo_interval 4 --yolo_imgsz 256 --depth_input_width 192 --depth_input_height 192
```

## Runtime Output

啟動時會印出 `RUNTIME DEVICE SUMMARY`，包含：

- `Input mode`
- `PyTorch CUDA`
- `XFeat model device`
- `YOLO model device`
- `ONNX providers`
- `Depth using GPU`
- `YOLO imgsz`
- `YOLO interval`

結束後會輸出效能摘要：

- 平均 FPS
- mean / std / P95 latency
- Mask / Feature / Pose 各階段耗時比例

## Output Files

常見輸出：

- `trajectory/*.txt`：軌跡結果
- `video/*.mp4`：執行影片
- `kitti/pose/*.txt`：KITTI pose 結果
- `kitti/pose/eval_output/`：KITTI 評估輸出圖表與統計

CLI 預設輸出檔名：

- `trajectory.txt`
- `trajectory.mp4`

## Evaluation Tools

### TUM Evaluation

位於 `tum_evaluation/`：

- `associate.py`
- `ate.py`
- `rpe.py`

### KITTI Evaluation

位於 `kitti_evaluation/`：

- `evaluate_odometry.cpp`
- `evaluate_odometry.py`

另外 `kitti/` 內也保留：

- `kitti_eval.py`
- `kitti_video.py`

## Camera Calibration

可使用內建校正工具：

```powershell
python .\camera_calibration\camera_calibration.py
```

完成後將得到的 `fx`、`fy`、`cx`、`cy` 帶回執行參數即可。

## Notes

- `TUMDatasetLoader` 目前以 `utf-16` 讀取 `associations.txt`，若資料檔編碼不同可能需要調整
- `YoloMaskTracker` 目前要求 CUDA；若沒有可用的 PyTorch CUDA，程式會直接停止
- `kitti_pose_path` 目前由 loader 載入，但主流程沒有直接做誤差比對
- `kitti/` 現在同時包含工具腳本與輸出資料；若之後持續整理，建議把 script 與 output 再切得更乾淨

## Git Ignore Suggestions

目前專案常見需要忽略的內容：

- `autosys/`
- `.venv/`
- `__pycache__/`
- `video/`
- `trajectory/`
- `kitti/`

如果你未來要把 `kitti/` 裡的腳本保留在版控，但忽略輸出，建議把資料與工具分開放置。
