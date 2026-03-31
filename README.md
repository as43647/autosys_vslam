# AutoSys Hybrid SLAM

這個專案實作一套以 RGB 為主、深度為輔的 Hybrid SLAM 流程，整合了：

- YOLO segmentation，用來排除動態物體區域
- XFeat 特徵點與描述子
- PnP + RANSAC 相機位姿估計
- ONNX 深度模型，或資料集提供的深度 / 雙目深度
- Open3D 即時軌跡視覺化

目前程式入口支援 3 種模式：

- `camera`: 即時攝影機輸入
- `tum`: TUM RGB-D dataset
- `kitti`: KITTI odometry sequence

## Project Structure

```text
.
|-- main.py
|-- README.md
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
|-- depth_model/
|   `-- autosys_mdepth_optm.onnx
|-- yolo/
|   `-- yolo11n-seg.pt
|-- camera_calibration/
|   |-- camera_calibration.py
|   `-- checkerboard.jpg
|-- kitti/
|   |-- kitti_eval.py
|   `-- kitti_video.py
|-- tum/
|   |-- associate.py
|   |-- ate.py
|   `-- rpe.py
|-- cpp/
|   |-- evaluate_odometry.cpp
|   `-- evaluate_odometry.py
|-- trajectory/
`-- video/
```

## How It Works

每張影像大致會經過以下流程：

1. 取得 RGB 畫面，以及對應深度來源
2. 用 YOLO segmentation 找出動態物體遮罩
3. 在靜態區域上抽取 XFeat 特徵
4. 以前一個 keyframe 的 3D 點和目前 2D 點做 PnP
5. 更新相機軌跡，輸出影片與 trajectory 檔案

不同模式的深度來源如下：

- `camera`: 使用 ONNX 深度模型推論 disparity，再換算成 depth
- `tum --use_gt_depth`: 直接使用資料集 ground-truth depth
- `tum`: 使用 ONNX 深度模型
- `kitti --kitti_depth_source model`: 使用 ONNX 深度模型
- `kitti --kitti_depth_source stereo`: 使用左右影像做 StereoSGBM 深度

## Environment Setup

目前倉庫已提供 `requirements.txt`，建議直接建立虛擬環境後安裝。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果你要使用 GPU：

- YOLO segmentation 目前要求 `PyTorch CUDA` 可用，否則程式會在啟動時直接拋錯
- ONNX depth 會優先嘗試 `CUDAExecutionProvider`，失敗時才 fallback 到 CPU
- `requirements.txt` 內有註解提醒：若要使用 GPU，建議另外安裝對應版本的 CUDA-enabled PyTorch wheel

`FeatureExtractor` 會優先使用本地的 `modules.xfeat`，若匯入失敗，才退回 `torch.hub` 載入 XFeat。第一次走 `torch.hub` 時可能需要網路。

## Required Files

請先確認以下模型檔存在：

- `depth_model/autosys_mdepth_optm.onnx`
- `yolo/yolo11n-seg.pt`

另外，不同模式還需要對應資料：

- `tum`: 資料夾內需要 `associations.txt`
- `kitti`: sequence 資料夾內需要 `calib.txt` 與左影像資料夾
- `kitti --kitti_depth_source stereo`: 還需要右影像資料夾，預設為 `image_3`

## Run

專案入口是：

```powershell
python .\main.py
```

### 1. Camera Mode

最基本執行方式：

```powershell
python .\main.py --input_mode camera
```

指定攝影機解析度：

```powershell
python .\main.py --input_mode camera --cam_width 640 --cam_height 480
```

指定相機內參：

```powershell
python .\main.py --input_mode camera --fx 823.0409 --fy 825.8716 --cx 330.1168 --cy 231.5212
```

### 2. TUM Mode

使用 ground-truth depth：

```powershell
python .\main.py --input_mode tum --dataset_path D:\dataset\tum\rgbd_dataset_freiburg3_walking_xyz --use_gt_depth
```

使用模型推論深度：

```powershell
python .\main.py --input_mode tum --dataset_path D:\dataset\tum\rgbd_dataset_freiburg3_walking_xyz
```

### 3. KITTI Mode

使用模型推論深度：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --kitti_depth_source model
```

使用雙目深度：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --kitti_depth_source stereo
```

如果你有 ground-truth pose 檔，也可以一起帶入：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --kitti_pose_path D:\dataset\kitti\poses\00.txt
```

## Important CLI Arguments

`hybrid_slam/cli.py` 目前主要參數如下。

### Input

- `--input_mode`: `camera`、`tum`、`kitti`
- `--dataset_path`: TUM 或 KITTI sequence 路徑
- `--use_gt_depth`: TUM 模式下使用 ground-truth depth

### KITTI

- `--kitti_image_folder`: 左影像資料夾，預設 `image_2`
- `--kitti_right_image_folder`: 右影像資料夾，預設 `image_3`
- `--kitti_pose_path`: ground-truth pose 檔案
- `--kitti_depth_source`: `model` 或 `stereo`

### Output

- `--trajectory_output`: 軌跡輸出檔名，預設 `trajectory.txt`
- `--video_output`: 輸出影片檔名，預設 `trajectory.mp4`

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

啟動後，終端機會先印出 `RUNTIME DEVICE SUMMARY`，包含：

- `Input mode`
- `PyTorch CUDA`
- `XFeat model device`
- `YOLO model device`
- `ONNX providers`
- `Depth using GPU`
- `YOLO imgsz`
- `YOLO interval`

流程結束後還會印出：

- 平均 FPS
- mean / std / P95 latency
- Mask / Feature / Pose 三部分耗時占比

## Output Files

程式目前會輸出：

- 軌跡文字檔，預設 `trajectory.txt`
- 結果影片，預設 `trajectory.mp4`

若你沒有傳入輸出參數，`core.py` 內也保留了舊的 fallback 名稱：

- trajectory fallback: `camera_trajectories_walking_testH.txt`
- video fallback: `testH.mp4`

建議直接明確指定：

```powershell
python .\main.py --input_mode kitti --dataset_path D:\dataset\kitti\sequences\00 --trajectory_output .\trajectory\00_model.txt --video_output .\video\00_model.mp4
```

## Evaluation Helpers

倉庫內另外附了幾個評估工具：

- `tum/ate.py`: 計算 Absolute Trajectory Error
- `tum/rpe.py`: 計算 Relative Pose Error
- `kitti/kitti_eval.py`: KITTI 相關評估輔助
- `cpp/evaluate_odometry.py`: odometry 評估腳本

如果要做 KITTI 影片整理，也可以使用：

- `kitti/kitti_video.py`

## Camera Calibration

若你要先校正相機內參，可以執行：

```powershell
python .\camera_calibration\camera_calibration.py
```

完成後把得到的 `fx`, `fy`, `cx`, `cy` 帶回 `main.py` 執行參數即可。

## Notes And Known Issues

- `README` 這次已依照目前程式碼更新，但部分程式內的 `print` 訊息仍有編碼異常，不影響核心流程
- `TUMDatasetLoader` 目前用 `utf-16` 讀取 `associations.txt`，如果你的檔案實際不是這個編碼，會讀取失敗
- `YoloMaskTracker` 目前硬性要求 CUDA；即使其他模組可在 CPU 跑，YOLO 這段仍會中止
- `trajectory_output` 與 `video_output` 在 CLI 裡有預設值，因此大多數情況下會直接輸出成 `trajectory.txt` 與 `trajectory.mp4`
- `kitti_pose_path` 目前會被 loader 載入，但主流程沒有直接拿來做誤差計算

## Suggested .gitignore Items

目前 `.gitignore` 已忽略：

- `__pycache__/`
- `xfeat/`
- `video/`
- `trajectory/`

若之後要整理版本庫，通常也建議忽略：

- 大型模型檔，如 `.onnx`、`.pt`
- 執行輸出影片，如 `.mp4`
- 臨時軌跡結果，如 `.txt`
- 本地虛擬環境，如 `.venv/` 或 `autosys/`
