# AutoSys Hybrid SLAM

這個專案是一套結合下列模組的 RGB-D / Monocular Hybrid SLAM 測試系統：

- YOLO segmentation：剔除人物等動態目標
- 幾何投影遮罩更新：在非 YOLO 幀補動態遮罩
- XFeat：特徵點提取與匹配
- PnP：估計相機相對位姿
- ONNX 深度模型或 TUM ground-truth depth：提供深度資訊

目前支援兩種輸入模式：

- `camera`：讀取即時相機
- `tum`：讀取 TUM RGB-D dataset

## Project Structure

```text
.
├─ main.py
├─ requirements.txt
├─ hybrid_slam/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ factory.py
│  ├─ core.py
│  ├─ camera_pipeline.py
│  ├─ tum_pipeline.py
│  ├─ loaders.py
│  ├─ depth.py
│  ├─ features.py
│  └─ segment.py
├─ camera_calibration/
│  ├─ camera_calibration.py
│  └─ checkerboard.jpg
├─ depth_model/
│  └─ autosys_mdepth_optm.onnx
├─ yolo/
│  └─ yolo11n-seg.pt
└─ modules/
   └─ xfeat.py
```

## Module Overview

- `main.py`
  - 程式入口，解析 CLI 並啟動對應 pipeline
- `hybrid_slam/cli.py`
  - 所有命令列參數
- `hybrid_slam/factory.py`
  - 根據 `input_mode` 建立 `camera` 或 `tum` pipeline
- `hybrid_slam/core.py`
  - 共用 SLAM 主流程
- `hybrid_slam/camera_pipeline.py`
  - 相機模式專用流程
- `hybrid_slam/tum_pipeline.py`
  - TUM 資料集模式專用流程
- `hybrid_slam/loaders.py`
  - 相機與資料集讀取
- `hybrid_slam/depth.py`
  - 深度模型載入與推論
- `hybrid_slam/features.py`
  - XFeat 特徵提取與匹配
- `hybrid_slam/segment.py`
  - YOLO 與幾何投影遮罩

## Environment Setup

建議使用虛擬環境。

```powershell
python -m venv xfeat
.\xfeat\Scripts\activate
pip install -r requirements.txt
```

## Required Models

請確認下列模型檔已存在：

- `depth_model/autosys_mdepth_optm.onnx`
- `yolo/yolo11n-seg.pt`

如果你有修改預設路徑，請同步更新 `hybrid_slam/cli.py` 或在執行時用參數指定。

## How To Run

### 1. Camera Mode

直接使用相機：

```powershell
python .\main.py --input_mode camera
```

使用指定解析度：

```powershell
python .\main.py --input_mode camera --cam_width 640 --cam_height 480
```

使用校正後的相機內參：

```powershell
python .\main.py --input_mode camera --fx 823.0409 --fy 825.8716 --cx 330.1168 --cy 231.5212
```

### 2. TUM Mode

使用 ground-truth depth：

```powershell
python .\main.py --input_mode tum --dataset_path .\rgbd_dataset_freiburg3_walking_xyz --use_gt_depth
```

使用深度模型推論：

```powershell
python .\main.py --input_mode tum --dataset_path .\rgbd_dataset_freiburg3_walking_xyz
```

## Important CLI Arguments

### Input

- `--input_mode`
  - `camera` 或 `tum`
- `--dataset_path`
  - TUM 資料集路徑
- `--use_gt_depth`
  - TUM 模式下使用 ground-truth depth

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
  - 每幾幀跑一次 YOLO
- `--yolo_imgsz`
  - YOLO 輸入尺寸
- `--geo_proj_step`
  - 幾何投影取樣步長

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

### 比較穩定的 camera 模式起手式

```powershell
python .\main.py --input_mode camera --yolo_interval 2 --geo_proj_step 1
```

### 比較省算力的 camera 模式

```powershell
python .\main.py --input_mode camera --yolo_interval 3 --geo_proj_step 2 --yolo_imgsz 320
```

### 比較輕量的設定

```powershell
python .\main.py --input_mode camera --yolo_interval 4 --yolo_imgsz 256 --depth_input_width 192 --depth_input_height 192
```

## Camera Calibration

專案附有相機校正工具：

```powershell
python .\camera_calibration\camera_calibration.py
```

使用方式：

1. 將棋盤格放到鏡頭前
2. 按 `c` 擷取多張不同角度影像
3. 按 `q` 開始計算
4. 取得 `fx`, `fy`, `cx`, `cy` 後，填回 `main.py` 執行參數或 `hybrid_slam/cli.py`

## Runtime Output

啟動程式時會輸出 `RUNTIME DEVICE SUMMARY`，用來確認：

- PyTorch 是否偵測到 CUDA
- XFeat 是否跑在 GPU
- YOLO 是否跑在 GPU
- ONNX Runtime 是否啟用 `CUDAExecutionProvider`

如果看到：

```text
PyTorch CUDA        : False
YOLO model device   : cpu
XFeat model device  : cpu
ONNX providers      : ['CPUExecutionProvider']
```

代表目前整體大多仍在 CPU 執行。

## Output Files

程式執行後會輸出：

- `testH.mp4`
  - 執行畫面錄影
- `camera_trajectories_walking_testH.txt`
  - 軌跡紀錄

## GitHub Upload Suggestions

建議不要把下列內容直接上傳到 GitHub：

- 虛擬環境資料夾 `xfeat/`
- 模型檔 `.onnx`, `.pt`
- 輸出影片 `.mp4`
- 軌跡輸出 `.txt`
- `__pycache__/`

## Known Issues

- 如果 `PyTorch CUDA = False`，則 YOLO / XFeat 會退回 CPU，FPS 會很低
- 如果 ONNX Runtime 缺少 CUDA / cuDNN / MSVC 相依，深度模型會退回 CPU
- 當 `yolo_interval > 1` 時，中間幀依賴幾何投影，可能出現遮罩邊界漏遮

## Future Improvements

- 將輸出檔名改為 CLI 參數
- 支援將 YOLO 類別與膨脹 kernel 參數化
- 顯示更完整的除錯資訊與每模組耗時

