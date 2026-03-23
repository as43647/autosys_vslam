import os
import time

import cv2
import numpy as np


class TUMDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.assoc_txt = os.path.join(dataset_path, "associations.txt")
        self.image_list = []
        self.depth_list = []
        self.timestamps = []

        if not os.path.exists(self.assoc_txt):
            raise FileNotFoundError(f"請先生成 associations.txt 並放在 {dataset_path}")

        with open(self.assoc_txt, "r", encoding="utf-16") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    self.timestamps.append(parts[0])
                    self.image_list.append(os.path.join(dataset_path, parts[1]))
                    self.depth_list.append(os.path.join(dataset_path, parts[3]))

        print(f"[Dataset] 加載了 {len(self.image_list)} 幀")

    def __len__(self):
        return len(self.image_list)

    def get_frame(self, idx):
        if idx >= len(self.image_list):
            return None, None, None

        img = cv2.imread(self.image_list[idx])
        depth_raw = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED)
        if depth_raw is not None and depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]

        return img, depth_raw, self.timestamps[idx]


class CameraLoader:
    def __init__(self, camera_id=0, width=None, height=None):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟相機 camera_id={camera_id}")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print(f"[Camera] 相機已開啟，camera_id={camera_id}")

    def get_frame(self, idx):
        del idx
        ret, img = self.cap.read()
        if not ret or img is None:
            return None, None, None

        return img, None, str(time.time())

    def release(self):
        if self.cap is not None:
            self.cap.release()


class KITTILoader:
    def __init__(self, sequence_path, image_folder="image_2", pose_path=None):
        self.sequence_path = sequence_path
        self.image_dir = os.path.join(sequence_path, image_folder)
        self.calib_path = os.path.join(sequence_path, "calib.txt")
        self.pose_path = pose_path
        self.image_list = []
        self.timestamps = []
        self.gt_poses = []
        self.K = None

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"找不到 KITTI 影像資料夾: {self.image_dir}")
        if not os.path.exists(self.calib_path):
            raise FileNotFoundError(f"找不到 KITTI calib.txt: {self.calib_path}")

        self._load_images()
        self._load_intrinsics()
        self._load_gt_poses()

        print(f"[KITTI] 加載了 {len(self.image_list)} 幀，image_folder={image_folder}")

    def _load_images(self):
        valid_exts = {".png", ".jpg", ".jpeg"}
        for name in sorted(os.listdir(self.image_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in valid_exts:
                continue
            self.image_list.append(os.path.join(self.image_dir, name))
            self.timestamps.append(os.path.splitext(name)[0])

    def _load_intrinsics(self):
        with open(self.calib_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, values = line.split(":", 1)
                key = key.strip()
                if key not in {"P0", "P1", "P2", "P3"}:
                    continue

                proj = np.fromstring(values, sep=" ", dtype=np.float64)
                if proj.size != 12:
                    continue

                if key == "P2":
                    self.K = proj.reshape(3, 4)[:, :3]
                    return

        raise ValueError(f"無法從 {self.calib_path} 解析 KITTI 相機內參 P2")

    def _load_gt_poses(self):
        if not self.pose_path or not os.path.exists(self.pose_path):
            return

        with open(self.pose_path, "r", encoding="utf-8") as f:
            for line in f:
                vals = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
                if vals.size == 12:
                    pose = np.eye(4, dtype=np.float64)
                    pose[:3, :4] = vals.reshape(3, 4)
                    self.gt_poses.append(pose)

    def __len__(self):
        return len(self.image_list)

    def get_frame(self, idx):
        if idx >= len(self.image_list):
            return None, None, None

        img = cv2.imread(self.image_list[idx])
        return img, None, self.timestamps[idx]
