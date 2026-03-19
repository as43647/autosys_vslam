import os
import time

import cv2


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
