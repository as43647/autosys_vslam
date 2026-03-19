import os

import cv2
import numpy as np
import onnxruntime as ort


class DepthInference:
    def __init__(self, model_path, input_size=(256, 256)):
        self.input_size = input_size
        self.session = None
        self.input_name = None

        if not os.path.exists(model_path):
            print(f"[DepthInference] 找不到模型：{model_path}")
            return

        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[DepthInference] 已載入模型：{model_path}")
        print(f"[DepthInference] 啟用 providers: {self.session.get_providers()}")

    def process_disparity(self, color_bgr):
        if self.session is None:
            return None

        h, w = color_bgr.shape[:2]
        img_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, self.input_size).astype(np.float32) / 255.0
        blob = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]
        out = self.session.run(None, {self.input_name: blob})[0][0]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def disparity_to_depth_mm(disparity, focal_px, baseline_m):
        return (focal_px * baseline_m) / (disparity + 1e-6) * 1000.0

    def get_runtime_info(self):
        if self.session is None:
            return {
                "model_loaded": False,
                "requested_backend": "CUDAExecutionProvider -> CPUExecutionProvider",
                "active_providers": [],
                "using_gpu": False,
            }

        providers = self.session.get_providers()
        return {
            "model_loaded": True,
            "requested_backend": "CUDAExecutionProvider -> CPUExecutionProvider",
            "active_providers": providers,
            "using_gpu": "CUDAExecutionProvider" in providers,
        }
