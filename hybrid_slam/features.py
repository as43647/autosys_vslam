import cv2
import torch


class FeatureExtractor:
    def __init__(self, max_kpts=500, match_threshold=0.82):
        self.match_threshold = match_threshold
        try:
            from modules.xfeat import XFeat

            self.model = XFeat(top_k=max_kpts)
            print("[XFeat] 使用本地 modules.xfeat")
        except Exception:
            self.model = torch.hub.load(
                "verlab/accelerated_features",
                "XFeat",
                pretrained=True,
                top_k=max_kpts,
                trust_repo=True,
            )
            print("[XFeat] 使用 torch.hub 載入")

    def extract(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float()[None]
        if torch.cuda.is_available():
            x_tensor = x_tensor.cuda()
        return self.model.detectAndCompute(x_tensor)[0]

    def match(self, desc1, desc2):
        idx0, idx1 = self.model.match(desc1, desc2, self.match_threshold)
        return idx0.cpu().numpy(), idx1.cpu().numpy()

    def get_runtime_info(self):
        model_device = "unknown"
        try:
            model_device = str(next(self.model.parameters()).device)
        except Exception:
            pass

        return {
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_device_count": torch.cuda.device_count(),
            "torch_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_device": model_device,
        }
