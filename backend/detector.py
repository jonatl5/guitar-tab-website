# backend/detector.py
from ultralytics import YOLO
import numpy as np

class TabDetector:
    def __init__(self, weights_path="backend/models/best.pt", conf=0.25):
        self.model = YOLO(weights_path)
        self.conf = conf
        self.prev_box = None

    def detect_box(self, bgr_image: np.ndarray):
        """
        Returns [x1, y1, x2, y2] or None.
        """
        res = self.model.predict(bgr_image, conf=self.conf, verbose=False)[0]
        if len(res.boxes) == 0:
            return self.prev_box
        # take highest-confidence box (single-class problem)
        i = int(res.boxes.conf.argmax().item())
        xyxy = res.boxes.xyxy.cpu().numpy()[i]
        self.prev_box = xyxy
        return xyxy
        # Ultralytics Python usage & predict mode docs:  # :contentReference[oaicite:4]{index=4}
