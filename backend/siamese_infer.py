from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .siamese_model import IMAGE_SIZE, SiameseCNN

DEFAULT_MODEL_PATH = Path("backend/models/siamese_cnn.pt")
DEFAULT_THRESHOLD = 0.75


class SiameseComparator:
    """Loads a trained Siamese model and compares tab crops."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_MODEL_PATH,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Siamese checkpoint not found at {checkpoint_path}. "
                "Train the model with backend/training/train_siamese.py first."
            )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        embedding_dim = int(checkpoint.get("embedding_dim", 128))

        self.model = SiameseCNN(embedding_dim=embedding_dim).to(self.device)
        state = checkpoint.get("model_state_dict") or checkpoint.get("state")
        if state is None:
            raise KeyError("Checkpoint does not contain model weights.")
        self.model.load_state_dict(state)
        self.model.eval()

        ckpt_threshold = checkpoint.get("threshold", checkpoint.get("thr", DEFAULT_THRESHOLD))
        self.threshold = float(threshold) if threshold is not None else float(ckpt_threshold)

    def _preprocess(self, bgr_image: np.ndarray) -> torch.Tensor:
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input crop is empty.")

        cleaned = self._remove_indicator_overlay(bgr_image)
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray,
            (IMAGE_SIZE[1], IMAGE_SIZE[0]),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(resized).float().div(255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _remove_indicator_overlay(bgr_image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Bright yellow indicator range
        yellow_lower = np.array([15, 60, 120], dtype=np.uint8)
        yellow_upper = np.array([45, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Bright blue indicator range
        blue_lower = np.array([85, 50, 80], dtype=np.uint8)
        blue_upper = np.array([135, 255, 255], dtype=np.uint8)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        indicator_mask = cv2.bitwise_or(yellow_mask, blue_mask)

        # Remove isolated noise, then slightly expand to cover indicator edges.
        kernel = np.ones((3, 3), np.uint8)
        indicator_mask = cv2.morphologyEx(indicator_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        indicator_mask = cv2.dilate(indicator_mask, kernel, iterations=1)

        return cv2.inpaint(bgr_image, indicator_mask, 3, cv2.INPAINT_TELEA)

    def embed(self, bgr_image: np.ndarray) -> torch.Tensor:
        x = self._preprocess(bgr_image)
        with torch.no_grad():
            emb = self.model(x)
        return emb.squeeze(0).cpu()

    @staticmethod
    def distance(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
        return torch.norm(embedding_a - embedding_b, p=2).item()

    def is_match(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> bool:
        limit = self.threshold if threshold is None else float(threshold)
        return self.distance(embedding_a, embedding_b) <= limit
