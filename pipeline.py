# backend/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import base64
import io
import tempfile
from uuid import uuid4
from hashlib import md5

import cv2
import numpy as np
from PIL import Image

from .detector import TabDetector

# ---------- Tunables (safe defaults) ----------
INTERVAL_SECONDS = 2.0              # Extract screenshot every N seconds
# Page layout
PAGE_SIZE       = (2480, 3508)      # A4 @ ~300 dpi (W,H)
MARGIN          = 48
GAP             = 16
LINE_HEIGHT     = 180
# ---------------------------------------------

def extract_screenshots(video_path: str, session_id: str) -> Tuple[List[Dict], str]:
    """
    Extract screenshots from video at regular time intervals.
    Returns (screenshots_metadata, crops_dir) where:
    - screenshots_metadata: List of dicts with 'image' (base64), 'timestamp', 'index', 'crop_path'
    - crops_dir: Directory where crop images are saved
    """
    detector = TabDetector(weights_path="backend/models/best.pt", conf=0.25)
    screenshots: List[Dict] = []
    
    # Create temp directory for crops
    workdir = Path("backend/outputs") / "temp_crops" / session_id
    workdir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return screenshots, str(workdir)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    
    frame_interval = int(round(fps * INTERVAL_SECONDS))
    frame_count = 0
    screenshot_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract screenshot at regular intervals
        if frame_count % frame_interval == 0:
            box = detector.detect_box(frame)
            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                crop = frame[y1:y2, x1:x2].copy()
                
                if crop.size > 0:
                    # Save crop to file
                    crop_path = workdir / f"crop_{screenshot_index}.png"
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_crop)
                    pil_image.save(crop_path)
                    
                    # Convert to base64 for frontend display
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    timestamp = frame_count / fps
                    screenshots.append({
                        'index': screenshot_index,
                        'image': img_base64,
                        'timestamp': round(timestamp, 2),
                        'crop_path': str(crop_path)
                    })
                    screenshot_index += 1
        
        frame_count += 1
    
    cap.release()
    return screenshots, str(workdir)


def create_pdf_from_selected(screenshots: List[Dict], selected_indices: List[int], crops_dir: str) -> str:
    """
    Create PDF from selected screenshots.
    screenshots: List of screenshot dicts (must have 'crop_path' key)
    selected_indices: List of indices to include in PDF
    crops_dir: Directory where crop images are stored
    """
    workdir = Path("backend/outputs")
    workdir.mkdir(parents=True, exist_ok=True)
    
    # Load selected crops
    selected_crops = []
    for idx in sorted(selected_indices):
        if 0 <= idx < len(screenshots):
            crop_path = screenshots[idx].get('crop_path')
            if crop_path and Path(crop_path).exists():
                # Load image
                img = cv2.imread(crop_path)
                if img is not None:
                    selected_crops.append(img)
    
    if not selected_crops:
        # Create empty PDF
        pages = [_blank_page(*PAGE_SIZE)]
    else:
        pages = _layout_pages(selected_crops, PAGE_SIZE, MARGIN, GAP, LINE_HEIGHT)
    
    indices_str = ','.join(map(str, sorted(selected_indices)))
    hash_str = md5(indices_str.encode()).hexdigest()[:8]
    out_pdf = str(workdir / f"selected_tabs_{hash_str}.pdf")
    _save_pages_as_pdf(pages, out_pdf)
    return out_pdf


def _layout_pages(crops: List[np.ndarray],
                  page_size: Tuple[int, int],
                  margin: int,
                  gap: int,
                  line_h: int) -> List[np.ndarray]:
    page_w, page_h = page_size
    content_w = page_w - 2 * margin

    pages: List[np.ndarray] = []
    y = margin
    page = _blank_page(page_w, page_h)

    for crop in crops:
        h, w = crop.shape[:2]
        if w <= 0 or h <= 0:
            continue
        
        # Preserve aspect ratio: scale to fit content width
        scale = content_w / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))  # Preserve aspect ratio instead of fixed height
        
        # Resize with aspect ratio preserved
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Check if image fits on current page
        if y + new_h + margin > page_h:
            pages.append(page)
            page = _blank_page(page_w, page_h)
            y = margin

        # Center the image horizontally
        x = margin + (content_w - new_w) // 2
        page[y:y + new_h, x:x + new_w] = resized
        y += new_h + gap

    pages.append(page)
    return pages


def _blank_page(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _save_pages_as_pdf(pages: List[np.ndarray], out_pdf: str):
    pil_pages = []
    for p in pages:
        im = Image.fromarray(p[:, :, ::-1])  # BGR->RGB
        if im.mode != "RGB":
            im = im.convert("RGB")
        pil_pages.append(im)
    pil_pages[0].save(out_pdf, save_all=True, append_images=pil_pages[1:])
