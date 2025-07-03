import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from src.utils.logger import setup_logger
from src.config import YOLO_CONFIG, TEMP_DIR

logger = setup_logger(__name__)


class MeterDetector:
    def __init__(self, model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Loaded trained model: {model_path}")
        else:
            self.model = YOLO(YOLO_CONFIG["model_name"])
            logger.info("Loaded pretrained YOLOv11s")

    def detect_meters(self, image_path: str, conf_threshold: float = None):
        """Detect meters on image"""
        results = self.model(
            image_path,
            conf=conf_threshold or YOLO_CONFIG["conf_threshold"],
            device=self.device
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': cls,
                        'class_name': self.model.names[cls]
                    })

        return detections

    def crop_meter_regions(self, image_path: str, detections: list):
        """Crop detected meter regions for OCR"""
        image = cv2.imread(image_path)
        cropped_regions = []

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            cropped = image[y1:y2, x1:x2]

            crop_path = TEMP_DIR / f"temp_crop_{i}.jpg"
            cv2.imwrite(str(crop_path), cropped)

            cropped_regions.append({
                'image_path': str(crop_path),
                'bbox': detection['bbox'],
                'class': detection['class_name']
            })

        return cropped_regions