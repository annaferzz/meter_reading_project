import os
from pathlib import Path

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "weights"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Дополнительные пути и файлы проекта
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "runs" / "detect" / "train_meter" / "weights" / "best.pt"
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "dataset_gaz" / "images" / "test" / "test1.jpg"
DATASET_YAML_PATH = PROJECT_ROOT / "dataset_gaz" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

# Временная директория для промежуточных кропов / артефактов
TEMP_DIR = PROJECT_ROOT / "data" / "tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Список классов, описывающих область показаний счётчика
TARGET_CLASSES = [
    "indications", "digits", "numbers", "reading", "value"
]

# Конфигурация YOLO
YOLO_CONFIG = {
    "model_name": "yolo11s.pt",
    "img_size": 640,
    "conf_threshold": 0.5,
    "iou_threshold": 0.45,
    "classes": ["gas_meter"]
}

# Конфигурация OCR
OCR_CONFIG = {
    "tesseract_cmd": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "tesseract_config": "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789."
}

# Целевые метрики
TARGET_METRICS = {
    "detection_map": 0.85,
    "ocr_accuracy": 0.95,
    "processing_time": 2.0
}
