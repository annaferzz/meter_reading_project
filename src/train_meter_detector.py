from ultralytics import YOLO
from src.config import YOLO_CONFIG, DATASET_YAML_PATH, RUNS_DIR

def train_meter_detector():
    model = YOLO(YOLO_CONFIG["model_name"])
    results = model.train(
        data=str(DATASET_YAML_PATH),
        epochs=100,
        imgsz=YOLO_CONFIG["img_size"],
        batch=16,
        device='cpu',
        project=str(RUNS_DIR),
        name='train_meter'
    )
    print("Обучение завершено. Веса сохранены в:", results.save_dir)

if __name__ == "__main__":
    train_meter_detector()
