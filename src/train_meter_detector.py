from ultralytics import YOLO
from src.config import YOLO_CONFIG, DATASET_YAML_PATH, RUNS_DIR

def train_meter_detector():
    model = YOLO(YOLO_CONFIG["model_name"])
    results = model.train(
        data=str(DATASET_YAML_PATH),
        epochs=100,
        imgsz=YOLO_CONFIG["img_size"],
        batch=16,
        device=0,
        project=str(RUNS_DIR),
        name='train_meter',
        # Параметры оптимизации
        optimizer='AdamW',  # Явно указываем оптимизатор
        lr0=0.001667,      # Learning rate из auto-подбора
        momentum=0.9,      # Momentum из auto-подбора
        weight_decay=0.0005,
        
        # Аугментация данных
        hsv_h=0.015,  # Изменение оттенка
        hsv_s=0.7,    # Изменение насыщенности
        hsv_v=0.4,    # Изменение яркости
        degrees=10.0,  # Поворот изображения
        translate=0.1, # Сдвиг
        scale=0.5,    # Масштабирование
        fliplr=0.5,   # Горизонтальное отражение
        mosaic=1.0,   # Мозаика изображений
        mixup=0.1,    # Смешивание изображений
        
        # Параметры для предотвращения переобучения
        patience=50,   # Ранняя остановка
        save_period=10,# Сохранение каждые N эпох
        exist_ok=True, # Перезаписывать существующие эксперименты
        
        # Дополнительные параметры
        workers=8,     # Количество worker'ов для загрузки данных
        cache='disk',  # Кэширование на диск для детерминированных результатов
        close_mosaic=10, # Отключить мозаику за N эпох до конца
        warmup_epochs=3, # Количество эпох для разогрева
    )
    print("Обучение завершено. Веса сохранены в:", results.save_dir)

if __name__ == "__main__":
    train_meter_detector()
