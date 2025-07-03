import sys
from pathlib import Path
import cv2
import pytesseract
import os

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detection.yolo_meter_detector import MeterDetector
from src.config import (
    OCR_CONFIG, TARGET_CLASSES, MODEL_WEIGHTS_PATH,
    SAMPLE_IMAGE_PATH, TEMP_DIR
)

# Настройки Tesseract
pytesseract.pytesseract.tesseract_cmd = OCR_CONFIG["tesseract_cmd"]


def recognize_digits(image_path, detector):
    image_path = str(image_path)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден!")

    # Детекция объектов
    detections = detector.detect_meters(image_path)

    print("\nРезультаты детекции:")
    for i, det in enumerate(detections):
        print(f"Объект {i + 1}: {det['class_name']} (conf: {det['confidence']:.2f})")

    # Поиск показаний счетчика
    indications = [d for d in detections if d['class_name'].lower() in TARGET_CLASSES]

    if not indications:
        # Визуализация для отладки
        img = cv2.imread(image_path)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{det['class_name']} {det['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        debug_path = TEMP_DIR / "debug_detection.jpg"
        cv2.imwrite(str(debug_path), img)
        print(f"Сохранен файл для отладки: {debug_path}")
        return "Показания не обнаружены"

    # Используем первый найденный элемент показаний
    det = indications[0]
    x1, y1, x2, y2 = det['bbox']

    # Кроп области счетчика
    img = cv2.imread(image_path)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(TEMP_DIR / "debug_crop.jpg"), crop)

    # Препроцессинг для OCR
    # 1. Преобразование в оттенки серого
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(TEMP_DIR / "debug_gray.jpg"), gray)
    
    # 2. Увеличение размера изображения
    scale_factor = 2
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 3. Увеличение контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 4. Удаление шума
    gray = cv2.medianBlur(gray, 3)
    
    # 5. Бинаризация (инвертированная, так как цифры темные на светлом фоне)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(str(TEMP_DIR / "debug_thresh.jpg"), thresh)
    
    # 6. Морфологические операции для улучшения цифр
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Удаляем шум
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)  # Соединяем части цифр
    cv2.imwrite(str(TEMP_DIR / "debug_cleaned.jpg"), cleaned)

    # 7. Добавляем отступы для лучшего распознавания
    border = 20
    cleaned = cv2.copyMakeBorder(cleaned, border, border, border, border, 
                                cv2.BORDER_CONSTANT, value=255)

    # Распознавание цифр с подробными параметрами
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789. -c tessedit_min_confidence=60'
    text = pytesseract.image_to_string(cleaned, config=custom_config)
    
    # Получаем также уверенность OCR
    confidence_data = pytesseract.image_to_data(cleaned, config=custom_config, output_type=pytesseract.Output.DICT)
    
    print("\nОтладка OCR:")
    if confidence_data['text']:
        for i, word in enumerate(confidence_data['text']):
            if word.strip():
                conf = confidence_data['conf'][i]
                print(f"Распознано: '{word}', уверенность: {conf}%")
    else:
        print("OCR не смог распознать текст. Проверьте отладочные изображения в папке tmp/")

    # Очистка результата
    text = ''.join(c for c in text if c.isdigit() or c == '.')
    
    return {
        'type': det['class_name'],
        'value': text.strip(),
        'confidence': det['confidence'],
        'ocr_debug': confidence_data if confidence_data['text'] else None
    }


if __name__ == "__main__":
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"ОШИБКА: Файл весов не найден: {MODEL_WEIGHTS_PATH}")
        print("Проверьте путь и перезапустите обучение!")
        exit(1)
    detector = MeterDetector(MODEL_WEIGHTS_PATH)
    print("Классы модели:", detector.model.names)
    reading = recognize_digits(SAMPLE_IMAGE_PATH, detector)
    print("Результат:", reading)
