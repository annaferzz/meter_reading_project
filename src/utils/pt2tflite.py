import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO


def convert_pt_to_tflite(
    pt_model_path: str, 
    tflite_output_path: str,
    input_shape: Tuple[int, int, int] = (1, 3, 640, 640),
    optimize_for_size: bool = True
) -> bool:
    """
    Конвертирует PyTorch модель (.pt) в TensorFlow Lite формат
    
    Args:
        pt_model_path: Путь к PyTorch модели
        tflite_output_path: Путь для сохранения TFLite модели
        input_shape: Форма входных данных (batch_size, channels, height, width)
        optimize_for_size: Оптимизировать модель для размера
    
    Returns:
        bool: True если конвертация прошла успешно, False иначе
    """

    # Load the YOLO11 model
    model = YOLO(pt_model_path)

    # Export the model to TFLite format
    model.export(format="tflite", imgsz=400)  # creates 'yolo11n_float32.tflite'

    return True

def main():
    """Основная функция для запуска конвертации из командной строки"""
    parser = argparse.ArgumentParser(description='Конвертация PyTorch модели в TensorFlow Lite')
    parser.add_argument('pt_model_path', type=str, help='Путь к PyTorch модели (.pt)')
    parser.add_argument('tflite_output_path', type=str, help='Путь для сохранения TFLite модели')
    parser.add_argument('--input-shape', type=str, default='1,3,640,640', 
                       help='Форма входных данных (batch,channels,height,width)')
    parser.add_argument('--optimize', action='store_true', 
                       help='Оптимизировать модель для размера')
    parser.add_argument('--validate', action='store_true', 
                       help='Валидировать сконвертированную модель')
    
    args = parser.parse_args()
    
    # Проверяем существование входного файла
    if not os.path.exists(args.pt_model_path):
        print(f"Ошибка: Файл {args.pt_model_path} не найден")
        sys.exit(1)
    
    # Парсим форму входных данных
    try:
        input_shape = tuple(map(int, args.input_shape.split(',')))
    except ValueError:
        print("Ошибка: Неверный формат input_shape. Используйте: batch,channels,height,width")
        sys.exit(1)
    
    # Создаем директорию для выходного файла если нужно
    output_dir = os.path.dirname(args.tflite_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Конвертация модели:")
    print(f"  Входной файл: {args.pt_model_path}")
    print(f"  Выходной файл: {args.tflite_output_path}")
    print(f"  Форма входных данных: {input_shape}")
    print(f"  Оптимизация: {'Включена' if args.optimize else 'Отключена'}")
    
    # Выполняем конвертацию
    success = convert_pt_to_tflite(
        args.pt_model_path,
        args.tflite_output_path,
        input_shape,
        args.optimize
    )
    
    if success:
        print("Конвертация завершена успешно!")
    else:
        print("Конвертация не удалась!")
        sys.exit(1)


if __name__ == "__main__":
    main()