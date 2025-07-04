import os
import shutil
import random
from pathlib import Path

# Константы
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_directory_structure(base_dir):
    """Создает структуру каталогов для нового датасета"""
    directories = [
        'dataset_gaz/images/train',
        'dataset_gaz/images/val',
        'dataset_gaz/images/test',
        'dataset_gaz/labels/train',
        'dataset_gaz/labels/val',
        'dataset_gaz/labels/test'
    ]
    for dir_path in directories:
        Path(os.path.join(base_dir, dir_path)).mkdir(parents=True, exist_ok=True)

def is_gas_meter(filename):
    """Проверяет, является ли файл изображением газового счетчика"""
    return 'closeup' in filename.lower()

def get_matching_files(source_dir):
    """Получает список файлов газовых счетчиков и их меток"""
    image_files = []
    for file in os.listdir(os.path.join(source_dir, 'train/images')):
        if file.endswith('.jpg') and is_gas_meter(file):
            image_name = os.path.splitext(file)[0]
            label_file = image_name + '.txt'
            if os.path.exists(os.path.join(source_dir, 'train/labels', label_file)):
                image_files.append(image_name)
    return image_files

def split_dataset(files):
    """Разделяет файлы на train, validation и test наборы"""
    random.shuffle(files)
    total = len(files)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    
    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]
    
    return train_files, val_files, test_files

def copy_files(source_dir, dest_dir, files, subset):
    """Копирует файлы в соответствующие директории"""
    for file in files:
        # Копирование изображений
        src_img = os.path.join(source_dir, 'train/images', file + '.jpg')
        dst_img = os.path.join(dest_dir, f'dataset_gaz/images/{subset}', file + '.jpg')
        shutil.copy2(src_img, dst_img)
        
        # Копирование меток
        src_label = os.path.join(source_dir, 'train/labels', file + '.txt')
        dst_label = os.path.join(dest_dir, f'dataset_gaz/labels/{subset}', file + '.txt')
        shutil.copy2(src_label, dst_label)

def create_data_yaml(base_dir):
    """Создает файл data.yaml для нового датасета"""
    yaml_content = """train: ../dataset_gaz/images/train
val: ../dataset_gaz/images/val
test: ../dataset_gaz/images/test

nc: 10
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

roboflow:
  workspace: ntd
  project: test-ppi3e
  version: 2
  license: CC BY 4.0
"""
    with open(os.path.join(base_dir, 'dataset_gaz/data.yaml'), 'w') as f:
        f.write(yaml_content)

def process_dataset():
    """Основная функция обработки датасета"""
    base_dir = os.getcwd()
    
    # Создание структуры каталогов
    create_directory_structure(base_dir)
    
    # Получение списка файлов газовых счетчиков
    files = get_matching_files(base_dir)
    
    # Разделение датасета
    train_files, val_files, test_files = split_dataset(files)
    
    # Копирование файлов
    copy_files(base_dir, base_dir, train_files, 'train')
    copy_files(base_dir, base_dir, val_files, 'val')
    copy_files(base_dir, base_dir, test_files, 'test')
    
    # Создание data.yaml
    create_data_yaml(base_dir)
    
    # Вывод статистики
    print(f"Всего файлов газовых счетчиков: {len(files)}")
    print(f"Тренировочный набор: {len(train_files)} файлов")
    print(f"Валидационный набор: {len(val_files)} файлов")
    print(f"Тестовый набор: {len(test_files)} файлов")

if __name__ == '__main__':
    process_dataset() 