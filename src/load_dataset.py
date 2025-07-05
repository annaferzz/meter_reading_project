import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import random

dataset_url = 'https://universe.roboflow.com/ds/HJ1zhypP3Q?key=o5LXfvCbPV'

def main():
    dataset_dir = 'dataset_gaz'
    zip_path = 'dataset.zip'
    # Скачивание архива, если его нет
    if not os.path.exists(zip_path):
        print(f"Скачивание датасета из {dataset_url}...")
        with requests.get(dataset_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc=zip_path,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print("Архив скачан.")
    else:
        print(f"Файл {zip_path} уже существует, скачивание пропущено.")
    
    # Удаляем старую папку dataset, если есть
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Распаковка архива
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Архив распакован.")
    # Удаление файлов без closeup в имени
    for subfolder in ['images', 'labels']:
        target_dir = os.path.join(dataset_dir, 'train', subfolder)
        if os.path.exists(target_dir):
            for fname in os.listdir(target_dir):
                if 'closeup' not in fname:
                    os.remove(os.path.join(target_dir, fname))
    print("Удаление файлов не содержащих 'closeup' завершено.")

    # Разделение на train/val/test
    split_cfg = {'train': 0.7, 'val': 0.2, 'test': 0.1}  # можно менять проценты
    shutil.move(os.path.join(dataset_dir, 'train'), os.path.join(dataset_dir, 'dataset'))
    images_dir = os.path.join(dataset_dir, 'dataset', 'images')
    labels_dir = os.path.join(dataset_dir, 'dataset', 'labels')
    images = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
    random.shuffle(images)
    n = len(images)
    n_train = int(n * split_cfg['train'])
    n_val = int(n * split_cfg['val'])
    n_test = n - n_train - n_val
    splits = (
        ('train', images[:n_train]),
        ('val', images[n_train:n_train+n_val]),
        ('test', images[n_train+n_val:]),
    )
    for split, split_files in splits:
        split_img_dir = os.path.join(dataset_dir, 'images', split)
        split_lbl_dir = os.path.join(dataset_dir, 'labels', split)
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        for fname in split_files:
            shutil.move(os.path.join(images_dir, fname), os.path.join(split_img_dir, fname))
            label_name = os.path.splitext(fname)[0] + '.txt'
            if os.path.exists(os.path.join(labels_dir, label_name)):
                shutil.move(os.path.join(labels_dir, label_name), os.path.join(split_lbl_dir, label_name))
    # Удаляем пустые исходные папки
    shutil.rmtree(os.path.join(dataset_dir, 'dataset'))
    
    print("Датасет разделён на train/val/test.")

    # Обновление data.yaml
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = [
            f"train: ../{dataset_dir}/images/train\n",
            f"val: ../{dataset_dir}/images/val\n",
            f"test: ../{dataset_dir}/images/test\n"
        ]
        # Сохраняем остальные строки, начиная с 4-й
        lines = new_lines + lines[3:]
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print('Пути train/val/test в data.yaml обновлены.')

    shutil.move(os.path.join(dataset_dir, 'data.yaml'), os.path.join(dataset_dir, 'dataset.yaml'))

if __name__ == '__main__':
    main()