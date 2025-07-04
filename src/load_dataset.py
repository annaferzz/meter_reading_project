import os
import requests
import zipfile
import shutil

dataset_url = 'https://universe.roboflow.com/ds/HJ1zhypP3Q?key=o5LXfvCbPV'

def main():
    dataset_dir = 'dataset'
    zip_path = 'dataset.zip'
    # Скачивание архива
    print(f"Скачивание датасета из {dataset_url}...")
    with requests.get(dataset_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Архив скачан.")
    # Удаляем старую папку dataset, если есть
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    # Распаковка архива
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Архив распакован.")
    os.remove(zip_path)
    # Удаление файлов без closeup в имени
    for subfolder in ['images', 'labels']:
        target_dir = os.path.join(dataset_dir, 'train', subfolder)
        if os.path.exists(target_dir):
            for fname in os.listdir(target_dir):
                if 'closeup' not in fname:
                    os.remove(os.path.join(target_dir, fname))
    print("Удаление файлов не содержащих 'closeup' завершено.")

if __name__ == '__main__':
    main()