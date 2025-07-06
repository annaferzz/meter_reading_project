import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from PIL import Image
import yaml
from src.config import DATASET_YAML_PATH

class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.splits = ['train', 'valid', 'test']
        self.classes = ['gaz_meter', 'indicator']
        self.stats = {}
        
    def load_yaml_config(self):
        """Загрузка конфигурации датасета из YAML"""
        try:
            with open(DATASET_YAML_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Ошибка при загрузке YAML конфигурации: {e}")
            return None
            
    def analyze_images(self):
        """Анализ изображений в датасете"""
        print("Анализ изображений...")
        for split in self.splits:
            split_path = self.dataset_path / split / 'images'
            if not split_path.exists():
                print(f"Предупреждение: директория {split_path} не найдена")
                continue
                
            images = list(split_path.glob('*.jpg')) + list(split_path.glob('*.jpeg')) + list(split_path.glob('*.png'))
            if not images:
                print(f"Предупреждение: изображения не найдены в {split_path}")
                continue
                
            resolutions = []
            sizes = []
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        resolutions.append((width, height))
                        sizes.append(os.path.getsize(img_path) / 1024)  # размер в KB
                except Exception as e:
                    print(f"Ошибка при обработке {img_path}: {e}")
                    continue
                    
            if resolutions:
                self.stats[f'{split}_images'] = {
                    'count': len(images),
                    'avg_resolution': np.mean(resolutions, axis=0).tolist(),
                    'min_resolution': np.min(resolutions, axis=0).tolist(),
                    'max_resolution': np.max(resolutions, axis=0).tolist(),
                    'avg_size_kb': np.mean(sizes),
                    'total_size_mb': sum(sizes) / 1024
                }
            
    def analyze_annotations(self):
        """Анализ аннотаций в датасете"""
        print("Анализ аннотаций...")
        for split in self.splits:
            labels_path = self.dataset_path / split / 'labels'
            if not labels_path.exists():
                print(f"Предупреждение: директория {labels_path} не найдена")
                continue
                
            class_counts = Counter()
            boxes_per_image = []
            
            label_files = list(labels_path.glob('*.txt'))
            if not label_files:
                print(f"Предупреждение: аннотации не найдены в {labels_path}")
                continue
                
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        boxes_per_image.append(len(lines))
                        for line in lines:
                            class_id = int(line.split()[0])
                            if class_id < len(self.classes):
                                class_counts[self.classes[class_id]] += 1
                            else:
                                print(f"Предупреждение: неизвестный класс {class_id} в {label_file}")
                except Exception as e:
                    print(f"Ошибка при обработке {label_file}: {e}")
                    continue
                    
            if boxes_per_image:
                self.stats[f'{split}_annotations'] = {
                    'class_distribution': dict(class_counts),
                    'total_annotations': sum(class_counts.values()),
                    'avg_boxes_per_image': np.mean(boxes_per_image),
                    'max_boxes_per_image': np.max(boxes_per_image)
                }
            
    def plot_class_distribution(self):
        """Построение графика распределения классов"""
        print("Создание визуализации распределения классов...")
        
        # Проверяем наличие данных для построения графика
        valid_splits = []
        for split in self.splits:
            if f'{split}_annotations' in self.stats and self.stats[f'{split}_annotations']['class_distribution']:
                valid_splits.append(split)
                
        if not valid_splits:
            print("Предупреждение: нет данных для построения графика распределения классов")
            return
            
        plt.figure(figsize=(12, 6))
        x = np.arange(len(valid_splits))
        width = 0.35
        
        for i, class_name in enumerate(self.classes):
            counts = [self.stats[f'{split}_annotations']['class_distribution'].get(class_name, 0) 
                     for split in valid_splits]
            plt.bar(x + i*width, counts, width, label=class_name)
            
        plt.xlabel('Splits')
        plt.ylabel('Number of Objects')
        plt.title('Class Distribution Across Dataset Splits')
        plt.xticks(x + width/2, valid_splits)
        plt.legend()
        
        save_path = self.dataset_path / 'class_distribution.png'
        try:
            plt.savefig(save_path)
            print(f"График сохранен в {save_path}")
        except Exception as e:
            print(f"Ошибка при сохранении графика: {e}")
        finally:
            plt.close()
        
    def generate_report(self):
        """Генерация отчета о датасете"""
        print("Генерация отчета...")
        report = []
        report.append("# Анализ датасета\n")
        
        # Общая статистика
        report.append("## Общая информация\n")
        total_images = sum(self.stats.get(f'{split}_images', {}).get('count', 0) for split in self.splits)
        report.append(f"Всего изображений: {total_images}\n")
        
        # Статистика по сплитам
        report.append("\n## Статистика по разделам датасета\n")
        for split in self.splits:
            if f'{split}_images' not in self.stats:
                continue
                
            report.append(f"\n### {split.capitalize()}\n")
            img_stats = self.stats[f'{split}_images']
            
            report.append(f"- Количество изображений: {img_stats['count']}")
            report.append(f"- Среднее разрешение: {img_stats['avg_resolution'][0]:.0f}x{img_stats['avg_resolution'][1]:.0f}")
            report.append(f"- Средний размер файла: {img_stats['avg_size_kb']:.1f}KB")
            
            if f'{split}_annotations' in self.stats:
                ann_stats = self.stats[f'{split}_annotations']
                report.append(f"- Всего аннотаций: {ann_stats['total_annotations']}")
                report.append(f"- Среднее количество боксов на изображение: {ann_stats['avg_boxes_per_image']:.2f}")
                
                report.append("\nРаспределение классов:")
                for cls, count in ann_stats['class_distribution'].items():
                    report.append(f"- {cls}: {count}")
                    
        # Сохранение отчета
        try:
            report_path = self.dataset_path / 'dataset_analysis.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            print(f"Отчет сохранен в {report_path}")
        except Exception as e:
            print(f"Ошибка при сохранении отчета: {e}")
            
    def run_analysis(self):
        """Запуск полного анализа датасета"""
        print("Начало анализа датасета...")
        try:
            self.analyze_images()
            self.analyze_annotations()
            if any(f'{split}_annotations' in self.stats for split in self.splits):
                self.plot_class_distribution()
            self.generate_report()
            print("Анализ успешно завершен")
        except Exception as e:
            print(f"Ошибка при выполнении анализа: {e}")
        finally:
            print(f"Результаты сохранены в {self.dataset_path}")

def main():
    """Основная функция для обработки и анализа датасета"""
    try:
        dataset_path = Path("../dataset_gaz")
        if not dataset_path.exists():
            print(f"Ошибка: директория датасета {dataset_path} не найдена")
            return
            
        analyzer = DatasetAnalyzer(dataset_path)
        analyzer.run_analysis()
    except Exception as e:
        print(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    main() 