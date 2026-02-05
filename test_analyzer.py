"""
Тестовый скрипт для проверки анализатора
"""
from pathlib import Path
from Data.Datasets.dataset_work import get_dataset_path, list_available_datasets
from Utils.image_analyzer import UniversalImageAnalyzer
from Utils.preprocessing_selector import AdaptivePreprocessingSelector


def main():
    print("=" * 70)
    print("ТЕСТ УНИВЕРСАЛЬНОГО АНАЛИЗАТОРА")
    print("=" * 70)

    # Показываем доступные датасеты
    print("\nДоступные датасеты:")
    datasets = list_available_datasets(verbose=False)
    for i, name in enumerate(datasets):
        print(f"  [{i}] {name}")

    # Выбираем датасет
    choice = input("\nВведите номер или название датасета: ")

    if choice.isdigit():
        dataset_name = datasets[int(choice)]
    else:
        dataset_name = choice

    dataset_path = get_dataset_path(dataset_name)

    if not dataset_path.exists():
        print(f"❌ Датасет '{dataset_name}' не найден!")
        return

    print(f"\n✅ Выбран датасет: {dataset_name}")

    # Создаём анализатор
    analyzer = UniversalImageAnalyzer(verbose=False)

    # Анализируем датасет
    print(f"\nАнализируем датасет...")
    dataset_metrics, image_metrics = analyzer.analyze_dataset(
        dataset_path,
        split='train',
        sample_size=None  # None = все изображения
    )

    # Печатаем отчёт
    analyzer.print_dataset_report(dataset_metrics)

    # Выбираем стратегию
    print("\n" + "=" * 70)
    print("ВЫБОР СТРАТЕГИИ ПРЕДОБРАБОТКИ")
    print("=" * 70)

    selector = AdaptivePreprocessingSelector(analyzer)
    strategy = selector.select_strategy(dataset_path, split='train')

    if strategy['strategy'] == 'global':
        print("\n📌 Рекомендация: ГЛОБАЛЬНАЯ предобработка")
        print(f"   Методы: {', '.join(strategy['methods']) if strategy['methods'] else 'не требуется'}")

    else:
        print("\n📌 Рекомендация: АДАПТИВНАЯ предобработка")
        print(f"   Кластеров: {strategy['n_clusters']}")
        for cluster_id, cluster_info in strategy['clusters'].items():
            print(f"\n   Кластер {cluster_id}:")
            print(f"     Размер: {cluster_info['size']} изображений")
            print(f"     SNR: {cluster_info['characteristics']['avg_snr']:.1f} dB")
            print(f"     Контраст: {cluster_info['characteristics']['avg_contrast']:.3f}")
            # ← ДОБАВЬТЕ ЭТИ СТРОКИ:
            # Показываем доминирующий тип шума в кластере
            cluster_metrics = [image_metrics[i] for i in cluster_info['image_indices']]
            noise_types = [m.noise_type for m in cluster_metrics]
            from collections import Counter
            dominant_noise = Counter(noise_types).most_common(1)[0][0]
            print(f"     Тип шума: {dominant_noise}")

            print(
                f"     Методы: {', '.join(cluster_info['preprocessing']) if cluster_info['preprocessing'] else 'нет'}")
            print(
                f"     Методы: {', '.join(cluster_info['preprocessing']) if cluster_info['preprocessing'] else 'нет'}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()