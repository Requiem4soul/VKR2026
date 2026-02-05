import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from Utils.image_analyzer import UniversalImageAnalyzer, ImageMetrics


class AdaptivePreprocessingSelector:
    """Выбирает стратегию предобработки на основе анализа датасета"""

    def __init__(self, analyzer: UniversalImageAnalyzer):
        self.analyzer = analyzer

    def select_strategy(
            self,
            dataset_path: Path,
            split: str = 'train'
    ) -> Dict:
        """Определяет оптимальную стратегию предобработки"""
        dataset_metrics, image_metrics = self.analyzer.analyze_dataset(
            dataset_path,
            split=split
        )

        if not dataset_metrics.needs_adaptive_preprocessing:
            return {
                'strategy': 'global',
                'methods': dataset_metrics.recommended_global_preprocessing,
                'dataset_metrics': dataset_metrics
            }

        else:
            clusters = self._cluster_images(
                image_metrics,
                n_clusters=dataset_metrics.suggested_clusters
            )

            return {
                'strategy': 'adaptive',
                'n_clusters': dataset_metrics.suggested_clusters,
                'clusters': clusters,
                'dataset_metrics': dataset_metrics,
                'image_metrics': image_metrics
            }

    def _cluster_images(
            self,
            image_metrics: List[ImageMetrics],
            n_clusters: int
    ) -> Dict:
        """Кластеризует изображения по характеристикам"""
        features = np.array([
            [
                m.snr_db,
                m.global_contrast,
                m.mean_brightness,
                m.sharpness_score
            ]
            for m in image_metrics
        ])

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_scaled)

        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_metrics = [image_metrics[i] for i in cluster_indices]

            avg_snr = np.mean([m.snr_db for m in cluster_metrics])
            avg_contrast = np.mean([m.global_contrast for m in cluster_metrics])
            avg_brightness = np.mean([m.mean_brightness for m in cluster_metrics])

            cluster_preprocessing = self._recommend_for_cluster(cluster_metrics)

            clusters[cluster_id] = {
                'size': len(cluster_metrics),
                'characteristics': {
                    'avg_snr': avg_snr,
                    'avg_contrast': avg_contrast,
                    'avg_brightness': avg_brightness
                },
                'preprocessing': cluster_preprocessing,
                'image_indices': cluster_indices.tolist()
            }

        return clusters

    def _recommend_for_cluster(
            self,
            cluster_metrics: List[ImageMetrics]
    ) -> List[str]:
        """Рекомендует предобработку для кластера"""
        high_noise_ratio = sum(1 for m in cluster_metrics if m.noise_level == 'high') / len(cluster_metrics)
        low_contrast_ratio = sum(1 for m in cluster_metrics if m.contrast_level == 'low') / len(cluster_metrics)
        dark_ratio = sum(1 for m in cluster_metrics if m.brightness_level == 'dark') / len(cluster_metrics)
        blur_ratio = sum(1 for m in cluster_metrics if m.blur_detected) / len(cluster_metrics)

        methods = []
        if high_noise_ratio > 0.5:
            methods.append('denoise')
        if low_contrast_ratio > 0.5:
            methods.append('contrast_enhancement')
        if dark_ratio > 0.5:
            methods.append('brightness_correction')
        if blur_ratio > 0.5:
            methods.append('sharpening')

        return methods