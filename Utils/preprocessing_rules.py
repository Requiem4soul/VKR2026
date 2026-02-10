"""
Правила предобработки для различных типов изображений

Каждый тип изображений требует специфического подхода к предобработке,
основанного на физических характеристиках процесса получения изображения.

Научное обоснование:
- SAR: Frost et al. (1982), Lee (1981) - speckle шум нельзя убирать стандартной коррекцией яркости
- Medical: Pisano et al. (1998) - CLAHE с консервативными параметрами
- Natural: Tomasi & Manduchi (1998) - bilateral filtering
- Infrared: Vollmer & Möllmann (2017) - агрессивное улучшение контраста
- Microscopy: Sternberg (1983) - сохранение биmodal распределения

Автор: Система адаптивной предобработки
Дата: 2025
"""

from typing import Dict, List, Any


class PreprocessingRules:
    """
    Правила предобработки для различных модальностей изображений
    
    Каждое правило содержит:
    - Разрешённые/запрещённые методы
    - Параметры методов
    - Научное обоснование
    """
    
    RULES = {
        'sar': {
            # === РАЗРЕШЁННЫЕ МЕТОДЫ ===
            'denoise': {
                'enabled': True,
                'method': 'median',  # Lee (1981) - лучший для speckle
                'params': {
                    'ksize': 5  # Небольшое окно для сохранения деталей
                },
                'rationale': 'Median фильтр эффективен против speckle шума (мультипликативного)'
            },
            
            # === ЗАПРЕЩЁННЫЕ МЕТОДЫ ===
            'brightness_correction': {
                'enabled': False,  # КРИТИЧЕСКИ ВАЖНО!
                'rationale': 'Низкая яркость — физическое свойство радарного рассеяния от воды. '
                            'Коррекция яркости исказит информацию о материалах поверхности. '
                            'Источник: Frost et al. (1982)'
            },
            
            'contrast_enhancement': {
                'enabled': False,  # Обычно не нужно
                'rationale': 'Высокий контраст уже присутствует из-за разницы между водой и объектами. '
                            'Дополнительное усиление может привести к артефактам.'
            },
            
            'sharpening': {
                'enabled': False,
                'rationale': 'Speckle уже создаёт эффект высокой резкости. Дополнительное sharpening усилит шум.'
            },
            
            # === ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ===
            'source': 'Oliver & Quegan (2004), Lee (1981), Frost et al. (1982)',
            'description': 'SAR изображения требуют минимальной обработки - только подавление speckle шума'
        },
        
        'medical_xray': {
            'denoise': {
                'enabled': True,
                'method': 'nlm',  # Non-Local Means
                'params': {
                    'h': 10,
                    'template_window_size': 7,
                    'search_window_size': 21
                },
                'rationale': 'NLM эффективен для Poisson шума, характерного для рентгена'
            },
            
            'brightness_correction': {
                'enabled': False,
                'rationale': 'Низкая яркость обусловлена поглощением рентгеновского излучения. '
                            'Это диагностически важная информация, которую нельзя искажать.'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 1.0,  # Консервативный параметр!
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'CLAHE с низким clip_limit улучшает видимость деталей без искажения '
                            'диагностически важной информации. Источник: Pisano et al. (1998)'
            },
            
            'sharpening': {
                'enabled': False,
                'rationale': 'Может создать ложные детали, критично для медицинской диагностики'
            },
            
            'source': 'Pham et al. (2000), Pisano et al. (1998)',
            'description': 'Медицинские снимки требуют осторожной обработки с сохранением диагностической информации'
        },
        
        'natural_photo': {
            'denoise': {
                'enabled': True,
                'method': 'bilateral',  # Tomasi & Manduchi (1998)
                'params': {
                    'd': 9,
                    'sigma_color': 75,
                    'sigma_space': 75
                },
                'rationale': 'Bilateral фильтр сохраняет края при подавлении Gaussian шума'
            },
            
            'brightness_correction': {
                'enabled': True,
                'target_brightness': 0.5,
                'params': {
                    'factor_range': (0.5, 2.0)  # Разумные пределы
                },
                'rationale': 'Естественные фото могут быть недо/переэкспонированы. Коррекция улучшает восприятие.'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 2.0,  # Стандартное значение
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'CLAHE улучшает локальный контраст без пересветов'
            },
            
            'sharpening': {
                'enabled': True,
                'method': 'unsharp_mask',
                'params': {
                    'amount': 1.0
                },
                'rationale': 'Умеренное sharpening улучшает детализацию'
            },
            
            'source': 'Gonzalez & Woods (2018), Tomasi & Manduchi (1998)',
            'description': 'Естественные фото допускают полный спектр предобработки'
        },
        
        'infrared': {
            'denoise': {
                'enabled': True,
                'method': 'bilateral',
                'params': {
                    'd': 9,
                    'sigma_color': 75,
                    'sigma_space': 75
                },
                'rationale': 'Bilateral фильтр эффективен против шума при сохранении температурных границ'
            },
            
            'brightness_correction': {
                'enabled': True,
                'target_brightness': 0.5,
                'params': {
                    'factor_range': (0.7, 1.5)
                },
                'rationale': 'Коррекция яркости допустима - она не меняет относительные температуры'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 3.0,  # АГРЕССИВНО!
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'Низкий естественный контраст требует агрессивного улучшения. '
                            'Источник: Vollmer & Möllmann (2017)'
            },
            
            'sharpening': {
                'enabled': True,
                'method': 'unsharp_mask',
                'params': {
                    'amount': 1.5  # Более агрессивно
                },
                'rationale': 'Размытие от атмосферных эффектов требует восстановления резкости'
            },
            
            'source': 'Vollmer & Möllmann (2017), Portmann et al. (2019)',
            'description': 'Тепловизионные изображения требуют агрессивного улучшения из-за низкого естественного контраста'
        },
        
        'microscopy': {
            'denoise': {
                'enabled': True,
                'method': 'nlm',
                'params': {
                    'h': 10,
                    'template_window_size': 7,
                    'search_window_size': 21
                },
                'rationale': 'NLM хорошо работает с комбинированным Gaussian + Poisson шумом'
            },
            
            'brightness_correction': {
                'enabled': False,  # ВАЖНО!
                'rationale': 'Микроскопия имеет биmodal распределение (фон + объекты). '
                            'Глобальная коррекция яркости разрушит это распределение. '
                            'Источник: Sternberg (1983)'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 1.5,
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'Умеренный CLAHE улучшает видимость объектов сохраняя биmodal структуру'
            },
            
            'sharpening': {
                'enabled': False,
                'rationale': 'Может усилить шум, критично для анализа мелких структур'
            },
            
            'source': 'Sternberg (1983), Vincent & Soille (1991)',
            'description': 'Микроскопия требует сохранения биmodal распределения (фон/объекты)'
        }
    }
    
    @classmethod
    def get_rules(cls, modality: str) -> Dict:
        """
        Получить правила предобработки для конкретного типа изображений
        
        Args:
            modality: Тип изображений ('sar', 'medical_xray', и т.д.)
            
        Returns:
            dict: Правила для данного типа
        """
        return cls.RULES.get(modality, cls.RULES['natural_photo'])  # По умолчанию - natural
    
    @classmethod
    def is_method_allowed(cls, modality: str, method: str) -> bool:
        """
        Проверяет разрешён ли метод для данного типа изображений
        
        Args:
            modality: Тип изображений
            method: Название метода ('denoise', 'brightness_correction', и т.д.)
            
        Returns:
            bool: True если метод разрешён
        """
        rules = cls.get_rules(modality)
        
        if method not in rules:
            return True  # Если правила нет - разрешаем по умолчанию
        
        return rules[method].get('enabled', True)
    
    @classmethod
    def get_method_params(cls, modality: str, method: str) -> Dict:
        """
        Получить рекомендуемые параметры метода для типа изображений
        
        Args:
            modality: Тип изображений
            method: Название метода
            
        Returns:
            dict: Параметры метода
        """
        rules = cls.get_rules(modality)
        
        if method in rules and 'params' in rules[method]:
            return rules[method]['params']
        
        return {}
    
    @classmethod
    def get_rationale(cls, modality: str, method: str) -> str:
        """
        Получить объяснение почему метод разрешён/запрещён
        
        Args:
            modality: Тип изображений
            method: Название метода
            
        Returns:
            str: Объяснение
        """
        rules = cls.get_rules(modality)
        
        if method in rules:
            return rules[method].get('rationale', 'Нет объяснения')
        
        return 'Правило не определено'
    
    @classmethod
    def print_rules_summary(cls, modality: str):
        """
        Красиво печатает правила для типа изображений
        
        Args:
            modality: Тип изображений
        """
        rules = cls.get_rules(modality)
        
        print(f"\n{'='*70}")
        print(f"ПРАВИЛА ПРЕДОБРАБОТКИ: {modality.upper()}")
        print(f"{'='*70}")
        
        print(f"\n📝 Описание: {rules.get('description', 'Нет описания')}")
        print(f"📚 Источник: {rules.get('source', 'Нет источника')}")
        
        print(f"\n🔧 Методы предобработки:")
        
        methods = ['denoise', 'brightness_correction', 'contrast_enhancement', 'sharpening']
        
        for method in methods:
            if method not in rules:
                continue
            
            method_info = rules[method]
            enabled = method_info.get('enabled', True)
            
            status = "✅ РАЗРЕШЁН" if enabled else "❌ ЗАПРЕЩЁН"
            print(f"\n   {method.upper()}: {status}")
            
            if enabled and 'method' in method_info:
                print(f"      Метод: {method_info['method']}")
            
            if 'params' in method_info:
                print(f"      Параметры: {method_info['params']}")
            
            print(f"      Обоснование: {method_info.get('rationale', 'Нет')}")
        
        print(f"\n{'='*70}")


def demonstrate_rules():
    """Демонстрация правил для всех типов"""
    
    modalities = ['sar', 'medical_xray', 'natural_photo', 'infrared', 'microscopy']
    
    print("\n" + "="*70)
    print("ДЕМОНСТРАЦИЯ ПРАВИЛ ПРЕДОБРАБОТКИ")
    print("="*70)
    
    for modality in modalities:
        PreprocessingRules.print_rules_summary(modality)
        print("\n")


if __name__ == '__main__':
    # Запускаем демонстрацию
    demonstrate_rules()
