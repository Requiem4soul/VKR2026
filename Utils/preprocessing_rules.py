"""
Правила предобработки для различных типов изображений

Каждый тип изображений требует специфического подхода к предобработке,
основанного на физических характеристиках процесса получения изображения.

Научное обоснование:
- SAR: Oliver & Quegan (2004), Lee (1981) - speckle шум нельзя убирать стандартной коррекцией яркости
- Medical: Pisano et al. (1998) - CLAHE с консервативными параметрами
- Natural: Tomasi & Manduchi (1998) - bilateral filtering
- Infrared: Vollmer & Möllmann (2017) - агрессивное улучшение контраста
- Microscopy: Kolarević et al. (2018) Journal of Microscopy 269(3):264-276 — локальные методы эффективнее глобальных

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
                'rationale': 'В SAR яркость пикселя кодирует коэффициент обратного рассеяния '
                            '(backscatter coefficient) — физическую характеристику поверхности. '
                            'Глобальная коррекция яркости исказит эту информацию и сделает '
                            'невозможным корректное сравнение объектов на снимке. '
                            'Oliver & Quegan (2004) "Understanding Synthetic Aperture Radar '
                            'Images", SciTech Publishing — фундаментальный источник по '
                            'физике SAR-изображений.'
            },

            # === ИСКЛЮЧЕНИЕ ===
            'contrast_enhancement': {
                'enabled': True,  # С оговорками
                'method': 'clahe',
                'params': {
                    'clip_limit': 1.0,
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'Разрешено с осторожностью: после подавления speckle-шумов '
                            'локальное контрастирование (CLAHE) с низким clip_limit может '
                            'может улучшать визуальную информативность при минимизации искажений статистики сигнала. '
                            'Однако методы контрастирования применяются только после шумоподавления и с консервативными параметрами. '
                            'Remote Sens. 2019, 11(13), 1532; ISPRS J. Photogramm. Remote Sens., 2024'
            },
            
            'sharpening': {
                'enabled': False,
                'rationale': 'Speckle-шум в SAR имеет высокочастотную природу — sharpening '
                            'фильтры усиливают именно высокие частоты, что приведёт к '
                            'многократному усилению speckle вместо его подавления. '
                            'Oliver & Quegan (2004), ibid.'
            },
            
            # === ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ===
            'source': 'Oliver & Quegan (2004) "Understanding Synthetic Aperture Radar Images", SciTech Publishing; '
                     'Lee J.S. (1981) IEEE Trans. Pattern Anal. Mach. Intell., 2:165-168',
            'description': 'SAR изображения требуют минимальной обработки - только подавление speckle шума'
        },
        
        'medical_xray': {
            'denoise': {
                'enabled': True,
                'method': 'wiener',
                'params': {
                    'size': 5
                },
                'rationale': 'Wiener filter адаптируется к локальной дисперсии шума — '
                            'эффективен для Poisson шума рентгена при O(N) сложности. '
                            'Заменяет NLM (O(N²)) ради вычислительной эффективности. '
                            'Wiener (1949); Fan et al. (2019) Visual Computing 2(1).'
            },
            
            'brightness_correction': {
                'enabled': True,
                'rationale': 'Коррекция яркости разрешена для рентгеновских снимков. '
                            'Pisano et al. (2000) "Image processing algorithms for digital '
                            'mammography: a pictorial essay", RadioGraphics 20:1479-1491 — '
                            'brightness adjustment перечислен среди применяемых методов '
                            'постобработки маммограмм наряду с contrast enhancement и '
                            'unsharp masking. Коррекция яркости может улучшать видимость '
                            'структур при недо- или переэкспонированных снимках.'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 1.0,  # Консервативный параметр
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'CLAHE с низким clip_limit улучшает видимость деталей без искажения '
                            'диагностически важной информации. '
                            'Pisano et al. (1998) J. Digital Imaging, 11(4):193-200 — '
                            'первая работа демонстрирующая эффективность CLAHE для маммографии.'
            },
            
            'sharpening': {
                'enabled': True,
                'method': 'unsharp_mask',
                'params': {
                    'amount': 0.5  # Умеренный параметр для медицинских снимков
                },
                'rationale': 'Unsharp masking — стандартный инструмент постобработки маммограмм. '
                            'Pisano et al. (2000) "Image processing algorithms for digital '
                            'mammography: a pictorial essay", RadioGraphics 20:1479-1491 — '
                            'unsharp masking прямо перечислен среди применяемых методов. '
                            'Умеренный alpha=0.5 улучшает видимость микрокальцинатов без '
                            'создания артефактов.'
            },
            
            'source': 'Pisano et al. (1998) J. Digital Imaging 11(4):193-200; '
                     'Pisano et al. (2000) RadioGraphics 20:1479-1491',
            'description': 'Медицинские рентгеновские снимки допускают все четыре группы предобработки; '
                          'Lee-фильтр исключён как предназначенный для мультипликативного speckle-шума SAR'
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
                    'clip_limit': 3.0,  # повышенное значение для низкоконтрастных снимков
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
                'method': 'wiener',
                'params': {
                    'size': 5
                },
                'rationale': 'Wiener filter адаптируется к смешанному Gaussian+Poisson шуму '
                            'микроскопии при O(N) сложности. '
                            'Заменяет NLM (O(N²)) ради вычислительной эффективности. '
                            'Wiener (1949); Fan et al. (2019) Visual Computing 2(1).'
            },
            
            'brightness_correction': {
                'enabled': True,
                'rationale': 'Коррекция яркости разрешена для гистопатологических снимков. '
                            'Murcia-Gomez et al. (2022) Applied Sciences 12(22):11375 — '
                            'сравнительное исследование методов предобработки на BreakHis '
                            'показало что статистически значимой разницы между фильтрами '
                            'нет: основную роль играет архитектура модели. '
                            'Следовательно, запрещать brightness нет научного основания — '
                            'система подберёт оптимальное решение эмпирически.'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 1.5,
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'Умеренный CLAHE улучшает локальный контраст без изменения '
                            'глобальных яркостных соотношений. '
                            'Kolarević et al. (2018) ibid. подтверждают эффективность '
                            'контрастирования для гистопатологических снимков.'
            },
            
            'sharpening': {
                'enabled': True,
                'method': 'unsharp_mask',
                'params': {
                    'amount': 0.5
                },
                'rationale': 'Sharpening разрешён для гистопатологических снимков. '
                            'Dziadosz et al. (2025) Scientific Reports — применение '
                            'sharpening для усиления краёв клеточных структур и повышения '
                            'контраста между светлыми и тёмными областями на BreakHis '
                            'дало точность классификации 99.60%. '
                            'Murcia-Gomez et al. (2022) Applied Sciences 12(22):11375 — '
                            'статистически значимой разницы между методами предобработки '
                            'нет, поэтому система определяет оптимальный метод эмпирически. '
                            'Умеренный alpha=0.5 снижает риск усиления артефактов.'
            },
            
            'source': 'Kolarević et al. (2018) Journal of Microscopy, 269(3):264-276; '
                     'Murcia-Gomez et al. (2022) Applied Sciences 12(22):11375; '
                     'Dziadosz et al. (2025) Scientific Reports',
            'description': 'Гистопатологические снимки допускают все четыре группы предобработки; '
                          'Lee-фильтр и Gaussian blur исключены как неподходящие для клеточных структур'
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

        print(f"\n  Описание: {rules.get('description', '')}")
        print(f"  Источник: {rules.get('source', '')}")

        print(f"\n  Методы предобработки:")
        
        methods = ['denoise', 'brightness_correction', 'contrast_enhancement', 'sharpening']
        
        for method in methods:
            if method not in rules:
                continue
            
            method_info = rules[method]
            enabled = method_info.get('enabled', True)
            
            status = "разрешён" if enabled else "запрещён"
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
