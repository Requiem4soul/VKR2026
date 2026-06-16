from typing import Dict, List, Any


class PreprocessingRules:
    """
    Правила предобработки для различных модальностей изображений
    """
    
    RULES = {
        'sar': {
            # Да
            'denoise': {
                'enabled': True,
                'method': 'median',
                'params': {
                    'ksize': 5
                },
                'rationale': 'Шумоподавление разрешено для SAR. Median фильтр подходит '
                            'как простая альтернатива Lee-фильтру. Lee-фильтр '
                            'оптимален для мультипликативного speckle, но median также '
                            'применяется в SAR. Конкретный метод '
                            'определяется алгоритмом подбора из пула кандидатов.'
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
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 1.0,
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'CLAHE разрешён с консервативными параметрами (clip_limit=1.0). '
                            'После подавления speckle локальное контрастирование может '
                            'улучшить различимость объектов на SAR-изображениях. '
                            'Pisano et al. (1998) J. Digital Imaging 11(4):193-200: CLAHE '
                            'эффективен для низкоконтрастных изображений. '
                            'Низкий clip_limit минимизирует искажения статистики '
                            'backscatter-сигнала (Oliver & Quegan, 2004).'
            },
            
            'sharpening': {
                'enabled': False,
                'rationale': 'Speckle-шум в SAR имеет высокочастотную природу — sharpening '
                            'фильтры усиливают именно высокие частоты, что приведёт к '
                            'многократному усилению speckle вместо его подавления. '
                            'Oliver & Quegan (2004), ibid.'
            },

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
                'method': 'bilateral',
                'params': {
                    'd': 9,
                    'sigma_color': 75,
                    'sigma_space': 75
                },
                'rationale': 'Bilateral фильтр подавляет Gaussian шум сенсора при сохранении '
                            'краёв объектов. Tomasi & Manduchi (1998) ICCV: bilateral filtering '
                            'оптимален для естественных изображений с аддитивным шумом.'
            },
            
            'brightness_correction': {
                'enabled': True,
                'target_brightness': 0.5,
                'params': {
                    'factor_range': (0.5, 2.0)
                },
                'rationale': 'Естественные фото часто имеют неоптимальную экспозицию. '
                            'Гамма-коррекция нормализует яркость без потери информации. '
                            'Gonzalez & Woods (2018), гл. 3.2: степенные преобразования — '
                            'базовый инструмент коррекции экспозиции.'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 2.0,
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'CLAHE улучшает локальный контраст без пересветов. '
                            'Pisano et al. (1998) J. Digital Imaging 11(4):193-200: CLAHE '
                            'эффективен для изображений с неравномерным освещением. '
                            'clip_limit=2.0 — стандартное значение (Gonzalez & Woods, 2018).'
            },
            
            'sharpening': {
                'enabled': True,
                'method': 'unsharp_mask',
                'params': {
                    'amount': 1.0
                },
                'rationale': 'Unsharp masking повышает визуальную детализацию. '
                            'Gonzalez & Woods (2018), гл. 3.6: стандартный метод '
                            'повышения резкости для фотографических изображений.'
            },
            
            'source': 'Gonzalez & Woods (2018) "Digital Image Processing"; '
                     'Tomasi & Manduchi (1998) ICCV',
            'description': 'Естественные фото допускают полный спектр предобработки. '
                          'Lee-фильтр исключён — предназначен для мультипликативного '
                          'speckle-шума SAR, не для аддитивного шума камер.'
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
                'rationale': 'Bilateral фильтр сохраняет температурные границы объектов '
                            'при подавлении сенсорного шума (Gaussian + fixed-pattern noise). '
                            'Tomasi & Manduchi (1998) ICCV: bilateral сохраняет края — '
                            'критично для IR, где границы объектов имеют малый контраст.'
            },
            
            'brightness_correction': {
                'enabled': True,
                'target_brightness': 0.5,
                'params': {
                    'factor_range': (0.7, 1.5)
                },
                'rationale': 'Коррекция яркости допустима для задач детекции/классификации: '
                            'нейросеть работает с визуальными паттернами, а не абсолютными '
                            'температурами. Гамма-коррекция (нелинейная) меняет относительные '
                            'значения, но для распознавания объектов это приемлемо. '
                            'Gonzalez & Woods (2018) "Digital Image Processing", гл. 3.2.'
            },
            
            'contrast_enhancement': {
                'enabled': True,
                'method': 'clahe',
                'params': {
                    'clip_limit': 3.0,
                    'tile_grid_size': (8, 8)
                },
                'rationale': 'IR-изображения имеют узкий динамический диапазон — CLAHE '
                            'с повышенным clip_limit расширяет контраст для выявления объектов. '
                            'Pisano et al. (1998) J. Digital Imaging 11(4):193-200: CLAHE '
                            'эффективен для низкоконтрастных изображений (аналогия с рентгеном). '
                            'Vollmer & Mollmann (2017) "Infrared Thermal Imaging", Wiley — '
                            'описывают низкий естественный контраст IR и необходимость '
                            'постобработки для визуального анализа.'
            },
            
            'sharpening': {
                'enabled': True,
                'method': 'unsharp_mask',
                'params': {
                    'amount': 1.5
                },
                'rationale': 'Unsharp masking повышает визуальную чёткость границ объектов '
                            'на IR-изображениях, где контраст между объектом и фоном мал. '
                            'Gonzalez & Woods (2018), гл. 3.6: unsharp masking — стандартный '
                            'метод повышения резкости для визуального анализа. Повышенный '
                            'alpha=1.5 компенсирует размытие от низкого контраста IR-сенсора.'
            },
            
            'source': 'Vollmer & Mollmann (2017) "Infrared Thermal Imaging", Wiley; '
                     'Gonzalez & Woods (2018) "Digital Image Processing"; '
                     'Tomasi & Manduchi (1998) ICCV',
            'description': 'Тепловизионные изображения допускают все четыре группы предобработки. '
                          'Низкий естественный контраст IR-сенсоров допускает агрессивные '
                          'параметры контрастирования и резкости.'
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
        """
        return cls.RULES.get(modality, cls.RULES['natural_photo'])  # По умолчанию - natural
    
    @classmethod
    def is_method_allowed(cls, modality: str, method: str) -> bool:
        """
        Проверяет разрешён ли метод для данного типа изображений
        """
        rules = cls.get_rules(modality)
        
        if method not in rules:
            return True  # Если правила нет - разрешаем по умолчанию
        
        return rules[method].get('enabled', True)

    # Мертво, но удалять страшно
    @classmethod
    def get_method_params(cls, modality: str, method: str) -> Dict:
        """
        Получить рекомендуемые параметры метода для типа изображений
        """
        rules = cls.get_rules(modality)
        
        if method in rules and 'params' in rules[method]:
            return rules[method]['params']
        
        return {}
    
    @classmethod
    def get_rationale(cls, modality: str, method: str) -> str:
        """
        Получить объяснение почему метод разрешён/запрещён
        """
        rules = cls.get_rules(modality)
        
        if method in rules:
            return rules[method].get('rationale', 'Нет объяснения')
        
        return 'Правило не определено'
    
    @classmethod
    def print_rules_summary(cls, modality: str):
        """
        Красиво печатает правила для типа изображений
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
