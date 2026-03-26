"""
Валидация процента скрининга SHA через коэффициент Спирмена и Jaccard similarity.

Спирмен (ρ) — стабильность ранжирования всех кандидатов Фазы 1.
Jaccard    — совпадение финальных survivors между двумя прогонами.

Пороги:
  Спирмен ρ ≥ 0.9  — ранжирование стабильно
  Jaccard   ≥ 0.7  — множества survivors достаточно похожи

Использование:
  Заполни секции "ВХОДНЫЕ ДАННЫЕ" для каждой пары прогонов и запусти:
  python screening_validation.py
"""

from scipy.stats import spearmanr

# ══════════════════════════════════════════════════════════════════════════════
# ВХОДНЫЕ ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════════════

# Названия методов — одинаковый порядок для обоих прогонов.
# Порядок должен совпадать с порядком в логе Фазы 1.
METHODS = [
    "Median k3", "Median k5", "Gaussian k3", "Gaussian k5", "Bilateral 75", "Bilateral 150",
    "CLAHE 1.0", "CLAHE 2.0", "CLAHE 3.0", "HistEq",
    "Gamma γ=0.5", "Gamma γ=0.8", "Gamma γ=1.2",
    "Unsharp 0.5", "Unsharp 1.0", "Unsharp 1.5",
]

# Scores всех кандидатов Фазы 1 — в том же порядке что METHODS.
# Заполни из лога: строки вида "score=0.XXXX"
SCORES_A = [  # первый прогон (например 30%)
    0.8234, 0.8247, 0.8288, 0.8267, 0.8256, 0.8265,
    0.8276, 0.8199, 0.8265, 0.8506,
    0.8256, 0.8201, 0.8275,
    0.8204, 0.8227, 0.8100
]
PERCENT_A = 40  # процент скрининга первого прогона

SCORES_B = [  # второй прогон (например 40%) — ЗАПОЛНИ ПОСЛЕ СЛЕДУЮЩЕГО ПРОГОНА
    0.8335, 0.8335, 0.8359, 0.8336, 0.8345, 0.8341,
    0.8340, 0.8248, 0.8318, 0.8558,
    0.8313, 0.8253, 0.8318,
    0.8261, 0.8282, 0.8161
]
PERCENT_B = 50  # процент скрининга второго прогона

# Финальные survivors (после всех фаз, перед финальным обучением).
# Используй точные названия методов как они выводятся в логе.
SURVIVORS_A = {
    "Histogram Equalization",
}
SURVIVORS_B = {
    "Histogram Equalization",
}

# ══════════════════════════════════════════════════════════════════════════════
# РАСЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

SPEARMAN_THRESHOLD = 0.9
JACCARD_THRESHOLD  = 0.7


def calc_spearman(scores_a, scores_b, methods, pct_a, pct_b):
    assert len(scores_a) == len(scores_b) == len(methods), \
        "Количество scores должно совпадать с количеством методов"

    rho, pvalue = spearmanr(scores_a, scores_b)

    print(f"\n{'='*60}")
    print(f"КОЭФФИЦИЕНТ СПИРМЕНА ({pct_a}% vs {pct_b}%)")
    print(f"{'='*60}")

    # Показываем ранги для наглядности
    def get_ranks(scores):
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_idx, 1):
            ranks[idx] = rank
        return ranks

    ranks_a = get_ranks(scores_a)
    ranks_b = get_ranks(scores_b)

    print(f"\n{'Метод':<22} {'Score A':>8} {'Ранг A':>7} {'Score B':>8} {'Ранг B':>7} {'|d|':>5}")
    print("-" * 60)
    for i, m in enumerate(methods):
        d = abs(ranks_a[i] - ranks_b[i])
        print(f"{m:<22} {scores_a[i]:>8.4f} {ranks_a[i]:>7} {scores_b[i]:>8.4f} {ranks_b[i]:>7} {d:>5}")

    print(f"\nρ = {rho:.4f}  (p-value={pvalue:.4f})")
    ok = rho >= SPEARMAN_THRESHOLD
    print(f"Порог ≥ {SPEARMAN_THRESHOLD}: {'ДОСТИГНУТ' if ok else 'НЕ ДОСТИГНУТ'}")
    return rho, ok


def calc_jaccard(survivors_a, survivors_b, pct_a, pct_b):
    print(f"\n{'='*60}")
    print(f"JACCARD SIMILARITY ({pct_a}% vs {pct_b}%)")
    print(f"{'='*60}")
    print(f"\nSurvivors {pct_a}%: {survivors_a}")
    print(f"Survivors {pct_b}%: {survivors_b}")

    intersection = survivors_a & survivors_b
    union        = survivors_a | survivors_b

    if not union:
        print("Оба множества пусты — Jaccard не определён.")
        return None, False

    jaccard = len(intersection) / len(union)
    print(f"\nПересечение: {intersection}")
    print(f"Объединение: {union}")
    print(f"\nJaccard = |A∩B|/|A∪B| = {len(intersection)}/{len(union)} = {jaccard:.4f}")
    ok = jaccard >= JACCARD_THRESHOLD
    print(f"Порог ≥ {JACCARD_THRESHOLD}: {'ДОСТИГНУТ' if ok else 'НЕ ДОСТИГНУТ'}")
    return jaccard, ok


def main():
    # Проверка что второй прогон заполнен
    if all(s == 0.0 for s in SCORES_B):
        print("SCORES_B не заполнен — запусти второй прогон и заполни данные.")
        print("   Пока показываю только ранги первого прогона:\n")
        ranked = sorted(zip(METHODS, SCORES_A), key=lambda x: x[1], reverse=True)
        for i, (m, s) in enumerate(ranked, 1):
            marker = " ← выше baseline" if s > max(SCORES_A) * 0.99 else ""
            print(f"  {i:2d}. {m:<22}  score={s:.4f}{marker}")
        return

    rho, spearman_ok = calc_spearman(SCORES_A, SCORES_B, METHODS, PERCENT_A, PERCENT_B)
    jac, jaccard_ok  = calc_jaccard(SURVIVORS_A, SURVIVORS_B, PERCENT_A, PERCENT_B)

    print(f"\n{'='*60}")
    print("ИТОГ")
    print(f"{'='*60}")
    both_ok = spearman_ok and jaccard_ok
    if both_ok:
        print(f" {PERCENT_A}% скрининга ДОСТАТОЧНО — оба критерия выполнены.")
        print(f"   Ρ={rho:.4f} ≥ {SPEARMAN_THRESHOLD},  Jaccard={jac:.4f} ≥ {JACCARD_THRESHOLD}")
    else:
        print(f" {PERCENT_A}% скрининга НЕДОСТАТОЧНО — не все критерии выполнены.")
        if not spearman_ok:
            print(f"   Спирмен ρ={rho:.4f} < {SPEARMAN_THRESHOLD}")
        if not jaccard_ok:
            print(f"   Jaccard={jac:.4f} < {JACCARD_THRESHOLD}")
        print(f"   Рекомендуется использовать {PERCENT_B}% скрининга.")


if __name__ == "__main__":
    main()
