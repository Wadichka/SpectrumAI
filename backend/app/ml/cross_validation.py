"""Cross-validation предсказанных и фактических функциональных групп (§4.4.5).

Глава 4 §4.4.5 говорит о «двухуровневом согласовании»: предсказание CNN и
табличное определение функциональных групп должны совпадать. Формальной
метрики нет — здесь применяется **Jaccard ≥ threshold** (согласовано с
автором; threshold по умолчанию 0.5).

Группы кандидата определяются через ту же SMARTS-разметку, что и labels для
синтетики (``pipelines.labeling.label_functional_groups``), — это
гарантирует семантическое совпадение между предсказаниями и реальностью.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.ml import _ml_path  # noqa: F401 — sys.path
from pipelines.labeling import label_functional_groups


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    """Результат сравнения двух наборов функциональных групп."""

    jaccard: float
    consistent: bool
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]


def compute_consistency(
    predicted: Sequence[str],
    candidate_groups: Sequence[str],
    *,
    threshold: float = 0.5,
) -> ConsistencyResult:
    """Считает Jaccard и пометку consistent.

    Args:
        predicted: предсказанные имена групп (вероятность > threshold классификатора).
        candidate_groups: группы, реально присутствующие в кандидате (SMARTS).
        threshold: порог Jaccard для пометки ``consistent``.
    """
    pred_set = frozenset(predicted)
    cand_set = frozenset(candidate_groups)
    matched = pred_set & cand_set
    union = pred_set | cand_set
    jaccard = len(matched) / len(union) if union else 0.0
    return ConsistencyResult(
        jaccard=jaccard,
        consistent=jaccard >= threshold,
        matched=tuple(sorted(matched)),
        missing=tuple(sorted(pred_set - cand_set)),
        extra=tuple(sorted(cand_set - pred_set)),
    )


def candidate_groups_from_smiles(smiles: str) -> tuple[str, ...]:
    """Возвращает имена функциональных групп, реально присутствующих в SMILES.

    Невалидные SMILES → ``ValueError`` (из ``label_functional_groups``).
    """
    counts = label_functional_groups(smiles)
    return tuple(sorted(name for name, count in counts.items() if count > 0))


__all__ = [
    "ConsistencyResult",
    "candidate_groups_from_smiles",
    "compute_consistency",
]
