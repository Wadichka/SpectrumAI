"""Unit-тесты cross-validation предсказанных и фактических групп (§4.4.5)."""

from __future__ import annotations

import pytest

from app.ml.cross_validation import (
    ConsistencyResult,
    candidate_groups_from_smiles,
    compute_consistency,
)


def test_consistency_identical_sets() -> None:
    res = compute_consistency(["alcohol_OH", "ether"], ["alcohol_OH", "ether"])
    assert res.jaccard == 1.0
    assert res.consistent is True
    assert res.matched == ("alcohol_OH", "ether")
    assert res.missing == ()
    assert res.extra == ()


def test_consistency_empty_sets() -> None:
    res = compute_consistency([], [])
    assert res.jaccard == 0.0
    assert res.consistent is False
    assert res.matched == res.missing == res.extra == ()


def test_consistency_partial_overlap_below_threshold() -> None:
    res = compute_consistency(
        ["alcohol_OH", "carbonyl", "amine_primary"],
        ["alcohol_OH", "ether"],
    )
    # Пересечение = {alcohol_OH}, объединение = 4 элемента → 1/4 = 0.25.
    assert res.jaccard == pytest.approx(0.25)
    assert res.consistent is False
    assert res.matched == ("alcohol_OH",)
    assert set(res.missing) == {"amine_primary", "carbonyl"}
    assert res.extra == ("ether",)


def test_consistency_meets_threshold() -> None:
    res = compute_consistency(
        ["alcohol_OH", "carbonyl"],
        ["alcohol_OH", "carbonyl", "ether"],
    )
    # 2 / 3 = 0.667.
    assert res.jaccard == pytest.approx(2 / 3)
    assert res.consistent is True


def test_consistency_custom_threshold() -> None:
    res = compute_consistency(["a", "b"], ["a", "c"], threshold=0.2)
    assert res.jaccard == pytest.approx(1 / 3)
    assert res.consistent is True


def test_consistency_returns_dataclass() -> None:
    res = compute_consistency(["a"], ["a"])
    assert isinstance(res, ConsistencyResult)


def test_candidate_groups_for_ethanol() -> None:
    groups = candidate_groups_from_smiles("CCO")
    assert "alcohol_OH" in groups


def test_candidate_groups_invalid_smiles_raises() -> None:
    with pytest.raises(ValueError):
        candidate_groups_from_smiles("not_a_smiles")
