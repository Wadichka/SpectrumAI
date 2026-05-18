"""Тесты автоматической разметки функциональных групп (§5.4)."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.labeling import (
    FUNCTIONAL_GROUPS,
    GROUP_NAMES,
    N_GROUPS,
    label_functional_groups,
    multi_hot_labels,
)


def _active(labels: dict[str, int]) -> dict[str, int]:
    return {name: count for name, count in labels.items() if count > 0}


def test_constants_consistent() -> None:
    assert N_GROUPS == 25
    assert len(FUNCTIONAL_GROUPS) == 25
    assert len(GROUP_NAMES) == 25
    assert GROUP_NAMES[0] == "alcohol_OH"
    assert GROUP_NAMES[-1] == "thiol_thioether"


def test_ethanol_has_hydroxyl_methyl_methylene() -> None:
    active = _active(label_functional_groups("CCO"))
    assert active.get("alcohol_OH") == 1
    assert active.get("ch3_group") == 1
    assert active.get("ch2_group") == 1


def test_benzene_has_aromatic_ring() -> None:
    active = _active(label_functional_groups("c1ccccc1"))
    assert active.get("aromatic_ring") == 1
    # Бензол не должен распознаваться как простой алкен — это разные группы.
    assert active.get("alkene", 0) == 0


def test_acetic_acid_has_carbonyl_and_acid() -> None:
    active = _active(label_functional_groups("CC(=O)O"))
    assert active.get("carbonyl") == 1
    assert active.get("carboxylic_acid") == 1
    assert active.get("ch3_group") == 1


def test_nitromethane_has_nitro_group() -> None:
    # Проверяет, что top-level split SMARTS работает (см. _split_top_level).
    active = _active(label_functional_groups("C[N+](=O)[O-]"))
    assert active.get("nitro") == 1


def test_dmso_has_sulfoxide() -> None:
    active = _active(label_functional_groups("CS(C)=O"))
    assert active.get("sulfoxide_sulfone") == 1


def test_invalid_smiles_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid SMILES"):
        label_functional_groups("not-a-smiles")


def test_multi_hot_vector_shape_and_dtype() -> None:
    vec = multi_hot_labels("CCO")
    assert vec.shape == (25,)
    assert vec.dtype == np.int8
    # Должны быть установлены alcohol_OH, ch2_group, ch3_group.
    assert int(vec.sum()) == 3


def test_multi_hot_ordering_matches_group_names() -> None:
    vec = multi_hot_labels("c1ccccc1")
    aromatic_idx = GROUP_NAMES.index("aromatic_ring")
    assert vec[aromatic_idx] == 1
