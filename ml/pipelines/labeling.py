"""Автоматическая разметка функциональных групп через RDKit (§5.4 главы 5).

Содержит фиксированный перечень 25 целевых функциональных групп с SMARTS-
паттернами (таблица 5.3) и функцию-разметчик ``label_functional_groups``,
возвращающую число вхождений каждой группы. Multi-hot вектор формируется
вызывающим кодом (см. :func:`multi_hot_labels`).

Список SMARTS — источник истины глава 5, таблица 5.3. Тот же список
сидируется в БД миграцией ``0001_initial.py``; здесь мы не зависим от БД,
чтобы ML-код был автономен.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import Mol


@dataclass(frozen=True, slots=True)
class FunctionalGroupDef:
    """Определение одной целевой функциональной группы."""

    code: str
    name: str
    smarts: str


FUNCTIONAL_GROUPS: tuple[FunctionalGroupDef, ...] = (
    FunctionalGroupDef("FG01", "alcohol_OH", "[OX2H][CX4]"),
    FunctionalGroupDef("FG02", "phenol_OH", "[OX2H][c]"),
    FunctionalGroupDef("FG03", "carbonyl", "[CX3]=[OX1]"),
    FunctionalGroupDef("FG04", "aldehyde", "[CX3H1](=O)[#6]"),
    FunctionalGroupDef("FG05", "ketone", "[#6][CX3](=O)[#6]"),
    FunctionalGroupDef("FG06", "carboxylic_acid", "[CX3](=O)[OX2H]"),
    FunctionalGroupDef("FG07", "ester", "[#6][CX3](=O)[OX2][#6]"),
    FunctionalGroupDef("FG08", "amide_primary", "[CX3](=O)[NX3H2]"),
    FunctionalGroupDef("FG09", "amide_secondary", "[CX3](=O)[NX3H1]"),
    FunctionalGroupDef("FG10", "amide_tertiary", "[CX3](=O)[NX3H0]"),
    FunctionalGroupDef("FG11", "amine_primary", "[NX3;H2;!$(NC=O)]"),
    FunctionalGroupDef("FG12", "amine_secondary", "[NX3;H1;!$(NC=O)]"),
    FunctionalGroupDef("FG13", "amine_tertiary", "[NX3;H0;!$(NC=O)]"),
    FunctionalGroupDef("FG14", "nitrile", "[CX2]#[NX1]"),
    FunctionalGroupDef("FG15", "nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    FunctionalGroupDef("FG16", "ether", "[OD2]([#6])[#6]"),
    FunctionalGroupDef("FG17", "alkene", "[CX3]=[CX3]"),
    FunctionalGroupDef("FG18", "alkyne", "[CX2]#[CX2]"),
    FunctionalGroupDef("FG19", "aromatic_ring", "c1ccccc1"),
    FunctionalGroupDef("FG20", "ch2_group", "[CH2]"),
    FunctionalGroupDef("FG21", "ch3_group", "[CH3]"),
    FunctionalGroupDef("FG22", "c_f_bond", "[CX4]F"),
    FunctionalGroupDef("FG23", "c_cl_bond", "[CX4]Cl"),
    FunctionalGroupDef("FG24", "sulfoxide_sulfone", "[#6][SX3](=O)[#6],[#6][SX4](=O)(=O)[#6]"),
    FunctionalGroupDef("FG25", "thiol_thioether", "[SX2H],[SX2]([#6])[#6]"),
)

GROUP_NAMES: tuple[str, ...] = tuple(g.name for g in FUNCTIONAL_GROUPS)
N_GROUPS: int = len(FUNCTIONAL_GROUPS)


def _parse_mol(smiles: str) -> Mol:
    """Парсит SMILES в RDKit-Mol или кидает ValueError."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return mol


def _split_top_level(smarts: str) -> list[str]:
    """Разбивает SMARTS по запятым верхнего уровня (вне ``[]`` и ``()``).

    SMARTS-альтернативы вида ``A,B`` разделяют независимые структурные паттерны,
    тогда как запятая внутри ``[...]`` (например, ``[$(...),$(...)]``) — атомная
    альтернатива и должна остаться как есть.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in smarts:
        if char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _count_matches(mol: Mol, smarts: str) -> int:
    """Возвращает число вхождений SMARTS-паттерна в молекулу."""
    total = 0
    for part in _split_top_level(smarts):
        patt = Chem.MolFromSmarts(part)
        if patt is None:
            continue
        total += len(mol.GetSubstructMatches(patt))
    return total


def label_functional_groups(smiles: str) -> dict[str, int]:
    """Возвращает словарь ``{group_name: count}`` для всех 25 групп.

    Если группа отсутствует, значение равно 0. SMARTS-паттерны с запятой
    рассматриваются как «или» (несколько вариантов).
    """
    mol = _parse_mol(smiles)
    return {group.name: _count_matches(mol, group.smarts) for group in FUNCTIONAL_GROUPS}


def multi_hot_labels(smiles: str) -> npt.NDArray[np.int8]:
    """Бинарный multi-hot вектор длины 25 (порядок — :data:`GROUP_NAMES`)."""
    counts = label_functional_groups(smiles)
    return np.fromiter(
        (1 if counts[name] > 0 else 0 for name in GROUP_NAMES),
        dtype=np.int8,
        count=N_GROUPS,
    )
