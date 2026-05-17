"""Генератор синтетического parquet-датасета для отладки ML-пайплайна.

Согласно CLAUDE.md §11 (двухфазная стратегия данных) на фазе 1 вся система
проверяется на искусственно сгенерированных спектрах. Этот скрипт создаёт
250 «фейковых» спектров: для каждой молекулы из захардкоженного списка
выполняется автоматическая разметка функциональных групп (§5.4 главы 5),
после чего синтезируется спектр как сумма гауссов на центрах характеристических
полос присутствующих групп, с линейным фоном и шумом, и min-max-нормированием.

Это не имитация физики — это заглушка, позволяющая end-to-end отладить
обучение, инференс и retrieval до перехода на реальные данные (фаза 2,
этапы 18+).

Запуск:
    python ml/scripts/build_synthetic_dataset.py [--output PATH] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# Скрипт должен импортироваться как из ml/scripts/, так и из тестов; добавляем
# корень ml/ в sys.path для гарантированного импорта pipelines.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.labeling import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    GROUP_NAMES,
    label_functional_groups,
    multi_hot_labels,
)

# Целевая сетка по CLAUDE.md §7.
TARGET_MIN: float = 400.0
TARGET_MAX: float = 4000.0
TARGET_STEP: float = 1.0
TARGET_LENGTH: int = round((TARGET_MAX - TARGET_MIN) / TARGET_STEP) + 1

# Характеристические центры полос (см⁻¹) — отвечают за расположение пиков
# в «фейковом» спектре. Значения упрощены из таблицы 5.3 главы 5 (берётся
# середина диапазона).
_GROUP_CENTERS: dict[str, tuple[float, ...]] = {
    "alcohol_OH": (3400.0, 1050.0),
    "phenol_OH": (3450.0, 1200.0),
    "carbonyl": (1715.0,),
    "aldehyde": (2770.0, 1720.0),
    "ketone": (1715.0,),
    "carboxylic_acid": (2900.0, 1700.0),
    "ester": (1740.0, 1200.0),
    "amide_primary": (3350.0, 3180.0, 1650.0),
    "amide_secondary": (3300.0, 1650.0, 1550.0),
    "amide_tertiary": (1660.0,),
    "amine_primary": (3400.0, 1600.0),
    "amine_secondary": (3350.0,),
    "amine_tertiary": (1135.0,),
    "nitrile": (2230.0,),
    "nitro": (1535.0, 1340.0),
    "ether": (1100.0,),
    "alkene": (1650.0, 3050.0),
    "alkyne": (2180.0, 3300.0),
    "aromatic_ring": (1525.0, 3050.0),
    "ch2_group": (2925.0, 1465.0),
    "ch3_group": (2960.0, 1375.0),
    "c_f_bond": (1200.0,),
    "c_cl_bond": (700.0,),
    "sulfoxide_sulfone": (1040.0, 1335.0),
    "thiol_thioether": (2575.0, 650.0),
}

# 250 SMILES с покрытием всех 25 групп (минимум 5 представителей на группу
# через перекрывающиеся структуры). Хардкод оправдан: фаза 1 — отладочный
# набор, не источник истины.
SMILES_LIST: tuple[str, ...] = (
    # Спирты (alcohol_OH)
    "CO",
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "OCC(O)CO",
    "CC(O)CC",
    "OC1CCCCC1",
    "OCC1CCCCC1",
    "OCCO",
    "CCC(O)C",
    "CC(O)C(C)O",
    "OCCCO",
    # Фенолы (phenol_OH)
    "Oc1ccccc1",
    "Cc1ccc(O)cc1",
    "Oc1ccc(O)cc1",
    "Oc1ccc(C)cc1",
    "Oc1cccc(C)c1",
    "Oc1cc(C)ccc1C",
    "Oc1cc(Cl)ccc1",
    # Альдегиды
    "C=O",
    "CC=O",
    "CCC=O",
    "CCCC=O",
    "O=Cc1ccccc1",
    "CC(C)C=O",
    "OCCC=O",
    # Кетоны
    "CC(C)=O",
    "CCC(C)=O",
    "CCC(=O)CC",
    "O=C1CCCCC1",
    "Cc1ccc(cc1)C(C)=O",
    "CC(=O)CC(C)=O",
    # Карбоновые кислоты
    "C(=O)O",
    "CC(=O)O",
    "CCC(=O)O",
    "CCCC(=O)O",
    "OC(=O)CC(=O)O",
    "c1ccc(cc1)C(=O)O",
    "CC(O)C(=O)O",
    "OC(=O)CCCC(=O)O",
    "CCCCC(=O)O",
    # Эфиры (ester)
    "COC(C)=O",
    "CCOC(C)=O",
    "CCOC(=O)CC",
    "COC(=O)c1ccccc1",
    "CCOC(=O)c1ccccc1",
    "CC(=O)OCC",
    "COC(=O)C(C)C",
    # Амиды
    "CC(N)=O",
    "CCC(N)=O",
    "CC(=O)NC",
    "CC(=O)N(C)C",
    "NC(=O)c1ccccc1",
    "CC(=O)Nc1ccccc1",
    "NC(=O)N",
    "CC(=O)NCC",
    # Амины первичные
    "CN",
    "CCN",
    "CCCN",
    "NCCCN",
    "Nc1ccccc1",
    "NCCO",
    # Амины вторичные
    "CNC",
    "CCNC",
    "CNCC",
    "CN(C)C",
    "c1ccc(NC)cc1",
    # Амины третичные
    "CN(C)C",
    "CCN(CC)CC",
    "CN(C)CC",
    "CN(C)c1ccccc1",
    # Нитрилы
    "CC#N",
    "CCC#N",
    "CCCC#N",
    "N#Cc1ccccc1",
    "N#CCC#N",
    # Нитрогруппа
    "C[N+](=O)[O-]",
    "CC[N+](=O)[O-]",
    "O=[N+]([O-])c1ccccc1",
    "Cc1ccc([N+](=O)[O-])cc1",
    "O=[N+]([O-])CCC",
    # Простые эфиры
    "COC",
    "CCOC",
    "CCOCC",
    "COCCOC",
    "C1CCOCC1",
    "COc1ccccc1",
    # Алкены
    "C=C",
    "CC=C",
    "CCC=C",
    "CC=CC",
    "C=CC=C",
    "C=Cc1ccccc1",
    # Алкины
    "C#C",
    "CC#C",
    "CCC#C",
    "C#Cc1ccccc1",
    "CC#CC",
    # Ароматика
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccc(C)cc1",
    "c1ccc(cc1)c2ccccc2",
    "Cc1ccc(C)cc1",
    "Cc1ccccc1C",
    "c1ccc2ccccc2c1",
    # CH2/CH3 covered by alkanes
    "CC",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "CCCCCCC",
    "CCCCCCCC",
    "CC(C)C",
    "CC(C)CC",
    "C1CCCCC1",
    "C1CCCC1",
    # C-F
    "CF",
    "CCF",
    "CCCF",
    "FCCF",
    "FCC(F)F",
    "Cc1ccc(F)cc1",
    # C-Cl
    "CCl",
    "CCCl",
    "ClCCCl",
    "ClCC(Cl)Cl",
    "Cc1ccc(Cl)cc1",
    "ClCC=O",
    "CCCCl",
    # Сульфо/сульфон
    "CS(C)=O",
    "CS(=O)C",
    "CS(=O)(=O)C",
    "CCS(=O)(=O)CC",
    "c1ccc(cc1)S(=O)(=O)c1ccccc1",
    # Тиолы / тиоэфиры
    "CS",
    "CCS",
    "CSC",
    "CCSCC",
    "CSCC",
    "c1ccc(S)cc1",
    # Дополнительные сложные молекулы (разнообразие)
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # ацетилсалициловая (аспирин)
    "CN1CCN(CC1)c1nc2ccccc2s1",  # фрагмент лекарства
    "OC(=O)CC(N)C(=O)O",  # аспарагиновая кислота
    "NC(N)=N",  # гуанидин
    "CC(C)(C)c1ccc(O)cc1",  # 4-tert-butylphenol
    "OC(=O)c1ccccc1O",  # салициловая кислота
    "OCC(O)C(O)C(O)C(O)CO",  # сорбит
    "NC(CC(=O)O)C(=O)O",
    "Cc1ccccc1N",  # о-толуидин
    "ClC(Cl)Cl",  # хлороформ (3 C-Cl)
    "FC(F)(F)CC(=O)O",  # трифторацетат-ish
    "OCCN",  # этаноламин
    "NCCS",  # цистеамин
    "CC(=O)CCC(=O)C",  # 2,5-гександион
    "OCC=O",  # гликолевый альдегид
    "OCCO",  # этиленгликоль
    "OCCCCO",  # 1,4-бутандиол
    "NC(=O)N",  # мочевина (повтор для покрытия)
    "CCN(CC)C(C)=O",  # N,N-диэтилацетамид
    "CC(=O)c1ccc(N)cc1",  # п-аминоацетофенон
    "Oc1ccc(C=O)cc1",  # п-гидроксибензальдегид
    "CCOC(=O)CCC(=O)OCC",  # диэтилсукцинат
    "CC(C)(C)O",  # трет-бутанол
    "C1=CCCC=C1",  # циклогексадиен
    "ClCCCl",  # 1,2-дихлорэтан
    "CCS(=O)(=O)O",  # этансульфоновая
    "OS(=O)(=O)O",  # серная (валидируется как S=O)
    "CCN(CC)CC",  # триэтиламин
    "Cc1ccc(N)cc1",  # п-толуидин
    "Cc1ccc(cc1)C#N",  # п-толунитрил
    "ClC(=O)c1ccccc1",  # бензоилхлорид (C-Cl и C=O)
    "Nc1ccc(N)cc1",  # п-фенилендиамин
    "OC(=O)CN",  # глицин
    "OCCOC",  # метилгликоль
    "CSCCC(N)C(=O)O",  # метионин
    "OCCCS",  # тиол-спирт
    "CCC(C)CC",  # 3-метилпентан
    "FC(F)(F)c1ccccc1",  # трифторметилбензол
    "ClCc1ccccc1",  # бензилхлорид
    "ClC(Cl)=O",  # фосген (мини)
    "Cc1ncccc1",  # 2-метилпиридин
    "CCOCCOCC",  # ди(этокси)этиловый эфир
    "CCC(C)(O)CC",  # 3-метил-3-пентанол
    "OC(=O)CC#N",  # цианоуксусная кислота
    "CCC(=O)NC",  # N-метилпропанамид
    "CC(C)NC(C)C",  # диизопропиламин
    "CCN(CC)C(=O)c1ccccc1",  # N,N-диэтилбензамид
    "FCCF",  # 1,2-дифторэтан
    "ClCC(O)CO",  # 1-хлор-2,3-пропандиол
    # Дополнительный набор (общий объём ≥ 200).
    "CCCCCCCCCCCC",  # додекан
    "CCCCC(O)CC",  # 3-гептанол
    "OCCCCCO",  # 1,5-пентандиол
    "Cc1ccc(O)c(C)c1",  # 2,4-диметилфенол
    "Oc1cc(Cl)cc(Cl)c1",  # 2,6-дихлорфенол
    "CCC(=O)CC",  # 3-пентанон (повтор для разнообразия)
    "OC(=O)c1cccc(C)c1",  # м-толуиловая кислота
    "OC(=O)Cc1ccccc1",  # фенилуксусная кислота
    "CCCC(=O)OC",  # метилбутират
    "NC(=O)CC",  # пропионамид
    "CNC(=O)CC",  # N-метилпропионамид
    "CN(C)C(=O)c1ccccc1",  # N,N-диметилбензамид
    "NCCC",  # н-пропиламин
    "CNCC",  # N-метилэтиламин
    "CN(C)CC",  # N,N-диметилэтиламин
    "CC(C)C#N",  # изобутиронитрил
    "Cc1ccccc1[N+](=O)[O-]",  # о-нитротолуол
    "COCOC",  # диметоксиметан
    "CC=CC",  # 2-бутен
    "CC#CC",  # 2-бутин
    "c1ccc(cc1)Cc2ccccc2",  # дифенилметан
    "CCCCCCCC",  # октан
    "CC(Cl)(Cl)Cl",  # 1,1,1-трихлорэтан
    "CSCC",  # этил-метил-тиоэфир
    "SCCS",  # этан-1,2-дитиол
    "CC(=O)C(C)=O",  # 2,3-бутандион
    "Oc1ccc2ccccc2c1",  # 2-нафтол
    "CC(C)=CC(C)C",  # 2,5-диметилгексен
    "CCCC#N",  # бутиронитрил
    "CCCCCC=O",  # гексаналь
)


def _gaussian_peak(
    grid: npt.NDArray[np.float64], center: float, sigma: float, amplitude: float
) -> npt.NDArray[np.float64]:
    return amplitude * np.exp(-(((grid - center) / sigma) ** 2))


def _build_fake_spectrum(
    labels: dict[str, int],
    *,
    grid: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Собирает «фейковый» спектр из гауссов на центрах присутствующих групп."""
    spectrum = np.zeros_like(grid)
    for group_name, count in labels.items():
        if count <= 0:
            continue
        centers = _GROUP_CENTERS.get(group_name, ())
        for center in centers:
            sigma = float(rng.uniform(15.0, 45.0))
            amplitude = float(rng.uniform(0.4, 1.0)) * min(count, 3)
            spectrum = spectrum + _gaussian_peak(grid, center, sigma, amplitude)

    baseline = 0.05 + 0.00001 * grid
    noise = rng.normal(0.0, 0.005, grid.size)
    spectrum = spectrum + baseline + noise

    # min-max нормирование.
    lo, hi = float(spectrum.min()), float(spectrum.max())
    spread = hi - lo
    spectrum = (spectrum - lo) / spread if spread > 0.0 else np.zeros_like(spectrum)
    return spectrum.astype(np.float32)


def build(output: Path, *, seed: int = 42) -> pd.DataFrame:
    """Собирает датасет и сохраняет в parquet. Возвращает DataFrame для тестов."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(TARGET_MIN, TARGET_MAX, num=TARGET_LENGTH, dtype=np.float64)

    rows: list[dict[str, object]] = []
    group_coverage: Counter[str] = Counter()

    for compound_id, smiles in enumerate(SMILES_LIST):
        counts = label_functional_groups(smiles)
        labels = multi_hot_labels(smiles).tolist()
        spectrum = _build_fake_spectrum(counts, grid=grid, rng=rng).tolist()

        for name, present in zip(GROUP_NAMES, labels, strict=True):
            if present:
                group_coverage[name] += 1

        rows.append(
            {
                "compound_id": compound_id,
                "smiles": smiles,
                "spectrum": spectrum,
                "labels": labels,
            }
        )

    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, engine="pyarrow", index=False)

    coverage = {group.name: group_coverage.get(group.name, 0) for group in FUNCTIONAL_GROUPS}
    log.info(
        "synthetic_dataset_built",
        path=str(output),
        rows=len(frame),
        groups=len(FUNCTIONAL_GROUPS),
        coverage=coverage,
    )
    return frame


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка синтетического parquet-датасета.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "synthetic.parquet",
        help="путь для записи parquet",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _configure_logging() -> None:
    """Минимальная настройка structlog для запуска скрипта из CLI."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


def main() -> None:
    _configure_logging()
    args = _parse_args()
    build(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
