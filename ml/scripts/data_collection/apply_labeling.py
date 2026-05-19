"""Пакетная RDKit-разметка функциональных групп для предзащитного датасета.

Читает ``predefense_normalized.parquet`` (после предобработки), для каждого
SMILES вычисляет multi-hot метку через :func:`pipelines.labeling.multi_hot_labels`
и сохраняет ``predefense_labeled.parquet`` с дополнительным столбцом
``labels`` (np.array int8, длина = ``pipelines.labeling.N_GROUPS``).

Запуск:
    python -m ml.scripts.data_collection.apply_labeling \\
        [--input ml/data/processed/predefense_normalized.parquet] \\
        [--output ml/data/processed/predefense_labeled.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_ROOT = _REPO_ROOT / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.labeling import GROUP_NAMES, N_GROUPS, multi_hot_labels  # noqa: E402

log = structlog.get_logger(__name__)


def apply_labeling(input_parquet: Path, output_parquet: Path) -> int:
    df = pd.read_parquet(input_parquet)
    log.info("apply_labeling_start", rows=len(df), input=str(input_parquet))

    labels: list[list[int]] = []
    keep_mask: list[bool] = []
    failures = 0
    for smiles in tqdm(df["smiles"].tolist(), desc="labeling"):
        try:
            vector = multi_hot_labels(str(smiles))
        except ValueError as exc:
            log.warning("labeling_failed", smiles=smiles, reason=str(exc))
            labels.append([])
            keep_mask.append(False)
            failures += 1
            continue
        labels.append(vector.astype(np.int8).tolist())
        keep_mask.append(True)

    df = df.loc[keep_mask].reset_index(drop=True)
    df["labels"] = [lst for lst, keep in zip(labels, keep_mask, strict=True) if keep]

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    log.info(
        "apply_labeling_done",
        output=str(output_parquet),
        kept=len(df),
        failures=failures,
        n_groups=N_GROUPS,
        first_group=GROUP_NAMES[0],
    )
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_normalized.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_labeled.parquet",
    )
    args = parser.parse_args()
    apply_labeling(args.input, args.output)


if __name__ == "__main__":
    main()
