"""Статистика по предзащитному датасету.

Читает ``predefense_labeled.parquet`` (после разметки), формирует
``predefense_stats.json``: количество спектров, распределение по 25
функциональным группам, длины исходных спектров (из ``wavenumbers`` уже
3601 для всех — берём из логов merge'а), предупреждения о редких классах
(< 10 примеров).

Запуск:
    python -m ml.scripts.data_collection.compute_stats \\
        [--input ml/data/processed/predefense_labeled.parquet] \\
        [--output ml/data/processed/predefense_stats.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_ROOT = _REPO_ROOT / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.labeling import GROUP_NAMES  # noqa: E402

log = structlog.get_logger(__name__)

_RARE_CLASS_THRESHOLD = 10


def compute_stats(input_parquet: Path, output_json: Path) -> dict[str, object]:
    df = pd.read_parquet(input_parquet)
    if df.empty:
        raise RuntimeError(f"empty parquet: {input_parquet}")

    labels = np.asarray(df["labels"].tolist(), dtype=np.int8)
    positives_per_class = labels.sum(axis=0)
    counts: dict[str, int] = {
        name: int(positives_per_class[i]) for i, name in enumerate(GROUP_NAMES)
    }
    rare_classes = [name for name, count in counts.items() if count < _RARE_CLASS_THRESHOLD]
    multi_label_density = float(labels.sum(axis=1).mean())

    stats: dict[str, object] = {
        "total_spectra": len(df),
        "spectrum_length": len(df.iloc[0]["spectrum"]),
        "n_groups": int(labels.shape[1]),
        "positives_per_class": counts,
        "rare_classes": rare_classes,
        "rare_class_threshold": _RARE_CLASS_THRESHOLD,
        "multi_label_density_mean": multi_label_density,
        "labels_per_spectrum_distribution": {
            "min": int(labels.sum(axis=1).min()),
            "max": int(labels.sum(axis=1).max()),
            "median": int(np.median(labels.sum(axis=1))),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "compute_stats_done",
        output=str(output_json),
        total=stats["total_spectra"],
        rare_classes=len(rare_classes),
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_labeled.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_stats.json",
    )
    args = parser.parse_args()
    stats = compute_stats(args.input, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
