"""Пакетная предобработка сырого parquet'а через backend/app/preprocessing.

Читает ``predefense_spectra.parquet`` (результат merge_nist_dataset), прогоняет
каждый ряд через :func:`app.preprocessing.pipeline.preprocess` и сохраняет
``predefense_normalized.parquet`` со столбцом ``spectrum`` фиксированной длины
3601 и ``wavenumbers`` (целевая сетка). Метаданные оригинального ряда
(``inchi_key``, ``smiles``, ``name``, ``cas``) сохраняются как есть.

Запуск:
    python -m ml.scripts.data_collection.apply_preprocessing \\
        [--input ml/data/processed/predefense_spectra.parquet] \\
        [--output ml/data/processed/predefense_normalized.parquet]
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
_BACKEND_ROOT = _REPO_ROOT / "backend"
for path in (_ML_ROOT, _BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.domain.errors import DomainError  # noqa: E402
from app.parsing.models import RawSpectrum  # noqa: E402
from app.preprocessing.config import PreprocessConfig  # noqa: E402
from app.preprocessing.pipeline import preprocess  # noqa: E402

log = structlog.get_logger(__name__)


def _row_to_processed(row: pd.Series, config: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    raw = RawSpectrum(
        wavenumbers=np.asarray(row["wavenumbers_raw"], dtype=np.float64),
        intensities=np.asarray(row["spectrum_raw"], dtype=np.float64),
        metadata={"format": "JCAMP-DX", "source": "NIST"},
    )
    processed = preprocess(raw, config)
    return processed.wavenumbers, processed.intensities


def apply_preprocessing(input_parquet: Path, output_parquet: Path) -> int:
    """Прогоняет предобработку и возвращает число успешных рядов."""
    df = pd.read_parquet(input_parquet)
    log.info("apply_preprocessing_start", rows=len(df), input=str(input_parquet))

    config = PreprocessConfig()
    grid: np.ndarray | None = None
    spectra: list[list[float]] = []
    keep_mask: list[bool] = []
    failures = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="preprocess"):
        try:
            wn, intensities = _row_to_processed(row, config)
            if grid is None:
                grid = wn.astype(np.float32)
            spectra.append(intensities.astype(np.float32).tolist())
            keep_mask.append(True)
        except (DomainError, ValueError) as exc:
            log.warning("preprocess_failed", inchi_key=row.get("inchi_key"), reason=str(exc))
            spectra.append([])
            keep_mask.append(False)
            failures += 1

    df = df.loc[keep_mask].reset_index(drop=True)
    df["spectrum"] = [s for s, keep in zip(spectra, keep_mask, strict=True) if keep]
    if grid is None:
        raise RuntimeError("No spectra were processed successfully.")
    df["wavenumbers"] = [grid.tolist()] * len(df)
    df = df.drop(columns=["spectrum_raw", "wavenumbers_raw"])

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    log.info(
        "apply_preprocessing_done",
        output=str(output_parquet),
        kept=len(df),
        failures=failures,
        spectrum_length=int(grid.size),
    )
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_spectra.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_normalized.parquet",
    )
    args = parser.parse_args()
    apply_preprocessing(args.input, args.output)


if __name__ == "__main__":
    main()
