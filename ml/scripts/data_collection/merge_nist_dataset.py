"""Сборка parquet'а из subset-IDs + каталога NistChemPy.

Используется после ``download_datasets.py --source nist_chemdata`` и
``select_subset.py``. На входе — CSV из ``select_subset`` (~2500 NIST ID) +
сами JDX-файлы вида ``{ID}_IR_{idx}.jdx`` в ``--jdx-dir`` +
``nist_compounds.csv`` (выгрузка ``nistchempy.get_all_data``).

Алгоритм:

1. Загружает subset-CSV.
2. Подгружает каталог NistChemPy (name, cas_rn, inchi, inchi_key).
3. Для каждого NIST ID:
   - Глобит ``{ID}_IR_*.jdx`` в ``--jdx-dir``;
   - Парсит каждый вариант через :func:`app.parsing.jcamp_parser.parse_jcamp`,
     выбирает с наибольшим n_points (proxy для разрешения);
   - Проверяет coverage 400–4000 см⁻¹ ≥ 80%, валидность InChI/SMILES.
4. Дедуплицирует по ``inchi_key``.
5. Пишет parquet: ``id``, ``inchi_key``, ``smiles``, ``spectrum_raw``,
   ``wavenumbers_raw``, ``name``, ``cas``, ``state`` (заполняется None —
   NistChemPy не отдаёт фазу в ``get_all_data``).

Запуск:
    python -m ml.scripts.data_collection.merge_nist_dataset \\
        [--ids       ml/data/processed/predefense_subset_ids.csv] \\
        [--catalog   ml/data/raw/nist/nist_compounds.csv] \\
        [--jdx-dir   ml/data/raw/nist] \\
        [--output    ml/data/processed/predefense_spectra.parquet] \\
        [--quarantine ml/data/quarantine]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from rdkit import Chem
from rdkit.Chem.inchi import InchiToInchiKey, MolFromInchi
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_ROOT = _REPO_ROOT / "ml"
_BACKEND_ROOT = _REPO_ROOT / "backend"
for path in (_ML_ROOT, _BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.domain.errors import DomainError  # noqa: E402
from app.parsing.jcamp_parser import parse_jcamp  # noqa: E402

log = structlog.get_logger(__name__)

_TARGET_MIN = 400.0
_TARGET_MAX = 4000.0
_COVERAGE_MIN_RATIO = 0.80


@dataclass(frozen=True)
class _RejectionReason:
    code: str
    message: str
    nist_id: str


def _coverage_ratio(wavenumbers: np.ndarray) -> float:
    if wavenumbers.size == 0:
        return 0.0
    src_min = float(wavenumbers.min())
    src_max = float(wavenumbers.max())
    overlap_min = max(src_min, _TARGET_MIN)
    overlap_max = min(src_max, _TARGET_MAX)
    overlap = max(0.0, overlap_max - overlap_min)
    return overlap / (_TARGET_MAX - _TARGET_MIN)


def _quarantine(quarantine_dir: Path, reason: _RejectionReason, source_path: Path) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{reason.nist_id}.{reason.code}.json"
    target.write_text(
        json.dumps(
            {
                "nist_id": reason.nist_id,
                "code": reason.code,
                "message": reason.message,
                "source": str(source_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_jdx_paths(
    nist_id: str,
    subset_path: Path | None,
    jdx_dir: Path,
) -> list[Path]:
    """Список JDX-файлов для NIST ID: один или несколько (разные индексы)."""
    paths: list[Path] = []
    if subset_path is not None and subset_path.exists():
        paths.append(subset_path)
    for candidate in sorted(jdx_dir.glob(f"{nist_id}_IR_*.jdx")):
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _pick_best_spectrum(paths: list[Path]) -> tuple[Path, Any] | None:
    """Парсит все варианты и возвращает (path, RawSpectrum) с максимальным n_points."""
    best: tuple[Path, Any] | None = None
    best_size = -1
    for path in paths:
        try:
            spectrum = parse_jcamp(path)
        except DomainError:
            continue
        size = int(spectrum.wavenumbers.size)
        if size > best_size:
            best = (path, spectrum)
            best_size = size
    return best


def merge_nist_dataset(
    *,
    ids_csv: Path,
    catalog_csv: Path,
    jdx_dir: Path,
    output_parquet: Path,
    quarantine_dir: Path,
) -> dict[str, int]:
    """Полный merge-пайплайн. Возвращает счётчики для логирования/тестов."""
    if not ids_csv.exists():
        raise FileNotFoundError(f"ids_csv не найден: {ids_csv}")
    if not catalog_csv.exists():
        raise FileNotFoundError(f"catalog_csv не найден: {catalog_csv}")

    subset = pd.read_csv(ids_csv)
    subset = subset.drop_duplicates(subset="nist_id", keep="first").reset_index(drop=True)
    log.info("subset_loaded", rows=len(subset))

    catalog = pd.read_csv(catalog_csv, dtype="str").set_index("ID", drop=False)
    log.info("catalog_loaded", rows=len(catalog))

    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {
        "total": len(subset),
        "valid": 0,
        "rejected": 0,
        "deduplicated": 0,
    }

    for _, entry in tqdm(subset.iterrows(), total=len(subset), desc="merge"):
        nist_id = str(entry["nist_id"])
        subset_path = Path(entry["jdx_path"]) if "jdx_path" in entry and pd.notna(
            entry.get("jdx_path")
        ) else None

        candidates = _resolve_jdx_paths(nist_id, subset_path, jdx_dir)
        if not candidates:
            counters["rejected"] += 1
            counters["jdx_not_found"] = counters.get("jdx_not_found", 0) + 1
            continue

        best = _pick_best_spectrum(candidates)
        if best is None:
            counters["rejected"] += 1
            counters["all_variants_failed_parse"] = counters.get("all_variants_failed_parse", 0) + 1
            _quarantine(
                quarantine_dir,
                _RejectionReason("parsing_error", "no variant parseable", nist_id),
                candidates[0],
            )
            continue
        jdx_path, spectrum = best

        coverage = _coverage_ratio(spectrum.wavenumbers)
        if coverage < _COVERAGE_MIN_RATIO:
            counters["rejected"] += 1
            counters["coverage_below_threshold"] = counters.get("coverage_below_threshold", 0) + 1
            _quarantine(
                quarantine_dir,
                _RejectionReason(
                    "coverage_below_threshold", f"coverage={coverage:.2f}", nist_id
                ),
                jdx_path,
            )
            continue

        catalog_row = catalog.loc[nist_id] if nist_id in catalog.index else None
        inchi = str(catalog_row["inchi"]) if catalog_row is not None else ""
        if not inchi.startswith("InChI="):
            counters["rejected"] += 1
            counters["missing_inchi_in_catalog"] = counters.get("missing_inchi_in_catalog", 0) + 1
            continue

        mol = MolFromInchi(inchi)
        if mol is None:
            counters["rejected"] += 1
            counters["invalid_inchi"] = counters.get("invalid_inchi", 0) + 1
            _quarantine(
                quarantine_dir,
                _RejectionReason("invalid_inchi", inchi[:50], nist_id),
                jdx_path,
            )
            continue
        smiles = Chem.MolToSmiles(mol)
        catalog_inchi_key = catalog_row.get("inchi_key") if catalog_row is not None else None
        inchi_key = str(catalog_inchi_key) if (
            catalog_inchi_key is not None and pd.notna(catalog_inchi_key)
        ) else InchiToInchiKey(inchi)

        rows.append(
            {
                "inchi_key": inchi_key,
                "smiles": smiles,
                "spectrum_raw": spectrum.intensities.astype(np.float32).tolist(),
                "wavenumbers_raw": spectrum.wavenumbers.astype(np.float32).tolist(),
                "name": catalog_row.get("name") if catalog_row is not None else None,
                "cas": catalog_row.get("cas_rn") if catalog_row is not None else None,
                "state": None,
                "nist_id": nist_id,
            }
        )
        counters["valid"] += 1

    if not rows:
        raise RuntimeError("merge_nist_dataset: ни один спектр не прошёл валидацию")

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset="inchi_key", keep="first").reset_index(drop=True)
    df.insert(0, "id", df.index.astype(np.int64))
    counters["deduplicated"] = before - len(df)
    counters["final"] = len(df)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    log.info(
        "merge_nist_dataset_done",
        output=str(output_parquet),
        **{k: v for k, v in counters.items() if v},
    )
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_subset_ids.csv",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=_ML_ROOT / "data" / "raw" / "nist" / "nist_compounds.csv",
    )
    parser.add_argument("--jdx-dir", type=Path, default=_ML_ROOT / "data" / "raw" / "nist")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_spectra.parquet",
    )
    parser.add_argument("--quarantine", type=Path, default=_ML_ROOT / "data" / "quarantine")
    args = parser.parse_args()

    stats = merge_nist_dataset(
        ids_csv=args.ids,
        catalog_csv=args.catalog,
        jdx_dir=args.jdx_dir,
        output_parquet=args.output,
        quarantine_dir=args.quarantine,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
