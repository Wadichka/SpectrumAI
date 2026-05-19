"""Сборка предзащитного parquet'а из subset-IDs + NistChemData CSV-каталога.

Используется после ``select_predefense_subset.py`` (фаза 2, этап 18). На
входе — CSV из ~2000 NIST ID + сами JDX-файлы из распакованного
``nist_IR.zip`` + CSV-каталоги соединений и метаданных от
``IvanChernyshov/NistChemData``.

Алгоритм:

1. Загружает ``predefense_subset_ids.csv`` (выход
   ``select_predefense_subset``).
2. Подгружает ``nist_compounds.csv`` (name, cas, inchi, inchi_key) и
   ``nist_ir_info.csv`` (physical state).
3. Для каждой строки subset:
   - Парсит JDX через :func:`app.parsing.jcamp_parser.parse_jcamp`;
   - Если для одного NIST ID несколько JDX (бывает с разными условиями
     регистрации) — берёт файл с наибольшим числом точек (proxy для
     разрешения).
   - Проверяет coverage 400–4000 см⁻¹ ≥ 80%, валидность InChI/SMILES.
4. Дедуплицирует по ``inchi_key`` (если разные NIST ID соответствуют
   одной молекуле — оставляет первое вхождение).
5. Пишет ``predefense_spectra.parquet``: ``id``, ``inchi_key``, ``smiles``,
   ``spectrum_raw``, ``wavenumbers_raw``, ``name``, ``cas``, ``state``
   (gas/liquid/solid/solution или null).

Запуск:
    python -m ml.scripts.data_collection.merge_predefense \\
        [--subset-ids ml/data/processed/predefense_subset_ids.csv] \\
        [--compounds-csv ml/data/raw/nist/nist_compounds.csv] \\
        [--ir-info-csv ml/data/raw/nist/nist_ir_info.csv] \\
        [--jdx-dir ml/data/raw/nist] \\
        [--output ml/data/processed/predefense_spectra.parquet] \\
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
    ir_info: pd.DataFrame | None,
    jdx_dir: Path,
) -> list[Path]:
    """Список JDX-файлов для NIST ID: один или несколько (разные условия)."""
    paths: list[Path] = []
    if subset_path is not None and subset_path.exists():
        paths.append(subset_path)
    if ir_info is not None and "file" in ir_info.columns:
        for value in ir_info.loc[ir_info["nist_id"] == nist_id, "file"].dropna().tolist():
            candidate = jdx_dir / str(value)
            if candidate.exists() and candidate not in paths:
                paths.append(candidate)
    if not paths:
        for fallback in (jdx_dir / f"{nist_id}.jdx", jdx_dir / f"IR_{nist_id}.jdx"):
            if fallback.exists():
                paths.append(fallback)
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


def _state_for(nist_id: str, ir_info: pd.DataFrame | None) -> str | None:
    if ir_info is None or "state" not in ir_info.columns:
        return None
    values = ir_info.loc[ir_info["nist_id"] == nist_id, "state"].dropna().tolist()
    return str(values[0]) if values else None


def merge_predefense(
    *,
    subset_csv: Path,
    compounds_csv: Path,
    ir_info_csv: Path | None,
    jdx_dir: Path,
    output_parquet: Path,
    quarantine_dir: Path,
) -> dict[str, int]:
    """Полный merge-пайплайн. Возвращает счётчики для логирования/тестов."""
    if not subset_csv.exists():
        raise FileNotFoundError(f"subset_csv не найден: {subset_csv}")
    if not compounds_csv.exists():
        raise FileNotFoundError(f"compounds_csv не найден: {compounds_csv}")

    subset = pd.read_csv(subset_csv)
    subset = subset.drop_duplicates(subset="nist_id", keep="first").reset_index(drop=True)
    log.info("subset_loaded", rows=len(subset))

    compounds = pd.read_csv(compounds_csv).set_index("nist_id", drop=False)
    ir_info: pd.DataFrame | None = None
    if ir_info_csv is not None and ir_info_csv.exists():
        ir_info = pd.read_csv(ir_info_csv)

    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {
        "total": len(subset),
        "valid": 0,
        "rejected": 0,
        "deduplicated": 0,
    }

    for _, entry in tqdm(subset.iterrows(), total=len(subset), desc="merge"):
        nist_id = str(entry["nist_id"])
        subset_path = Path(entry["jdx_path"]) if "jdx_path" in entry else None

        candidates = _resolve_jdx_paths(nist_id, subset_path, ir_info, jdx_dir)
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

        compound_row = compounds.loc[nist_id] if nist_id in compounds.index else None
        inchi = str(compound_row["inchi"]) if compound_row is not None else ""
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
        inchi_key = str(compound_row.get("inchi_key") or InchiToInchiKey(inchi))

        rows.append(
            {
                "inchi_key": inchi_key,
                "smiles": smiles,
                "spectrum_raw": spectrum.intensities.astype(np.float32).tolist(),
                "wavenumbers_raw": spectrum.wavenumbers.astype(np.float32).tolist(),
                "name": compound_row.get("name") if compound_row is not None else None,
                "cas": compound_row.get("cas") if compound_row is not None else None,
                "state": _state_for(nist_id, ir_info),
                "nist_id": nist_id,
            }
        )
        counters["valid"] += 1

    if not rows:
        raise RuntimeError("merge_predefense: ни один спектр не прошёл валидацию")

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset="inchi_key", keep="first").reset_index(drop=True)
    df.insert(0, "id", df.index.astype(np.int64))
    counters["deduplicated"] = before - len(df)
    counters["final"] = len(df)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    log.info(
        "merge_predefense_done",
        output=str(output_parquet),
        **{k: v for k, v in counters.items() if v},
    )
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-ids",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_subset_ids.csv",
    )
    parser.add_argument(
        "--compounds-csv",
        type=Path,
        default=_ML_ROOT / "data" / "raw" / "nist" / "nist_compounds.csv",
    )
    parser.add_argument(
        "--ir-info-csv",
        type=Path,
        default=_ML_ROOT / "data" / "raw" / "nist" / "nist_ir_info.csv",
    )
    parser.add_argument("--jdx-dir", type=Path, default=_ML_ROOT / "data" / "raw" / "nist")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_spectra.parquet",
    )
    parser.add_argument("--quarantine", type=Path, default=_ML_ROOT / "data" / "quarantine")
    args = parser.parse_args()

    stats = merge_predefense(
        subset_csv=args.subset_ids,
        compounds_csv=args.compounds_csv,
        ir_info_csv=args.ir_info_csv if args.ir_info_csv.exists() else None,
        jdx_dir=args.jdx_dir,
        output_parquet=args.output,
        quarantine_dir=args.quarantine,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
