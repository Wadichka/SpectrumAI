"""Объединение и валидация сырых NIST-спектров в parquet (этап 18, фаза 2).

Берёт JCAMP-DX-файлы из ``ml/data/raw/nist/`` (результат
``download_datasets.py --source nist_scrape``) и сопровождающий
``inchi.txt`` (мапа CAS → InChI), формирует ``predefense_spectra.parquet``
со столбцами ``id``, ``inchi_key``, ``smiles``, ``spectrum_raw``,
``wavenumbers_raw``, ``name``, ``cas``. Дедуплицирует по InChI Key.
Битые записи отправляет в ``ml/data/quarantine/`` с json-логом причины.

Валидация (глава 5, §5.3):
- спектр непустой и его длина совпадает с длиной оси волновых чисел;
- диапазон 400–4000 см⁻¹ покрыт ≥ 80%;
- SMILES валиден через RDKit (InChI → Mol → SMILES → MolFromSmiles).

Запуск (из корня репо, .venv активирован):
    python -m ml.scripts.data_collection.merge_predefense \\
        [--raw-dir ml/data/raw/nist] \\
        [--inchi ml/data/raw/nist/inchi.txt] \\
        [--output ml/data/processed/predefense_spectra.parquet]
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_ROOT = _REPO_ROOT / "ml"
_BACKEND_ROOT = _REPO_ROOT / "backend"
for path in (_ML_ROOT, _BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.domain.errors import DomainError  # noqa: E402
from app.parsing.jcamp_parser import parse_jcamp  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem.inchi import InchiToInchiKey, MolFromInchi  # noqa: E402

log = structlog.get_logger(__name__)

# Покрытие целевого диапазона (400–4000 см⁻¹) хотя бы на 80% (см. §5.3).
_TARGET_MIN = 400.0
_TARGET_MAX = 4000.0
_COVERAGE_MIN_RATIO = 0.80


@dataclass(frozen=True)
class _RejectionReason:
    code: str
    message: str
    cas: str | None = None


def _load_inchi_map(inchi_path: Path) -> dict[str, str]:
    """Парсит ``inchi.txt`` candiy_spectrum'а: одна строка — CAS<TAB>InChI."""
    if not inchi_path.exists():
        log.warning("inchi_file_missing", path=str(inchi_path))
        return {}
    mapping: dict[str, str] = {}
    for raw in inchi_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        cas, inchi = parts[0].strip(), parts[1].strip()
        if cas and inchi.startswith("InChI="):
            mapping[cas] = inchi
    log.info("inchi_map_loaded", count=len(mapping), source=str(inchi_path))
    return mapping


def _coverage_ratio(wavenumbers: np.ndarray) -> float:
    """Доля целевого диапазона 400–4000 см⁻¹, покрытая источником."""
    if wavenumbers.size == 0:
        return 0.0
    src_min = float(wavenumbers.min())
    src_max = float(wavenumbers.max())
    overlap_min = max(src_min, _TARGET_MIN)
    overlap_max = min(src_max, _TARGET_MAX)
    overlap = max(0.0, overlap_max - overlap_min)
    return overlap / (_TARGET_MAX - _TARGET_MIN)


def _quarantine(quarantine_dir: Path, jdx_path: Path, reason: _RejectionReason) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{jdx_path.stem}.{reason.code}.json"
    target.write_text(
        json.dumps(
            {
                "file": str(jdx_path),
                "code": reason.code,
                "message": reason.message,
                "cas": reason.cas,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _process_jdx(jdx_path: Path, inchi_map: dict[str, str]) -> dict[str, Any] | _RejectionReason:
    cas = jdx_path.stem
    try:
        spectrum = parse_jcamp(jdx_path)
    except DomainError as exc:
        return _RejectionReason("parsing_error", str(exc), cas)

    coverage = _coverage_ratio(spectrum.wavenumbers)
    if coverage < _COVERAGE_MIN_RATIO:
        return _RejectionReason("coverage_below_threshold", f"coverage={coverage:.2f}", cas)

    inchi = inchi_map.get(cas)
    if inchi is None:
        return _RejectionReason("inchi_not_found", "InChI mapping missing", cas)

    mol = MolFromInchi(inchi)
    if mol is None:
        return _RejectionReason("invalid_inchi", f"RDKit rejected: {inchi[:50]}", cas)

    inchi_key = InchiToInchiKey(inchi)
    smiles = Chem.MolToSmiles(mol)
    # Двойная проверка: SMILES → Mol → не пуст.
    if Chem.MolFromSmiles(smiles) is None:
        return _RejectionReason("invalid_smiles_roundtrip", smiles, cas)

    return {
        "inchi_key": inchi_key,
        "smiles": smiles,
        "spectrum_raw": spectrum.intensities.astype(np.float32).tolist(),
        "wavenumbers_raw": spectrum.wavenumbers.astype(np.float32).tolist(),
        "name": spectrum.metadata.get("title"),
        "cas": cas,
    }


def merge_predefense(
    *,
    raw_dir: Path,
    inchi_path: Path,
    output_parquet: Path,
    quarantine_dir: Path,
) -> dict[str, int]:
    """Полный merge-пайплайн; возвращает статистику для логирования/тестов."""
    inchi_map = _load_inchi_map(inchi_path)
    jdx_paths = sorted(raw_dir.glob("*.jdx")) + sorted(raw_dir.glob("*.dx"))
    if not jdx_paths:
        raise RuntimeError(f"No JCAMP-DX files in {raw_dir}")

    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {"total": len(jdx_paths), "valid": 0, "rejected": 0}

    for jdx_path in jdx_paths:
        result = _process_jdx(jdx_path, inchi_map)
        if isinstance(result, _RejectionReason):
            _quarantine(quarantine_dir, jdx_path, result)
            counters[result.code] = counters.get(result.code, 0) + 1
            counters["rejected"] += 1
            continue
        rows.append(result)
        counters["valid"] += 1

    if not rows:
        raise RuntimeError("No valid spectra after validation; check quarantine.")

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
    parser.add_argument("--raw-dir", type=Path, default=_ML_ROOT / "data" / "raw" / "nist")
    parser.add_argument(
        "--inchi",
        type=Path,
        default=_ML_ROOT / "data" / "raw" / "nist" / "inchi.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_spectra.parquet",
    )
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=_ML_ROOT / "data" / "quarantine",
    )
    args = parser.parse_args()

    stats = merge_predefense(
        raw_dir=args.raw_dir,
        inchi_path=args.inchi,
        output_parquet=args.output,
        quarantine_dir=args.quarantine,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
