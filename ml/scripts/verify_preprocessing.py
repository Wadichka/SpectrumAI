"""Сверка препроцессинга: parquet vs свежий прогон из JDX.

Берёт 10 случайных строк из ``ml/data/processed/predefense_normalized.parquet``,
находит для каждой сырой JDX в ``ml/data/raw/nist/{nist_id}_IR_*.jdx``,
прогоняет через ``backend.app.preprocessing.preprocess`` и сравнивает с
сохранённым в parquet ``spectrum``.

Если хотя бы для одного спектра ``max_abs_diff > 1e-3`` — препроцессинг
расходится между обучением и инференсом. Это блокирующий баг (модель
видела одни значения, инференс отдаёт другие → плохой top-K).

Запуск:
    python ml/scripts/verify_preprocessing.py
    python ml/scripts/verify_preprocessing.py --n 20 --seed 7

Exit code 0 — препроцессинг идентичен. Exit code 1 — дрейф найден.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "backend"
_ML_ROOT = _REPO_ROOT / "ml"
for path in (_BACKEND_ROOT, _ML_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.domain.errors import DomainError  # noqa: E402
from app.parsing.jcamp_parser import parse_jcamp  # noqa: E402
from app.parsing.models import RawSpectrum  # noqa: E402
from app.preprocessing.config import PreprocessConfig  # noqa: E402
from app.preprocessing.pipeline import preprocess  # noqa: E402

_PARQUET = _ML_ROOT / "data" / "processed" / "predefense_normalized.parquet"
_RAW_DIR = _ML_ROOT / "data" / "raw" / "nist"
_DRIFT_THRESHOLD = 1e-3


def _pick_best_jdx(nist_id: str) -> tuple[Path, RawSpectrum] | None:
    """Эмулирует ``merge_nist_dataset._pick_best_spectrum`` — выбирает JDX с
    максимальным числом точек среди всех ``{nist_id}_IR_*.jdx``.

    Без этой эмуляции скрипт сравнивает не тот спектр, что попал в parquet
    (для compound'ов с несколькими IR — например, C113382 имеет _IR_0 на 908
    точек и _IR_1 на 3333 точки, merge берёт второй).
    """
    best: tuple[Path, RawSpectrum] | None = None
    best_size = -1
    for path in sorted(_RAW_DIR.glob(f"{nist_id}_IR_*.jdx")):
        try:
            raw = parse_jcamp(path)
        except DomainError:
            continue
        size = int(raw.wavenumbers.size)
        if size > best_size:
            best = (path, raw)
            best_size = size
    return best


def _diff(parquet_spectrum: list[float], fresh: np.ndarray) -> tuple[float, float]:
    a = np.asarray(parquet_spectrum, dtype=np.float64)
    b = fresh.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch parquet={a.shape} fresh={b.shape}")
    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def verify(n_samples: int, seed: int) -> int:
    if not _PARQUET.exists():
        print(f"[FATAL] parquet not found: {_PARQUET}")
        return 1
    df = pd.read_parquet(_PARQUET)
    if df.empty:
        print(f"[FATAL] parquet empty: {_PARQUET}")
        return 1

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)

    cfg = PreprocessConfig()
    rows: list[dict] = []
    drift_found = False
    skipped = 0

    print(f"sampling {len(indices)} rows from {len(df)} (seed={seed})")
    print(f"jdx dir: {_RAW_DIR}")
    print(f"config:  {cfg.model_dump()}")
    print("-" * 90)
    print(f"{'idx':>5} {'nist_id':>11} {'max_abs_diff':>15} {'mean_abs_diff':>15} {'verdict':>10}")
    print("-" * 90)

    for idx in indices:
        row = df.iloc[int(idx)]
        nist_id = str(row["nist_id"])
        best = _pick_best_jdx(nist_id)
        if best is None:
            print(f"{idx:>5} {nist_id:>11} {'':>15} {'':>15} {'NO_JDX':>10}")
            skipped += 1
            continue
        jdx, raw = best
        try:
            processed = preprocess(raw, cfg)
        except Exception as exc:
            print(f"{idx:>5} {nist_id:>11} {'':>15} {'':>15} {'PARSE_FAIL':>10}  ({exc})")
            skipped += 1
            continue

        max_d, mean_d = _diff(row["spectrum"], processed.intensities)
        verdict = "OK" if max_d <= _DRIFT_THRESHOLD else "DRIFT"
        if verdict == "DRIFT":
            drift_found = True
        rows.append({"nist_id": nist_id, "max_d": max_d, "mean_d": mean_d, "verdict": verdict})
        print(f"{idx:>5} {nist_id:>11} {max_d:15.6e} {mean_d:15.6e} {verdict:>10}")

    print("-" * 90)
    if not rows:
        print("[FATAL] no rows successfully compared (all skipped)")
        return 1
    overall_max = max(r["max_d"] for r in rows)
    overall_mean = float(np.mean([r["mean_d"] for r in rows]))
    print(f"overall: max_abs_diff = {overall_max:.6e}, mean_abs_diff = {overall_mean:.6e}")
    print(f"compared {len(rows)} rows, skipped {skipped}")

    if drift_found:
        print()
        print("[FAIL] PREPROCESSING DRIFT DETECTED")
        print(f"  threshold = {_DRIFT_THRESHOLD:.0e}, max observed = {overall_max:.3e}")
        print("  → train (parquet) и inference (свежий preprocess) расходятся.")
        print("  → модели в models/checkpoints/*-predefense-0.5.0/ обучались на")
        print("    другой версии спектров → реальные предсказания смещены.")
        return 1
    print()
    print("[OK] препроцессинг идентичен в train и inference.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="число случайных спектров")
    parser.add_argument("--seed", type=int, default=42, help="seed для случайной выборки")
    args = parser.parse_args()
    return verify(args.n, args.seed)


if __name__ == "__main__":
    sys.exit(main())
