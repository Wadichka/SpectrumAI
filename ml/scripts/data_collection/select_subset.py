"""Стратифицированный отбор соединений из каталога NistChemPy для фазы 2.

Каталог формируется ``download_datasets.py --source nist_chemdata``:
``ml/data/raw/nist/nist_compounds.csv`` (выгрузка ``nistchempy.get_all_data``).
Скрипт:

1. Загружает каталог, оставляет соединения с непустым полем ``IR Spectrum``.
2. Для каждого валидирует InChI → SMILES через RDKit, размечает функциональные
   группы (``pipelines.labeling.multi_hot_labels``).
3. Фильтрует кандидатов по наличию JDX-файлов на диске
   (шаблон ``{ID}_IR_*.jdx``).
4. Делает greedy round-robin sampling: для каждой из 25 групп добирает
   ``--target-per-group`` примеров, общий лимит — ``--target-size``.
5. Пишет CSV: ``nist_id``, ``inchi_key``, ``smiles``, ``target_groups``,
   ``jdx_path``.

Запуск:
    python -m ml.scripts.data_collection.select_subset \\
        [--target-size 2500] [--target-per-group 100] [--seed 42] \\
        [--input-catalog ml/data/raw/nist/nist_compounds.csv] \\
        [--jdx-dir       ml/data/raw/nist] \\
        [--output        ml/data/processed/predefense_subset_ids.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from rdkit import Chem
from rdkit.Chem.inchi import InchiToInchiKey, MolFromInchi
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_ROOT = _REPO_ROOT / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.labeling import GROUP_NAMES, N_GROUPS, multi_hot_labels  # noqa: E402

log = structlog.get_logger(__name__)


def _find_jdx_for_id(jdx_dir: Path, nist_id: str) -> Path | None:
    """Возвращает первый JDX-файл вида ``{nist_id}_IR_*.jdx`` или None."""
    matches = sorted(jdx_dir.glob(f"{nist_id}_IR_*.jdx"))
    return matches[0] if matches else None


def _compute_labels(inchi: str) -> tuple[str, str, np.ndarray] | None:
    """Возвращает (canonical_smiles, inchi_key, labels) или None при ошибке."""
    try:
        mol = MolFromInchi(inchi)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol)
        if Chem.MolFromSmiles(smiles) is None:
            return None
        labels = multi_hot_labels(smiles)
        inchi_key = InchiToInchiKey(inchi)
        return smiles, inchi_key, labels
    except (ValueError, RuntimeError):
        return None


def _stratified_sample(
    df: pd.DataFrame,
    *,
    total: int,
    target_per_group: int,
    seed: int,
) -> pd.DataFrame:
    """Greedy round-robin: добираем «бедную» группу, пока не сядем в лимит."""
    rng = np.random.default_rng(seed)
    indices = np.array(df.index.to_numpy(), copy=True)
    rng.shuffle(indices)
    df_shuffled = df.loc[indices].reset_index(drop=True)

    label_matrix = np.stack(df_shuffled["labels"].to_numpy())
    used = np.zeros(len(df_shuffled), dtype=bool)
    group_counts = np.zeros(N_GROUPS, dtype=int)

    candidates_by_group: list[list[int]] = [
        list(np.where(label_matrix[:, g] == 1)[0]) for g in range(N_GROUPS)
    ]
    cursors = [0] * N_GROUPS

    selected_rows: list[int] = []
    while len(selected_rows) < total:
        deficient = np.where(group_counts < target_per_group)[0]
        if deficient.size == 0:
            break
        target_group = int(deficient[np.argmin(group_counts[deficient])])

        idx_to_add = -1
        candidates = candidates_by_group[target_group]
        while cursors[target_group] < len(candidates):
            candidate = candidates[cursors[target_group]]
            cursors[target_group] += 1
            if not used[candidate]:
                idx_to_add = candidate
                break

        if idx_to_add < 0:
            group_counts[target_group] = max(group_counts[target_group], target_per_group)
            continue

        used[idx_to_add] = True
        selected_rows.append(idx_to_add)
        group_counts += label_matrix[idx_to_add]

    return df_shuffled.iloc[selected_rows].reset_index(drop=True)


def select_subset(
    *,
    input_catalog: Path,
    jdx_dir: Path,
    output_csv: Path,
    target_size: int = 2500,
    target_per_group: int = 100,
    seed: int = 42,
) -> dict[str, int]:
    """Полный пайплайн отбора. Возвращает статистику для логирования/тестов."""
    if not input_catalog.exists():
        raise FileNotFoundError(f"input_catalog не найден: {input_catalog}")
    if not jdx_dir.exists():
        raise FileNotFoundError(f"jdx_dir не найден: {jdx_dir}")

    df_catalog = pd.read_csv(input_catalog, dtype="str")
    required_cols = {"ID", "inchi"}
    missing = required_cols - set(df_catalog.columns)
    if missing:
        raise ValueError(
            f"в {input_catalog} отсутствуют колонки {missing}, "
            f"получено: {list(df_catalog.columns)}"
        )

    if "IR Spectrum" in df_catalog.columns:
        ir_mask = df_catalog["IR Spectrum"].notna() & (
            df_catalog["IR Spectrum"].astype(str).str.len() > 0
        )
        df_catalog = df_catalog.loc[ir_mask].reset_index(drop=True)
    log.info("catalog_loaded", rows=len(df_catalog), source=str(input_catalog))

    enriched: list[dict[str, object]] = []
    label_failures = 0
    missing_jdx = 0
    for _, row in tqdm(df_catalog.iterrows(), total=len(df_catalog), desc="label+filter"):
        inchi = str(row.get("inchi") or "")
        if not inchi.startswith("InChI="):
            label_failures += 1
            continue
        result = _compute_labels(inchi)
        if result is None:
            label_failures += 1
            continue
        smiles, inchi_key, labels = result
        if labels.sum() == 0:
            continue
        nist_id = str(row["ID"])
        jdx_path = _find_jdx_for_id(jdx_dir, nist_id)
        if jdx_path is None:
            missing_jdx += 1
            continue
        enriched.append(
            {
                "nist_id": nist_id,
                "smiles": smiles,
                "inchi_key": inchi_key,
                "jdx_path": str(jdx_path),
                "labels": labels,
            }
        )

    if not enriched:
        raise RuntimeError(
            "ни одно соединение не прошло фильтр — проверьте структуру "
            "nist_compounds.csv и наличие JDX-файлов"
        )

    df = pd.DataFrame(enriched)
    df = df.drop_duplicates(subset="inchi_key", keep="first").reset_index(drop=True)
    log.info(
        "candidates_built",
        candidates=len(df),
        label_failures=label_failures,
        missing_jdx=missing_jdx,
    )

    subset = _stratified_sample(
        df, total=target_size, target_per_group=target_per_group, seed=seed
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = subset[["nist_id", "inchi_key", "smiles", "jdx_path"]].copy()
    out["target_groups"] = subset["labels"].apply(
        lambda arr: ",".join(GROUP_NAMES[i] for i, v in enumerate(arr) if v == 1)
    )
    out = out[["nist_id", "inchi_key", "smiles", "target_groups", "jdx_path"]]
    out.to_csv(output_csv, index=False)

    label_matrix = np.stack(subset["labels"].to_numpy())
    coverage = {GROUP_NAMES[i]: int(label_matrix[:, i].sum()) for i in range(N_GROUPS)}
    log.info(
        "subset_written",
        output=str(output_csv),
        rows=len(subset),
        coverage=coverage,
    )
    return {
        "candidates": len(df),
        "selected": len(subset),
        "label_failures": label_failures,
        "missing_jdx": missing_jdx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-catalog",
        type=Path,
        default=_ML_ROOT / "data" / "raw" / "nist" / "nist_compounds.csv",
    )
    parser.add_argument(
        "--jdx-dir",
        type=Path,
        default=_ML_ROOT / "data" / "raw" / "nist",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ML_ROOT / "data" / "processed" / "predefense_subset_ids.csv",
    )
    parser.add_argument("--target-size", type=int, default=2500)
    parser.add_argument("--target-per-group", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    select_subset(
        input_catalog=args.input_catalog,
        jdx_dir=args.jdx_dir,
        output_csv=args.output,
        target_size=args.target_size,
        target_per_group=args.target_per_group,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
