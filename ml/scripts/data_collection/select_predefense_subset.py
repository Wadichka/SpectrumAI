"""Стратифицированный отбор соединений для предзащитного датасета (фаза 2, этап 18).

Берёт каталог ``nist_compounds.csv`` из NistChemData (распакован командой
``download_datasets.py --source nist_chemdata``) и выбирает ~2000 соединений,
балансируя представленность 25 функциональных групп. Для каждой группы
целевая квота — ``--target-per-group`` (по умолчанию 100); общий лимит —
``--total`` (по умолчанию 2000).

Использует RDKit (``MolFromInchi`` → ``MolToSmiles``) и
``pipelines.labeling.multi_hot_labels``, чтобы определить, какие группы
присутствуют в каждом соединении. Отбирает только те NIST ID, для которых
JDX-файл реально существует в ``--jdx-dir`` (после `nist_IR.zip` extract).

Запуск:
    python -m ml.scripts.data_collection.select_predefense_subset \\
        [--compounds-csv ml/data/raw/nist/nist_compounds.csv] \\
        [--ir-info-csv  ml/data/raw/nist/nist_ir_info.csv] \\
        [--jdx-dir      ml/data/raw/nist] \\
        [--output       ml/data/processed/predefense_subset_ids.csv] \\
        [--total 2000] [--target-per-group 100] [--seed 42]
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


def _find_jdx_for_id(jdx_dir: Path, nist_id: str, ir_info: pd.DataFrame | None) -> Path | None:
    """Возвращает путь к JDX-файлу для данного NIST ID или None.

    Поиск идёт по нескольким именам, чтобы быть устойчивыми к небольшим
    различиям в схеме именования NistChemData (точный формат уточняется
    при первом прогоне на реальных данных).
    """
    candidates: list[Path] = []
    if ir_info is not None and "file" in ir_info.columns:
        match = ir_info[ir_info["nist_id"] == nist_id]
        if not match.empty:
            for value in match["file"].dropna().tolist():
                candidates.append(jdx_dir / str(value))
    candidates.extend(
        [
            jdx_dir / f"{nist_id}.jdx",
            jdx_dir / f"{nist_id}.dx",
            jdx_dir / f"IR_{nist_id}.jdx",
            jdx_dir / f"{nist_id}_IR.jdx",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback — глоб по подстроке.
    matches = list(jdx_dir.glob(f"*{nist_id}*.jdx"))
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
    """Greedy round-robin: добираем «бедную» группу, пока не сядем в лимит.

    Соединение считается доступным для группы, если соответствующий
    label-bit равен 1. Один и тот же InChI Key выбирается не больше одного
    раза.
    """
    rng = np.random.default_rng(seed)
    indices = np.array(df.index.to_numpy(), copy=True)
    rng.shuffle(indices)
    df_shuffled = df.loc[indices].reset_index(drop=True)

    label_matrix = np.stack(df_shuffled["labels"].to_numpy())
    used = np.zeros(len(df_shuffled), dtype=bool)
    group_counts = np.zeros(N_GROUPS, dtype=int)

    # Индексы соединений, у которых активна каждая группа.
    candidates_by_group: list[list[int]] = [
        list(np.where(label_matrix[:, g] == 1)[0]) for g in range(N_GROUPS)
    ]
    # Курсор по списку кандидатов каждой группы.
    cursors = [0] * N_GROUPS

    selected_rows: list[int] = []
    while len(selected_rows) < total:
        deficient = np.where(group_counts < target_per_group)[0]
        if deficient.size == 0:
            break
        # Берём группу с наименьшим текущим покрытием (стабильность).
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
            # Группа исчерпана — помечаем как «достигнутую», чтобы не
            # зацикливаться, и идём дальше.
            group_counts[target_group] = max(group_counts[target_group], target_per_group)
            continue

        used[idx_to_add] = True
        selected_rows.append(idx_to_add)
        # Обновляем counts для всех групп, в которых активен этот пример.
        group_counts += label_matrix[idx_to_add]

    return df_shuffled.iloc[selected_rows].reset_index(drop=True)


def select_predefense_subset(
    *,
    compounds_csv: Path,
    jdx_dir: Path,
    output_csv: Path,
    ir_info_csv: Path | None = None,
    total: int = 2000,
    target_per_group: int = 100,
    seed: int = 42,
) -> dict[str, int]:
    """Полный пайплайн отбора. Возвращает статистику для логирования/тестов."""
    if not compounds_csv.exists():
        raise FileNotFoundError(f"compounds_csv не найден: {compounds_csv}")
    if not jdx_dir.exists():
        raise FileNotFoundError(f"jdx_dir не найден: {jdx_dir}")

    compounds = pd.read_csv(compounds_csv)
    if "inchi" not in compounds.columns or "nist_id" not in compounds.columns:
        raise ValueError(
            f"в {compounds_csv} ожидаются колонки 'nist_id' и 'inchi', "
            f"получено: {list(compounds.columns)}"
        )
    log.info("compounds_loaded", rows=len(compounds), source=str(compounds_csv))

    ir_info: pd.DataFrame | None = None
    if ir_info_csv is not None and ir_info_csv.exists():
        ir_info = pd.read_csv(ir_info_csv)
        log.info("ir_info_loaded", rows=len(ir_info), source=str(ir_info_csv))

    # 1. Считаем метки и фильтруем по наличию JDX-файла.
    enriched: list[dict[str, object]] = []
    label_failures = 0
    missing_jdx = 0
    for _, row in tqdm(compounds.iterrows(), total=len(compounds), desc="label+filter"):
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
            # Соединение без целевых групп — не полезно для стратификации.
            continue
        nist_id = str(row["nist_id"])
        jdx_path = _find_jdx_for_id(jdx_dir, nist_id, ir_info)
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
    # Дедупликация по inchi_key до сэмплинга — оставляем первый встретившийся.
    df = df.drop_duplicates(subset="inchi_key", keep="first").reset_index(drop=True)
    log.info(
        "candidates_built",
        candidates=len(df),
        label_failures=label_failures,
        missing_jdx=missing_jdx,
    )

    subset = _stratified_sample(df, total=total, target_per_group=target_per_group, seed=seed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = subset[["nist_id", "smiles", "inchi_key", "jdx_path"]].copy()
    out["labels_csv"] = subset["labels"].apply(
        lambda arr: ",".join(GROUP_NAMES[i] for i, v in enumerate(arr) if v == 1)
    )
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
        default=_ML_ROOT / "data" / "processed" / "predefense_subset_ids.csv",
    )
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--target-per-group", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    select_predefense_subset(
        compounds_csv=args.compounds_csv,
        ir_info_csv=args.ir_info_csv if args.ir_info_csv.exists() else None,
        jdx_dir=args.jdx_dir,
        output_csv=args.output,
        total=args.total,
        target_per_group=args.target_per_group,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
