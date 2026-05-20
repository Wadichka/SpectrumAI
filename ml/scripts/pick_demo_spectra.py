"""Аналитический отбор демо-спектров для предзащиты (этап 20, фаза 2).

Берёт ``predefense_labeled.parquet`` + предобученные чекпойнты + FAISS,
прогоняет каждую строку через CNN и contrastive tower, считает уверенность
модели и отбирает 7 показательных кейсов:

- 4 «чистых» (label_f1 ≥ 0.7, top1_hit=1, 1–2 функциональные группы);
- 2 «множественных» (label_f1 ≥ 0.5, ≥ 3 групп);
- 1 «трудный» (top1_hit=0, top5_hit=1, label_f1 ≥ 0.3).

Для каждого выбранного NIST_ID копирует ``ml/data/raw/nist/{ID}_IR_0.jdx``
в ``demo/predefense_spectra/{ID}.jdx`` и записывает метаданные в
``demo/predefense_spectra/selection.json``.

Запуск:
    python ml/scripts/pick_demo_spectra.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import torch
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import f1_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ML_ROOT = _REPO_ROOT / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.labeling import GROUP_NAMES  # noqa: E402
from pipelines.models.cnn1d import build_model as build_cnn  # noqa: E402
from pipelines.models.molecule_tower import MoleculeTower  # noqa: E402
from pipelines.models.spectrum_tower import SpectrumTower  # noqa: E402

PARQUET = _REPO_ROOT / "ml" / "data" / "processed" / "predefense_labeled.parquet"
CNN_CKPT = _REPO_ROOT / "models" / "checkpoints" / "cnn1d-predefense-0.5.0" / "best.pt"
CONTRASTIVE_CKPT = _REPO_ROOT / "models" / "checkpoints" / "contrastive-predefense-0.5.0" / "best.pt"
FAISS_DIR = _REPO_ROOT / "models" / "faiss" / "contrastive-predefense-0.5.0"
RAW_NIST_DIR = _REPO_ROOT / "ml" / "data" / "raw" / "nist"
DEMO_DIR = _REPO_ROOT / "demo" / "predefense_spectra"
CONFIG_PATH = _REPO_ROOT / "ml" / "configs" / "contrastive_predefense.yaml"


def _score_rows() -> pd.DataFrame:
    """Прогоняет parquet через CNN + contrastive, возвращает DataFrame со score'ами."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    device = torch.device("cpu")
    df = pd.read_parquet(PARQUET)
    print(f"loaded parquet: {len(df)} rows")

    # CNN
    cnn = build_cnn(cfg["cnn"]).to(device).eval()
    cnn_state = torch.load(CNN_CKPT, map_location=device, weights_only=False)
    cnn.load_state_dict(cnn_state["state_dict"])
    thresholds = np.asarray(cnn_state["thresholds"], dtype=np.float64)

    spectra = np.stack([np.asarray(s, dtype=np.float32) for s in df["spectrum"].tolist()])
    labels_true = np.stack([np.asarray(l, dtype=np.int8) for l in df["labels"].tolist()])

    batch_size = 64
    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            chunk = torch.from_numpy(spectra[start:start + batch_size]).to(device)
            logits = cnn(chunk)
            probs_chunks.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs_chunks, axis=0)
    preds = (probs >= thresholds[None, :]).astype(np.int8)
    print(f"cnn forward done; preds shape={preds.shape}")

    # per-row F1
    label_f1 = np.array(
        [
            f1_score(labels_true[i], preds[i], average="macro", zero_division=0)
            for i in range(len(df))
        ]
    )

    # Contrastive tower → spectrum embeddings
    spectrum_cnn = build_cnn(cfg["cnn"]).to(device).eval()
    spectrum_tower = SpectrumTower(
        spectrum_cnn,
        projection_dim=int(cfg["spectrum_tower"].get("projection_dim", 128)),
        hidden_dim=int(cfg["spectrum_tower"].get("projection_hidden", 256)),
        dropout=float(cfg["spectrum_tower"].get("projection_dropout", 0.10)),
        freeze_encoder=bool(cfg["spectrum_tower"].get("freeze_encoder", False)),
    ).to(device).eval()
    contrastive_state = torch.load(CONTRASTIVE_CKPT, map_location=device, weights_only=False)
    spectrum_tower.load_state_dict(contrastive_state["spectrum_tower_state_dict"])

    z_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            chunk = torch.from_numpy(spectra[start:start + batch_size]).to(device)
            z = spectrum_tower(chunk)
            z_chunks.append(z.cpu().numpy().astype(np.float32))
    z_spec = np.concatenate(z_chunks, axis=0)
    print(f"spectrum embeddings done; shape={z_spec.shape}")

    # FAISS lookup
    index = faiss.read_index(str(FAISS_DIR / "index.faiss"))
    mapping = json.loads((FAISS_DIR / "mapping.json").read_text(encoding="utf-8"))
    # Build inchi_key → vector_id by joining mapping.compound_id with df.id
    id_to_vec = {entry["compound_id"]: entry["id"] for entry in mapping}

    distances, indices = index.search(z_spec, k=10)
    ranks = np.full(len(df), index.ntotal + 1, dtype=np.int64)
    top1_hit = np.zeros(len(df), dtype=np.int8)
    top5_hit = np.zeros(len(df), dtype=np.int8)
    for i, row_id in enumerate(df["id"].tolist()):
        true_vec = id_to_vec.get(int(row_id))
        if true_vec is None:
            continue
        neighbors = indices[i].tolist()
        if true_vec in neighbors:
            rank = neighbors.index(true_vec) + 1
            ranks[i] = rank
            top1_hit[i] = 1 if rank == 1 else 0
            top5_hit[i] = 1 if rank <= 5 else 0

    scored = pd.DataFrame(
        {
            "nist_id": df["nist_id"].astype(str),
            "name": df["name"],
            "smiles": df["smiles"],
            "inchi_key": df["inchi_key"],
            "label_f1": label_f1,
            "top1_hit": top1_hit,
            "top5_hit": top5_hit,
            "rank": ranks,
            "n_labels_true": labels_true.sum(axis=1),
            "true_groups": [
                ",".join(GROUP_NAMES[g] for g in np.where(row == 1)[0]) for row in labels_true
            ],
            "predicted_groups": [
                ",".join(GROUP_NAMES[g] for g in np.where(row == 1)[0]) for row in preds
            ],
        }
    )
    return scored


def _pick_categories(scored: pd.DataFrame) -> list[dict]:
    """Отбирает 4 clean, 2 multi-label, 1 difficult."""
    picks: list[dict] = []

    # 4 clean: 1-2 groups, f1>=0.7, top1=1
    clean_pool = scored[
        (scored["n_labels_true"].between(1, 2))
        & (scored["label_f1"] >= 0.7)
        & (scored["top1_hit"] == 1)
    ].sort_values("label_f1", ascending=False)
    seen_inchi: set[str] = set()
    for _, row in clean_pool.iterrows():
        if len(picks) >= 4:
            break
        if row["inchi_key"] in seen_inchi:
            continue
        seen_inchi.add(row["inchi_key"])
        picks.append({**row.to_dict(), "category": "clean"})

    # 2 multi-label: 3+ groups, f1>=0.5
    multi_pool = scored[
        (scored["n_labels_true"] >= 3) & (scored["label_f1"] >= 0.5)
    ].sort_values("label_f1", ascending=False)
    for _, row in multi_pool.iterrows():
        if len([p for p in picks if p["category"] == "multi"]) >= 2:
            break
        if row["inchi_key"] in seen_inchi:
            continue
        seen_inchi.add(row["inchi_key"])
        picks.append({**row.to_dict(), "category": "multi"})

    # 1 difficult: top1=0, top5=1, f1>=0.3
    diff_pool = scored[
        (scored["top1_hit"] == 0) & (scored["top5_hit"] == 1) & (scored["label_f1"] >= 0.3)
    ].sort_values("label_f1", ascending=False)
    for _, row in diff_pool.iterrows():
        if any(p["category"] == "difficult" for p in picks):
            break
        if row["inchi_key"] in seen_inchi:
            continue
        seen_inchi.add(row["inchi_key"])
        picks.append({**row.to_dict(), "category": "difficult"})

    return picks


def _relax_picks(scored: pd.DataFrame, picks: list[dict]) -> list[dict]:
    """Если строгие критерии не дали 7 шт., ослабляем по очереди."""
    target_by_cat = {"clean": 4, "multi": 2, "difficult": 1}
    seen_inchi = {p["inchi_key"] for p in picks}

    def _have(cat: str) -> int:
        return sum(1 for p in picks if p["category"] == cat)

    relaxations = [
        ("clean", scored[(scored["n_labels_true"].between(1, 2)) & (scored["label_f1"] >= 0.6)]),
        ("clean", scored[(scored["n_labels_true"].between(1, 3)) & (scored["label_f1"] >= 0.5)]),
        ("multi", scored[(scored["n_labels_true"] >= 3) & (scored["label_f1"] >= 0.4)]),
        ("multi", scored[(scored["n_labels_true"] >= 3) & (scored["label_f1"] >= 0.3)]),
        (
            "difficult",
            scored[(scored["top1_hit"] == 0) & (scored["top5_hit"] == 1)].sort_values(
                "label_f1", ascending=False
            ),
        ),
        (
            "difficult",
            scored[(scored["top1_hit"] == 0) & (scored["rank"] <= 10)].sort_values(
                "label_f1", ascending=False
            ),
        ),
    ]
    for cat, pool in relaxations:
        if _have(cat) >= target_by_cat[cat]:
            continue
        pool_sorted = pool.sort_values("label_f1", ascending=False)
        for _, row in pool_sorted.iterrows():
            if _have(cat) >= target_by_cat[cat]:
                break
            if row["inchi_key"] in seen_inchi:
                continue
            seen_inchi.add(row["inchi_key"])
            picks.append({**row.to_dict(), "category": cat})
    return picks


def _copy_jdx_files(picks: list[dict]) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for pick in picks:
        nist_id = str(pick["nist_id"])
        src_candidates = sorted(RAW_NIST_DIR.glob(f"{nist_id}_IR_*.jdx"))
        if not src_candidates:
            print(f"  WARN: no JDX for {nist_id}")
            pick["jdx_file"] = None
            continue
        src = src_candidates[0]
        dst = DEMO_DIR / f"{nist_id}.jdx"
        if not dst.exists():
            shutil.copy2(src, dst)
        pick["jdx_file"] = f"demo/predefense_spectra/{dst.name}"


def _to_serializable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    scored = _score_rows()
    print("score stats:")
    print(scored[["label_f1", "top1_hit", "top5_hit", "n_labels_true"]].describe())

    picks = _pick_categories(scored)
    picks = _relax_picks(scored, picks)
    print(f"final picks: {len(picks)} (cats: {[p['category'] for p in picks]})")

    _copy_jdx_files(picks)

    # Serialize selection.json
    serializable = []
    for p in picks:
        entry = {k: _to_serializable(v) for k, v in p.items()}
        serializable.append(entry)
    (DEMO_DIR / "selection.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {DEMO_DIR / 'selection.json'} with {len(serializable)} entries")


if __name__ == "__main__":
    main()
