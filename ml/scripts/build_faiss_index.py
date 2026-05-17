"""Сборка FAISS-индекса из чекпойнта контрастной модели (Этап 6, §6.9).

Восстанавливает :class:`MoleculeTower` из чекпойнта ``ContrastiveTrainer``,
проходит по parquet с молекулами, считает L2-нормированные эмбеддинги и
сохраняет ``IndexFlatIP`` (inner product = cosine similarity на нормированных
векторах) + mapping ``vector_id → (compound_id, smiles)`` в JSON.

Также обновляет ``models/MANIFEST.json`` записью типа ``faiss_index``.

Запуск:
    python ml/scripts/build_faiss_index.py [--checkpoint PATH] [--parquet PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import structlog
import torch

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.models.molecule_tower import MoleculeTower, resolve_model_name  # noqa: E402

log = structlog.get_logger(__name__)


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка FAISS-индекса.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/ircnn-contrastive-0.2.0/best.pt"),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("ml/data/synthetic.parquet"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/faiss"),
        help="каталог, в котором создаётся подкаталог по имени модели",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA запрошена, но недоступна")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_molecule_tower(
    checkpoint: Path, device: torch.device
) -> tuple[MoleculeTower, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_name = payload.get("molecule_model_name") or resolve_model_name()
    tower = MoleculeTower(model_name).to(device)
    projection_state = payload.get("molecule_projection_state_dict")
    if projection_state is not None:
        tower.projection.load_state_dict(projection_state, strict=True)
    tower.eval()
    return tower, payload


def _embed_batches(
    tower: MoleculeTower,
    smiles: list[str],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start : start + batch_size]
            tower.to(device)
            z = tower(batch)
            embeddings.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)


def _update_manifest(
    manifest_path: Path,
    *,
    name: str,
    version: str,
    relative_file: str,
    n_vectors: int,
    dim: int,
    source_checkpoint: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    models = list(raw.get("models", []))
    models = [m for m in models if not (m.get("name") == name and m.get("version") == version)]
    models.append(
        {
            "name": name,
            "version": version,
            "type": "faiss_index",
            "file": relative_file,
            "phase": 1,
            "data_source": "synthetic",
            "built_at": dt.date.today().isoformat(),
            "source_checkpoint": source_checkpoint,
            "metrics": {"n_vectors": int(n_vectors), "dim": int(dim)},
            "notes": "Этап 6 фазы 1. IndexFlatIP на L2-нормированных молекулярных эмбеддингах.",
        }
    )
    raw["models"] = models
    manifest_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    args = _parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    parquet = args.parquet.resolve()
    output_root = args.output_root.resolve()
    device = _resolve_device(args.device)

    if not checkpoint.exists():
        raise FileNotFoundError(f"чекпойнт не найден: {checkpoint}")
    if not parquet.exists():
        raise FileNotFoundError(f"parquet не найден: {parquet}")

    log.info("loading_checkpoint", path=str(checkpoint))
    tower, payload = _load_molecule_tower(checkpoint, device)

    frame = pd.read_parquet(parquet, engine="pyarrow")
    smiles = [str(s) for s in frame["smiles"].to_list()]
    compound_ids = [int(cid) for cid in frame["compound_id"].to_list()]
    log.info("dataset_loaded", rows=len(smiles), parquet=str(parquet))

    embeddings = _embed_batches(tower, smiles, batch_size=int(args.batch_size), device=device)
    if embeddings.size == 0:
        raise RuntimeError("эмбеддинги пусты — проверь датасет")

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings))

    # Версия и имя берутся из конфига внутри payload, fallback — параметры по умолчанию.
    config = payload.get("config", {}) or {}
    output_cfg = config.get("config", {}).get("output", {}) if isinstance(config, dict) else {}
    model_name = str(output_cfg.get("model_name", "ircnn-contrastive"))
    model_version = str(output_cfg.get("model_version", "0.2.0"))
    sub_dir = output_root / f"{model_name}-{model_version}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    index_path = sub_dir / "index.faiss"
    mapping_path = sub_dir / "mapping.json"
    meta_path = sub_dir / "meta.json"

    faiss.write_index(index, str(index_path))
    mapping = [
        {"id": i, "compound_id": cid, "smiles": s}
        for i, (cid, s) in enumerate(zip(compound_ids, smiles, strict=True))
    ]
    mapping_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    meta_path.write_text(
        json.dumps(
            {
                "dim": dim,
                "n_vectors": int(index.ntotal),
                "index_type": "IndexFlatIP",
                "source_checkpoint": str(checkpoint),
                "molecule_model_name": tower.model_name,
                "built_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info(
        "index_written",
        index=str(index_path),
        mapping=str(mapping_path),
        n_vectors=int(index.ntotal),
        dim=dim,
    )

    repo_root = Path.cwd()
    try:
        relative_checkpoint = checkpoint.relative_to(repo_root).as_posix()
    except ValueError:
        relative_checkpoint = str(checkpoint)
    manifest_path = Path("models/MANIFEST.json").resolve()
    _update_manifest(
        manifest_path,
        name=f"faiss-index-{model_name}",
        version=model_version,
        relative_file=f"faiss/{model_name}-{model_version}/index.faiss",
        n_vectors=int(index.ntotal),
        dim=dim,
        source_checkpoint=relative_checkpoint,
    )
    log.info("manifest_updated", path=str(manifest_path))


if __name__ == "__main__":
    main()
