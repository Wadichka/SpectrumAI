"""Точка входа контрастного обучения двухбашенной схемы (Этап 6).

Поднимает SpectrumTower (с опциональной инициализацией CNN из чекпойнта
Этапа 5), MoleculeTower (ChemBERTa с заморозкой), BCE + symmetric InfoNCE,
AdamW + cosine + warmup. Прогоняет warm-up → joint фазы согласно §6.8.2,
сохраняет чекпойнты и обновляет ``models/MANIFEST.json``.

Запуск:
    python ml/scripts/train_contrastive.py --config ml/configs/contrastive.yaml \
        [--dry-run] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
import yaml  # type: ignore[import-untyped]
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.augmentations import default_train_augmentation  # noqa: E402
from pipelines.dataset import SpectraDataset  # noqa: E402
from pipelines.labeling import GROUP_NAMES, N_GROUPS  # noqa: E402
from pipelines.losses import compute_pos_weight, make_loss  # noqa: E402
from pipelines.models.cnn1d import build_model, count_parameters  # noqa: E402
from pipelines.models.molecule_tower import MoleculeTower  # noqa: E402
from pipelines.models.spectrum_tower import SpectrumTower  # noqa: E402
from pipelines.training import (  # noqa: E402
    ContrastiveTrainer,
    set_global_seed,
    split_by_inchi_key,
    stratified_multilabel_split,
)

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
    parser = argparse.ArgumentParser(description="Обучение двухбашенной схемы.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ML_ROOT / "configs" / "contrastive.yaml",
        help="путь к YAML-конфигу",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA запрошена, но недоступна")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_label_matrix(parquet_path: Path, n_labels: int) -> np.ndarray:
    frame = pd.read_parquet(parquet_path, engine="pyarrow")
    rows = [np.asarray(row, dtype=np.int64) for row in frame["labels"].to_list()]
    matrix = np.stack(rows, axis=0)
    if matrix.shape[1] != n_labels:
        raise ValueError(f"в parquet метки длины {matrix.shape[1]}, конфиг ожидает {n_labels}")
    return matrix


def _build_loaders(
    cfg: dict[str, Any],
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    rng: np.random.Generator,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    parquet_path = Path(data_cfg["parquet"])
    spectrum_length = int(data_cfg["spectrum_length"])
    n_labels = int(data_cfg["n_labels"])
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 0))

    use_aug = bool(data_cfg.get("augmentation", True))
    transform = default_train_augmentation(rng) if use_aug else None

    train_dataset = SpectraDataset(
        parquet_path,
        transform=transform,
        spectrum_length=spectrum_length,
        n_labels=n_labels,
        return_smiles=True,
    )
    val_dataset = SpectraDataset(
        parquet_path,
        transform=None,
        spectrum_length=spectrum_length,
        n_labels=n_labels,
        return_smiles=True,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def _make_scheduler(optimizer: AdamW, *, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


def _make_run_dir(experiment_root: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = experiment_root / f"contrastive-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_cnn_with_optional_checkpoint(cnn_cfg: dict[str, Any], device: torch.device) -> Any:
    model = build_model(cnn_cfg)
    checkpoint_path = cnn_cfg.get("init_checkpoint")
    if checkpoint_path:
        path = Path(str(checkpoint_path))
        if path.exists():
            payload = torch.load(path, map_location=device, weights_only=False)
            state_dict = payload.get("state_dict", payload)
            try:
                model.load_state_dict(state_dict, strict=True)
                log.info("cnn_init_loaded", path=str(path))
            except RuntimeError as exc:
                log.warning("cnn_init_skipped", path=str(path), reason=str(exc))
        else:
            log.warning("cnn_init_not_found", path=str(path))
    return model


def _update_manifest(
    manifest_path: Path,
    *,
    name: str,
    version: str,
    relative_file: str,
    metrics: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    models = list(raw.get("models", []))
    models = [m for m in models if not (m.get("name") == name and m.get("version") == version)]
    retrieval = metrics.get("retrieval", {}) if isinstance(metrics, dict) else {}
    models.append(
        {
            "name": name,
            "version": version,
            "file": relative_file,
            "phase": 1,
            "data_source": "synthetic",
            "trained_at": dt.date.today().isoformat(),
            "metrics": {
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "top1": float(retrieval.get("top1", 0.0)),
                "top5": float(retrieval.get("top5", 0.0)),
                "mrr": float(retrieval.get("mrr", 0.0)),
                "temperature": float(metrics.get("temperature", 0.0)),
            },
            "notes": "Этап 6 фазы 1. Двухбашенная схема, BCE + InfoNCE, синтетика.",
        }
    )
    raw["models"] = models
    manifest_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    args = _parse_args(argv)
    cfg_path = args.config.resolve()
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    if int(cfg["data"]["n_labels"]) != N_GROUPS:
        raise ValueError(
            f"data.n_labels={cfg['data']['n_labels']} ≠ N_GROUPS={N_GROUPS}; проверь конфиг"
        )

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    device = _resolve_device(args.device)
    log.info(
        "session_started",
        device=str(device),
        config=str(cfg_path),
        tiny_bert=os.environ.get("SPECTRUMAI_USE_TINY_BERT") == "1",
    )

    data_cfg = cfg["data"]
    parquet_path = Path(data_cfg["parquet"]).resolve()
    labels = _load_label_matrix(parquet_path, int(data_cfg["n_labels"]))
    split_cfg = data_cfg["split"]
    split_strategy = str(split_cfg.get("strategy", "stratified_multilabel"))
    ratios = (
        float(split_cfg["train"]),
        float(split_cfg["val"]),
        float(split_cfg["test"]),
    )
    if split_strategy == "by_inchi_key":
        inchi_keys = pd.read_parquet(parquet_path, columns=["inchi_key"])["inchi_key"].tolist()
        if len(inchi_keys) != labels.shape[0]:
            raise ValueError(
                f"inchi_key rows ({len(inchi_keys)}) != labels rows ({labels.shape[0]}); "
                f"проверь parquet {parquet_path}"
            )
        train_idx, val_idx, test_idx = split_by_inchi_key(inchi_keys, ratios=ratios, seed=seed)
    elif split_strategy == "stratified_multilabel":
        train_idx, val_idx, test_idx = stratified_multilabel_split(
            labels, ratios=ratios, seed=seed
        )
    else:
        raise ValueError(
            f"unknown data.split.strategy={split_strategy!r}; "
            "supported: 'stratified_multilabel', 'by_inchi_key'"
        )
    log.info(
        "split_done",
        strategy=split_strategy,
        train=len(train_idx),
        val=len(val_idx),
        test=len(test_idx),
    )

    train_loader, val_loader = _build_loaders(cfg, train_idx=train_idx, val_idx=val_idx, rng=rng)

    cnn = _load_cnn_with_optional_checkpoint(cfg["cnn"], device)
    spectrum_tower = SpectrumTower(
        cnn,
        projection_dim=int(cfg["spectrum_tower"].get("projection_dim", 128)),
        hidden_dim=int(cfg["spectrum_tower"].get("projection_hidden", 256)),
        dropout=float(cfg["spectrum_tower"].get("projection_dropout", 0.10)),
        freeze_encoder=bool(cfg["spectrum_tower"].get("freeze_encoder", False)),
    )
    mt_model_name = cfg["molecule_tower"].get("model_name")
    # SPECTRUMAI_USE_TINY_BERT=1 перебивает конфиг — это нужно для CI,
    # чтобы не качать реальный ChemBERTa на runner.
    if os.environ.get("SPECTRUMAI_USE_TINY_BERT") == "1":
        mt_model_name = None
    molecule_tower = MoleculeTower(
        mt_model_name,
        projection_dim=int(cfg["molecule_tower"].get("projection_dim", 128)),
        hidden_dim=int(cfg["molecule_tower"].get("projection_hidden", 384)),
        dropout=float(cfg["molecule_tower"].get("projection_dropout", 0.10)),
        freeze_encoder=bool(cfg["molecule_tower"].get("freeze_encoder", True)),
        max_length=int(cfg["molecule_tower"].get("max_length", 128)),
    )
    log.info(
        "towers_built",
        spectrum_params=count_parameters(spectrum_tower),
        molecule_model=molecule_tower.model_name,
        molecule_projection_params=sum(
            int(p.numel()) for p in molecule_tower.projection.parameters()
        ),
    )

    loss_cfg = cfg.get("loss", {})
    bce_cfg = loss_cfg.get("bce", {}) or {}
    pos_weight_setting = bce_cfg.get("pos_weight", "auto")
    pos_weight_tensor: torch.Tensor | None
    if pos_weight_setting == "auto":
        pos_weight_tensor = compute_pos_weight(
            labels[train_idx], cap=float(bce_cfg.get("pos_weight_cap", 100.0))
        ).to(device)
    elif pos_weight_setting in (None, "none", "null"):
        pos_weight_tensor = None
    else:
        pos_weight_tensor = torch.tensor(
            [float(x) for x in pos_weight_setting], dtype=torch.float32, device=device
        )

    bce_loss = make_loss({"type": "bce"}, pos_weight=pos_weight_tensor)
    infonce_loss = make_loss(
        {"type": "infonce", "infonce": loss_cfg.get("infonce", {})},
        pos_weight=None,
    )

    optim_cfg = cfg.get("optim", {})
    trainable: list[torch.nn.Parameter] = [
        p for p in spectrum_tower.parameters() if p.requires_grad
    ]
    trainable.extend(p for p in molecule_tower.parameters() if p.requires_grad)
    trainable.extend(p for p in infonce_loss.parameters() if p.requires_grad)
    if not trainable:
        raise RuntimeError("нет обучаемых параметров")

    optimizer = AdamW(
        trainable,
        lr=float(optim_cfg.get("lr", 1.0e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 1.0e-4)),
    )

    training_cfg = cfg["training"]
    epochs = int(training_cfg["epochs"])
    total_steps = max(1, epochs * max(1, len(train_loader)))
    scheduler = _make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=float(optim_cfg.get("warmup_ratio", 0.05)),
    )

    output_cfg = cfg["output"]
    model_name = str(output_cfg["model_name"])
    model_version = str(output_cfg["model_version"])
    model_dir = Path(output_cfg["model_dir"]).resolve() / f"{model_name}-{model_version}"
    experiment_root = Path(output_cfg["experiment_root"]).resolve()
    run_dir = _make_run_dir(experiment_root)

    config_snapshot = {
        "config_path": str(cfg_path),
        "config": cfg,
        "seed": seed,
        "device": str(device),
        "split": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "molecule_model_name": molecule_tower.model_name,
    }

    trainer = ContrastiveTrainer(
        spectrum_tower=spectrum_tower,
        molecule_tower=molecule_tower,
        bce_loss=bce_loss,
        infonce_loss=infonce_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=model_dir,
        run_dir=run_dir,
        class_names=GROUP_NAMES,
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        nce_weight=float(loss_cfg.get("nce_weight", 0.5)),
        warmup_epochs=int(training_cfg.get("warmup_epochs", 2)),
        threshold_mode=str(training_cfg.get("threshold_search", "per_class")),
        fixed_threshold=float(training_cfg.get("fixed_threshold", 0.5)),
        grad_clip=float(training_cfg.get("grad_clip", 1.0)),
    )

    if args.dry_run:
        log.info("dry_run_started", batches=len(train_loader))
        spectra, labels_batch, smiles_list = next(iter(train_loader))
        spectra = spectra.to(device)
        labels_batch = labels_batch.to(device)
        if isinstance(smiles_list, tuple):
            smiles_list = list(smiles_list)
        embedding, z_spec = spectrum_tower.forward_with_embedding(spectra)
        logits = spectrum_tower.encoder.classifier(embedding)
        z_mol = molecule_tower(smiles_list)
        loss_b = bce_loss(logits, labels_batch)
        loss_n = infonce_loss(z_spec, z_mol)
        loss = loss_b + 0.5 * loss_n
        loss.backward()
        log.info(
            "dry_run_passed",
            logits_shape=list(logits.shape),
            z_spec_shape=list(z_spec.shape),
            z_mol_shape=list(z_mol.shape),
            bce=float(loss_b.detach().cpu()),
            infonce=float(loss_n.detach().cpu()),
            temperature=float(infonce_loss.temperature),  # type: ignore[attr-defined]
        )
        trainer.writer.flush()
        trainer.writer.close()
        return

    state = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        patience=int(training_cfg.get("early_stop_patience", 5)),
        config_snapshot=config_snapshot,
    )

    log.info(
        "training_finished",
        best_epoch=state.best_epoch,
        best_score=round(state.best_score, 4),
        best_macro_f1=round(state.best_macro_f1, 4),
        best_top1=round(state.best_top1, 4),
        stopped_early=state.stopped_early,
        run_dir=str(run_dir),
        model_dir=str(model_dir),
    )

    if state.best_epoch < 0 or not state.history:
        log.warning("no_epochs_completed_skip_manifest_update")
        return
    best_metrics = next(s for s in state.history if s.epoch == state.best_epoch).metrics
    manifest_path = Path(output_cfg["model_dir"]).resolve() / "MANIFEST.json"
    _update_manifest(
        manifest_path,
        name=model_name,
        version=model_version,
        relative_file=f"{model_name}-{model_version}/best.pt",
        metrics=best_metrics,
    )
    log.info("manifest_updated", path=str(manifest_path))


if __name__ == "__main__":
    main()
