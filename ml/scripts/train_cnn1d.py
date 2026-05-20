"""Точка входа обучения 1D-CNN multi-label классификатора (Этап 5).

Читает YAML-конфиг (`ml/configs/cnn1d.yaml`), собирает датасет, модель,
loss и оптимизатор по главам 6.2–6.8 пояснительной записки, прогоняет
обучение через :class:`pipelines.training.Trainer`. После успешного
прогона дописывает запись о модели в `models/MANIFEST.json` (CLAUDE.md §10).

Запуск:
    python ml/scripts/train_cnn1d.py --config ml/configs/cnn1d.yaml [--dry-run] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
from pipelines.training import (  # noqa: E402
    Trainer,
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
    parser = argparse.ArgumentParser(description="Обучение 1D-CNN по конфигу cnn1d.yaml.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ML_ROOT / "configs" / "cnn1d.yaml",
        help="путь к YAML-конфигу",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="устройство обучения",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="собрать всё и прогнать один батч в forward+backward без полного цикла",
    )
    return parser.parse_args(argv)


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA запрошена, но недоступна на этой машине")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_label_matrix(parquet_path: Path, n_labels: int) -> np.ndarray:
    """Читает матрицу меток (N, n_labels) напрямую из parquet."""
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

    use_aug = bool(data_cfg.get("augmentation", True))
    transform = default_train_augmentation(rng) if use_aug else None
    train_full = SpectraDataset(
        parquet_path,
        transform=transform,
        spectrum_length=spectrum_length,
        n_labels=n_labels,
    )
    val_full = SpectraDataset(
        parquet_path,
        transform=None,
        spectrum_length=spectrum_length,
        n_labels=n_labels,
    )
    train_loader = DataLoader(
        Subset(train_full, train_idx.tolist()),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx.tolist()),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
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
    run_dir = experiment_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _update_manifest(
    manifest_path: Path,
    *,
    name: str,
    version: str,
    relative_file: str,
    metrics: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    models = list(raw.get("models", []))
    models = [m for m in models if not (m.get("name") == name and m.get("version") == version)]
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
                "macro_ap": float(metrics.get("macro_ap", 0.0)),
                "hamming_loss": float(metrics.get("hamming_loss", 1.0)),
            },
            "notes": "Этап 5 фазы 1. Цель — отладка пайплайна, не достижение метрик §6.4.4.",
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

    model = build_model(cfg["model"])
    log.info("model_built", parameters=count_parameters(model))

    loss_cfg = cfg.get("loss", {})
    pos_weight_setting = loss_cfg.get("pos_weight", "auto")
    pos_weight_tensor: torch.Tensor | None
    if pos_weight_setting == "auto":
        pos_weight_tensor = compute_pos_weight(
            labels[train_idx], cap=float(loss_cfg.get("pos_weight_cap", 100.0))
        ).to(device)
    elif pos_weight_setting in (None, "none", "null"):
        pos_weight_tensor = None
    else:
        pos_weight_tensor = torch.tensor(
            [float(x) for x in pos_weight_setting], dtype=torch.float32, device=device
        )
    loss_fn = make_loss(loss_cfg, pos_weight=pos_weight_tensor)

    optim_cfg = cfg.get("optim", {})
    if str(optim_cfg.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("на этом этапе поддерживается только AdamW")
    optimizer = AdamW(
        model.parameters(),
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
    }

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=model_dir,
        run_dir=run_dir,
        class_names=GROUP_NAMES,
        threshold_mode=str(training_cfg.get("threshold_search", "per_class")),
        fixed_threshold=float(training_cfg.get("fixed_threshold", 0.5)),
    )

    if args.dry_run:
        log.info("dry_run_started", batches=len(train_loader))
        spectra, targets = next(iter(train_loader))
        spectra = spectra.to(device)
        targets = targets.to(device)
        logits = model(spectra)
        loss = loss_fn(logits, targets)
        loss.backward()
        log.info(
            "dry_run_passed",
            logits_shape=list(logits.shape),
            loss=float(loss.detach().cpu()),
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
        best_macro_f1=round(state.best_macro_f1, 4),
        stopped_early=state.stopped_early,
        run_dir=str(run_dir),
        model_dir=str(model_dir),
    )

    if state.best_epoch < 0 or not state.history:
        log.warning("no_epochs_completed_skip_manifest_update")
        return

    best_metrics = max(state.history, key=lambda s: s.metrics.get("macro_f1", -1.0)).metrics
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
