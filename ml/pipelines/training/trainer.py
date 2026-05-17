"""Цикл обучения 1D-CNN с early stopping, чекпойнтами и логированием.

Поведение соответствует §6.5 (валидация), §6.8 (процесс обучения), §6.9.3
(per-class пороги) пояснительной записки. Логирование — TensorBoard +
``metrics.json``, как в промпте Этапа 5 `DEVELOPMENT_PLAN.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pipelines.metrics import compute_metrics, search_thresholds

log = structlog.get_logger(__name__)


@dataclass
class EpochStats:
    """Снимок метрик одной эпохи."""

    epoch: int
    train_loss: float
    val_loss: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainState:
    """Итоговое состояние обучения."""

    best_epoch: int
    best_macro_f1: float
    best_thresholds: list[float]
    history: list[EpochStats]
    stopped_early: bool


class Trainer:
    """Управляет циклом обучения и валидацией.

    Args:
        model: обучаемая модель.
        loss_fn: loss-функция.
        optimizer: оптимизатор (AdamW по умолчанию в `train_cnn1d.py`).
        scheduler: LR-scheduler с шагом ``step()`` на каждый батч.
        device: устройство (``cpu`` или ``cuda``).
        output_dir: каталог для чекпойнтов модели (``models/<name>-<ver>/``).
        run_dir: каталог эксперимента (``ml/experiments/<run_id>/``).
        class_names: имена классов для per-class метрик.
        threshold_mode: ``per_class`` (грид-поиск) или ``fixed``.
        fixed_threshold: значение порога в режиме ``fixed``.
        grad_clip: норма для gradient clipping (None — выключено).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
        output_dir: Path,
        run_dir: Path,
        class_names: list[str] | tuple[str, ...],
        threshold_mode: str = "per_class",
        fixed_threshold: float = 0.5,
        grad_clip: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.run_dir = run_dir
        self.class_names = tuple(class_names)
        self.threshold_mode = threshold_mode
        self.fixed_threshold = float(fixed_threshold)
        self.grad_clip = grad_clip

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        n_seen = 0
        for spectra, targets in loader:
            spectra = spectra.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(spectra)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            batch = spectra.size(0)
            running_loss += float(loss.detach().cpu()) * batch
            n_seen += batch
        return running_loss / max(n_seen, 1)

    def _validate(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        n_seen = 0
        all_probs: list[Tensor] = []
        all_targets: list[Tensor] = []
        with torch.no_grad():
            for spectra, targets in loader:
                spectra = spectra.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(spectra)
                loss = self.loss_fn(logits, targets)
                running_loss += float(loss.detach().cpu()) * spectra.size(0)
                n_seen += spectra.size(0)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_targets.append(targets.cpu())
        probs = torch.cat(all_probs).numpy() if all_probs else np.zeros((0, 0))
        targets_arr = (
            torch.cat(all_targets).numpy().astype(np.int64) if all_targets else np.zeros((0, 0))
        )
        return running_loss / max(n_seen, 1), targets_arr, probs

    def _resolve_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        if self.threshold_mode == "fixed":
            return np.full(y_true.shape[1], self.fixed_threshold, dtype=np.float64)
        return search_thresholds(y_true, y_prob, default=self.fixed_threshold)

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        thresholds: np.ndarray,
        metrics: dict[str, Any],
        config: dict[str, Any] | None,
    ) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "thresholds": thresholds.tolist(),
            "metrics": metrics,
            "epoch": epoch,
            "class_names": list(self.class_names),
            "config": config,
        }
        torch.save(payload, path)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
        patience: int,
        config_snapshot: dict[str, Any] | None = None,
    ) -> TrainState:
        history: list[EpochStats] = []
        best_macro_f1 = -1.0
        best_epoch = -1
        best_thresholds = np.full(len(self.class_names), self.fixed_threshold, dtype=np.float64)
        epochs_without_improvement = 0
        stopped_early = False

        last_path = self.output_dir / "last.pt"
        best_path = self.output_dir / "best.pt"

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, y_true, y_prob = self._validate(val_loader)
            thresholds = self._resolve_thresholds(y_true, y_prob)
            metrics = compute_metrics(
                y_true, y_prob, thresholds, class_names=list(self.class_names)
            )

            stats = EpochStats(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
            )
            history.append(stats)
            self._log_epoch(stats)

            self._save_checkpoint(
                last_path,
                epoch=epoch,
                thresholds=thresholds,
                metrics=metrics,
                config=config_snapshot,
            )

            improved = metrics["macro_f1"] > best_macro_f1
            if improved:
                best_macro_f1 = float(metrics["macro_f1"])
                best_epoch = epoch
                best_thresholds = thresholds
                epochs_without_improvement = 0
                self._save_checkpoint(
                    best_path,
                    epoch=epoch,
                    thresholds=thresholds,
                    metrics=metrics,
                    config=config_snapshot,
                )
            else:
                epochs_without_improvement += 1

            self._write_metrics_json(history)

            if epochs_without_improvement >= patience:
                stopped_early = True
                log.info(
                    "early_stopping",
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_macro_f1=best_macro_f1,
                )
                break

        self.writer.flush()
        self.writer.close()
        return TrainState(
            best_epoch=best_epoch,
            best_macro_f1=best_macro_f1,
            best_thresholds=best_thresholds.tolist(),
            history=history,
            stopped_early=stopped_early,
        )

    def _log_epoch(self, stats: EpochStats) -> None:
        m = stats.metrics
        log.info(
            "epoch_completed",
            epoch=stats.epoch,
            train_loss=round(stats.train_loss, 5),
            val_loss=round(stats.val_loss, 5),
            macro_f1=round(float(m.get("macro_f1", 0.0)), 4),
            macro_ap=round(float(m.get("macro_ap", 0.0)), 4),
            hamming=round(float(m.get("hamming_loss", 0.0)), 4),
        )
        self.writer.add_scalar("loss/train", stats.train_loss, stats.epoch)
        self.writer.add_scalar("loss/val", stats.val_loss, stats.epoch)
        for key in (
            "macro_f1",
            "micro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "macro_ap",
            "macro_auc",
            "hamming_loss",
            "subset_accuracy",
        ):
            value = m.get(key)
            if isinstance(value, int | float):
                fvalue = float(value)
                if not np.isnan(fvalue):
                    self.writer.add_scalar(f"val/{key}", fvalue, stats.epoch)

    def _write_metrics_json(self, history: list[EpochStats]) -> None:
        path = self.run_dir / "metrics.json"
        payload = {
            "history": [
                {
                    "epoch": s.epoch,
                    "train_loss": s.train_loss,
                    "val_loss": s.val_loss,
                    "metrics": _strip_nan(s.metrics),
                }
                for s in history
            ]
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _strip_nan(metrics: dict[str, Any]) -> dict[str, Any]:
    """Заменяет NaN на null для совместимости с JSON."""
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            cleaned[key] = None
        elif isinstance(value, dict):
            cleaned[key] = _strip_nan(value)
        else:
            cleaned[key] = value
    return cleaned


__all__ = ["EpochStats", "TrainState", "Trainer"]
