"""Цикл контрастного обучения двухбашенной схемы (§6.6, §6.8 главы 6).

Поддерживает два режима эпох:

- **Warm-up** (эпохи 1 .. ``warmup_epochs``): backward только по L_BCE.
  Молекулярная башня не вызывается. Это соответствует §6.8.2 «Этап 1
  (warm-up)»: спектральный энкодер и классификационная голова прогреваются
  до подключения контрастной потери.

- **Joint** (эпохи ``warmup_epochs + 1`` .. ``epochs``): multi-task
  ``L_total = bce_weight·L_BCE + nce_weight·L_NCE`` (§6.6.6).
  SMILES батча канонизируется, прогоняется через :class:`MoleculeTower`,
  даёт L2-нормированный молекулярный эмбеддинг.

На валидации считаем классификационные метрики (как Этап 5) и
retrieval-метрики (top-1/5/10, MRR) — по всему набору валидации в виде
полной матрицы сходств. Чекпойнтинг и early-stopping — по комбинированному
критерию ``0.5·macro_f1 + 0.5·top1_acc`` (§6.8.3). В warm-up top1 ещё
неинформативный, поэтому критерий вырождается в macro-F1.
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
from pipelines.models.molecule_tower import MoleculeTower
from pipelines.models.spectrum_tower import SpectrumTower
from pipelines.retrieval_metrics import mean_reciprocal_rank, topk_accuracy

log = structlog.get_logger(__name__)


@dataclass
class ContrastiveEpochStats:
    """Метрики одной эпохи контрастного обучения."""

    epoch: int
    phase: str  # "warmup" | "joint"
    train_loss: float
    train_loss_bce: float
    train_loss_nce: float
    val_loss: float
    val_loss_bce: float
    val_loss_nce: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContrastiveTrainState:
    """Итог обучения."""

    best_epoch: int
    best_score: float  # комбинированный
    best_macro_f1: float
    best_top1: float
    best_thresholds: list[float]
    history: list[ContrastiveEpochStats]
    stopped_early: bool


class ContrastiveTrainer:
    """Тренер для warm-up + joint multi-task обучения двухбашенной схемы."""

    def __init__(
        self,
        *,
        spectrum_tower: SpectrumTower,
        molecule_tower: MoleculeTower,
        bce_loss: nn.Module,
        infonce_loss: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
        output_dir: Path,
        run_dir: Path,
        class_names: list[str] | tuple[str, ...],
        bce_weight: float = 1.0,
        nce_weight: float = 0.5,
        warmup_epochs: int = 2,
        threshold_mode: str = "per_class",
        fixed_threshold: float = 0.5,
        grad_clip: float | None = 1.0,
    ) -> None:
        self.spectrum_tower = spectrum_tower.to(device)
        self.molecule_tower = molecule_tower.to(device)
        self.bce_loss = bce_loss.to(device)
        self.infonce_loss = infonce_loss.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.run_dir = run_dir
        self.class_names = tuple(class_names)
        self.bce_weight = float(bce_weight)
        self.nce_weight = float(nce_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.threshold_mode = threshold_mode
        self.fixed_threshold = float(fixed_threshold)
        self.grad_clip = grad_clip

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

    # ------------------------------------------------------------------
    # Train / validate loops.
    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader, *, phase: str) -> tuple[float, float, float]:
        self.spectrum_tower.train()
        if phase == "joint":
            self.molecule_tower.train()
        total = 0.0
        total_bce = 0.0
        total_nce = 0.0
        n_seen = 0
        for batch in loader:
            spectra, labels, smiles_list = _unpack_batch(batch, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)

            embedding, z_spec = self.spectrum_tower.forward_with_embedding(spectra)
            logits = self.spectrum_tower.encoder.classifier(embedding)
            loss_bce = self.bce_loss(logits, labels)

            if phase == "joint":
                z_mol = self.molecule_tower(smiles_list)
                loss_nce = self.infonce_loss(z_spec, z_mol)
                loss = self.bce_weight * loss_bce + self.nce_weight * loss_nce
            else:
                loss_nce = torch.zeros((), device=self.device)
                loss = self.bce_weight * loss_bce

            loss.backward()
            if self.grad_clip is not None:
                trainable = [p for p in self._all_trainable_params() if p.grad is not None]
                if trainable:
                    nn.utils.clip_grad_norm_(trainable, self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            batch_size = spectra.size(0)
            total += float(loss.detach().cpu()) * batch_size
            total_bce += float(loss_bce.detach().cpu()) * batch_size
            total_nce += float(loss_nce.detach().cpu()) * batch_size
            n_seen += batch_size
        n = max(n_seen, 1)
        return total / n, total_bce / n, total_nce / n

    def _validate(
        self, loader: DataLoader, *, phase: str
    ) -> tuple[
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        Tensor | None,
        Tensor | None,
    ]:
        self.spectrum_tower.eval()
        self.molecule_tower.eval()
        total_bce = 0.0
        total_nce = 0.0
        n_seen = 0
        all_probs: list[Tensor] = []
        all_targets: list[Tensor] = []
        all_z_spec: list[Tensor] = []
        all_z_mol: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                spectra, labels, smiles_list = _unpack_batch(batch, device=self.device)
                embedding, z_spec = self.spectrum_tower.forward_with_embedding(spectra)
                logits = self.spectrum_tower.encoder.classifier(embedding)
                loss_bce = self.bce_loss(logits, labels)
                if phase == "joint":
                    z_mol = self.molecule_tower(smiles_list)
                    loss_nce = self.infonce_loss(z_spec, z_mol)
                    all_z_spec.append(z_spec.cpu())
                    all_z_mol.append(z_mol.cpu())
                else:
                    loss_nce = torch.zeros((), device=self.device)
                batch_size = spectra.size(0)
                total_bce += float(loss_bce.detach().cpu()) * batch_size
                total_nce += float(loss_nce.detach().cpu()) * batch_size
                n_seen += batch_size
                all_probs.append(torch.sigmoid(logits).cpu())
                all_targets.append(labels.cpu())

        n = max(n_seen, 1)
        probs = torch.cat(all_probs).numpy() if all_probs else np.zeros((0, 0))
        targets_arr = (
            torch.cat(all_targets).numpy().astype(np.int64) if all_targets else np.zeros((0, 0))
        )
        z_spec_full = torch.cat(all_z_spec) if all_z_spec else None
        z_mol_full = torch.cat(all_z_mol) if all_z_mol else None
        total_loss = self.bce_weight * (total_bce / n) + (
            self.nce_weight * (total_nce / n) if phase == "joint" else 0.0
        )
        return (
            total_loss,
            total_bce / n,
            total_nce / n,
            targets_arr,
            probs,
            z_spec_full,
            z_mol_full,
        )

    def _all_trainable_params(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for module in (self.spectrum_tower, self.molecule_tower, self.infonce_loss):
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    # ------------------------------------------------------------------
    # Threshold / metric helpers.
    # ------------------------------------------------------------------
    def _resolve_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        if self.threshold_mode == "fixed":
            return np.full(y_true.shape[1], self.fixed_threshold, dtype=np.float64)
        return search_thresholds(y_true, y_prob, default=self.fixed_threshold)

    # ------------------------------------------------------------------
    # Checkpoints.
    # ------------------------------------------------------------------
    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        phase: str,
        thresholds: np.ndarray,
        metrics: dict[str, Any],
        config: dict[str, Any] | None,
    ) -> None:
        payload = {
            "spectrum_tower_state_dict": self.spectrum_tower.state_dict(),
            # Энкодер ChemBERTa не сохраняем — он заморожен; восстановим
            # из HF-кэша по имени модели. Сохраняем только trainable части.
            "molecule_projection_state_dict": self.molecule_tower.projection.state_dict(),
            "molecule_model_name": self.molecule_tower.model_name,
            "infonce_state_dict": self.infonce_loss.state_dict(),
            "thresholds": thresholds.tolist(),
            "metrics": metrics,
            "epoch": epoch,
            "phase": phase,
            "class_names": list(self.class_names),
            "config": config,
        }
        torch.save(payload, path)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
        patience: int,
        config_snapshot: dict[str, Any] | None = None,
    ) -> ContrastiveTrainState:
        history: list[ContrastiveEpochStats] = []
        best_score = -1.0
        best_macro_f1 = -1.0
        best_top1 = -1.0
        best_epoch = -1
        best_thresholds = np.full(len(self.class_names), self.fixed_threshold, dtype=np.float64)
        epochs_without_improvement = 0
        stopped_early = False

        last_path = self.output_dir / "last.pt"
        best_path = self.output_dir / "best.pt"

        for epoch in range(1, epochs + 1):
            phase = "warmup" if epoch <= self.warmup_epochs else "joint"
            train_loss, train_bce, train_nce = self._train_epoch(train_loader, phase=phase)
            (
                val_loss,
                val_bce,
                val_nce,
                y_true,
                y_prob,
                z_spec_val,
                z_mol_val,
            ) = self._validate(val_loader, phase=phase)
            thresholds = self._resolve_thresholds(y_true, y_prob)
            metrics = compute_metrics(
                y_true, y_prob, thresholds, class_names=list(self.class_names)
            )
            metrics["temperature"] = float(self.infonce_loss.temperature)  # type: ignore[attr-defined]
            if z_spec_val is not None and z_mol_val is not None:
                metrics["retrieval"] = self._retrieval_metrics(z_spec_val, z_mol_val)
            else:
                metrics["retrieval"] = {"top1": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0}

            stats = ContrastiveEpochStats(
                epoch=epoch,
                phase=phase,
                train_loss=train_loss,
                train_loss_bce=train_bce,
                train_loss_nce=train_nce,
                val_loss=val_loss,
                val_loss_bce=val_bce,
                val_loss_nce=val_nce,
                metrics=metrics,
            )
            history.append(stats)
            self._log_epoch(stats)

            self._save_checkpoint(
                last_path,
                epoch=epoch,
                phase=phase,
                thresholds=thresholds,
                metrics=metrics,
                config=config_snapshot,
            )

            score = self._combined_score(metrics, phase=phase)
            # Warm-up даёт неинформативный top1 (retrieval не считается),
            # поэтому best/early-stopping применяются только в joint-фазе. До
            # первой joint-эпохи best_path всё равно создаётся — чтобы FAISS
            # был воспроизводим, даже если обучение прервётся.
            if phase == "warmup":
                if best_epoch < 0:
                    self._save_checkpoint(
                        best_path,
                        epoch=epoch,
                        phase=phase,
                        thresholds=thresholds,
                        metrics=metrics,
                        config=config_snapshot,
                    )
                    best_macro_f1 = float(metrics["macro_f1"])
                    best_epoch = epoch
                    best_thresholds = thresholds
            else:
                improved = score > best_score or best_epoch <= self.warmup_epochs
                if improved:
                    best_score = score
                    best_macro_f1 = float(metrics["macro_f1"])
                    best_top1 = float(metrics["retrieval"]["top1"])
                    best_epoch = epoch
                    best_thresholds = thresholds
                    epochs_without_improvement = 0
                    self._save_checkpoint(
                        best_path,
                        epoch=epoch,
                        phase=phase,
                        thresholds=thresholds,
                        metrics=metrics,
                        config=config_snapshot,
                    )
                else:
                    epochs_without_improvement += 1

            self._write_metrics_json(history)

            if phase == "joint" and epochs_without_improvement >= patience:
                stopped_early = True
                log.info(
                    "early_stopping",
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_score=round(best_score, 4),
                )
                break

        self.writer.flush()
        self.writer.close()
        return ContrastiveTrainState(
            best_epoch=best_epoch,
            best_score=best_score,
            best_macro_f1=best_macro_f1,
            best_top1=best_top1,
            best_thresholds=best_thresholds.tolist(),
            history=history,
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------------------
    # Logging.
    # ------------------------------------------------------------------
    def _retrieval_metrics(self, z_spec: Tensor, z_mol: Tensor) -> dict[str, float]:
        topk = topk_accuracy(z_spec, z_mol, ks=(1, 5, 10))
        mrr = mean_reciprocal_rank(z_spec, z_mol)
        return {**topk, "mrr": mrr}

    @staticmethod
    def _combined_score(metrics: dict[str, Any], *, phase: str) -> float:
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        if phase == "joint":
            top1 = float(metrics.get("retrieval", {}).get("top1", 0.0))
            return 0.5 * macro_f1 + 0.5 * top1
        return macro_f1

    def _log_epoch(self, stats: ContrastiveEpochStats) -> None:
        retrieval = stats.metrics.get("retrieval", {})
        log.info(
            "epoch_completed",
            epoch=stats.epoch,
            phase=stats.phase,
            train_loss=round(stats.train_loss, 5),
            val_loss=round(stats.val_loss, 5),
            macro_f1=round(float(stats.metrics.get("macro_f1", 0.0)), 4),
            top1=round(float(retrieval.get("top1", 0.0)), 4),
            top5=round(float(retrieval.get("top5", 0.0)), 4),
            mrr=round(float(retrieval.get("mrr", 0.0)), 4),
            temperature=round(float(stats.metrics.get("temperature", 0.0)), 4),
        )
        self.writer.add_scalar("loss/train_total", stats.train_loss, stats.epoch)
        self.writer.add_scalar("loss/train_bce", stats.train_loss_bce, stats.epoch)
        self.writer.add_scalar("loss/train_nce", stats.train_loss_nce, stats.epoch)
        self.writer.add_scalar("loss/val_total", stats.val_loss, stats.epoch)
        self.writer.add_scalar("loss/val_bce", stats.val_loss_bce, stats.epoch)
        self.writer.add_scalar("loss/val_nce", stats.val_loss_nce, stats.epoch)
        for key in ("macro_f1", "micro_f1", "macro_ap", "macro_auc", "hamming_loss"):
            value = stats.metrics.get(key)
            if isinstance(value, int | float):
                fvalue = float(value)
                if not np.isnan(fvalue):
                    self.writer.add_scalar(f"val/{key}", fvalue, stats.epoch)
        for key, value in retrieval.items():
            fvalue = float(value)
            if not np.isnan(fvalue):
                self.writer.add_scalar(f"retrieval/{key}", fvalue, stats.epoch)
        self.writer.add_scalar(
            "infonce/temperature",
            float(stats.metrics.get("temperature", 0.0)),
            stats.epoch,
        )

    def _write_metrics_json(self, history: list[ContrastiveEpochStats]) -> None:
        path = self.run_dir / "metrics.json"
        payload = {
            "history": [
                {
                    "epoch": s.epoch,
                    "phase": s.phase,
                    "train_loss": s.train_loss,
                    "train_loss_bce": s.train_loss_bce,
                    "train_loss_nce": s.train_loss_nce,
                    "val_loss": s.val_loss,
                    "val_loss_bce": s.val_loss_bce,
                    "val_loss_nce": s.val_loss_nce,
                    "metrics": _strip_nan(s.metrics),
                }
                for s in history
            ]
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _unpack_batch(batch: Any, *, device: torch.device) -> tuple[Tensor, Tensor, list[str]]:
    """Извлекает (spectrum, labels, smiles_list) из батча DataLoader."""
    if len(batch) != 3:
        raise ValueError(
            "DataLoader для contrastive должен возвращать (spectrum, labels, smiles); "
            "проверь, что SpectraDataset создан с return_smiles=True"
        )
    spectra, labels, smiles_list = batch
    spectra = spectra.to(device)
    labels = labels.to(device)
    if isinstance(smiles_list, tuple):
        smiles_list = list(smiles_list)
    return spectra, labels, smiles_list


def _strip_nan(metrics: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            cleaned[key] = None
        elif isinstance(value, dict):
            cleaned[key] = _strip_nan(value)
        else:
            cleaned[key] = value
    return cleaned


__all__ = [
    "ContrastiveEpochStats",
    "ContrastiveTrainState",
    "ContrastiveTrainer",
]
