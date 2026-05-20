"""PyTorch-датасет ``SpectraDataset`` для синтетического parquet-датасета.

Читает заранее подготовленный parquet (``ml/data/synthetic.parquet`` на фазе 1)
с колонками:
- ``spectrum``: список/массив float длины 3601 (по CLAUDE.md §7);
- ``labels``: список/массив 0/1 длины 25 (multi-hot, см. :mod:`labeling`);
- ``smiles``: строка SMILES;
- ``compound_id``: целочисленный идентификатор.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset

SpectrumTransform = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

# Тип элемента датасета: (spectrum, labels) либо (spectrum, labels, smiles)
# в зависимости от ``return_smiles``. PyTorch DataLoader корректно собирает
# батч из тензоров и список строк через стандартный collate_fn.
DatasetItem = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]


class SpectraDataset(Dataset[DatasetItem]):
    """ИК-спектры с multi-hot метками для обучения 1D-CNN и контрастной схемы.

    Args:
        parquet_path: путь к parquet-файлу.
        transform: опциональная аугментация ``ndarray -> ndarray``.
            Применяется к каждому спектру в ``__getitem__``.
        spectrum_length: ожидаемая длина вектора спектра (3601 по умолчанию).
        n_labels: размерность multi-hot метки (25 функциональных групп).
        return_smiles: если True, элемент возвращается как
            ``(spectrum, labels, smiles)`` — нужно для contrastive-обучения
            (Этап 6). По умолчанию False, обратная совместимость с Этапами 4/5.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        transform: SpectrumTransform | None = None,
        spectrum_length: int = 3601,
        n_labels: int = 25,
        return_smiles: bool = False,
    ) -> None:
        self._frame: pd.DataFrame = pd.read_parquet(parquet_path, engine="pyarrow")
        self._transform = transform
        self._spectrum_length = spectrum_length
        self._n_labels = n_labels
        self._return_smiles = return_smiles

        missing = {"spectrum", "labels", "smiles"} - set(self._frame.columns)
        if missing:
            raise ValueError(f"в parquet не хватает колонок: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self._frame.iloc[idx]
        spectrum = np.asarray(row["spectrum"], dtype=np.float64)
        if spectrum.size != self._spectrum_length:
            raise ValueError(
                f"строка {idx}: длина спектра {spectrum.size} != {self._spectrum_length}"
            )
        if self._transform is not None:
            spectrum = self._transform(spectrum)

        labels = np.asarray(row["labels"], dtype=np.float32)
        if labels.size != self._n_labels:
            raise ValueError(f"строка {idx}: размер меток {labels.size} != {self._n_labels}")

        spectrum_tensor = torch.from_numpy(spectrum.astype(np.float32))
        labels_tensor = torch.from_numpy(labels)
        if self._return_smiles:
            return spectrum_tensor, labels_tensor, str(row["smiles"])
        return spectrum_tensor, labels_tensor

    def get_metadata(self, idx: int) -> dict[str, object]:
        """Метаданные строки без преобразования в тензор."""
        row = self._frame.iloc[idx]
        if "compound_id" in self._frame.columns:
            compound_id = int(row["compound_id"])
        elif "id" in self._frame.columns:
            compound_id = int(row["id"])
        else:
            compound_id = int(idx)
        return {
            "smiles": cast(str, row["smiles"]),
            "compound_id": compound_id,
        }
