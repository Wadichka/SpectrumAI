"""Коррекция базовой линии методом Asymmetric Least Squares (AsLS).

Алгоритм Eilers & Boelens (2005). Минимизирует
:math:`\\sum_i w_i (y_i - z_i)^2 + \\lambda \\sum_i (\\Delta^2 z_i)^2`
с асимметричным правилом обновления весов ``w``: точки, лежащие выше
текущей оценки, штрафуются весом ``p``; ниже — ``1-p``. Это позволяет
оценке «прижиматься» к нижней огибающей и формировать гладкую базовую линию.
Реализация на ``scipy.sparse``, без внешних зависимостей (CLAUDE.md §4).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags, eye, spdiags
from scipy.sparse.linalg import spsolve

from app.preprocessing.config import PreprocessConfig


def asls_baseline(
    intensities: npt.NDArray[np.float64],
    *,
    lam: float,
    p: float,
    niter: int,
) -> npt.NDArray[np.float64]:
    """Возвращает оценку базовой линии методом AsLS.

    Args:
        intensities: вектор интенсивностей y, длина ≥ 3.
        lam: параметр гладкости λ; чем больше, тем плавнее линия.
        p: асимметрия p ∈ (0, 1); типично 0.001–0.05.
        niter: число итераций ревешивания.

    Returns:
        Массив той же длины, что и ``intensities``, оценивающий базовую линию.
    """
    y = np.asarray(intensities, dtype=np.float64)
    length = y.size
    # Вторая разностная матрица D размером (L-2, L): второе центральное
    # разностное оператор. lam * D.T @ D штрафует кривизну оценки.
    diff = diags([1.0, -2.0, 1.0], offsets=[0, 1, 2], shape=(length - 2, length)).tocsc()
    weights = np.ones(length, dtype=np.float64)
    baseline = np.zeros(length, dtype=np.float64)
    for _ in range(niter):
        w_diag = spdiags(weights, 0, length, length)
        system = w_diag + lam * (diff.T @ diff)
        baseline = spsolve(system.tocsc(), weights * y)
        weights = p * (y > baseline) + (1.0 - p) * (y <= baseline)
    return np.asarray(baseline, dtype=np.float64)


def subtract_baseline(
    intensities: npt.NDArray[np.float64],
    *,
    config: PreprocessConfig,
) -> npt.NDArray[np.float64]:
    """Вычитает AsLS-базовую линию из спектра.

    Параметры λ/p/niter берутся из ``config``.
    """
    baseline = asls_baseline(
        intensities,
        lam=config.asls_lam,
        p=config.asls_p,
        niter=config.asls_niter,
    )
    return np.asarray(intensities - baseline, dtype=np.float64)


# eye импортирован для возможного будущего использования (стартовое значение
# системы); в текущей реализации не задействован, но оставлен в импортах ради
# консистентности с описанием алгоритма в главе 4 §4.4.2.
_ = eye
