"""ML-инференс backend'а (§4.4.3, §4.4.5 главы 4).

Импорт ``_ml_path`` обязателен первым: он подкладывает корень ``ml/`` в
``sys.path`` для последующих ``from pipelines.* import ...``.
"""

from app.ml import _ml_path  # noqa: F401 — side-effect: sys.path += ml/
from app.ml.components import MLComponents, get_ml_components, reset_ml_components

__all__ = ["MLComponents", "get_ml_components", "reset_ml_components"]
