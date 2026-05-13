"""ORM-модели SpectrumAI.

Импорт всех модулей регистрирует таблицы в ``Base.metadata`` — это критично
для Alembic-автогенерации и для ``Base.metadata.create_all`` в тестах.
"""

from app.db.base import Base
from app.db.models.cache_entry import CacheEntry
from app.db.models.compound import Compound
from app.db.models.compound_functional_group import CompoundFunctionalGroup
from app.db.models.functional_group import FunctionalGroup
from app.db.models.identification_request import IdentificationRequest
from app.db.models.identification_result import IdentificationResult
from app.db.models.model_version import ModelVersion
from app.db.models.predicted_functional_group import PredictedFunctionalGroup
from app.db.models.spectrum import Spectrum
from app.db.models.spectrum_embedding import SpectrumEmbedding
from app.db.models.user import User

__all__ = [
    "Base",
    "CacheEntry",
    "Compound",
    "CompoundFunctionalGroup",
    "FunctionalGroup",
    "IdentificationRequest",
    "IdentificationResult",
    "ModelVersion",
    "PredictedFunctionalGroup",
    "Spectrum",
    "SpectrumEmbedding",
    "User",
]
