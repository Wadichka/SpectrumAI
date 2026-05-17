"""Сервис добавления спектра в БД (UC-03)."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from sqlalchemy.orm import Session

from app.db.models.compound import Compound
from app.db.models.spectrum import Spectrum
from app.db.repositories.compound import CompoundRepository
from app.domain.errors import SpectrumValidationError
from app.parsing import parse_spectrum


@dataclass(frozen=True, slots=True)
class SpectrumCreated:
    spectrum_id: int
    compound_id: int


class SpectrumService:
    """Добавление пары (соединение, спектр) в локальную базу.

    FAISS-индекс не обновляется в этом методе — это будет Этап 21 (сидинг
    реальных данных). Здесь только запись в реляционную БД.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._compound_repo = CompoundRepository(session)

    def add_spectrum(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        smiles: str,
        name: str | None = None,
        source: str = "manual",
    ) -> SpectrumCreated:
        raw = parse_spectrum(file_bytes)
        wavenumbers = raw.wavenumbers
        n_points = int(wavenumbers.size)
        if n_points < 2:
            raise SpectrumValidationError("спектр слишком короткий: нужно ≥ 2 точек")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise SpectrumValidationError(f"invalid SMILES: {smiles!r}", field="smiles")

        canonical = Chem.MolToSmiles(mol, canonical=True)
        inchi = Chem.MolToInchi(mol)
        inchi_key = Chem.MolToInchiKey(mol)
        formula = CalcMolFormula(mol)
        mw = float(Descriptors.MolWt(mol))

        compound = self._compound_repo.get_by_inchi_key(inchi_key)
        if compound is None:
            compound = Compound(
                name=name,
                smiles_canonical=canonical,
                inchi=inchi,
                inchi_key=inchi_key,
                molecular_formula=formula,
                molecular_weight=mw,
            )
            self._session.add(compound)
            self._session.flush()

        spectrum = Spectrum(
            compound_id=compound.id,
            source=source,
            wavenumber_min=float(wavenumbers.min()),
            wavenumber_max=float(wavenumbers.max()),
            n_points=n_points,
            file_path=filename,
        )
        self._session.add(spectrum)
        self._session.flush()
        self._session.commit()

        # `AllChem` импортирован для будущей генерации Morgan-fingerprint
        # при обновлении FAISS (Этап 21); чтобы линтер не ругался на F401 —
        # обращаемся к классу здесь.
        _ = AllChem.__name__
        return SpectrumCreated(spectrum_id=spectrum.id, compound_id=compound.id)


__all__ = ["SpectrumCreated", "SpectrumService"]
