"""Тесты ``SpectrumRepository`` и ``SpectrumEmbeddingRepository``."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models.compound import Compound
from app.db.models.spectrum import Spectrum
from app.db.repositories.compound import CompoundRepository
from app.db.repositories.spectrum import SpectrumRepository
from app.db.repositories.spectrum_embedding import SpectrumEmbeddingRepository


def _make_compound(inchi_key: str = "K-UNIQ-0001") -> Compound:
    return Compound(
        smiles_canonical="CCO",
        inchi="InChI=1S/C2H6O",
        inchi_key=inchi_key,
    )


def _make_spectrum(compound_id: int, source: str = "NIST") -> Spectrum:
    return Spectrum(
        compound_id=compound_id,
        source=source,
        phase="liquid",
        technique="ATR",
        wavenumber_min=400.0,
        wavenumber_max=4000.0,
        n_points=3601,
        file_path=f"/data/spectra/{compound_id}.jdx",
    )


def test_add_spectrum_under_compound(db_session: Session) -> None:
    compounds = CompoundRepository(db_session)
    spectra = SpectrumRepository(db_session)
    c = compounds.add(_make_compound())
    s = spectra.add(_make_spectrum(c.id))
    assert s.id is not None
    assert s.compound_id == c.id


def test_list_by_compound_id(db_session: Session) -> None:
    compounds = CompoundRepository(db_session)
    spectra = SpectrumRepository(db_session)
    c = compounds.add(_make_compound())
    spectra.add(_make_spectrum(c.id, source="NIST"))
    spectra.add(_make_spectrum(c.id, source="SDBS"))
    result = spectra.list_by_compound_id(c.id)
    assert {s.source for s in result} == {"NIST", "SDBS"}


def test_compound_delete_cascades_to_spectrum(db_session: Session) -> None:
    compounds = CompoundRepository(db_session)
    spectra = SpectrumRepository(db_session)
    c = compounds.add(_make_compound())
    spectra.add(_make_spectrum(c.id))
    db_session.commit()
    compounds.delete(c)
    db_session.commit()
    assert spectra.count() == 0


def test_embedding_upsert_inserts_then_updates(db_session: Session) -> None:
    compounds = CompoundRepository(db_session)
    spectra = SpectrumRepository(db_session)
    embeddings = SpectrumEmbeddingRepository(db_session)
    c = compounds.add(_make_compound())
    s = spectra.add(_make_spectrum(c.id))

    first = embeddings.upsert(
        spectrum_id=s.id,
        embedding_vector=b"\x00\x01\x02",
        model_version="encoder-0.1.0",
    )
    assert first.embedding_vector == b"\x00\x01\x02"

    second = embeddings.upsert(
        spectrum_id=s.id,
        embedding_vector=b"\xff",
        model_version="encoder-0.2.0",
    )
    assert second.spectrum_id == first.spectrum_id
    assert second.embedding_vector == b"\xff"
    assert second.model_version == "encoder-0.2.0"
    assert embeddings.count() == 1


def test_get_with_embedding_eager_loads(db_session: Session) -> None:
    compounds = CompoundRepository(db_session)
    spectra = SpectrumRepository(db_session)
    embeddings = SpectrumEmbeddingRepository(db_session)
    c = compounds.add(_make_compound())
    s = spectra.add(_make_spectrum(c.id))
    embeddings.upsert(spectrum_id=s.id, embedding_vector=b"v", model_version="v1")
    db_session.commit()
    loaded = spectra.get_with_embedding(s.id)
    assert loaded is not None
    assert loaded.embedding is not None
    assert loaded.embedding.embedding_vector == b"v"
