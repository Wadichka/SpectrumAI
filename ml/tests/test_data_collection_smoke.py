"""Smoke-тест предзащитного data-collection-пайплайна (фаза 2, этап 18).

Цепочка: select_subset → merge_nist_dataset → apply_preprocessing →
apply_labeling → compute_stats. Имитирует структуру выгрузки NistChemPy
(JDX-файлы вида ``{ID}_IR_{idx}.jdx`` + ``nist_compounds.csv`` со схемой
``ID``/``inchi``/``inchi_key``/``cas_rn``/``name``/``IR Spectrum``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ML_ROOT = _REPO_ROOT / "ml"
_BACKEND_ROOT = _REPO_ROOT / "backend"
for path in (_ML_ROOT, _BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml.scripts.data_collection.apply_labeling import apply_labeling  # noqa: E402
from ml.scripts.data_collection.apply_preprocessing import apply_preprocessing  # noqa: E402
from ml.scripts.data_collection.compute_stats import compute_stats  # noqa: E402
from ml.scripts.data_collection.merge_nist_dataset import merge_nist_dataset  # noqa: E402
from ml.scripts.data_collection.select_subset import select_subset  # noqa: E402


def _build_synthetic_jcamp(start: float, stop: float, npoints: int, title: str) -> bytes:
    """Минимальный валидный JCAMP-DX-файл с покрытием 400–4000 см⁻¹."""
    step = (stop - start) / (npoints - 1)
    intensities = [0.1 + 0.001 * i for i in range(npoints)]
    data_line = f"{start:.0f} " + " ".join(f"{v:.3f}" for v in intensities)
    return (
        f"##TITLE={title}\n"
        f"##JCAMP-DX=4.24\n"
        f"##DATA TYPE=INFRARED SPECTRUM\n"
        f"##XUNITS=1/CM\n"
        f"##YUNITS=ABSORBANCE\n"
        f"##FIRSTX={start:.1f}\n"
        f"##LASTX={start + step * (npoints - 1):.1f}\n"
        f"##NPOINTS={npoints}\n"
        f"##XFACTOR=1\n"
        f"##YFACTOR=1\n"
        f"##XYDATA=(X++(Y..Y))\n"
        f"{data_line}\n"
        f"##END=\n"
    ).encode()


@pytest.fixture()
def fake_nistchempy_catalog(tmp_path: Path) -> Path:
    """Создаёт мини-каталог NistChemPy: 4 JDX + nist_compounds.csv."""
    raw_dir = tmp_path / "nist"
    raw_dir.mkdir()

    (raw_dir / "C64175_IR_0.jdx").write_bytes(
        _build_synthetic_jcamp(390.0, 4010.0, 250, "ethanol")
    )
    (raw_dir / "C67641_IR_0.jdx").write_bytes(
        _build_synthetic_jcamp(395.0, 3995.0, 250, "acetone")
    )
    (raw_dir / "C50000_IR_0.jdx").write_bytes(
        _build_synthetic_jcamp(388.0, 4005.0, 230, "formaldehyde")
    )
    (raw_dir / "C99999_IR_0.jdx").write_bytes(
        _build_synthetic_jcamp(500.0, 700.0, 50, "narrow_band")
    )

    ir_url = "https://webbook.nist.gov/cgi/cbook.cgi?ID=stub&Mask=80"
    pd.DataFrame(
        [
            {
                "ID": "C64175",
                "name": "ethanol",
                "cas_rn": "64-17-5",
                "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "IR Spectrum": ir_url,
            },
            {
                "ID": "C67641",
                "name": "acetone",
                "cas_rn": "67-64-1",
                "inchi": "InChI=1S/C3H6O/c1-3(2)4/h1-2H3",
                "inchi_key": "CSCPPACGZOOCGX-UHFFFAOYSA-N",
                "IR Spectrum": ir_url,
            },
            {
                "ID": "C50000",
                "name": "formaldehyde",
                "cas_rn": "50-00-0",
                "inchi": "InChI=1S/CH2O/c1-2/h1H2",
                "inchi_key": "WSFSSNUMVMOOMR-UHFFFAOYSA-N",
                "IR Spectrum": ir_url,
            },
            {
                "ID": "C99999",
                "name": "narrow_band",
                "cas_rn": "99-99-9",
                "inchi": "InChI=1S/CH4/h1H4",
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                "IR Spectrum": ir_url,
            },
        ]
    ).to_csv(raw_dir / "nist_compounds.csv", index=False)

    return raw_dir


def test_full_pipeline_on_synthetic_nistchempy_catalog(
    tmp_path: Path, fake_nistchempy_catalog: Path
) -> None:
    processed_dir = tmp_path / "processed"
    quarantine_dir = tmp_path / "quarantine"

    subset_csv = processed_dir / "predefense_subset_ids.csv"
    select_stats = select_subset(
        input_catalog=fake_nistchempy_catalog / "nist_compounds.csv",
        jdx_dir=fake_nistchempy_catalog,
        output_csv=subset_csv,
        target_size=10,
        target_per_group=5,
        seed=42,
    )
    assert select_stats["selected"] >= 2
    df_subset = pd.read_csv(subset_csv)
    assert "nist_id" in df_subset.columns
    assert "smiles" in df_subset.columns
    assert "target_groups" in df_subset.columns

    raw_parquet = processed_dir / "predefense_spectra.parquet"
    merge_stats = merge_nist_dataset(
        ids_csv=subset_csv,
        catalog_csv=fake_nistchempy_catalog / "nist_compounds.csv",
        jdx_dir=fake_nistchempy_catalog,
        output_parquet=raw_parquet,
        quarantine_dir=quarantine_dir,
    )
    assert merge_stats["valid"] >= 2
    assert merge_stats["final"] >= 2
    df_raw = pd.read_parquet(raw_parquet)
    assert "state" in df_raw.columns
    assert df_raw["state"].isna().all()

    normalized_parquet = processed_dir / "predefense_normalized.parquet"
    kept = apply_preprocessing(raw_parquet, normalized_parquet)
    assert kept == merge_stats["final"]
    df_norm = pd.read_parquet(normalized_parquet)
    assert len(df_norm.iloc[0]["spectrum"]) == 3601
    assert "spectrum_raw" not in df_norm.columns
    assert "state" in df_norm.columns

    labeled_parquet = processed_dir / "predefense_labeled.parquet"
    labeled_count = apply_labeling(normalized_parquet, labeled_parquet)
    assert labeled_count == kept
    df_lab = pd.read_parquet(labeled_parquet)
    assert "labels" in df_lab.columns
    assert len(df_lab.iloc[0]["labels"]) == 25

    stats_json = processed_dir / "predefense_stats.json"
    stats = compute_stats(labeled_parquet, stats_json)
    assert stats["total_spectra"] == labeled_count
    assert stats["n_groups"] == 25
    assert stats["spectrum_length"] == 3601
    assert isinstance(stats["state_distribution"], dict)
