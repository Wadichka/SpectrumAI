"""Smoke-тест предзащитного data-collection-пайплайна (фаза 2, этап 18).

Цепочка: select_predefense_subset → merge_predefense → apply_preprocessing →
apply_labeling → compute_stats. Использует синтезированные NistChemData-
структуры в tmp-каталоге, чтобы не зависеть от реального git clone.
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
from ml.scripts.data_collection.merge_predefense import merge_predefense  # noqa: E402
from ml.scripts.data_collection.select_predefense_subset import (  # noqa: E402
    select_predefense_subset,
)


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
def fake_nist_chemdata(tmp_path: Path) -> Path:
    """Создаёт мини-NistChemData-структуру: 4 JDX + nist_compounds.csv + nist_ir_info.csv."""
    raw_dir = tmp_path / "nist"
    raw_dir.mkdir()

    (raw_dir / "C64175.jdx").write_bytes(
        _build_synthetic_jcamp(390.0, 4010.0, 250, "ethanol")
    )
    (raw_dir / "C67641.jdx").write_bytes(
        _build_synthetic_jcamp(395.0, 3995.0, 250, "acetone")
    )
    (raw_dir / "C50000.jdx").write_bytes(
        _build_synthetic_jcamp(388.0, 4005.0, 230, "formaldehyde")
    )
    (raw_dir / "C99999.jdx").write_bytes(
        _build_synthetic_jcamp(500.0, 700.0, 50, "narrow_band")
    )

    pd.DataFrame(
        [
            {
                "nist_id": "C64175",
                "name": "ethanol",
                "cas": "64-17-5",
                "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            },
            {
                "nist_id": "C67641",
                "name": "acetone",
                "cas": "67-64-1",
                "inchi": "InChI=1S/C3H6O/c1-3(2)4/h1-2H3",
                "inchi_key": "CSCPPACGZOOCGX-UHFFFAOYSA-N",
            },
            {
                "nist_id": "C50000",
                "name": "formaldehyde",
                "cas": "50-00-0",
                "inchi": "InChI=1S/CH2O/c1-2/h1H2",
                "inchi_key": "WSFSSNUMVMOOMR-UHFFFAOYSA-N",
            },
            {
                "nist_id": "C99999",
                "name": "narrow_band",
                "cas": "99-99-9",
                "inchi": "InChI=1S/CH4/h1H4",
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            },
        ]
    ).to_csv(raw_dir / "nist_compounds.csv", index=False)

    pd.DataFrame(
        [
            {"nist_id": "C64175", "file": "C64175.jdx", "state": "liquid"},
            {"nist_id": "C67641", "file": "C67641.jdx", "state": "liquid"},
            {"nist_id": "C50000", "file": "C50000.jdx", "state": "gas"},
            {"nist_id": "C99999", "file": "C99999.jdx", "state": "solid"},
        ]
    ).to_csv(raw_dir / "nist_ir_info.csv", index=False)

    return raw_dir


def test_full_pipeline_on_synthetic_nist_chemdata(
    tmp_path: Path, fake_nist_chemdata: Path
) -> None:
    processed_dir = tmp_path / "processed"
    quarantine_dir = tmp_path / "quarantine"

    # 1. select — формируем subset из 4 соединений (с target_per_group=5 берёт всех).
    subset_csv = processed_dir / "predefense_subset_ids.csv"
    select_stats = select_predefense_subset(
        compounds_csv=fake_nist_chemdata / "nist_compounds.csv",
        ir_info_csv=fake_nist_chemdata / "nist_ir_info.csv",
        jdx_dir=fake_nist_chemdata,
        output_csv=subset_csv,
        total=10,
        target_per_group=5,
        seed=42,
    )
    assert select_stats["selected"] >= 2
    df_subset = pd.read_csv(subset_csv)
    assert "nist_id" in df_subset.columns
    assert "smiles" in df_subset.columns

    # 2. merge — узкий C99999 отсекается по coverage, если он попал в subset.
    raw_parquet = processed_dir / "predefense_spectra.parquet"
    merge_stats = merge_predefense(
        subset_csv=subset_csv,
        compounds_csv=fake_nist_chemdata / "nist_compounds.csv",
        ir_info_csv=fake_nist_chemdata / "nist_ir_info.csv",
        jdx_dir=fake_nist_chemdata,
        output_parquet=raw_parquet,
        quarantine_dir=quarantine_dir,
    )
    assert merge_stats["valid"] >= 2
    assert merge_stats["final"] >= 2
    df_raw = pd.read_parquet(raw_parquet)
    assert "state" in df_raw.columns
    assert df_raw["state"].notna().any()

    # 3. preprocess.
    normalized_parquet = processed_dir / "predefense_normalized.parquet"
    kept = apply_preprocessing(raw_parquet, normalized_parquet)
    assert kept == merge_stats["final"]
    df_norm = pd.read_parquet(normalized_parquet)
    assert len(df_norm.iloc[0]["spectrum"]) == 3601
    assert "spectrum_raw" not in df_norm.columns
    assert "state" in df_norm.columns

    # 4. labeling.
    labeled_parquet = processed_dir / "predefense_labeled.parquet"
    labeled_count = apply_labeling(normalized_parquet, labeled_parquet)
    assert labeled_count == kept
    df_lab = pd.read_parquet(labeled_parquet)
    assert "labels" in df_lab.columns
    assert len(df_lab.iloc[0]["labels"]) == 25

    # 5. stats.
    stats_json = processed_dir / "predefense_stats.json"
    stats = compute_stats(labeled_parquet, stats_json)
    assert stats["total_spectra"] == labeled_count
    assert stats["n_groups"] == 25
    assert stats["spectrum_length"] == 3601
    assert isinstance(stats["state_distribution"], dict)
    assert sum(stats["state_distribution"].values()) == labeled_count
