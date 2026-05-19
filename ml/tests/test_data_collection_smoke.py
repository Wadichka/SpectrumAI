"""Smoke-тест предзащитного data-collection-пайплайна (фаза 2, этап 18).

Цепочка: merge_predefense → apply_preprocessing → apply_labeling →
compute_stats. Использует синтезированные JCAMP-DX в tmp-каталоге,
чтобы не зависеть от реального NIST-скрейпинга.
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
def fake_nist_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "nist"
    raw_dir.mkdir()
    # 2 валидных + 1 битый по покрытию диапазона.
    (raw_dir / "64-17-5.jdx").write_bytes(
        _build_synthetic_jcamp(390.0, 4010.0, 250, "ethanol")
    )
    (raw_dir / "67-64-1.jdx").write_bytes(
        _build_synthetic_jcamp(395.0, 3995.0, 250, "acetone")
    )
    # Слишком узкий диапазон (200 см⁻¹ из 3600 нужных) → должен быть отброшен.
    (raw_dir / "999-99-9.jdx").write_bytes(
        _build_synthetic_jcamp(500.0, 700.0, 50, "narrow_band")
    )
    inchi_path = raw_dir / "inchi.txt"
    inchi_path.write_text(
        "\n".join(
            [
                "64-17-5\tInChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # ethanol
                "67-64-1\tInChI=1S/C3H6O/c1-3(2)4/h1-2H3",  # acetone
                "999-99-9\tInChI=1S/CH4/h1H4",  # methane (won't reach this)
            ]
        ),
        encoding="utf-8",
    )
    return raw_dir


def test_full_pipeline_on_synthetic_nist(tmp_path: Path, fake_nist_dir: Path) -> None:
    processed_dir = tmp_path / "processed"
    quarantine_dir = tmp_path / "quarantine"

    raw_parquet = processed_dir / "predefense_spectra.parquet"
    merge_stats = merge_predefense(
        raw_dir=fake_nist_dir,
        inchi_path=fake_nist_dir / "inchi.txt",
        output_parquet=raw_parquet,
        quarantine_dir=quarantine_dir,
    )
    assert merge_stats["valid"] >= 2
    assert merge_stats["rejected"] >= 1  # узкий 500-700 см⁻¹
    assert merge_stats["final"] >= 2
    assert raw_parquet.exists()

    normalized_parquet = processed_dir / "predefense_normalized.parquet"
    kept = apply_preprocessing(raw_parquet, normalized_parquet)
    assert kept == merge_stats["final"]
    df_norm = pd.read_parquet(normalized_parquet)
    assert len(df_norm.iloc[0]["spectrum"]) == 3601
    assert "spectrum_raw" not in df_norm.columns

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
    assert isinstance(stats["positives_per_class"], dict)
