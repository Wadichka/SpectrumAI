"""
Скрипт-помощник для скачивания ИК-спектральных датасетов для проекта SpectrumAI.

Запускается из активированного .venv:
    python download_datasets.py --source nist_chemdata
    python download_datasets.py --source zenodo_chunk --chunk 1
    python download_datasets.py --source chemotion

``--source nist_chemdata`` использует пакет ``nistchempy`` для прямого
скрейпинга NIST Chemistry WebBook (см. ``download_nist_chemdata``).
Прогресс пишется в ``ml/data/raw/nist/scrape.log``, в stdout ничего не
выводится — расчёт на фоновый запуск через ``Start-Process``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "ml" / "data" / "raw"
NIST_DIR = DATA_DIR / "nist"

_LOG_FORMAT = "%(asctime)sZ [%(levelname)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def _setup_nist_logger(log_path: Path) -> logging.Logger:
    """Логгер пишет только в файл, не в stdout/stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nist_scrape")
    logger.setLevel(logging.INFO)
    # Очищаем хэндлеры на случай повторного вызова в одном процессе.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def progress_bar(block_num: int, block_size: int, total_size: int) -> None:
    """Прогресс-бар для urlretrieve (используется только в Zenodo-ветке)."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar = "█" * (percent // 2) + "·" * (50 - percent // 2)
        mb_done = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  [{bar}] {percent}%  {mb_done:.1f}/{mb_total:.1f} МБ", end="", flush=True)


def _spectra_already_downloaded(target_dir: Path, nist_id: str) -> bool:
    """Истина, если на диске есть хотя бы один JDX-файл вида ``{nist_id}_IR_*.jdx``."""
    return any(target_dir.glob(f"{nist_id}_IR_*.jdx"))


def _scrape_single_compound(
    nist_id: str,
    target_dir: Path,
    request_config: object,
    logger: logging.Logger,
    max_attempts: int = 5,
) -> tuple[int, str | None]:
    """Загружает все IR-спектры одного NIST ID.

    Возвращает ``(downloaded_spectra, error_message_or_None)``. При сетевых
    ошибках делает экспоненциальный retry до ``max_attempts`` раз.
    """
    import nistchempy

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            compound = nistchempy.get_compound(nist_id, request_config=request_config)
            if compound is None:
                return 0, "get_compound returned None"
            compound.get_ir_spectra()
            spectra = getattr(compound, "ir_specs", None) or []
            if not spectra:
                logger.info("no_ir_after_load %s", nist_id)
                return 0, None
            downloaded = 0
            for spectrum in spectra:
                try:
                    spectrum.save(path_dir=str(target_dir))
                    downloaded += 1
                except Exception as save_err:  # noqa: BLE001
                    logger.warning("save_failed %s idx=%s err=%s", nist_id,
                                   getattr(spectrum, "spec_idx", "?"), save_err)
            return downloaded, None
        except Exception as err:  # noqa: BLE001
            last_error = err
            if attempt + 1 >= max_attempts:
                break
            backoff = min(60, 2 ** attempt)
            logger.warning(
                "retry %s attempt=%d/%d wait=%ds err=%s",
                nist_id, attempt + 1, max_attempts, backoff, err,
            )
            time.sleep(backoff)
    return 0, f"{type(last_error).__name__}: {last_error}"


def download_nist_chemdata(
    *,
    refresh_catalog: bool = False,
    limit: int | None = None,
    delay: float = 1.0,
) -> None:
    """
    Скачивает каталог NIST Chemistry WebBook и ИК-спектры через ``nistchempy``.

    Алгоритм:
      1. ``nistchempy.get_all_data()`` → DataFrame на 144 795 соединений; кладём в
         ``ml/data/raw/nist/nist_compounds.csv`` (полный каталог).
      2. Отфильтровываем строки с непустым полем ``IR Spectrum`` (~16 500) и
         сохраняем их в ``nist_compounds_with_ir.csv``.
      3. Итеративно вызываем ``get_compound`` + ``get_ir_spectra`` для каждого
         ID, сохраняем JDX в ``ml/data/raw/nist/<ID>_IR_<idx>.jdx`` (имя задаёт
         сама библиотека).
      4. Прогресс пишется в ``ml/data/raw/nist/scrape.log``, в stdout/stderr
         ничего не выводится (расчёт на фоновый запуск через ``Start-Process``).
      5. Идемпотентно: при повторном запуске пропускает соединения, у которых
         JDX уже есть.

    Args:
        refresh_catalog: если True, перезагружает каталог даже при наличии
            кэшированного CSV.
        limit: если задан, обрабатывает только первые ``limit`` IDs
            (для smoke-теста).
        delay: задержка между запросами, секунд. По умолчанию 1.0 (минимум
            из ТЗ); ``nistchempy.utils.get_crawl_delay()`` возвращает 5 для
            NIST.robots.txt — этот floor сознательно не соблюдаем по
            требованию пользователя.
    """
    import nistchempy

    NIST_DIR.mkdir(parents=True, exist_ok=True)
    logger = _setup_nist_logger(NIST_DIR / "scrape.log")
    logger.info("=== nist scrape session start ===")
    logger.info("python=%s nistchempy=%s delay=%.2fs limit=%s",
                sys.version.split()[0], nistchempy.__version__, delay, limit)

    catalog_csv = NIST_DIR / "nist_compounds.csv"
    if catalog_csv.exists() and not refresh_catalog:
        logger.info("catalog_cache_hit %s", catalog_csv)
        import pandas as pd
        df = pd.read_csv(catalog_csv, dtype="str")
    else:
        logger.info("fetching nistchempy.get_all_data()")
        df = nistchempy.get_all_data()
        df.to_csv(catalog_csv, index=False)
        logger.info("catalog_saved rows=%d path=%s", len(df), catalog_csv)

    if "IR Spectrum" not in df.columns:
        logger.error("column 'IR Spectrum' missing in catalog — schema changed?")
        raise RuntimeError("Каталог nistchempy не содержит колонку 'IR Spectrum'.")

    ir_mask = df["IR Spectrum"].notna() & (df["IR Spectrum"].astype(str).str.len() > 0)
    df_ir = df.loc[ir_mask, ["ID", "name", "cas_rn", "inchi", "inchi_key", "IR Spectrum"]].copy()
    ir_index_csv = NIST_DIR / "nist_compounds_with_ir.csv"
    df_ir.to_csv(ir_index_csv, index=False)
    logger.info("ir_index_built rows=%d path=%s", len(df_ir), ir_index_csv)

    if limit is not None:
        df_ir = df_ir.head(limit)
        logger.info("limit_applied rows=%d", len(df_ir))

    request_config = nistchempy.RequestConfig(delay=delay, max_attempts=1)

    started_at = time.monotonic()
    processed = 0
    downloaded_total = 0
    skipped = 0
    errors = 0
    total = len(df_ir)
    ids = df_ir["ID"].astype(str).tolist()

    for nist_id in ids:
        processed += 1
        if _spectra_already_downloaded(NIST_DIR, nist_id):
            skipped += 1
            if processed % 500 == 0:
                logger.info("skip_batch nist_id=%s processed=%d/%d skipped=%d",
                            nist_id, processed, total, skipped)
            continue

        n, err = _scrape_single_compound(nist_id, NIST_DIR, request_config, logger)
        if err is not None:
            errors += 1
            logger.error("failure %s: %s", nist_id, err)
        else:
            downloaded_total += n
            logger.info("ok %s spectra=%d", nist_id, n)

        if processed % 100 == 0:
            elapsed = time.monotonic() - started_at
            logger.info(
                "checkpoint processed=%d/%d downloaded=%d skipped=%d errors=%d elapsed=%.1fs",
                processed, total, downloaded_total, skipped, errors, elapsed,
            )

    elapsed_total = time.monotonic() - started_at
    hh = int(elapsed_total // 3600)
    mm = int((elapsed_total % 3600) // 60)
    ss = int(elapsed_total % 60)
    logger.info(
        "обработано %d соединений, скачано %d спектров, пропущено %d, ошибок %d, общее время %02d:%02d:%02d",
        processed, downloaded_total, skipped, errors, hh, mm, ss,
    )
    logger.info("=== nist scrape session end ===")


def download_zenodo_chunk(chunk_num: int = 1) -> None:
    """
    Качает один чанк расчётного датасета IR-NMR Multimodal (Zipoli et al., IBM, 2025).
    Один чанк = ~900 МБ, ~20 000 молекул с SMILES и квантово-скорректированными ИК-спектрами.

    Лицензия: CDLA Permissive 2.0 — можно использовать в академических работах.
    Внимание: спектры РАСЧЁТНЫЕ (молекулярная динамика + квантовая коррекция), не экспериментальные.

    Цитирование: Zipoli F., Alberts M., Laino T. (2025).
    DOI: 10.5281/zenodo.16417648
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://zenodo.org/records/16417648/files/IR_data_chunk{chunk_num:03d}_of_009.parquet?download=1"
    dest = DATA_DIR / f"IR_data_chunk{chunk_num:03d}_of_009.parquet"

    if dest.exists():
        print(f"[skip] Уже скачан: {dest}")
        return

    print(f"[zenodo] Скачивание чанка {chunk_num} (~900 МБ) → {dest}")
    print("        Это может занять 5–20 минут в зависимости от скорости интернета.")
    urlretrieve(url, str(dest), reporthook=progress_bar)
    print(f"\n[zenodo] Готово: {dest.stat().st_size / 1024 / 1024:.1f} МБ")


def fetch_chemotion_via_api() -> None:
    """
    Подсказка по доступу к Chemotion repository.
    Chemotion содержит ~4175 ЭКСПЕРИМЕНТАЛЬНЫХ ИК-спектров.
    Не имеет публичного API-эндпоинта «выгрузить всё одним архивом» —
    нужна работа через REST API или ручное скачивание подборок.
    """
    print("[chemotion] Этот источник требует индивидуального подхода:")
    print("  1. Зайдите на https://www.chemotion.net/")
    print("  2. Зарегистрируйтесь (бесплатно для академического использования)")
    print("  3. Используйте поиск по типу данных = IR, экспорт через интерфейс")
    print("  4. Альтернативно — REST API, документация по адресу:")
    print("     https://www.chemotion.net/docs/repo/api")
    print()
    print("Это не автоматизируется в скрипте — нужны учётные данные и интерактивный выбор.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["nist_chemdata", "zenodo_chunk", "chemotion"],
        required=True,
        help="Какой источник использовать (по умолчанию рекомендуется nist_chemdata)",
    )
    parser.add_argument(
        "--chunk", type=int, default=1, help="Номер чанка для zenodo_chunk (1–9)"
    )
    parser.add_argument(
        "--keep-repo",
        action="store_true",
        help="Не используется в текущей реализации (оставлено для совместимости).",
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Перезагрузить каталог NIST даже при наличии локального CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Обработать только первые N соединений (для smoke-тестов).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Задержка между запросами к NIST WebBook, секунд (default 1.0).",
    )
    args = parser.parse_args()

    if args.source == "nist_chemdata":
        download_nist_chemdata(
            refresh_catalog=args.refresh_catalog,
            limit=args.limit,
            delay=args.delay,
        )
    elif args.source == "zenodo_chunk":
        download_zenodo_chunk(args.chunk)
    elif args.source == "chemotion":
        fetch_chemotion_via_api()


if __name__ == "__main__":
    main()
