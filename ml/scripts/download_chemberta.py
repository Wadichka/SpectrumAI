"""Одноразовая загрузка ChemBERTa в локальный кэш HuggingFace.

Запуск:
    python ml/scripts/download_chemberta.py [--model NAME]

После выполнения модель и токенайзер доступны без сети
(``HF_HUB_OFFLINE=1`` тоже сработает).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.models.molecule_tower import DEFAULT_MODEL_NAME  # noqa: E402

log = structlog.get_logger(__name__)


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Загружает ChemBERTa в HF-кэш.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args(argv)
    log.info("download_started", model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    log.info(
        "download_finished",
        model=args.model,
        vocab_size=len(tokenizer),
        hidden_size=int(model.config.hidden_size),
    )


if __name__ == "__main__":
    main()
