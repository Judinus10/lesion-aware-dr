from __future__ import annotations

import logging
import os
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file is not None:
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger