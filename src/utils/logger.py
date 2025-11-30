import logging
from pathlib import Path

def get_logger(name: str = "dr_project", log_dir: str = "outputs/logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_fmt)

    # File handler
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(ch_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
