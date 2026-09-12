"""
Setup logging terpusat untuk aplikasi.
Output ke console + file dengan format konsisten.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import settings


# ============================================
# FORMAT
# ============================================
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Format warna untuk console (opsional, biar enak dibaca)
COLOR_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"


# ============================================
# COLORED FORMATTER (untuk console)
# ============================================
class ColoredFormatter(logging.Formatter):
    """
    Formatter dengan warna untuk level log berbeda.
    Hanya dipakai untuk console (bukan file).
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ============================================
# SETUP LOGGER
# ============================================
def setup_logger() -> logging.Logger:
    """
    Setup root logger aplikasi.
    - Console handler (dengan warna)
    - File handler (rotating, max 5MB x 5 file)
    """
    logger = logging.getLogger("app")
    logger.setLevel(settings.LOG_LEVEL.upper())

    # Hindari duplicate handler kalau function dipanggil ulang
    if logger.handlers:
        return logger

    # ---------- CONSOLE HANDLER ----------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.LOG_LEVEL.upper())

    # Pakai warna kalau di terminal (bukan di-pipe ke file)
    if sys.stdout.isatty():
        console_formatter = ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ---------- FILE HANDLER ----------
    log_dir: Path = settings.log_dir_absolute
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "app.log"

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,             # Simpan 5 file lama
        encoding="utf-8",
    )
    file_handler.setLevel(settings.LOG_LEVEL.upper())
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Jangan propagate ke root logger (biar tidak dobel)
    logger.propagate = False

    return logger


# ============================================
# GET LOGGER
# ============================================
def get_logger(name: str = "app") -> logging.Logger:
    """
    Return logger dengan nama tertentu.
    
    Contoh:
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    # Pastikan root logger sudah di-setup
    setup_logger()

    # Kalau name tidak dimulai dengan "app.", tambahkan
    if not name.startswith("app"):
        name = f"app.{name}"

    return logging.getLogger(name)


# ============================================
# SILENCE LIBRARY NOISY
# ============================================
def silence_noisy_loggers() -> None:
    """
    Turunkan level log dari library yang terlalu berisik.
    """
    noisy = [
        "uvicorn.access",
        "uvicorn.error",
        "multipart",
        "PIL",
        "matplotlib",
        "tensorflow",
        "keras",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)


# ============================================
# INIT (dijalankan saat import)
# ============================================
setup_logger()
silence_noisy_loggers()