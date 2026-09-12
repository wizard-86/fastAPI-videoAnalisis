"""
Konfigurasi aplikasi.
Semua nilai dibaca dari file .env menggunakan Pydantic Settings.
"""

from pathlib import Path
from typing import List
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================
# BASE PATH
# ============================================
# Root project = folder yang berisi folder 'app/'
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================
# SETTINGS CLASS
# ============================================
class Settings(BaseSettings):
    """
    Semua konfigurasi aplikasi.
    Nilai dibaca dari .env, fallback ke default di bawah.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- APPLICATION ----------
    APP_NAME: str = "Shoplifting Detection API"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ---------- MODEL ----------
    USE_DUMMY_MODEL: bool = True
    MODEL_PATH: str = "app/ml_models/best_model.keras"
    FRAMES: int = 32
    IMG_SIZE: int = 160
    NUM_CLASSES: int = 1
    CONFIDENCE_THRESHOLD: float = 0.5

    # ---------- UPLOAD ----------
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = ".mp4,.avi,.mov,.mkv,.webm"
    UPLOAD_DIR: str = "tmp/uploads"

    # ---------- OUTPUT ----------
    TOP_K_FRAMES: int = 3
    IMAGE_FORMAT: str = "PNG"

    # ---------- LOGGING ----------
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    # ============================================
    # PROPERTIES (computed)
    # ============================================
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse string '.mp4,.avi' → list ['.mp4', '.avi']"""
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB → bytes"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def model_path_absolute(self) -> Path:
        """Path absolut ke file model"""
        return BASE_DIR / self.MODEL_PATH

    @property
    def upload_dir_absolute(self) -> Path:
        """Path absolut folder upload"""
        return BASE_DIR / self.UPLOAD_DIR

    @property
    def log_dir_absolute(self) -> Path:
        """Path absolut folder log"""
        return BASE_DIR / self.LOG_DIR

    @property
    def is_dummy_mode(self) -> bool:
        """Shortcut untuk cek dummy mode"""
        return self.USE_DUMMY_MODEL

    # ============================================
    # VALIDATORS
    # ============================================
    @field_validator("CONFIDENCE_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD harus antara 0.0 dan 1.0")
        return v

    @field_validator("FRAMES", "IMG_SIZE", "TOP_K_FRAMES")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Nilai harus positif")
        return v

    @field_validator("IMAGE_FORMAT")
    @classmethod
    def validate_image_format(cls, v: str) -> str:
        v = v.upper()
        if v not in ("PNG", "JPEG", "JPG"):
            raise ValueError("IMAGE_FORMAT harus PNG, JPEG, atau JPG")
        return v


# ============================================
# SINGLETON INSTANCE
# ============================================
@lru_cache()
def get_settings() -> Settings:
    """
    Return instance Settings (cached).
    Pakai lru_cache supaya .env cuma dibaca sekali.
    """
    return Settings()


# Shortcut
settings = get_settings()