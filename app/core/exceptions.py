"""
Custom exceptions + handler untuk aplikasi.
Semua error dikembalikan dalam format JSON yang konsisten.
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# CUSTOM EXCEPTION CLASSES
# ============================================
class AppException(Exception):
    """
    Base exception untuk aplikasi.
    Semua custom exception turunan dari class ini.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "APP_ERROR",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[dict] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class InvalidFileFormatError(AppException):
    """File yang di-upload bukan format video yang didukung."""

    def __init__(self, message: str = "Format file tidak didukung", details: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code="INVALID_FILE_FORMAT",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class FileTooLargeError(AppException):
    """Ukuran file melebihi batas maksimum."""

    def __init__(self, message: str = "File terlalu besar", details: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code="FILE_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details=details,
        )


class VideoUnreadableError(AppException):
    """Video tidak bisa dibaca / corrupt."""

    def __init__(self, message: str = "Video tidak bisa dibaca", details: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code="VIDEO_UNREADABLE",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class ModelNotLoadedError(AppException):
    """Model belum di-load / gagal load."""

    def __init__(self, message: str = "Model belum siap", details: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code="MODEL_NOT_LOADED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class InferenceError(AppException):
    """Error saat menjalankan prediksi."""

    def __init__(self, message: str = "Gagal melakukan prediksi", details: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code="INFERENCE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


# ============================================
# RESPONSE BUILDER
# ============================================
def _error_response(
    message: str,
    error_code: str,
    status_code: int,
    details: Optional[dict] = None,
) -> JSONResponse:
    """Bikin JSON response error yang konsisten."""
    body: dict[str, Any] = {
        "detail": message,
        "error_code": error_code,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if details:
        body["details"] = details

    return JSONResponse(status_code=status_code, content=body)


# ============================================
# EXCEPTION HANDLERS
# ============================================
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handler untuk custom AppException."""
    logger.warning(
        f"[{exc.error_code}] {exc.message} | path={request.url.path} | details={exc.details}"
    )
    return _error_response(
        message=exc.message,
        error_code=exc.error_code,
        status_code=exc.status_code,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handler untuk HTTPException bawaan FastAPI."""
    logger.warning(f"[HTTP_{exc.status_code}] {exc.detail} | path={request.url.path}")
    return _error_response(
        message=str(exc.detail),
        error_code=f"HTTP_{exc.status_code}",
        status_code=exc.status_code,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handler untuk error validasi request (422)."""
    logger.warning(f"[VALIDATION_ERROR] path={request.url.path} | errors={exc.errors()}")
    return _error_response(
        message="Request tidak valid",
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"errors": exc.errors()},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler untuk error tak terduga (500)."""
    logger.exception(f"[UNHANDLED_ERROR] path={request.url.path} | {type(exc).__name__}: {exc}")
    return _error_response(
        message="Terjadi kesalahan internal pada server",
        error_code="INTERNAL_SERVER_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# ============================================
# REGISTER HANDLERS
# ============================================
def register_exception_handlers(app: FastAPI) -> None:
    """
    Daftarkan semua exception handler ke app FastAPI.
    Panggil di main.py setelah app dibuat.
    """
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("✅ Exception handlers terdaftar")