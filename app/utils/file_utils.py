"""
Helper untuk validasi & manajemen file upload.
"""

import uuid
import shutil
from pathlib import Path
from typing import List

from fastapi import UploadFile

from app.config import settings
from app.core.exceptions import InvalidFileFormatError, FileTooLargeError
from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# MIME TYPE YANG DIIZINKAN
# ============================================
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/x-msvideo",       # .avi
    "video/quicktime",       # .mov
    "video/x-matroska",      # .mkv
    "video/webm",            # .webm
    "application/octet-stream",  # fallback (beberapa client tidak set MIME)
}


# ============================================
# VALIDASI EKSTENSI
# ============================================
def validate_extension(filename: str) -> str:
    """
    Cek ekstensi file.
    Return ekstensi lowercase (.mp4) kalau valid.
    Raise InvalidFileFormatError kalau tidak valid.
    """
    if not filename:
        raise InvalidFileFormatError(
            message="Nama file kosong",
            details={"allowed": settings.allowed_extensions_list},
        )

    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_extensions_list:
        raise InvalidFileFormatError(
            message=f"Format '{ext}' tidak didukung. "
                    f"Hanya menerima: {', '.join(settings.allowed_extensions_list)}",
            details={
                "received": ext,
                "allowed": settings.allowed_extensions_list,
            },
        )

    return ext


# ============================================
# VALIDASI MIME TYPE
# ============================================
def validate_mime_type(content_type: str | None) -> None:
    """
    Cek MIME type file.
    Raise InvalidFileFormatError kalau tidak valid.
    """
    if content_type is None:
        # Beberapa client tidak set content-type
        logger.warning("Content-Type tidak diset, skip validasi MIME")
        return

    if content_type not in ALLOWED_MIME_TYPES:
        raise InvalidFileFormatError(
            message=f"MIME type '{content_type}' tidak didukung",
            details={
                "received": content_type,
                "allowed": sorted(ALLOWED_MIME_TYPES),
            },
        )


# ============================================
# VALIDASI UKURAN
# ============================================
def validate_size(size_bytes: int) -> None:
    """
    Cek ukuran file.
    Raise FileTooLargeError kalau melebihi batas.
    """
    max_bytes = settings.max_upload_size_bytes

    if size_bytes <= 0:
        raise InvalidFileFormatError(
            message="File kosong (0 byte)",
        )

    if size_bytes > max_bytes:
        raise FileTooLargeError(
            message=f"Ukuran file melebihi batas maksimum "
                    f"{settings.MAX_UPLOAD_SIZE_MB} MB. "
                    f"File Anda: {size_bytes / (1024 * 1024):.2f} MB",
            details={
                "max_mb": settings.MAX_UPLOAD_SIZE_MB,
                "received_mb": round(size_bytes / (1024 * 1024), 2),
            },
        )


# ============================================
# VALIDASI LENGKAP
# ============================================
def validate_video_upload(file: UploadFile) -> str:
    """
    Validasi lengkap: nama, ekstensi, MIME type, ukuran.
    Return ekstensi kalau valid.
    Raise exception kalau tidak valid.

    Catatan: file.size bisa None kalau belum dibaca.
    """
    if file is None:
        raise InvalidFileFormatError(message="File tidak ditemukan")

    if not file.filename:
        raise InvalidFileFormatError(message="Nama file tidak ada")

    # Ekstensi
    ext = validate_extension(file.filename)

    # MIME type
    validate_mime_type(file.content_type)

    # Ukuran (kalau tersedia)
    if file.size is not None:
        validate_size(file.size)

    logger.debug(
        f"File valid: name={file.filename}, "
        f"ext={ext}, mime={file.content_type}, size={file.size}"
    )

    return ext


# ============================================
# GENERATE NAMA FILE UNIK
# ============================================
def generate_unique_filename(original_filename: str, prefix: str = "upload") -> str:
    """
    Generate nama file unik dengan UUID.
    Contoh: upload_a1b2c3d4.mp4
    """
    ext = Path(original_filename).suffix.lower()
    unique_id = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_id}{ext}"


# ============================================
# SIMPAN FILE SEMENTARA
# ============================================
def save_upload_to_temp(file: UploadFile) -> Path:
    """
    Simpan UploadFile ke folder sementara.
    Return Path ke file yang disimpan.
    Folder: settings.upload_dir_absolute
    """
    upload_dir: Path = settings.upload_dir_absolute
    upload_dir.mkdir(parents=True, exist_ok=True)

    unique_name = generate_unique_filename(file.filename or "video.mp4")
    dest_path = upload_dir / unique_name

    try:
        # Reset posisi file ke awal (antisipasi kalau sudah dibaca)
        file.file.seek(0)

        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File disimpan: {dest_path} ({dest_path.stat().st_size} bytes)")
        return dest_path

    except Exception as e:
        # Hapus file parsial kalau gagal
        if dest_path.exists():
            dest_path.unlink()
        logger.error(f"Gagal simpan file: {e}")
        raise


# ============================================
# HAPUS FILE
# ============================================
def cleanup_file(path: Path | str | None) -> None:
    """
    Hapus file dengan aman (tidak raise kalau gagal).
    """
    if path is None:
        return

    path = Path(path) if isinstance(path, str) else path

    try:
        if path.exists():
            path.unlink()
            logger.debug(f"File dihapus: {path}")
    except Exception as e:
        logger.warning(f"Gagal hapus file {path}: {e}")


# ============================================
# BERSIHKAN FOLDER UPLOAD
# ============================================
def cleanup_upload_dir() -> None:
    """
    Hapus semua file di folder upload.
    Berguna saat startup / shutdown.
    """
    upload_dir: Path = settings.upload_dir_absolute

    if not upload_dir.exists():
        return

    count = 0
    for f in upload_dir.iterdir():
        if f.is_file():
            try:
                f.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Gagal hapus {f}: {e}")

    if count > 0:
        logger.info(f"🧹 {count} file dibersihkan dari {upload_dir}")