"""
Helper untuk konversi gambar / frame video ke base64.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# MIME HELPER
# ============================================
def _get_mime_type() -> str:
    """Return MIME type sesuai settings.IMAGE_FORMAT."""
    fmt = settings.IMAGE_FORMAT.upper()
    if fmt in ("JPG", "JPEG"):
        return "image/jpeg"
    return "image/png"


def _normalize_format() -> str:
    """PIL hanya kenal 'JPEG', bukan 'JPG'."""
    fmt = settings.IMAGE_FORMAT.upper()
    if fmt == "JPG":
        return "JPEG"
    return fmt


# ============================================
# NUMPY -> BASE64
# ============================================
def numpy_to_base64(
    image_array: np.ndarray,
    format: str | None = None,
    add_data_uri: bool = True,
) -> str:
    """
    Convert numpy array (H, W, 3) uint8 RGB ke base64.

    Args:
        image_array: numpy array shape (H, W, 3), dtype uint8, RGB
        format: "PNG" / "JPEG". Default dari settings.
        add_data_uri: kalau True, tambah prefix "data:image/png;base64,..."

    Returns:
        String base64 (dengan/ tanpa data URI prefix)
    """
    if image_array is None or image_array.size == 0:
        raise ValueError("Array gambar kosong")

    # Pastikan uint8
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Kalau grayscale (H, W), convert ke RGB
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    # Kalau ada alpha channel (H, W, 4), buang
    if image_array.ndim == 3 and image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    fmt = (format or _normalize_format()).upper()
    if fmt == "JPG":
        fmt = "JPEG"

    # Convert ke PIL
    img = Image.fromarray(image_array, mode="RGB")

    # Simpan ke BytesIO
    buffer = BytesIO()
    img.save(buffer, format=fmt, optimize=True)
    buffer.seek(0)

    # Encode base64
    encoded = base64.b64encode(buffer.read()).decode("utf-8")

    if add_data_uri:
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        return f"data:{mime};base64,{encoded}"

    return encoded


# ============================================
# ANNOTATE FRAME
# ============================================
def annotate_frame(
    frame: np.ndarray,
    rank: int,
    frame_index: int,
    weight: float,
) -> np.ndarray:
    """
    Tambah label di atas frame (rank, index, weight).
    Berguna buat visualisasi top frame di frontend.

    Args:
        frame: numpy array (H, W, 3) uint8 RGB
        rank: 1/2/3 (urutan tertinggi)
        frame_index: index frame di 32 frame (0-31)
        weight: bobot attention (0-1)

    Returns:
        numpy array dengan label tertempel
    """
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Convert ke PIL
    img = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Siapkan teks
    text_line1 = f"Rank #{rank}"
    text_line2 = f"Frame {frame_index + 1} | {weight * 100:.2f}%"

    # Pakai font default (selalu tersedia)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
        font_small = ImageFont.truetype("arial.ttf", size=12)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # Hitung ukuran text box
    padding = 6
    try:
        bbox1 = draw.textbbox((0, 0), text_line1, font=font)
        bbox2 = draw.textbbox((0, 0), text_line2, font=font_small)
    except AttributeError:
        # Pillow versi lama
        w1, h1 = draw.textsize(text_line1, font=font)
        w2, h2 = draw.textsize(text_line2, font=font_small)
        bbox1 = (0, 0, w1, h1)
        bbox2 = (0, 0, w2, h2)

    text_width = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
    text_height = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + 4

    # Background semi-transparan di atas (hitam)
    box_h = text_height + padding * 2
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(0, 0), (text_width + padding * 2, box_h)],
        fill=(0, 0, 0, 180),
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Tulis teks (putih)
    draw.text((padding, padding), text_line1, fill=(255, 255, 255), font=font)
    draw.text(
        (padding, padding + (bbox1[3] - bbox1[1]) + 4),
        text_line2,
        fill=(255, 255, 255),
        font=font_small,
    )

    return np.array(img)


# ============================================
# FRAME -> BASE64 DENGAN ANNOTASI
# ============================================
def frame_to_base64_annotated(
    frame: np.ndarray,
    rank: int,
    frame_index: int,
    weight: float,
) -> str:
    """
    Shortcut: annotate frame + convert ke base64.
    """
    annotated = annotate_frame(frame, rank, frame_index, weight)
    return numpy_to_base64(annotated)


# ============================================
# RESIZE FRAME (opsional, kalau mau hemat bandwidth)
# ============================================
def resize_frame(frame: np.ndarray, size: tuple[int, int] = (320, 320)) -> np.ndarray:
    """
    Resize frame ke ukuran tertentu.
    Berguna kalau frame asli 160x160 terlalu kecil, atau 1920x1080 terlalu besar.
    """
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    img = Image.fromarray(frame, mode="RGB")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)


# ============================================
# VALIDASI BASE64
# ============================================
def is_valid_base64(data_uri: str) -> bool:
    """
    Cek apakah string adalah data URI base64 valid.
    """
    if not isinstance(data_uri, str):
        return False
    if not data_uri.startswith("data:image/"):
        return False
    if ";base64," not in data_uri:
        return False

    try:
        # Coba decode
        _, encoded = data_uri.split(",", 1)
        base64.b64decode(encoded, validate=True)
        return True
    except Exception:
        return False