"""
Service untuk load video & sampling frame.

PENTING (untuk model E3):
- Model EfficientNetV2B0 pakai `include_preprocessing=True`, jadi
  frame TIDAK perlu di-preprocess lagi di sini.
- Frame di-return sebagai float32 (0-255), BUKAN uint8.
- Return juga sampled_indices untuk timestamp akurat.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings
from app.core.exceptions import VideoUnreadableError
from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# VIDEO METADATA
# ============================================
class VideoMetadata:
    """Info video yang dibaca."""

    def __init__(
        self,
        total_frames: int,
        fps: float,
        width: int,
        height: int,
        duration_sec: float,
    ):
        self.total_frames = total_frames
        self.fps = fps
        self.width = width
        self.height = height
        self.duration_sec = duration_sec

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 2),
            "resolution": f"{self.width}x{self.height}",
        }


def _read_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    """Baca metadata dari VideoCapture."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        logger.warning("FPS tidak terbaca, fallback ke 30")
        fps = 30.0

    duration_sec = total_frames / fps if fps > 0 else 0.0

    return VideoMetadata(
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )


# ============================================
# UNIFORM SAMPLING
# ============================================
def _uniform_sample_indices(total: int, target: int) -> np.ndarray:
    """
    Hasilkan index frame yang tersebar uniform.
    Sama persis dengan training (np.linspace).

    Contoh:
        total=320, target=32 -> [0, 10, 20, ..., 319]
        total=10, target=32  -> [0, 1, 2, ..., 9, 9, 9, ...]  (pad)
    """
    if total <= 0:
        return np.array([], dtype=int)

    if total >= target:
        indices = np.linspace(0, total - 1, target, dtype=int)
        return indices
    else:
        indices = list(range(total))
        while len(indices) < target:
            indices.append(total - 1)
        return np.array(indices, dtype=int)


# ============================================
# LOAD VIDEO
# ============================================
def load_video(
    video_path: str | Path,
    max_frames: Optional[int] = None,
    img_size: Optional[int] = None,
) -> tuple[np.ndarray, VideoMetadata, np.ndarray]:
    """
    Load video & sampling ke N frame.

    Args:
        video_path: path ke file video
        max_frames: jumlah frame target (default dari settings.FRAMES = 32)
        img_size: ukuran resize (default dari settings.IMG_SIZE = 160)

    Returns:
        (frames_array, metadata, sampled_indices)
        - frames_array: (max_frames, img_size, img_size, 3) float32 (0-255)
        - metadata: VideoMetadata
        - sampled_indices: index frame asli yang dipilih (untuk timestamp)

    Raises:
        VideoUnreadableError: kalau video tidak bisa dibaca
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoUnreadableError(
            message=f"File video tidak ditemukan: {video_path}",
        )

    max_frames = max_frames or settings.FRAMES
    img_size = img_size or settings.IMG_SIZE

    # ---------- BUKA VIDEO ----------
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise VideoUnreadableError(
            message="Video tidak bisa dibuka. Pastikan format valid & codec didukung.",
            details={"path": str(video_path)},
        )

    try:
        metadata = _read_metadata(cap)
        logger.info(
            f"Video dibaca: {video_path.name} | "
            f"{metadata.total_frames} frames, {metadata.fps:.1f} fps, "
            f"{metadata.width}x{metadata.height}"
        )

        if metadata.total_frames <= 0:
            raise VideoUnreadableError(
                message="Video tidak punya frame (corrupt atau kosong)",
            )

        # ---------- BACA SEMUA FRAME ----------
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)

        if len(raw_frames) == 0:
            raise VideoUnreadableError(
                message="Tidak ada frame yang berhasil dibaca dari video",
            )

        original_frame_count = len(raw_frames)
        logger.debug(f"Total frame dibaca: {original_frame_count}")

        # ---------- SAMPLING UNIFORM ----------
        sampled_indices = _uniform_sample_indices(
            total=original_frame_count,
            target=max_frames,
        )

        sampled_frames = []
        for idx in sampled_indices:
            frame = raw_frames[idx]
            # Resize
            frame = cv2.resize(
                frame,
                (img_size, img_size),
                interpolation=cv2.INTER_AREA,
            )
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(frame)

        # Pad kalau kurang (jaga-jaga)
        while len(sampled_frames) < max_frames:
            sampled_frames.append(sampled_frames[-1])

        sampled_frames = sampled_frames[:max_frames]

        # ⚠️ PENTING: float32 (0-255), BUKAN uint8 & BUKAN preprocess
        frames_array = np.array(sampled_frames, dtype=np.float32)

        logger.info(
            f"Sampling selesai: {len(sampled_frames)} frames, "
            f"shape={frames_array.shape}, dtype={frames_array.dtype}"
        )

        return frames_array, metadata, sampled_indices

    finally:
        cap.release()


# ============================================
# AMBIL FRAME SPESIFIK (untuk top-K)
# ============================================
def get_frames_at_indices(
    video_path: str | Path,
    indices: list[int],
    target_frames: int,
    img_size: Optional[int] = None,
) -> list[np.ndarray]:
    """
    Ambil frame tertentu dari video (sesuai index di 32 sampling).
    Return list of numpy array (H, W, 3) uint8 RGB — untuk display.

    Args:
        video_path: path video
        indices: list index di 32 frame (0-31)
        target_frames: jumlah sampling (32)
        img_size: ukuran resize

    Returns:
        list numpy array (H, W, 3) uint8 RGB
    """
    video_path = Path(video_path)
    img_size = img_size or settings.IMG_SIZE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoUnreadableError(message="Video tidak bisa dibuka ulang")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = _uniform_sample_indices(
            total=total,
            target=target_frames,
        )

        results = []
        for idx in indices:
            if idx < 0 or idx >= len(sample_indices):
                logger.warning(f"Index {idx} di luar range, skip")
                continue

            real_idx = int(sample_indices[idx])
            cap.set(cv2.CAP_PROP_POS_FRAMES, real_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Gagal baca frame ke-{real_idx}")
                continue

            frame = cv2.resize(
                frame,
                (img_size, img_size),
                interpolation=cv2.INTER_AREA,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Untuk display: uint8 (biar base64 PNG benar)
            results.append(frame.astype(np.uint8))

        return results

    finally:
        cap.release()


# ============================================
# HELPER: HITUNG TIMESTAMP
# ============================================
def indices_to_timestamps(
    sampled_indices: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Konversi index frame asli -> timestamp (detik).

    Contoh:
        sampled_indices = [0, 10, 20, 30, ..., 319]
        fps = 30
        -> timestamps = [0.0, 0.33, 0.67, 1.0, ..., 10.63]
    """
    if fps <= 0:
        return np.zeros(len(sampled_indices))
    return sampled_indices / fps


# ============================================
# QUICK CHECK: VIDEO BISA DIBACA?
# ============================================
def is_video_readable(video_path: str | Path) -> bool:
    """Cek cepat apakah video bisa dibuka tanpa load semua frame."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        readable = cap.isOpened()
        cap.release()
        return readable
    except Exception:
        return False