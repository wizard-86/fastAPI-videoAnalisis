"""
Service untuk ekstrak top-K frame dari video.

PENTING:
- timestamp dihitung dari `sampled_indices` (index frame asli / fps)
- ini akurat, bukan perkiraan
"""

from pathlib import Path
from typing import List

import numpy as np

from app.config import settings
from app.core.exceptions import VideoUnreadableError
from app.core.logger import get_logger
from app.services.video_loader import get_frames_at_indices
from app.utils.image_utils import frame_to_base64_annotated


logger = get_logger(__name__)


# ============================================
# EXTRACT TOP FRAMES
# ============================================
def extract_top_frames(
    video_path: str | Path,
    top_indices: List[int],
    top_weights: List[float],
    sampled_indices: np.ndarray,
    fps: float,
    total_frames: int = 32,
    img_size: int | None = None,
) -> List[dict]:
    """
    Ambil top-K frame dari video, annotate, dan convert ke base64.

    Args:
        video_path: path ke file video
        top_indices: list index frame (di 32 sampling), sorted desc by weight
        top_weights: list bobot attention untuk masing-masing index
        sampled_indices: index frame asli yang dipilih saat sampling
        fps: FPS video (untuk hitung timestamp)
        total_frames: jumlah total frame sampling (32)
        img_size: ukuran resize (default dari settings.IMG_SIZE)

    Returns:
        list of dict:
        [
            {
                "rank": 1,
                "frame_index": 12,
                "weight": 0.1521,
                "timestamp_sec": 3.75,
                "image": "data:image/png;base64,..."
            },
            ...
        ]
    """
    if not top_indices:
        logger.warning("top_indices kosong, return list kosong")
        return []

    if len(top_indices) != len(top_weights):
        raise ValueError(
            f"Panjang top_indices ({len(top_indices)}) != "
            f"top_weights ({len(top_weights)})"
        )

    img_size = img_size or settings.IMG_SIZE

    logger.debug(f"Mengambil {len(top_indices)} frame: {top_indices}")

    # ---------- AMBIL FRAME DARI VIDEO ----------
    try:
        frames = get_frames_at_indices(
            video_path=video_path,
            indices=top_indices,
            target_frames=total_frames,
            img_size=img_size,
        )
    except VideoUnreadableError:
        raise
    except Exception as e:
        raise VideoUnreadableError(
            message=f"Gagal ambil frame dari video: {e}",
        ) from e

    if len(frames) != len(top_indices):
        logger.warning(
            f"Hanya dapat {len(frames)} frame dari {len(top_indices)} yang diminta"
        )

    # ---------- BUILD RESULT ----------
    results: List[dict] = []

    for rank, (frame, idx, weight) in enumerate(
        zip(frames, top_indices[: len(frames)], top_weights[: len(frames)]),
        start=1,
    ):
        try:
            # Hitung timestamp akurat dari sampled_indices
            timestamp_sec = _get_timestamp(
                idx=idx,
                sampled_indices=sampled_indices,
                fps=fps,
            )

            # Annotate + convert ke base64
            image_b64 = frame_to_base64_annotated(
                frame=frame,
                rank=rank,
                frame_index=idx,
                weight=weight,
            )

            results.append({
                "rank": rank,
                "frame_index": int(idx),
                "weight": round(float(weight), 6),
                "timestamp_sec": timestamp_sec,
                "image": image_b64,
            })

            logger.debug(
                f"Frame rank #{rank}: index={idx}, weight={weight:.4f}, "
                f"timestamp={timestamp_sec:.2f}s, "
                f"base64_len={len(image_b64)}"
            )

        except Exception as e:
            # Kalau 1 frame gagal, log & skip, jangan crash semua
            logger.error(f"Gagal proses frame rank #{rank} (index={idx}): {e}")
            continue

    logger.info(f"✅ {len(results)} top frame berhasil diekstrak")
    return results


# ============================================
# HELPER: TIMESTAMP AKURAT
# ============================================
def _get_timestamp(
    idx: int,
    sampled_indices: np.ndarray,
    fps: float,
) -> float:
    """
    Hitung timestamp (detik) dari index sampling.

    Contoh:
        idx = 12
        sampled_indices = [0, 10, 20, ..., 319]
        fps = 30

        -> sampled_indices[12] = 120  (frame asli ke-120)
        -> 120 / 30 = 4.0 detik
    """
    if fps <= 0:
        return 0.0

    if idx < 0 or idx >= len(sampled_indices):
        logger.warning(f"Index {idx} di luar range sampled_indices")
        return 0.0

    real_frame_idx = int(sampled_indices[idx])
    return round(real_frame_idx / fps, 2)