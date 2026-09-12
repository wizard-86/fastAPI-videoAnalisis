"""
Service untuk ekstrak top-K frame dari video.

Alur:
1. Terima video_path + top_indices (dari inference)
2. Ambil frame di index tersebut (re-open video)
3. Annotate setiap frame dengan rank, index, weight
4. Convert ke base64 PNG
5. Return list of dict (siap masuk response JSON)
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
    total_frames: int = 32,
    img_size: int | None = None,
) -> List[dict]:
    """
    Ambil top-K frame dari video, annotate, dan convert ke base64.

    Args:
        video_path: path ke file video
        top_indices: list index frame (di 32 sampling), sorted descending by weight
        top_weights: list bobot attention untuk masing-masing index
        total_frames: jumlah total frame sampling (32)
        img_size: ukuran resize (default dari settings.IMG_SIZE)

    Returns:
        list of dict:
        [
            {
                "rank": 1,
                "frame_index": 12,
                "weight": 0.1521,
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
                "image": image_b64,
            })

            logger.debug(
                f"Frame rank #{rank}: index={idx}, weight={weight:.4f}, "
                f"base64_len={len(image_b64)}"
            )

        except Exception as e:
            # Kalau 1 frame gagal, log & skip, jangan crash semua
            logger.error(f"Gagal proses frame rank #{rank} (index={idx}): {e}")
            continue

    logger.info(f"✅ {len(results)} top frame berhasil diekstrak")
    return results


# ============================================
# HELPER: KONVERSI INDEX -> TIMESTAMP
# ============================================
def indices_to_timestamps(
    indices: List[int],
    total_frames: int,
    video_duration_sec: float,
) -> List[float]:
    """
    Konversi index sampling (0-31) ke timestamp (detik) di video asli.

    Args:
        indices: list index di 32 sampling
        total_frames: jumlah sampling (32)
        video_duration_sec: durasi video (detik)

    Returns:
        list timestamp dalam detik

    Contoh:
        indices=[12, 27], total_frames=32, duration=10.0
        -> [3.75, 8.44]
    """
    if total_frames <= 1:
        return [0.0] * len(indices)

    timestamps = []
    for idx in indices:
        # Posisi relatif (0.0 - 1.0)
        position = idx / (total_frames - 1)
        # Konversi ke detik
        timestamp = position * video_duration_sec
        timestamps.append(round(timestamp, 2))

    return timestamps


# ============================================
# EXTRACT + ENRICH DENGAN TIMESTAMP
# ============================================
def extract_top_frames_with_timestamp(
    video_path: str | Path,
    top_indices: List[int],
    top_weights: List[float],
    total_frames: int = 32,
    video_duration_sec: float = 0.0,
    img_size: int | None = None,
) -> List[dict]:
    """
    Sama seperti extract_top_frames, tapi menambahkan field 'timestamp_sec'.
    Berguna biar user tahu frame itu di menit ke-berapa.
    """
    results = extract_top_frames(
        video_path=video_path,
        top_indices=top_indices,
        top_weights=top_weights,
        total_frames=total_frames,
        img_size=img_size,
    )

    if video_duration_sec > 0 and results:
        timestamps = indices_to_timestamps(
            indices=[r["frame_index"] for r in results],
            total_frames=total_frames,
            video_duration_sec=video_duration_sec,
        )
        for i, r in enumerate(results):
            r["timestamp_sec"] = timestamps[i]

    return results