"""
Service untuk menjalankan inference.

PENTING (untuk model E3):
- Frame dari video_loader sudah float32 (0-255), TIDAK perlu preprocess lagi
- Model EfficientNetV2B0 pakai include_preprocessing=True
- Jadi _preprocess() hanya: tambah batch dim + pastikan float32
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from app.config import settings
from app.core.exceptions import InferenceError
from app.core.logger import get_logger
from app.services.model_loader import get_model


logger = get_logger(__name__)


# ============================================
# RESULT DATACLASS
# ============================================
@dataclass
class InferenceResult:
    """Hasil inference siap dipakai route."""
    prediction: str                       # "Shoplifting" / "Normal"
    confidence: float                     # 0.0 - 1.0
    shoplifting_probability: float        # probabilitas Shoplifting
    normal_probability: float             # probabilitas Normal
    attention_weights: np.ndarray         # shape (32,), sum = 1.0
    total_frames: int                     # 32
    top_indices: List[int] = field(default_factory=list)
    top_weights: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert ke dict (tanpa numpy) untuk response JSON."""
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "total_frames": self.total_frames,
            "attention_weights": [
                round(float(w), 6) for w in self.attention_weights
            ],
        }


# ============================================
# PREPROCESS (minimal)
# ============================================
def _preprocess(frames: np.ndarray) -> np.ndarray:
    """
    Preprocess minimal: tambah batch dim + pastikan float32.

    TIDAK ada preprocess_input di sini, karena model sudah
    include_preprocessing=True.

    Args:
        frames: (32, 160, 160, 3) float32 (0-255)

    Returns:
        (1, 32, 160, 160, 3) float32
    """
    if frames.ndim != 4:
        raise InferenceError(
            message=f"Shape frames salah: {frames.shape}. "
                    f"Expected (32, 160, 160, 3)",
        )

    expected_frames = settings.FRAMES
    expected_size = settings.IMG_SIZE

    if frames.shape[0] != expected_frames:
        raise InferenceError(
            message=f"Jumlah frame salah: {frames.shape[0]}, "
                    f"expected {expected_frames}",
        )

    if frames.shape[1] != expected_size or frames.shape[2] != expected_size:
        raise InferenceError(
            message=f"Ukuran frame salah: {frames.shape[1]}x{frames.shape[2]}, "
                    f"expected {expected_size}x{expected_size}",
        )

    # Pastikan float32
    if frames.dtype != np.float32:
        logger.debug(f"Convert frames dari {frames.dtype} ke float32")
        frames = frames.astype(np.float32)

    # Tambah batch dimension
    batch = np.expand_dims(frames, axis=0)

    return batch


# ============================================
# POST-PROCESSING
# ============================================
def _determine_label(confidence: float) -> str:
    """
    Tentukan label dari confidence score.
    confidence > threshold -> "Shoplifting"
    confidence <= threshold -> "Normal"
    """
    threshold = settings.CONFIDENCE_THRESHOLD
    return "Shoplifting" if confidence > threshold else "Normal"


def _get_top_k_indices(
    attention: np.ndarray,
    k: int,
) -> tuple[List[int], List[float]]:
    """
    Ambil k index dengan attention weight tertinggi.

    Returns:
        (indices, weights) - sorted descending by weight
    """
    if k <= 0:
        return [], []

    if k > len(attention):
        k = len(attention)

    # argsort ascending, ambil k terakhir, lalu reverse
    top_indices = np.argsort(attention)[-k:][::-1]
    top_weights = attention[top_indices]

    return top_indices.tolist(), top_weights.tolist()


# ============================================
# MAIN INFERENCE
# ============================================
def run_inference(frames: np.ndarray) -> InferenceResult:
    """
    Jalankan inference lengkap.

    Args:
        frames: numpy array (32, 160, 160, 3) float32 (0-255)

    Returns:
        InferenceResult

    Raises:
        InferenceError: kalau prediksi gagal
    """
    model = get_model()

    # ---------- 1. PREPROCESS ----------
    batch = _preprocess(frames)
    logger.debug(f"Input batch shape: {batch.shape}, dtype: {batch.dtype}")

    # ---------- 2. PREDICT ----------
    try:
        confidence, attention = model.predict(batch)
    except Exception as e:
        logger.exception("Model predict gagal")
        raise InferenceError(
            message=f"Gagal menjalankan model: {e}",
        ) from e

    # ---------- 3. VALIDASI OUTPUT ----------
    confidence = float(confidence)
    attention = np.asarray(attention, dtype=np.float32).flatten()

    # Clip confidence ke [0, 1]
    if not 0.0 <= confidence <= 1.0:
        logger.warning(
            f"Confidence di luar range [0,1]: {confidence}. Clip."
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

    # Cek panjang attention
    if len(attention) != settings.FRAMES:
        raise InferenceError(
            message=f"Panjang attention salah: {len(attention)}, "
                    f"expected {settings.FRAMES}",
        )

    # Normalisasi attention (jaga-jaga, harusnya sudah sum=1 dari model)
    att_sum = attention.sum()
    if att_sum > 0:
        attention = attention / att_sum

    # ---------- 4. HITUNG PROBABILITAS ----------
    shoplifting_prob = confidence
    normal_prob = 1.0 - confidence

    # ---------- 5. TENTUKAN LABEL ----------
    prediction = _determine_label(confidence)

    # ---------- 6. AMBIL TOP-K ----------
    top_indices, top_weights = _get_top_k_indices(
        attention,
        settings.TOP_K_FRAMES,
    )

    logger.info(
        f"Inference selesai: prediction={prediction}, "
        f"confidence={confidence:.4f}, "
        f"top-{settings.TOP_K_FRAMES}="
        f"{list(zip(top_indices, [round(w, 4) for w in top_weights]))}"
    )

    return InferenceResult(
        prediction=prediction,
        confidence=confidence,
        shoplifting_probability=shoplifting_prob,
        normal_probability=normal_prob,
        attention_weights=attention,
        total_frames=settings.FRAMES,
        top_indices=top_indices,
        top_weights=top_weights,
    )