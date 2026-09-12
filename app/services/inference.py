"""
Service untuk menjalankan inference.

Alur:
1. Terima frames (32, 160, 160, 3)
2. Preprocess (tambah batch dim)
3. Jalankan model.predict() -> confidence, attention_weights
4. Post-processing:
   - Tentukan label (Shoplifting/Normal) dari confidence
   - Sort attention weights -> ambil top-K index
5. Return dataclass hasil inference
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
    prediction: str                    # "Shoplifting" / "Normal"
    confidence: float                  # 0.0 - 1.0
    attention_weights: np.ndarray      # shape (32,), sum = 1.0
    total_frames: int                  # 32
    top_indices: List[int] = field(default_factory=list)  # [12, 27, 11]
    top_weights: List[float] = field(default_factory=list)  # [0.15, 0.12, 0.10]

    def to_dict(self) -> dict:
        """Convert ke dict (tanpa numpy) untuk response JSON."""
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "total_frames": self.total_frames,
            "attention_weights": [round(float(w), 6) for w in self.attention_weights],
        }


# ============================================
# PREPROCESS
# ============================================
def _preprocess(frames: np.ndarray) -> np.ndarray:
    """
    Preprocess frames untuk masuk ke model.

    Args:
        frames: (32, 160, 160, 3) uint8 RGB

    Returns:
        (1, 32, 160, 160, 3) float32
    """
    if frames.ndim != 4:
        raise InferenceError(
            message=f"Shape frames salah: {frames.shape}. Expected (32, 160, 160, 3)",
        )

    expected_frames = settings.FRAMES
    expected_size = settings.IMG_SIZE

    if frames.shape[0] != expected_frames:
        raise InferenceError(
            message=f"Jumlah frame salah: {frames.shape[0]}, expected {expected_frames}",
        )

    if frames.shape[1] != expected_size or frames.shape[2] != expected_size:
        raise InferenceError(
            message=f"Ukuran frame salah: {frames.shape[1]}x{frames.shape[2]}, "
                    f"expected {expected_size}x{expected_size}",
        )

    # Tambah batch dimension
    batch = np.expand_dims(frames, axis=0).astype(np.float32)

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
        frames: numpy array (32, 160, 160, 3) uint8 RGB

    Returns:
        InferenceResult

    Raises:
        InferenceError: kalau prediksi gagal
    """
    model = get_model()

    # 1. Preprocess
    batch = _preprocess(frames)
    logger.debug(f"Input batch shape: {batch.shape}")

    # 2. Predict
    try:
        confidence, attention = model.predict(batch)
    except Exception as e:
        logger.exception("Model predict gagal")
        raise InferenceError(
            message=f"Gagal menjalankan model: {e}",
        ) from e

    # 3. Validasi output
    confidence = float(confidence)
    attention = np.asarray(attention, dtype=np.float32).flatten()

    if not 0.0 <= confidence <= 1.0:
        logger.warning(f"Confidence di luar range [0,1]: {confidence}. Clip.")
        confidence = float(np.clip(confidence, 0.0, 1.0))

    if len(attention) != settings.FRAMES:
        raise InferenceError(
            message=f"Panjang attention salah: {len(attention)}, expected {settings.FRAMES}",
        )

    # Normalisasi (jaga-jaga)
    att_sum = attention.sum()
    if att_sum > 0:
        attention = attention / att_sum

    # 4. Tentukan label
    prediction = _determine_label(confidence)

    # 5. Ambil top-K
    top_indices, top_weights = _get_top_k_indices(attention, settings.TOP_K_FRAMES)

    logger.info(
        f"Inference selesai: prediction={prediction}, "
        f"confidence={confidence:.4f}, "
        f"top-{settings.TOP_K_FRAMES}={list(zip(top_indices, [round(w, 4) for w in top_weights]))}"
    )

    return InferenceResult(
        prediction=prediction,
        confidence=confidence,
        attention_weights=attention,
        total_frames=settings.FRAMES,
        top_indices=top_indices,
        top_weights=top_weights,
    )