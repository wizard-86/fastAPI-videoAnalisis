"""
Service untuk load model.

Dua mode:
1. DUMMY MODE (USE_DUMMY_MODEL=True)
   - Return DummyModel yang generate attention weights random

2. PRODUCTION MODE (USE_DUMMY_MODEL=False)
   - Build arsitektur E3 (build_e3_model)
   - Load weights dari .keras (load_weights)
   - Bikin attention model multi-output

Karena model .keras disimpan sebagai weights (custom layer + tuple output),
kita REBUILD arsitektur lalu load_weights(), BUKAN load_model().
"""

from typing import Any, Optional, Protocol
from pathlib import Path

import numpy as np

from app.config import settings
from app.core.exceptions import ModelNotLoadedError, InferenceError
from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# PROTOCOL
# ============================================
class BaseModel(Protocol):
    """Interface yang harus dipenuhi model (dummy atau asli)."""

    def predict(self, frames: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Args:
            frames: numpy array shape (1, 32, 160, 160, 3)

        Returns:
            (confidence, attention_weights)
        """
        ...


# ============================================
# DUMMY MODEL
# ============================================
class DummyModel:
    """
    Model dummy untuk development.
    - Confidence: random 0-1
    - Attention weights: random Dirichlet (sum = 1.0)
    """

    def __init__(self, num_frames: int = 32, seed: Optional[int] = None):
        self.num_frames = num_frames
        self.rng = np.random.default_rng(seed)

    def predict(self, frames: np.ndarray) -> tuple[float, np.ndarray]:
        confidence = float(self.rng.uniform(0.0, 1.0))

        alpha = np.ones(self.num_frames) * 0.5
        attention = self.rng.dirichlet(alpha).astype(np.float32)

        assert attention.shape == (self.num_frames,)
        assert abs(attention.sum() - 1.0) < 1e-5

        return confidence, attention


# ============================================
# BUILD ARSITEKTUR E3
# ============================================
def build_e3_model():
    """
    Build arsitektur E3 (sama persis dengan training).

    Arsitektur:
        Input (32, 160, 160, 3)
        -> TimeDistributed(EfficientNetV2B0)   [frozen, include_preprocessing]
        -> TimeDistributed(SEBlock)
        -> TimeDistributed(GlobalAveragePooling2D)
        -> Bidirectional(LSTM(128, return_sequences=True))
        -> TemporalAttention -> (context, attention_weights)
        -> Dropout(0.5)
        -> Dense(128, relu)
        -> Dense(1, sigmoid)   [output]
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetV2B0

    from app.models.custom_layers import (
        get_seblock_class,
        get_temporal_attention_class,
    )

    SEBlock = get_seblock_class()
    TemporalAttention = get_temporal_attention_class()

    NUM_FRAMES = settings.FRAMES
    IMG_SIZE = settings.IMG_SIZE
    CHANNELS = 3

    logger.info(f"Building E3 model: {NUM_FRAMES} frames, {IMG_SIZE}x{IMG_SIZE}")

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------
    video_input = layers.Input(
        shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS),
        name="video_input",
    )

    # --------------------------------------------------------
    # BACKBONE
    # --------------------------------------------------------
    backbone = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        include_preprocessing=True,
    )
    backbone.trainable = False

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------
    x = layers.TimeDistributed(
        backbone,
        name="frame_feature_extractor",
    )(video_input)

    # --------------------------------------------------------
    # SE ATTENTION
    # --------------------------------------------------------
    x = layers.TimeDistributed(
        SEBlock(reduction=16),
        name="se_attention",
    )(x)

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------
    x = layers.TimeDistributed(
        layers.GlobalAveragePooling2D(),
        name="global_average_pooling",
    )(x)

    # --------------------------------------------------------
    # BiLSTM
    # --------------------------------------------------------
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.0,
            recurrent_dropout=0.0,
        ),
        name="bilstm",
    )(x)

    # --------------------------------------------------------
    # TEMPORAL ATTENTION
    # --------------------------------------------------------
    temporal_attention = TemporalAttention(name="temporal_attention")
    context, attention_weights = temporal_attention(x)

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------
    x = layers.Dropout(0.5, name="dropout")(context)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    # --------------------------------------------------------
    # MODEL (single output)
    # --------------------------------------------------------
    model = models.Model(
        inputs=video_input,
        outputs=output,
        name="E3_PROPOSED_DUAL_ATTENTION",
    )

    return model


# ============================================
# BUILD ATTENTION MODEL (multi-output)
# ============================================
def build_attention_model(model):
    """
    Bikin model multi-output: [prediction, attention_weights].
    Dipakai untuk ambil attention weights saat inference.
    """
    from tensorflow.keras import models

    attention_layer = model.get_layer("temporal_attention")

    attention_model = models.Model(
        inputs=model.input,
        outputs=[
            model.output,
            attention_layer.output[1],
        ],
    )

    return attention_model


# ============================================
# REAL MODEL
# ============================================
class RealModel:
    """
    Wrapper untuk model E3 asli.
    - Build arsitektur dari kode
    - Load weights dari .keras
    - Bikin attention_model (multi-output)
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.attention_model = None
        self._load()

    def _load(self) -> None:
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ModelNotLoadedError(
                message="TensorFlow belum terinstall. "
                        "Aktifkan tensorflow di requirements.txt.",
            ) from e

        if not self.model_path.exists():
            raise ModelNotLoadedError(
                message=f"File model tidak ditemukan: {self.model_path}",
                details={"path": str(self.model_path)},
            )

        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Loading model dari: {self.model_path} ({size_mb:.1f} MB)")

        # --------------------------------------------------------
        # 1. BUILD ARSITEKTUR
        # --------------------------------------------------------
        logger.info("Membangun arsitektur E3...")
        try:
            self.model = build_e3_model()
            logger.info("✅ Arsitektur E3 berhasil dibangun")
        except Exception as e:
            raise ModelNotLoadedError(
                message=f"Gagal build arsitektur E3: {e}",
            ) from e

        # --------------------------------------------------------
        # 2. LOAD WEIGHTS
        # --------------------------------------------------------
        logger.info("Loading weights dari .keras...")
        try:
            self.model.load_weights(str(self.model_path))
            logger.info("✅ Weights berhasil di-load")
        except Exception as e:
            raise ModelNotLoadedError(
                message=f"Gagal load weights: {e}",
                details={"path": str(self.model_path)},
            ) from e

        # --------------------------------------------------------
        # 3. BUILD ATTENTION MODEL
        # --------------------------------------------------------
        logger.info("Membangun attention model (multi-output)...")
        try:
            self.attention_model = build_attention_model(self.model)
            logger.info("✅ Attention model berhasil dibuat")
        except Exception as e:
            raise ModelNotLoadedError(
                message=f"Gagal bangun attention model: {e}",
            ) from e

    def predict(self, frames: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Args:
            frames: (1, 32, 160, 160, 3) float32 (0-255), BELUM preprocess
                    (karena EfficientNet pakai include_preprocessing=True)

        Returns:
            (confidence, attention_weights)
        """
        if self.attention_model is None:
            raise ModelNotLoadedError(message="Attention model belum siap")

        try:
            # Predict — model multi-output
            prediction, attention = self.attention_model.predict(
                frames,
                verbose=0,
            )

            # Confidence: probabilitas Shoplifting
            confidence = float(prediction[0][0])

            # Attention: shape (1, 32) -> (32,)
            attention = np.asarray(attention[0]).flatten().astype(np.float32)

            # Normalisasi
            att_sum = attention.sum()
            if att_sum > 0:
                attention = attention / att_sum

            # Validasi panjang
            if len(attention) != settings.FRAMES:
                logger.warning(
                    f"Panjang attention={len(attention)}, "
                    f"expected={settings.FRAMES}. Resize..."
                )
                attention = _resize_attention(attention, settings.FRAMES)

            return confidence, attention

        except Exception as e:
            raise InferenceError(
                message=f"Gagal menjalankan prediksi: {e}",
            ) from e


# ============================================
# HELPER: RESIZE ATTENTION
# ============================================
def _resize_attention(attention: np.ndarray, target_len: int) -> np.ndarray:
    """Resize attention kalau panjangnya tidak sesuai."""
    if len(attention) == target_len:
        return attention

    x_old = np.linspace(0, 1, len(attention))
    x_new = np.linspace(0, 1, target_len)
    resized = np.interp(x_new, x_old, attention)

    total = resized.sum()
    if total > 0:
        resized = resized / total

    return resized.astype(np.float32)


# ============================================
# SINGLETON
# ============================================
_model_instance: Optional[Any] = None


def load_model() -> Any:
    """
    Load model (dummy atau asli) dan cache sebagai singleton.
    Panggil di startup FastAPI.
    """
    global _model_instance

    if _model_instance is not None:
        logger.debug("Model sudah di-load, pakai instance yang ada")
        return _model_instance

    if settings.is_dummy_mode:
        logger.warning(
            "⚠️  DUMMY MODE aktif. Prediksi & attention weights di-generate random. "
            "Set USE_DUMMY_MODEL=False di .env kalau model asli siap."
        )
        _model_instance = DummyModel(num_frames=settings.FRAMES)
    else:
        logger.info("Loading model asli (production mode)...")
        model_path = settings.model_path_absolute
        _model_instance = RealModel(model_path=model_path)

    return _model_instance


def get_model() -> Any:
    """Return model yang sudah di-load. Raise kalau belum."""
    if _model_instance is None:
        raise ModelNotLoadedError(
            message="Model belum di-load. Pastikan load_model() dipanggil di startup.",
        )
    return _model_instance


def is_model_loaded() -> bool:
    """Cek apakah model sudah di-load."""
    return _model_instance is not None


def get_model_info() -> dict:
    """Return info model untuk endpoint /health."""
    if _model_instance is None:
        return {"loaded": False, "mode": "none"}

    if isinstance(_model_instance, DummyModel):
        return {
            "loaded": True,
            "mode": "dummy",
            "num_frames": _model_instance.num_frames,
        }

    return {
        "loaded": True,
        "mode": "production",
        "model_path": str(_model_instance.model_path),
        "has_attention_model": _model_instance.attention_model is not None,
    }