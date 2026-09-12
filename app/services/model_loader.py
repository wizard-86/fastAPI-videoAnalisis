"""
Service untuk load model.

Dua mode:
1. DUMMY MODE (USE_DUMMY_MODEL=True)
   - Tidak butuh TensorFlow
   - Return DummyModel yang generate attention weights random
   - Berguna untuk development & testing endpoint

2. PRODUCTION MODE (USE_DUMMY_MODEL=False)
   - Load model .keras asli
   - Butuh TensorFlow/Keras
   - Ambil layer TemporalAttention untuk ekstrak attention weights

Model di-load SEKALI saja (singleton) supaya tidak lambat.
"""

from typing import Any, Optional, Protocol
from pathlib import Path

import numpy as np

from app.config import settings
from app.core.exceptions import ModelNotLoadedError
from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# PROTOCOL: Interface Model
# ============================================
class BaseModel(Protocol):
    """Interface yang harus dipenuhi model (dummy atau asli)."""

    def predict(self, frames: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Args:
            frames: numpy array shape (1, 32, 160, 160, 3)

        Returns:
            (confidence, attention_weights)
            confidence: float 0-1
            attention_weights: numpy array shape (32,) sum=1.0
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
    
    Sengaja bikin 1-2 frame punya bobot tinggi biar mirip
    dengan distribusi attention asli.
    """

    def __init__(self, num_frames: int = 32, seed: Optional[int] = None):
        self.num_frames = num_frames
        self.rng = np.random.default_rng(seed)

    def predict(self, frames: np.ndarray) -> tuple[float, np.ndarray]:
        """Generate prediksi & attention weights random."""
        # Confidence random 0-1
        confidence = float(self.rng.uniform(0.0, 1.0))

        # Attention weights pakai Dirichlet (sum otomatis 1.0)
        # alpha < 1 bikin distribusi "sparse" (beberapa frame dominan)
        alpha = np.ones(self.num_frames) * 0.5
        attention = self.rng.dirichlet(alpha).astype(np.float32)

        # Pastikan shape sesuai
        assert attention.shape == (self.num_frames,), \
            f"Shape attention salah: {attention.shape}"
        assert abs(attention.sum() - 1.0) < 1e-5, \
            f"Sum attention bukan 1.0: {attention.sum()}"

        return confidence, attention


# ============================================
# REAL MODEL WRAPPER
# ============================================
class RealModel:
    """
    Wrapper untuk model Keras asli.
    - Load model .keras
    - Bangun sub-model untuk ekstrak attention weights
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.inference_model = None
        self._load()

    def _load(self) -> None:
        """Load model dari disk."""
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

        logger.info(f"Loading model dari: {self.model_path}")

        # Import custom layers
        from app.models.custom_layers import get_custom_layers
        custom_objects = get_custom_layers()

        try:
            self.model = tf.keras.models.load_model(
                str(self.model_path),
                custom_objects=custom_objects,
                compile=False,  # Tidak butuh loss/optimizer untuk inference
            )
            logger.info("✅ Model berhasil di-load")
        except Exception as e:
            raise ModelNotLoadedError(
                message=f"Gagal load model: {e}",
                details={"path": str(self.model_path)},
            ) from e

        # Bangun inference model (output: confidence + attention)
        self._build_inference_model(tf)

    def _build_inference_model(self, tf: Any) -> None:
        """
        Bangun sub-model yang output-nya (confidence, attention_weights).

        Model asli punya arsitektur:
            ...
            -> Bidirectional(LSTM) -> TemporalAttention() -> context, att_weights
            -> Dropout -> Dense(128) -> Dense(1) sigmoid

        Kita perlu akses output `att_weights` dari layer TemporalAttention.
        """
        try:
            # Cari layer TemporalAttention
            att_layer = None
            for layer in self.model.layers:
                if layer.__class__.__name__ == "TemporalAttention":
                    att_layer = layer
                    break

            if att_layer is None:
                raise ModelNotLoadedError(
                    message="Layer TemporalAttention tidak ditemukan di model. "
                            "Pastikan model di-training dengan custom layer ini.",
                )

            # TemporalAttention return tuple (context, att_weights)
            # Kita perlu bangun model baru dengan output [prediction, att_weights]
            #
            # Cara paling aman: pakai functional API dari model asli
            # Cari layer setelah attention (Dropout, Dense) untuk prediction output
            #
            # Strategi: buat model dengan output = [model.output, att_layer.output]
            #
            # Keterbatasan: Keras 3 kadang tidak bisa akses layer.output kalau
            # layer return tuple. Alternatif: bikin layer wrapper.

            logger.info(
                f"Attention layer ditemukan: {att_layer.name} "
                f"(class={att_layer.__class__.__name__})"
            )

            # Coba bangun inference model
            # Catatan: ini bergantung pada bagaimana model asli dibangun
            # Kalau error, kita fallback ke model.predict() + akses manual
            try:
                self.inference_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=[self.model.output, att_layer.output],
                )
                logger.info("✅ Inference model (multi-output) berhasil dibuat")
            except Exception as e:
                logger.warning(
                    f"Gagal bangun inference model: {e}. "
                    f"Fallback ke mode single-output."
                )
                self.inference_model = None

        except ModelNotLoadedError:
            raise
        except Exception as e:
            raise ModelNotLoadedError(
                message=f"Gagal bangun inference model: {e}",
            ) from e

    def predict(self, frames: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Args:
            frames: shape (1, 32, 160, 160, 3) uint8 atau float32

        Returns:
            (confidence, attention_weights)
        """
        if self.model is None:
            raise ModelNotLoadedError(message="Model belum di-load")

        # Preprocess
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

        # Model Anda sudah punya Lambda(preprocess_input) di dalamnya,
        # tapi untuk aman kita tidak preprocess lagi di sini.
        # Kalau model TIDAK punya Lambda, uncomment baris di bawah:
        # x = preprocess_input(frames.astype(np.float32))

        x = frames.astype(np.float32)

        try:
            if self.inference_model is not None:
                # Multi-output: [prediction, attention]
                outputs = self.inference_model.predict(x, verbose=0)

                if isinstance(outputs, list) and len(outputs) == 2:
                    pred, att = outputs
                else:
                    # Fallback: mungkin output-nya single
                    pred = outputs
                    att = None

                confidence = float(pred.flatten()[0])

                if att is None:
                    # Kalau attention tidak ter-extract, fallback uniform
                    logger.warning("Attention tidak ter-extract, pakai uniform")
                    att = np.ones(settings.FRAMES, dtype=np.float32) / settings.FRAMES

                attention = np.asarray(att).flatten().astype(np.float32)

            else:
                # Fallback: single-output prediction
                pred = self.model.predict(x, verbose=0)
                confidence = float(np.asarray(pred).flatten()[0])
                logger.warning("Attention tidak tersedia, pakai uniform")
                attention = np.ones(settings.FRAMES, dtype=np.float32) / settings.FRAMES

            # Normalisasi attention (jaga-jaga)
            att_sum = attention.sum()
            if att_sum > 0:
                attention = attention / att_sum

            # Pastikan panjang = FRAMES
            if len(attention) != settings.FRAMES:
                logger.warning(
                    f"Panjang attention={len(attention)}, "
                    f"expected={settings.FRAMES}. Resize..."
                )
                attention = _resize_attention(attention, settings.FRAMES)

            return confidence, attention

        except Exception as e:
            from app.core.exceptions import InferenceError
            raise InferenceError(
                message=f"Gagal menjalankan prediksi: {e}",
            ) from e


# ============================================
# HELPER: RESIZE ATTENTION
# ============================================
def _resize_attention(attention: np.ndarray, target_len: int) -> np.ndarray:
    """Resize attention weights kalau panjangnya tidak sesuai."""
    if len(attention) == target_len:
        return attention

    # Interpolasi linear
    x_old = np.linspace(0, 1, len(attention))
    x_new = np.linspace(0, 1, target_len)
    resized = np.interp(x_new, x_old, attention)

    # Normalisasi ulang
    total = resized.sum()
    if total > 0:
        resized = resized / total

    return resized.astype(np.float32)


# ============================================
# SINGLETON MODEL
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
        logger.info("Loading model asli...")
        model_path = settings.model_path_absolute
        _model_instance = RealModel(model_path=model_path)

    return _model_instance


def get_model() -> Any:
    """
    Return model yang sudah di-load.
    Raise kalau belum di-load.
    """
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
        return {
            "loaded": False,
            "mode": "none",
        }

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
        "has_inference_model": _model_instance.inference_model is not None,
    }