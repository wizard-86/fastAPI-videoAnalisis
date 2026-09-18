"""
Custom layer Keras — SAMA PERSIS dengan training.

Layer:
1. SEBlock          -> Squeeze-and-Excitation (pakai Conv2D 1x1)
2. TemporalAttention -> Attention per frame + return tuple

File ini di-import secara LAZY (hanya saat load model asli).
Kalau masih dummy mode, file ini tidak di-import.
"""

from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# LAZY IMPORT GUARD
# ============================================
def _ensure_keras_available():
    """Pastikan TensorFlow/Keras sudah terinstall."""
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TensorFlow belum terinstall. "
            "Aktifkan tensorflow & keras di requirements.txt, "
            "lalu jalankan: pip install -r requirements.txt"
        ) from e


# ============================================
# SEBLOCK — VERSI TRAINING
# ============================================
def _make_seblock():
    """
    Factory function untuk SEBlock (butuh closure TF).
    Dipanggil setiap kali butuh class baru.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    class SEBlock(layers.Layer):
        """
        Squeeze-and-Excitation Block (versi training).
        Pakai Conv2D 1x1, BUKAN Dense.
        """

        def __init__(self, reduction=16, **kwargs):
            super().__init__(**kwargs)
            self.reduction = reduction

            self.gap = layers.GlobalAveragePooling2D(
                keepdims=True,
                name="se_gap",
            )

            self.dense1 = None
            self.dense2 = None

        def build(self, input_shape):
            channels = int(input_shape[-1])
            reduced_channels = max(channels // self.reduction, 1)

            # Reduce channel
            self.dense1 = layers.Conv2D(
                reduced_channels,
                kernel_size=1,
                activation="relu",
                padding="same",
                name="se_reduce",
            )

            # Expand channel
            self.dense2 = layers.Conv2D(
                channels,
                kernel_size=1,
                activation="sigmoid",
                padding="same",
                name="se_expand",
            )

            # Explicit build child layers
            self.dense1.build((None, 1, 1, channels))
            self.dense2.build((None, 1, 1, reduced_channels))

            super().build(input_shape)

        def call(self, inputs):
            scale = self.gap(inputs)
            scale = self.dense1(scale)
            scale = self.dense2(scale)
            return inputs * scale

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            config = super().get_config()
            config.update({"reduction": self.reduction})
            return config

    return SEBlock


# ============================================
# TEMPORAL ATTENTION — VERSI TRAINING
# ============================================
def _make_temporal_attention():
    """
    Factory function untuk TemporalAttention (butuh closure TF).
    Return tuple: (context, attention_weights).
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    class TemporalAttention(layers.Layer):
        """
        Temporal Attention (versi training).
        Pakai W (feature_dim, 1) + bias b.
        Return tuple (context, attention_weights).
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            feature_dim = int(input_shape[-1])

            self.W = self.add_weight(
                name="attention_weight",
                shape=(feature_dim, 1),
                initializer="glorot_uniform",
                trainable=True,
            )

            self.b = self.add_weight(
                name="attention_bias",
                shape=(1,),
                initializer="zeros",
                trainable=True,
            )

            super().build(input_shape)

        def call(self, inputs):
            # Score tiap frame
            score = tf.tanh(
                tf.matmul(inputs, self.W) + self.b
            )

            # Softmax antar frame
            attention_weights = tf.nn.softmax(score, axis=1)

            # Weighted sum
            context = tf.reduce_sum(
                inputs * attention_weights,
                axis=1,
            )

            return (
                context,
                tf.squeeze(attention_weights, axis=-1),
            )

        def compute_output_shape(self, input_shape):
            batch_size = input_shape[0]
            time_steps = input_shape[1]
            feature_dim = input_shape[2]

            return (
                (batch_size, feature_dim),
                (batch_size, time_steps),
            )

        def get_config(self):
            return super().get_config()

    return TemporalAttention


# ============================================
# EXPORT: UNTUK load_model(custom_objects=...)
# ============================================
def get_custom_layers() -> dict:
    """
    Return dict {nama_layer: class} untuk dipakai di
    `tf.keras.models.load_model(..., custom_objects=...)`.

    Contoh:
        from app.models.custom_layers import get_custom_layers
        model = tf.keras.models.load_model(
            "best_model.keras",
            custom_objects=get_custom_layers()
        )
    """
    _ensure_keras_available()

    SEBlock = _make_seblock()
    TemporalAttention = _make_temporal_attention()

    logger.info("✅ Custom layers (SEBlock, TemporalAttention) siap dipakai")

    return {
        "SEBlock": SEBlock,
        "TemporalAttention": TemporalAttention,
    }


# ============================================
# EXPORT: UNTUK build_e3_model()
# ============================================
def get_seblock_class():
    """
    Return class SEBlock untuk dipakai di build_e3_model().
    """
    _ensure_keras_available()
    return _make_seblock()


def get_temporal_attention_class():
    """
    Return class TemporalAttention untuk dipakai di build_e3_model().
    """
    _ensure_keras_available()
    return _make_temporal_attention()