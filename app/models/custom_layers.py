"""
Definisi custom layer Keras yang dipakai di model:
1. SEBlock       -> Squeeze-and-Excitation (channel attention)
2. TemporalAttention -> attention per frame (temporal)

File ini di-import secara LAZY (hanya saat load model asli).
Kalau masih dummy mode, file ini tidak di-import.
"""

from app.core.logger import get_logger


logger = get_logger(__name__)


# ============================================
# LAZY IMPORT GUARD
# ============================================
def _ensure_keras_available():
    """
    Pastikan TensorFlow/Keras sudah terinstall.
    Kalau belum, raise error yang jelas.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras import layers, backend as K  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TensorFlow belum terinstall. "
            "Aktifkan tensorflow & keras di requirements.txt, "
            "lalu jalankan: pip install -r requirements.txt"
        ) from e


# ============================================
# CUSTOM LAYERS
# ============================================
def get_custom_layers() -> dict:
    """
    Return dict {nama_layer: class} untuk dipakai
    di `tf.keras.models.load_model(..., custom_objects=...)`.

    Contoh pemakaian:
        from app.models.custom_layers import get_custom_layers
        model = tf.keras.models.load_model(
            "best_model.keras",
            custom_objects=get_custom_layers()
        )
    """
    _ensure_keras_available()

    import tensorflow as tf
    from tensorflow.keras import layers, backend as K

    # ============================================
    # SEBlock (Squeeze-and-Excitation)
    # ============================================
    class SEBlock(layers.Layer):
        """
        Squeeze-and-Excitation block.
        - Squeeze: Global Average Pooling -> (batch, channels)
        - Excitation: Dense -> Dense (sigmoid) -> (batch, channels)
        - Scale: Multiply input dengan attention weights
        """

        def __init__(self, reduction: int = 16, **kwargs):
            super().__init__(**kwargs)
            self.reduction = reduction

        def build(self, input_shape):
            channels = int(input_shape[-1])
            self.dense1 = layers.Dense(
                max(1, channels // self.reduction),
                activation="relu",
                name=f"{self.name}_dense1",
            )
            self.dense2 = layers.Dense(
                channels,
                activation="sigmoid",
                name=f"{self.name}_dense2",
            )
            super().build(input_shape)

        def call(self, inputs):
            x = layers.GlobalAveragePooling2D()(inputs)
            x = self.dense1(x)
            x = self.dense2(x)
            x = layers.Reshape((1, 1, int(inputs.shape[-1])))(x)
            return layers.Multiply()([inputs, x])

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            config = super().get_config()
            config.update({"reduction": self.reduction})
            return config

    # ============================================
    # TemporalAttention
    # ============================================
    class TemporalAttention(layers.Layer):
        """
        Attention per frame (temporal dimension).
        Input : (batch, T, features)
        Output: context_vector (batch, features), attention_weights (batch, T)
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name="att_weight",
                shape=(input_shape[-1],),
                initializer="random_normal",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):
            # inputs: (batch, T, features)
            scores = K.tanh(K.sum(inputs * self.W, axis=-1))  # (batch, T)
            alpha = K.softmax(scores)                          # (batch, T)
            alpha_exp = K.expand_dims(alpha, axis=-1)          # (batch, T, 1)
            context = K.sum(inputs * alpha_exp, axis=1)        # (batch, features)
            return context, alpha

        def compute_output_shape(self, input_shape):
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]

    logger.info("✅ Custom layers (SEBlock, TemporalAttention) siap dipakai")

    return {
        "SEBlock": SEBlock,
        "TemporalAttention": TemporalAttention,
    }


# ============================================
# CATATAN UNTUK NANTI
# ============================================
# Saat model asli sudah siap:
#
# 1. Uncomment tensorflow & keras di requirements.txt
# 2. pip install -r requirements.txt
# 3. Set USE_DUMMY_MODEL=False di .env
# 4. Taruh file best_model.keras di app/ml_models/
# 5. model_loader.py akan otomatis load dengan custom_objects
#
# Kalau ada error seperti:
#   "Unknown layer: SEBlock"
#   "Unknown layer: TemporalAttention"
#
# Berarti custom_objects belum terpasang dengan benar.
# Cek: pastikan get_custom_layers() dipanggil saat load_model().