"""
Script test load model .keras (versi E3).

Tujuan: Memastikan model bisa di-load SEBELUM masuk ke server FastAPI.
Kalau ada error, lebih mudah debug di sini.

Jalankan: python test_load.py
"""

import sys
import time
from pathlib import Path

# Pastikan root project ada di sys.path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def print_header(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main():
    print_header("🧪 TEST LOAD MODEL E3")

    # ============================================================
    # STEP 1: Cek TensorFlow
    # ============================================================
    print("\n[1/7] Cek TensorFlow...")
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow: {tf.__version__}")
        try:
            import keras
            print(f"   ✅ Keras     : {keras.__version__}")
        except ImportError:
            print(f"   ⚠️  Keras standalone tidak ada (pakai tf.keras)")
    except ImportError as e:
        print(f"   ❌ TensorFlow TIDAK terinstall: {e}")
        print("\n   Solusi: pip install -r requirements.txt")
        return 1

    # Cek GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"   ✅ GPU tersedia: {len(gpus)} device")
    else:
        print(f"   ℹ️  CPU mode (tidak ada GPU)")

    # ============================================================
    # STEP 2: Cek Config & Path Model
    # ============================================================
    print("\n[2/7] Cek config & path model...")
    try:
        from app.config import settings
    except Exception as e:
        print(f"   ❌ Gagal import config: {e}")
        return 1

    print(f"   Mode        : {'DUMMY' if settings.is_dummy_mode else 'PRODUCTION'}")
    print(f"   FRAMES      : {settings.FRAMES}")
    print(f"   IMG_SIZE    : {settings.IMG_SIZE}")
    print(f"   MODEL_PATH  : {settings.MODEL_PATH}")

    model_path = settings.model_path_absolute
    print(f"   Path absolut: {model_path}")

    if not model_path.exists():
        print(f"   ❌ File model TIDAK ADA!")
        print(f"\n   Solusi: copy file .keras ke: {model_path.parent}")
        return 1

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ File ada ({size_mb:.2f} MB)")

    # ============================================================
    # STEP 3: Build Arsitektur E3
    # ============================================================
    print("\n[3/7] Build arsitektur E3...")
    try:
        from app.services.model_loader import build_e3_model
        t0 = time.perf_counter()
        model = build_e3_model()
        t1 = time.perf_counter()
        print(f"   ✅ Arsitektur E3 berhasil dibangun ({t1 - t0:.1f}s)")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total params: {model.count_params():,}")
    except Exception as e:
        print(f"   ❌ Gagal build arsitektur: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================
    # STEP 4: Load Weights dari .keras
    # ============================================================
    print("\n[4/7] Load weights dari .keras...")
    try:
        t0 = time.perf_counter()
        model.load_weights(str(model_path))
        t1 = time.perf_counter()
        print(f"   ✅ Weights berhasil di-load ({t1 - t0:.1f}s)")
    except Exception as e:
        print(f"   ❌ Gagal load weights: {type(e).__name__}")
        print(f"   {e}")
        print("\n   Kemungkinan penyebab:")
        print("   1. Versi TF/Keras beda dengan Colab training")
        print("   2. Custom layer tidak cocok (cek custom_layers.py)")
        print("   3. Model corrupt / bukan format E3")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================
    # STEP 5: Build Attention Model (Multi-Output)
    # ============================================================
    print("\n[5/7] Build attention model (multi-output)...")
    try:
        from app.services.model_loader import build_attention_model
        attention_model = build_attention_model(model)
        print(f"   ✅ Attention model berhasil dibuat")
        print(f"   Inputs : {len(attention_model.inputs)}")
        print(f"   Outputs: {len(attention_model.outputs)}")
    except Exception as e:
        print(f"   ❌ Gagal build attention model: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================
    # STEP 6: Test Inference dengan Input Dummy
    # ============================================================
    print("\n[6/7] Test inference dengan input dummy...")
    try:
        # Bikin input dummy: (1, 32, 160, 160, 3) float32 0-255
        batch = np.random.rand(1, 32, 160, 160, 3).astype(np.float32) * 255

        t0 = time.perf_counter()
        prediction, attention = attention_model.predict(batch, verbose=0)
        t1 = time.perf_counter()

        print(f"   ✅ Inference berhasil ({t1 - t0:.2f}s)")
        print()
        print(f"   Prediction shape : {prediction.shape}")
        print(f"   Confidence       : {float(prediction[0][0]):.4f}")
        print()
        print(f"   Attention shape  : {attention.shape}")
        print(f"   Attention sum    : {attention.sum():.4f}")

        # Top-3
        att_flat = attention[0]
        top3 = np.argsort(att_flat)[-3:][::-1]
        print(f"   Top-3 index      : {top3.tolist()}")
        print(f"   Top-3 weights    : {[round(float(att_flat[i]), 4) for i in top3]}")

    except Exception as e:
        print(f"   ❌ Gagal inference: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================
    # STEP 7: Test RealModel Class (Full Pipeline)
    # ============================================================
    print("\n[7/7] Test RealModel class (full pipeline)...")
    try:
        from app.services.model_loader import RealModel

        t0 = time.perf_counter()
        real_model = RealModel(model_path=model_path)
        t1 = time.perf_counter()

        print(f"   ✅ RealModel berhasil di-load ({t1 - t0:.1f}s)")

        # Test predict
        conf, att = real_model.predict(batch)
        print(f"   ✅ Predict OK")
        print(f"   Confidence      : {conf:.4f}")
        print(f"   Attention shape : {att.shape}")
        print(f"   Attention sum   : {att.sum():.4f}")

    except Exception as e:
        print(f"   ❌ Gagal: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================
    # SELESAI
    # ============================================================
    print()
    print("=" * 65)
    print("  🎉 SEMUA TEST LULUS! Model siap production.")
    print("=" * 65)
    print()
    print("  Langkah selanjutnya:")
    print("  1. Set USE_DUMMY_MODEL=False di .env")
    print("  2. Restart server: uvicorn app.main:app --reload --port 8001")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Dibatalkan oleh user")
        exit_code = 130
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)