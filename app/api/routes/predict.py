"""
Endpoint POST /predict.

Alur:
1. Terima upload video
2. Validasi file (ekstensi, MIME, ukuran)
3. Simpan ke folder sementara
4. Load video -> sampling 32 frame
5. Jalankan inference -> confidence + attention weights
6. Ekstrak top-K frame -> base64
7. Return response JSON
8. Cleanup file sementara
"""

import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.core.exceptions import AppException, VideoUnreadableError
from app.core.logger import get_logger
from app.schemas.prediction import (
    PREDICTION_RESPONSE_EXAMPLE,
    PredictionMetadata,
    PredictionResponse,
    TopFrame,
    VideoMetadata,
)
from app.services.frame_extractor import extract_top_frames_with_timestamp
from app.services.inference import run_inference
from app.services.video_loader import load_video
from app.utils.file_utils import (
    cleanup_file,
    save_upload_to_temp,
    validate_video_upload,
)


logger = get_logger(__name__)


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


# ============================================
# MODEL VERSION (untuk metadata)
# ============================================
def _get_model_version() -> str:
    """Return string versi model untuk metadata."""
    if settings.is_dummy_mode:
        return "dummy-random-v1"
    return "efficientnetv2b0-bilstm-attention-v1"


# ============================================
# POST /predict
# ============================================
@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Deteksi shoplifting dari video",
    description=(
        "Upload video (mp4/avi/mov/mkv/webm), sistem akan:\n\n"
        "1. Sampling 32 frame dari video\n"
        "2. Klasifikasi: **Normal** atau **Shoplifting**\n"
        "3. Return confidence score\n"
        "4. Return attention weights per frame (untuk grafik)\n"
        "5. Return top-3 frame dengan bobot tertinggi (base64 PNG)\n\n"
        "**Catatan**: Video akan dihapus setelah diproses."
    ),
    responses={
        200: {
            "description": "Prediksi berhasil",
            "content": {
                "application/json": {
                    "example": PREDICTION_RESPONSE_EXAMPLE,
                }
            },
        },
        400: {"description": "Format file tidak valid"},
        413: {"description": "File terlalu besar"},
        422: {"description": "Video tidak bisa dibaca"},
        500: {"description": "Error saat inference"},
        503: {"description": "Model belum siap"},
    },
)
async def predict_video(
    video: UploadFile = File(
        ...,
        description="File video (mp4, avi, mov, mkv, webm). Max 50 MB.",
    ),
) -> PredictionResponse:
    """
    Endpoint utama untuk deteksi shoplifting.

    Terima file video via `multipart/form-data`, return hasil prediksi
    beserta attention weights dan top-3 frame.
    """
    start_time = time.perf_counter()
    temp_path: Path | None = None

    logger.info(
        f"📥 Request masuk: filename={video.filename}, "
        f"content_type={video.content_type}"
    )

    try:
        # ---------- [1] VALIDASI FILE ----------
        validate_video_upload(video)
        logger.debug("✅ Validasi file lolos")

        # ---------- [2] SIMPAN KE TEMP ----------
        temp_path = save_upload_to_temp(video)
        logger.debug(f"File disimpan sementara: {temp_path}")

        # ---------- [3] LOAD VIDEO ----------
        frames, video_meta = await run_in_threadpool(
            load_video, temp_path
        )
        logger.debug(f"Frame ter-load: shape={frames.shape}")

        # ---------- [4] INFERENCE ----------
        result = await run_in_threadpool(run_inference, frames)
        logger.debug(
            f"Inference selesai: prediction={result.prediction}, "
            f"confidence={result.confidence:.4f}"
        )

        # ---------- [5] EKSTRAK TOP-K FRAME ----------
        top_frames_raw = await run_in_threadpool(
            extract_top_frames_with_timestamp,
            temp_path,
            result.top_indices,
            result.top_weights,
            result.total_frames,
            video_meta.duration_sec,
        )
        logger.debug(f"Top frames diekstrak: {len(top_frames_raw)}")

        # ---------- [6] SUSUN RESPONSE ----------
        top_frames = [TopFrame(**tf) for tf in top_frames_raw]

        video_metadata = VideoMetadata(
            filename=video.filename or "unknown.mp4",
            total_frames=video_meta.total_frames,
            fps=round(video_meta.fps, 2),
            width=video_meta.width,
            height=video_meta.height,
            duration_sec=round(video_meta.duration_sec, 2),
            resolution=f"{video_meta.width}x{video_meta.height}",
        )

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        prediction_metadata = PredictionMetadata(
            processing_time_ms=processing_time_ms,
            model_version=_get_model_version(),
            mode="dummy" if settings.is_dummy_mode else "production",
            threshold=settings.CONFIDENCE_THRESHOLD,
        )

        response = PredictionResponse(
            prediction=result.prediction,
            confidence=round(result.confidence, 4),
            total_frames=result.total_frames,
            attention_weights=[
                round(float(w), 6) for w in result.attention_weights
            ],
            top_frames=top_frames,
            video_metadata=video_metadata,
            prediction_metadata=prediction_metadata,
        )

        logger.info(
            f"✅ Request selesai dalam {processing_time_ms} ms | "
            f"prediction={result.prediction} | "
            f"confidence={result.confidence:.4f}"
        )

        return response

    except AppException:
        # Sudah ditangani handler di core/exceptions.py
        raise

    except Exception as e:
        logger.exception(f"Unexpected error saat prediksi: {e}")
        # Bungkus jadi error generic 500
        from app.core.exceptions import InferenceError
        raise InferenceError(
            message=f"Terjadi kesalahan saat memproses video: {e}",
        ) from e

    finally:
        # ---------- [7] CLEANUP ----------
        cleanup_file(temp_path)


# ============================================
# POST /predict/batch (opsional, untuk nanti)
# ============================================
# Kalau nanti mau support batch upload, tinggal tambah endpoint baru
# di sini tanpa mengubah yang sudah ada.