"""
Pydantic schemas untuk request & response API.
FastAPI otomatis pakai ini untuk validasi & dokumentasi Swagger.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================
# TOP FRAME
# ============================================
class TopFrame(BaseModel):
    """Satu frame dengan bobot attention tertinggi."""

    rank: int = Field(
        ...,
        ge=1,
        description="Urutan ranking (1 = bobot tertinggi)",
        examples=[1],
    )
    frame_index: int = Field(
        ...,
        ge=0,
        description="Index frame dalam 32 sampling (0-31)",
        examples=[12],
    )
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Bobot attention (0-1)",
        examples=[0.1521],
    )
    timestamp_sec: Optional[float] = Field(
        None,
        ge=0.0,
        description="Posisi frame dalam video (detik)",
        examples=[3.75],
    )
    image: str = Field(
        ...,
        description="Frame dalam base64 PNG (data URI)",
        examples=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."],
    )


# ============================================
# METADATA VIDEO
# ============================================
class VideoMetadata(BaseModel):
    """Info video yang diproses."""

    filename: str = Field(
        ...,
        description="Nama file video yang di-upload",
        examples=["contoh_shoplifting.mp4"],
    )
    total_frames: int = Field(
        ...,
        ge=0,
        description="Total frame di video asli",
        examples=[306],
    )
    fps: float = Field(
        ...,
        ge=0.0,
        description="Frame per second video asli",
        examples=[30.0],
    )
    width: int = Field(
        ...,
        ge=0,
        description="Lebar video (piksel)",
        examples=[1920],
    )
    height: int = Field(
        ...,
        ge=0,
        description="Tinggi video (piksel)",
        examples=[1080],
    )
    duration_sec: float = Field(
        ...,
        ge=0.0,
        description="Durasi video (detik)",
        examples=[10.2],
    )
    resolution: str = Field(
        ...,
        description="Resolusi dalam format WxH",
        examples=["1920x1080"],
    )


# ============================================
# PREDICTION METADATA
# ============================================
class PredictionMetadata(BaseModel):
    """Info tentang proses prediksi."""

    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Waktu proses prediksi (milidetik)",
        examples=[1842],
    )
    model_version: str = Field(
        ...,
        description="Versi model yang dipakai",
        examples=["efficientnetv2b0-bilstm-attention-v1"],
    )
    mode: str = Field(
        ...,
        description="Mode model: 'production' atau 'dummy'",
        examples=["production"],
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold confidence untuk klasifikasi",
        examples=[0.5],
    )


# ============================================
# PREDICTION RESPONSE
# ============================================
class PredictionResponse(BaseModel):
    """
    Response dari endpoint POST /predict.

    Berisi hasil klasifikasi, attention weights per frame,
    dan top-K frame dengan bobot tertinggi.
    """

    prediction: str = Field(
        ...,
        description="Label klasifikasi: 'Normal' atau 'Shoplifting'",
        examples=["Shoplifting"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1). > threshold = Shoplifting",
        examples=[0.8734],
    )
    total_frames: int = Field(
        ...,
        ge=1,
        description="Jumlah frame yang dianalisis (32)",
        examples=[32],
    )
    attention_weights: List[float] = Field(
        ...,
        description=(
            "Bobot attention per frame. "
            "Panjang = total_frames. Sum ≈ 1.0. "
            "Nilai lebih tinggi = frame lebih penting."
        ),
        examples=[[0.0125, 0.0421, 0.0312, 0.1521, 0.0234]],
    )
    top_frames: List[TopFrame] = Field(
        ...,
        description="Top-K frame dengan bobot attention tertinggi",
    )
    video_metadata: VideoMetadata = Field(
        ...,
        description="Info video yang diproses",
    )
    prediction_metadata: PredictionMetadata = Field(
        ...,
        description="Info proses prediksi",
    )


# ============================================
# HEALTH RESPONSE
# ============================================
class HealthResponse(BaseModel):
    """Response dari endpoint GET /health."""

    status: str = Field(
        ...,
        description="Status aplikasi: 'ok' atau 'degraded'",
        examples=["ok"],
    )
    app_name: str = Field(
        ...,
        examples=["Shoplifting Detection API"],
    )
    app_version: str = Field(
        ...,
        examples=["0.1.0"],
    )
    environment: str = Field(
        ...,
        description="development / staging / production",
        examples=["development"],
    )
    model_loaded: bool = Field(
        ...,
        description="Apakah model sudah di-load",
        examples=[True],
    )
    model_mode: str = Field(
        ...,
        description="Mode model: 'dummy' atau 'production'",
        examples=["dummy"],
    )
    timestamp: str = Field(
        ...,
        description="Waktu server (ISO format)",
        examples=["2025-01-15T10:23:45.123456"],
    )


# ============================================
# ERROR RESPONSE
# ============================================
class ErrorResponse(BaseModel):
    """Response error standar."""

    detail: str = Field(
        ...,
        description="Pesan error untuk user",
        examples=["Format file tidak didukung"],
    )
    error_code: str = Field(
        ...,
        description="Kode error terstruktur",
        examples=["INVALID_FILE_FORMAT"],
    )
    timestamp: str = Field(
        ...,
        description="Waktu error (ISO format)",
        examples=["2025-01-15T10:23:45.123456"],
    )
    details: Optional[dict] = Field(
        None,
        description="Info tambahan (opsional)",
        examples=[{"allowed": [".mp4", ".avi"]}],
    )


# ============================================
# CONTOH RESPONSE (untuk dokumentasi)
# ============================================
PREDICTION_RESPONSE_EXAMPLE = {
    "prediction": "Shoplifting",
    "confidence": 0.8734,
    "total_frames": 32,
    "attention_weights": [
        0.0125, 0.0089, 0.0152, 0.0210, 0.0187, 0.0251, 0.0314, 0.0283,
        0.0352, 0.0421, 0.0387, 0.0452, 0.1521, 0.0483, 0.0391, 0.0332,
        0.0294, 0.0261, 0.0223, 0.0195, 0.0164, 0.0142, 0.0118, 0.0093,
        0.0131, 0.0172, 0.0243, 0.1214, 0.0271, 0.0203, 0.0156, 0.0104,
    ],
    "top_frames": [
        {
            "rank": 1,
            "frame_index": 12,
            "weight": 0.1521,
            "timestamp_sec": 3.75,
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
        },
        {
            "rank": 2,
            "frame_index": 27,
            "weight": 0.1214,
            "timestamp_sec": 8.45,
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
        },
        {
            "rank": 3,
            "frame_index": 11,
            "weight": 0.0452,
            "timestamp_sec": 3.45,
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
        },
    ],
    "video_metadata": {
        "filename": "contoh_shoplifting.mp4",
        "total_frames": 306,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "duration_sec": 10.2,
        "resolution": "1920x1080",
    },
    "prediction_metadata": {
        "processing_time_ms": 1842,
        "model_version": "efficientnetv2b0-bilstm-attention-v1",
        "mode": "production",
        "threshold": 0.5,
    },
}