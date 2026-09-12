"""
Endpoint /health untuk cek status aplikasi.
"""

from datetime import datetime

from fastapi import APIRouter, status

from app.config import settings
from app.core.logger import get_logger
from app.schemas.prediction import HealthResponse
from app.services.model_loader import get_model_info, is_model_loaded


logger = get_logger(__name__)


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


# ============================================
# GET /health
# ============================================
@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Cek status aplikasi",
    description=(
        "Endpoint untuk cek apakah API hidup dan model sudah di-load. "
        "Berguna untuk monitoring / load balancer healthcheck."
    ),
    responses={
        200: {
            "description": "Aplikasi sehat",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "app_name": "Shoplifting Detection API",
                        "app_version": "0.1.0",
                        "environment": "development",
                        "model_loaded": True,
                        "model_mode": "dummy",
                        "timestamp": "2025-01-15T10:23:45.123456",
                    }
                }
            },
        },
        503: {
            "description": "Model belum siap",
        },
    },
)
async def health_check() -> HealthResponse:
    """
    Return status aplikasi.
    
    - **status**: `ok` kalau model sudah di-load, `degraded` kalau belum
    - **model_loaded**: apakah model sudah siap
    - **model_mode**: `dummy` (development) atau `production`
    """
    model_info = get_model_info()
    loaded = model_info.get("loaded", False)
    mode = model_info.get("mode", "none")

    app_status = "ok" if loaded else "degraded"

    response = HealthResponse(
        status=app_status,
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        model_loaded=loaded,
        model_mode=mode,
        timestamp=datetime.utcnow().isoformat(),
    )

    logger.debug(f"Health check: status={app_status}, mode={mode}")
    return response


# ============================================
# GET /health/ready (readiness probe)
# ============================================
@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description=(
        "Cek apakah aplikasi siap menerima request. "
        "Return 200 kalau model sudah di-load, 503 kalau belum. "
        "Cocok untuk Kubernetes readiness probe."
    ),
    responses={
        200: {"description": "Siap menerima request"},
        503: {"description": "Belum siap"},
    },
)
async def readiness_check() -> dict:
    """
    Readiness probe untuk orchestrator (Kubernetes/Docker Swarm).
    
    Return 200 kalau model sudah di-load, 503 kalau belum.
    """
    from fastapi import HTTPException

    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum di-load",
        )

    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================
# GET /health/live (liveness probe)
# ============================================
@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description=(
        "Cek apakah aplikasi masih hidup (proses tidak hang). "
        "Selalu return 200 selama server bisa merespon. "
        "Cocok untuk Kubernetes liveness probe."
    ),
)
async def liveness_check() -> dict:
    """
    Liveness probe untuk orchestrator.
    
    Selalu return 200 selama server bisa merespon.
    Kalau server hang, probe ini timeout & orchestrator restart container.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }