"""
Entry point FastAPI aplikasi.

Cara jalankan:
    uvicorn app.main:app --reload
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, predict
from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import get_logger
from app.services.model_loader import get_model_info, load_model
from app.utils.file_utils import cleanup_upload_dir


logger = get_logger(__name__)


# ============================================
# LIFESPAN (startup + shutdown)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event:
    - Startup: load model, bersihkan folder temp
    - Shutdown: bersihkan resource
    """
    # ========== STARTUP ==========
    logger.info("=" * 70)
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"   Environment : {settings.APP_ENV}")
    logger.info(f"   Debug       : {settings.DEBUG}")
    logger.info(f"   Host:Port   : {settings.HOST}:{settings.PORT}")
    logger.info("=" * 70)

    # 1. Bersihkan folder upload dari sisa session sebelumnya
    try:
        cleanup_upload_dir()
    except Exception as e:
        logger.warning(f"Gagal cleanup upload dir: {e}")

    # 2. Load model (singleton)
    try:
        logger.info("📦 Loading model...")
        start = time.perf_counter()
        load_model()
        elapsed = (time.perf_counter() - start) * 1000
        info = get_model_info()
        logger.info(
            f"✅ Model siap dalam {elapsed:.0f} ms | "
            f"mode={info.get('mode')}"
        )
    except Exception as e:
        logger.exception(f"❌ Gagal load model: {e}")
        logger.warning(
            "⚠️  Server tetap jalan, tapi /predict akan error "
            "sampai model berhasil di-load."
        )

    logger.info("=" * 70)
    logger.info("✅ Application ready")
    logger.info("=" * 70)

    yield  # <-- aplikasi jalan di sini

    # ========== SHUTDOWN ==========
    logger.info("=" * 70)
    logger.info("🛑 Shutting down...")

    # Bersihkan folder upload
    try:
        cleanup_upload_dir()
    except Exception as e:
        logger.warning(f"Gagal cleanup upload dir saat shutdown: {e}")

    logger.info("👋 Bye!")
    logger.info("=" * 70)


# ============================================
# BUAT APP
# ============================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "REST API untuk deteksi **shoplifting** dari video.\n\n"
        "### Fitur\n"
        "- Klasifikasi video: **Normal** atau **Shoplifting**\n"
        "- Confidence score\n"
        "- **Attention weights** per frame (32 frame) untuk grafik\n"
        "- **Top-3 frame** dengan bobot tertinggi (base64 PNG)\n\n"
        "### Cara pakai\n"
        "1. Buka `POST /predict`\n"
        "2. Klik **Try it out**\n"
        "3. Upload video (mp4/avi/mov/mkv/webm, max 50 MB)\n"
        "4. Klik **Execute**\n"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG,
)


# ============================================
# CORS MIDDLEWARE
# ============================================
# Untuk development: izinkan semua origin
# Untuk production: ganti allow_origins ke domain spesifik
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Ganti ke ["https://domain.com"] di production
    allow_credentials=False,       # Tidak bisa True kalau allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# REQUEST LOGGING MIDDLEWARE
# ============================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log setiap request: method, path, status, durasi.
    """
    start = time.perf_counter()

    # Skip logging untuk docs & static biar tidak berisik
    skip_paths = ("/docs", "/redoc", "/openapi.json", "/favicon.ico")
    if request.url.path in skip_paths:
        return await call_next(request)

    logger.info(f"➡️  {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"⬅️  {request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed_ms:.0f} ms)"
        )
        return response
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            f"💥 {request.method} {request.url.path} "
            f"→ ERROR ({elapsed_ms:.0f} ms): {e}"
        )
        raise


# ============================================
# REGISTER EXCEPTION HANDLERS
# ============================================
register_exception_handlers(app)


# ============================================
# REGISTER ROUTERS
# ============================================
app.include_router(health.router)
app.include_router(predict.router)


# ============================================
# ROOT ENDPOINT
# ============================================
@app.get(
    "/",
    tags=["Root"],
    summary="Info API",
    description="Endpoint root untuk cek API & lihat link ke dokumentasi.",
)
async def root() -> dict:
    """Info singkat tentang API."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "predict": "/predict",
        "message": "Shoplifting Detection API is running 🚀",
    }


# ============================================
# FAVICON (biar tidak 404)
# ============================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return 204 No Content untuk favicon (biar tidak 404 di browser)."""
    return JSONResponse(content=None, status_code=204)


# ============================================
# ENTRY POINT (opsional, untuk `python app/main.py`)
# ============================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )