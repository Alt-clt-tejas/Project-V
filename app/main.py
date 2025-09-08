# app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import search, health, metrics, enrichment
from app.config.base import AppSettings
from app.api.dependencies import lifespan, get_settings

from app.api.search import router as search_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.enrichment import router as enrichment_router

from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.metrics_middleware import MetricsMiddleware
from app.utils.logging_config import setup_logging

# --- Initialization ---
settings: AppSettings = get_settings()
setup_logging(log_level=settings.LOG_LEVEL, log_file="logs/app.log" if not settings.DEBUG else None)
logger = logging.getLogger(__name__)

# --- FastAPI App Creation ---
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# --- Middleware Configuration ---
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Router Configuration ---
# Use the imported router objects directly.
app.include_router(search_router, prefix="/api/v1")
app.include_router(enrichment_router, prefix="/api/v1")  # Added enrichment router
app.include_router(health_router, prefix="/api/v1")
app.include_router(metrics_router, prefix="/api/v1")

# --- Root Endpoint ---
@app.get("/", tags=["Root"], summary="API Root Endpoint")
async def root():
    """Provides basic information about the API."""
    return {
        "name": settings.APP_NAME,
        "status": "running",
        "version": "1.0.0",
        "docs": app.docs_url,
        "health_check": "/api/v1/health/ready",
        "enrichment_health": "/api/v1/enrich/health"  # Added enrichment health endpoint
    }

logger.info(f"'{settings.APP_NAME}' application initialization complete.")