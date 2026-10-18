"""FastAPI backend for EpiWatch."""
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Securely load environment variables on backend startup
load_dotenv()

from src.api.routes import router
from src.surveillance.worker import get_surveillance_worker

logger = logging.getLogger("epiwatch_backend")

app = FastAPI(
    title="EpiWatch Epidemiological Intelligence API",
    description="Programmatic REST API for disease surveillance, early outbreak detection, forecasting, and risk intelligence.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Starts background surveillance scheduler on server startup."""
    logger.info("Initializing EpiWatch background surveillance engine...")
    try:
        worker = get_surveillance_worker()
        worker.start()
    except Exception as e:
        logger.error(f"Failed to auto-start surveillance worker: {e}")

@app.on_event("shutdown")
def on_shutdown():
    """Stops background worker on server shutdown."""
    logger.info("Shutting down EpiWatch background surveillance worker...")
    try:
        worker = get_surveillance_worker()
        worker.stop()
    except Exception as e:
        logger.error(f"Error stopping worker on shutdown: {e}")

# Support both /api/... (and /api/v1/...) as well as root-level routing
app.include_router(router, prefix="/api")
app.include_router(router)

@app.get("/health")
def root_health():
    return {
        "status": "healthy",
        "service": "EpiWatch API Core",
        "mode": "DEMO_MODE",
        "provenance_banner": "DEMO MODE — Synthetic epidemiological data. Results are illustrative and are not live public-health surveillance."
    }
