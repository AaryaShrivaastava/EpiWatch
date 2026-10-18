"""End-to-End Smoke Test."""
import sys
from pathlib import Path
import logging

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmokeTest")

def check_file(path: str) -> bool:
    exists = Path(path).exists()
    status = "OK" if exists else "MISSING"
    logger.info(f"[{status}] File check: {path}")
    return exists

def run_smoke_test():
    logger.info("Starting EpiWatch Smoke Test...")
    
    # 1. Check Data Dependencies
    required_files = [
        "data/alerts/alerts.csv",
        "data/risk_engine/regional_risk_assessments.csv",
        "data/recommendations/recommendations.csv",
        "data/forecasts/disease_forecasts.csv",
    ]
    
    missing = 0
    for f in required_files:
        if not check_file(f):
            missing += 1
            
    if missing > 0:
        logger.warning(f"{missing} required output files are missing. Ensure all analytical pipelines have run.")
    else:
        logger.info("All primary analytical outputs found.")
        
    # 2. Check imports (ensures environment is healthy)
    try:
        from src.dashboard.data_service import get_alerts
        logger.info("[OK] Dashboard data service import")
    except ImportError as e:
        logger.error(f"[FAIL] Failed to import dashboard service: {e}")
        sys.exit(1)
        
    try:
        from src.api.main import app
        logger.info("[OK] FastAPI application import")
    except ImportError as e:
        logger.error(f"[FAIL] Failed to import API module: {e}")
        sys.exit(1)
        
    # 3. Quick Data Load
    alerts = get_alerts()
    logger.info(f"Loaded {len(alerts)} alerts from persistence.")
    
    logger.info("Smoke test completed successfully.")
    
if __name__ == "__main__":
    run_smoke_test()
