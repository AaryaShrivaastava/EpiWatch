"""Managed In-Process Background Surveillance Worker.

Architecture & Concurrency:
- Runs in a managed background daemon thread (`threading.Thread`).
- Controlled via `threading.Event` for clean, responsive stop/startup.
- Guarded by `threading.Lock` to guarantee protection against overlapping scans.
- Provides immediate manual execution via `trigger_scan(source='manual')`.
- Persists scan history, status, and scheduled next execution timestamp.

Multi-Instance Note:
- In single-node deployments (like standard local / container deployments), this in-process
  worker guarantees zero external infrastructure overhead (no Redis/Celery/Kafka needed).
- For horizontally scaled multi-worker clusters (e.g., behind a load balancer with multiple uvicorn workers),
  a distributed lock (such as Redis or database advisory lock) or dedicated singleton worker service
  should be used to avoid redundant simultaneous execution.
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from src.surveillance.config import get_surveillance_config
from src.surveillance.engine import get_surveillance_engine

logger = logging.getLogger("surveillance_worker")


class SurveillanceWorker:
    def __init__(self):
        self.config = get_surveillance_config()
        self.engine = get_surveillance_engine()
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._scan_lock = threading.Lock()
        
        self.is_scanning: bool = False
        self.agent_status: str = "STOPPED"
        self.last_successful_scan: Optional[str] = None
        self.next_scheduled_scan: Optional[str] = None
        self.total_scans_completed: int = 0
        self.consecutive_errors: int = 0

    def start(self) -> bool:
        """Starts the scheduled surveillance worker thread."""
        with self._scan_lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("Surveillance worker is already running.")
                self.agent_status = "RUNNING"
                return True

            logger.info("Starting surveillance worker thread...")
            self._stop_event.clear()
            self.agent_status = "RUNNING"
            self.consecutive_errors = 0
            
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="EpiWatchSurveillanceWorker",
                daemon=True
            )
            self._thread.start()
            self.engine.log_timeline_event("worker_started", "Surveillance Worker Started", "Background surveillance scheduler activated.", "success")
            return True

    def stop(self) -> bool:
        """Signals the surveillance worker to stop gracefully."""
        logger.info("Stopping surveillance worker thread...")
        self._stop_event.set()
        self.agent_status = "STOPPED"
        self.next_scheduled_scan = None
        self.engine.log_timeline_event("worker_stopped", "Surveillance Worker Stopped", "Background surveillance scheduler deactivated.", "warning")
        return True

    def _worker_loop(self) -> None:
        """Worker main loop running scheduled scans."""
        logger.info("Surveillance worker loop entered.")
        
        # Run an initial scan shortly after startup (after 5 seconds grace period)
        if not self._stop_event.wait(5.0):
            self.trigger_scan(source="scheduled_initial")

        while not self._stop_event.is_set():
            interval = int(self.config.get("surveillance.scan_interval_seconds", 900))
            next_time = datetime.now() + timedelta(seconds=interval)
            self.next_scheduled_scan = next_time.isoformat()
            
            logger.info(f"Next scheduled surveillance scan in {interval}s at {self.next_scheduled_scan}")
            
            # Sleep in 1-second chunks to respond immediately to stop_event
            elapsed = 0
            while elapsed < interval and not self._stop_event.is_set():
                time.sleep(1.0)
                elapsed += 1
                
            if not self._stop_event.is_set():
                self.trigger_scan(source="scheduled")

        self.agent_status = "STOPPED"
        self.next_scheduled_scan = None
        logger.info("Surveillance worker loop exited.")

    def trigger_scan(self, source: str = "manual") -> Dict[str, Any]:
        """Runs an immediate scan of the surveillance pipeline with concurrency lock."""
        if not self._scan_lock.acquire(blocking=False):
            logger.warning("Scan requested while a previous scan is still active. Suppressing duplicate overlap.")
            return {
                "scan_id": "REJECTED_OVERLAP",
                "status": "REJECTED",
                "message": "A surveillance scan is currently in progress. Overlapping execution prevented.",
                "duration_seconds": 0.0,
                "findings_generated": 0,
                "alerts_triggered": 0,
                "errors": ["Scan overlap rejected"]
            }

        try:
            self.is_scanning = True
            scan_result = self.engine.execute_scan(source=source)
            
            if scan_result.get("status") == "COMPLETED":
                self.last_successful_scan = datetime.now().isoformat()
                self.total_scans_completed += 1
                self.consecutive_errors = 0
                if self.agent_status != "RUNNING":
                    self.agent_status = "RUNNING"
            else:
                self.consecutive_errors += 1
                if self.consecutive_errors > 2:
                    self.agent_status = "DEGRADED"

            # Update next scheduled scan if running
            if not self._stop_event.is_set():
                interval = int(self.config.get("surveillance.scan_interval_seconds", 900))
                self.next_scheduled_scan = (datetime.now() + timedelta(seconds=interval)).isoformat()

            return scan_result
        except Exception as e:
            logger.exception("Error in trigger_scan")
            self.consecutive_errors += 1
            self.agent_status = "ERROR"
            return {
                "scan_id": "ERROR",
                "status": "FAILED",
                "message": str(e),
                "duration_seconds": 0.0,
                "findings_generated": 0,
                "alerts_triggered": 0,
                "errors": [str(e)]
            }
        finally:
            self.is_scanning = False
            self._scan_lock.release()

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status for the surveillance worker and agent."""
        # Check LLM status
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        llm_status = "GEMINI_ACTIVE" if gemini_key else "DETERMINISTIC_FALLBACK"

        findings = self.engine.get_findings()
        stats = self.engine.get_statistics()

        return {
            "agent_status": self.agent_status,
            "is_scanning": self.is_scanning,
            "last_successful_scan": self.last_successful_scan,
            "next_scheduled_scan": self.next_scheduled_scan,
            "scan_interval_seconds": int(self.config.get("surveillance.scan_interval_seconds", 900)),
            "total_scans_completed": self.total_scans_completed,
            "findings_detected": len(findings),
            "active_alerts": stats.get("active_alerts", 0),
            "llm_status": llm_status,
            "monitored_regions": self.config.get("surveillance.monitored_regions", ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain"]),
            "monitored_diseases": self.config.get("surveillance.monitored_diseases", ["COVID-19", "Dengue", "Malaria", "Influenza", "Typhoid"])
        }


_worker_instance: Optional[SurveillanceWorker] = None

def get_surveillance_worker() -> SurveillanceWorker:
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = SurveillanceWorker()
    return _worker_instance
