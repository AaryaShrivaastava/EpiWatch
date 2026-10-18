"""Pydantic schemas for Surveillance Engine."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SurveillanceConfigUpdate(BaseModel):
    scan_interval_seconds: Optional[int] = Field(None, ge=10, le=86400, description="Scan interval in seconds")
    monitored_regions: Optional[List[str]] = Field(None, description="List of regions to scan")
    monitored_diseases: Optional[List[str]] = Field(None, description="List of diseases to scan")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Threshold updates")

class FindingModel(BaseModel):
    finding_id: str
    timestamp: str
    region: str
    disease: str
    triggering_metric: str
    observed_value: float
    historical_baseline: Dict[str, Any]
    threshold: float
    detection_method: str
    severity: str
    supporting_evidence: Dict[str, Any]
    recommended_next_step: str
    related_alert_id: Optional[str] = None
    data_provenance: str
    status: str
    summary: str

class ScanRecordModel(BaseModel):
    scan_id: str
    started_at: str
    completed_at: str
    duration_seconds: float
    status: str
    source: str
    records_evaluated: int
    findings_generated: int
    alerts_triggered: int
    errors: List[str] = []

class SurveillanceStatusResponse(BaseModel):
    agent_status: str
    is_scanning: bool
    last_successful_scan: Optional[str]
    next_scheduled_scan: Optional[str]
    scan_interval_seconds: int
    total_scans_completed: int
    findings_detected: int
    active_alerts: int
    llm_status: str
    monitored_regions: List[str]
    monitored_diseases: List[str]
