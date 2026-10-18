"""Tool definitions and implementations."""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.agent.config import get_agent_config

def _load_csv_if_exists(path_key: str) -> pd.DataFrame:
    config = get_agent_config()
    p = Path(config.get(path_key, ""))
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

def detect_outbreaks(region_id: str) -> Dict[str, Any]:
    """Retrieves outbreak detection results for a region."""
    df = _load_csv_if_exists("agent.data.outbreak_path")
    if df.empty:
        return {"error": "Outbreak detection data is unavailable."}
        
    if 'region' in df.columns:
        df = df[df['region'] == region_id]
        
    if df.empty:
        return {"status": "no data", "message": f"No outbreak data for region {region_id}"}
        
    # Get latest
    if 'date' in df.columns:
        df = df.sort_values('date')
    latest = df.iloc[-1].to_dict()
    
    return {
        "status": "success",
        "region_id": region_id,
        "anomaly_score": latest.get('anomaly_score'),
        "date": latest.get('date'),
        "limitation": "Anomaly scores are model outputs and not confirmed outbreaks."
    }

def get_recommendations(region_id: str) -> Dict[str, Any]:
    """Retrieves public-health recommendations for a region."""
    df = _load_csv_if_exists("agent.data.recommendations_path")
    if df.empty:
        return {"error": "Recommendations data is unavailable."}
        
    if 'region' in df.columns:
        df = df[df['region'] == region_id]
        
    if df.empty:
        return {"status": "no data", "message": f"No recommendations for region {region_id}"}
        
    recs = df.to_dict('records')
    return {
        "status": "success",
        "region_id": region_id,
        "recommendations": recs,
        "limitation": "Recommendations require review by qualified professionals."
    }

def forecast_disease(region_id: str) -> Dict[str, Any]:
    """Retrieves disease forecast."""
    df = _load_csv_if_exists("agent.data.forecast_path")
    if df.empty:
        p = Path("data/forecasts/disease_forecasts.csv")
        df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if df.empty:
        return {"error": "Forecast data is unavailable."}
        
    if 'region' in df.columns:
        df = df[df['region'].str.lower() == region_id.lower()]
        
    if df.empty:
        return {"status": "no data", "message": f"No forecasts for region {region_id}"}
        
    if 'date' in df.columns:
        df = df.sort_values('date')
        
    latest = df.iloc[-1].to_dict()
    return {
        "status": "success",
        "region_id": region_id,
        "predicted_cases": latest.get('predicted_cases'),
        "lower_bound": latest.get('lower_bound'),
        "upper_bound": latest.get('upper_bound'),
        "horizon": latest.get('horizon', 14),
        "limitation": "Forecasts are predictions with statistical confidence bounds."
    }

def get_regional_risk_assessment(region_id: str) -> Dict[str, Any]:
    """Retrieves regional composite risk assessment for a district."""
    p = Path("data/risk_engine/regional_risk_assessments.csv")
    if not p.exists():
        return {"error": "Regional risk assessment data is unavailable."}
    df = pd.read_csv(p)
    sub = df[df["region"].str.lower() == region_id.lower()]
    if sub.empty:
        return {"status": "no data", "message": f"No risk assessment found for region {region_id}"}
    latest = sub.iloc[-1].to_dict()
    return {
        "status": "success",
        "region_id": region_id,
        "risk_score": latest.get("integrated_regional_risk_score"),
        "risk_category": latest.get("integrated_risk_category"),
        "primary_driver": latest.get("primary_driver"),
        "healthcare_stress_score": latest.get("healthcare_stress_score"),
        "preparedness_score": latest.get("healthcare_preparedness_score"),
        "limitation": "Risk scores are synthetic decision-support models for demonstration."
    }

def get_surveillance_findings(region_id: Optional[str] = None, disease: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves active epidemiological surveillance findings and anomalies."""
    from src.surveillance.engine import get_surveillance_engine
    engine = get_surveillance_engine()
    findings = engine.get_findings(region=region_id, disease=disease)
    return {
        "status": "success",
        "findings_count": len(findings),
        "findings": findings[:5],
        "limitation": "Findings reflect statistical anomalies and require human epidemiological verification."
    }

def get_healthcare_capacity(region_id: str) -> Dict[str, Any]:
    """Retrieves healthcare capacity, bed stress, and preparedness metrics."""
    p = Path("data/healthcare/analytics/healthcare_stress.csv")
    if not p.exists():
        p = Path("data/healthcare/healthcare_stress.csv")
    if not p.exists():
        return {"status": "no data", "message": "Healthcare stress data is unavailable."}
    df = pd.read_csv(p)
    sub = df[df["region"].str.lower() == region_id.lower()]
    if sub.empty:
        return {"status": "no data", "message": f"No healthcare capacity data for {region_id}"}
    latest = sub.iloc[-1].to_dict()
    return {
        "status": "success",
        "region_id": region_id,
        "healthcare_stress": latest.get("healthcare_stress_score", 1.0),
        "preparedness_score": latest.get("preparedness_score", 85.0),
        "limitation": "Healthcare metrics represent calibrated capacity approximations."
    }

def get_environmental_risk(region_id: str) -> Dict[str, Any]:
    """Retrieves environmental risk factors (temperature, humidity, rainfall)."""
    p = Path("data/processed/environmental_clean.csv")
    if not p.exists():
        return {"status": "no data", "message": "Environmental data is unavailable."}
    df = pd.read_csv(p)
    if "region" in df.columns:
        sub = df[df["region"].str.lower() == region_id.lower()]
        if not sub.empty:
            df = sub
    latest = df.iloc[-1].to_dict() if not df.empty else {}
    return {
        "status": "success",
        "region_id": region_id,
        "temperature_c": latest.get("temperature_c"),
        "humidity_percent": latest.get("humidity_percent"),
        "rainfall_mm": latest.get("rainfall_mm"),
        "air_quality_index": latest.get("air_quality_index"),
        "limitation": "Environmental data indicates meteorological correlation, not causality."
    }
