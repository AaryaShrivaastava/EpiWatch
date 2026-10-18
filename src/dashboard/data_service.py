"""Data service for dashboard."""
import pandas as pd
from pathlib import Path
import streamlit as st

@st.cache_data(ttl=300)
def load_csv(path_str: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def get_alerts() -> pd.DataFrame:
    return load_csv("data/alerts/alerts.csv")

def get_recommendations() -> pd.DataFrame:
    return load_csv("data/recommendations/recommendations.csv")

def get_risk_assessments() -> pd.DataFrame:
    return load_csv("data/risk_engine/regional_risk_assessments.csv")

def get_forecasts() -> pd.DataFrame:
    return load_csv("data/forecasts/disease_forecasts.csv")

def get_outbreaks() -> pd.DataFrame:
    return load_csv("data/analytics/outbreak_detection_results.csv")
