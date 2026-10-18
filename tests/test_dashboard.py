"""Tests for the Dashboard module."""
from src.dashboard.data_service import get_alerts, get_recommendations, get_risk_assessments, get_forecasts

def test_data_service_no_crash():
    # Calling these should not crash even if files are missing or empty
    alerts = get_alerts()
    assert alerts is not None
    
    recs = get_recommendations()
    assert recs is not None
    
    risk = get_risk_assessments()
    assert risk is not None
    
    forecasts = get_forecasts()
    assert forecasts is not None
