"""Explanations for environmental risk score."""
import pandas as pd

def generate_explanations(df: pd.DataFrame, region: str, disease: str) -> str:
    """Generate deterministic textual explanation for the most recent observation."""
    if df.empty:
        return "No data available."
        
    latest = df.iloc[-1]
    
    score = latest.get('environmental_risk_score', 'UNAVAILABLE')
    if pd.isna(score):
        score_str = "UNAVAILABLE"
    else:
        score_str = f"{score:.0f}/100"
        
    text = f"Region: {region}\nDisease: {disease}\n\nEnvironmental Risk Indicator: {score_str}\n\nContributing signals:\n"
    
    available = str(latest.get('available_indicators', ''))
    
    if 'rainfall_anomaly' in available:
        val = latest.get('rainfall_mm_14d_sum', 0)
        text += f"- Rainfall is considered in the anomaly score (14d sum: {val:.1f}).\n"
        
    if 'humidity_anomaly' in available:
        val = latest.get('humidity_percent_14d_mean', 0)
        text += f"- Humidity is considered in the anomaly score (14d mean: {val:.1f}%).\n"
        
    if 'temperature_anomaly' in available:
        val = latest.get('temperature_c_14d_mean', 0)
        text += f"- Temperature is considered in the anomaly score (14d mean: {val:.1f}°C).\n"
        
    if not available:
        text += "- Insufficient indicators to compute signals.\n"
        
    text += "\nLimitations:\n- These associations do not establish causation.\n- The underlying dataset may include synthetic observations.\n- The score is not a clinically validated risk estimate."
    
    return text
