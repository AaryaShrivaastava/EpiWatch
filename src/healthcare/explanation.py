"""Explanation generator for healthcare."""
import pandas as pd

def generate_explanations(df: pd.DataFrame, region: str) -> str:
    """Generate textual deterministic explanations for the most recent observation."""
    if df.empty:
        return "No data available."
        
    latest = df.iloc[-1]
    
    score = latest.get('healthcare_stress_score', 'UNAVAILABLE')
    category = latest.get('stress_category', 'UNAVAILABLE')
    
    if pd.isna(score):
        score_str = "UNAVAILABLE"
    else:
        score_str = f"{score:.0f}/100"
        
    text = f"Region: {region}\n\nHealthcare Stress: {category} ({score_str})\n\nContributing signals:\n"
    
    available = str(latest.get('available_stress_indicators', ''))
    
    if 'bed_occupancy' in available:
        val = latest.get('bed_occupancy_percent', 0)
        text += f"- Bed occupancy is estimated at {val:.1f}%.\n"
        
    if 'icu_occupancy' in available:
        val = latest.get('icu_occupancy_percent', 0)
        text += f"- ICU occupancy is estimated at {val:.1f}%.\n"
        
    if 'oxygen_utilization' in available:
        val = latest.get('available_oxygen_percent', 0)
        text += f"- Available oxygen is at {val:.1f}%.\n"
        
    missing = str(latest.get('missing_stress_indicators', ''))
    if missing:
        text += "\nMissing information:\n"
        for m in missing.split(','):
            if m:
                text += f"- {m} is unavailable.\n"
                
    text += "\nAlerts:\n"
    
    if latest.get('bed_occupancy_ratio', 0) > 0.85:
        text += "- WATCH: High bed utilization (>85%).\n"
        
    if latest.get('icu_occupancy_ratio', 0) > 0.85:
        text += "- WATCH: High ICU utilization (>85%).\n"
        
    text += "\nLimitations:\n- Scores are based on provisional engineering thresholds.\n- Occupancy is estimated from admissions using length-of-stay assumptions.\n- This is a decision-support prototype, not a medical operations tool."
    
    return text
