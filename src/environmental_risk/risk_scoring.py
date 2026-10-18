"""Environmental Risk Scoring."""
import pandas as pd
import numpy as np

def generate_risk_score(
    df: pd.DataFrame,
    weights: dict,
    thresholds: dict
) -> pd.DataFrame:
    """
    Generate interpretable risk score (0-100).
    Normalizes indicators using historical z-scores or percentiles, then combines.
    """
    df = df.copy()
    
    # Indicators: we'll define simple anomalies:
    # 1. rainfall_anomaly = (current_rainfall_sum - historical_mean_rainfall)
    # Actually, let's just use rolling 28d vs overall mean as an example.
    # To keep it generic, we expect the pipeline to have passed basic features.
    
    score = np.zeros(len(df))
    total_weight = 0
    available_indicators = []
    missing_indicators = []
    
    # Helper for 0-100 scale
    def scale_0_100(series):
        if series.std() == 0:
            return pd.Series(50, index=series.index)
        # z-score to roughly 0-100 (cap at -3, +3)
        z = (series - series.mean()) / series.std()
        z = z.clip(-3, 3)
        return (z + 3) / 6 * 100
        
    if 'rainfall_anomaly' in weights:
        if 'rainfall_mm' in df.columns and 'rainfall_mm_14d_sum' in df.columns:
            val = scale_0_100(df['rainfall_mm_14d_sum'])
            score += val * weights['rainfall_anomaly']
            total_weight += weights['rainfall_anomaly']
            available_indicators.append('rainfall_anomaly')
        else:
            missing_indicators.append('rainfall_anomaly')
            
    if 'humidity_anomaly' in weights:
        if 'humidity_percent' in df.columns and 'humidity_percent_14d_mean' in df.columns:
            val = scale_0_100(df['humidity_percent_14d_mean'])
            score += val * weights['humidity_anomaly']
            total_weight += weights['humidity_anomaly']
            available_indicators.append('humidity_anomaly')
        else:
            missing_indicators.append('humidity_anomaly')
            
    if 'temperature_anomaly' in weights:
        if 'temperature_c' in df.columns and 'temperature_c_14d_mean' in df.columns:
            val = scale_0_100(df['temperature_c_14d_mean'])
            score += val * weights['temperature_anomaly']
            total_weight += weights['temperature_anomaly']
            available_indicators.append('temperature_anomaly')
        else:
            missing_indicators.append('temperature_anomaly')
            
    if total_weight > 0:
        final_score = score / total_weight
    else:
        final_score = np.nan
        
    df['environmental_risk_score'] = final_score
    
    # Categorize
    def categorize(x):
        if pd.isna(x):
            return "UNAVAILABLE"
        if x < thresholds.get('moderate', 40): return "LOW"
        if x < thresholds.get('high', 60): return "MODERATE"
        if x < thresholds.get('very_high', 80): return "HIGH"
        return "VERY_HIGH"
        
    df['risk_category'] = df['environmental_risk_score'].apply(categorize)
    df['available_indicators'] = ",".join(available_indicators)
    df['missing_indicators'] = ",".join(missing_indicators)
    df['score_method'] = "z_score_weighted"
    
    return df
