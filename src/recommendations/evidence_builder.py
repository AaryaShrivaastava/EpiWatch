"""Evidence extraction for recommendations."""
import pandas as pd
from typing import Dict, Any

def extract_evidence(datasets: Dict[str, pd.DataFrame], region: str) -> Dict[str, Any]:
    """Collect relevant latest evidence for a region."""
    evidence = {}
    
    # Outbreak
    df_outbreak = datasets.get('outbreak_path')
    if df_outbreak is not None and not df_outbreak.empty:
        if 'region' in df_outbreak.columns:
            df = df_outbreak[df_outbreak['region'] == region]
            if not df.empty:
                if 'date' in df.columns:
                    df = df.sort_values('date')
                latest = df.iloc[-1]
                evidence['outbreak_anomaly_score'] = float(latest.get('anomaly_score', 0.0))
                
    # Forecast
    df_forecast = datasets.get('forecast_path')
    if df_forecast is not None and not df_forecast.empty:
        df = df_forecast[df_forecast['region'] == region] if 'region' in df_forecast.columns else df_forecast
        if not df.empty:
            if 'date' in df.columns:
                df = df.sort_values('date')
            latest = df.iloc[-1]
            evidence['forecast_growth_rate'] = float(latest.get('forecast_growth_rate', 0.0))
            
    # Env
    df_env = datasets.get('env_path')
    if df_env is not None and not df_env.empty:
        df = df_env[df_env['region'] == region]
        if not df.empty:
            if 'date' in df.columns:
                df = df.sort_values('date')
            latest = df.iloc[-1]
            evidence['environmental_risk_score'] = float(latest.get('environmental_risk_score', 0.0))
            
    # HC Stress
    df_stress = datasets.get('hc_stress_path')
    if df_stress is not None and not df_stress.empty:
        df = df_stress[df_stress['region'] == region]
        if not df.empty:
            if 'date' in df.columns:
                df = df.sort_values('date')
            latest = df.iloc[-1]
            evidence['healthcare_stress_score'] = float(latest.get('healthcare_stress_score', 0.0))
            
    # HC Prep
    df_prep = datasets.get('hc_prep_path')
    if df_prep is not None and not df_prep.empty:
        df = df_prep[df_prep['region'] == region]
        if not df.empty:
            latest = df.iloc[-1]
            evidence['healthcare_preparedness_score'] = float(latest.get('preparedness_score', 0.0))
            
    # Gaps
    df_gap = datasets.get('resource_gap_path')
    if df_gap is not None and not df_gap.empty:
        df = df_gap[df_gap['region'] == region]
        if not df.empty:
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Find the worst gap among recent records
            beds = df[df['resource_type'] == 'hospital_beds']
            if not beds.empty:
                min_gap = beds['capacity_gap'].min()
                evidence['hospital_bed_gap'] = float(min_gap)
                
    return evidence
