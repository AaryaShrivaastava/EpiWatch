"""Rule Evaluator for Alerts."""
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

def evaluate_rules(datasets: Dict[str, pd.DataFrame], region: str, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluates datasets against configured rules to generate raw candidate alerts."""
    candidates = []
    timestamp = datetime.utcnow().isoformat()
    
    # 1. Outbreak Signal
    if rules.get('outbreak_signal', {}).get('enabled'):
        df = datasets.get('outbreak_path')
        if df is not None and not df.empty:
            df_reg = df[df['region'] == region] if 'region' in df.columns else df
            if not df_reg.empty:
                if 'date' in df_reg.columns: df_reg = df_reg.sort_values('date')
                latest = df_reg.iloc[-1]
                val = float(latest.get('anomaly_score', 0))
                th = rules['outbreak_signal'].get('threshold', 2.0)
                if val >= th:
                    candidates.append({
                        "rule_id": "outbreak_signal",
                        "category": rules['outbreak_signal']['category'],
                        "region": region,
                        "severity": rules['outbreak_signal']['severity'],
                        "triggering_metric": "anomaly_score",
                        "observed_value": val,
                        "threshold": th,
                        "explanation": f"Outbreak anomaly score {val:.2f} exceeded threshold {th}. This signal requires verification and does not confirm an outbreak.",
                        "timestamp": timestamp
                    })
                    
    # 2. Disease Forecast Growth
    if rules.get('disease_growth', {}).get('enabled'):
        df = datasets.get('forecast_path')
        if df is not None and not df.empty:
            df_reg = df[df['region'] == region] if 'region' in df.columns else df
            if not df_reg.empty:
                if 'date' in df_reg.columns: df_reg = df_reg.sort_values('date')
                latest = df_reg.iloc[-1]
                val = float(latest.get('forecast_growth_rate', 0))
                th = rules['disease_growth'].get('threshold', 0.15)
                if val >= th:
                    candidates.append({
                        "rule_id": "disease_growth",
                        "category": rules['disease_growth']['category'],
                        "region": region,
                        "severity": rules['disease_growth']['severity'],
                        "triggering_metric": "forecast_growth_rate",
                        "observed_value": val,
                        "threshold": th,
                        "explanation": f"Forecasted growth rate {val:.2f} exceeded threshold {th}. Forecasts are uncertain and not guaranteed outcomes.",
                        "timestamp": timestamp
                    })
                    
    # 3. Regional Risk
    if rules.get('regional_risk', {}).get('enabled'):
        df = datasets.get('risk_assessment_path')
        if df is not None and not df.empty:
            df_reg = df[df['region'] == region] if 'region' in df.columns else df
            if not df_reg.empty:
                latest = df_reg.iloc[-1]
                val = float(latest.get('integrated_regional_risk_score', 0))
                th = rules['regional_risk'].get('threshold', 75.0)
                if val >= th:
                    candidates.append({
                        "rule_id": "regional_risk",
                        "category": rules['regional_risk']['category'],
                        "region": region,
                        "severity": rules['regional_risk']['severity'],
                        "triggering_metric": "integrated_regional_risk_score",
                        "observed_value": val,
                        "threshold": th,
                        "explanation": f"Integrated regional risk score {val:.2f} crossed the threshold {th}.",
                        "timestamp": timestamp
                    })

    # 4. Healthcare Stress
    if rules.get('hc_stress', {}).get('enabled'):
        df = datasets.get('hc_stress_path')
        if df is not None and not df.empty:
            df_reg = df[df['region'] == region] if 'region' in df.columns else df
            if not df_reg.empty:
                if 'date' in df_reg.columns: df_reg = df_reg.sort_values('date')
                latest = df_reg.iloc[-1]
                val = float(latest.get('healthcare_stress_score', 0))
                th = rules['hc_stress'].get('threshold', 60.0)
                if val >= th:
                    candidates.append({
                        "rule_id": "hc_stress",
                        "category": rules['hc_stress']['category'],
                        "region": region,
                        "severity": rules['hc_stress']['severity'],
                        "triggering_metric": "healthcare_stress_score",
                        "observed_value": val,
                        "threshold": th,
                        "explanation": f"Healthcare stress score {val:.2f} crossed the critical threshold {th}. Review capacity metrics.",
                        "timestamp": timestamp
                    })

    return candidates
