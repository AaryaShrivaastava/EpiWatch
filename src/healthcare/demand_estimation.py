"""Demand estimation for future forecasts."""
import pandas as pd
import numpy as np
from typing import Optional

def estimate_future_demand(
    forecast_df: Optional[pd.DataFrame], 
    hosp_rate: float, 
    icu_rate: float,
    los_days: int = 7,
    icu_los_days: int = 10
) -> Optional[pd.DataFrame]:
    """Estimate future hospitalizations and ICU admissions from forecasted cases."""
    
    if forecast_df is None or forecast_df.empty:
        return None
        
    df = forecast_df.copy().sort_values('date')
    
    if 'predicted_cases' not in df.columns:
        return None
        
    # Projected daily admissions
    df['projected_hospital_admissions'] = df['predicted_cases'] * hosp_rate
    df['projected_icu_admissions'] = df['predicted_cases'] * icu_rate
    
    # Projected point-in-time occupancy using length of stay (rolling sum)
    df['expected_occupied_beds'] = df['projected_hospital_admissions'].rolling(window=los_days, min_periods=1).sum()
    df['expected_occupied_icu_beds'] = df['projected_icu_admissions'].rolling(window=icu_los_days, min_periods=1).sum()
    
    return df
