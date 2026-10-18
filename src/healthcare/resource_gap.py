"""Resource gap analysis."""
import pandas as pd

def calculate_resource_gaps(hc_df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
    """Compare estimated demand against point-in-time capacity."""
    if demand_df is None or demand_df.empty:
        return pd.DataFrame()
        
    # Demand is daily. We'll join against daily or forward-filled monthly HC data.
    hc = hc_df.copy().set_index('date').resample('D').ffill()
    demand = demand_df.copy().set_index('date')
    
    merged = demand.join(hc, lsuffix='_demand', rsuffix='_hc', how='inner').reset_index()
    
    results = []
    
    for _, row in merged.iterrows():
        date = row['date']
        region = row['region_demand'] if 'region_demand' in row else row.get('region', 'Unknown')
        
        # Bed Gap
        if 'hospital_beds' in row and 'expected_occupied_beds' in row and not pd.isna(row['expected_occupied_beds']):
            capacity = row['hospital_beds']
            demand_val = row['expected_occupied_beds']
            gap = capacity - demand_val
            ratio = demand_val / capacity if capacity > 0 else float('inf')
            
            results.append({
                'date': date,
                'region': region,
                'resource_type': 'hospital_beds',
                'available_capacity': capacity,
                'estimated_demand': demand_val,
                'capacity_gap': gap,
                'utilization_ratio': ratio,
                'scenario': 'forecast',
                'data_provenance': 'DERIVED'
            })
            
        # ICU Gap
        if 'icu_beds' in row and 'expected_occupied_icu_beds' in row and not pd.isna(row['expected_occupied_icu_beds']):
            capacity = row['icu_beds']
            demand_val = row['expected_occupied_icu_beds']
            gap = capacity - demand_val
            ratio = demand_val / capacity if capacity > 0 else float('inf')
            
            results.append({
                'date': date,
                'region': region,
                'resource_type': 'icu_beds',
                'available_capacity': capacity,
                'estimated_demand': demand_val,
                'capacity_gap': gap,
                'utilization_ratio': ratio,
                'scenario': 'forecast',
                'data_provenance': 'DERIVED'
            })
            
    return pd.DataFrame(results)
