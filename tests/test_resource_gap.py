"""Tests for resource gap analysis."""
import pandas as pd
from src.healthcare.resource_gap import calculate_resource_gaps

def test_calculate_resource_gaps():
    hc = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=2),
        'region': ['Bhopal', 'Bhopal'],
        'hospital_beds': [1000, 1000],
        'icu_beds': [100, 100]
    })
    
    demand = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=2),
        'region': ['Bhopal', 'Bhopal'],
        'expected_occupied_beds': [800, 1200],
        'expected_occupied_icu_beds': [50, 110]
    })
    
    res = calculate_resource_gaps(hc, demand)
    
    assert len(res) == 4 # 2 days * 2 resources (beds, icu)
    
    # Day 1 bed gap = 1000 - 800 = +200
    day1_beds = res[(res['date'] == pd.Timestamp('2024-01-01')) & (res['resource_type'] == 'hospital_beds')]
    assert day1_beds['capacity_gap'].iloc[0] == 200.0
    
    # Day 2 bed gap = 1000 - 1200 = -200 (shortfall)
    day2_beds = res[(res['date'] == pd.Timestamp('2024-01-02')) & (res['resource_type'] == 'hospital_beds')]
    assert day2_beds['capacity_gap'].iloc[0] == -200.0
