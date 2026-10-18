"""Tests for preparedness score."""
import pandas as pd
import numpy as np
from src.healthcare.preparedness_score import calculate_preparedness_score

def test_calculate_preparedness_score():
    df = pd.DataFrame({
        'beds_per_100k': [150], # half of 300
        'icu_beds_per_100k': [30], # full 30
        'medicine_availability_percent': [80]
    })
    
    weights = {
        'beds_per_100k': 30,
        'icu_beds_per_100k': 30,
        'medicine_availability': 20
    }
    
    benchmarks = {
        'beds_per_100k': 300,
        'icu_beds_per_100k': 30
    }
    
    res = calculate_preparedness_score(df, weights, benchmarks)
    
    # bed score = 50 * 30 = 1500
    # icu score = 100 * 30 = 3000
    # med score = 80 * 20 = 1600
    # total = 6100 / 80 = 76.25
    
    assert np.isclose(res['preparedness_score'].iloc[0], 76.25)
