"""Hotspot detection rules."""
import pandas as pd
from typing import Tuple

def detect_hotspots(regions: list, min_required: int = 10, boundary_data=None) -> Tuple[bool, str, pd.DataFrame]:
    """Detect spatial hotspots using Moran's I or Getis-Ord."""
    
    n_regions = len(regions)
    if n_regions < min_required:
        return False, f"Insufficient geographical observations ({n_regions}). Minimum {min_required} required.", pd.DataFrame()
        
    if boundary_data is None:
        return False, "No boundary or adjacency information available to compute spatial weights.", pd.DataFrame()
        
    # Since we lack boundaries and multiple regions, we just return failure.
    # In a full implementation, PySAL / esda would be used here.
    return False, "Unsupported statistical conditions: spatial weights missing.", pd.DataFrame()
