"""Data loading for recommendations."""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

def load_recommendation_data(config_paths: Dict[str, str]) -> Dict[str, Optional[pd.DataFrame]]:
    """Load upstream datasets for recommendations."""
    datasets = {}
    for key, path_str in config_paths.items():
        if not path_str:
            datasets[key] = None
            continue
            
        p = Path(path_str)
        if p.exists() and p.suffix == '.csv':
            try:
                datasets[key] = pd.read_csv(p)
                if 'date' in datasets[key].columns:
                    datasets[key]['date'] = pd.to_datetime(datasets[key]['date'])
            except Exception:
                datasets[key] = None
        else:
            datasets[key] = None
            
    return datasets
