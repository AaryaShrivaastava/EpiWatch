"""Data Loading for Spatial Module."""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

def load_spatial_data(config_paths: Dict[str, str]) -> Dict[str, Optional[pd.DataFrame]]:
    """Load spatial datasets gracefully."""
    datasets = {}
    
    for key, path_str in config_paths.items():
        if not path_str:
            datasets[key] = None
            continue
            
        p = Path(path_str)
        if p.exists():
            if p.suffix == '.csv':
                try:
                    datasets[key] = pd.read_csv(p)
                except Exception:
                    datasets[key] = None
            elif p.suffix in ['.json', '.geojson']:
                try:
                    import json
                    with open(p, 'r', encoding='utf-8') as f:
                        datasets[key] = json.load(f)
                except Exception:
                    datasets[key] = None
            else:
                datasets[key] = None
        else:
            datasets[key] = None
            
    return datasets
