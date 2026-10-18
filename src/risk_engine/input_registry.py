"""Data loading and validation."""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple

def load_risk_data(config_paths: Dict[str, str]) -> Dict[str, Optional[pd.DataFrame]]:
    """Load upstream datasets gracefully."""
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

def validate_risk_data(datasets: Dict[str, Optional[pd.DataFrame]], required: List[str] = None) -> Tuple[bool, List[str]]:
    """Check coverage of inputs."""
    issues = []
    if required is None:
        required = ['epi_path']
        
    for req in required:
        if datasets.get(req) is None or datasets[req].empty:
            issues.append(f"Required dataset {req} is missing or empty.")
            
    if datasets.get('epi_path') is not None:
        if 'region' not in datasets['epi_path'].columns:
            issues.append("Epidemiology dataset missing 'region' column.")
            
    return len(issues) == 0, issues
