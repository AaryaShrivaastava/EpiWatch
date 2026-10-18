"""Model registry for loading models and preprocessors."""
import joblib
import json
from pathlib import Path
from typing import Tuple, List, Any

def load_model_and_preprocessor(model_path: str, scaler_path: str, features_path: str) -> Tuple[Any, Any, List[str]]:
    """Load the model, scaler, and feature list."""
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler file missing: {scaler_path}")
    if not Path(features_path).exists():
        raise FileNotFoundError(f"Features file missing: {features_path}")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        meta = json.load(f)
        features = meta.get('feature_columns', [])
        
    return model, scaler, features
