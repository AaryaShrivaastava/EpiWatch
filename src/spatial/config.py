"""Configuration for Spatial module."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class SpatialConfig:
    def __init__(self, config_path: str = "configs/spatial_config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / "configs" / "spatial_config.yaml"
            if alt_path.exists():
                with open(alt_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

_spatial_config_instance: Optional[SpatialConfig] = None

def get_spatial_config() -> SpatialConfig:
    global _spatial_config_instance
    if _spatial_config_instance is None:
        _spatial_config_instance = SpatialConfig()
    return _spatial_config_instance
