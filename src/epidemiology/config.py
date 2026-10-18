"""Epidemiological Modeling Configuration."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import os

class EpiConfig:
    def __init__(self, config_path: str = "configs/epidemiology_config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            # Try from root if current dir is different
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / "configs" / "epidemiology_config.yaml"
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

_epi_config_instance: Optional[EpiConfig] = None

def get_epi_config() -> EpiConfig:
    global _epi_config_instance
    if _epi_config_instance is None:
        _epi_config_instance = EpiConfig()
    return _epi_config_instance
