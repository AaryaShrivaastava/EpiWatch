"""Surveillance configuration loader and persistence."""
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

class SurveillanceConfig:
    def __init__(self, config_path: str = "configs/surveillance_config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / "configs" / "surveillance_config.yaml"
            if alt_path.exists():
                with open(alt_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key_path: str, value: Any) -> None:
        if not key_path.startswith("surveillance.") and "surveillance" in self._config:
            key_path = f"surveillance.{key_path}"
        keys = key_path.split(".")
        d = self._config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        self.save()

    def update_dict(self, updates: Dict[str, Any]) -> None:
        for k, v in updates.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    self.set(f"{k}.{sub_k}", sub_v)
            else:
                self.set(k, v)

    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)

    def as_dict(self) -> Dict[str, Any]:
        return self._config.get("surveillance", {})

_instance: Optional[SurveillanceConfig] = None

def get_surveillance_config() -> SurveillanceConfig:
    global _instance
    if _instance is None:
        _instance = SurveillanceConfig()
    return _instance
