"""Configuration loader and manager for EpiWatch platform."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Class to load and provide easy access to configuration parameters."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Get nested configuration value using dot notation (e.g. 'paths.raw_data_dir')."""
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def raw_data_dir(self) -> Path:
        return Path(self.get("paths.raw_data_dir", "data/raw"))

    @property
    def processed_data_dir(self) -> Path:
        return Path(self.get("paths.processed_data_dir", "data/processed"))

    @property
    def features_dir(self) -> Path:
        return Path(self.get("paths.features_dir", "data/features"))

    @property
    def reports_dir(self) -> Path:
        return Path(self.get("paths.reports_dir", "reports/data_quality"))

    @property
    def eda_dir(self) -> Path:
        return Path(self.get("paths.eda_dir", "reports/eda"))

    @property
    def figures_dir(self) -> Path:
        return Path(self.get("paths.figures_dir", "reports/figures"))

    @property
    def analytics_data_dir(self) -> Path:
        return Path(self.get("paths.analytics_data_dir", "data/analytics"))

    @property
    def models_dir(self) -> Path:
        return Path(self.get("paths.models_dir", "models"))

    @property
    def ml_reports_dir(self) -> Path:
        return Path(self.get("paths.ml_reports_dir", "reports/ml"))

    @property
    def outbreak_reports_dir(self) -> Path:
        return Path(self.get("outbreak_detection.reports_dir", "reports/outbreak_detection"))

    @property
    def outbreak_models_dir(self) -> Path:
        return Path(self.get("outbreak_detection.models_dir", "models/outbreak"))


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or initialize singleton Config instance."""
    global _config_instance
    if _config_instance is None or str(_config_instance.config_path) != config_path:
        _config_instance = Config(config_path)
    return _config_instance

