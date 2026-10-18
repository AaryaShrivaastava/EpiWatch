"""Configuration manager for EpiWatch Phase 4 Forecasting."""

from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml

from src.utils.config import get_config as get_base_config


class ForecastingConfig:
    """Manages forecasting configuration from configs/forecasting_config.yaml."""

    def __init__(self, config_path: str = "configs/forecasting_config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            # Fallback to main config.yaml if specific config doesn't exist
            self.config_path = Path("config.yaml")
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

    @property
    def target_col(self) -> str:
        return self.get("forecasting.target", "new_cases")

    @property
    def frequency(self) -> str:
        return self.get("forecasting.frequency", "daily")

    @property
    def horizons(self) -> List[int]:
        freq = self.frequency
        return self.get(f"forecasting.forecast_horizons.{freq}", [7, 14, 28])


_forecasting_config_instance: Optional[ForecastingConfig] = None


def get_forecasting_config(config_path: str = "configs/forecasting_config.yaml") -> ForecastingConfig:
    global _forecasting_config_instance
    if _forecasting_config_instance is None or str(_forecasting_config_instance.config_path) != config_path:
        _forecasting_config_instance = ForecastingConfig(config_path)
    return _forecasting_config_instance
