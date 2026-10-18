"""Config loader for Agent."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

class AgentConfig:
    def __init__(self, config_path: str = "configs/agent_config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / "configs" / "agent_config.yaml"
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
        
    def get_api_key(self) -> Optional[str]:
        # Check env first
        key = os.environ.get("LLM_API_KEY")
        if key: return key
        return self.get("agent.api_key")

_agent_config_instance: Optional[AgentConfig] = None

def get_agent_config() -> AgentConfig:
    global _agent_config_instance
    if _agent_config_instance is None:
        _agent_config_instance = AgentConfig()
    return _agent_config_instance
