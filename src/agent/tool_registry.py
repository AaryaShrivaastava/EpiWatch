"""Tool Registry for LLM Agent.
Strictly approved and explicitly registered EpiWatch tools.
No shell access, no arbitrary code execution, no unrestricted filesystem access.
"""
from typing import Dict, Any, Callable, List
from src.agent.epiwatch_tools import (
    detect_outbreaks,
    get_recommendations,
    forecast_disease,
    get_regional_risk_assessment,
    get_surveillance_findings,
    get_healthcare_capacity,
    get_environmental_risk
)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []
        
    def register(self, name: str, func: Callable, schema: Dict[str, Any]):
        self._tools[name] = func
        self._schemas.append(schema)
        
    def get_tool(self, name: str) -> Callable:
        return self._tools.get(name)
        
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    
    registry.register(
        "detect_outbreaks",
        detect_outbreaks,
        {
            "name": "detect_outbreaks",
            "description": "Checks for epidemiological anomalies and outbreak signals in a given district/region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "The district or region name (e.g. Bhopal, Indore)."}
                },
                "required": ["region_id"]
            }
        }
    )
    
    registry.register(
        "get_recommendations",
        get_recommendations,
        {
            "name": "get_recommendations",
            "description": "Retrieves evidence-based public-health action directives and recommendations for a region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "The district or region name."}
                },
                "required": ["region_id"]
            }
        }
    )
    
    registry.register(
        "forecast_disease",
        forecast_disease,
        {
            "name": "forecast_disease",
            "description": "Retrieves 14-day disease trajectory forecast metrics and uncertainty bounds for a region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "The district or region name."}
                },
                "required": ["region_id"]
            }
        }
    )

    registry.register(
        "get_regional_risk_assessment",
        get_regional_risk_assessment,
        {
            "name": "get_regional_risk_assessment",
            "description": "Retrieves composite regional risk score (0-100), risk category, hospital stress, and preparedness for a district.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "The district or region name."}
                },
                "required": ["region_id"]
            }
        }
    )

    registry.register(
        "get_surveillance_findings",
        get_surveillance_findings,
        {
            "name": "get_surveillance_findings",
            "description": "Queries the active surveillance engine for recent anomaly findings, triggering metrics, and severity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "Optional filter for district name."},
                    "disease": {"type": "string", "description": "Optional filter for disease name."}
                }
            }
        }
    )

    registry.register(
        "get_healthcare_capacity",
        get_healthcare_capacity,
        {
            "name": "get_healthcare_capacity",
            "description": "Retrieves hospital bed stress index, ICU availability, and healthcare preparedness scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "The district or region name."}
                },
                "required": ["region_id"]
            }
        }
    )

    registry.register(
        "get_environmental_risk",
        get_environmental_risk,
        {
            "name": "get_environmental_risk",
            "description": "Retrieves meteorological indicators (temperature, humidity, rainfall, air quality) correlated with disease vector risk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "The district or region name."}
                },
                "required": ["region_id"]
            }
        }
    )
    
    return registry
