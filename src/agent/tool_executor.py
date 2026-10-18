"""Tool Executor."""
import logging
from typing import Dict, Any
from src.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Executing tool {tool_name} with args {arguments}")
        func = self.registry.get_tool(tool_name)
        if not func:
            return {"error": f"Tool '{tool_name}' not found in registry."}
            
        try:
            result = func(**arguments)
            return result
        except TypeError as e:
            logger.error(f"Invalid arguments for {tool_name}: {e}")
            return {"error": "Invalid arguments provided.", "details": str(e)}
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}")
            return {"error": "Tool execution failed.", "details": str(e)}
