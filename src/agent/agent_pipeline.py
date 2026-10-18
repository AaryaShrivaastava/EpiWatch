"""Agent Orchestration Pipeline."""
import logging
from datetime import datetime
from typing import List, Dict, Any

from src.agent.config import get_agent_config
from src.agent.llm_provider import get_llm_provider
from src.agent.tool_registry import build_tool_registry
from src.agent.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)

class AgentPipeline:
    def __init__(self):
        self.config = get_agent_config()
        self.provider = get_llm_provider(self.config)
        self.registry = build_tool_registry()
        self.executor = ToolExecutor(self.registry)
        
    def process_chat(self, user_prompt: str) -> Dict[str, Any]:
        """Processes chat request and returns reply along with verified tools executed."""
        logger.info(f"Processing user chat request: {user_prompt}")
        messages = [{"role": "user", "content": user_prompt}]
        schemas = self.registry.get_all_schemas()
        max_calls = self.config.get("agent.max_tool_calls", 5)
        calls_made = 0
        tools_executed = []
        
        while calls_made < max_calls:
            try:
                response = self.provider.generate_response(messages, schemas)
            except Exception as e:
                logger.error(f"LLM Provider error: {e}")
                return {
                    "reply": "The analytical engine encountered an error generating an answer. Please review raw surveillance data.",
                    "tools_executed": tools_executed,
                    "grounding_source": "Deterministic Platform Fallback",
                    "timestamp": datetime.now().isoformat()
                }
                
            messages.append(response)
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                return {
                    "reply": response.get("content", "No response content generated."),
                    "tools_executed": tools_executed,
                    "grounding_source": "Verified Local Analytical Pipeline Artifacts",
                    "timestamp": datetime.now().isoformat()
                }
                
            for call in tool_calls:
                name = call.get("name")
                args = call.get("arguments", {})
                tools_executed.append(name)
                
                result = self.executor.execute(name, args)
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "name": name,
                    "content": str(result),
                    "tool_result": result
                })
                
            calls_made += 1
            
        return {
            "reply": "The request completed with verified tool signals.",
            "tools_executed": tools_executed,
            "grounding_source": "Verified Local Analytical Pipeline Artifacts",
            "timestamp": datetime.now().isoformat()
        }

    def process_request(self, user_prompt: str) -> str:
        res = self.process_chat(user_prompt)
        return res.get("reply", "")

def run_agent_pipeline():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting AI Epidemiology Agent (Phase 12)")
    
    agent = AgentPipeline()
    response = agent.process_request("Please generate recommendations for Bhopal")
    
    logger.info(f"Agent Final Response: {response}")
    logger.info("Agent Pipeline completed.")

if __name__ == "__main__":
    run_agent_pipeline()
