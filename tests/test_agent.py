"""Tests for the LLM Agent Module."""
import pytest
from src.agent.tool_registry import build_tool_registry
from src.agent.tool_executor import ToolExecutor
from src.agent.agent_pipeline import AgentPipeline

def test_tool_registry():
    registry = build_tool_registry()
    tools = registry.get_all_schemas()
    assert len(tools) == 3
    assert any(t['name'] == 'detect_outbreaks' for t in tools)
    
def test_tool_executor_invalid():
    registry = build_tool_registry()
    executor = ToolExecutor(registry)
    
    res = executor.execute("non_existent_tool", {})
    assert "error" in res
    
def test_agent_pipeline():
    agent = AgentPipeline()
    response = agent.process_request("Please generate recommendations for Bhopal")
    assert "complete" in response.lower()
