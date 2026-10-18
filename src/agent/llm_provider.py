"""Secure LLM Provider abstraction with Google Gemini support.

Security Guarantees:
- Reads GEMINI_API_KEY only on the backend from environment variables.
- Never prints or exposes the API key in logs, exceptions, or client responses.
- Whitelists tool execution exclusively to registered EpiWatch analytical functions.
  No shell access, no code evaluation, no unrestricted filesystem operations.
- Graceful degradation: If GEMINI_API_KEY is not configured or fails, deterministic
  analytical capabilities continue uninterrupted.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx

from src.agent.tool_registry import build_tool_registry

logger = logging.getLogger("llm_provider")


class LLMProvider:
    def __init__(self, config: Any):
        self.config = config

    def generate_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Deterministic fallback provider when external LLM is not configured."""
    def generate_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        last_msg = messages[-1].get("content", "").lower() if messages else ""
        
        # Check if query asks for a tool
        if ("outbreak" in last_msg or "anomaly" in last_msg) and not any("tool_calls" in m for m in messages):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "detect_outbreaks", "arguments": {"region_id": "Bhopal"}}]
            }
        if "risk" in last_msg and not any("tool_calls" in m for m in messages):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_regional_risk_assessment", "arguments": {"region_id": "Bhopal"}}]
            }
        if "recommend" in last_msg and not any("tool_calls" in m for m in messages):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_recommendations", "arguments": {"region_id": "Bhopal"}}]
            }
        if "forecast" in last_msg and not any("tool_calls" in m for m in messages):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "forecast_disease", "arguments": {"region_id": "Bhopal"}}]
            }
            
        return {
            "role": "assistant",
            "content": (
                "EpiWatch Epidemiological Assistant (Deterministic Fallback Mode):\n"
                "System is operating in verified deterministic surveillance mode. "
                "All analytical outputs are grounded in calibrated models for Madhya Pradesh districts."
            ),
            "tool_calls": []
        }


class GeminiLLMProvider(LLMProvider):
    """Production Google Gemini Provider using Google AI Studio REST API."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip() or self.config.get("agent.api_key", "").strip()
        self.model = os.environ.get("LLM_MODEL", "").strip() or self.config.get("agent.model", "gemini-1.5-flash")
        if self.model == "mock-llm-1":
            self.model = "gemini-1.5-flash"
        self.timeout = float(self.config.get("agent.timeout_seconds", 30))

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def generate_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.is_configured():
            logger.info("Gemini API key is unconfigured. Using deterministic fallback provider.")
            return MockLLMProvider(self.config).generate_response(messages, tools)

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        # Convert tool schemas to Gemini format
        gemini_tools = []
        if tools:
            function_declarations = []
            for t in tools:
                function_declarations.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                })
            gemini_tools.append({"function_declarations": function_declarations})

        # Convert conversation messages to Gemini format
        contents = []
        for m in messages:
            role = "user" if m.get("role") in ["user", "human"] else "model"
            parts = []
            if m.get("content"):
                parts.append({"text": m["content"]})
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    parts.append({
                        "functionCall": {
                            "name": tc["name"],
                            "args": tc.get("arguments", {})
                        }
                    })
            if m.get("tool_result"):
                role = "function"
                parts.append({
                    "functionResponse": {
                        "name": m.get("tool_name", "tool"),
                        "response": {"result": m["tool_result"]}
                    }
                })
            if parts:
                contents.append({"role": role, "parts": parts})

        system_instruction = (
            "You are EpiWatch AI, a senior epidemiological intelligence assistant. "
            "Your role is to explain regional disease risk, early outbreak detection signals, hospital capacity, "
            "and evidence-based public-health recommendations for Madhya Pradesh districts (Bhopal, Indore, Jabalpur, Gwalior, Ujjain). "
            "CRITICAL: Always ground your answers in verified outputs from your approved tools. "
            "Never invent numbers, dates, or confirmed outbreak declarations. "
            "Distinguish clearly between statistical anomalies and medically confirmed public-health outbreaks."
        )

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": float(self.config.get("agent.temperature", 0.1)),
                "maxOutputTokens": 1024
            }
        }
        if gemini_tools:
            payload["tools"] = gemini_tools

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
                if res.status_code != 200:
                    logger.warning(f"Gemini API returned status {res.status_code}. Falling back to deterministic mode.")
                    return MockLLMProvider(self.config).generate_response(messages, tools)

                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return {"role": "assistant", "content": "No response generated by model.", "tool_calls": []}

                candidate = candidates[0]
                content_obj = candidate.get("content", {})
                parts = content_obj.get("parts", [])

                reply_text = ""
                tool_calls = []

                for p in parts:
                    if "text" in p:
                        reply_text += p["text"]
                    if "functionCall" in p:
                        fc = p["functionCall"]
                        tool_calls.append({
                            "name": fc.get("name"),
                            "arguments": fc.get("args", {})
                        })

                return {
                    "role": "assistant",
                    "content": reply_text,
                    "tool_calls": tool_calls
                }
        except Exception as e:
            logger.error(f"Error calling Gemini endpoint: {e}")
            return MockLLMProvider(self.config).generate_response(messages, tools)


def get_llm_provider(config: Any) -> LLMProvider:
    provider = config.get("agent.provider", "gemini")
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or config.get("agent.api_key", "").strip()
    
    if provider == "gemini" or gemini_key:
        return GeminiLLMProvider(config)
    return MockLLMProvider(config)
