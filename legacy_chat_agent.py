"""Legacy deterministic chat router for comparison with the OpenAI agent.

This module is intentionally not imported by ``web_app.py`` or
``chat_agent.py``. It is a teaching alternative that uses keyword matching,
fixed MCP arguments, and hard-coded response templates.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

Intent = Literal["train", "predict", "status", "help"]


def route_message(message: str) -> tuple[Intent, str, dict[str, Any]]:
    """Route a message without an LLM."""
    text = message.lower()
    if "current accuracy" in text or "latest accuracy" in text or "status" in text:
        return "status", "model_status", {}
    if any(word in text for word in ("train", "fit", "evaluate", "retrain")):
        match = re.search(r"(?:accuracy|threshold|target)\D*(0?\.\d+|1(?:\.0+)?)", text)
        target = float(match.group(1)) if match else 0.95
        return "train", "train_mnist", {"target_accuracy": target}
    if any(word in text for word in ("predict", "classify", "recognize")):
        match = re.search(r"(?:sample|index)\D*(\d+)", text)
        index = int(match.group(1)) if match else 0
        return "predict", "predict_digit", {"sample_index": index}
    return "help", "", {}


async def _call_mcp(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
    if result.isError:
        raise RuntimeError(f"MCP tool failed: {tool_name}")
    text = next((item.text for item in result.content if hasattr(item, "text")), "")
    if not text:
        raise RuntimeError(f"MCP tool returned no result: {tool_name}")
    return json.loads(text)


async def run_legacy_chat(message: str) -> str:
    """Run the deterministic alternative; production chat does not call this."""
    intent, tool_name, arguments = route_message(message.strip())
    if intent == "help":
        return (
            "Try 'train with accuracy 0.95', 'predict sample 12', "
            "or 'what is the current accuracy?'"
        )
    result = await _call_mcp(tool_name, arguments)
    if intent == "status":
        if result["status"] == "not_trained":
            return "No model is trained yet. Ask me to train it."
        return (
            f"Model {result['model']} is ready with "
            f"{result['accuracy']:.1%} accuracy."
        )
    response = (
        f"Used {result['model']} ({result['attempts']} attempt(s)) with "
        f"{result['accuracy']:.1%} accuracy."
    )
    if intent == "predict":
        response += (
            f" Sample {result['sample_index']} was classified as digit "
            f"{result['prediction']}."
        )
    return response
