"""OpenAI-powered LangGraph chat agent backed by MCP ML tools."""

from __future__ import annotations

import operator
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing_extensions import TypedDict

load_dotenv()

SYSTEM_PROMPT = """You are an assistant for a classical ML digit-classification project.
Use the available tools for any claim about training, accuracy, predictions, or model
status. Do not invent metrics. Use model_status for questions about current or latest
accuracy or F1. Train only when the user asks to train, fit, evaluate, or retrain.
Honor an explicit classifier request such as Random Forest or Logistic Regression;
otherwise use the automatic candidate selection mode.
If the user asks to maximize F1, use objective=f1_macro. Explain that the result is
the best score among the implemented candidate models, not a mathematical global
maximum. Explain tool results clearly and briefly. If the request is unrelated or
ambiguous, ask a concise clarifying question."""

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "train_mnist",
            "description": "Train and evaluate the classical digit classifier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_accuracy": {
                        "type": "number",
                        "description": "Minimum desired accuracy between 0 and 1.",
                    },
                    "objective": {
                        "type": "string",
                        "enum": ["accuracy", "f1_macro"],
                        "description": "Metric used to select the best candidate model.",
                    },
                    "target_score": {
                        "type": "number",
                        "description": "Optional target for the selected objective.",
                    },
                    "classifier": {
                        "type": "string",
                        "enum": ["auto", "logistic_regression", "random_forest"],
                        "description": "Classifier to train, or auto for model selection.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_digit",
            "description": "Predict a held-out digit by its sample index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_index": {
                        "type": "integer",
                        "description": "Held-out sample index from 0 through 359.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "model_status",
            "description": "Return the latest completed model accuracy, F1, and status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]
MCP_TOOL_NAMES = {tool["function"]["name"] for tool in MCP_TOOLS}
MCP_LOG_FILE = Path(__file__).with_name("logs") / "mnist_mcp.log"


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def _forward_mcp_logs(offset: int) -> int:
    """Forward new MCP log lines when child-process stderr is not attached."""
    if not MCP_LOG_FILE.exists():
        return offset
    with MCP_LOG_FILE.open("r", encoding="utf-8") as log_file:
        log_file.seek(offset)
        new_output = log_file.read()
        new_offset = log_file.tell()
    if new_output:
        sys.stderr.write(new_output)
        sys.stderr.flush()
    return new_offset


def _create_model():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for chat. Set it in the .env file."
        )
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    return init_chat_model(
        model_name,
        model_provider="openai",
        temperature=0,
    ).bind_tools(MCP_TOOLS)


@traceable(name="openai_chat_reasoning", tags=["chat", "openai"])
def _llm_node(model: Any, state: ChatState) -> dict[str, Any]:
    response = model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


@traceable(name="call_mnist_mcp_tool", tags=["chat", "mcp"])
async def _mcp_node(state: ChatState) -> dict[str, list[ToolMessage]]:
    """Execute every tool call requested by the OpenAI response."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise RuntimeError("The tool node requires an AI tool-call response.")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )
    tool_messages: list[ToolMessage] = []
    log_offset = (
        MCP_LOG_FILE.stat().st_size if MCP_LOG_FILE.exists() else 0
    )
    with open(os.devnull, "w") as mcp_errlog:
        async with stdio_client(server, errlog=mcp_errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    if tool_name not in MCP_TOOL_NAMES:
                        raise ValueError(f"Unknown MCP tool requested: {tool_name}")
                    result = await session.call_tool(
                        tool_name,
                        arguments=tool_call.get("args", {}),
                    )
                    log_offset = _forward_mcp_logs(log_offset)
                    if result.isError:
                        raise RuntimeError(f"MCP tool failed: {tool_name}")
                    text = next(
                        (
                            item.text
                            for item in result.content
                            if hasattr(item, "text")
                        ),
                        "",
                    )
                    if not text:
                        raise RuntimeError(f"MCP tool returned no result: {tool_name}")
                    tool_messages.append(
                        ToolMessage(content=text, tool_call_id=tool_call["id"])
                    )
    return {"messages": tool_messages}


def _next_step(state: ChatState) -> Literal["mcp", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "mcp"
    return END


def create_chat_agent(model: Any | None = None):
    """Build the OpenAI tool-calling graph."""
    # LangGraph passes a config dict when loading callable graph factories.
    selected_model = (
        _create_model() if model is None or isinstance(model, dict) else model
    )
    builder = StateGraph(ChatState)
    builder.add_node(
        "llm",
        lambda state: _llm_node(selected_model, state),
    )
    builder.add_node("mcp", _mcp_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", _next_step, ["mcp", END])
    builder.add_edge("mcp", "llm")
    return builder.compile()


async def run_chat(message: str, model: Any | None = None) -> str:
    """Run one user message through the OpenAI/MCP agent."""
    message = message.strip()
    if not message:
        raise ValueError("message must not be empty")
    result = await create_chat_agent(model).ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config={
            "run_name": "mnist-chat-agent",
            "tags": ["chat", "mcp", "openai", "classical-ml"],
        },
    )
    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("The chat agent did not return an assistant response.")
    return str(final_message.content)
