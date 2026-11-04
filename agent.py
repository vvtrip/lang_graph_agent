"""LangGraph arithmetic agent."""

import operator
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tools import TOOLS, TOOLS_BY_NAME

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with performing arithmetic "
    "on a set of inputs."
)


class MessagesState(TypedDict):
    """State accumulated while the agent runs."""

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def create_agent():
    """Build and compile the arithmetic agent graph."""
    model = init_chat_model("gpt-5-nano", temperature=0)
    model_with_tools = model.bind_tools(TOOLS)

    def llm_call(state: MessagesState) -> dict[str, object]:
        """Ask the model whether a tool is needed."""
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: MessagesState) -> dict[str, list[ToolMessage]]:
        """Execute all tool calls from the latest model response."""
        last_message = state["messages"][-1]
        result: list[ToolMessage] = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name not in TOOLS_BY_NAME:
                raise ValueError(f"Unknown tool requested by model: {tool_name}")
            observation = TOOLS_BY_NAME[tool_name].invoke(tool_call["args"])
            result.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": result}

    def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
        """Continue when the model returned tool calls; otherwise finish."""
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile()


def main() -> None:
    """Run a small example request."""
    agent = create_agent()
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="Add 3 and 4.")],
            "llm_calls": 0,
        }
    )
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
