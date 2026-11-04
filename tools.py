"""Arithmetic tools used by the LangGraph agent."""

from langchain.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide one integer by another."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


TOOLS = [add, multiply, divide]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
