import pytest

from langchain.messages import AIMessage

from chat_agent import run_chat
from mcp_server import train_mnist


class FakeModel:
    def invoke(self, messages):
        if messages[-1].type == "human":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "train_mnist",
                        "args": {"target_accuracy": 0.9},
                        "id": "call_train",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="The current accuracy is 97.2%.")


@pytest.mark.anyio
async def test_chat_can_train() -> None:
    response = await run_chat("train the model", model=FakeModel())

    assert response == "The current accuracy is 97.2%."


@pytest.mark.anyio
async def test_chat_help() -> None:
    response = await run_chat("hello", model=FakeModel())

    assert response == "The current accuracy is 97.2%."


@pytest.mark.anyio
async def test_mcp_can_optimize_macro_f1() -> None:
    import json

    result = json.loads(await train_mnist(objective="f1_macro"))

    assert result["objective"] == "f1_macro"
    assert result["f1_macro"] == result["score"]
    assert 0.0 <= result["f1_macro"] <= 1.0


@pytest.mark.anyio
async def test_mcp_can_train_requested_random_forest() -> None:
    import json

    result = json.loads(await train_mnist(classifier="random_forest"))

    assert result["requested_classifier"] == "random_forest"
    assert result["model"] == "random_forest"
