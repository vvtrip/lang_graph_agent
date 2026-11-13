import pytest

from langchain.messages import AIMessage

from chat_agent import run_chat


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
