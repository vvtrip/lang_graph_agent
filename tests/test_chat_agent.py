import pytest

from langchain.messages import AIMessage

from chat_agent import create_chat_agent, run_chat
from mcp_server import model_status, train_mnist


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


def test_graph_factory_ignores_langgraph_config(monkeypatch) -> None:
    created = []
    monkeypatch.setattr(
        "chat_agent._create_model",
        lambda: created.append(True) or FakeModel(),
    )

    graph = create_chat_agent({})

    assert graph is not None
    assert created == [True]


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


@pytest.mark.anyio
async def test_status_recomputes_metrics_from_model_artifact() -> None:
    import json

    await train_mnist(classifier="random_forest")
    result = json.loads(model_status())

    assert result["metrics_source"] == "persisted_model_recomputed"
    assert result["metrics_verified"] is True


def test_status_is_not_trained_when_artifacts_are_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mcp_server._MODEL_FILE", tmp_path / ".mnist_model.pkl"
    )
    status_file = tmp_path / ".mnist_status.json"
    status_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("mcp_server._STATUS_FILE", status_file)

    import json

    assert json.loads(model_status()) == {"status": "not_trained"}
