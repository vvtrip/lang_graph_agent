"""MCP tools for training and using the classical digit classifier."""

from __future__ import annotations

import json
import asyncio
from typing import Any
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mcp = FastMCP("mnist-classical-ml")
_last_result: dict[str, Any] | None = None
_model: Any | None = None
_X_test: Any | None = None
_STATUS_FILE = Path(__file__).with_name(".mnist_status.json")


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    accuracy = result.get("accuracy")
    if accuracy is None:
        accuracy = result["metrics"]["accuracy"]
    return {
        "model": result.get("model_name", result.get("model", "unknown")),
        "attempts": result["attempt"],
        "accuracy": round(accuracy, 4),
        "sample_index": result.get("sample_index", 0),
        "prediction": result.get("prediction"),
    }


def _save_status(result: dict[str, Any]) -> None:
    _STATUS_FILE.write_text(
        json.dumps({"status": "ready", **_summary(result)}),
        encoding="utf-8",
    )


@mcp.tool()
async def train_mnist(target_accuracy: float = 0.95) -> str:
    """Train and evaluate the digit classifier, retrying when needed."""
    global _last_result, _model, _X_test
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be greater than 0 and at most 1")
    def train() -> dict[str, Any]:
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data,
            digits.target,
            test_size=0.2,
            random_state=42,
            stratify=digits.target,
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=42),
        )
        model.fit(X_train, y_train)
        accuracy = float(accuracy_score(y_test, model.predict(X_test)))
        return {
            "model": model,
            "X_test": X_test,
            "model_name": "logistic_regression",
            "attempt": 1,
            "accuracy": accuracy,
            "sample_index": 0,
            "prediction": int(model.predict(X_test[:1])[0]),
        }

    trained = await asyncio.to_thread(train)
    _model = trained.pop("model")
    _X_test = trained.pop("X_test")
    _last_result = trained
    _save_status(_last_result)
    return json.dumps(_summary(_last_result))


@mcp.tool()
async def predict_digit(sample_index: int = 0) -> str:
    """Predict one held-out digit, training a model first if necessary."""
    global _last_result, _model, _X_test
    if not 0 <= sample_index < 360:
        raise ValueError("sample_index must be between 0 and 359")
    if _model is None or _X_test is None:
        await train_mnist()
    _last_result = {
        **(_last_result or {}),
        "sample_index": sample_index,
        "prediction": int(_model.predict(_X_test[sample_index : sample_index + 1])[0]),
    }
    _save_status(_last_result)
    return json.dumps(_summary(_last_result))


@mcp.tool()
def model_status() -> str:
    """Return the latest model result, or explain that no run exists yet."""
    if _last_result is None:
        if not _STATUS_FILE.exists():
            return json.dumps({"status": "not_trained"})
        return _STATUS_FILE.read_text(encoding="utf-8")
    return json.dumps({"status": "ready", **_summary(_last_result)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
