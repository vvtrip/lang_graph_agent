"""MCP tools for training and using the classical digit classifier."""

from __future__ import annotations

import json
import asyncio
from typing import Any
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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
        "f1_macro": round(result.get("f1_macro", accuracy), 4),
        "objective": result.get("objective", "accuracy"),
        "score": round(result.get("score", accuracy), 4),
        "target_score": result.get("target_score"),
        "target_met": result.get("target_met"),
        "requested_classifier": result.get("requested_classifier"),
        "sample_index": result.get("sample_index", 0),
        "prediction": result.get("prediction"),
    }


def _save_status(result: dict[str, Any]) -> None:
    _STATUS_FILE.write_text(
        json.dumps({"status": "ready", **_summary(result)}),
        encoding="utf-8",
    )


@mcp.tool()
async def train_mnist(
    target_accuracy: float = 0.95,
    objective: str = "accuracy",
    target_score: float | None = None,
    classifier: str = "auto",
) -> str:
    """Train a requested classifier or select the best candidate automatically."""
    global _last_result, _model, _X_test
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be greater than 0 and at most 1")
    if objective not in {"accuracy", "f1_macro"}:
        raise ValueError("objective must be 'accuracy' or 'f1_macro'")
    if classifier not in {
        "auto",
        "logistic_regression",
        "random_forest",
    }:
        raise ValueError(
            "classifier must be 'auto', 'logistic_regression', or 'random_forest'"
        )
    if target_score is not None and not 0.0 < target_score <= 1.0:
        raise ValueError("target_score must be greater than 0 and at most 1")

    def train() -> dict[str, Any]:
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data,
            digits.target,
            test_size=0.2,
            random_state=42,
            stratify=digits.target,
        )
        candidates = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, random_state=42),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=1,
            ),
        }
        if classifier != "auto":
            candidates = {
                classifier: candidates[classifier],
            }
        evaluated: list[dict[str, Any]] = []
        for model_name, model in candidates.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, predictions))
            f1_macro = float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            )
            evaluated.append(
                {
                    "model": model,
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "f1_macro": f1_macro,
                }
            )
        selected = max(
            evaluated,
            key=lambda item: item["f1_macro"]
            if objective == "f1_macro"
            else item["accuracy"],
        )
        return {
            "model": selected["model"],
            "X_test": X_test,
            "model_name": selected["model_name"],
            "attempt": len(evaluated),
            "accuracy": selected["accuracy"],
            "f1_macro": selected["f1_macro"],
            "objective": objective,
            "score": selected[objective],
            "target_score": target_score,
            "target_met": (
                target_score is None or selected[objective] >= target_score
            ),
            "requested_classifier": classifier,
            "sample_index": 0,
            "prediction": int(selected["model"].predict(X_test[:1])[0]),
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
