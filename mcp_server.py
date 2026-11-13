"""MCP tools for training and using the classical digit classifier."""

from __future__ import annotations

import json
import asyncio
import logging
import pickle
import sys
import time
import uuid
from typing import Any
from datetime import datetime, timedelta, timezone
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
_y_test: Any | None = None
_STATUS_FILE = Path(__file__).with_name(".mnist_status.json")
_MODEL_FILE = Path(__file__).with_name(".mnist_model.pkl")
_LOG_DIR = Path(__file__).with_name("logs")
_LOG_FILE = _LOG_DIR / "mnist_mcp.log"
_LOG_DIR.mkdir(exist_ok=True)
_IST = timezone(timedelta(hours=5, minutes=30))


class _IstFormatter(logging.Formatter):
    """Format log timestamps in India Standard Time."""

    def formatTime(self, record, datefmt=None):
        timestamp = datetime.fromtimestamp(record.created, _IST)
        return timestamp.isoformat(timespec="seconds")

logger = logging.getLogger("mnist.mcp")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = _IstFormatter("%(asctime)s %(levelname)s [mnist-mcp] %(message)s")
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)


def _log_event(operation_id: str, event: str, message: str, **details: Any) -> None:
    timestamp = datetime.now(_IST).isoformat()
    logger.info(
        "%s timestamp=%s operation=%s details=%s",
        message,
        timestamp,
        operation_id,
        json.dumps({"event": event, **details}, sort_keys=True),
    )


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


def _load_model_artifact() -> dict[str, Any]:
    if not _MODEL_FILE.exists():
        raise RuntimeError("The persisted model artifact is missing.")
    with _MODEL_FILE.open("rb") as model_file:
        artifact = pickle.load(model_file)
    required = {"model", "X_test", "y_test", "model_name"}
    if not required.issubset(artifact):
        raise RuntimeError("The persisted model artifact is incomplete.")
    return artifact


@mcp.tool()
async def train_mnist(
    target_accuracy: float = 0.95,
    objective: str = "accuracy",
    target_score: float | None = None,
    classifier: str = "auto",
) -> str:
    """Train a requested classifier or select the best candidate automatically."""
    global _last_result, _model, _X_test, _y_test
    operation_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
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
    _log_event(
        operation_id,
        "training_started",
        "Training operation started",
        tool="train_mnist",
        objective=objective,
        classifier=classifier,
    )

    def train() -> dict[str, Any]:
        digits = load_digits()
        _log_event(
            operation_id,
            "dataset_loaded",
            "Digits dataset loaded",
            samples=len(digits.data),
            features=digits.data.shape[1],
        )
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data,
            digits.target,
            test_size=0.2,
            random_state=42,
            stratify=digits.target,
        )
        _log_event(
            operation_id,
            "train_test_split_created",
            "Created train/test split",
            train_samples=len(X_train),
            test_samples=len(X_test),
            test_size=0.2,
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
            _log_event(
                operation_id,
                "model_training_started",
                f"Training {model_name}",
                train_samples=len(X_train),
            )
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
            _log_event(
                operation_id,
                "model_evaluated",
                f"Evaluated {model_name}",
                accuracy=round(accuracy, 4),
                f1_macro=round(f1_macro, 4),
            )
        selected = max(
            evaluated,
            key=lambda item: item["f1_macro"]
            if objective == "f1_macro"
            else item["accuracy"],
        )
        _log_event(
            operation_id,
            "model_selected",
            f"Selected {selected['model_name']}",
            objective=objective,
            score=round(selected[objective], 4),
        )
        sample_prediction = int(selected["model"].predict(X_test[:1])[0])
        _log_event(
            operation_id,
            "sample_prediction_completed",
            "Sample prediction completed",
            model=selected["model_name"],
            sample_index=0,
            prediction=sample_prediction,
            expected=int(y_test[0]),
        )
        return {
            "model": selected["model"],
            "X_test": X_test,
            "y_test": y_test,
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
            "operation_id": operation_id,
            "sample_index": 0,
            "prediction": sample_prediction,
        }

    try:
        trained = await asyncio.to_thread(train)
    except Exception:
        logger.exception(
            "Training operation failed operation=%s duration_ms=%.2f",
            operation_id,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    _model = trained.pop("model")
    _X_test = trained.pop("X_test")
    _y_test = trained.pop("y_test")
    _last_result = trained
    with _MODEL_FILE.open("wb") as model_file:
        pickle.dump(
            {
                "model": _model,
                "X_test": _X_test,
                "y_test": _y_test,
                "model_name": _last_result["model_name"],
            },
            model_file,
        )
    _log_event(
        operation_id,
        "training_completed",
        "Training operation completed",
        tool="train_mnist",
        model=_last_result["model_name"],
        score=round(_last_result["score"], 4),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    _save_status(_last_result)
    return json.dumps(_summary(_last_result))


@mcp.tool()
async def predict_digit(sample_index: int = 0) -> str:
    """Predict one held-out digit, training a model first if necessary."""
    global _last_result, _model, _X_test, _y_test
    operation_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    if not 0 <= sample_index < 360:
        raise ValueError("sample_index must be between 0 and 359")
    _log_event(
        operation_id,
        "prediction_started",
        "Prediction requested",
        tool="predict_digit",
        sample_index=sample_index,
    )
    if _model is None or _X_test is None:
        if _MODEL_FILE.exists():
            try:
                persisted = _load_model_artifact()
            except RuntimeError as error:
                logger.warning(
                    "Persisted model unavailable; retraining operation=%s reason=%s",
                    operation_id,
                    error,
                )
                await train_mnist()
                persisted = None
            if persisted is None:
                persisted = {
                    "model": _model,
                    "X_test": _X_test,
                    "y_test": _y_test,
                    "model_name": _last_result["model_name"],
                }
            _model = persisted["model"]
            _X_test = persisted["X_test"]
            _y_test = persisted["y_test"]
            if _last_result is None and _STATUS_FILE.exists():
                persisted_status = json.loads(
                    _STATUS_FILE.read_text(encoding="utf-8")
                )
                persisted_status.pop("status", None)
                persisted_status["model_name"] = persisted_status.get("model")
                persisted_status["attempt"] = persisted_status.get("attempts", 1)
                _last_result = persisted_status
            _log_event(
                operation_id,
                "model_loaded",
                "Persisted model loaded for prediction",
                model=_last_result.get("model_name")
                if _last_result
                else "unknown",
            )
        else:
            await train_mnist()
    _last_result = {
        **(_last_result or {}),
        "sample_index": sample_index,
        "prediction": int(_model.predict(_X_test[sample_index : sample_index + 1])[0]),
        "operation_id": operation_id,
    }
    _log_event(
        operation_id,
        "prediction_completed",
        "Prediction completed",
        tool="predict_digit",
        sample_index=sample_index,
        prediction=_last_result["prediction"],
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    _save_status(_last_result)
    return json.dumps(_summary(_last_result))


@mcp.tool()
def model_status() -> str:
    """Return the latest model result, or explain that no run exists yet."""
    operation_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    status_text = (
        _STATUS_FILE.read_text(encoding="utf-8")
        if _STATUS_FILE.exists()
        else ""
    )
    if not _MODEL_FILE.exists() and not status_text.strip():
        _log_event(
            operation_id,
            "status_checked",
            "No trained model found",
            tool="model_status",
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return json.dumps({"status": "not_trained"})
    artifact = _load_model_artifact()
    if not status_text.strip():
        raise RuntimeError("Model status metadata is missing.")
    persisted_status = json.loads(status_text)
    predictions = artifact["model"].predict(artifact["X_test"])
    verified_accuracy = float(
        accuracy_score(artifact["y_test"], predictions)
    )
    verified_f1 = float(
        f1_score(
            artifact["y_test"],
            predictions,
            average="macro",
            zero_division=0,
        )
    )
    if (
        persisted_status.get("model") != artifact["model_name"]
        or
        abs(verified_accuracy - persisted_status["accuracy"]) > 1e-4
        or abs(verified_f1 - persisted_status["f1_macro"]) > 1e-4
    ):
        raise RuntimeError("Persisted model metrics do not match status metadata.")
    verified_status = {
        **persisted_status,
        "accuracy": round(verified_accuracy, 4),
        "f1_macro": round(verified_f1, 4),
        "metrics_source": "persisted_model_recomputed",
        "metrics_verified": True,
    }
    _log_event(
        operation_id,
        "status_checked",
        "Persisted model metrics verified and returned",
        tool="model_status",
        model=verified_status.get("model"),
        accuracy=verified_status["accuracy"],
        f1_macro=verified_status["f1_macro"],
        metrics_source=verified_status["metrics_source"],
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )
    return json.dumps(verified_status)


if __name__ == "__main__":
    mcp.run(transport="stdio")
