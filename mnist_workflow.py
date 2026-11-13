"""Agentic classical ML workflow for an MNIST-like digit classifier.

The workflow is deliberately LLM-free: LangGraph orchestrates data loading,
training, evaluation, and prediction, while an evaluation node decides whether
to retry with a stronger model. This makes the graph reproducible and cheap to
run while still producing useful LangSmith traces.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing_extensions import TypedDict

load_dotenv()


class WorkflowState(TypedDict, total=False):
    """State passed between the graph nodes."""

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    model: Any
    model_name: str
    metrics: dict[str, Any]
    prediction: int
    sample_index: int
    target_accuracy: float
    attempt: int
    decision: str


def _model_for_attempt(attempt: int) -> tuple[str, Any]:
    if attempt == 1:
        return (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, random_state=42),
            ),
        )
    return (
        "random_forest",
        RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            n_jobs=1,
        ),
    )


@traceable(name="load_digits_dataset", tags=["mnist", "data"])
def load_data(_: WorkflowState) -> dict[str, Any]:
    """Load and split the local MNIST-like digits dataset."""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=42,
        stratify=digits.target,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sample_index": 0,
    }


@traceable(name="train_classifier", tags=["mnist", "training"])
def train_model(state: WorkflowState) -> dict[str, Any]:
    """Train the model selected by the current retry attempt."""
    attempt = state.get("attempt", 0) + 1
    model_name, model = _model_for_attempt(attempt)
    model.fit(state["X_train"], state["y_train"])
    return {"model": model, "model_name": model_name, "attempt": attempt}


@traceable(name="evaluate_classifier", tags=["mnist", "evaluation"])
def evaluate_model(state: WorkflowState) -> dict[str, Any]:
    """Measure accuracy and decide whether another model should be tried."""
    predictions = state["model"].predict(state["X_test"])
    accuracy = float(accuracy_score(state["y_test"], predictions))
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(
            state["y_test"], predictions
        ).tolist(),
    }
    target = state.get("target_accuracy", 0.95)
    decision = "accept" if accuracy >= target or state["attempt"] >= 2 else "retrain"
    return {"metrics": metrics, "decision": decision}


@traceable(name="predict_digit", tags=["mnist", "prediction"])
def predict_digit(state: WorkflowState) -> dict[str, int]:
    """Predict one held-out digit after the model has been accepted."""
    index = state.get("sample_index", 0)
    prediction = int(state["model"].predict(state["X_test"][index : index + 1])[0])
    return {"prediction": prediction}


def _next_step(state: WorkflowState) -> Literal["train_model", "predict_digit"]:
    if state.get("decision") == "retrain":
        return "train_model"
    return "predict_digit"


def create_workflow():
    """Build and compile the agentic ML workflow."""
    builder = StateGraph(WorkflowState)
    builder.add_node("load_data", load_data)
    builder.add_node("train_model", train_model)
    builder.add_node("evaluate_model", evaluate_model)
    builder.add_node("predict_digit", predict_digit)
    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "train_model")
    builder.add_edge("train_model", "evaluate_model")
    builder.add_conditional_edges(
        "evaluate_model",
        _next_step,
        {"train_model": "train_model", "predict_digit": "predict_digit"},
    )
    builder.add_edge("predict_digit", END)
    return builder.compile()


def run_workflow(
    *,
    target_accuracy: float = 0.95,
    sample_index: int = 0,
) -> WorkflowState:
    """Run the workflow with LangSmith metadata when tracing is enabled."""
    if not 0 <= sample_index < 360:
        raise ValueError("sample_index must be between 0 and 359")
    workflow = create_workflow()
    return workflow.invoke(
        {
            "target_accuracy": target_accuracy,
            "sample_index": sample_index,
            "attempt": 0,
        },
        config={
            "run_name": "mnist-agentic-classical-ml",
            "tags": ["mnist", "classical-ml", "langgraph"],
            "metadata": {
                "target_accuracy": target_accuracy,
                "langsmith_enabled": (
                    os.getenv("LANGSMITH_TRACING", "").lower() == "true"
                    or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
                ),
            },
        },
    )


def main() -> None:
    """Run the demo and print the trace-friendly summary."""
    result = run_workflow()
    print(f"Model: {result['model_name']}")
    print(f"Attempts: {result['attempt']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.3f}")
    print(f"Predicted held-out digit: {result['prediction']}")


if __name__ == "__main__":
    main()
