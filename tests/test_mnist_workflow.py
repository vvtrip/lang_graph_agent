from mnist_workflow import create_workflow, run_workflow


def test_workflow_trains_and_predicts() -> None:
    result = run_workflow(target_accuracy=0.90)

    assert result["metrics"]["accuracy"] >= 0.90
    assert result["model_name"] in {"logistic_regression", "random_forest"}
    assert 0 <= result["prediction"] <= 9
    assert len(result["metrics"]["confusion_matrix"]) == 10


def test_workflow_retries_when_target_is_unreachable() -> None:
    result = run_workflow(target_accuracy=1.01)

    assert result["attempt"] == 2
    assert result["model_name"] == "random_forest"
    assert result["decision"] == "accept"


def test_graph_is_compiled() -> None:
    workflow = create_workflow()

    assert workflow is not None
