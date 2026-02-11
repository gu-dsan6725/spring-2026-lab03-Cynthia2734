"""Tests for the Wine XGBoost classification pipeline."""

import importlib
import json

import numpy as np
import pytest
from xgboost import XGBClassifier

# Module name starts with a digit, so use importlib
_module = importlib.import_module("part1_claude_code.src.03_xgboost_model")
_compute_metrics = _module._compute_metrics
_run_cross_validation = _module._run_cross_validation
_run_hyperparameter_tuning = _module._run_hyperparameter_tuning
_save_tuning_results = _module._save_tuning_results
_write_evaluation_report = _module._write_evaluation_report


# Test fixtures
@pytest.fixture
def sample_data():
    """Create small sample classification data for testing."""
    rng = np.random.RandomState(42)
    x = rng.rand(100, 5)
    y = (x[:, 0] + x[:, 1] > 1.0).astype(int) + (x[:, 2] > 0.7).astype(int)
    return x, y


@pytest.fixture
def trained_model(
    sample_data,
):
    """Return a trained XGBClassifier on sample data."""
    x, y = sample_data
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(x, y)
    return model


@pytest.fixture
def sample_metrics():
    """Return sample classification metrics dictionary."""
    return {
        "accuracy": 0.9444,
        "precision_macro": 0.9500,
        "recall_macro": 0.9333,
        "f1_macro": 0.9394,
        "per_class": {
            "class_0": {"precision": 1.0, "recall": 0.9, "f1": 0.9474},
            "class_1": {"precision": 0.9, "recall": 1.0, "f1": 0.9474},
            "class_2": {"precision": 0.95, "recall": 0.9, "f1": 0.9234},
        },
    }


class TestComputeMetrics:
    """Tests for _compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = _compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_imperfect_predictions(self):
        """Test metrics with imperfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 0])

        metrics = _compute_metrics(y_true, y_pred)

        assert 0 < metrics["accuracy"] < 1
        assert 0 < metrics["f1_macro"] < 1

    def test_returns_expected_keys(self):
        """Test that all expected metric keys are present."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])

        metrics = _compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "per_class" in metrics

    def test_per_class_metrics_present(self):
        """Test that per-class metrics are included."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = _compute_metrics(y_true, y_pred)

        assert "class_0" in metrics["per_class"]
        assert "class_1" in metrics["per_class"]
        assert "class_2" in metrics["per_class"]
        for class_metrics in metrics["per_class"].values():
            assert "precision" in class_metrics
            assert "recall" in class_metrics
            assert "f1" in class_metrics


class TestRunCrossValidation:
    """Tests for _run_cross_validation function."""

    def test_returns_expected_keys(
        self,
        sample_data,
        trained_model,
    ):
        """Test that CV results contain expected keys."""
        x, y = sample_data
        results = _run_cross_validation(x, y, trained_model)

        assert "cv_mean_accuracy" in results
        assert "cv_std_accuracy" in results
        assert "cv_scores" in results

    def test_cv_scores_length(
        self,
        sample_data,
        trained_model,
    ):
        """Test that CV produces the correct number of fold scores."""
        x, y = sample_data
        results = _run_cross_validation(x, y, trained_model)

        assert len(results["cv_scores"]) == 5

    def test_cv_mean_is_valid(
        self,
        sample_data,
        trained_model,
    ):
        """Test that CV mean accuracy is between 0 and 1."""
        x, y = sample_data
        results = _run_cross_validation(x, y, trained_model)

        assert 0 <= results["cv_mean_accuracy"] <= 1
        assert results["cv_std_accuracy"] >= 0


class TestRunHyperparameterTuning:
    """Tests for _run_hyperparameter_tuning function."""

    def test_returns_model_search_and_time(
        self,
        sample_data,
        monkeypatch,
    ):
        """Test that tuning returns a model, search object, and time."""
        monkeypatch.setattr(_module, "N_ITER_SEARCH", 2)
        monkeypatch.setattr(_module, "CV_FOLDS", 2)

        x, y = sample_data
        model, search, tuning_time = _run_hyperparameter_tuning(x, y)

        assert isinstance(model, XGBClassifier)
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert tuning_time > 0

    def test_best_params_are_valid(
        self,
        sample_data,
        monkeypatch,
    ):
        """Test that best params contain expected hyperparameters."""
        monkeypatch.setattr(_module, "N_ITER_SEARCH", 2)
        monkeypatch.setattr(_module, "CV_FOLDS", 2)

        x, y = sample_data
        _, search, _ = _run_hyperparameter_tuning(x, y)

        assert "n_estimators" in search.best_params_
        assert "max_depth" in search.best_params_
        assert "learning_rate" in search.best_params_


class TestSaveTuningResults:
    """Tests for _save_tuning_results function."""

    def test_saves_json_file(
        self,
        sample_data,
        tmp_path,
        monkeypatch,
    ):
        """Test that tuning results are saved as JSON."""
        monkeypatch.setattr(_module, "N_ITER_SEARCH", 2)
        monkeypatch.setattr(_module, "CV_FOLDS", 2)

        x, y = sample_data
        _, search, tuning_time = _run_hyperparameter_tuning(x, y)

        _save_tuning_results(search, tuning_time, tmp_path)

        filepath = tmp_path / "tuning_results.json"
        assert filepath.exists()

        results = json.loads(filepath.read_text())
        assert "best_params" in results
        assert "best_cv_accuracy" in results
        assert "all_candidates" in results
        assert "tuning_time_seconds" in results


class TestWriteEvaluationReport:
    """Tests for _write_evaluation_report function."""

    def test_basic_report(
        self,
        sample_metrics,
        tmp_path,
    ):
        """Test basic report without CV or tuning info."""
        _write_evaluation_report(sample_metrics, tmp_path)

        filepath = tmp_path / "evaluation_report.md"
        assert filepath.exists()

        content = filepath.read_text()
        assert "# Model Evaluation Report" in content
        assert "0.9444" in content
        assert "Per-Class Metrics" in content
        assert "Cross-Validation" not in content

    def test_report_with_cv_results(
        self,
        sample_metrics,
        tmp_path,
    ):
        """Test report includes CV section when provided."""
        cv_results = {
            "cv_mean_accuracy": 0.95,
            "cv_std_accuracy": 0.02,
            "cv_scores": [0.94, 0.96, 0.95, 0.94, 0.96],
        }

        _write_evaluation_report(
            sample_metrics,
            tmp_path,
            cv_results=cv_results,
        )

        content = (tmp_path / "evaluation_report.md").read_text()
        assert "Cross-Validation Results" in content
        assert "0.95" in content

    def test_report_with_tuning(
        self,
        sample_metrics,
        tmp_path,
    ):
        """Test report includes tuning section when provided."""
        best_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
        }

        _write_evaluation_report(
            sample_metrics,
            tmp_path,
            best_params=best_params,
            tuning_time=42.5,
        )

        content = (tmp_path / "evaluation_report.md").read_text()
        assert "Hyperparameter Tuning" in content
        assert "42.5 seconds" in content
        assert "tuning_results.json" in content
        assert "300" in content
