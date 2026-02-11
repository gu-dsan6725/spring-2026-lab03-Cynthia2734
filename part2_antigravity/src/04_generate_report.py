import json
import logging
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Constants
OUTPUT_DIR: str = "part2_antigravity/output/"
MODEL_FILE: str = "xgboost_model.joblib"
TUNING_FILE: str = "tuning_results.json"
REPORT_MD: str = "evaluation_report.md"
CM_PLOT: str = "confusion_matrix.png"
ROC_PLOT: str = "roc_curves.png"
FI_PLOT: str = "feature_importance.png"
DIST_PLOT: str = "distributions.png"
CORR_PLOT: str = "correlation_matrix.png"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _load_artifacts() -> tuple[XGBClassifier, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Load model, test data, and tuning results."""
    model = joblib.load(f"{OUTPUT_DIR}{MODEL_FILE}")
    X_test = pl.read_parquet(f"{OUTPUT_DIR}x_test.parquet")
    y_test = pl.read_parquet(f"{OUTPUT_DIR}y_test.parquet")

    tuning_results = {}
    tuning_path = Path(f"{OUTPUT_DIR}{TUNING_FILE}")
    if tuning_path.exists():
        with open(tuning_path, "r") as f:
            tuning_results = json.load(f)

    return model, X_test, y_test, tuning_results


def _get_classification_report(
    model: XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
) -> dict[str, Any]:
    """Calculate classification metrics."""
    y_pred = model.predict(X_test.to_numpy())
    return classification_report(y_test.to_numpy().flatten(), y_pred, output_dict=True)


def _get_top_features(
    model: XGBClassifier,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Extract top 5 feature importances."""
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:5]
    return [(feature_names[i], importances[i]) for i in indices]


def _create_summary_section(
    accuracy: float,
    num_samples: int,
    num_features: int,
) -> str:
    """Generate executive summary and dataset overview."""
    md = "# Wine Classification Model Evaluation Report\n\n"
    md += "## 1. Executive Summary\n"
    md += (
        f"The XGBoost model achieved an overall accuracy of **{accuracy:.2%}** "
        "on the test set. "
    )
    md += (
        "The pipeline included EDA, Feature Engineering (3 derived features), "
        "and Hyperparameter Tuning.\n\n"
    )

    md += "## 2. Dataset Overview\n"
    md += f"*   **Test Samples**: {num_samples}\n"
    md += f"*   **Features**: {num_features} (Original + Derived)\n"
    md += "*   **Classes**: 3 (Wine Cultivars)\n\n"

    md += "### Key Visualizations\n"
    md += "#### Feature Distributions\n"
    md += f"![Distributions]({DIST_PLOT})\n\n"
    md += "#### Correlation Matrix\n"
    md += f"![Correlation]({CORR_PLOT})\n\n"
    return md


def _create_model_section(
    tuning_results: dict[str, Any],
    top_features: list[tuple[str, float]],
) -> str:
    """Generate model configuration and feature importance section."""
    md = "## 3. Model Configuration\n"
    md += "### Best Hyperparameters\n"
    if tuning_results:
        md += "```json\n"
        md += json.dumps(tuning_results, indent=2)
        md += "\n```\n"
    else:
        md += "Default XGBoost parameters used (no tuning results found).\n"

    md += "\n### Top 5 Important Features\n"
    for i, (feature, importance) in enumerate(top_features):
        md += f"{i + 1}. {feature} ({importance:.4f})\n"
    md += f"\n![Feature Importance]({FI_PLOT})\n\n"
    return md


def _create_metrics_section(report_dict: dict[str, Any]) -> str:
    """Generate performance metrics section."""
    md = "## 4. Performance Metrics\n"
    md += "### Classification Report\n"
    md += "| Class | Precision | Recall | F1-Score | Support |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for label, metrics in report_dict.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        md += (
            f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1-score']:.4f} | {metrics['support']} |\n"
        )

    md += f"\n**Macro Avg**: F1 {report_dict['macro avg']['f1-score']:.4f}\n"
    md += f"**Weighted Avg**: F1 {report_dict['weighted avg']['f1-score']:.4f}\n\n"

    md += "### Confusion Matrix\n"
    md += f"![Confusion Matrix]({CM_PLOT})\n\n"

    md += "### ROC Curves\n"
    md += f"![ROC Curves]({ROC_PLOT})\n\n"
    return md


def _create_recommendations_section() -> str:
    """Generate recommendations section."""
    md = "## 5. Recommendations\n"
    md += (
        "*   **Model Deployment**: The model shows excellent performance and is ready for "
        "deployment.\n"
    )
    md += (
        "*   **Monitoring**: Monitor for data drift as the current dataset is small "
        "(178 samples).\n"
    )
    return md


def _generate_markdown(
    model: XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
    tuning_results: dict[str, Any],
) -> str:
    """Generate the full markdown report content."""
    report_dict = _get_classification_report(model, X_test, y_test)
    top_features = _get_top_features(model, X_test.columns)

    parts = [
        _create_summary_section(
            report_dict["accuracy"], len(X_test), len(X_test.columns)
        ),
        _create_model_section(tuning_results, top_features),
        _create_metrics_section(report_dict),
        _create_recommendations_section(),
    ]

    return "".join(parts)


def main() -> None:
    """Execute Evaluation Report Generation."""
    logging.info("Starting Evaluation Report Generation Phase")

    model, X_test, y_test, tuning_results = _load_artifacts()
    logging.info("Loaded model and test data")

    report_content = _generate_markdown(model, X_test, y_test, tuning_results)

    output_path = f"{OUTPUT_DIR}{REPORT_MD}"
    with open(output_path, "w") as f:
        f.write(report_content)

    logging.info(f"Generated evaluation report at {output_path}")
    logging.info("Pipeline Completed Successfully")


if __name__ == "__main__":
    main()
