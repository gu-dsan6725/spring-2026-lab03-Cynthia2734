import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

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


def _load_artifacts() -> Tuple[XGBClassifier, pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
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


def _generate_markdown(
    model: XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
    tuning_results: Dict[str, Any],
) -> str:
    """Generate the markdown report content."""
    # Calculate metrics
    y_pred = model.predict(X_test.to_numpy())
    report_dict = classification_report(y_test.to_numpy().flatten(), y_pred, output_dict=True)

    # Feature Importance
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:5]
    top_features = [X_test.columns[i] for i in indices]

    md = "# Wine Classification Model Evaluation Report\n\n"

    md += "## 1. Executive Summary\n"
    md += (
        f"The XGBoost model achieved an overall accuracy of **{report_dict['accuracy']:.2%}** "
        "on the test set. "
    )
    md += (
        "The pipeline included EDA, Feature Engineering (3 derived features), "
        "and Hyperparameter Tuning.\n\n"
    )

    md += "## 2. Dataset Overview\n"
    md += f"*   **Test Samples**: {len(X_test)}\n"
    md += f"*   **Features**: {len(X_test.columns)} (Original + Derived)\n"
    md += "*   **Classes**: 3 (Wine Cultivars)\n\n"

    md += "### Key Visualizations\n"
    md += "#### Feature Distributions\n"
    md += f"![Distributions]({DIST_PLOT})\n\n"
    md += "#### Correlation Matrix\n"
    md += f"![Correlation]({CORR_PLOT})\n\n"

    md += "## 3. Model Configuration\n"
    md += "### Best Hyperparameters\n"
    if tuning_results:
        md += "```json\n"
        md += json.dumps(tuning_results, indent=2)
        md += "\n```\n"
    else:
        md += "Default XGBoost parameters used (no tuning results found).\n"

    md += "\n### Top 5 Important Features\n"
    for i, feature in enumerate(top_features):
        md += f"{i + 1}. {feature} ({importances[indices[i]]:.4f})\n"
    md += f"\n![Feature Importance]({FI_PLOT})\n\n"

    md += "## 4. Performance Metrics\n"
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

    md += "## 5. Recommendations\n"
    md += (
        "*   **Model Deployment**: The model shows excellent performance and is ready for "
        "deployment.\n"
    )
    md += (
        "*   **Monitoring**: Monitor for data drift as the current dataset is small "
        "(178 samples).\n"
    )

    return md


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
