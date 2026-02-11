"""Generate a comprehensive model evaluation report from trained model artifacts.

Loads a trained model, test data, and existing evaluation metrics,
then builds a detailed report and saves the completed report.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
TOP_N_FEATURES: int = 5
DEFAULT_OUTPUT_DIR: str = "output"
REPORT_FILENAME: str = "full_report.md"
CLASS_NAMES: list[str] = ["class_0", "class_1", "class_2"]


def _find_model_file(
    output_dir: Path,
) -> Path:
    """Find the first .joblib or .pkl model file in the output directory."""
    for pattern in ["*.joblib", "*.pkl"]:
        files = list(output_dir.glob(pattern))
        if files:
            logger.info(f"Found model file: {files[0]}")
            return files[0]

    raise FileNotFoundError(f"No .joblib or .pkl model file found in {output_dir}")


def _extract_model_info(
    model_path: Path,
) -> dict:
    """Load model and extract type, hyperparameters, and feature importance."""
    model = joblib.load(model_path)
    model_type = type(model).__name__

    params = model.get_params()
    key_params = {
        k: v
        for k, v in params.items()
        if v is not None
        and k
        in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "objective",
            "random_state",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "num_class",
        ]
    }

    logger.info(f"Model type: {model_type}")
    logger.info(
        f"Hyperparameters:\n{json.dumps({k: str(v) for k, v in key_params.items()}, indent=2)}"
    )

    return {
        "model": model,
        "model_type": model_type,
        "params": key_params,
    }


def _load_test_data(
    output_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load test features and labels from parquet files."""
    x_test = pl.read_parquet(output_dir / "x_test.parquet")
    y_test = pl.read_parquet(output_dir / "y_test.parquet")

    logger.info(f"Loaded test data: x={x_test.shape}, y={y_test.shape}")
    return x_test, y_test


def _compute_metrics(
    model: object,
    x_test: pl.DataFrame,
    y_test: pl.DataFrame,
) -> dict:
    """Compute classification metrics."""
    preds = model.predict(x_test.to_numpy())
    y_arr = y_test.to_numpy().flatten().astype(int)

    accuracy = float(accuracy_score(y_arr, preds))
    precision_macro = float(precision_score(y_arr, preds, average="macro"))
    recall_macro = float(recall_score(y_arr, preds, average="macro"))
    f1_macro = float(f1_score(y_arr, preds, average="macro"))

    precision_per_class = precision_score(y_arr, preds, average=None).tolist()
    recall_per_class = recall_score(y_arr, preds, average=None).tolist()
    f1_per_class = f1_score(y_arr, preds, average=None).tolist()

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision (macro)": round(precision_macro, 4),
        "Recall (macro)": round(recall_macro, 4),
        "F1-Score (macro)": round(f1_macro, 4),
    }

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(precision_per_class[i], 4),
            "recall": round(recall_per_class[i], 4),
            "f1": round(f1_per_class[i], 4),
        }

    logger.info(f"Computed metrics:\n{json.dumps(metrics, indent=2)}")
    return {"summary": metrics, "per_class": per_class}


def _get_feature_importance(
    model: object,
    feature_names: list[str],
    top_n: int = TOP_N_FEATURES,
) -> list:
    """Extract top N features ranked by importance."""
    importance = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importance),
        key=lambda x: -x[1],
    )

    top_features = [
        {"rank": i + 1, "name": name, "score": round(float(score), 4)}
        for i, (name, score) in enumerate(ranked[:top_n])
    ]

    logger.info(f"Top {top_n} features: {top_features}")
    return top_features


def _get_dataset_info(
    output_dir: Path,
    x_test: pl.DataFrame,
) -> dict:
    """Gather dataset size information."""
    x_train = pl.read_parquet(output_dir / "x_train.parquet")

    total = x_train.shape[0] + x_test.shape[0]
    info = {
        "total": total,
        "train": x_train.shape[0],
        "test": x_test.shape[0],
        "n_features": x_test.shape[1],
        "feature_names": x_test.columns,
        "class_names": CLASS_NAMES,
    }

    logger.info(
        f"Dataset: {info['total']} total, "
        f"{info['train']} train, "
        f"{info['test']} test, "
        f"{info['n_features']} features"
    )
    return info


def _build_report(
    dataset_info: dict,
    model_info: dict,
    metrics: dict,
    top_features: list,
) -> str:
    """Fill in the report template with actual data."""
    summary = metrics["summary"]
    per_class = metrics["per_class"]
    lines = []

    lines.append("# Model Evaluation Report")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    accuracy = summary.get("Accuracy", "N/A")
    f1_val = summary.get("F1-Score (macro)", "N/A")
    lines.append(
        f"An {model_info['model_type']} classification model was trained to predict "
        f"wine classes ({', '.join(dataset_info['class_names'])}). "
        f"The model achieves an accuracy of {accuracy} "
        f"with a macro F1-score of {f1_val}."
    )
    lines.append("")

    # Dataset Overview
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Total samples | {dataset_info['total']:,} |")
    lines.append(f"| Training samples | {dataset_info['train']:,} |")
    lines.append(f"| Test samples | {dataset_info['test']:,} |")
    lines.append(f"| Number of features | {dataset_info['n_features']} |")
    lines.append(f"| Number of classes | {len(dataset_info['class_names'])} |")
    lines.append(f"| Class names | {', '.join(dataset_info['class_names'])} |")
    lines.append("")

    # Model Configuration
    lines.append("## Model Configuration")
    lines.append("")
    lines.append("| Hyperparameter | Value |")
    lines.append("|----------------|-------|")
    lines.append(f"| Model type | {model_info['model_type']} |")
    for param_name, param_value in model_info["params"].items():
        lines.append(f"| {param_name} | {param_value} |")
    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for metric_name, metric_value in summary.items():
        lines.append(f"| {metric_name} | {metric_value} |")
    lines.append("")

    # Per-Class Metrics
    lines.append("## Per-Class Metrics")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1-Score |")
    lines.append("|-------|-----------|--------|----------|")
    for class_name, class_metrics in per_class.items():
        lines.append(
            f"| {class_name} | {class_metrics['precision']} "
            f"| {class_metrics['recall']} | {class_metrics['f1']} |"
        )
    lines.append("")

    # Confusion Matrix
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("See `confusion_matrix.png` for the confusion matrix heatmap.")
    lines.append("")

    # Feature Importance
    lines.append(f"## Feature Importance (Top {len(top_features)})")
    lines.append("")
    lines.append("| Rank | Feature | Importance Score |")
    lines.append("|------|---------|-----------------:|")
    for feat in top_features:
        lines.append(f"| {feat['rank']} | {feat['name']} | {feat['score']} |")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations for Improvement")
    lines.append("")
    lines.append(
        "1. **Hyperparameter tuning**: Run cross-validated randomized or Bayesian "
        "search over learning_rate, max_depth, n_estimators, and subsample "
        "to find a better configuration."
    )
    lines.append(
        "2. **Feature engineering**: The top features suggest room for "
        "derived features. Consider interactions, polynomial terms, or "
        "domain-specific transformations."
    )
    lines.append(
        "3. **Ensemble methods**: Consider stacking or blending XGBoost with "
        "other classifiers (e.g., Random Forest, SVM) to improve robustness."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate a model evaluation report from artifacts in the output dir."""
    parser = argparse.ArgumentParser(
        description="Generate a model evaluation report from trained artifacts",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing model artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Set logging level to DEBUG",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    start_time = time.time()
    output_dir = Path(args.output_dir)

    logger.info(f"Generating report from artifacts in: {output_dir}")

    model_path = _find_model_file(output_dir)
    model_info = _extract_model_info(model_path)

    x_test, y_test = _load_test_data(output_dir)
    metrics = _compute_metrics(model_info["model"], x_test, y_test)

    top_features = _get_feature_importance(model_info["model"], x_test.columns)

    dataset_info = _get_dataset_info(output_dir, x_test)

    report_content = _build_report(dataset_info, model_info, metrics, top_features)

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(report_content)
    logger.info(f"Report saved to {report_path}")

    elapsed = time.time() - start_time
    logger.info(f"Report generation completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
