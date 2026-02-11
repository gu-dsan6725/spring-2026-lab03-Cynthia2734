"""Train and evaluate an XGBoost classification model on Wine data.

Loads the prepared train/test splits, trains an XGBoost classifier,
evaluates performance, and saves the model and evaluation artifacts.
Supports cross-validation and hyperparameter tuning via CLI flags.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
MODEL_FILENAME: str = "xgboost_model.joblib"
FIGURE_DPI: int = 150
N_ESTIMATORS: int = 200
MAX_DEPTH: int = 6
LEARNING_RATE: float = 0.1
RANDOM_STATE: int = 42
NUM_CLASSES: int = 3
CLASS_NAMES: list[str] = ["class_0", "class_1", "class_2"]

# Cross-validation constants
CV_FOLDS: int = 5
CV_SCORING: str = "accuracy"
N_ITER_SEARCH: int = 20

# Hyperparameter search space
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}


def _load_splits(
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test splits from parquet files."""
    path = Path(output_dir)

    x_train = pl.read_parquet(path / "x_train.parquet").to_numpy()
    x_test = pl.read_parquet(path / "x_test.parquet").to_numpy()
    y_train = pl.read_parquet(path / "y_train.parquet").to_numpy().ravel()
    y_test = pl.read_parquet(path / "y_test.parquet").to_numpy().ravel()

    logger.info(f"Loaded splits: train={x_train.shape}, test={x_test.shape}")
    return x_train, x_test, y_train, y_test


def _train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Train an XGBoost classifier with default hyperparameters."""
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    model.fit(x_train, y_train)
    logger.info(
        f"Trained XGBoost classifier with n_estimators={N_ESTIMATORS}, "
        f"max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}"
    )
    return model


def _run_cross_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: XGBClassifier,
) -> dict:
    """Run stratified k-fold cross-validation and return score statistics."""
    logger.info(f"Running {CV_FOLDS}-fold stratified cross-validation...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
    )

    cv_results = {
        "cv_mean_accuracy": round(float(np.mean(scores)), 4),
        "cv_std_accuracy": round(float(np.std(scores)), 4),
        "cv_scores": [round(float(s), 4) for s in scores],
    }

    logger.info(
        f"Cross-validation accuracy: {cv_results['cv_mean_accuracy']} "
        f"(+/- {cv_results['cv_std_accuracy']})"
    )
    logger.info(f"Per-fold accuracy scores: {cv_results['cv_scores']}")

    return cv_results


def _run_hyperparameter_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[XGBClassifier, RandomizedSearchCV, float]:
    """Run randomized search for hyperparameter tuning.

    Returns:
        Tuple of (best estimator, full RandomizedSearchCV object, tuning time in seconds).
    """
    logger.warning(
        f"Starting hyperparameter tuning with {N_ITER_SEARCH} iterations "
        f"and {CV_FOLDS}-fold CV. This may take a while."
    )

    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    tune_start = time.time()
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring=CV_SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(x_train, y_train)
    tuning_time = time.time() - tune_start

    logger.info(f"Best CV accuracy: {search.best_score_:.4f}")
    logger.info(f"Tuning completed in {tuning_time:.1f} seconds")
    logger.info(f"Best parameters:\n{json.dumps(search.best_params_, indent=2, default=str)}")

    return search.best_estimator_, search, tuning_time


def _save_tuning_results(
    search: RandomizedSearchCV,
    tuning_time: float,
    output_path: Path,
) -> None:
    """Save hyperparameter tuning results to a JSON file."""
    cv_results = search.cv_results_

    candidates = []
    for i in range(len(cv_results["params"])):
        candidates.append(
            {
                "rank": int(cv_results["rank_test_score"][i]),
                "mean_accuracy": round(float(cv_results["mean_test_score"][i]), 4),
                "std_accuracy": round(float(cv_results["std_test_score"][i]), 4),
                "params": {
                    k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                    for k, v in cv_results["params"][i].items()
                },
            }
        )

    candidates.sort(key=lambda x: x["rank"])

    results = {
        "best_params": {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in search.best_params_.items()
        },
        "best_cv_accuracy": round(float(search.best_score_), 4),
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
        "tuning_time_seconds": round(tuning_time, 2),
        "all_candidates": candidates,
    }

    filepath = output_path / "tuning_results.json"
    filepath.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Tuning results saved to {filepath}")


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute classification evaluation metrics."""
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro"))
    recall_macro = float(recall_score(y_true, y_pred, average="macro"))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    precision_per_class = precision_score(y_true, y_pred, average=None).tolist()
    recall_per_class = recall_score(y_true, y_pred, average=None).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None).tolist()

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(precision_per_class[i], 4),
                "recall": round(recall_per_class[i], 4),
                "f1": round(f1_per_class[i], 4),
            }
            for i in range(len(CLASS_NAMES))
        },
    }

    logger.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")
    return metrics


def _save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save the sklearn classification report to a text file."""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    filepath = output_path / "classification_report.txt"
    filepath.write_text(report)
    logger.info(f"Classification report saved to {filepath}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Generate a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    filepath = output_path / "confusion_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {filepath}")


def _plot_roc_curves(
    model: XGBClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Generate per-class ROC curves with AUC scores."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_prob = model.predict_proba(x_test)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-Class ROC Curves")
    ax.legend(loc="lower right")

    plt.tight_layout()
    filepath = output_path / "roc_curves.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"ROC curves saved to {filepath}")


def _plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Generate a feature importance bar chart."""
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        range(len(importances)),
        importances[sorted_indices],
        align="center",
        alpha=0.8,
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in sorted_indices],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("XGBoost Feature Importance")

    plt.tight_layout()
    filepath = output_path / "feature_importance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Feature importance plot saved to {filepath}")


def _save_model(
    model: XGBClassifier,
    output_path: Path,
) -> None:
    """Save the trained model to disk."""
    filepath = output_path / MODEL_FILENAME
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def _write_evaluation_report(
    metrics: dict,
    output_path: Path,
    cv_results: Optional[dict] = None,
    best_params: Optional[dict] = None,
    tuning_time: Optional[float] = None,
) -> None:
    """Write an evaluation report to a markdown file."""
    report = "# Model Evaluation Report\n\n"

    report += "## Metrics Summary\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| Accuracy | {metrics['accuracy']} |\n"
    report += f"| Precision (macro) | {metrics['precision_macro']} |\n"
    report += f"| Recall (macro) | {metrics['recall_macro']} |\n"
    report += f"| F1-Score (macro) | {metrics['f1_macro']} |\n\n"

    report += "## Per-Class Metrics\n\n"
    report += "| Class | Precision | Recall | F1-Score |\n"
    report += "|-------|-----------|--------|----------|\n"
    for class_name, class_metrics in metrics["per_class"].items():
        report += (
            f"| {class_name} | {class_metrics['precision']} "
            f"| {class_metrics['recall']} | {class_metrics['f1']} |\n"
        )
    report += "\n"

    report += "## Confusion Matrix\n\n"
    report += "The confusion matrix heatmap shows the distribution of predicted vs actual\n"
    report += "classes. See `confusion_matrix.png` for the visualization.\n\n"

    if cv_results is not None:
        report += "## Cross-Validation Results\n\n"
        report += f"- **Folds**: {CV_FOLDS}\n"
        report += f"- **Mean Accuracy**: {cv_results['cv_mean_accuracy']}\n"
        report += f"- **Std Accuracy**: {cv_results['cv_std_accuracy']}\n"
        report += f"- **Per-fold Accuracy**: {cv_results['cv_scores']}\n\n"

    if best_params is not None:
        report += "## Hyperparameter Tuning\n\n"
        if tuning_time is not None:
            report += f"- **Tuning time**: {tuning_time:.1f} seconds\n"
        report += f"- **Search iterations**: {N_ITER_SEARCH}\n"
        report += f"- **CV folds**: {CV_FOLDS}\n\n"
        report += "### Best Hyperparameters\n\n"
        report += "| Parameter | Value |\n"
        report += "|-----------|-------|\n"
        for param, value in sorted(best_params.items()):
            report += f"| {param} | {value} |\n"
        report += "\n"

    report += "## Artifacts\n\n"
    report += "- `confusion_matrix.png`: Confusion matrix heatmap\n"
    report += "- `feature_importance.png`: XGBoost feature importance ranking\n"
    report += "- `roc_curves.png`: Per-class ROC curves with AUC scores\n"
    report += "- `classification_report.txt`: Detailed classification report\n"
    report += "- `xgboost_model.joblib`: Trained model file\n"

    if best_params is not None:
        report += "- `tuning_results.json`: Hyperparameter tuning results\n"

    filepath = output_path / "evaluation_report.md"
    filepath.write_text(report)
    logger.info(f"Evaluation report saved to {filepath}")


def run_training_and_evaluation(
    tune: bool = False,
    cv_only: bool = False,
) -> None:
    """Run the full model training and evaluation pipeline."""
    start_time = time.time()
    logger.info("Starting model training and evaluation...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_splits(OUTPUT_DIR)

    cv_results = None
    best_params = None
    tuning_time = None

    if tune:
        model, search, tuning_time = _run_hyperparameter_tuning(x_train, y_train)
        best_params = search.best_params_
        _save_tuning_results(search, tuning_time, output_path)
        cv_results = _run_cross_validation(x_train, y_train, model)

    elif cv_only:
        model = _train_model(x_train, y_train)
        cv_results = _run_cross_validation(x_train, y_train, model)

    else:
        model = _train_model(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = _compute_metrics(y_test, y_pred)

    _save_classification_report(y_test, y_pred, output_path)
    _plot_confusion_matrix(y_test, y_pred, output_path)
    _plot_roc_curves(model, x_test, y_test, output_path)
    feature_names = pl.read_parquet(output_path / "x_train.parquet").columns
    _plot_feature_importance(model, feature_names, output_path)

    _save_model(model, output_path)
    _write_evaluation_report(
        metrics,
        output_path,
        cv_results=cv_results,
        best_params=best_params,
        tuning_time=tuning_time,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    if minutes > 0:
        logger.info(
            f"Training and evaluation completed in {minutes} minutes and {seconds:.1f} seconds"
        )
    else:
        logger.info(f"Training and evaluation completed in {seconds:.1f} seconds")


def main() -> None:
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost classifier on Wine data",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run randomized hyperparameter search before training",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        default=False,
        help="Run cross-validation on the default model (no tuning)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Set logging level to DEBUG",
    )

    args = parser.parse_args()

    if args.tune and args.cv_only:
        parser.error("--tune and --cv-only are mutually exclusive")

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    run_training_and_evaluation(
        tune=args.tune,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
