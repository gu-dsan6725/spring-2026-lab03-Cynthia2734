import argparse
import json
import logging
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# Constants
OUTPUT_DIR: str = "part2_antigravity/output/"
MODEL_FILE: str = "xgboost_model.joblib"
TUNING_FILE: str = "tuning_results.json"
REPORT_FILE: str = "classification_report.txt"
CM_PLOT: str = "confusion_matrix.png"
ROC_PLOT: str = "roc_curves.png"
FI_PLOT: str = "feature_importance.png"
RANDOM_STATE: int = 42

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _load_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train and test data from parquet files."""
    X_train = pl.read_parquet(f"{OUTPUT_DIR}x_train.parquet")
    X_test = pl.read_parquet(f"{OUTPUT_DIR}x_test.parquet")
    y_train = pl.read_parquet(f"{OUTPUT_DIR}y_train.parquet")
    y_test = pl.read_parquet(f"{OUTPUT_DIR}y_test.parquet")
    return X_train, X_test, y_train, y_test


def _get_tuning_params() -> Dict[str, Any]:
    """Define hyperparameter search space."""
    return {
        "max_depth": np.arange(3, 11),
        "learning_rate": np.linspace(0.01, 0.3, 30),
        "n_estimators": np.arange(100, 501, 50),
        "min_child_weight": np.arange(1, 8),
        "subsample": np.linspace(0.6, 1.0, 5),
        "colsample_bytree": np.linspace(0.6, 1.0, 5),
    }


def _tune_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Perform RandomizedSearchCV to find best hyperparameters."""
    logging.info("Starting Hyperparameter Tuning...")
    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_dist = _get_tuning_params()

    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_search.fit(X, y)

    best_params = random_search.best_params_
    logging.info(f"Best Parameters: {json.dumps(best_params, indent=2, default=str)}")

    # Save results
    with open(f"{OUTPUT_DIR}{TUNING_FILE}", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    return best_params


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}{CM_PLOT}")
    plt.close()


def _plot_roc_curves(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Generate and save ROC curves for each class."""
    y_prob = model.predict_proba(X)
    n_classes = 3
    y_test_bin = label_binarize(y, classes=[0, 1, 2])

    plt.figure(figsize=(10, 8))
    colors = ["blue", "red", "green"]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}{ROC_PLOT}")
    plt.close()


def _plot_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
) -> None:
    """Generate and save feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(
        range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right"
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}{FI_PLOT}")
    plt.close()


def main() -> None:
    """Execute Model Training Phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--cv-only", action="store_true", help="Run CV only")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Starting Model Training Phase")

    # Load Data (Polars -> Numpy)
    X_train_pl, X_test_pl, y_train_pl, y_test_pl = _load_data()
    X_train = X_train_pl.to_numpy()
    X_test = X_test_pl.to_numpy()
    y_train = y_train_pl.to_numpy().flatten()
    y_test = y_test_pl.to_numpy().flatten()
    feature_names = X_train_pl.columns

    # Configure Model
    model_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    if args.tune:
        best_params = _tune_model(X_train, y_train)
        model_params.update(best_params)

    model = XGBClassifier(**model_params)

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    logging.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    if args.cv_only:
        logging.info("CV Only mode. Exiting.")
        return

    # Train Final Model
    model.fit(X_train, y_train)
    logging.info("Model Training Completed")

    # Save Model
    joblib.dump(model, f"{OUTPUT_DIR}{MODEL_FILE}")
    logging.info(f"Saved model to {OUTPUT_DIR}{MODEL_FILE}")

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info("Classification Report:\n" + report)

    with open(f"{OUTPUT_DIR}{REPORT_FILE}", "w") as f:
        f.write(report)

    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
    logging.info(f"Test Metrics: {json.dumps(scores, indent=2)}")

    # Plots
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curves(model, X_test, y_test)
    _plot_feature_importance(model, feature_names)

    logging.info("Model Phase Completed")


if __name__ == "__main__":
    main()
