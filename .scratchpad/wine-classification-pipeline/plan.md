## Plan: Wine Classification Pipeline (v2 — updated per user review)

### Objective

Build a complete Wine classification pipeline using `sklearn.datasets.load_wine`. The pipeline covers exploratory data analysis, feature engineering, XGBoost classification with cross-validation, and a comprehensive evaluation report. Scripts live in `part1_claude_code/src/`, output artifacts in `output/`.

### Steps

1. **Create `part1_claude_code/src/01_eda.py`**
   - Load Wine dataset via `load_wine`, convert to polars DataFrame (13 features + `target` column with class names mapped)
   - Compute summary statistics (mean, median, std, min, max) per feature
   - Check for missing values
   - Plot feature distributions (histogram grid) → `output/distributions.png`
   - Plot correlation matrix heatmap → `output/correlation_matrix.png`
   - Plot class distribution bar chart → `output/class_distribution.png`
   - Identify outliers via IQR method
   - **`--debug` CLI flag** to set logging level to DEBUG
   - Public function: `run_eda()`
   - Dependencies: none

2. **Create `part1_claude_code/src/02_feature_engineering.py`**
   - Load Wine dataset independently
   - Create derived features:
     - `FlavanoidsToNonflavRatio` = flavanoids / nonflavanoid_phenols
     - `ColorIntensityHue` = color_intensity * hue
     - `TotalPhenols` = total_phenols + flavanoids
   - Replace infinite/NaN values with column medians
   - Scale features with `StandardScaler` (exclude target)
   - Stratified train/test split (80/20, `random_state=42`)
   - Save parquet files: `output/{x_train,x_test,y_train,y_test}.parquet`
   - **`--debug` CLI flag** to set logging level to DEBUG
   - Public function: `run_feature_engineering()`
   - Dependencies: none

3. **Create `part1_claude_code/src/03_xgboost_model.py`**
   - Load train/test splits from parquet files
   - Train `XGBClassifier` with defaults: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `objective="multi:softprob"`, `num_class=3`
   - Run 5-fold stratified cross-validation (`scoring="accuracy"`)
   - **Enhanced `--tune` flag**: `RandomizedSearchCV` with **20 iterations** over broad search space:
     - `max_depth`: [3, 4, 5, 6, 7, 8, 9, 10]
     - `learning_rate`: [0.01, 0.05, 0.1, 0.2, 0.3]
     - `n_estimators`: [100, 200, 300, 400, 500]
     - `min_child_weight`: [1, 3, 5, 7]
     - `subsample`: [0.6, 0.7, 0.8, 0.9, 1.0]
     - `colsample_bytree`: [0.6, 0.7, 0.8, 0.9, 1.0]
   - **Save tuning results** → `output/tuning_results.json` with: best parameters, best CV score, all tried combinations with scores, tuning time
   - Compute test-set metrics: accuracy, precision/recall/F1 (macro) **plus per-class precision, recall, F1**
   - **Generate per-class ROC curves with AUC scores** → `output/roc_curves.png`
   - **Save classification report** → `output/classification_report.txt`
   - Generate confusion matrix heatmap → `output/confusion_matrix.png`
   - Generate feature importance bar chart → `output/feature_importance.png`
   - Save model → `output/xgboost_model.joblib`
   - **Enhanced `evaluation_report.md`** including:
     - Hyperparameter tuning section (if --tune was used) with tuning time
     - Per-class precision, recall, F1 breakdown table
     - Confusion matrix description and reference to `confusion_matrix.png`
   - CLI flags: `--tune`, `--cv-only`, **`--debug`**
   - Public function: `run_training_and_evaluation(tune, cv_only)`
   - Dependencies: Step 2 parquet files

4. **Create `part1_claude_code/src/04_generate_report.py`**
   - Load model from `output/xgboost_model.joblib`
   - Load test data, re-compute classification metrics
   - Extract top-5 feature importances
   - Gather dataset info (sample counts, feature count, class names)
   - Build markdown report with sections: Executive Summary, Dataset Overview, Model Configuration, Performance Metrics, Confusion Matrix reference, Feature Importance (top 5), Recommendations
   - Save → `output/full_report.md`
   - **`--debug` CLI flag** to set logging level to DEBUG
   - Dependencies: Step 3 artifacts

5. **Create `tests/test_wine_xgboost_model.py`**
   - Test `_compute_metrics` with perfect and imperfect predictions
   - Test `_run_cross_validation` returns expected keys and fold count
   - Test `_run_hyperparameter_tuning` returns model + search (monkeypatched small n_iter)
   - Test `_write_evaluation_report` produces valid markdown
   - Fixtures: sample data, trained model, sample metrics
   - Dependencies: Step 3 module

6. **Lint, compile, and test**
   - `uv run ruff check --fix` on all four src files
   - `uv run python -m py_compile` on all four src files
   - `uv run pytest tests/test_wine_xgboost_model.py -v`
   - Dependencies: Steps 1-5

7. **Run the full pipeline end-to-end**
   - Execute in order: 01_eda → 02_feature_engineering → 03_xgboost_model --cv-only → 04_generate_report
   - Verify all output artifacts exist
   - Dependencies: Step 6

### Technical Decisions

- **polars** for all data manipulation (no pandas, per CLAUDE.md)
- **XGBClassifier** (not Regressor) — multi-class classification with 3 wine classes
- **Classification metrics** (accuracy, precision, recall, F1, confusion matrix) replace the demo's regression metrics (RMSE, MAE, R2)
- **Stratified splits and CV** to preserve class proportions across folds
- **Both macro-averaged and per-class** precision/recall/F1 for thorough reporting
- **Per-class ROC curves with AUC** for visual discrimination analysis
- **Confusion matrix heatmap** replaces the demo's residual plots
- **StandardScaler** included for pipeline consistency with the demo, though XGBoost doesn't strictly require it
- **4-file structure** mirrors the demo pattern for consistency
- **`--debug` flag on all scripts** for consistent developer experience

### Testing Strategy

- **Unit tests**: metric computation, CV structure, tuning output, report markdown content
- **Static analysis**: ruff check + py_compile on every file
- **Integration**: full pipeline run, verify all output files exist in `output/`

### Expected Output

**Source files:**
- `part1_claude_code/src/01_eda.py`
- `part1_claude_code/src/02_feature_engineering.py`
- `part1_claude_code/src/03_xgboost_model.py`
- `part1_claude_code/src/04_generate_report.py`
- `tests/test_wine_xgboost_model.py`

**Artifacts in `output/`:**
- `distributions.png` — feature histograms
- `correlation_matrix.png` — correlation heatmap
- `class_distribution.png` — target class counts
- `x_train.parquet`, `x_test.parquet`, `y_train.parquet`, `y_test.parquet`
- `confusion_matrix.png` — classification confusion matrix
- `feature_importance.png` — XGBoost feature importances
- `roc_curves.png` — per-class ROC curves with AUC
- `classification_report.txt` — sklearn classification report
- `tuning_results.json` — hyperparameter search results (when --tune used)
- `xgboost_model.joblib` — serialized model
- `evaluation_report.md` — metrics summary (with per-class breakdown and tuning section)
- `full_report.md` — comprehensive report
