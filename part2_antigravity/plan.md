# Wine Classification Pipeline Plan

## 1. Project Overview
This project builds a reproducible Machine Learning pipeline for classifying wine varieties using the Wine dataset from sklearn. It leverages modern Python tooling (`uv`, `ruff`, `polars`) and implements a full workflow from Exploratory Data Analysis (EDA) to Model Evaluation with XGBoost.

## 2. File Structure
```
part2_antigravity/
├── plan.md                 # This implementation plan
├── src/
│   ├── __init__.py
│   ├── 01_eda.py           # Phase 1: EDA
│   ├── 02_feature_engineering.py # Phase 2: Feature Engineering
│   ├── 03_xgboost_model.py # Phase 3: Model Training
│   └── 04_generate_report.py # Phase 4: Evaluation Report
└── output/                 # All generated artifacts (images, parquet, models)
    ├── distributions.png
    ├── correlation_matrix.png
    ├── ... (other outputs)
    └── evaluation_report.md
```

## 3. Detailed Implementation Steps

### Phase 1: EDA (`src/01_eda.py`)
*   **Goal**: Understand dataset characteristics and data quality.
*   **Input**: `sklearn.datasets.load_wine()`
*   **Output**: Plots in `output/`
*   **Steps**:
    1.  Load wine data using `sklearn`.
    2.  Convert to `polars.DataFrame`.
    3.  Compute summary statistics (mean, median, std, min, max).
    4.  Check for and log missing values/duplicates.
    5.  Generate and save:
        *   `distributions.png`: Histograms for all 13 features.
        *   `correlation_matrix.png`: Heatmap of feature correlations.
        *   `class_distribution.png`: Bar chart of target classes.
        *   `outlier_boxplots.png`: Boxplots for all features.

### Phase 2: Feature Engineering (`src/02_feature_engineering.py`)
*   **Goal**: Prepare data for modeling.
*   **Input**: `sklearn.datasets.load_wine()` (raw data)
*   **Output**: Parquet files in `output/`
*   **Steps**:
    1.  Load data and convert to Polars.
    2.  Create derived features:
        *   `alcohol_to_phenols_ratio` = `alcohol` / `total_phenols`
        *   `flavor_intensity` = `flavanoids` * `color_intensity`
        *   `phenol_richness` = `total_phenols` + `flavanoids`
    3.  Impute infinite/NaN values with median (if any).
    4.  Apply `StandardScaler` to all features.
    5.  Perform stratified train/test split (80/20, `random_state=42`).
    6.  Save sets to `output/x_train.parquet`, `x_test.parquet`, `y_train.parquet`, `y_test.parquet`.

### Phase 3: Model Training (`src/03_xgboost_model.py`)
*   **Goal**: Train and maximize model performance.
*   **Input**: Parquet files from Phase 2.
*   **Output**: Model artifacts and performance plots.
*   **CLI Flags**: `--tune`, `--debug`, `--cv-only`
*   **Steps**:
    1.  Load Parquet data (Polars).
    2.  Initialize `XGBClassifier` (`objective='multi:softprob'`, `num_class=3`).
    3.  If `--tune` is set:
        *   Run `RandomizedSearchCV` (20 iterations, 5-fold stratified CV).
        *   Search space: `max_depth` (3-10), `learning_rate` (0.01-0.3), `n_estimators` (100-500), `min_child_weight` (1-7), `subsample` (0.6-1.0), `colsample_bytree` (0.6-1.0).
        *   Save `tuning_results.json`.
    4.  Perform 5-fold stratified cross-validation on the (best) model.
    5.  Train on full training set.
    6.  Evaluate on train/test sets (Accuracy, Precision, Recall, F1 - macro & per-class).
    7.  Generate artifacts:
        *   `confusion_matrix.png`
        *   `roc_curves.png`
        *   `feature_importance.png`
        *   `xgboost_model.joblib`
        *   `classification_report.txt`

### Phase 4: Evaluation Report (`src/04_generate_report.py`)
*   **Goal**: Summarize findings in a readable format.
*   **Input**: Model and Test data.
*   **Output**: `output/evaluation_report.md`
*   **Steps**:
    1.  Load model and test data.
    2.  Re-calculate metrics.
    3.  Generate Markdown report containing:
        *   Executive Summary
        *   Dataset Overview
        *   Model Configuration
        *   Performance Metrics table
        *   Embedded images (Confusion Matrix, ROC, etc.)
        *   Top 5 Feature Importance
        *   Recommendations.

## 4. Technical Decisions
*   **Polars over Pandas**: Chosen for performance, type strictness, and adherence to user constraints (NEVER pandas).
*   **XGBClassifier**: State-of-the-art gradient boosting tailored for structured data; handles non-linear relationships well.
*   **Stratified Sampling**: Essential for classification to maintain class distribution in splits, preventing bias.
*   **Parquet Format**: Efficient columnar storage, preserves schema/types better than CSV, faster to read/write.

## 5. Coding Standards Compliance
This plan strictly adheres to `.gemini/GEMINI.md` and `.agent/rules/code-style-guide.md`:

*   **Language & Tools**:
    *   [x] Python 3.11+
    *   [x] Package Manager: `uv` (NEVER pip)
    *   [x] Data Manipulation: `polars` (NEVER pandas)
    *   [x] Linting/Formatting: `ruff`
    *   [x] Testing: `pytest`

*   **Code Organization**:
    *   [x] **Private functions** (`_func`) at the **TOP** of the file.
    *   [x] **Public functions** follow private functions.
    *   [x] **Two blank lines** between function definitions.
    *   [x] **Function Size**: Max 30-50 lines.
    *   [x] **Imports**: Multi-line, grouped (stdlib, third-party, local).

*   **Type Annotations**:
    *   [x] **ALL** parameters typed.
    *   [x] One parameter per line.
    *   [x] Return types specified.

*   **Constants**:
    *   [x] Defined at module level (top).
    *   [x] Type annotated.
    *   [x] No hard-coded magic numbers in functions.

*   **Logging**:
    *   [x] Exact configuration used:
        ```python
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
        )
        ```
    *   [x] `logging.info()` for progress.
    *   [x] Dictionaries formatted with `json.dumps(data, indent=2, default=str)`.

*   **Quality Checks (MANDATORY)**:
    1.  `uv run ruff check --fix <filename>`
    2.  `uv run ruff format <filename>`
    3.  `uv run python -m py_compile <filename>`

*   **Output**:
    *   [x] All plots saved to `output/`.
    *   [x] Data saved as `.parquet`.
    *   [x] Models saved as `.joblib`.

## 6. CLI Arguments
*   `--tune`: (Phase 3) Trigger RandomizedSearchCV hyperparameter tuning.
*   `--debug`: Set logging level to DEBUG.
*   `--cv-only`: (Phase 3) Run Cross-Validation without final model training/saving.

## 7. Expected Output Files
1.  `distributions.png`
2.  `correlation_matrix.png`
3.  `class_distribution.png`
4.  `outlier_boxplots.png`
5.  `x_train.parquet`
6.  `x_test.parquet`
7.  `y_train.parquet`
8.  `y_test.parquet`
9.  `confusion_matrix.png`
10. `roc_curves.png`
11. `feature_importance.png`
12. `xgboost_model.joblib`
13. `tuning_results.json` (if --tune used)
14. `classification_report.txt`
15. `evaluation_report.md`

## 8. Testing Strategy
*   **Unit Tests (`tests/`)**:
    *   Test feature engineering calculations (e.g., check `alcohol_to_phenols_ratio` calculation).
    *   Test data cleaning functions (e.g., check infinite value replacement).
*   **Integration Tests**:
    *   Run the full pipeline (`01` -> `02` -> `03` -> `04`) on a small subset or synthetic data to ensure valid artifact generation.

## 9. Success Criteria
*   Model Accuracy > 90% on test set.
*   All 4 scripts execute without errors.
*   All 15 expected artifacts are generated in `output/`.
*   Code passes all `ruff` checks and compiles successfully.

## 10. Verification Checklist
*   [ ] References `.gemini/GEMINI.md` explicitly.
*   [ ] References `.agent/rules/code-style-guide.md` explicitly.
*   [ ] Specifies exact logging format with code snippet.
*   [ ] Lists all 3 quality check commands.
*   [ ] Explains private function placement at top of file.
*   [ ] Specifies function size limit (30-50 lines).
*   [ ] Shows multi-line import example.
*   [ ] Lists all constants with type annotations.
*   [ ] Documents all CLI flags.
*   [ ] Lists all 15 expected output files.
