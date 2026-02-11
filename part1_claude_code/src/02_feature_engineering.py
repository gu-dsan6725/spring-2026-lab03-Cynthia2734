"""Feature engineering for the Wine dataset.

Creates derived features, scales numeric columns, and splits
the data into training and test sets with stratification.
"""

import argparse
import logging
import time
from pathlib import Path

import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "target"


def _ensure_output_dir(
    output_dir: str,
) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_dataset() -> pl.DataFrame:
    """Load the Wine dataset and return as a polars DataFrame."""
    wine = load_wine()
    feature_names = list(wine.feature_names)
    data = wine.data
    target = wine.target

    df = pl.DataFrame({name: data[:, i] for i, name in enumerate(feature_names)})
    df = df.with_columns(pl.Series(TARGET_COLUMN, target.astype(int)))

    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def _create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create new features from existing columns."""
    df = df.with_columns(
        [
            (pl.col("flavanoids") / pl.col("nonflavanoid_phenols")).alias(
                "FlavanoidsToNonflavRatio"
            ),
            (pl.col("color_intensity") * pl.col("hue")).alias("ColorIntensityHue"),
            (pl.col("total_phenols") + pl.col("flavanoids")).alias("TotalPhenols"),
        ]
    )

    logger.info(f"Created 3 derived features. New shape: {df.shape}")
    logger.info(f"Columns: {df.columns}")
    return df


def _handle_infinite_values(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace infinite values with column medians."""
    for col in df.columns:
        if df[col].dtype in [pl.Float64, pl.Float32]:
            median_val = float(df[col].filter(df[col].is_finite()).median())

            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(median_val)
                .otherwise(pl.col(col))
                .alias(col)
            )

    logger.info("Replaced infinite values with column medians")
    return df


def _scale_features(
    df: pl.DataFrame,
    target_column: str,
) -> tuple[pl.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler."""
    feature_columns = [c for c in df.columns if c != target_column]

    scaler = StandardScaler()
    feature_values = df.select(feature_columns).to_numpy()
    scaled_values = scaler.fit_transform(feature_values)

    scaled_df = pl.DataFrame({col: scaled_values[:, i] for i, col in enumerate(feature_columns)})
    scaled_df = scaled_df.with_columns(df[target_column])

    logger.info(f"Scaled {len(feature_columns)} features using StandardScaler")
    return scaled_df, scaler


def _split_data(
    df: pl.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data into training and test sets with stratification."""
    feature_columns = [c for c in df.columns if c != target_column]

    x_data = df.select(feature_columns).to_numpy()
    y_data = df[target_column].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )

    x_train_df = pl.DataFrame({col: x_train[:, i] for i, col in enumerate(feature_columns)})
    x_test_df = pl.DataFrame({col: x_test[:, i] for i, col in enumerate(feature_columns)})
    y_train_df = pl.DataFrame({target_column: y_train})
    y_test_df = pl.DataFrame({target_column: y_test})

    logger.info(f"Train set: {x_train_df.shape[0]} samples")
    logger.info(f"Test set: {x_test_df.shape[0]} samples")

    return x_train_df, x_test_df, y_train_df, y_test_df


def _save_splits(
    x_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    output_path: Path,
) -> None:
    """Save train/test splits to parquet files."""
    x_train.write_parquet(output_path / "x_train.parquet")
    x_test.write_parquet(output_path / "x_test.parquet")
    y_train.write_parquet(output_path / "y_train.parquet")
    y_test.write_parquet(output_path / "y_test.parquet")

    logger.info(f"Saved train/test splits to {output_path}")


def run_feature_engineering() -> None:
    """Run the full feature engineering pipeline."""
    start_time = time.time()
    logger.info("Starting feature engineering...")

    output_path = _ensure_output_dir(OUTPUT_DIR)

    df = _load_dataset()
    df = _create_derived_features(df)
    df = _handle_infinite_values(df)
    scaled_df, _scaler = _scale_features(df, TARGET_COLUMN)

    x_train, x_test, y_train, y_test = _split_data(
        scaled_df,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
    )

    _save_splits(x_train, x_test, y_train, y_test, output_path)

    elapsed = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed:.1f} seconds")


def main() -> None:
    """Parse CLI arguments and run the feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Run feature engineering on the Wine dataset",
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

    run_feature_engineering()


if __name__ == "__main__":
    main()
