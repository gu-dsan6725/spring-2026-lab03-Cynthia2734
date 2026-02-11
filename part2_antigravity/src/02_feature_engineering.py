import logging
from pathlib import Path
from typing import Tuple

import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
OUTPUT_DIR: str = "part2_antigravity/output/"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _load_data() -> pl.DataFrame:
    """Load wine dataset and return as Polars DataFrame."""
    data = load_wine()
    # Create Polars DataFrame
    df = pl.DataFrame(data.data, schema=data.feature_names)
    df = df.with_columns(pl.Series("target", data.target))
    return df


def _create_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create derived features."""
    return df.with_columns(
        (pl.col("alcohol") / pl.col("total_phenols")).alias("alcohol_to_phenols_ratio"),
        (pl.col("flavanoids") * pl.col("color_intensity")).alias("flavor_intensity"),
        (pl.col("total_phenols") + pl.col("flavanoids")).alias("phenol_richness"),
    )


def _handle_missing_inf(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Handle missing and infinite values by replacing with median."""
    features = [col for col in df.columns if col != "target"]

    # Step 1: Replace Inf with Null
    replace_inf_exprs = [
        pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)
        for col in features
        if df[col].dtype in [pl.Float32, pl.Float64]
    ]

    if replace_inf_exprs:
        df = df.with_columns(replace_inf_exprs)

    # Step 2: Fill Null with Median
    fill_null_exprs = [
        pl.col(col).fill_null(pl.col(col).median()).alias(col)
        for col in features
        if df[col].dtype in [pl.Float32, pl.Float64]
    ]

    if fill_null_exprs:
        df = df.with_columns(fill_null_exprs)

    return df


def _scale_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply StandardScaler and split data."""
    features = [col for col in df.columns if col != "target"]
    X = df.select(features).to_numpy()
    y = df.select("target").to_numpy().flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to Polars
    # Use features list for columns since derived features are included
    X_train_pl = pl.DataFrame(X_train_scaled, schema=features)
    X_test_pl = pl.DataFrame(X_test_scaled, schema=features)
    y_train_pl = pl.DataFrame({"target": y_train})
    y_test_pl = pl.DataFrame({"target": y_test})

    return X_train_pl, X_test_pl, y_train_pl, y_test_pl


def main() -> None:
    """Execute Feature Engineering Phase."""
    logging.info("Starting Feature Engineering Phase")

    # Load
    df = _load_data()
    logging.info(f"Loaded data: {df.shape}")

    # Feature Engineering
    df = _create_features(df=df)
    logging.info(f"Created derived features: {df.shape}")

    # Clean
    df = _handle_missing_inf(df=df)
    logging.info("Handled missing/infinite values")

    # Split and Scale
    X_train, X_test, y_train, y_test = _scale_features(df=df)
    logging.info(f"Split and Scaled Data. Train: {X_train.shape}, Test: {X_test.shape}")

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    X_train.write_parquet(f"{OUTPUT_DIR}x_train.parquet")
    X_test.write_parquet(f"{OUTPUT_DIR}x_test.parquet")
    y_train.write_parquet(f"{OUTPUT_DIR}y_train.parquet")
    y_test.write_parquet(f"{OUTPUT_DIR}y_test.parquet")

    logging.info(f"Saved parquet files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
