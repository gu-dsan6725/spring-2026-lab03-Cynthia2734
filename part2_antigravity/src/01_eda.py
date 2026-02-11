import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Constants
OUTPUT_DIR: str = "part2_antigravity/output/"
DIST_PLOT: str = "distributions.png"
CORR_PLOT: str = "correlation_matrix.png"
CLASS_PLOT: str = "class_distribution.png"
BOX_PLOT: str = "outlier_boxplots.png"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _setup_plotting_style() -> None:
    """Set up the plotting style/theme."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)


def _plot_distributions(
    df: pl.DataFrame,
) -> None:
    """Generate and save distribution plots for all features."""
    features = [col for col in df.columns if col != "target"]
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {feature}")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}{DIST_PLOT}"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved distribution plot to {output_path}")


def _plot_correlation(
    df: pl.DataFrame,
) -> None:
    """Generate and save feature correlation heatmap."""
    plt.figure(figsize=(12, 10))
    # Convert to pandas for correlation calculation and plotting
    corr = df.to_pandas().corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}{CORR_PLOT}"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved correlation plot to {output_path}")


def _plot_class_distribution(
    df: pl.DataFrame,
) -> None:
    """Generate and save class distribution bar chart."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df["target"].to_pandas())
    plt.title("Class Distribution")
    output_path = f"{OUTPUT_DIR}{CLASS_PLOT}"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved class distribution plot to {output_path}")


def _plot_outliers(
    df: pl.DataFrame,
) -> None:
    """Generate and save boxplots for outlier detection."""
    features = [col for col in df.columns if col != "target"]
    plt.figure(figsize=(15, 8))
    # Melt for boxplot
    df_melted = df.select(features).to_pandas().melt(var_name="Feature", value_name="Value")
    sns.boxplot(x="Feature", y="Value", data=df_melted)
    plt.xticks(rotation=45)
    plt.title("Outlier Detection Boxplots")
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}{BOX_PLOT}"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved outlier boxplots to {output_path}")


def main() -> None:
    """Execute the EDA pipeline."""
    logging.info("Starting EDA Phase")

    # Ensure output directory exists (redundant if main script does it, but safe)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_wine()
    # Create Polars DataFrame directly from numpy array
    df = pl.DataFrame(data.data, schema=data.feature_names)
    df = df.with_columns(pl.Series("target", data.target))

    logging.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")

    # Statistics
    stats = df.describe()
    logging.info("Dataset Statistics:\n" + str(stats))

    # Missing values
    missing = df.null_count()
    logging.info("Missing Values:\n" + str(missing))

    # Check duplicates
    n_duplicates = df.is_duplicated().sum()
    logging.info(f"Number of duplicate rows: {n_duplicates}")

    # Plots
    _setup_plotting_style()
    _plot_distributions(
        df=df,
    )
    _plot_correlation(
        df=df,
    )
    _plot_class_distribution(
        df=df,
    )
    _plot_outliers(
        df=df,
    )

    logging.info("EDA Phase Completed")


if __name__ == "__main__":
    main()
