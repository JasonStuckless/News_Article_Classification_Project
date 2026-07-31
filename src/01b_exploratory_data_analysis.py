"""
01b_exploratory_data_analysis.py

Perform exploratory data analysis on the prepared CNN News Articles dataset.

This stage adapts the exploratory methods developed in the project notebook and
integrates them into the reproducible project pipeline. It examines dataset
quality, class distributions, publication dates, text lengths, and the prepared
train/test split. Important tabular results are retained as CSV files and are
also converted into figures where visualization improves interpretation.

Inputs:
- data/raw/cnn_original.csv
- data/interim/train_original.csv
- data/interim/test_original.csv

Outputs:
- results/exploratory_data_analysis/tables/*.csv
- results/exploratory_data_analysis/figures/*.png
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any, Iterable
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Allow this script to import modules from src/utils when run directly.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.io import read_csv, write_csv
from utils.logging import get_logger


logger = get_logger(
    name=__name__,
    log_directory=PROJECT_ROOT / "logs",
    log_file="project.log",
)


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

ID_COLUMN = "Index"
DATE_COLUMN = "Date published"
CATEGORY_COLUMN = "Category"
SECTION_COLUMN = "Section"
URL_COLUMN = "Url"

TEXT_COLUMNS: tuple[str, ...] = (
    "Headline",
    "Description",
    "Article text",
)

ADDITIONAL_TEXT_COLUMNS: tuple[str, ...] = (
    "Keywords",
    "Second headline",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "Author",
    DATE_COLUMN,
    CATEGORY_COLUMN,
    SECTION_COLUMN,
    URL_COLUMN,
    *TEXT_COLUMNS,
    *ADDITIONAL_TEXT_COLUMNS,
)

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

DEFAULT_OUTPUT_SUBDIRECTORY = "exploratory_data_analysis"
FIGURE_DPI = 300


class ExploratoryDataAnalysisError(RuntimeError):
    """Raised when exploratory analysis cannot be completed safely."""


# ---------------------------------------------------------------------------
# Configuration and paths
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the configuration required by this stage."""
    if not config_path.exists():
        raise ExploratoryDataAnalysisError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ExploratoryDataAnalysisError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise ExploratoryDataAnalysisError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    output_config = config.get("output")

    if not isinstance(dataset_config, dict):
        raise ExploratoryDataAnalysisError(
            "The configuration must contain a 'dataset' section."
        )

    if not isinstance(output_config, dict):
        raise ExploratoryDataAnalysisError(
            "The configuration must contain an 'output' section."
        )

    required_dataset_settings = {
        "original",
        "train_file",
        "test_file",
    }
    missing_settings = required_dataset_settings - set(dataset_config)

    if missing_settings:
        missing = ", ".join(sorted(missing_settings))
        raise ExploratoryDataAnalysisError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    if "results" not in output_config:
        raise ExploratoryDataAnalysisError(
            "Missing output.results setting in config.yaml."
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_output_directories(results_root: Path) -> tuple[Path, Path]:
    """Create and return the EDA table and figure directories."""
    stage_root = results_root / DEFAULT_OUTPUT_SUBDIRECTORY
    table_directory = stage_root / "tables"
    figure_directory = stage_root / "figures"

    table_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    return table_directory, figure_directory


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------

def validate_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    require_all_columns: bool = True,
) -> None:
    """Validate the minimum structure needed for exploratory analysis."""
    if dataframe.empty:
        raise ExploratoryDataAnalysisError(
            f"The {dataset_name} contains no rows."
        )

    required = set(REQUIRED_COLUMNS if require_all_columns else (
        ID_COLUMN,
        CATEGORY_COLUMN,
        SECTION_COLUMN,
        *TEXT_COLUMNS,
    ))
    missing_columns = required - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ExploratoryDataAnalysisError(
            f"The {dataset_name} is missing required columns: {missing}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise ExploratoryDataAnalysisError(
            f"The {dataset_name} contains missing {ID_COLUMN} values."
        )


def normalize_label_series(series: pd.Series) -> pd.Series:
    """Return labels normalized for stable analysis and plotting."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def prepare_analysis_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a non-destructive normalized copy for exploratory analysis."""
    prepared = dataframe.copy()

    for column in (*TEXT_COLUMNS, *ADDITIONAL_TEXT_COLUMNS):
        if column in prepared.columns:
            prepared[column] = (
                prepared[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    prepared[CATEGORY_COLUMN] = normalize_label_series(
        prepared[CATEGORY_COLUMN]
    )
    prepared[SECTION_COLUMN] = normalize_label_series(
        prepared[SECTION_COLUMN]
    )
    prepared[DATE_COLUMN] = pd.to_datetime(
        prepared[DATE_COLUMN],
        errors="coerce",
    )

    return prepared


# ---------------------------------------------------------------------------
# Text processing adapted from the project notebook
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """Normalize Unicode, markup, URLs, and repeated whitespace."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_headline(value: Any) -> str:
    """Apply the notebook's headline cleanup without altering source data."""
    text = normalize_text(value)
    return re.sub(
        r"\s*-\s*CNN\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def clean_article_text(value: Any) -> str:
    """Remove a leading CNN dateline pattern used in the notebook."""
    text = normalize_text(value)
    return re.sub(
        r"^[A-Z][A-Za-z .'-]+\s*\(CNN\)\s*[-—:]?\s*",
        "",
        text,
    ).strip()


def combine_baseline_text(dataframe: pd.DataFrame) -> pd.Series:
    """Combine the same three text fields used by the TF-IDF pipeline."""
    return (
        dataframe.loc[:, list(TEXT_COLUMNS)]
        .fillna("")
        .astype(str)
        .apply(
            lambda row: "\n\n".join(
                value.strip() for value in row if value.strip()
            ),
            axis=1,
        )
    )


def create_enhanced_text(dataframe: pd.DataFrame) -> pd.Series:
    """Create the notebook's enhanced text representation for comparison."""
    components = pd.DataFrame(
        {
            "headline": dataframe["Headline"].apply(clean_headline),
            "description": dataframe["Description"].apply(normalize_text),
            "article": dataframe["Article text"].apply(clean_article_text),
        },
        index=dataframe.index,
    )

    return components.apply(
        lambda row: ". ".join(value for value in row if value),
        axis=1,
    )


def count_words(value: Any) -> int:
    """Count word-like tokens using the notebook's token definition."""
    text = "" if pd.isna(value) else str(value)
    return len(WORD_PATTERN.findall(text))


def add_text_length_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add word and character counts for each core text component."""
    featured = dataframe.copy()

    for column in TEXT_COLUMNS:
        featured[f"{column}_words"] = featured[column].map(count_words)
        featured[f"{column}_characters"] = (
            featured[column].fillna("").astype(str).str.len()
        )

    featured["baseline_text"] = combine_baseline_text(featured)
    featured["enhanced_text"] = create_enhanced_text(featured)
    featured["baseline_text_characters"] = (
        featured["baseline_text"].str.len()
    )
    featured["enhanced_text_characters"] = (
        featured["enhanced_text"].str.len()
    )

    return featured


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def build_column_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize data types, missingness, and cardinality by column."""
    profile = pd.DataFrame(
        {
            "Column": dataframe.columns,
            "Data type": dataframe.dtypes.astype(str).values,
            "Missing values": dataframe.isna().sum().values,
            "Missing percentage": (
                dataframe.isna().mean().mul(100).round(2).values
            ),
            "Unique values": dataframe.nunique(dropna=True).values,
        }
    )

    return profile.sort_values(
        ["Missing values", "Column"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_quality_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Recreate and extend the notebook's dataset-quality summary."""
    date_values = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")

    measures = {
        "Rows": len(dataframe),
        "Columns": len(dataframe.columns),
        "Exact duplicate rows": int(dataframe.duplicated().sum()),
        "Duplicate Index values": int(
            dataframe[ID_COLUMN].duplicated().sum()
        ),
        "Duplicate URLs": int(dataframe[URL_COLUMN].duplicated().sum()),
        "Missing values": int(dataframe.isna().sum().sum()),
        "Unparseable publication dates": int(date_values.isna().sum()),
        "Missing Category labels": int(
            dataframe[CATEGORY_COLUMN].isna().sum()
        ),
        "Missing Section labels": int(
            dataframe[SECTION_COLUMN].isna().sum()
        ),
        "Empty headlines": int(
            dataframe["Headline"].fillna("").astype(str).str.strip().eq("").sum()
        ),
        "Empty descriptions": int(
            dataframe["Description"].fillna("").astype(str).str.strip().eq("").sum()
        ),
        "Empty article texts": int(
            dataframe["Article text"].fillna("").astype(str).str.strip().eq("").sum()
        ),
    }

    return pd.DataFrame(
        {"Measure": list(measures), "Value": list(measures.values())}
    )


def build_text_feature_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the word-count distribution of each core text field."""
    records: list[dict[str, Any]] = []

    for column in TEXT_COLUMNS:
        values = dataframe[f"{column}_words"]
        records.append(
            {
                "Feature": column,
                "Mean words": round(float(values.mean()), 1),
                "Median words": round(float(values.median()), 1),
                "Standard deviation": round(float(values.std()), 1),
                "90th percentile": round(float(values.quantile(0.90)), 1),
                "95th percentile": round(float(values.quantile(0.95)), 1),
                "Maximum words": int(values.max()),
                "Zero-length values": int(values.eq(0).sum()),
            }
        )

    return pd.DataFrame(records)


def build_category_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Recreate the notebook's category-level article-length table."""
    return (
        dataframe.groupby(CATEGORY_COLUMN, dropna=False)
        .agg(
            Articles=(CATEGORY_COLUMN, "size"),
            Median_headline_words=("Headline_words", "median"),
            Median_description_words=("Description_words", "median"),
            Median_article_words=("Article text_words", "median"),
            Mean_article_words=("Article text_words", "mean"),
        )
        .round(1)
        .sort_values("Articles", ascending=False)
        .reset_index()
    )


def build_processing_comparison(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline and enhanced text lengths from the notebook."""
    return pd.DataFrame(
        {
            "Representation": ["Baseline", "Enhanced"],
            "Mean characters": [
                dataframe["baseline_text_characters"].mean(),
                dataframe["enhanced_text_characters"].mean(),
            ],
            "Median characters": [
                dataframe["baseline_text_characters"].median(),
                dataframe["enhanced_text_characters"].median(),
            ],
            "90th percentile": [
                dataframe["baseline_text_characters"].quantile(0.90),
                dataframe["enhanced_text_characters"].quantile(0.90),
            ],
            "Maximum characters": [
                dataframe["baseline_text_characters"].max(),
                dataframe["enhanced_text_characters"].max(),
            ],
        }
    ).round(1)


def build_label_distribution(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Return counts and percentages for one label column."""
    distribution = (
        dataframe[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="Articles")
    )
    distribution["Percentage"] = (
        distribution["Articles"] / len(dataframe) * 100
    ).round(2)
    return distribution


def build_split_distribution(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Compare train and test label proportions."""
    train_counts = train_dataframe[column].value_counts().rename("Train articles")
    test_counts = test_dataframe[column].value_counts().rename("Test articles")

    comparison = pd.concat([train_counts, test_counts], axis=1).fillna(0)
    comparison["Train articles"] = comparison["Train articles"].astype(int)
    comparison["Test articles"] = comparison["Test articles"].astype(int)
    comparison["Train percentage"] = (
        comparison["Train articles"] / len(train_dataframe) * 100
    )
    comparison["Test percentage"] = (
        comparison["Test articles"] / len(test_dataframe) * 100
    )
    comparison["Percentage-point difference"] = (
        comparison["Train percentage"] - comparison["Test percentage"]
    ).abs()

    return comparison.round(2).reset_index(names=column)


def build_publication_year_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Count articles by parsed publication year."""
    dated = dataframe[dataframe[DATE_COLUMN].notna()].copy()

    if dated.empty:
        return pd.DataFrame(columns=["Publication year", "Articles"])

    return (
        dated[DATE_COLUMN]
        .dt.year
        .value_counts()
        .sort_index()
        .rename_axis("Publication year")
        .reset_index(name="Articles")
    )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save a figure consistently and release its resources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure to %s.", output_path)


def add_bar_labels(axis: plt.Axes, decimal_places: int = 0) -> None:
    """Add readable labels to bars when the plotted range permits."""
    formatting = f"%.{decimal_places}f"

    for container in axis.containers:
        try:
            axis.bar_label(
                container,
                fmt=formatting,
                padding=3,
                fontsize=8,
            )
        except (AttributeError, TypeError):
            continue


def plot_quality_issues(
    quality_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize quality issues while excluding structural row/column counts."""
    excluded = {"Rows", "Columns"}
    plot_data = quality_summary[
        ~quality_summary["Measure"].isin(excluded)
    ].copy()
    plot_data = plot_data.sort_values("Value", ascending=True)

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_data, x="Value", y="Measure", color="#4C78A8", ax=axis)
    axis.set_title("Dataset Quality Checks")
    axis.set_xlabel("Count")
    axis.set_ylabel("")
    add_bar_labels(axis)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_missing_values(
    column_profile: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize missing values for every source column."""
    plot_data = column_profile.sort_values(
        "Missing values",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_data,
        x="Missing values",
        y="Column",
        color="#4C78A8",
        ax=axis,
    )
    axis.set_title("Missing Values by Dataset Column")
    axis.set_xlabel("Missing values")
    axis.set_ylabel("")
    add_bar_labels(axis)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_label_distribution(
    distribution: pd.DataFrame,
    label_column: str,
    output_path: Path,
) -> None:
    """Plot class counts for Category or Section."""
    plot_data = distribution.sort_values("Articles", ascending=True)
    height = max(5.0, 0.33 * len(plot_data) + 1.5)

    figure, axis = plt.subplots(figsize=(10, height))
    sns.barplot(
        data=plot_data,
        x="Articles",
        y=label_column,
        color="#4C78A8",
        ax=axis,
    )
    axis.set_title(f"{label_column} Distribution")
    axis.set_xlabel("Articles")
    axis.set_ylabel("")
    add_bar_labels(axis)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_category_section_heatmap(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize the hierarchical relationship between Category and Section."""
    contingency = pd.crosstab(
        dataframe[SECTION_COLUMN],
        dataframe[CATEGORY_COLUMN],
    )

    figure_height = max(8.0, 0.32 * len(contingency) + 2.0)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    sns.heatmap(
        contingency,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.3,
        cbar_kws={"label": "Articles"},
        ax=axis,
    )
    axis.set_title("Section Distribution Within CNN Categories")
    axis.set_xlabel("Category")
    axis.set_ylabel("Section")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_text_word_distributions(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize word-count distributions for the three text components."""
    long_data = dataframe[
        [f"{column}_words" for column in TEXT_COLUMNS]
    ].rename(
        columns={f"{column}_words": column for column in TEXT_COLUMNS}
    ).melt(var_name="Text component", value_name="Words")

    # A log scale keeps headline and article distributions visible together.
    long_data["Words plus one"] = long_data["Words"] + 1

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=long_data,
        x="Text component",
        y="Words plus one",
        showfliers=False,
        color="#4C78A8",
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set_title("Word-Count Distributions by Text Component")
    axis.set_xlabel("")
    axis.set_ylabel("Words + 1 (log scale)")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_article_length_histogram(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the article-text word-count distribution."""
    values = dataframe["Article text_words"]
    upper_limit = float(values.quantile(0.99))
    plotted = values[values <= upper_limit]

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(plotted, bins=50, kde=True, color="#4C78A8", ax=axis)
    axis.axvline(
        values.median(),
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {values.median():.0f} words",
    )
    axis.set_title(
        "Article Text Length Distribution "
        "(Values Through the 99th Percentile)"
    )
    axis.set_xlabel("Article text words")
    axis.set_ylabel("Articles")
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_category_article_lengths(
    category_statistics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize typical article length by Category."""
    plot_data = category_statistics.sort_values(
        "Median_article_words",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_data,
        x="Median_article_words",
        y=CATEGORY_COLUMN,
        color="#4C78A8",
        ax=axis,
    )
    axis.set_title("Median Article Length by CNN Category")
    axis.set_xlabel("Median article-text words")
    axis.set_ylabel("")
    add_bar_labels(axis, decimal_places=1)
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_processing_comparison(
    processing_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize baseline and enhanced text lengths from the notebook."""
    plot_data = processing_comparison.melt(
        id_vars="Representation",
        value_vars=["Mean characters", "Median characters"],
        var_name="Statistic",
        value_name="Characters",
    )

    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=plot_data,
        x="Representation",
        y="Characters",
        hue="Statistic",
        palette="deep",
        ax=axis,
    )
    axis.set_title("Baseline and Enhanced Text Lengths")
    axis.set_xlabel("")
    axis.set_ylabel("Characters")
    add_bar_labels(axis, decimal_places=1)
    axis.legend(title="")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_split_comparison(
    split_distribution: pd.DataFrame,
    label_column: str,
    output_path: Path,
) -> None:
    """Compare train and test class percentages."""
    plot_data = split_distribution.melt(
        id_vars=label_column,
        value_vars=["Train percentage", "Test percentage"],
        var_name="Dataset",
        value_name="Percentage",
    )
    plot_data["Dataset"] = plot_data["Dataset"].str.replace(
        " percentage",
        "",
        regex=False,
    )

    height = max(5.0, 0.33 * len(split_distribution) + 1.5)
    figure, axis = plt.subplots(figsize=(10, height))
    sns.barplot(
        data=plot_data,
        x="Percentage",
        y=label_column,
        hue="Dataset",
        palette="deep",
        ax=axis,
    )
    axis.set_title(f"Train and Test {label_column} Distributions")
    axis.set_xlabel("Articles (%)")
    axis.set_ylabel("")
    axis.legend(title="")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_publication_years(
    year_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize the number of articles by publication year."""
    if year_distribution.empty:
        logger.warning(
            "Publication-year figure was not generated because no dates "
            "could be parsed."
        )
        return

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=year_distribution,
        x="Publication year",
        y="Articles",
        marker="o",
        color="#4C78A8",
        ax=axis,
    )
    axis.set_title("Articles by Publication Year")
    axis.set_xlabel("Publication year")
    axis.set_ylabel("Articles")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    save_figure(figure, output_path)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    include_index: bool = False,
) -> None:
    """Save a table using the project's CSV utility."""
    output = dataframe.copy()

    if include_index:
        output = output.reset_index()

    write_csv(output, output_path)
    logger.info("Saved analysis table to %s.", output_path)


def save_all_tables(
    tables: Iterable[tuple[str, pd.DataFrame]],
    table_directory: Path,
) -> None:
    """Save named analysis tables to CSV."""
    for filename, table in tables:
        save_table(table, table_directory / filename)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete exploratory analysis stage."""
    logger.info("Starting exploratory data analysis.")

    config = load_config(CONFIG_PATH)
    dataset_config = config["dataset"]
    output_config = config["output"]

    original_path = resolve_project_path(str(dataset_config["original"]))
    train_path = resolve_project_path(str(dataset_config["train_file"]))
    test_path = resolve_project_path(str(dataset_config["test_file"]))
    results_root = resolve_project_path(str(output_config["results"]))

    for input_name, input_path in (
        ("original dataset", original_path),
        ("training dataset", train_path),
        ("testing dataset", test_path),
    ):
        if not input_path.exists():
            raise ExploratoryDataAnalysisError(
                f"The {input_name} was not found: {input_path}"
            )

    table_directory, figure_directory = create_output_directories(
        results_root
    )

    logger.info("Loading original dataset from %s.", original_path)
    original_raw = read_csv(original_path)
    logger.info("Loading training dataset from %s.", train_path)
    train_raw = read_csv(train_path)
    logger.info("Loading testing dataset from %s.", test_path)
    test_raw = read_csv(test_path)

    validate_dataframe(
        original_raw,
        dataset_name="original dataset",
        require_all_columns=True,
    )
    validate_dataframe(
        train_raw,
        dataset_name="training dataset",
        require_all_columns=False,
    )
    validate_dataframe(
        test_raw,
        dataset_name="testing dataset",
        require_all_columns=False,
    )

    original = add_text_length_features(
        prepare_analysis_dataframe(original_raw)
    )
    train = prepare_analysis_dataframe(train_raw)
    test = prepare_analysis_dataframe(test_raw)

    column_profile = build_column_profile(original_raw)
    quality_summary = build_quality_summary(original_raw)
    feature_statistics = build_text_feature_statistics(original)
    category_statistics = build_category_statistics(original)
    processing_comparison = build_processing_comparison(original)
    category_distribution = build_label_distribution(
        original,
        CATEGORY_COLUMN,
    )
    section_distribution = build_label_distribution(
        original,
        SECTION_COLUMN,
    )
    category_split_distribution = build_split_distribution(
        train,
        test,
        CATEGORY_COLUMN,
    )
    section_split_distribution = build_split_distribution(
        train,
        test,
        SECTION_COLUMN,
    )
    publication_year_distribution = build_publication_year_distribution(
        original
    )
    category_section_table = pd.crosstab(
        original[SECTION_COLUMN],
        original[CATEGORY_COLUMN],
    ).reset_index()

    save_all_tables(
        (
            ("dataset_quality_summary.csv", quality_summary),
            ("column_profile.csv", column_profile),
            ("text_feature_statistics.csv", feature_statistics),
            ("category_statistics.csv", category_statistics),
            ("processing_comparison.csv", processing_comparison),
            ("category_distribution.csv", category_distribution),
            ("section_distribution.csv", section_distribution),
            ("category_section_contingency.csv", category_section_table),
            (
                "train_test_category_distribution.csv",
                category_split_distribution,
            ),
            (
                "train_test_section_distribution.csv",
                section_split_distribution,
            ),
            (
                "publication_year_distribution.csv",
                publication_year_distribution,
            ),
        ),
        table_directory,
    )

    plot_quality_issues(
        quality_summary,
        figure_directory / "dataset_quality_checks.png",
    )
    plot_missing_values(
        column_profile,
        figure_directory / "missing_values_by_column.png",
    )
    plot_label_distribution(
        category_distribution,
        CATEGORY_COLUMN,
        figure_directory / "category_distribution.png",
    )
    plot_label_distribution(
        section_distribution,
        SECTION_COLUMN,
        figure_directory / "section_distribution.png",
    )
    plot_category_section_heatmap(
        original,
        figure_directory / "category_section_heatmap.png",
    )
    plot_text_word_distributions(
        original,
        figure_directory / "text_component_word_distributions.png",
    )
    plot_article_length_histogram(
        original,
        figure_directory / "article_text_length_distribution.png",
    )
    plot_category_article_lengths(
        category_statistics,
        figure_directory / "median_article_length_by_category.png",
    )
    plot_processing_comparison(
        processing_comparison,
        figure_directory / "baseline_enhanced_text_length_comparison.png",
    )
    plot_split_comparison(
        category_split_distribution,
        CATEGORY_COLUMN,
        figure_directory / "train_test_category_distribution.png",
    )
    plot_split_comparison(
        section_split_distribution,
        SECTION_COLUMN,
        figure_directory / "train_test_section_distribution.png",
    )
    plot_publication_years(
        publication_year_distribution,
        figure_directory / "articles_by_publication_year.png",
    )

    parsed_dates = original[DATE_COLUMN].dropna()
    if not parsed_dates.empty:
        logger.info(
            "Publication date range: %s to %s.",
            parsed_dates.min().date(),
            parsed_dates.max().date(),
        )

    logger.info(
        "Exploratory data analysis completed successfully. "
        "Tables: %s; figures: %s.",
        table_directory,
        figure_directory,
    )


if __name__ == "__main__":
    try:
        main()

    except ExploratoryDataAnalysisError as error:
        logger.error("Exploratory data analysis failed: %s", error)
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "Exploratory data analysis failed because of an unexpected error."
        )
        raise SystemExit(1)
