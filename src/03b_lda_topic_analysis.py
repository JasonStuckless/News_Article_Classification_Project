"""
03b_lda_topic_analysis.py

Perform Latent Dirichlet Allocation (LDA) topic analysis on the validated CNN
News Articles dataset.

This stage adapts the topic-modelling methodology developed in the project
notebook and integrates it into the reproducible Python pipeline. It models
latent topics from article headlines and descriptions, assigns each article its
most probable topic, compares the discovered topics with the original CNN
Category and Section labels, and saves both tabular results and visualizations.

Inputs:
- data/raw/cnn_original.csv

Outputs:
- results/lda_topic_analysis/tables/*.csv
- results/lda_topic_analysis/figures/*.png
- results/lda_topic_analysis/artifacts/*.pkl
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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
# Constants
# ---------------------------------------------------------------------------

ID_COLUMN = "Index"
CATEGORY_COLUMN = "Category"
SECTION_COLUMN = "Section"
HEADLINE_COLUMN = "Headline"
DESCRIPTION_COLUMN = "Description"

REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    CATEGORY_COLUMN,
    SECTION_COLUMN,
    HEADLINE_COLUMN,
    DESCRIPTION_COLUMN,
)

DEFAULT_OUTPUT_SUBDIRECTORY = "lda_topic_analysis"
DEFAULT_NUMBER_OF_TOPICS = 12
DEFAULT_TOP_TERMS = 12
FIGURE_DPI = 300


class LDATopicAnalysisError(RuntimeError):
    """Raised when the LDA topic-analysis stage cannot be completed safely."""


# ---------------------------------------------------------------------------
# Configuration and paths
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the configuration required by this stage."""
    if not config_path.exists():
        raise LDATopicAnalysisError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise LDATopicAnalysisError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise LDATopicAnalysisError(
            "The configuration file must contain a YAML mapping."
        )

    project_config = config.get("project")
    dataset_config = config.get("dataset")
    output_config = config.get("output")

    if not isinstance(project_config, dict):
        raise LDATopicAnalysisError(
            "The configuration must contain a 'project' section."
        )

    if not isinstance(dataset_config, dict):
        raise LDATopicAnalysisError(
            "The configuration must contain a 'dataset' section."
        )

    if not isinstance(output_config, dict):
        raise LDATopicAnalysisError(
            "The configuration must contain an 'output' section."
        )

    if "random_seed" not in project_config:
        raise LDATopicAnalysisError(
            "Missing project.random_seed setting in config.yaml."
        )

    if "original" not in dataset_config:
        raise LDATopicAnalysisError(
            "Missing dataset.original setting in config.yaml."
        )

    if "results" not in output_config:
        raise LDATopicAnalysisError(
            "Missing output.results setting in config.yaml."
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_output_directories(
    results_root: Path,
) -> tuple[Path, Path, Path]:
    """Create and return the LDA table, figure, and artifact directories."""
    stage_root = results_root / DEFAULT_OUTPUT_SUBDIRECTORY
    table_directory = stage_root / "tables"
    figure_directory = stage_root / "figures"
    artifact_directory = stage_root / "artifacts"

    table_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)

    return table_directory, figure_directory, artifact_directory


def get_lda_settings(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return LDA settings, preserving the partner notebook defaults.

    An optional ``lda`` section can later be added to config.yaml without
    requiring changes to this script. Until then, the notebook settings are
    used exactly as defaults.
    """
    configured = config.get("lda", {})
    if configured is None:
        configured = {}

    if not isinstance(configured, dict):
        raise LDATopicAnalysisError(
            "The optional 'lda' configuration section must be a YAML mapping."
        )

    settings = {
        "n_components": configured.get("n_components", 12),
        "top_terms": configured.get("top_terms", 12),
        "min_df": configured.get("min_df", 5),
        "max_df": configured.get("max_df", 0.85),
        "max_features": configured.get("max_features", 8000),
        "ngram_range": configured.get("ngram_range", [1, 2]),
        "max_iter": configured.get("max_iter", 30),
        "learning_method": configured.get("learning_method", "batch"),
    }

    integer_settings = (
        "n_components",
        "top_terms",
        "min_df",
        "max_features",
        "max_iter",
    )
    for setting_name in integer_settings:
        value = settings[setting_name]
        if not isinstance(value, int) or value < 1:
            raise LDATopicAnalysisError(
                f"lda.{setting_name} must be a positive integer."
            )

    max_df = settings["max_df"]
    if not isinstance(max_df, (int, float)) or not 0 < float(max_df) <= 1:
        raise LDATopicAnalysisError(
            "lda.max_df must be greater than 0 and no greater than 1."
        )

    ngram_range = settings["ngram_range"]
    if (
        not isinstance(ngram_range, (list, tuple))
        or len(ngram_range) != 2
        or not all(isinstance(value, int) for value in ngram_range)
        or ngram_range[0] < 1
        or ngram_range[1] < ngram_range[0]
    ):
        raise LDATopicAnalysisError(
            "lda.ngram_range must contain two valid integers, such as [1, 2]."
        )

    settings["ngram_range"] = tuple(ngram_range)
    return settings


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate the dataset structure required for LDA analysis."""
    if dataframe.empty:
        raise LDATopicAnalysisError(
            "The original dataset contains no rows."
        )

    missing_columns = set(REQUIRED_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise LDATopicAnalysisError(
            f"The original dataset is missing required columns: {missing}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise LDATopicAnalysisError(
            f"The original dataset contains missing {ID_COLUMN} values."
        )

    if dataframe[ID_COLUMN].duplicated().any():
        duplicate_count = int(dataframe[ID_COLUMN].duplicated().sum())
        raise LDATopicAnalysisError(
            f"The original dataset contains {duplicate_count} duplicated "
            f"{ID_COLUMN} values."
        )

    for label_column in (CATEGORY_COLUMN, SECTION_COLUMN):
        if dataframe[label_column].isna().any():
            raise LDATopicAnalysisError(
                f"The original dataset contains missing {label_column} labels."
            )


def normalize_label_series(series: pd.Series) -> pd.Series:
    """Normalize labels for stable comparisons and plotting."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def combine_lda_text(dataframe: pd.DataFrame) -> pd.Series:
    """
    Combine Headline and Description using the notebook methodology.

    Article text is intentionally excluded here to preserve the topic-modelling
    design contributed in the partner notebook.
    """
    text_frame = (
        dataframe[[HEADLINE_COLUMN, DESCRIPTION_COLUMN]]
        .fillna("")
        .astype(str)
    )

    combined = text_frame.apply(
        lambda row: ". ".join(
            value.strip()
            for value in row
            if value.strip()
        ),
        axis=1,
    )

    empty_count = int(combined.str.strip().eq("").sum())
    if empty_count:
        raise LDATopicAnalysisError(
            f"{empty_count} articles have neither a usable headline nor "
            "description for LDA analysis."
        )

    return combined


# ---------------------------------------------------------------------------
# LDA modelling and summaries
# ---------------------------------------------------------------------------


def build_vectorizer(settings: dict[str, Any]) -> CountVectorizer:
    """Create the CountVectorizer used by the partner notebook."""
    return CountVectorizer(
        stop_words="english",
        min_df=settings["min_df"],
        max_df=settings["max_df"],
        max_features=settings["max_features"],
        ngram_range=settings["ngram_range"],
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
    )


def build_lda_model(
    settings: dict[str, Any],
    random_seed: int,
) -> LatentDirichletAllocation:
    """Create the configured LDA model."""
    return LatentDirichletAllocation(
        n_components=settings["n_components"],
        random_state=random_seed,
        learning_method=settings["learning_method"],
        max_iter=settings["max_iter"],
        n_jobs=-1,
    )


def build_topic_summary(
    lda_model: LatentDirichletAllocation,
    feature_names: np.ndarray,
    assigned_topics: pd.Series,
    top_terms: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create topic summaries and a long-form table of top-term weights."""
    summary_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []

    for topic_id, component_weights in enumerate(lda_model.components_):
        top_indices = component_weights.argsort()[-top_terms:][::-1]
        topic_terms = feature_names[top_indices]
        topic_weights = component_weights[top_indices]

        summary_rows.append(
            {
                "Topic": topic_id,
                "Top terms": ", ".join(topic_terms),
                "Articles": int((assigned_topics == topic_id).sum()),
                "Article share": float((assigned_topics == topic_id).mean()),
            }
        )

        for rank, (term, weight) in enumerate(
            zip(topic_terms, topic_weights, strict=True),
            start=1,
        ):
            term_rows.append(
                {
                    "Topic": topic_id,
                    "Rank": rank,
                    "Term": term,
                    "Weight": float(weight),
                }
            )

    topic_summary = pd.DataFrame(summary_rows).sort_values(
        "Articles",
        ascending=False,
    )
    top_term_weights = pd.DataFrame(term_rows)
    return topic_summary, top_term_weights


def build_contingency_tables(
    article_topics: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return count and row-normalized topic-to-label contingency tables."""
    counts = pd.crosstab(
        article_topics["LDA_topic"],
        article_topics[label_column],
    ).sort_index()

    proportions = counts.div(
        counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    counts.index.name = "LDA_topic"
    proportions.index.name = "LDA_topic"

    return counts.reset_index(), proportions.reset_index()


def build_alignment_metrics(article_topics: pd.DataFrame) -> pd.DataFrame:
    """Calculate NMI and ARI for Category and Section label alignment."""
    metric_rows: list[dict[str, Any]] = []

    for label_column in (CATEGORY_COLUMN, SECTION_COLUMN):
        labels = article_topics[label_column]
        topics = article_topics["LDA_topic"]

        metric_rows.extend(
            (
                {
                    "Label target": label_column,
                    "Metric": "Normalized Mutual Information",
                    "Value": float(
                        normalized_mutual_info_score(labels, topics)
                    ),
                },
                {
                    "Label target": label_column,
                    "Metric": "Adjusted Rand Index",
                    "Value": float(adjusted_rand_score(labels, topics)),
                },
            )
        )

    return pd.DataFrame(metric_rows)


def build_news_topic_distribution(
    article_topics: pd.DataFrame,
    topic_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the notebook's focused analysis of the broad news category."""
    news_rows = article_topics[
        article_topics[CATEGORY_COLUMN] == "news"
    ]

    if news_rows.empty:
        return pd.DataFrame(
            columns=("LDA_topic", "Articles", "Share of news", "Top terms")
        )

    counts = news_rows["LDA_topic"].value_counts().sort_index()
    result = counts.rename("Articles").to_frame()
    result["Share of news"] = result["Articles"] / result["Articles"].sum()
    result = result.join(
        topic_summary.set_index("Topic")[["Top terms"]],
        how="left",
    )
    result.index.name = "LDA_topic"
    return result.reset_index()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame through the project's CSV utility."""
    write_csv(dataframe, output_path)
    logger.info("Saved table: %s", output_path)


def save_all_tables(
    tables: Iterable[tuple[str, pd.DataFrame]],
    table_directory: Path,
) -> None:
    """Save all named result tables."""
    for filename, dataframe in tables:
        save_dataframe(dataframe, table_directory / filename)


def save_pickle(value: Any, output_path: Path) -> None:
    """Serialize an analysis artifact safely."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(value, file)
    logger.info("Saved artifact: %s", output_path)


def save_figure(output_path: Path) -> None:
    """Apply common layout settings, save the current figure, and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", output_path)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_topic_sizes(topic_summary: pd.DataFrame, output_path: Path) -> None:
    """Visualize the number of articles assigned to each topic."""
    plotting_data = topic_summary.sort_values("Articles", ascending=True)

    plt.figure(figsize=(10, 7))
    axis = plt.gca()
    bars = axis.barh(
        plotting_data["Topic"].astype(str),
        plotting_data["Articles"],
    )
    axis.set_title("Article Distribution Across LDA Topics")
    axis.set_xlabel("Number of articles")
    axis.set_ylabel("LDA topic")
    axis.bar_label(bars, padding=3, fmt="%d")

    save_figure(output_path)


def plot_topic_label_heatmap(
    contingency_table: pd.DataFrame,
    label_column: str,
    output_path: Path,
    normalize: bool,
) -> None:
    """Visualize topic-to-label counts or within-topic proportions."""
    matrix = contingency_table.set_index("LDA_topic")

    figure_width = max(10.0, 0.48 * matrix.shape[1] + 4.0)
    figure_height = max(7.0, 0.55 * matrix.shape[0] + 2.5)

    plt.figure(figsize=(figure_width, figure_height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar_kws={
            "label": "Within-topic proportion" if normalize else "Articles"
        },
    )

    suffix = "Proportions" if normalize else "Counts"
    plt.title(f"LDA Topics Versus CNN {label_column} Labels ({suffix})")
    plt.xlabel(f"CNN {label_column}")
    plt.ylabel("LDA topic")
    plt.xticks(rotation=45, ha="right")

    save_figure(output_path)


def plot_alignment_metrics(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize NMI and ARI for Category and Section alignment."""
    plt.figure(figsize=(9, 6))
    axis = sns.barplot(
        data=metrics,
        x="Label target",
        y="Value",
        hue="Metric",
    )
    axis.set_title("Alignment Between LDA Topics and CNN Labels")
    axis.set_xlabel("CNN label target")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, max(1.0, float(metrics["Value"].max()) * 1.15))
    axis.legend(title="Metric")

    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3)

    save_figure(output_path)


def plot_top_terms_by_topic(
    top_term_weights: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize the top weighted terms for every discovered topic."""
    topics = sorted(top_term_weights["Topic"].unique())
    number_of_columns = 3
    number_of_rows = int(np.ceil(len(topics) / number_of_columns))

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(18, 4.6 * number_of_rows),
        squeeze=False,
    )

    for axis, topic_id in zip(axes.flat, topics, strict=False):
        topic_data = (
            top_term_weights[top_term_weights["Topic"] == topic_id]
            .sort_values("Weight", ascending=True)
        )
        axis.barh(topic_data["Term"], topic_data["Weight"])
        axis.set_title(f"Topic {topic_id}")
        axis.set_xlabel("LDA term weight")
        axis.set_ylabel("")

    for axis in axes.flat[len(topics):]:
        axis.set_visible(False)

    figure.suptitle("Top Terms for Each LDA Topic", fontsize=16, y=1.002)
    save_figure(output_path)


def plot_topic_confidence_distribution(
    article_topics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize confidence in each article's dominant-topic assignment."""
    plt.figure(figsize=(10, 6))
    plt.hist(
        article_topics["LDA_topic_probability"],
        bins=30,
        edgecolor="black",
    )
    plt.title("Distribution of Dominant LDA Topic Probabilities")
    plt.xlabel("Probability assigned to dominant topic")
    plt.ylabel("Number of articles")
    save_figure(output_path)


def plot_topic_confidence_by_topic(
    article_topics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare dominant-topic probability distributions across topics."""
    plt.figure(figsize=(12, 7))
    topics = sorted(article_topics["LDA_topic"].unique())
    probability_groups = [
        article_topics.loc[
            article_topics["LDA_topic"] == topic,
            "LDA_topic_probability",
        ].to_numpy()
        for topic in topics
    ]
    plt.boxplot(
        probability_groups,
        tick_labels=[str(topic) for topic in topics],
    )
    plt.title("Dominant-Topic Confidence by LDA Topic")
    plt.xlabel("LDA topic")
    plt.ylabel("Dominant-topic probability")
    save_figure(output_path)


def plot_news_topic_distribution(
    news_topic_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Visualize how the broad CNN news category separates across topics."""
    if news_topic_distribution.empty:
        logger.warning(
            "Skipping news-topic figure because no Category='news' rows exist."
        )
        return

    plotting_data = news_topic_distribution.sort_values(
        "Share of news",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    axis = plt.gca()
    bars = axis.barh(
        plotting_data["LDA_topic"].astype(str),
        plotting_data["Share of news"],
    )
    axis.set_title("Distribution of CNN News Articles Across LDA Topics")
    axis.set_xlabel("Share of articles labelled news")
    axis.set_ylabel("LDA topic")
    axis.xaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=1.0)
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.1%}" for value in plotting_data["Share of news"]],
        padding=3,
    )

    save_figure(output_path)


# ---------------------------------------------------------------------------
# Main pipeline stage
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete LDA topic-analysis stage."""
    config = load_config(CONFIG_PATH)
    settings = get_lda_settings(config)

    project_config = config["project"]
    dataset_config = config["dataset"]
    output_config = config["output"]

    random_seed = project_config["random_seed"]
    if not isinstance(random_seed, int):
        raise LDATopicAnalysisError(
            "project.random_seed must be an integer."
        )

    original_path = resolve_project_path(dataset_config["original"])
    results_root = resolve_project_path(output_config["results"])

    if not original_path.exists():
        raise LDATopicAnalysisError(
            f"The original dataset was not found: {original_path}"
        )

    table_directory, figure_directory, artifact_directory = (
        create_output_directories(results_root)
    )

    logger.info("Loading original dataset from %s.", original_path)
    original = read_csv(original_path)
    validate_dataframe(original)

    analysis_data = original.copy()
    analysis_data[CATEGORY_COLUMN] = normalize_label_series(
        analysis_data[CATEGORY_COLUMN]
    )
    analysis_data[SECTION_COLUMN] = normalize_label_series(
        analysis_data[SECTION_COLUMN]
    )

    lda_text = combine_lda_text(analysis_data)

    logger.info(
        "Vectorizing %d articles for LDA topic analysis.",
        len(analysis_data),
    )
    vectorizer = build_vectorizer(settings)
    document_term_matrix = vectorizer.fit_transform(lda_text)

    if document_term_matrix.shape[1] == 0:
        raise LDATopicAnalysisError(
            "CountVectorizer produced no features. Review the LDA frequency "
            "thresholds and input text."
        )

    logger.info(
        "Fitting %d-topic LDA model with %d vocabulary terms.",
        settings["n_components"],
        document_term_matrix.shape[1],
    )
    lda_model = build_lda_model(settings, random_seed)
    topic_probabilities = lda_model.fit_transform(document_term_matrix)

    analysis_data["LDA_topic"] = topic_probabilities.argmax(axis=1)
    analysis_data["LDA_topic_probability"] = topic_probabilities.max(axis=1)

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    topic_summary, top_term_weights = build_topic_summary(
        lda_model=lda_model,
        feature_names=feature_names,
        assigned_topics=analysis_data["LDA_topic"],
        top_terms=settings["top_terms"],
    )

    article_topics = analysis_data.loc[
        :,
        [
            ID_COLUMN,
            HEADLINE_COLUMN,
            CATEGORY_COLUMN,
            SECTION_COLUMN,
            "LDA_topic",
            "LDA_topic_probability",
        ],
    ].copy()

    category_counts, category_proportions = build_contingency_tables(
        article_topics,
        CATEGORY_COLUMN,
    )
    section_counts, section_proportions = build_contingency_tables(
        article_topics,
        SECTION_COLUMN,
    )
    alignment_metrics = build_alignment_metrics(article_topics)
    news_topic_distribution = build_news_topic_distribution(
        article_topics,
        topic_summary,
    )

    model_summary = pd.DataFrame(
        [
            {
                "Articles": len(analysis_data),
                "Vocabulary size": document_term_matrix.shape[1],
                "Number of topics": settings["n_components"],
                "Top terms per topic": settings["top_terms"],
                "LDA iterations": int(lda_model.n_iter_),
                "Perplexity": float(lda_model.perplexity(document_term_matrix)),
                "Mean dominant-topic probability": float(
                    article_topics["LDA_topic_probability"].mean()
                ),
                "Median dominant-topic probability": float(
                    article_topics["LDA_topic_probability"].median()
                ),
            }
        ]
    )

    save_all_tables(
        (
            ("lda_model_summary.csv", model_summary),
            ("lda_topic_summary.csv", topic_summary),
            ("lda_top_term_weights.csv", top_term_weights),
            ("lda_article_topics.csv", article_topics),
            ("lda_topic_category_counts.csv", category_counts),
            (
                "lda_topic_category_proportions.csv",
                category_proportions,
            ),
            ("lda_topic_section_counts.csv", section_counts),
            (
                "lda_topic_section_proportions.csv",
                section_proportions,
            ),
            ("lda_label_alignment_metrics.csv", alignment_metrics),
            ("lda_news_topic_distribution.csv", news_topic_distribution),
        ),
        table_directory,
    )

    save_pickle(
        vectorizer,
        artifact_directory / "lda_count_vectorizer.pkl",
    )
    save_pickle(
        lda_model,
        artifact_directory / "lda_model.pkl",
    )

    plot_topic_sizes(
        topic_summary,
        figure_directory / "lda_topic_article_distribution.png",
    )
    plot_top_terms_by_topic(
        top_term_weights,
        figure_directory / "lda_top_terms_by_topic.png",
    )
    plot_topic_label_heatmap(
        category_counts,
        CATEGORY_COLUMN,
        figure_directory / "lda_topic_category_counts.png",
        normalize=False,
    )
    plot_topic_label_heatmap(
        category_proportions,
        CATEGORY_COLUMN,
        figure_directory / "lda_topic_category_proportions.png",
        normalize=True,
    )
    plot_topic_label_heatmap(
        section_counts,
        SECTION_COLUMN,
        figure_directory / "lda_topic_section_counts.png",
        normalize=False,
    )
    plot_topic_label_heatmap(
        section_proportions,
        SECTION_COLUMN,
        figure_directory / "lda_topic_section_proportions.png",
        normalize=True,
    )
    plot_alignment_metrics(
        alignment_metrics,
        figure_directory / "lda_label_alignment_metrics.png",
    )
    plot_topic_confidence_distribution(
        article_topics,
        figure_directory / "lda_topic_probability_distribution.png",
    )
    plot_topic_confidence_by_topic(
        article_topics,
        figure_directory / "lda_topic_probability_by_topic.png",
    )
    plot_news_topic_distribution(
        news_topic_distribution,
        figure_directory / "lda_news_topic_distribution.png",
    )

    category_nmi = alignment_metrics.loc[
        (alignment_metrics["Label target"] == CATEGORY_COLUMN)
        & (
            alignment_metrics["Metric"]
            == "Normalized Mutual Information"
        ),
        "Value",
    ].iloc[0]
    category_ari = alignment_metrics.loc[
        (alignment_metrics["Label target"] == CATEGORY_COLUMN)
        & (alignment_metrics["Metric"] == "Adjusted Rand Index"),
        "Value",
    ].iloc[0]

    logger.info(
        "LDA topic analysis completed successfully. Category NMI=%.3f; "
        "Category ARI=%.3f. Tables: %s; figures: %s; artifacts: %s.",
        category_nmi,
        category_ari,
        table_directory,
        figure_directory,
        artifact_directory,
    )


if __name__ == "__main__":
    try:
        main()

    except LDATopicAnalysisError as error:
        logger.error("LDA topic analysis failed: %s", error)
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "LDA topic analysis failed because of an unexpected error."
        )
        raise SystemExit(1)
