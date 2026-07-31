"""
12_compare_representations.py

Compare the existing TF-IDF classifier results with the Sentence-BERT
embedding classifier results produced by 11_evaluate_embedding_models.py.

The script creates two complementary comparisons:

1. Complete comparison
   Includes every evaluated model from both representations.

2. Matched-model comparison
   Directly compares only classifiers available for both representations,
   normally Logistic Regression and Linear SVM. This is the fairest estimate
   of the effect of changing the text representation because the classifier,
   target, training-label source, and test set are held constant.

Inputs:
- TF-IDF evaluation summary produced by 06_evaluate_models.py
- results/tables/sentence_bert/embedding_model_evaluation_summary.csv

The TF-IDF summary is located using a set of expected paths followed by a
controlled search under results/. This supports the existing project without
requiring changes to 06_evaluate_models.py.

Outputs:
- results/tables/representation_comparison/all_representation_results.csv
- results/tables/representation_comparison/matched_model_comparison.csv
- results/tables/representation_comparison/representation_summary.csv
- results/tables/representation_comparison/best_models_by_condition.csv
- results/tables/representation_comparison/metric_differences.csv
- results/figures/representation_comparison/macro_f1_comparison.png
- results/figures/representation_comparison/accuracy_comparison.png
- results/figures/representation_comparison/matched_model_differences.png
- results/figures/representation_comparison/best_model_comparison.png
- results/representation_comparison/comparison_manifest.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

RESULTS_DIRECTORY = PROJECT_ROOT / "results"

EMBEDDING_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "tables"
    / "sentence_bert"
    / "embedding_model_evaluation_summary.csv"
)

TABLE_DIRECTORY = (
    RESULTS_DIRECTORY / "tables" / "representation_comparison"
)
ALL_RESULTS_PATH = TABLE_DIRECTORY / "all_representation_results.csv"
MATCHED_RESULTS_PATH = TABLE_DIRECTORY / "matched_model_comparison.csv"
REPRESENTATION_SUMMARY_PATH = TABLE_DIRECTORY / "representation_summary.csv"
BEST_MODELS_PATH = TABLE_DIRECTORY / "best_models_by_condition.csv"
DIFFERENCES_PATH = TABLE_DIRECTORY / "metric_differences.csv"

FIGURE_DIRECTORY = (
    RESULTS_DIRECTORY / "figures" / "representation_comparison"
)
MACRO_F1_FIGURE_PATH = FIGURE_DIRECTORY / "macro_f1_comparison.png"
ACCURACY_FIGURE_PATH = FIGURE_DIRECTORY / "accuracy_comparison.png"
DIFFERENCE_FIGURE_PATH = FIGURE_DIRECTORY / "matched_model_differences.png"
BEST_MODEL_FIGURE_PATH = FIGURE_DIRECTORY / "best_model_comparison.png"

MANIFEST_PATH = (
    RESULTS_DIRECTORY
    / "representation_comparison"
    / "comparison_manifest.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from utils.io import write_csv
except ImportError:
    # This fallback keeps the stage independently executable while preserving
    # compatibility with the project's normal utility module.
    def write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)

try:
    from utils.logging import get_logger

    logger = get_logger(
        name=__name__,
        log_directory=PROJECT_ROOT / "logs",
        log_file="project.log",
    )
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_METRICS = ("accuracy", "macro_f1", "weighted_f1")
OPTIONAL_METRICS = (
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "weighted_precision",
    "weighted_recall",
)

COMPARISON_KEYS = ("target", "label_source", "model")

REPRESENTATION_DISPLAY = {
    "tfidf": "TF-IDF",
    "sentence_bert": "Sentence-BERT",
}

LABEL_SOURCE_DISPLAY = {
    "real": "CNN labels",
    "llm": "LLM labels",
}

MODEL_ALIASES = {
    "logistic": "logistic",
    "logistic_regression": "logistic",
    "logisticregression": "logistic",
    "lr": "logistic",
    "svm": "svm",
    "linear_svm": "svm",
    "linear_svc": "svm",
    "linearsvc": "svm",
    "svc": "svm",
    "nb": "nb",
    "naive_bayes": "nb",
    "multinomial_nb": "nb",
    "multinomialnb": "nb",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "neural_network": "neural_network",
    "small_neural_network": "neural_network",
    "mlp": "neural_network",
}

MODEL_DISPLAY_NAMES = {
    "logistic": "Logistic Regression",
    "svm": "Linear SVM",
    "nb": "Multinomial Naive Bayes",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Small Neural Network",
}

TFIDF_PREFERRED_PATHS = (
    RESULTS_DIRECTORY / "tables" / "model_evaluation_summary.csv",
    RESULTS_DIRECTORY / "tables" / "model_performance_summary.csv",
    RESULTS_DIRECTORY / "tables" / "model_comparison.csv",
    RESULTS_DIRECTORY / "tables" / "evaluation_summary.csv",
    RESULTS_DIRECTORY / "metrics" / "model_evaluation_summary.csv",
    RESULTS_DIRECTORY / "metrics" / "evaluation_summary.csv",
    RESULTS_DIRECTORY / "metrics" / "model_metrics.csv",
    RESULTS_DIRECTORY / "evaluation_summary.csv",
    RESULTS_DIRECTORY / "model_comparison.csv",
)


class RepresentationComparisonError(RuntimeError):
    """Raised when representation results cannot be compared safely."""


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def canonical_name(value: Any) -> str:
    """Convert a column name or categorical value to a stable identifier."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize common result-column variants to the comparison schema."""
    result = dataframe.copy()

    aliases = {
        "task": "target",
        "prediction_target": "target",
        "label_type": "target",
        "label": "target",
        "training_label_source": "label_source",
        "training_labels": "label_source",
        "source": "label_source",
        "label_origin": "label_source",
        "classifier": "model",
        "model_name": "model",
        "algorithm": "model",
        "acc": "accuracy",
        "f1_macro": "macro_f1",
        "macro_f1_score": "macro_f1",
        "f1_weighted": "weighted_f1",
        "weighted_f1_score": "weighted_f1",
        "precision_macro": "macro_precision",
        "recall_macro": "macro_recall",
        "precision_weighted": "weighted_precision",
        "recall_weighted": "weighted_recall",
    }

    normalized_columns = {}
    for column in result.columns:
        normalized = canonical_name(column)
        normalized_columns[column] = aliases.get(normalized, normalized)
    result = result.rename(columns=normalized_columns)

    # Avoid silently using ambiguous duplicate columns.
    if result.columns.duplicated().any():
        duplicates = sorted(set(result.columns[result.columns.duplicated()]))
        raise RepresentationComparisonError(
            "A result table contains duplicate normalized columns: "
            + ", ".join(duplicates)
        )

    return result


def normalize_target(value: Any) -> str:
    """Normalize Category/Section target values."""
    normalized = canonical_name(value)
    target_aliases = {
        "categories": "category",
        "category_label": "category",
        "sections": "section",
        "section_label": "section",
    }
    return target_aliases.get(normalized, normalized)


def normalize_label_source(value: Any) -> str:
    """Normalize original/CNN and LLM label-source values."""
    normalized = canonical_name(value)
    if normalized in {
        "real",
        "original",
        "cnn",
        "cnn_label",
        "cnn_labels",
        "original_label",
        "original_labels",
        "human",
        "human_labels",
        "real_label",
        "real_labels",
    }:
        return "real"
    if normalized in {
        "llm",
        "llm_label",
        "llm_labels",
        "generated",
        "generated_labels",
        "ai",
        "ai_labels",
    }:
        return "llm"
    return normalized


def normalize_model(value: Any) -> str:
    """Normalize classifier names shared across pipeline stages."""
    normalized = canonical_name(value)
    return MODEL_ALIASES.get(normalized, normalized)


def coerce_metric_columns(dataframe: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Convert metric columns to numeric values and reject invalid rows."""
    result = dataframe.copy()
    for metric in REQUIRED_METRICS:
        result[metric] = pd.to_numeric(result[metric], errors="coerce")

    invalid = result[list(REQUIRED_METRICS)].isna().any(axis=1)
    if invalid.any():
        count = int(invalid.sum())
        raise RepresentationComparisonError(
            f"{source_path} contains {count} rows with missing or non-numeric "
            "required metrics."
        )

    for metric in OPTIONAL_METRICS:
        if metric in result.columns:
            result[metric] = pd.to_numeric(result[metric], errors="coerce")

    for metric in REQUIRED_METRICS:
        outside_range = ~result[metric].between(0, 1, inclusive="both")
        if outside_range.any():
            raise RepresentationComparisonError(
                f"{source_path} contains {metric} values outside [0, 1]."
            )

    return result


def prepare_result_table(
    dataframe: pd.DataFrame,
    representation: str,
    source_path: Path,
) -> pd.DataFrame:
    """Transform one evaluation table into the common comparison schema."""
    result = normalize_columns(dataframe)

    required_columns = {
        "target",
        "label_source",
        "model",
        *REQUIRED_METRICS,
    }
    missing = required_columns - set(result.columns)
    if missing:
        raise RepresentationComparisonError(
            f"{source_path} is missing required comparison columns: "
            + ", ".join(sorted(missing))
        )

    result["target"] = result["target"].map(normalize_target)
    result["label_source"] = result["label_source"].map(normalize_label_source)
    result["model"] = result["model"].map(normalize_model)

    invalid_targets = sorted(
        set(result["target"]) - {"category", "section"}
    )
    if invalid_targets:
        raise RepresentationComparisonError(
            f"{source_path} contains unsupported targets: "
            + ", ".join(invalid_targets)
        )

    invalid_sources = sorted(
        set(result["label_source"]) - {"real", "llm"}
    )
    if invalid_sources:
        raise RepresentationComparisonError(
            f"{source_path} contains unsupported label sources: "
            + ", ".join(invalid_sources)
        )

    result = coerce_metric_columns(result, source_path)

    duplicated = result.duplicated(list(COMPARISON_KEYS), keep=False)
    if duplicated.any():
        duplicate_keys = (
            result.loc[duplicated, list(COMPARISON_KEYS)]
            .drop_duplicates()
            .astype(str)
            .agg("/".join, axis=1)
            .tolist()
        )
        raise RepresentationComparisonError(
            f"{source_path} contains duplicate experiments: "
            + ", ".join(duplicate_keys[:10])
        )

    result["representation"] = representation
    result["representation_display"] = REPRESENTATION_DISPLAY[representation]
    result["label_source_display"] = result["label_source"].map(
        LABEL_SOURCE_DISPLAY
    )
    result["model_display_name"] = result["model"].map(
        lambda value: MODEL_DISPLAY_NAMES.get(value, value.replace("_", " ").title())
    )

    preferred_columns = [
        "representation",
        "representation_display",
        "target",
        "label_source",
        "label_source_display",
        "model",
        "model_display_name",
        *REQUIRED_METRICS,
        *[metric for metric in OPTIONAL_METRICS if metric in result.columns],
    ]
    remaining_columns = [
        column for column in result.columns if column not in preferred_columns
    ]
    return result[preferred_columns + remaining_columns]


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """Read a CSV and provide a path-specific error."""
    try:
        return pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise RepresentationComparisonError(
            f"Could not read CSV file {file_path}: {error}"
        ) from error


def table_has_comparison_schema(file_path: Path) -> bool:
    """Check whether a CSV appears to contain aggregate model metrics."""
    try:
        sample = pd.read_csv(file_path, nrows=5)
        sample = normalize_columns(sample)
    except Exception:
        return False

    required = {
        "target",
        "label_source",
        "model",
        *REQUIRED_METRICS,
    }
    return required.issubset(sample.columns)


def score_tfidf_candidate(file_path: Path) -> tuple[int, str]:
    """Rank possible TF-IDF summary files deterministically."""
    name = canonical_name(file_path.stem)
    path_text = canonical_name(str(file_path.relative_to(RESULTS_DIRECTORY)))

    score = 0
    if "summary" in name:
        score += 6
    if "evaluation" in name:
        score += 5
    if "model" in name:
        score += 3
    if "comparison" in name:
        score += 2
    if "metric" in path_text:
        score += 2
    if "table" in path_text:
        score += 1
    if "sentence_bert" in path_text or "embedding" in path_text:
        score -= 100
    if "representation_comparison" in path_text:
        score -= 100

    return score, str(file_path)


def locate_tfidf_summary() -> Path:
    """Locate the aggregate TF-IDF evaluation table without changing stage 6."""
    for candidate in TFIDF_PREFERRED_PATHS:
        if candidate.exists() and table_has_comparison_schema(candidate):
            logger.info("Using TF-IDF evaluation summary: %s", candidate)
            return candidate

    if not RESULTS_DIRECTORY.exists():
        raise RepresentationComparisonError(
            f"Results directory not found: {RESULTS_DIRECTORY}. "
            "Run 06_evaluate_models.py first."
        )

    candidates = [
        path
        for path in RESULTS_DIRECTORY.rglob("*.csv")
        if path != EMBEDDING_SUMMARY_PATH
        and table_has_comparison_schema(path)
    ]
    candidates.sort(key=score_tfidf_candidate, reverse=True)

    if not candidates:
        expected = "\n- ".join(str(path) for path in TFIDF_PREFERRED_PATHS)
        raise RepresentationComparisonError(
            "Could not locate the TF-IDF aggregate evaluation summary. "
            "Run 06_evaluate_models.py first. Checked preferred paths:\n- "
            + expected
        )

    selected = candidates[0]
    logger.info(
        "TF-IDF summary was discovered automatically at: %s", selected
    )
    return selected


def load_evaluation_results() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load and normalize TF-IDF and Sentence-BERT evaluation summaries."""
    tfidf_path = locate_tfidf_summary()

    if not EMBEDDING_SUMMARY_PATH.exists():
        raise RepresentationComparisonError(
            "Sentence-BERT evaluation summary not found. Run "
            "11_evaluate_embedding_models.py first: "
            f"{EMBEDDING_SUMMARY_PATH}"
        )

    tfidf = prepare_result_table(
        read_csv_safely(tfidf_path),
        representation="tfidf",
        source_path=tfidf_path,
    )
    sentence_bert = prepare_result_table(
        read_csv_safely(EMBEDDING_SUMMARY_PATH),
        representation="sentence_bert",
        source_path=EMBEDDING_SUMMARY_PATH,
    )

    logger.info(
        "Loaded %d TF-IDF and %d Sentence-BERT evaluation rows.",
        len(tfidf),
        len(sentence_bert),
    )
    return tfidf, sentence_bert, tfidf_path


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------


def create_all_results(
    tfidf: pd.DataFrame,
    sentence_bert: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all model results from both representations."""
    common_columns = [
        "representation",
        "representation_display",
        "target",
        "label_source",
        "label_source_display",
        "model",
        "model_display_name",
        *REQUIRED_METRICS,
    ]

    optional_common = [
        metric
        for metric in OPTIONAL_METRICS
        if metric in tfidf.columns and metric in sentence_bert.columns
    ]
    common_columns.extend(optional_common)

    combined = pd.concat(
        [tfidf[common_columns], sentence_bert[common_columns]],
        ignore_index=True,
    )
    return combined.sort_values(
        ["target", "label_source", "representation", "macro_f1"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def create_matched_comparison(
    tfidf: pd.DataFrame,
    sentence_bert: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare identical classifiers across representations."""
    tfidf_keys = tfidf[list(COMPARISON_KEYS)].drop_duplicates()
    embedding_keys = sentence_bert[list(COMPARISON_KEYS)].drop_duplicates()
    shared_keys = tfidf_keys.merge(
        embedding_keys,
        on=list(COMPARISON_KEYS),
        how="inner",
        validate="one_to_one",
    )

    if shared_keys.empty:
        raise RepresentationComparisonError(
            "No matched target/label-source/model experiments exist across "
            "TF-IDF and Sentence-BERT results."
        )

    tfidf_metrics = tfidf[
        [*COMPARISON_KEYS, "model_display_name", *REQUIRED_METRICS]
    ].rename(
        columns={
            metric: f"tfidf_{metric}" for metric in REQUIRED_METRICS
        }
    )
    embedding_metrics = sentence_bert[
        [*COMPARISON_KEYS, "model_display_name", *REQUIRED_METRICS]
    ].rename(
        columns={
            metric: f"sentence_bert_{metric}" for metric in REQUIRED_METRICS
        }
    )
    embedding_metrics = embedding_metrics.drop(
        columns=["model_display_name"]
    )

    matched = shared_keys.merge(
        tfidf_metrics,
        on=list(COMPARISON_KEYS),
        how="left",
        validate="one_to_one",
    ).merge(
        embedding_metrics,
        on=list(COMPARISON_KEYS),
        how="left",
        validate="one_to_one",
    )

    matched["label_source_display"] = matched["label_source"].map(
        LABEL_SOURCE_DISPLAY
    )
    for metric in REQUIRED_METRICS:
        matched[f"{metric}_difference"] = (
            matched[f"sentence_bert_{metric}"]
            - matched[f"tfidf_{metric}"]
        )
        matched[f"{metric}_winner"] = np.select(
            [
                matched[f"{metric}_difference"] > 0,
                matched[f"{metric}_difference"] < 0,
            ],
            ["Sentence-BERT", "TF-IDF"],
            default="Tie",
        )

    matched = matched.sort_values(
        ["target", "label_source", "model"]
    ).reset_index(drop=True)

    difference_columns = [
        "target",
        "label_source",
        "label_source_display",
        "model",
        "model_display_name",
        *[f"{metric}_difference" for metric in REQUIRED_METRICS],
        *[f"{metric}_winner" for metric in REQUIRED_METRICS],
    ]
    differences = matched[difference_columns].copy()

    return matched, differences


def create_representation_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scores by representation, target, and label source."""
    summary = (
        all_results.groupby(
            [
                "representation",
                "representation_display",
                "target",
                "label_source",
                "label_source_display",
            ],
            as_index=False,
        )
        .agg(
            model_count=("model", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_weighted_f1=("weighted_f1", "mean"),
            best_accuracy=("accuracy", "max"),
            best_macro_f1=("macro_f1", "max"),
            best_weighted_f1=("weighted_f1", "max"),
        )
        .sort_values(
            ["target", "label_source", "representation"]
        )
        .reset_index(drop=True)
    )
    return summary


def create_best_models(all_results: pd.DataFrame) -> pd.DataFrame:
    """Select the best macro-F1 model for each experimental condition."""
    sorted_results = all_results.sort_values(
        [
            "target",
            "label_source",
            "representation",
            "macro_f1",
            "accuracy",
            "model_display_name",
        ],
        ascending=[True, True, True, False, False, True],
    )
    best = (
        sorted_results.groupby(
            ["target", "label_source", "representation"],
            as_index=False,
        )
        .first()
        .sort_values(["target", "label_source", "representation"])
        .reset_index(drop=True)
    )
    return best


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def condition_label(dataframe: pd.DataFrame) -> pd.Series:
    """Create compact labels for target and training-label source."""
    return (
        dataframe["target"].str.title()
        + "\n"
        + dataframe["label_source_display"]
    )


def save_grouped_metric_figure(
    matched: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot matched TF-IDF and Sentence-BERT scores for one metric."""
    plot_data = matched.copy()
    plot_data["experiment"] = (
        plot_data["target"].str.title()
        + " | "
        + plot_data["label_source_display"]
        + " | "
        + plot_data["model_display_name"]
    )
    plot_data = plot_data.sort_values(
        ["target", "label_source", "model_display_name"]
    )

    positions = np.arange(len(plot_data))
    width = 0.38

    figure_height = max(6.5, len(plot_data) * 0.58)
    figure, axis = plt.subplots(figsize=(13, figure_height))
    axis.barh(
        positions - width / 2,
        plot_data[f"tfidf_{metric}"],
        height=width,
        label="TF-IDF",
    )
    axis.barh(
        positions + width / 2,
        plot_data[f"sentence_bert_{metric}"],
        height=width,
        label="Sentence-BERT",
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(plot_data["experiment"])
    axis.set_xlim(0, 1)
    axis.set_xlabel(metric.replace("_", " ").title())
    axis.set_title(title)
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.3)
    axis.invert_yaxis()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_difference_figure(matched: pd.DataFrame) -> None:
    """Plot Sentence-BERT minus TF-IDF macro-F1 differences."""
    plot_data = matched.copy()
    plot_data["experiment"] = (
        plot_data["target"].str.title()
        + " | "
        + plot_data["label_source_display"]
        + " | "
        + plot_data["model_display_name"]
    )
    plot_data = plot_data.sort_values("macro_f1_difference")

    positions = np.arange(len(plot_data))
    figure_height = max(6.5, len(plot_data) * 0.55)
    figure, axis = plt.subplots(figsize=(13, figure_height))
    axis.barh(positions, plot_data["macro_f1_difference"])
    axis.axvline(0, linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(plot_data["experiment"])
    axis.set_xlabel("Macro F1 difference (Sentence-BERT - TF-IDF)")
    axis.set_title("Matched-Classifier Representation Differences")
    axis.grid(axis="x", alpha=0.3)

    max_absolute = float(
        np.nanmax(np.abs(plot_data["macro_f1_difference"].to_numpy()))
    )
    if max_absolute > 0:
        axis.set_xlim(-max_absolute * 1.15, max_absolute * 1.15)

    DIFFERENCE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(DIFFERENCE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_best_model_figure(best_models: pd.DataFrame) -> None:
    """Compare the best macro-F1 result from each representation."""
    plot_data = best_models.copy()
    plot_data["condition"] = condition_label(plot_data)
    pivot = plot_data.pivot_table(
        index="condition",
        columns="representation_display",
        values="macro_f1",
        aggfunc="first",
    )

    preferred_columns = [
        name for name in ("TF-IDF", "Sentence-BERT") if name in pivot.columns
    ]
    pivot = pivot.reindex(columns=preferred_columns)

    figure, axis = plt.subplots(figsize=(11, 7))
    pivot.plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Evaluation condition")
    axis.set_ylabel("Best macro F1")
    axis.set_title("Best Model by Text Representation")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Representation")
    axis.grid(axis="y", alpha=0.3)

    BEST_MODEL_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(BEST_MODEL_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_json_atomic(value: dict[str, Any], output_path: Path) -> None:
    """Write JSON atomically to avoid partially written manifests."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary_path.replace(output_path)
    except (OSError, TypeError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RepresentationComparisonError(
            f"Could not save JSON output {output_path}: {error}"
        ) from error


def relative_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


def run_representation_comparison() -> None:
    """Run the complete TF-IDF versus Sentence-BERT comparison."""
    logger.info("Starting representation comparison.")
    stage_start = time.perf_counter()

    tfidf, sentence_bert, tfidf_path = load_evaluation_results()

    all_results = create_all_results(tfidf, sentence_bert)
    matched, differences = create_matched_comparison(tfidf, sentence_bert)
    representation_summary = create_representation_summary(all_results)
    best_models = create_best_models(all_results)

    write_csv(all_results, ALL_RESULTS_PATH)
    write_csv(matched, MATCHED_RESULTS_PATH)
    write_csv(representation_summary, REPRESENTATION_SUMMARY_PATH)
    write_csv(best_models, BEST_MODELS_PATH)
    write_csv(differences, DIFFERENCES_PATH)

    save_grouped_metric_figure(
        matched=matched,
        metric="macro_f1",
        title="TF-IDF versus Sentence-BERT: Matched-Classifier Macro F1",
        output_path=MACRO_F1_FIGURE_PATH,
    )
    save_grouped_metric_figure(
        matched=matched,
        metric="accuracy",
        title="TF-IDF versus Sentence-BERT: Matched-Classifier Accuracy",
        output_path=ACCURACY_FIGURE_PATH,
    )
    create_difference_figure(matched)
    create_best_model_figure(best_models)

    elapsed = time.perf_counter() - stage_start

    manifest = {
        "stage": "12_compare_representations",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "elapsed_seconds": elapsed,
        "inputs": {
            "tfidf_summary": relative_path(tfidf_path),
            "sentence_bert_summary": relative_path(EMBEDDING_SUMMARY_PATH),
        },
        "row_counts": {
            "tfidf_results": int(len(tfidf)),
            "sentence_bert_results": int(len(sentence_bert)),
            "all_results": int(len(all_results)),
            "matched_experiments": int(len(matched)),
            "best_model_rows": int(len(best_models)),
        },
        "matched_models": sorted(matched["model"].unique().tolist()),
        "outputs": [
            relative_path(ALL_RESULTS_PATH),
            relative_path(MATCHED_RESULTS_PATH),
            relative_path(REPRESENTATION_SUMMARY_PATH),
            relative_path(BEST_MODELS_PATH),
            relative_path(DIFFERENCES_PATH),
            relative_path(MACRO_F1_FIGURE_PATH),
            relative_path(ACCURACY_FIGURE_PATH),
            relative_path(DIFFERENCE_FIGURE_PATH),
            relative_path(BEST_MODEL_FIGURE_PATH),
        ],
    }
    write_json_atomic(manifest, MANIFEST_PATH)

    logger.info(
        "Representation comparison complete in %.2f seconds. "
        "Compared %d matched experiments across %d shared classifiers.",
        elapsed,
        len(matched),
        matched["model"].nunique(),
    )


def main() -> int:
    """Command-line entry point."""
    try:
        run_representation_comparison()
    except RepresentationComparisonError as error:
        logger.error("Representation comparison failed: %s", error)
        return 1
    except Exception:
        logger.exception("Unexpected representation-comparison failure.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
