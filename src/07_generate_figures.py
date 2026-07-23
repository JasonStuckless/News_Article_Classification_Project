"""
07_generate_figures.py

Generate all visual results for the news article classification project.

This script creates:

1. Confusion matrices for every trained classifier.
2. Overall macro-metric comparison figures:
   - Accuracy
   - Macro Precision
   - Macro Recall
   - Macro F1
3. Overall weighted-metric comparison figures:
   - Accuracy
   - Weighted Precision
   - Weighted Recall
   - Weighted F1
4. Per-class metric heatmaps:
   - Precision
   - Recall
   - F1
5. Direct agreement figures comparing original and LLM-generated labels.
6. Summary CSV tables.

Inputs:
- data/interim/test_original.csv
- data/processed/X_test.pkl
- data/processed/category_encoder.pkl
- data/processed/section_encoder.pkl
- models/category/*.pkl
- models/section/*.pkl
- results/metrics/category_metrics.csv
- results/metrics/section_metrics.csv
- results/metrics/llm_category_agreement.csv
- results/metrics/llm_section_agreement.csv

Outputs:
- results/confusion_matrices/category/*.png
- results/confusion_matrices/section/*.png
- results/figures/*.png
- results/tables/*.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIRECTORY = PROJECT_ROOT / "models"

RESULTS_DIRECTORY = PROJECT_ROOT / "results"
METRICS_DIRECTORY = RESULTS_DIRECTORY / "metrics"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"
TABLES_DIRECTORY = RESULTS_DIRECTORY / "tables"

CONFUSION_MATRIX_DIRECTORY = (
    RESULTS_DIRECTORY / "confusion_matrices"
)

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"

CATEGORY_ENCODER_PATH = (
    PROCESSED_DATA_DIR / "category_encoder.pkl"
)

SECTION_ENCODER_PATH = (
    PROCESSED_DATA_DIR / "section_encoder.pkl"
)

CATEGORY_METRICS_PATH = (
    METRICS_DIRECTORY / "category_metrics.csv"
)

SECTION_METRICS_PATH = (
    METRICS_DIRECTORY / "section_metrics.csv"
)

LLM_CATEGORY_AGREEMENT_PATH = (
    METRICS_DIRECTORY / "llm_category_agreement.csv"
)

LLM_SECTION_AGREEMENT_PATH = (
    METRICS_DIRECTORY / "llm_section_agreement.csv"
)

CATEGORY_SUMMARY_PATH = (
    TABLES_DIRECTORY / "category_model_summary.csv"
)

SECTION_SUMMARY_PATH = (
    TABLES_DIRECTORY / "section_model_summary.csv"
)

LLM_AGREEMENT_SUMMARY_PATH = (
    TABLES_DIRECTORY / "llm_label_agreement_summary.csv"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.io import load_pickle, read_csv, write_csv
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

TARGET_CATEGORY = "category"
TARGET_SECTION = "section"

LABEL_SOURCE_REAL = "real"
LABEL_SOURCE_LLM = "llm"

MODEL_LOGISTIC = "logistic"
MODEL_SVM = "svm"
MODEL_NAIVE_BAYES = "nb"

SUPPORTED_MODELS: tuple[str, ...] = (
    MODEL_LOGISTIC,
    MODEL_SVM,
    MODEL_NAIVE_BAYES,
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    MODEL_LOGISTIC: "Logistic Regression",
    MODEL_SVM: "Linear SVM",
    MODEL_NAIVE_BAYES: "Multinomial Naive Bayes",
}

LABEL_SOURCE_DISPLAY_NAMES: dict[str, str] = {
    LABEL_SOURCE_REAL: "Real Labels",
    LABEL_SOURCE_LLM: "LLM Labels",
}

MACRO_METRICS: tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

WEIGHTED_METRICS: tuple[str, ...] = (
    "accuracy",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)

METRIC_DISPLAY_NAMES: dict[str, str] = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
    "weighted_precision": "Weighted Precision",
    "weighted_recall": "Weighted Recall",
    "weighted_f1": "Weighted F1",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


class FigureGenerationError(RuntimeError):
    """Raised when figures or tables cannot be generated."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate config/config.yaml."""
    if not config_path.exists():
        raise FigureGenerationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise FigureGenerationError(
            f"Could not parse configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise FigureGenerationError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    classifier_config = config.get("classifiers")

    if not isinstance(dataset_config, dict):
        raise FigureGenerationError(
            "The configuration must contain a dataset section."
        )

    if not isinstance(classifier_config, dict):
        raise FigureGenerationError(
            "The configuration must contain a classifiers section."
        )

    if "test_file" not in dataset_config:
        raise FigureGenerationError(
            "Missing dataset.test_file in config.yaml."
        )

    missing_models = (
        set(SUPPORTED_MODELS) - set(classifier_config)
    )

    if missing_models:
        missing = ", ".join(sorted(missing_models))

        raise FigureGenerationError(
            f"Missing classifier settings: {missing}"
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_enabled_models(
    classifier_config: dict[str, Any],
) -> list[str]:
    """Return the classifier names enabled in config.yaml."""
    enabled_models: list[str] = []

    for model_name in SUPPORTED_MODELS:
        setting = classifier_config[model_name]

        if isinstance(setting, bool):
            enabled = setting

        elif isinstance(setting, dict):
            enabled = bool(setting.get("enabled", True))

        else:
            raise FigureGenerationError(
                f"classifiers.{model_name} must be a Boolean "
                "or YAML mapping."
            )

        if enabled:
            enabled_models.append(model_name)

    if not enabled_models:
        raise FigureGenerationError(
            "At least one classifier must be enabled."
        )

    return enabled_models


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def load_required_csv(
    file_path: Path,
    file_name: str,
) -> pd.DataFrame:
    """Load a required CSV file."""
    if not file_path.exists():
        raise FigureGenerationError(
            f"{file_name} not found: {file_path}"
        )

    dataframe = read_csv(file_path)

    if dataframe.empty:
        raise FigureGenerationError(
            f"{file_name} contains no rows."
        )

    return dataframe


def validate_test_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """Validate fields required for confusion matrices."""
    required_columns = {
        ID_COLUMN,
        CATEGORY_COLUMN,
        SECTION_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise FigureGenerationError(
            f"The testing dataset is missing columns: {missing}"
        )

    for column in (
        CATEGORY_COLUMN,
        SECTION_COLUMN,
    ):
        if dataframe[column].isna().any():
            raise FigureGenerationError(
                f"The testing dataset contains missing {column} values."
            )


def validate_metric_dataframe(
    dataframe: pd.DataFrame,
    metric_name: str,
) -> None:
    """Validate a classifier metrics CSV."""
    required_columns = {
        "target",
        "model",
        "training_labels",
        "metric_scope",
        "class_label",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "support",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise FigureGenerationError(
            f"{metric_name} is missing columns: {missing}"
        )


def validate_agreement_dataframe(
    dataframe: pd.DataFrame,
    metric_name: str,
) -> None:
    """Validate a direct LLM agreement metrics CSV."""
    required_columns = {
        "target",
        "metric_scope",
        "class_label",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "support",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise FigureGenerationError(
            f"{metric_name} is missing columns: {missing}"
        )


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(
    dataframe: pd.DataFrame,
    label_column: str,
    encoder: Any,
) -> np.ndarray:
    """Encode test labels using a saved LabelEncoder."""
    labels = (
        dataframe[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unknown_labels = set(labels) - set(encoder.classes_)

    if unknown_labels:
        unknown = ", ".join(sorted(unknown_labels))

        raise FigureGenerationError(
            f"Testing data contains labels not recognized by the "
            f"{label_column} encoder: {unknown}"
        )

    return encoder.transform(labels)


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def get_model_path(
    target_name: str,
    model_name: str,
    label_source: str,
) -> Path:
    """Return the saved path of a trained model."""
    return (
        MODEL_DIRECTORY
        / target_name
        / f"{model_name}_{label_source}.pkl"
    )


def create_confusion_matrix_figure(
    X_test: Any,
    y_test: np.ndarray,
    encoder: Any,
    target_name: str,
    model_name: str,
    label_source: str,
) -> None:
    """Generate and save one normalized confusion matrix."""
    model_path = get_model_path(
        target_name=target_name,
        model_name=model_name,
        label_source=label_source,
    )

    if not model_path.exists():
        raise FigureGenerationError(
            f"Trained model not found: {model_path}"
        )

    model = load_pickle(model_path)

    try:
        predictions = np.asarray(model.predict(X_test))

    except Exception as error:
        raise FigureGenerationError(
            f"Prediction failed for "
            f"{target_name}/{model_name}/{label_source}: {error}"
        ) from error

    encoded_labels = np.arange(len(encoder.classes_))

    matrix = confusion_matrix(
        y_true=y_test,
        y_pred=predictions,
        labels=encoded_labels,
        normalize="true",
    )

    figure_width = max(
        10.0,
        len(encoder.classes_) * 0.45,
    )

    figure_height = max(
        8.0,
        len(encoder.classes_) * 0.40,
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=encoder.classes_,
    )

    display.plot(
        ax=axis,
        include_values=len(encoder.classes_) <= 15,
        xticks_rotation=90,
        values_format=".2f",
        colorbar=True,
    )

    axis.set_title(
        f"{target_name.capitalize()} Confusion Matrix\n"
        f"{MODEL_DISPLAY_NAMES[model_name]}, "
        f"Trained on "
        f"{LABEL_SOURCE_DISPLAY_NAMES[label_source]}"
    )

    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("Original Test Label")

    figure.tight_layout()

    output_directory = (
        CONFUSION_MATRIX_DIRECTORY / target_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{model_name}_{label_source}.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Saved confusion matrix to %s.",
        output_path,
    )


def generate_all_confusion_matrices(
    X_test: Any,
    test_dataframe: pd.DataFrame,
    category_encoder: Any,
    section_encoder: Any,
    enabled_models: list[str],
) -> None:
    """Generate confusion matrices for all model conditions."""
    category_labels = encode_labels(
        dataframe=test_dataframe,
        label_column=CATEGORY_COLUMN,
        encoder=category_encoder,
    )

    section_labels = encode_labels(
        dataframe=test_dataframe,
        label_column=SECTION_COLUMN,
        encoder=section_encoder,
    )

    target_settings = (
        (
            TARGET_CATEGORY,
            category_labels,
            category_encoder,
        ),
        (
            TARGET_SECTION,
            section_labels,
            section_encoder,
        ),
    )

    for target_name, y_test, encoder in target_settings:
        for model_name in enabled_models:
            for label_source in (
                LABEL_SOURCE_REAL,
                LABEL_SOURCE_LLM,
            ):
                create_confusion_matrix_figure(
                    X_test=X_test,
                    y_test=y_test,
                    encoder=encoder,
                    target_name=target_name,
                    model_name=model_name,
                    label_source=label_source,
                )


# ---------------------------------------------------------------------------
# Overall metric preparation
# ---------------------------------------------------------------------------

def extract_overall_metrics(
    metrics_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Extract overall model-level metric rows."""
    overall_metrics = metrics_dataframe[
        metrics_dataframe["metric_scope"] == "overall"
    ].copy()

    required_numeric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    for column in required_numeric_columns:
        overall_metrics[column] = pd.to_numeric(
            overall_metrics[column],
            errors="raise",
        )

    overall_metrics["model_display"] = (
        overall_metrics["model"]
        .map(MODEL_DISPLAY_NAMES)
    )

    overall_metrics["training_labels_display"] = (
        overall_metrics["training_labels"]
        .map(LABEL_SOURCE_DISPLAY_NAMES)
    )

    if overall_metrics["model_display"].isna().any():
        raise FigureGenerationError(
            "Metrics contain an unrecognized model name."
        )

    if (
        overall_metrics[
            "training_labels_display"
        ].isna().any()
    ):
        raise FigureGenerationError(
            "Metrics contain an unrecognized label source."
        )

    overall_metrics["experiment"] = (
        overall_metrics["model_display"]
        + "\n"
        + overall_metrics["training_labels_display"]
    )

    return overall_metrics.reset_index(drop=True)


def create_summary_table(
    overall_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save overall classifier metrics as a CSV table."""
    summary = overall_metrics[
        [
            "target",
            "model_display",
            "training_labels_display",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ]
    ].copy()

    summary = summary.rename(
        columns={
            "model_display": "model",
            "training_labels_display": "training_labels",
        }
    )

    summary = summary.sort_values(
        by=[
            "model",
            "training_labels",
        ]
    ).reset_index(drop=True)

    write_csv(
        dataframe=summary,
        file_path=output_path,
    )

    logger.info(
        "Saved summary table to %s.",
        output_path,
    )


# ---------------------------------------------------------------------------
# Overall metric figures
# ---------------------------------------------------------------------------

def create_overall_metric_figure(
    overall_metrics: pd.DataFrame,
    target_name: str,
    metric_columns: tuple[str, ...],
    metric_group_name: str,
    output_path: Path,
) -> None:
    """
    Create a grouped bar chart for a set of overall metrics.

    Examples:
    - Accuracy, Macro Precision, Macro Recall, Macro F1
    - Accuracy, Weighted Precision, Weighted Recall, Weighted F1
    """
    target_metrics = overall_metrics[
        overall_metrics["target"] == target_name
    ].copy()

    if target_metrics.empty:
        raise FigureGenerationError(
            f"No metrics found for target: {target_name}"
        )

    experiment_order = list(
        target_metrics["experiment"]
    )

    x_positions = np.arange(
        len(experiment_order)
    )

    bar_width = 0.18

    figure, axis = plt.subplots(
        figsize=(15, 8)
    )

    for metric_index, metric_name in enumerate(
        metric_columns
    ):
        values = target_metrics[metric_name].to_numpy()

        offset = (
            metric_index
            - (len(metric_columns) - 1) / 2
        ) * bar_width

        bars = axis.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=METRIC_DISPLAY_NAMES[metric_name],
        )

        axis.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=8,
            rotation=90,
        )

    axis.set_title(
        f"{target_name.capitalize()} Classification "
        f"{metric_group_name} Metrics"
    )

    axis.set_xlabel(
        "Classifier and Training-Label Source"
    )

    axis.set_ylabel("Score")

    axis.set_xticks(
        x_positions,
        experiment_order,
        rotation=30,
        ha="right",
    )

    axis.set_ylim(0.0, 1.08)
    axis.legend()
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Saved overall metric figure to %s.",
        output_path,
    )


# ---------------------------------------------------------------------------
# Per-class metric heatmaps
# ---------------------------------------------------------------------------

def extract_class_metrics(
    metrics_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Extract and clean per-class metric rows."""
    class_metrics = metrics_dataframe[
        metrics_dataframe["metric_scope"] == "class"
    ].copy()

    for column in (
        "precision",
        "recall",
        "f1",
        "support",
    ):
        class_metrics[column] = pd.to_numeric(
            class_metrics[column],
            errors="raise",
        )

    class_metrics["model_display"] = (
        class_metrics["model"]
        .map(MODEL_DISPLAY_NAMES)
    )

    class_metrics["training_labels_display"] = (
        class_metrics["training_labels"]
        .map(LABEL_SOURCE_DISPLAY_NAMES)
    )

    class_metrics["experiment"] = (
        class_metrics["model_display"]
        + " - "
        + class_metrics["training_labels_display"]
    )

    return class_metrics.reset_index(drop=True)


def create_per_class_heatmap(
    class_metrics: pd.DataFrame,
    target_name: str,
    metric_name: str,
    output_path: Path,
) -> None:
    """
    Create a heatmap showing one per-class metric for every experiment.

    Rows:
        Classifier and training-label source.

    Columns:
        Category or Section labels.
    """
    target_metrics = class_metrics[
        class_metrics["target"] == target_name
    ].copy()

    if target_metrics.empty:
        raise FigureGenerationError(
            f"No per-class metrics found for target: {target_name}"
        )

    pivot_table = target_metrics.pivot_table(
        index="experiment",
        columns="class_label",
        values=metric_name,
        aggfunc="first",
    )

    pivot_table = pivot_table.sort_index()

    figure_width = max(
        12.0,
        len(pivot_table.columns) * 0.45,
    )

    figure_height = max(
        6.0,
        len(pivot_table.index) * 0.65,
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    image = axis.imshow(
        pivot_table.to_numpy(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    axis.set_title(
        f"{target_name.capitalize()} Per-Class "
        f"{METRIC_DISPLAY_NAMES[metric_name]}"
    )

    axis.set_xlabel(
        f"{target_name.capitalize()} Label"
    )

    axis.set_ylabel(
        "Classifier and Training-Label Source"
    )

    axis.set_xticks(
        np.arange(len(pivot_table.columns)),
        pivot_table.columns,
        rotation=90,
    )

    axis.set_yticks(
        np.arange(len(pivot_table.index)),
        pivot_table.index,
    )

    if len(pivot_table.columns) <= 12:
        for row_index in range(
            len(pivot_table.index)
        ):
            for column_index in range(
                len(pivot_table.columns)
            ):
                value = pivot_table.iloc[
                    row_index,
                    column_index,
                ]

                if not pd.isna(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        METRIC_DISPLAY_NAMES[metric_name]
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Saved per-class heatmap to %s.",
        output_path,
    )


def generate_per_class_heatmaps(
    class_metrics: pd.DataFrame,
    target_name: str,
) -> None:
    """Generate precision, recall, and F1 heatmaps."""
    for metric_name in (
        "precision",
        "recall",
        "f1",
    ):
        output_path = (
            FIGURES_DIRECTORY
            / f"{target_name}_per_class_{metric_name}.png"
        )

        create_per_class_heatmap(
            class_metrics=class_metrics,
            target_name=target_name,
            metric_name=metric_name,
            output_path=output_path,
        )


# ---------------------------------------------------------------------------
# Direct LLM agreement figures
# ---------------------------------------------------------------------------

def extract_overall_agreement(
    agreement_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Extract overall direct-label agreement metrics."""
    overall = agreement_dataframe[
        agreement_dataframe["metric_scope"] == "overall"
    ].copy()

    if len(overall) != 1:
        raise FigureGenerationError(
            "Each LLM agreement file must contain exactly "
            "one overall row."
        )

    selected_columns = [
        "target",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    overall = overall[selected_columns]

    for column in selected_columns:
        if column != "target":
            overall[column] = pd.to_numeric(
                overall[column],
                errors="raise",
            )

    return overall.reset_index(drop=True)


def create_llm_agreement_summary(
    category_agreement: pd.DataFrame,
    section_agreement: pd.DataFrame,
) -> pd.DataFrame:
    """Combine Category and Section agreement summaries."""
    return pd.concat(
        [
            extract_overall_agreement(
                category_agreement
            ),
            extract_overall_agreement(
                section_agreement
            ),
        ],
        ignore_index=True,
    )


def create_llm_agreement_figure(
    agreement_summary: pd.DataFrame,
    metric_columns: tuple[str, ...],
    metric_group_name: str,
    output_path: Path,
) -> None:
    """
    Create a direct LLM-label agreement comparison chart.
    """
    target_order = [
        TARGET_CATEGORY,
        TARGET_SECTION,
    ]

    agreement_summary = (
        agreement_summary
        .set_index("target")
        .reindex(target_order)
        .reset_index()
    )

    x_positions = np.arange(
        len(target_order)
    )

    bar_width = 0.18

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    for metric_index, metric_name in enumerate(
        metric_columns
    ):
        offset = (
            metric_index
            - (len(metric_columns) - 1) / 2
        ) * bar_width

        bars = axis.bar(
            x_positions + offset,
            agreement_summary[metric_name],
            width=bar_width,
            label=METRIC_DISPLAY_NAMES[metric_name],
        )

        axis.bar_label(
            bars,
            fmt="%.3f",
            padding=3,
            fontsize=9,
        )

    axis.set_title(
        "Direct Agreement Between Original and "
        f"LLM-Generated Labels\n{metric_group_name} Metrics"
    )

    axis.set_xlabel("Prediction Target")
    axis.set_ylabel("Agreement Score")

    axis.set_xticks(
        x_positions,
        [
            "Category",
            "Section",
        ],
    )

    axis.set_ylim(0.0, 1.08)
    axis.legend()
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Saved LLM agreement figure to %s.",
        output_path,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all project figures and summary tables."""
    logger.info(
        "Starting figure and table generation."
    )

    config = load_config(CONFIG_PATH)

    dataset_config = config["dataset"]
    classifier_config = config["classifiers"]

    enabled_models = get_enabled_models(
        classifier_config
    )

    test_path = resolve_project_path(
        str(dataset_config["test_file"])
    )

    required_files = (
        test_path,
        X_TEST_PATH,
        CATEGORY_ENCODER_PATH,
        SECTION_ENCODER_PATH,
        CATEGORY_METRICS_PATH,
        SECTION_METRICS_PATH,
        LLM_CATEGORY_AGREEMENT_PATH,
        LLM_SECTION_AGREEMENT_PATH,
    )

    for file_path in required_files:
        if not file_path.exists():
            raise FigureGenerationError(
                f"Required file not found: {file_path}"
            )

    test_dataframe = load_required_csv(
        test_path,
        "Testing dataset",
    )

    validate_test_dataset(test_dataframe)

    category_metrics = load_required_csv(
        CATEGORY_METRICS_PATH,
        "Category metrics",
    )

    section_metrics = load_required_csv(
        SECTION_METRICS_PATH,
        "Section metrics",
    )

    category_agreement = load_required_csv(
        LLM_CATEGORY_AGREEMENT_PATH,
        "LLM Category agreement metrics",
    )

    section_agreement = load_required_csv(
        LLM_SECTION_AGREEMENT_PATH,
        "LLM Section agreement metrics",
    )

    validate_metric_dataframe(
        category_metrics,
        "Category metrics",
    )

    validate_metric_dataframe(
        section_metrics,
        "Section metrics",
    )

    validate_agreement_dataframe(
        category_agreement,
        "LLM Category agreement metrics",
    )

    validate_agreement_dataframe(
        section_agreement,
        "LLM Section agreement metrics",
    )

    X_test = load_pickle(X_TEST_PATH)

    category_encoder = load_pickle(
        CATEGORY_ENCODER_PATH
    )

    section_encoder = load_pickle(
        SECTION_ENCODER_PATH
    )

    if X_test.shape[0] != len(test_dataframe):
        raise FigureGenerationError(
            "X_test row count does not match the testing dataset."
        )

    # ------------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------------

    generate_all_confusion_matrices(
        X_test=X_test,
        test_dataframe=test_dataframe,
        category_encoder=category_encoder,
        section_encoder=section_encoder,
        enabled_models=enabled_models,
    )

    # ------------------------------------------------------------------
    # Overall model metrics
    # ------------------------------------------------------------------

    category_overall = extract_overall_metrics(
        category_metrics
    )

    section_overall = extract_overall_metrics(
        section_metrics
    )

    create_summary_table(
        overall_metrics=category_overall,
        output_path=CATEGORY_SUMMARY_PATH,
    )

    create_summary_table(
        overall_metrics=section_overall,
        output_path=SECTION_SUMMARY_PATH,
    )

    create_overall_metric_figure(
        overall_metrics=category_overall,
        target_name=TARGET_CATEGORY,
        metric_columns=MACRO_METRICS,
        metric_group_name="Macro",
        output_path=(
            FIGURES_DIRECTORY
            / "category_macro_metrics.png"
        ),
    )

    create_overall_metric_figure(
        overall_metrics=category_overall,
        target_name=TARGET_CATEGORY,
        metric_columns=WEIGHTED_METRICS,
        metric_group_name="Weighted",
        output_path=(
            FIGURES_DIRECTORY
            / "category_weighted_metrics.png"
        ),
    )

    create_overall_metric_figure(
        overall_metrics=section_overall,
        target_name=TARGET_SECTION,
        metric_columns=MACRO_METRICS,
        metric_group_name="Macro",
        output_path=(
            FIGURES_DIRECTORY
            / "section_macro_metrics.png"
        ),
    )

    create_overall_metric_figure(
        overall_metrics=section_overall,
        target_name=TARGET_SECTION,
        metric_columns=WEIGHTED_METRICS,
        metric_group_name="Weighted",
        output_path=(
            FIGURES_DIRECTORY
            / "section_weighted_metrics.png"
        ),
    )

    # ------------------------------------------------------------------
    # Per-class model metrics
    # ------------------------------------------------------------------

    category_class_metrics = extract_class_metrics(
        category_metrics
    )

    section_class_metrics = extract_class_metrics(
        section_metrics
    )

    generate_per_class_heatmaps(
        class_metrics=category_class_metrics,
        target_name=TARGET_CATEGORY,
    )

    generate_per_class_heatmaps(
        class_metrics=section_class_metrics,
        target_name=TARGET_SECTION,
    )

    # ------------------------------------------------------------------
    # Direct LLM-label agreement
    # ------------------------------------------------------------------

    agreement_summary = create_llm_agreement_summary(
        category_agreement=category_agreement,
        section_agreement=section_agreement,
    )

    write_csv(
        dataframe=agreement_summary,
        file_path=LLM_AGREEMENT_SUMMARY_PATH,
    )

    create_llm_agreement_figure(
        agreement_summary=agreement_summary,
        metric_columns=MACRO_METRICS,
        metric_group_name="Macro",
        output_path=(
            FIGURES_DIRECTORY
            / "llm_label_agreement_macro_metrics.png"
        ),
    )

    create_llm_agreement_figure(
        agreement_summary=agreement_summary,
        metric_columns=WEIGHTED_METRICS,
        metric_group_name="Weighted",
        output_path=(
            FIGURES_DIRECTORY
            / "llm_label_agreement_weighted_metrics.png"
        ),
    )

    logger.info(
        "Figure and table generation completed successfully."
    )


if __name__ == "__main__":
    try:
        main()

    except FigureGenerationError as error:
        logger.error(
            "Figure generation failed: %s",
            error,
        )
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "Figure generation failed because of an unexpected error."
        )
        raise SystemExit(1)