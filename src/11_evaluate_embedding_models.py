"""
11_evaluate_embedding_models.py

Evaluate the Sentence-BERT classifiers trained by
10_train_embedding_models.py against the held-out CNN test set.

The evaluation preserves the metrics used in the project notebook while
integrating them into the repository's reproducible pipeline. Every enabled
classifier is evaluated for four conditions:

1. Category prediction using original CNN training labels.
2. Category prediction using LLM-generated training labels.
3. Section prediction using original CNN training labels.
4. Section prediction using LLM-generated training labels.

All conditions are evaluated against the original CNN test labels. This makes
performance directly comparable with the existing TF-IDF experiments and
measures whether replacing the training labels changes downstream predictive
performance.

Inputs:
- data/interim/test_original.csv
- data/processed/sentence_embeddings/test_embeddings.npy
- data/processed/sentence_embeddings/test_metadata.csv
- models/sentence_bert/<target>/<label_source>/*.pkl

Outputs:
- results/tables/sentence_bert/embedding_model_evaluation_summary.csv
- results/tables/sentence_bert/embedding_per_class_metrics.csv
- results/tables/sentence_bert/embedding_predictions.csv
- results/tables/sentence_bert/confusion_matrices/*.csv
- results/figures/sentence_bert/confusion_matrices/*.png
- results/figures/sentence_bert/embedding_model_performance.png
- results/figures/sentence_bert/embedding_macro_f1_comparison.png
- results/sentence_bert/evaluation_manifest.json
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDING_DIRECTORY = PROCESSED_DATA_DIR / "sentence_embeddings"
TEST_EMBEDDINGS_PATH = EMBEDDING_DIRECTORY / "test_embeddings.npy"
TEST_METADATA_PATH = EMBEDDING_DIRECTORY / "test_metadata.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "sentence_bert"

TABLE_DIRECTORY = PROJECT_ROOT / "results" / "tables" / "sentence_bert"
SUMMARY_PATH = TABLE_DIRECTORY / "embedding_model_evaluation_summary.csv"
PER_CLASS_PATH = TABLE_DIRECTORY / "embedding_per_class_metrics.csv"
PREDICTIONS_PATH = TABLE_DIRECTORY / "embedding_predictions.csv"
REPORT_DIRECTORY = TABLE_DIRECTORY / "classification_reports"
CONFUSION_TABLE_DIRECTORY = TABLE_DIRECTORY / "confusion_matrices"

FIGURE_DIRECTORY = PROJECT_ROOT / "results" / "figures" / "sentence_bert"
CONFUSION_FIGURE_DIRECTORY = FIGURE_DIRECTORY / "confusion_matrices"
PERFORMANCE_FIGURE_PATH = FIGURE_DIRECTORY / "embedding_model_performance.png"
MACRO_F1_FIGURE_PATH = FIGURE_DIRECTORY / "embedding_macro_f1_comparison.png"

MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "sentence_bert" / "evaluation_manifest.json"
)

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
HEADLINE_COLUMN = "Headline"
CATEGORY_COLUMN = "Category"
SECTION_COLUMN = "Section"

TARGET_TO_COLUMN = {
    "category": CATEGORY_COLUMN,
    "section": SECTION_COLUMN,
}

LABEL_SOURCE_DISPLAY = {
    "real": "CNN labels",
    "llm": "LLM labels",
}

MODEL_DISPLAY_NAMES = {
    "logistic": "Logistic Regression",
    "svm": "Linear SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Small Neural Network",
}


class EmbeddingModelEvaluationError(RuntimeError):
    """Raised when embedding models cannot be evaluated safely."""


# ---------------------------------------------------------------------------
# Configuration and input loading
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the project configuration and verify the test-file setting."""
    if not config_path.exists():
        raise EmbeddingModelEvaluationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise EmbeddingModelEvaluationError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise EmbeddingModelEvaluationError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise EmbeddingModelEvaluationError(
            "The configuration must contain a 'dataset' section."
        )
    if "test_file" not in dataset_config:
        raise EmbeddingModelEvaluationError(
            "Missing dataset.test_file in config.yaml."
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_test_dataset(file_path: Path) -> pd.DataFrame:
    """Load and validate the original-label held-out test dataset."""
    if not file_path.exists():
        raise EmbeddingModelEvaluationError(
            f"Test dataset not found: {file_path}"
        )

    test_data = read_csv(file_path)
    if test_data.empty:
        raise EmbeddingModelEvaluationError("The test dataset contains no rows.")

    required_columns = {
        ID_COLUMN,
        HEADLINE_COLUMN,
        CATEGORY_COLUMN,
        SECTION_COLUMN,
    }
    missing_columns = required_columns - set(test_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise EmbeddingModelEvaluationError(
            f"Test dataset is missing required columns: {missing}"
        )

    if test_data[ID_COLUMN].isna().any():
        raise EmbeddingModelEvaluationError(
            f"Test dataset contains missing {ID_COLUMN} values."
        )
    if not test_data[ID_COLUMN].is_unique:
        raise EmbeddingModelEvaluationError(
            f"Test dataset contains duplicate {ID_COLUMN} values."
        )

    for column in (CATEGORY_COLUMN, SECTION_COLUMN):
        if test_data[column].isna().any():
            raise EmbeddingModelEvaluationError(
                f"Test dataset contains missing {column} labels."
            )
        if test_data[column].astype(str).str.strip().eq("").any():
            raise EmbeddingModelEvaluationError(
                f"Test dataset contains blank {column} labels."
            )

    logger.info("Loaded test dataset: %d rows.", len(test_data))
    return test_data


def load_test_embeddings(expected_rows: int) -> np.ndarray:
    """Load and validate the held-out Sentence-BERT embedding matrix."""
    if not TEST_EMBEDDINGS_PATH.exists():
        raise EmbeddingModelEvaluationError(
            "Test embeddings were not found. Run "
            "08_generate_sentence_embeddings.py first: "
            f"{TEST_EMBEDDINGS_PATH}"
        )

    try:
        embeddings = np.load(TEST_EMBEDDINGS_PATH, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise EmbeddingModelEvaluationError(
            f"Could not load test embeddings: {error}"
        ) from error

    if embeddings.ndim != 2:
        raise EmbeddingModelEvaluationError(
            "Test embeddings must be a two-dimensional matrix."
        )
    if embeddings.shape[0] != expected_rows:
        raise EmbeddingModelEvaluationError(
            "Embedding row count does not match the test dataset. "
            f"Embeddings: {embeddings.shape[0]}; rows: {expected_rows}."
        )
    if embeddings.shape[1] <= 0:
        raise EmbeddingModelEvaluationError(
            "Test embeddings contain no features."
        )
    if not np.issubdtype(embeddings.dtype, np.number):
        raise EmbeddingModelEvaluationError(
            "Test embeddings must contain numeric values."
        )
    if not np.isfinite(embeddings).all():
        raise EmbeddingModelEvaluationError(
            "Test embeddings contain NaN or infinite values."
        )

    logger.info(
        "Loaded test embeddings: %d rows and %d dimensions.",
        embeddings.shape[0],
        embeddings.shape[1],
    )
    return np.asarray(embeddings, dtype=np.float32)


def validate_embedding_metadata(test_data: pd.DataFrame) -> None:
    """Confirm that embedding rows align exactly with test-dataset rows."""
    if not TEST_METADATA_PATH.exists():
        raise EmbeddingModelEvaluationError(
            "Test embedding metadata was not found. Run "
            "08_generate_sentence_embeddings.py first: "
            f"{TEST_METADATA_PATH}"
        )

    metadata = read_csv(TEST_METADATA_PATH)
    required_columns = {ID_COLUMN, "embedding_row"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise EmbeddingModelEvaluationError(
            f"Test embedding metadata is missing columns: {missing}"
        )

    if len(metadata) != len(test_data):
        raise EmbeddingModelEvaluationError(
            "Test embedding metadata row count does not match the test dataset."
        )

    expected_rows = np.arange(len(metadata), dtype=int)
    actual_rows = pd.to_numeric(
        metadata["embedding_row"], errors="coerce"
    ).to_numpy()
    if np.isnan(actual_rows).any() or not np.array_equal(
        actual_rows.astype(int), expected_rows
    ):
        raise EmbeddingModelEvaluationError(
            "Test embedding metadata has invalid or reordered embedding_row values."
        )

    if list(metadata[ID_COLUMN].astype(str)) != list(
        test_data[ID_COLUMN].astype(str)
    ):
        raise EmbeddingModelEvaluationError(
            "Test embedding metadata does not align with test dataset IDs."
        )

    logger.info("Test embedding metadata is aligned with the test dataset.")


# ---------------------------------------------------------------------------
# Model discovery and prediction
# ---------------------------------------------------------------------------


def discover_model_paths() -> list[Path]:
    """Find all trained Sentence-BERT model bundles in stable order."""
    if not MODEL_DIRECTORY.exists():
        raise EmbeddingModelEvaluationError(
            "Sentence-BERT model directory was not found. Run "
            f"10_train_embedding_models.py first: {MODEL_DIRECTORY}"
        )

    paths = sorted(MODEL_DIRECTORY.glob("*/*/*.pkl"))
    if not paths:
        raise EmbeddingModelEvaluationError(
            "No Sentence-BERT model artifacts were found. Run "
            "10_train_embedding_models.py first."
        )

    logger.info("Discovered %d embedding model artifacts.", len(paths))
    return paths


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load and validate one model bundle produced by stage 10."""
    try:
        with model_path.open("rb") as file:
            bundle = pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        raise EmbeddingModelEvaluationError(
            f"Could not load model artifact {model_path}: {error}"
        ) from error

    if not isinstance(bundle, dict):
        raise EmbeddingModelEvaluationError(
            f"Model artifact does not contain a bundle dictionary: {model_path}"
        )

    required_keys = {
        "representation",
        "model_name",
        "target",
        "label_source",
        "classifier",
        "label_encoder",
        "embedding_dimension",
    }
    missing_keys = required_keys - set(bundle)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise EmbeddingModelEvaluationError(
            f"Model artifact {model_path} is missing keys: {missing}"
        )

    if bundle["representation"] != "sentence_bert":
        raise EmbeddingModelEvaluationError(
            f"Unexpected representation in {model_path}: "
            f"{bundle['representation']}"
        )
    if bundle["target"] not in TARGET_TO_COLUMN:
        raise EmbeddingModelEvaluationError(
            f"Unsupported target in {model_path}: {bundle['target']}"
        )
    if bundle["label_source"] not in LABEL_SOURCE_DISPLAY:
        raise EmbeddingModelEvaluationError(
            f"Unsupported label source in {model_path}: "
            f"{bundle['label_source']}"
        )
    if not isinstance(bundle["label_encoder"], LabelEncoder):
        raise EmbeddingModelEvaluationError(
            f"Invalid LabelEncoder in model artifact: {model_path}"
        )
    if not hasattr(bundle["classifier"], "predict"):
        raise EmbeddingModelEvaluationError(
            f"Classifier in {model_path} does not support predict()."
        )

    return bundle


def generate_predictions(
    bundle: dict[str, Any],
    embeddings: np.ndarray,
    model_path: Path,
) -> np.ndarray:
    """Predict encoded labels and convert them back to string labels."""
    expected_dimension = int(bundle["embedding_dimension"])
    if embeddings.shape[1] != expected_dimension:
        raise EmbeddingModelEvaluationError(
            f"Embedding dimension mismatch for {model_path}. Model expects "
            f"{expected_dimension}; test matrix has {embeddings.shape[1]}."
        )

    classifier = bundle["classifier"]
    encoder: LabelEncoder = bundle["label_encoder"]

    try:
        encoded_predictions = np.asarray(classifier.predict(embeddings))
    except Exception as error:
        raise EmbeddingModelEvaluationError(
            f"Prediction failed for {model_path}: {error}"
        ) from error

    if encoded_predictions.ndim != 1:
        encoded_predictions = encoded_predictions.reshape(-1)
    if len(encoded_predictions) != len(embeddings):
        raise EmbeddingModelEvaluationError(
            f"Prediction count mismatch for {model_path}."
        )

    try:
        encoded_predictions = encoded_predictions.astype(np.int64)
        predictions = encoder.inverse_transform(encoded_predictions)
    except (ValueError, TypeError) as error:
        raise EmbeddingModelEvaluationError(
            f"Could not decode predictions from {model_path}: {error}"
        ) from error

    return np.asarray(predictions, dtype=str)


# ---------------------------------------------------------------------------
# Metric calculation and outputs
# ---------------------------------------------------------------------------


def safe_slug(value: str) -> str:
    """Convert a label or name to a filesystem-safe lowercase slug."""
    return "".join(
        character if character.isalnum() else "_"
        for character in value.lower()
    ).strip("_")


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Calculate aggregate classification metrics."""
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_precision": precision_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            actual, predicted, labels=labels, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            actual, predicted, labels=labels, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(
            actual, predicted, labels=labels, average="weighted", zero_division=0
        ),
    }


def create_per_class_rows(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
    metadata: dict[str, str],
) -> list[dict[str, Any]]:
    """Return one precision/recall/F1/support row per evaluation class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        zero_division=0,
    )

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        rows.append(
            {
                **metadata,
                "class_label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "seen_during_training": label in metadata["training_classes"].split(" | "),
            }
        )
    return rows


def save_classification_report(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    """Save sklearn's detailed classification report as a CSV table."""
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose().reset_index()
    report_frame = report_frame.rename(columns={"index": "label_or_average"})
    write_csv(report_frame, output_path)


def save_confusion_outputs(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    file_stem: str,
) -> tuple[Path, Path]:
    """Save a confusion matrix as both CSV and publication-ready PNG."""
    table_path = CONFUSION_TABLE_DIRECTORY / f"{file_stem}.csv"
    figure_path = CONFUSION_FIGURE_DIRECTORY / f"{file_stem}.png"

    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = "actual_label"
    matrix_frame.columns.name = "predicted_label"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_frame.to_csv(table_path, encoding="utf-8")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    label_count = len(labels)
    figure_size = max(8.0, min(18.0, 0.48 * label_count + 5.0))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, aspect="auto")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_xticks(np.arange(label_count))
    axis.set_yticks(np.arange(label_count))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)

    # Annotate smaller matrices; Section matrices can become unreadable otherwise.
    if label_count <= 20:
        threshold = matrix.max() / 2.0 if matrix.size else 0
        for row in range(label_count):
            for column in range(label_count):
                value = int(matrix[row, column])
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )

    figure.tight_layout()
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return table_path, figure_path


def create_performance_figure(summary: pd.DataFrame) -> None:
    """Plot accuracy, macro F1, and weighted F1 for every experiment."""
    if summary.empty:
        return

    plot_data = summary.copy()
    plot_data["experiment"] = (
        plot_data["target"].str.title()
        + " | "
        + plot_data["label_source_display"]
        + " | "
        + plot_data["model_display_name"]
    )
    plot_data = plot_data.sort_values(
        ["target", "label_source", "macro_f1"],
        ascending=[True, True, False],
    )

    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]
    positions = np.arange(len(plot_data))
    width = 0.25

    figure_height = max(8.0, len(plot_data) * 0.42)
    figure, axis = plt.subplots(figsize=(13, figure_height))
    for index, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        axis.barh(
            positions + (index - 1) * width,
            plot_data[metric],
            height=width,
            label=metric_name,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(plot_data["experiment"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.set_title("Sentence-BERT Classifier Performance")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.3)
    axis.invert_yaxis()

    PERFORMANCE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(PERFORMANCE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_macro_f1_figure(summary: pd.DataFrame) -> None:
    """Compare macro F1 across targets, label sources, and classifiers."""
    if summary.empty:
        return

    plot_data = summary.copy()
    plot_data["condition"] = (
        plot_data["target"].str.title()
        + "\n"
        + plot_data["label_source_display"]
    )
    pivot = plot_data.pivot_table(
        index="model_display_name",
        columns="condition",
        values="macro_f1",
        aggfunc="first",
    )
    pivot = pivot.reindex(
        [name for name in MODEL_DISPLAY_NAMES.values() if name in pivot.index]
    )

    figure, axis = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Classifier")
    axis.set_ylabel("Macro F1")
    axis.set_title("Sentence-BERT Macro F1 by Training-Label Source")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Evaluation condition")
    axis.grid(axis="y", alpha=0.3)

    MACRO_F1_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(MACRO_F1_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_json_atomic(value: dict[str, Any], output_path: Path) -> None:
    """Write a formatted JSON object atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary_path.replace(output_path)
    except (OSError, TypeError) as error:
        temporary_path.unlink(missing_ok=True)
        raise EmbeddingModelEvaluationError(
            f"Could not save JSON output {output_path}: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


def run_embedding_model_evaluation(config_path: Path = CONFIG_PATH) -> None:
    """Evaluate every trained Sentence-BERT classifier."""
    logger.info("Starting Sentence-BERT model evaluation.")
    stage_start = time.perf_counter()

    config = load_config(config_path)
    test_path = resolve_project_path(str(config["dataset"]["test_file"]))
    test_data = load_test_dataset(test_path)
    validate_embedding_metadata(test_data)
    embeddings = load_test_embeddings(expected_rows=len(test_data))
    model_paths = discover_model_paths()

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    evaluated_artifacts: list[dict[str, Any]] = []

    for model_path in model_paths:
        bundle = load_model_bundle(model_path)
        target = str(bundle["target"])
        label_source = str(bundle["label_source"])
        model_name = str(bundle["model_name"])
        model_display_name = str(
            bundle.get(
                "model_display_name",
                MODEL_DISPLAY_NAMES.get(model_name, model_name),
            )
        )
        label_column = TARGET_TO_COLUMN[target]
        encoder: LabelEncoder = bundle["label_encoder"]
        training_classes = [str(value) for value in encoder.classes_]

        actual = test_data[label_column].astype(str).str.strip().to_numpy()
        prediction_start = time.perf_counter()
        predicted = generate_predictions(bundle, embeddings, model_path)
        prediction_seconds = time.perf_counter() - prediction_start

        # Include labels appearing only in the held-out set. Such classes cannot
        # be predicted by the model but must contribute zero recall and F1 rather
        # than causing the Section experiment to be skipped.
        labels = sorted(set(actual) | set(predicted) | set(training_classes))
        metrics = evaluate_predictions(actual, predicted, labels)

        unseen_test_classes = sorted(set(actual) - set(training_classes))
        file_stem = "_".join(
            [safe_slug(target), safe_slug(label_source), safe_slug(model_name)]
        )

        matrix = confusion_matrix(actual, predicted, labels=labels)
        confusion_table_path, confusion_figure_path = save_confusion_outputs(
            matrix=matrix,
            labels=labels,
            title=(
                f"{target.title()}: {model_display_name} | "
                f"{LABEL_SOURCE_DISPLAY[label_source]}"
            ),
            file_stem=file_stem,
        )

        report_path = REPORT_DIRECTORY / f"{file_stem}.csv"
        save_classification_report(
            actual=actual,
            predicted=predicted,
            labels=labels,
            output_path=report_path,
        )

        correct = actual == predicted
        prediction_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: test_data[ID_COLUMN].to_numpy(),
                    HEADLINE_COLUMN: test_data[HEADLINE_COLUMN].fillna("").astype(str),
                    "target": target,
                    "label_source": label_source,
                    "label_source_display": LABEL_SOURCE_DISPLAY[label_source],
                    "model": model_name,
                    "model_display_name": model_display_name,
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "correct": correct,
                    "actual_label_seen_during_training": np.isin(
                        actual, training_classes
                    ),
                }
            )
        )

        metadata = {
            "representation": "Sentence-BERT",
            "target": target,
            "label_source": label_source,
            "label_source_display": LABEL_SOURCE_DISPLAY[label_source],
            "model": model_name,
            "model_display_name": model_display_name,
            "training_classes": " | ".join(training_classes),
        }
        per_class_rows.extend(
            create_per_class_rows(actual, predicted, labels, metadata)
        )

        summary_row = {
            **metadata,
            "test_rows": int(len(actual)),
            "training_class_count": int(len(training_classes)),
            "evaluation_class_count": int(len(labels)),
            "unseen_test_class_count": int(len(unseen_test_classes)),
            "unseen_test_classes": " | ".join(unseen_test_classes),
            **metrics,
            "prediction_time_seconds": prediction_seconds,
            "predictions_per_second": (
                len(actual) / prediction_seconds if prediction_seconds > 0 else np.nan
            ),
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "classification_report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_table_path": str(
                confusion_table_path.relative_to(PROJECT_ROOT)
            ),
            "confusion_matrix_figure_path": str(
                confusion_figure_path.relative_to(PROJECT_ROOT)
            ),
            "status": "success",
        }
        summary_rows.append(summary_row)

        evaluated_artifacts.append(
            {
                "target": target,
                "label_source": label_source,
                "model": model_name,
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
                "unseen_test_classes": unseen_test_classes,
            }
        )

        logger.info(
            "Evaluated %s for %s/%s: accuracy=%.4f, macro_f1=%.4f, "
            "weighted_f1=%.4f.",
            model_display_name,
            target,
            label_source,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["target", "label_source", "macro_f1"],
        ascending=[True, True, False],
    )
    per_class = pd.DataFrame(per_class_rows).sort_values(
        ["target", "label_source", "model", "class_label"]
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)

    write_csv(summary, SUMMARY_PATH)
    write_csv(per_class, PER_CLASS_PATH)
    write_csv(predictions, PREDICTIONS_PATH)

    create_performance_figure(summary)
    create_macro_f1_figure(summary)

    total_elapsed = time.perf_counter() - stage_start
    manifest = {
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "sentence_bert",
        "evaluation_truth": "original CNN test labels",
        "test_rows": int(len(test_data)),
        "embedding_dimension": int(embeddings.shape[1]),
        "models_evaluated": int(len(summary)),
        "total_evaluation_time_seconds": total_elapsed,
        "evaluated_artifacts": evaluated_artifacts,
        "outputs": {
            "summary": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "per_class_metrics": str(PER_CLASS_PATH.relative_to(PROJECT_ROOT)),
            "predictions": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "classification_report_directory": str(
                REPORT_DIRECTORY.relative_to(PROJECT_ROOT)
            ),
            "confusion_matrix_table_directory": str(
                CONFUSION_TABLE_DIRECTORY.relative_to(PROJECT_ROOT)
            ),
            "confusion_matrix_figure_directory": str(
                CONFUSION_FIGURE_DIRECTORY.relative_to(PROJECT_ROOT)
            ),
            "performance_figure": str(
                PERFORMANCE_FIGURE_PATH.relative_to(PROJECT_ROOT)
            ),
            "macro_f1_figure": str(
                MACRO_F1_FIGURE_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    }
    write_json_atomic(manifest, MANIFEST_PATH)

    logger.info(
        "Sentence-BERT model evaluation completed: %d models in %.3f seconds.",
        len(summary),
        total_elapsed,
    )


if __name__ == "__main__":
    try:
        run_embedding_model_evaluation()
    except EmbeddingModelEvaluationError:
        logger.exception("Sentence-BERT model evaluation failed.")
        raise
    except Exception:
        logger.exception(
            "Sentence-BERT model evaluation failed unexpectedly."
        )
        raise
