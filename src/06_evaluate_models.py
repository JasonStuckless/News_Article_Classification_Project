"""
06_evaluate_models.py

Evaluate all trained classification models against the original test labels.

This script evaluates:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

For both prediction targets:

- Category
- Section

And both training-label sources:

- Original labels
- LLM-generated labels

All models are evaluated using the same held-out test set and the original
test labels as ground truth.

The script also measures direct agreement between the original and
LLM-generated training labels.

Inputs:
- data/interim/train_original.csv
- data/interim/train_llm_labeled.csv
- data/interim/test_original.csv
- data/processed/X_test.pkl
- data/processed/category_encoder.pkl
- data/processed/section_encoder.pkl
- Trained model files in models/category/ and models/section/

Outputs:
- results/metrics/category_metrics.csv
- results/metrics/section_metrics.csv
- results/metrics/llm_category_agreement.csv
- results/metrics/llm_section_agreement.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
METRICS_DIRECTORY = PROJECT_ROOT / "results" / "metrics"

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


class ModelEvaluationError(RuntimeError):
    """Raised when model evaluation cannot be completed safely."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate config/config.yaml.
    """
    if not config_path.exists():
        raise ModelEvaluationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise ModelEvaluationError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise ModelEvaluationError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    classifier_config = config.get("classifiers")

    if not isinstance(dataset_config, dict):
        raise ModelEvaluationError(
            "The configuration must contain a 'dataset' section."
        )

    if not isinstance(classifier_config, dict):
        raise ModelEvaluationError(
            "The configuration must contain a 'classifiers' section."
        )

    required_dataset_settings = {
        "train_file",
        "test_file",
        "llm_train_file",
    }

    missing_dataset_settings = (
        required_dataset_settings - set(dataset_config)
    )

    if missing_dataset_settings:
        missing = ", ".join(
            sorted(missing_dataset_settings)
        )

        raise ModelEvaluationError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    missing_classifier_settings = (
        set(SUPPORTED_MODELS) - set(classifier_config)
    )

    if missing_classifier_settings:
        missing = ", ".join(
            sorted(missing_classifier_settings)
        )

        raise ModelEvaluationError(
            f"Missing classifier settings in config.yaml: {missing}"
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
    """
    Return classifier names enabled in config.yaml.

    Supports either:

    classifiers:
      logistic: true

    or:

    classifiers:
      logistic:
        enabled: true
    """
    enabled_models: list[str] = []

    for model_name in SUPPORTED_MODELS:
        setting = classifier_config[model_name]

        if isinstance(setting, bool):
            enabled = setting

        elif isinstance(setting, dict):
            enabled = bool(setting.get("enabled", True))

        else:
            raise ModelEvaluationError(
                f"classifiers.{model_name} must be a Boolean "
                "or YAML mapping."
            )

        if enabled:
            enabled_models.append(model_name)

    if not enabled_models:
        raise ModelEvaluationError(
            "At least one classifier must be enabled."
        )

    return enabled_models


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

def load_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Load and validate a dataset used during evaluation."""
    if not file_path.exists():
        raise ModelEvaluationError(
            f"{dataset_name} not found: {file_path}"
        )

    dataframe = read_csv(file_path)

    if dataframe.empty:
        raise ModelEvaluationError(
            f"{dataset_name} contains no rows."
        )

    required_columns = {
        ID_COLUMN,
        CATEGORY_COLUMN,
        SECTION_COLUMN,
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ModelEvaluationError(
            f"{dataset_name} is missing required columns: {missing}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise ModelEvaluationError(
            f"{dataset_name} contains missing {ID_COLUMN} values."
        )

    if not dataframe[ID_COLUMN].is_unique:
        raise ModelEvaluationError(
            f"{dataset_name} contains duplicated {ID_COLUMN} values."
        )

    for label_column in (
        CATEGORY_COLUMN,
        SECTION_COLUMN,
    ):
        if dataframe[label_column].isna().any():
            raise ModelEvaluationError(
                f"{dataset_name} contains missing "
                f"{label_column} values."
            )

        blank_mask = (
            dataframe[label_column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if blank_mask.any():
            raise ModelEvaluationError(
                f"{dataset_name} contains blank "
                f"{label_column} values."
            )

    logger.info(
        "Loaded %s: %d rows.",
        dataset_name,
        len(dataframe),
    )

    return dataframe


def validate_training_alignment(
    original_training: pd.DataFrame,
    llm_training: pd.DataFrame,
) -> None:
    """
    Verify that the real-label and LLM-label training datasets contain
    the same articles in the same order.
    """
    if len(original_training) != len(llm_training):
        raise ModelEvaluationError(
            "The original-label and LLM-label training datasets "
            "have different row counts."
        )

    if list(original_training[ID_COLUMN]) != list(
        llm_training[ID_COLUMN]
    ):
        raise ModelEvaluationError(
            "The original-label and LLM-label training datasets "
            "do not have the same article order."
        )


def validate_test_matrix(
    X_test: Any,
    expected_rows: int,
) -> None:
    """Verify that X_test matches the held-out test dataset."""
    if not hasattr(X_test, "shape"):
        raise ModelEvaluationError(
            "X_test does not provide a valid matrix shape."
        )

    if len(X_test.shape) != 2:
        raise ModelEvaluationError(
            "X_test must be a two-dimensional feature matrix."
        )

    if X_test.shape[0] != expected_rows:
        raise ModelEvaluationError(
            "X_test row count does not match the testing dataset. "
            f"Features: {X_test.shape[0]}; "
            f"test rows: {expected_rows}."
        )

    if X_test.shape[1] <= 0:
        raise ModelEvaluationError(
            "X_test contains no features."
        )


# ---------------------------------------------------------------------------
# Label preparation
# ---------------------------------------------------------------------------

def encode_labels(
    dataframe: pd.DataFrame,
    label_column: str,
    encoder: Any,
    dataset_name: str,
) -> np.ndarray:
    """Encode labels using a previously saved LabelEncoder."""
    labels = (
        dataframe[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unknown_labels = (
        set(labels) - set(encoder.classes_)
    )

    if unknown_labels:
        unknown = ", ".join(sorted(unknown_labels))

        raise ModelEvaluationError(
            f"{dataset_name} contains labels not recognized by "
            f"the {label_column} encoder: {unknown}"
        )

    return encoder.transform(labels)


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def calculate_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate overall classification metrics.

    Returns:
    - Accuracy
    - Macro precision, recall, and F1
    - Weighted precision, recall, and F1
    """
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: Any,
) -> list[dict[str, Any]]:
    """
    Calculate precision, recall, F1, and support for every valid class.

    Classes absent from the test set receive support 0.
    """
    encoded_class_values = np.arange(
        len(encoder.classes_)
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=encoded_class_values,
            average=None,
            zero_division=0,
        )
    )

    records: list[dict[str, Any]] = []

    for class_index, class_name in enumerate(
        encoder.classes_
    ):
        records.append(
            {
                "class_label": class_name,
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
        )

    return records


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def get_model_path(
    target_name: str,
    model_name: str,
    label_source: str,
) -> Path:
    """Return the path of one trained model."""
    return (
        MODEL_DIRECTORY
        / target_name
        / f"{model_name}_{label_source}.pkl"
    )


def evaluate_one_model(
    X_test: Any,
    y_test: np.ndarray,
    encoder: Any,
    target_name: str,
    model_name: str,
    label_source: str,
) -> list[dict[str, Any]]:
    """
    Load and evaluate one trained classifier.

    Returns one overall metrics row and one row for every class.
    """
    model_path = get_model_path(
        target_name=target_name,
        model_name=model_name,
        label_source=label_source,
    )

    if not model_path.exists():
        raise ModelEvaluationError(
            f"Trained model not found: {model_path}. "
            "Run 05_train_models.py first."
        )

    model = load_pickle(model_path)

    logger.info(
        "Evaluating %s model for %s using %s training labels.",
        model_name,
        target_name,
        label_source,
    )

    try:
        predictions = model.predict(X_test)

    except Exception as error:
        raise ModelEvaluationError(
            f"Prediction failed for model={model_name}, "
            f"target={target_name}, "
            f"label source={label_source}: {error}"
        ) from error

    predictions = np.asarray(predictions)

    if len(predictions) != len(y_test):
        raise ModelEvaluationError(
            f"{model_name} produced an unexpected prediction count "
            f"for {target_name} using {label_source} labels."
        )

    valid_encoded_labels = set(
        range(len(encoder.classes_))
    )

    unexpected_predictions = (
        set(predictions) - valid_encoded_labels
    )

    if unexpected_predictions:
        unexpected = ", ".join(
            str(value)
            for value in sorted(unexpected_predictions)
        )

        raise ModelEvaluationError(
            f"The {model_name} model produced invalid encoded "
            f"predictions: {unexpected}"
        )

    summary_metrics = calculate_summary_metrics(
        y_true=y_test,
        y_pred=predictions,
    )

    records: list[dict[str, Any]] = [
        {
            "target": target_name,
            "model": model_name,
            "training_labels": label_source,
            "metric_scope": "overall",
            "class_label": "",
            "accuracy": summary_metrics["accuracy"],
            "precision": "",
            "recall": "",
            "f1": "",
            "support": len(y_test),
            "macro_precision": (
                summary_metrics["macro_precision"]
            ),
            "macro_recall": summary_metrics["macro_recall"],
            "macro_f1": summary_metrics["macro_f1"],
            "weighted_precision": (
                summary_metrics["weighted_precision"]
            ),
            "weighted_recall": (
                summary_metrics["weighted_recall"]
            ),
            "weighted_f1": (
                summary_metrics["weighted_f1"]
            ),
        }
    ]

    per_class_metrics = calculate_per_class_metrics(
        y_true=y_test,
        y_pred=predictions,
        encoder=encoder,
    )

    for class_metrics in per_class_metrics:
        records.append(
            {
                "target": target_name,
                "model": model_name,
                "training_labels": label_source,
                "metric_scope": "class",
                "class_label": (
                    class_metrics["class_label"]
                ),
                "accuracy": "",
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1": class_metrics["f1"],
                "support": class_metrics["support"],
                "macro_precision": "",
                "macro_recall": "",
                "macro_f1": "",
                "weighted_precision": "",
                "weighted_recall": "",
                "weighted_f1": "",
            }
        )

    logger.info(
        "%s | %s | %s labels | Accuracy: %.4f | "
        "Macro F1: %.4f | Weighted F1: %.4f",
        target_name,
        model_name,
        label_source,
        summary_metrics["accuracy"],
        summary_metrics["macro_f1"],
        summary_metrics["weighted_f1"],
    )

    return records


def evaluate_target_models(
    X_test: Any,
    y_test: np.ndarray,
    encoder: Any,
    target_name: str,
    enabled_models: list[str],
) -> pd.DataFrame:
    """Evaluate every enabled model for one prediction target."""
    records: list[dict[str, Any]] = []

    for model_name in enabled_models:
        for label_source in (
            LABEL_SOURCE_REAL,
            LABEL_SOURCE_LLM,
        ):
            model_records = evaluate_one_model(
                X_test=X_test,
                y_test=y_test,
                encoder=encoder,
                target_name=target_name,
                model_name=model_name,
                label_source=label_source,
            )

            records.extend(model_records)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Direct LLM-label agreement
# ---------------------------------------------------------------------------

def calculate_label_agreement(
    real_labels: np.ndarray,
    llm_labels: np.ndarray,
    encoder: Any,
    target_name: str,
) -> pd.DataFrame:
    """
    Compare LLM-generated training labels directly with original labels.

    The original labels are treated as the reference labels.
    """
    if len(real_labels) != len(llm_labels):
        raise ModelEvaluationError(
            f"Cannot calculate {target_name} agreement because "
            "the label arrays have different lengths."
        )

    summary_metrics = calculate_summary_metrics(
        y_true=real_labels,
        y_pred=llm_labels,
    )

    records: list[dict[str, Any]] = [
        {
            "target": target_name,
            "metric_scope": "overall",
            "class_label": "",
            "accuracy": summary_metrics["accuracy"],
            "precision": "",
            "recall": "",
            "f1": "",
            "support": len(real_labels),
            "macro_precision": (
                summary_metrics["macro_precision"]
            ),
            "macro_recall": summary_metrics["macro_recall"],
            "macro_f1": summary_metrics["macro_f1"],
            "weighted_precision": (
                summary_metrics["weighted_precision"]
            ),
            "weighted_recall": (
                summary_metrics["weighted_recall"]
            ),
            "weighted_f1": (
                summary_metrics["weighted_f1"]
            ),
        }
    ]

    per_class_metrics = calculate_per_class_metrics(
        y_true=real_labels,
        y_pred=llm_labels,
        encoder=encoder,
    )

    for class_metrics in per_class_metrics:
        records.append(
            {
                "target": target_name,
                "metric_scope": "class",
                "class_label": (
                    class_metrics["class_label"]
                ),
                "accuracy": "",
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1": class_metrics["f1"],
                "support": class_metrics["support"],
                "macro_precision": "",
                "macro_recall": "",
                "macro_f1": "",
                "weighted_precision": "",
                "weighted_recall": "",
                "weighted_f1": "",
            }
        )

    logger.info(
        "Direct LLM %s-label agreement: "
        "Accuracy %.4f | Macro F1 %.4f | Weighted F1 %.4f",
        target_name,
        summary_metrics["accuracy"],
        summary_metrics["macro_f1"],
        summary_metrics["weighted_f1"],
    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute all model and label-agreement evaluations."""
    logger.info("Starting model evaluation.")

    config = load_config(CONFIG_PATH)

    dataset_config = config["dataset"]
    classifier_config = config["classifiers"]

    enabled_models = get_enabled_models(
        classifier_config
    )

    train_path = resolve_project_path(
        str(dataset_config["train_file"])
    )

    llm_train_path = resolve_project_path(
        str(dataset_config["llm_train_file"])
    )

    test_path = resolve_project_path(
        str(dataset_config["test_file"])
    )

    original_training = load_dataset(
        train_path,
        "Original-label training dataset",
    )

    llm_training = load_dataset(
        llm_train_path,
        "LLM-labeled training dataset",
    )

    testing_dataframe = load_dataset(
        test_path,
        "Original-label testing dataset",
    )

    validate_training_alignment(
        original_training=original_training,
        llm_training=llm_training,
    )

    required_processed_files = (
        X_TEST_PATH,
        CATEGORY_ENCODER_PATH,
        SECTION_ENCODER_PATH,
    )

    for required_path in required_processed_files:
        if not required_path.exists():
            raise ModelEvaluationError(
                f"Required processed file not found: {required_path}. "
                "Run 04_vectorize_dataset.py first."
            )

    X_test = load_pickle(X_TEST_PATH)
    category_encoder = load_pickle(
        CATEGORY_ENCODER_PATH
    )
    section_encoder = load_pickle(
        SECTION_ENCODER_PATH
    )

    validate_test_matrix(
        X_test=X_test,
        expected_rows=len(testing_dataframe),
    )

    test_category_labels = encode_labels(
        dataframe=testing_dataframe,
        label_column=CATEGORY_COLUMN,
        encoder=category_encoder,
        dataset_name="Testing dataset",
    )

    test_section_labels = encode_labels(
        dataframe=testing_dataframe,
        label_column=SECTION_COLUMN,
        encoder=section_encoder,
        dataset_name="Testing dataset",
    )

    real_training_category_labels = encode_labels(
        dataframe=original_training,
        label_column=CATEGORY_COLUMN,
        encoder=category_encoder,
        dataset_name="Original-label training dataset",
    )

    llm_training_category_labels = encode_labels(
        dataframe=llm_training,
        label_column=CATEGORY_COLUMN,
        encoder=category_encoder,
        dataset_name="LLM-labeled training dataset",
    )

    real_training_section_labels = encode_labels(
        dataframe=original_training,
        label_column=SECTION_COLUMN,
        encoder=section_encoder,
        dataset_name="Original-label training dataset",
    )

    llm_training_section_labels = encode_labels(
        dataframe=llm_training,
        label_column=SECTION_COLUMN,
        encoder=section_encoder,
        dataset_name="LLM-labeled training dataset",
    )

    category_metrics = evaluate_target_models(
        X_test=X_test,
        y_test=test_category_labels,
        encoder=category_encoder,
        target_name=TARGET_CATEGORY,
        enabled_models=enabled_models,
    )

    section_metrics = evaluate_target_models(
        X_test=X_test,
        y_test=test_section_labels,
        encoder=section_encoder,
        target_name=TARGET_SECTION,
        enabled_models=enabled_models,
    )

    category_agreement = calculate_label_agreement(
        real_labels=real_training_category_labels,
        llm_labels=llm_training_category_labels,
        encoder=category_encoder,
        target_name=TARGET_CATEGORY,
    )

    section_agreement = calculate_label_agreement(
        real_labels=real_training_section_labels,
        llm_labels=llm_training_section_labels,
        encoder=section_encoder,
        target_name=TARGET_SECTION,
    )

    write_csv(
        dataframe=category_metrics,
        file_path=CATEGORY_METRICS_PATH,
    )

    write_csv(
        dataframe=section_metrics,
        file_path=SECTION_METRICS_PATH,
    )

    write_csv(
        dataframe=category_agreement,
        file_path=LLM_CATEGORY_AGREEMENT_PATH,
    )

    write_csv(
        dataframe=section_agreement,
        file_path=LLM_SECTION_AGREEMENT_PATH,
    )

    logger.info(
        "Saved Category metrics to %s.",
        CATEGORY_METRICS_PATH,
    )

    logger.info(
        "Saved Section metrics to %s.",
        SECTION_METRICS_PATH,
    )

    logger.info(
        "Saved direct LLM Category agreement to %s.",
        LLM_CATEGORY_AGREEMENT_PATH,
    )

    logger.info(
        "Saved direct LLM Section agreement to %s.",
        LLM_SECTION_AGREEMENT_PATH,
    )

    logger.info(
        "Model evaluation completed successfully."
    )


if __name__ == "__main__":
    try:
        main()

    except ModelEvaluationError as error:
        logger.error(
            "Model evaluation failed: %s",
            error,
        )
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "Model evaluation failed because of an unexpected error."
        )
        raise SystemExit(1)