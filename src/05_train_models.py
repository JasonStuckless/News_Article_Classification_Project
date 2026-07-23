"""
05_train_models.py

Train the classical news article classification models.

This script trains three classifier types:

- Logistic Regression
- Linear Support Vector Machine
- Multinomial Naive Bayes

Each classifier is trained under four experimental conditions:

1. Category prediction using original labels.
2. Category prediction using LLM-generated labels.
3. Section prediction using original labels.
4. Section prediction using LLM-generated labels.

This produces 12 trained models in total.

Inputs:
- data/interim/train_original.csv
- data/interim/train_llm_labeled.csv
- data/processed/X_train.pkl
- data/processed/category_encoder.pkl
- data/processed/section_encoder.pkl

Outputs:
- models/category/logistic_real.pkl
- models/category/logistic_llm.pkl
- models/category/svm_real.pkl
- models/category/svm_llm.pkl
- models/category/nb_real.pkl
- models/category/nb_llm.pkl
- models/section/logistic_real.pkl
- models/section/logistic_llm.pkl
- models/section/svm_real.pkl
- models/section/svm_llm.pkl
- models/section/nb_real.pkl
- models/section/nb_llm.pkl
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIRECTORY = PROJECT_ROOT / "models"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.pkl"
CATEGORY_ENCODER_PATH = (
    PROCESSED_DATA_DIR / "category_encoder.pkl"
)
SECTION_ENCODER_PATH = (
    PROCESSED_DATA_DIR / "section_encoder.pkl"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.io import (
    load_pickle,
    read_csv,
    save_pickle,
)
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

LABEL_SOURCE_REAL = "real"
LABEL_SOURCE_LLM = "llm"

TARGET_CATEGORY = "category"
TARGET_SECTION = "section"

MODEL_LOGISTIC = "logistic"
MODEL_SVM = "svm"
MODEL_NAIVE_BAYES = "nb"

SUPPORTED_MODELS: tuple[str, ...] = (
    MODEL_LOGISTIC,
    MODEL_SVM,
    MODEL_NAIVE_BAYES,
)


class ModelTrainingError(RuntimeError):
    """Raised when one or more models cannot be trained safely."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate the project configuration.

    Parameters
    ----------
    config_path:
        Path to config/config.yaml.

    Returns
    -------
    dict[str, Any]
        Parsed project configuration.

    Raises
    ------
    ModelTrainingError
        If the configuration file is missing or malformed.
    """
    if not config_path.exists():
        raise ModelTrainingError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise ModelTrainingError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise ModelTrainingError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    classifier_config = config.get("classifiers")
    project_config = config.get("project")

    if not isinstance(dataset_config, dict):
        raise ModelTrainingError(
            "The configuration must contain a 'dataset' section."
        )

    if not isinstance(classifier_config, dict):
        raise ModelTrainingError(
            "The configuration must contain a 'classifiers' section."
        )

    if not isinstance(project_config, dict):
        raise ModelTrainingError(
            "The configuration must contain a 'project' section."
        )

    required_dataset_settings = {
        "train_file",
        "llm_train_file",
    }

    missing_dataset_settings = (
        required_dataset_settings - set(dataset_config)
    )

    if missing_dataset_settings:
        missing = ", ".join(
            sorted(missing_dataset_settings)
        )

        raise ModelTrainingError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    if "random_seed" not in project_config:
        raise ModelTrainingError(
            "Missing project.random_seed in config.yaml."
        )

    missing_classifier_settings = (
        set(SUPPORTED_MODELS) - set(classifier_config)
    )

    if missing_classifier_settings:
        missing = ", ".join(
            sorted(missing_classifier_settings)
        )

        raise ModelTrainingError(
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
    Return the enabled classifier names.

    Supports either simple Boolean entries:

    classifiers:
      logistic: true

    or nested entries:

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
            raise ModelTrainingError(
                f"classifiers.{model_name} must be a Boolean "
                "or a YAML mapping."
            )

        if enabled:
            enabled_models.append(model_name)

    if not enabled_models:
        raise ModelTrainingError(
            "At least one classifier must be enabled."
        )

    return enabled_models


def get_model_parameters(
    classifier_config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """
    Return optional model-specific parameters from config.yaml.

    For example:

    classifiers:
      logistic:
        enabled: true
        C: 1.0
        max_iter: 2000
    """
    setting = classifier_config[model_name]

    if isinstance(setting, bool):
        return {}

    parameters = dict(setting)
    parameters.pop("enabled", None)

    return parameters


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def load_training_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load and validate a training dataset.
    """
    if not file_path.exists():
        raise ModelTrainingError(
            f"{dataset_name} not found: {file_path}"
        )

    dataframe = read_csv(file_path)

    if dataframe.empty:
        raise ModelTrainingError(
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

        raise ModelTrainingError(
            f"{dataset_name} is missing required columns: {missing}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise ModelTrainingError(
            f"{dataset_name} contains missing {ID_COLUMN} values."
        )

    if not dataframe[ID_COLUMN].is_unique:
        raise ModelTrainingError(
            f"{dataset_name} contains duplicated {ID_COLUMN} values."
        )

    for label_column in (
        CATEGORY_COLUMN,
        SECTION_COLUMN,
    ):
        if dataframe[label_column].isna().any():
            raise ModelTrainingError(
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
            raise ModelTrainingError(
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
    Verify that both training datasets contain identical articles
    in the same order.

    Category and Section are the only fields permitted to differ.
    """
    if list(original_training.columns) != list(
        llm_training.columns
    ):
        raise ModelTrainingError(
            "The original-label and LLM-label training datasets "
            "have different columns or column order."
        )

    if len(original_training) != len(llm_training):
        raise ModelTrainingError(
            "The original-label and LLM-label training datasets "
            "have different row counts."
        )

    if list(original_training[ID_COLUMN]) != list(
        llm_training[ID_COLUMN]
    ):
        raise ModelTrainingError(
            "The original-label and LLM-label training datasets "
            "do not have the same row order."
        )

    non_label_columns = [
        column
        for column in original_training.columns
        if column not in (
            CATEGORY_COLUMN,
            SECTION_COLUMN,
        )
    ]

    try:
        pd.testing.assert_frame_equal(
            original_training[
                non_label_columns
            ].reset_index(drop=True),
            llm_training[
                non_label_columns
            ].reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )

    except AssertionError as error:
        raise ModelTrainingError(
            "The two training datasets differ in one or more "
            "non-label values."
        ) from error

    logger.info(
        "Original-label and LLM-label training datasets "
        "are correctly aligned."
    )


def validate_feature_matrix(
    X_train: Any,
    expected_rows: int,
) -> None:
    """
    Validate the saved training feature matrix.
    """
    if not hasattr(X_train, "shape"):
        raise ModelTrainingError(
            "X_train does not provide a valid matrix shape."
        )

    if len(X_train.shape) != 2:
        raise ModelTrainingError(
            "X_train must be a two-dimensional feature matrix."
        )

    if X_train.shape[0] != expected_rows:
        raise ModelTrainingError(
            "X_train row count does not match the training datasets. "
            f"Features: {X_train.shape[0]}; "
            f"dataset rows: {expected_rows}."
        )

    if X_train.shape[1] <= 0:
        raise ModelTrainingError(
            "X_train contains no features."
        )

    if hasattr(X_train, "data"):
        feature_values = X_train.data
    else:
        feature_values = np.asarray(X_train)

    if not np.isfinite(feature_values).all():
        raise ModelTrainingError(
            "X_train contains NaN or infinite feature values."
        )

    if np.any(feature_values < 0):
        raise ModelTrainingError(
            "X_train contains negative values. "
            "Multinomial Naive Bayes requires non-negative features."
        )

    logger.info(
        "Validated X_train: %d rows and %d features.",
        X_train.shape[0],
        X_train.shape[1],
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
    """
    Encode one target column using its saved LabelEncoder.
    """
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

        raise ModelTrainingError(
            f"{dataset_name} contains labels not recognized by "
            f"the encoder for {label_column}: {unknown}"
        )

    encoded_labels = encoder.transform(labels)

    if len(encoded_labels) != len(dataframe):
        raise ModelTrainingError(
            f"Encoded {label_column} label count does not match "
            f"{dataset_name}."
        )

    return encoded_labels


def validate_target_classes(
    encoded_labels: np.ndarray,
    target_name: str,
    label_source: str,
) -> None:
    """
    Confirm that a target contains at least two distinct classes.
    """
    class_count = len(np.unique(encoded_labels))

    if class_count < 2:
        raise ModelTrainingError(
            f"The {target_name} target using {label_source} labels "
            "contains fewer than two classes."
        )

    logger.info(
        "%s target using %s labels contains %d classes.",
        target_name.capitalize(),
        label_source,
        class_count,
    )


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_classifier(
    model_name: str,
    random_seed: int,
    configured_parameters: dict[str, Any],
) -> ClassifierMixin:
    """
    Create one configured scikit-learn classifier.

    Parameters supplied in config.yaml override the defaults below.
    """
    if model_name == MODEL_LOGISTIC:
        parameters: dict[str, Any] = {
            "C": 1.0,
            "max_iter": 2000,
            "solver": "liblinear",
            "random_state": random_seed,
        }

        parameters.update(configured_parameters)

        return LogisticRegression(**parameters)

    if model_name == MODEL_SVM:
        parameters = {
            "C": 1.0,
            "random_state": random_seed,
            "max_iter": 5000,
        }

        parameters.update(configured_parameters)

        return LinearSVC(**parameters)

    if model_name == MODEL_NAIVE_BAYES:
        parameters = {
            "alpha": 1.0,
        }

        parameters.update(configured_parameters)

        return MultinomialNB(**parameters)

    raise ModelTrainingError(
        f"Unsupported classifier: {model_name}"
    )


def get_model_output_path(
    target_name: str,
    model_name: str,
    label_source: str,
) -> Path:
    """
    Construct the output path for one trained model.
    """
    return (
        MODEL_DIRECTORY
        / target_name
        / f"{model_name}_{label_source}.pkl"
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_and_save_model(
    X_train: Any,
    y_train: np.ndarray,
    model_name: str,
    target_name: str,
    label_source: str,
    random_seed: int,
    configured_parameters: dict[str, Any],
) -> None:
    """
    Create, train, verify, and save one classifier.
    """
    logger.info(
        "Training %s model for %s using %s labels.",
        model_name,
        target_name,
        label_source,
    )

    classifier = create_classifier(
        model_name=model_name,
        random_seed=random_seed,
        configured_parameters=configured_parameters,
    )

    start_time = time.perf_counter()

    try:
        classifier.fit(X_train, y_train)

    except Exception as error:
        raise ModelTrainingError(
            f"Training failed for {model_name}, "
            f"target={target_name}, "
            f"labels={label_source}: {error}"
        ) from error

    training_duration = time.perf_counter() - start_time

    if not hasattr(classifier, "classes_"):
        raise ModelTrainingError(
            f"The trained {model_name} model does not contain "
            "a classes_ attribute."
        )

    learned_classes = set(classifier.classes_)
    expected_classes = set(np.unique(y_train))

    if learned_classes != expected_classes:
        raise ModelTrainingError(
            f"The trained {model_name} model did not learn the "
            f"expected class set for {target_name} using "
            f"{label_source} labels."
        )

    output_path = get_model_output_path(
        target_name=target_name,
        model_name=model_name,
        label_source=label_source,
    )

    save_pickle(
        obj=classifier,
        file_path=output_path,
    )

    logger.info(
        "Saved %s model for %s using %s labels to %s. "
        "Training time: %.3f seconds.",
        model_name,
        target_name,
        label_source,
        output_path,
        training_duration,
    )


def train_target_models(
    X_train: Any,
    original_labels: np.ndarray,
    llm_labels: np.ndarray,
    target_name: str,
    enabled_models: list[str],
    classifier_config: dict[str, Any],
    random_seed: int,
) -> None:
    """
    Train all enabled classifiers for one prediction target.
    """
    validate_target_classes(
        encoded_labels=original_labels,
        target_name=target_name,
        label_source=LABEL_SOURCE_REAL,
    )

    validate_target_classes(
        encoded_labels=llm_labels,
        target_name=target_name,
        label_source=LABEL_SOURCE_LLM,
    )

    for model_name in enabled_models:
        model_parameters = get_model_parameters(
            classifier_config=classifier_config,
            model_name=model_name,
        )

        train_and_save_model(
            X_train=X_train,
            y_train=original_labels,
            model_name=model_name,
            target_name=target_name,
            label_source=LABEL_SOURCE_REAL,
            random_seed=random_seed,
            configured_parameters=model_parameters,
        )

        train_and_save_model(
            X_train=X_train,
            y_train=llm_labels,
            model_name=model_name,
            target_name=target_name,
            label_source=LABEL_SOURCE_LLM,
            random_seed=random_seed,
            configured_parameters=model_parameters,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the model-training pipeline."""
    logger.info("Starting classifier training.")

    config = load_config(CONFIG_PATH)

    project_config = config["project"]
    dataset_config = config["dataset"]
    classifier_config = config["classifiers"]

    random_seed = project_config["random_seed"]

    if not isinstance(random_seed, int):
        raise ModelTrainingError(
            "project.random_seed must be an integer."
        )

    enabled_models = get_enabled_models(
        classifier_config
    )

    logger.info(
        "Enabled classifiers: %s",
        ", ".join(enabled_models),
    )

    original_training_path = resolve_project_path(
        str(dataset_config["train_file"])
    )

    llm_training_path = resolve_project_path(
        str(dataset_config["llm_train_file"])
    )

    original_training = load_training_dataset(
        file_path=original_training_path,
        dataset_name="Original-label training dataset",
    )

    llm_training = load_training_dataset(
        file_path=llm_training_path,
        dataset_name="LLM-labeled training dataset",
    )

    validate_training_alignment(
        original_training=original_training,
        llm_training=llm_training,
    )

    required_processed_files = (
        X_TRAIN_PATH,
        CATEGORY_ENCODER_PATH,
        SECTION_ENCODER_PATH,
    )

    for required_path in required_processed_files:
        if not required_path.exists():
            raise ModelTrainingError(
                f"Required processed file not found: {required_path}. "
                "Run 04_vectorize_dataset.py first."
            )

    X_train = load_pickle(X_TRAIN_PATH)

    category_encoder = load_pickle(
        CATEGORY_ENCODER_PATH
    )

    section_encoder = load_pickle(
        SECTION_ENCODER_PATH
    )

    validate_feature_matrix(
        X_train=X_train,
        expected_rows=len(original_training),
    )

    real_category_labels = encode_labels(
        dataframe=original_training,
        label_column=CATEGORY_COLUMN,
        encoder=category_encoder,
        dataset_name="Original-label training dataset",
    )

    llm_category_labels = encode_labels(
        dataframe=llm_training,
        label_column=CATEGORY_COLUMN,
        encoder=category_encoder,
        dataset_name="LLM-labeled training dataset",
    )

    real_section_labels = encode_labels(
        dataframe=original_training,
        label_column=SECTION_COLUMN,
        encoder=section_encoder,
        dataset_name="Original-label training dataset",
    )

    llm_section_labels = encode_labels(
        dataframe=llm_training,
        label_column=SECTION_COLUMN,
        encoder=section_encoder,
        dataset_name="LLM-labeled training dataset",
    )

    train_target_models(
        X_train=X_train,
        original_labels=real_category_labels,
        llm_labels=llm_category_labels,
        target_name=TARGET_CATEGORY,
        enabled_models=enabled_models,
        classifier_config=classifier_config,
        random_seed=random_seed,
    )

    train_target_models(
        X_train=X_train,
        original_labels=real_section_labels,
        llm_labels=llm_section_labels,
        target_name=TARGET_SECTION,
        enabled_models=enabled_models,
        classifier_config=classifier_config,
        random_seed=random_seed,
    )

    expected_model_count = (
        len(enabled_models)
        * 2  # label sources
        * 2  # prediction targets
    )

    logger.info(
        "Classifier training completed successfully. "
        "Trained and saved %d models.",
        expected_model_count,
    )


if __name__ == "__main__":
    try:
        main()

    except ModelTrainingError as error:
        logger.error(
            "Classifier training failed: %s",
            error,
        )
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "Classifier training failed because of an "
            "unexpected error."
        )
        raise SystemExit(1)