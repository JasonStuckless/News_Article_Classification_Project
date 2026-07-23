"""
04_vectorize_dataset.py

Create the shared TF-IDF feature representation used by all classification
experiments.

This script:
1. Loads the original-label training and testing datasets.
2. Loads the LLM-labeled training dataset.
3. Verifies that both training datasets contain the same articles in the
   same order.
4. Combines Headline, Description, and Article text into one document per row.
5. Fits a TF-IDF vectorizer using only the training articles.
6. Transforms both training and testing articles.
7. Creates Category and Section label encoders.
8. Saves:
   - data/processed/X_train.pkl
   - data/processed/X_test.pkl
   - data/processed/vectorizer.pkl
   - data/processed/category_encoder.pkl
   - data/processed/section_encoder.pkl

The same X_train representation is used for the real-label and LLM-label
experiments. Only the training labels change.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.pkl"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
VECTORIZER_PATH = PROCESSED_DATA_DIR / "vectorizer.pkl"
CATEGORY_ENCODER_PATH = PROCESSED_DATA_DIR / "category_encoder.pkl"
SECTION_ENCODER_PATH = PROCESSED_DATA_DIR / "section_encoder.pkl"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.validate_labels import VALID_CATEGORIES, VALID_SECTIONS
from utils.io import read_csv, save_pickle
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
CATEGORY_COLUMN = "Category"
SECTION_COLUMN = "Section"

TEXT_COLUMNS: tuple[str, ...] = (
    "Headline",
    "Description",
    "Article text",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    CATEGORY_COLUMN,
    SECTION_COLUMN,
    *TEXT_COLUMNS,
)


class DatasetVectorizationError(RuntimeError):
    """Raised when the datasets cannot be vectorized safely."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate config/config.yaml.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    DatasetVectorizationError
        If the configuration file is missing or invalid.
    """
    if not config_path.exists():
        raise DatasetVectorizationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise DatasetVectorizationError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise DatasetVectorizationError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    tfidf_config = config.get("tfidf")

    if not isinstance(dataset_config, dict):
        raise DatasetVectorizationError(
            "The configuration must contain a 'dataset' section."
        )

    if not isinstance(tfidf_config, dict):
        raise DatasetVectorizationError(
            "The configuration must contain a 'tfidf' section."
        )

    required_dataset_settings = {
        "train_file",
        "test_file",
        "llm_train_file",
    }

    required_tfidf_settings = {
        "max_features",
        "ngram_range",
        "stop_words",
    }

    missing_dataset_settings = (
        required_dataset_settings - set(dataset_config)
    )

    missing_tfidf_settings = (
        required_tfidf_settings - set(tfidf_config)
    )

    if missing_dataset_settings:
        missing = ", ".join(sorted(missing_dataset_settings))

        raise DatasetVectorizationError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    if missing_tfidf_settings:
        missing = ", ".join(sorted(missing_tfidf_settings))

        raise DatasetVectorizationError(
            f"Missing TF-IDF settings in config.yaml: {missing}"
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def validate_tfidf_settings(
    max_features: int,
    ngram_range: tuple[int, int],
    stop_words: str | None,
) -> None:
    """Validate the TF-IDF configuration values."""
    if max_features <= 0:
        raise DatasetVectorizationError(
            "tfidf.max_features must be greater than zero."
        )

    if len(ngram_range) != 2:
        raise DatasetVectorizationError(
            "tfidf.ngram_range must contain exactly two integers."
        )

    minimum_ngram, maximum_ngram = ngram_range

    if minimum_ngram < 1:
        raise DatasetVectorizationError(
            "The minimum n-gram size must be at least 1."
        )

    if maximum_ngram < minimum_ngram:
        raise DatasetVectorizationError(
            "The maximum n-gram size cannot be less than the minimum."
        )

    if stop_words not in (None, "english"):
        raise DatasetVectorizationError(
            "tfidf.stop_words must be 'english' or null."
        )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def load_required_dataset(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load and validate a required dataset.

    Parameters
    ----------
    file_path:
        Path to the CSV file.

    dataset_name:
        Human-readable name used in messages.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if not file_path.exists():
        raise DatasetVectorizationError(
            f"{dataset_name} not found: {file_path}"
        )

    dataframe = read_csv(file_path)

    if dataframe.empty:
        raise DatasetVectorizationError(
            f"{dataset_name} contains no rows."
        )

    missing_columns = set(REQUIRED_COLUMNS) - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise DatasetVectorizationError(
            f"{dataset_name} is missing required columns: {missing}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise DatasetVectorizationError(
            f"{dataset_name} contains missing {ID_COLUMN} values."
        )

    if not dataframe[ID_COLUMN].is_unique:
        raise DatasetVectorizationError(
            f"{dataset_name} contains duplicated {ID_COLUMN} values."
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
    Verify that the real-label and LLM-label training datasets contain the
    same articles in the same order.

    Only Category and Section are permitted to differ.
    """
    if list(original_training.columns) != list(llm_training.columns):
        raise DatasetVectorizationError(
            "The original-label and LLM-label training datasets do not "
            "have identical columns and column order."
        )

    if len(original_training) != len(llm_training):
        raise DatasetVectorizationError(
            "The original-label and LLM-label training datasets have "
            "different row counts."
        )

    if list(original_training[ID_COLUMN]) != list(
        llm_training[ID_COLUMN]
    ):
        raise DatasetVectorizationError(
            "The original-label and LLM-label training datasets do not "
            "have the same article order."
        )

    non_label_columns = [
        column
        for column in original_training.columns
        if column not in (CATEGORY_COLUMN, SECTION_COLUMN)
    ]

    try:
        pd.testing.assert_frame_equal(
            original_training[non_label_columns].reset_index(drop=True),
            llm_training[non_label_columns].reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )

    except AssertionError as error:
        raise DatasetVectorizationError(
            "The original-label and LLM-label training datasets differ "
            "in one or more non-label values."
        ) from error

    logger.info(
        "Training datasets are aligned. Only their labels may differ."
    )


def validate_labels(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that Category and Section labels belong to the allowed sets."""
    categories = (
        dataframe[CATEGORY_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sections = (
        dataframe[SECTION_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if categories.isna().any() or sections.isna().any():
        raise DatasetVectorizationError(
            f"{dataset_name} contains missing labels."
        )

    invalid_categories = (
        set(categories) - set(VALID_CATEGORIES)
    )

    invalid_sections = (
        set(sections) - set(VALID_SECTIONS)
    )

    if invalid_categories:
        invalid = ", ".join(sorted(invalid_categories))

        raise DatasetVectorizationError(
            f"{dataset_name} contains invalid Category labels: {invalid}"
        )

    if invalid_sections:
        invalid = ", ".join(sorted(invalid_sections))

        raise DatasetVectorizationError(
            f"{dataset_name} contains invalid Section labels: {invalid}"
        )


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def clean_text_value(value: Any) -> str:
    """
    Convert one dataset value into clean text.

    Missing values become an empty string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def combine_article_text(dataframe: pd.DataFrame) -> pd.Series:
    """
    Combine Headline, Description, and Article text into one document.

    The same fields are used for every article and for both label-source
    conditions.
    """
    combined_documents = dataframe.apply(
        lambda row: "\n\n".join(
            text
            for text in (
                clean_text_value(row["Headline"]),
                clean_text_value(row["Description"]),
                clean_text_value(row["Article text"]),
            )
            if text
        ),
        axis=1,
    )

    empty_documents = combined_documents.str.strip().eq("")

    if empty_documents.any():
        empty_count = int(empty_documents.sum())

        raise DatasetVectorizationError(
            f"{empty_count} articles contain no usable text."
        )

    return combined_documents


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

def create_vectorizer(
    max_features: int,
    ngram_range: tuple[int, int],
    stop_words: str | None,
) -> TfidfVectorizer:
    """
    Create the TF-IDF vectorizer used by all experiments.

    The vectorizer is fitted only on the training articles.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )


def create_label_encoders() -> tuple[LabelEncoder, LabelEncoder]:
    """
    Create encoders containing every permitted Category and Section value.

    Fitting to the predefined allowed label sets avoids using test-label
    frequencies while ensuring that every valid label can be transformed.
    """
    category_encoder = LabelEncoder()
    category_encoder.fit(list(VALID_CATEGORIES))

    section_encoder = LabelEncoder()
    section_encoder.fit(list(VALID_SECTIONS))

    return category_encoder, section_encoder


def verify_feature_matrices(
    X_train: Any,
    X_test: Any,
    train_rows: int,
    test_rows: int,
) -> None:
    """Verify the dimensions of the generated feature matrices."""
    if X_train.shape[0] != train_rows:
        raise DatasetVectorizationError(
            "X_train row count does not match the training dataset."
        )

    if X_test.shape[0] != test_rows:
        raise DatasetVectorizationError(
            "X_test row count does not match the testing dataset."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise DatasetVectorizationError(
            "X_train and X_test have different feature counts."
        )

    if X_train.shape[1] == 0:
        raise DatasetVectorizationError(
            "TF-IDF produced no features."
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the TF-IDF vectorization pipeline."""
    logger.info("Starting dataset vectorization.")

    config = load_config(CONFIG_PATH)

    dataset_config = config["dataset"]
    tfidf_config = config["tfidf"]

    train_path = resolve_project_path(
        str(dataset_config["train_file"])
    )

    test_path = resolve_project_path(
        str(dataset_config["test_file"])
    )

    llm_train_path = resolve_project_path(
        str(dataset_config["llm_train_file"])
    )

    max_features = int(tfidf_config["max_features"])

    raw_ngram_range = tfidf_config["ngram_range"]

    if (
        not isinstance(raw_ngram_range, (list, tuple))
        or len(raw_ngram_range) != 2
    ):
        raise DatasetVectorizationError(
            "tfidf.ngram_range must be a two-element list."
        )

    ngram_range = (
        int(raw_ngram_range[0]),
        int(raw_ngram_range[1]),
    )

    raw_stop_words = tfidf_config["stop_words"]

    stop_words = (
        None
        if raw_stop_words is None
        else str(raw_stop_words)
    )

    validate_tfidf_settings(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )

    original_training = load_required_dataset(
        train_path,
        "Original-label training dataset",
    )

    testing_dataframe = load_required_dataset(
        test_path,
        "Testing dataset",
    )

    llm_training = load_required_dataset(
        llm_train_path,
        "LLM-labeled training dataset",
    )

    validate_training_alignment(
        original_training=original_training,
        llm_training=llm_training,
    )

    validate_labels(
        original_training,
        "Original-label training dataset",
    )

    validate_labels(
        llm_training,
        "LLM-labeled training dataset",
    )

    validate_labels(
        testing_dataframe,
        "Testing dataset",
    )

    logger.info("Combining article text fields.")

    training_documents = combine_article_text(
        original_training
    )

    testing_documents = combine_article_text(
        testing_dataframe
    )

    vectorizer = create_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )

    logger.info(
        "Fitting TF-IDF vectorizer on %d training articles.",
        len(training_documents),
    )

    X_train = vectorizer.fit_transform(
        training_documents
    )

    logger.info(
        "Transforming %d testing articles.",
        len(testing_documents),
    )

    X_test = vectorizer.transform(
        testing_documents
    )

    verify_feature_matrices(
        X_train=X_train,
        X_test=X_test,
        train_rows=len(original_training),
        test_rows=len(testing_dataframe),
    )

    category_encoder, section_encoder = (
        create_label_encoders()
    )

    logger.info(
        "TF-IDF matrix created: %d training rows, %d testing rows, "
        "%d features.",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )

    save_pickle(
        obj=X_train,
        file_path=X_TRAIN_PATH,
    )

    save_pickle(
        obj=X_test,
        file_path=X_TEST_PATH,
    )

    save_pickle(
        obj=vectorizer,
        file_path=VECTORIZER_PATH,
    )

    save_pickle(
        obj=category_encoder,
        file_path=CATEGORY_ENCODER_PATH,
    )

    save_pickle(
        obj=section_encoder,
        file_path=SECTION_ENCODER_PATH,
    )

    logger.info(
        "Saved X_train to %s.",
        X_TRAIN_PATH,
    )

    logger.info(
        "Saved X_test to %s.",
        X_TEST_PATH,
    )

    logger.info(
        "Saved TF-IDF vectorizer to %s.",
        VECTORIZER_PATH,
    )

    logger.info(
        "Saved Category encoder to %s.",
        CATEGORY_ENCODER_PATH,
    )

    logger.info(
        "Saved Section encoder to %s.",
        SECTION_ENCODER_PATH,
    )

    logger.info(
        "Dataset vectorization completed successfully."
    )


if __name__ == "__main__":
    try:
        main()

    except DatasetVectorizationError as error:
        logger.error(
            "Dataset vectorization failed: %s",
            error,
        )
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "Dataset vectorization failed because of an unexpected error."
        )
        raise SystemExit(1)