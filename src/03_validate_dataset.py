"""
03_validate_dataset.py

Validate all datasets produced by the preparation and LLM-labeling stages.

This script verifies that:

1. The original dataset, unlabeled dataset, training split, and testing split
   all exist and have the expected structure.
2. The unlabeled dataset differs from the original dataset only because the
   Category and Section columns were removed.
3. The training and testing datasets contain every original row exactly once.
4. The LLM-labeled training dataset contains the same articles and original
   non-label values as the real-label training dataset.
5. Every LLM-generated Category and Section value is valid.
6. The LLM audit file contains one successful record for every training row.
7. Audit labels agree with the labels stored in train_llm_labeled.csv.

This script does not modify any dataset.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.validate_labels import VALID_CATEGORIES, VALID_SECTIONS
from utils.io import read_csv
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

LABEL_COLUMNS: tuple[str, ...] = (
    CATEGORY_COLUMN,
    SECTION_COLUMN,
)

AUDIT_REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "original_category",
    "original_section",
    "llm_category",
    "llm_section",
    "llm_model",
    "prompt_version",
    "raw_response",
    "status",
    "validation_error",
    "request_attempts",
    "processing_time_seconds",
    "prompt_token_count",
    "response_token_count",
    "generated_at_utc",
)


class DatasetValidationError(RuntimeError):
    """Raised when one or more dataset integrity checks fail."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate config/config.yaml.

    Parameters
    ----------
    config_path:
        Path to the project configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.

    Raises
    ------
    DatasetValidationError
        If the configuration is missing or invalid.
    """
    if not config_path.exists():
        raise DatasetValidationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise DatasetValidationError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise DatasetValidationError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")

    if not isinstance(dataset_config, dict):
        raise DatasetValidationError(
            "The configuration must contain a 'dataset' section."
        )

    required_settings = {
        "original",
        "unlabeled",
        "train_file",
        "test_file",
        "llm_train_file",
        "audit_file",
    }

    missing_settings = required_settings - set(dataset_config)

    if missing_settings:
        missing = ", ".join(sorted(missing_settings))
        raise DatasetValidationError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# General validation helpers
# ---------------------------------------------------------------------------

def load_required_csv(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load a required CSV file.

    Raises
    ------
    DatasetValidationError
        If the file is missing, empty, or cannot be read.
    """
    if not file_path.exists():
        raise DatasetValidationError(
            f"{dataset_name} not found: {file_path}"
        )

    try:
        dataframe = read_csv(file_path)

    except Exception as error:
        raise DatasetValidationError(
            f"Could not read {dataset_name}: {file_path}"
        ) from error

    if dataframe.empty:
        raise DatasetValidationError(
            f"{dataset_name} contains no rows."
        )

    logger.info(
        "Loaded %s: %d rows and %d columns.",
        dataset_name,
        len(dataframe),
        len(dataframe.columns),
    )

    return dataframe


def validate_unique_identifiers(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that a dataset has a complete, unique Index column."""
    if ID_COLUMN not in dataframe.columns:
        raise DatasetValidationError(
            f"{dataset_name} is missing the {ID_COLUMN} column."
        )

    if dataframe[ID_COLUMN].isna().any():
        missing_count = int(dataframe[ID_COLUMN].isna().sum())

        raise DatasetValidationError(
            f"{dataset_name} contains {missing_count} missing "
            f"{ID_COLUMN} values."
        )

    if not dataframe[ID_COLUMN].is_unique:
        duplicate_count = int(
            dataframe[ID_COLUMN].duplicated(keep=False).sum()
        )

        raise DatasetValidationError(
            f"{dataset_name} contains {duplicate_count} rows with "
            f"duplicated {ID_COLUMN} values."
        )


def validate_label_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that Category and Section are present and non-empty."""
    for column in LABEL_COLUMNS:
        if column not in dataframe.columns:
            raise DatasetValidationError(
                f"{dataset_name} is missing the {column} column."
            )

        missing_count = int(dataframe[column].isna().sum())

        if missing_count > 0:
            raise DatasetValidationError(
                f"{dataset_name} contains {missing_count} missing "
                f"{column} values."
            )

        blank_mask = (
            dataframe[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        blank_count = int(blank_mask.sum())

        if blank_count > 0:
            raise DatasetValidationError(
                f"{dataset_name} contains {blank_count} blank "
                f"{column} values."
            )


def compare_dataframes_exactly(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    comparison_name: str,
    check_dtype: bool = False,
) -> None:
    """
    Compare two DataFrames after resetting their indexes.

    Raises
    ------
    DatasetValidationError
        If the DataFrames differ.
    """
    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_dtype=check_dtype,
            check_like=False,
        )

    except AssertionError as error:
        raise DatasetValidationError(
            f"Validation failed for {comparison_name}: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Unlabeled dataset validation
# ---------------------------------------------------------------------------

def validate_unlabeled_dataset(
    original: pd.DataFrame,
    unlabeled: pd.DataFrame,
) -> None:
    """
    Verify that the unlabeled dataset differs only by removal of labels.
    """
    expected_columns = [
        column
        for column in original.columns
        if column not in LABEL_COLUMNS
    ]

    if list(unlabeled.columns) != expected_columns:
        raise DatasetValidationError(
            "The unlabeled dataset has unexpected columns or column order."
        )

    if len(unlabeled) != len(original):
        raise DatasetValidationError(
            "The unlabeled dataset row count differs from the original."
        )

    compare_dataframes_exactly(
        expected=original[expected_columns],
        actual=unlabeled,
        comparison_name="original versus unlabeled dataset",
    )

    logger.info(
        "Unlabeled dataset validated: only Category and Section "
        "were removed."
    )


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------

def validate_train_test_split(
    original: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """
    Verify that training and testing contain every original row once.
    """
    validate_unique_identifiers(
        original,
        "Original dataset",
    )
    validate_unique_identifiers(
        train_dataframe,
        "Training dataset",
    )
    validate_unique_identifiers(
        test_dataframe,
        "Testing dataset",
    )

    validate_label_columns(
        original,
        "Original dataset",
    )
    validate_label_columns(
        train_dataframe,
        "Training dataset",
    )
    validate_label_columns(
        test_dataframe,
        "Testing dataset",
    )

    original_ids = set(original[ID_COLUMN])
    train_ids = set(train_dataframe[ID_COLUMN])
    test_ids = set(test_dataframe[ID_COLUMN])

    overlap = train_ids.intersection(test_ids)

    if overlap:
        raise DatasetValidationError(
            f"Training and testing datasets share "
            f"{len(overlap)} article identifiers."
        )

    combined_ids = train_ids.union(test_ids)

    missing_ids = original_ids - combined_ids
    unexpected_ids = combined_ids - original_ids

    if missing_ids or unexpected_ids:
        raise DatasetValidationError(
            "Training and testing datasets do not reconstruct the "
            "original dataset. "
            f"Missing identifiers: {len(missing_ids)}; "
            f"unexpected identifiers: {len(unexpected_ids)}."
        )

    if len(train_dataframe) + len(test_dataframe) != len(original):
        raise DatasetValidationError(
            "Training and testing row counts do not sum to the "
            "original row count."
        )

    original_lookup = original.set_index(ID_COLUMN)

    for split_name, split_dataframe in (
            ("training dataset", train_dataframe),
            ("testing dataset", test_dataframe),
    ):
        split_lookup = split_dataframe.set_index(ID_COLUMN)

        expected_split = original_lookup.loc[
            split_lookup.index,
            split_lookup.columns,
        ]

        compare_dataframes_exactly(
            expected=expected_split.reset_index(),
            actual=split_lookup.reset_index(),
            comparison_name=f"original data versus {split_name}",
        )

    logger.info(
        "Train/test split validated: %d training rows and "
        "%d testing rows.",
        len(train_dataframe),
        len(test_dataframe),
    )


# ---------------------------------------------------------------------------
# LLM-labeled dataset validation
# ---------------------------------------------------------------------------

def validate_generated_label_values(
    llm_dataframe: pd.DataFrame,
) -> None:
    """Verify that every generated label belongs to the allowed sets."""
    validate_label_columns(
        llm_dataframe,
        "LLM-labeled training dataset",
    )

    categories = set(
        llm_dataframe[CATEGORY_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sections = set(
        llm_dataframe[SECTION_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_categories = categories - set(VALID_CATEGORIES)
    invalid_sections = sections - set(VALID_SECTIONS)

    if invalid_categories:
        invalid = ", ".join(sorted(invalid_categories))

        raise DatasetValidationError(
            f"LLM-labeled dataset contains invalid Category values: "
            f"{invalid}"
        )

    if invalid_sections:
        invalid = ", ".join(sorted(invalid_sections))

        raise DatasetValidationError(
            f"LLM-labeled dataset contains invalid Section values: "
            f"{invalid}"
        )

    logger.info(
        "All LLM-generated Category and Section labels are valid."
    )


def validate_llm_dataset_integrity(
    original_training: pd.DataFrame,
    llm_training: pd.DataFrame,
) -> None:
    """
    Verify that only Category and Section differ between training versions.
    """
    validate_unique_identifiers(
        original_training,
        "Original-label training dataset",
    )
    validate_unique_identifiers(
        llm_training,
        "LLM-labeled training dataset",
    )

    if list(llm_training.columns) != list(original_training.columns):
        raise DatasetValidationError(
            "The LLM-labeled training dataset does not have the same "
            "columns or column order as the original-label training dataset."
        )

    if len(llm_training) != len(original_training):
        raise DatasetValidationError(
            "The LLM-labeled training dataset row count differs from "
            "the original-label training dataset."
        )

    if list(llm_training[ID_COLUMN]) != list(
        original_training[ID_COLUMN]
    ):
        raise DatasetValidationError(
            "The LLM-labeled training dataset row order differs from "
            "the original-label training dataset."
        )

    non_label_columns = [
        column
        for column in original_training.columns
        if column not in LABEL_COLUMNS
    ]

    compare_dataframes_exactly(
        expected=original_training[non_label_columns],
        actual=llm_training[non_label_columns],
        comparison_name=(
            "non-label data in original-label and LLM-labeled "
            "training datasets"
        ),
    )

    validate_generated_label_values(llm_training)

    logger.info(
        "LLM-labeled training dataset integrity validated."
    )


# ---------------------------------------------------------------------------
# Audit validation
# ---------------------------------------------------------------------------

def validate_audit_file(
    original_training: pd.DataFrame,
    llm_training: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
) -> None:
    """
    Verify that the audit file matches both training datasets.
    """
    missing_columns = (
        set(AUDIT_REQUIRED_COLUMNS)
        - set(audit_dataframe.columns)
    )

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise DatasetValidationError(
            f"The audit file is missing required columns: {missing}"
        )

    validate_unique_identifiers(
        audit_dataframe,
        "LLM audit dataset",
    )

    if len(audit_dataframe) != len(original_training):
        raise DatasetValidationError(
            "The audit file row count differs from the training row count."
        )

    unsuccessful_rows = audit_dataframe[
        audit_dataframe["status"] != "success"
    ]

    if not unsuccessful_rows.empty:
        raise DatasetValidationError(
            f"The audit file contains {len(unsuccessful_rows)} "
            "unsuccessful labeling records."
        )

    original_ids = set(original_training[ID_COLUMN])
    audit_ids = set(audit_dataframe[ID_COLUMN])

    if audit_ids != original_ids:
        missing_ids = original_ids - audit_ids
        unexpected_ids = audit_ids - original_ids

        raise DatasetValidationError(
            "The audit file identifiers do not match the training set. "
            f"Missing identifiers: {len(missing_ids)}; "
            f"unexpected identifiers: {len(unexpected_ids)}."
        )

    original_lookup = original_training.set_index(ID_COLUMN)
    llm_lookup = llm_training.set_index(ID_COLUMN)
    audit_lookup = audit_dataframe.set_index(ID_COLUMN)

    expected_original_categories = (
        original_lookup.loc[
            audit_lookup.index,
            CATEGORY_COLUMN,
        ]
        .astype(str)
    )

    expected_original_sections = (
        original_lookup.loc[
            audit_lookup.index,
            SECTION_COLUMN,
        ]
        .astype(str)
    )

    expected_llm_categories = (
        llm_lookup.loc[
            audit_lookup.index,
            CATEGORY_COLUMN,
        ]
        .astype(str)
    )

    expected_llm_sections = (
        llm_lookup.loc[
            audit_lookup.index,
            SECTION_COLUMN,
        ]
        .astype(str)
    )

    actual_original_categories = (
        audit_lookup["original_category"].astype(str)
    )
    actual_original_sections = (
        audit_lookup["original_section"].astype(str)
    )
    actual_llm_categories = (
        audit_lookup["llm_category"].astype(str)
    )
    actual_llm_sections = (
        audit_lookup["llm_section"].astype(str)
    )

    if not expected_original_categories.equals(
        actual_original_categories
    ):
        raise DatasetValidationError(
            "Original Category values in the audit file do not match "
            "train_original.csv."
        )

    if not expected_original_sections.equals(
        actual_original_sections
    ):
        raise DatasetValidationError(
            "Original Section values in the audit file do not match "
            "train_original.csv."
        )

    if not expected_llm_categories.equals(
        actual_llm_categories
    ):
        raise DatasetValidationError(
            "LLM Category values in the audit file do not match "
            "train_llm_labeled.csv."
        )

    if not expected_llm_sections.equals(
        actual_llm_sections
    ):
        raise DatasetValidationError(
            "LLM Section values in the audit file do not match "
            "train_llm_labeled.csv."
        )

    if audit_dataframe["llm_model"].isna().any():
        raise DatasetValidationError(
            "The audit file contains missing LLM model values."
        )

    if audit_dataframe["prompt_version"].isna().any():
        raise DatasetValidationError(
            "The audit file contains missing prompt-version values."
        )

    logger.info(
        "LLM audit file validated successfully."
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def log_validation_summary(
    original: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    llm_dataframe: pd.DataFrame,
) -> None:
    """Log a concise summary of the validated datasets."""
    category_agreement = (
        train_dataframe[CATEGORY_COLUMN].astype(str)
        == llm_dataframe[CATEGORY_COLUMN].astype(str)
    ).mean()

    section_agreement = (
        train_dataframe[SECTION_COLUMN].astype(str)
        == llm_dataframe[SECTION_COLUMN].astype(str)
    ).mean()

    logger.info(
        "Validation summary:"
        "\nOriginal rows: %d"
        "\nTraining rows: %d"
        "\nTesting rows: %d"
        "\nLLM-labeled rows: %d"
        "\nDirect Category agreement: %.4f"
        "\nDirect Section agreement: %.4f",
        len(original),
        len(train_dataframe),
        len(test_dataframe),
        len(llm_dataframe),
        category_agreement,
        section_agreement,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute all dataset validation checks."""
    logger.info("Starting dataset validation.")

    config = load_config(CONFIG_PATH)
    dataset_config = config["dataset"]

    original_path = resolve_project_path(
        str(dataset_config["original"])
    )
    unlabeled_path = resolve_project_path(
        str(dataset_config["unlabeled"])
    )
    train_path = resolve_project_path(
        str(dataset_config["train_file"])
    )
    test_path = resolve_project_path(
        str(dataset_config["test_file"])
    )
    llm_train_path = resolve_project_path(
        str(dataset_config["llm_train_file"])
    )
    audit_path = resolve_project_path(
        str(dataset_config["audit_file"])
    )

    original_dataframe = load_required_csv(
        original_path,
        "Original dataset",
    )
    unlabeled_dataframe = load_required_csv(
        unlabeled_path,
        "Unlabeled dataset",
    )
    train_dataframe = load_required_csv(
        train_path,
        "Original-label training dataset",
    )
    test_dataframe = load_required_csv(
        test_path,
        "Original-label testing dataset",
    )
    llm_dataframe = load_required_csv(
        llm_train_path,
        "LLM-labeled training dataset",
    )
    audit_dataframe = load_required_csv(
        audit_path,
        "LLM label audit dataset",
    )

    validate_unlabeled_dataset(
        original=original_dataframe,
        unlabeled=unlabeled_dataframe,
    )

    validate_train_test_split(
        original=original_dataframe,
        train_dataframe=train_dataframe,
        test_dataframe=test_dataframe,
    )

    validate_llm_dataset_integrity(
        original_training=train_dataframe,
        llm_training=llm_dataframe,
    )

    validate_audit_file(
        original_training=train_dataframe,
        llm_training=llm_dataframe,
        audit_dataframe=audit_dataframe,
    )

    log_validation_summary(
        original=original_dataframe,
        train_dataframe=train_dataframe,
        test_dataframe=test_dataframe,
        llm_dataframe=llm_dataframe,
    )

    logger.info(
        "All dataset validation checks completed successfully."
    )


if __name__ == "__main__":
    try:
        main()

    except DatasetValidationError as error:
        logger.error(
            "Dataset validation failed: %s",
            error,
        )
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "Dataset validation failed because of an unexpected error."
        )
        raise SystemExit(1)