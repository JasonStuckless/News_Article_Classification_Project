"""
03_validate_dataset.py

Validate all datasets produced by the preparation and hierarchical
LLM-labeling stages.

This script verifies that:

1. The original dataset, unlabeled dataset, training split, and testing split
   exist and have the expected structure.
2. The unlabeled dataset differs from the original dataset only because the
   Category and Section columns were removed.
3. The training and testing datasets contain every original row exactly once.
4. The LLM-labeled training dataset contains the same articles and original
   non-label values as the original-label training dataset.
5. Every generated Category and Section is valid.
6. Every generated Category-Section pair follows the CNN hierarchy.
7. The hierarchical LLM audit file contains one successful record for every
   training article.
8. Audit labels agree with the labels in train_llm_labeled.csv.
9. Original labels stored in the audit agree with train_original.csv.

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

from llm.validate_labels import (
    CATEGORY_SECTION_MAP,
    VALID_CATEGORIES,
    VALID_SECTIONS,
    LabelValidationError,
    normalize_label,
    validate_label_pair,
)
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
    "raw_category_response",
    "raw_section_response",
    "status",
    "category_validation_error",
    "section_validation_error",
    "category_request_attempts",
    "section_request_attempts",
    "category_processing_time_seconds",
    "section_processing_time_seconds",
    "total_processing_time_seconds",
    "category_prompt_token_count",
    "category_response_token_count",
    "section_prompt_token_count",
    "section_response_token_count",
    "generated_at_utc",
)


class DatasetValidationError(RuntimeError):
    """Raised when one or more dataset-integrity checks fail."""


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
        If the file is missing, empty, or unreadable.
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
    """Verify that a dataset has a complete and unique Index column."""
    if ID_COLUMN not in dataframe.columns:
        raise DatasetValidationError(
            f"{dataset_name} is missing the {ID_COLUMN} column."
        )

    if dataframe[ID_COLUMN].isna().any():
        missing_count = int(
            dataframe[ID_COLUMN].isna().sum()
        )

        raise DatasetValidationError(
            f"{dataset_name} contains {missing_count} missing "
            f"{ID_COLUMN} values."
        )

    if not dataframe[ID_COLUMN].is_unique:
        duplicate_count = int(
            dataframe[ID_COLUMN]
            .duplicated(keep=False)
            .sum()
        )

        raise DatasetValidationError(
            f"{dataset_name} contains {duplicate_count} rows with "
            f"duplicated {ID_COLUMN} values."
        )


def validate_label_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that Category and Section columns are present and non-empty."""
    for column in LABEL_COLUMNS:
        if column not in dataframe.columns:
            raise DatasetValidationError(
                f"{dataset_name} is missing the {column} column."
            )

        missing_count = int(
            dataframe[column].isna().sum()
        )

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


def normalize_label_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize every label in a pandas Series."""
    return (
        series
        .astype(str)
        .map(normalize_label)
    )


# ---------------------------------------------------------------------------
# Original dataset hierarchy validation
# ---------------------------------------------------------------------------

def validate_dataset_hierarchy(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Verify that every Category-Section pair follows the CNN hierarchy.

    This validation is applied to both the original labels and generated
    labels. The hierarchy is defined centrally in validate_labels.py.
    """
    validate_label_columns(
        dataframe=dataframe,
        dataset_name=dataset_name,
    )

    invalid_records: list[str] = []

    for _, row in dataframe.iterrows():
        category = row[CATEGORY_COLUMN]
        section = row[SECTION_COLUMN]

        try:
            validate_label_pair(
                category=str(category),
                section=str(section),
            )

        except LabelValidationError as error:
            invalid_records.append(
                f"Index={row.get(ID_COLUMN, 'unknown')}: {error}"
            )

            if len(invalid_records) >= 10:
                break

    if invalid_records:
        details = "\n".join(invalid_records)

        raise DatasetValidationError(
            f"{dataset_name} contains invalid hierarchical "
            f"Category-Section pairs. First errors:\n{details}"
        )

    logger.info(
        "%s hierarchy validated successfully.",
        dataset_name,
    )


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
    Verify that training and testing reconstruct the original dataset.
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
    """
    Verify that all generated labels are globally valid and hierarchical.
    """
    validate_label_columns(
        llm_dataframe,
        "LLM-labeled training dataset",
    )

    normalized_categories = normalize_label_series(
        llm_dataframe[CATEGORY_COLUMN]
    )

    normalized_sections = normalize_label_series(
        llm_dataframe[SECTION_COLUMN]
    )

    invalid_categories = (
        set(normalized_categories)
        - set(VALID_CATEGORIES)
    )

    invalid_sections = (
        set(normalized_sections)
        - set(VALID_SECTIONS)
    )

    if invalid_categories:
        invalid = ", ".join(
            sorted(invalid_categories)
        )

        raise DatasetValidationError(
            "LLM-labeled dataset contains invalid Category values: "
            f"{invalid}"
        )

    if invalid_sections:
        invalid = ", ".join(
            sorted(invalid_sections)
        )

        raise DatasetValidationError(
            "LLM-labeled dataset contains invalid Section values: "
            f"{invalid}"
        )

    validate_dataset_hierarchy(
        dataframe=llm_dataframe,
        dataset_name="LLM-labeled training dataset",
    )

    logger.info(
        "All LLM-generated labels and hierarchical pairs are valid."
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

    validate_generated_label_values(
        llm_training
    )

    logger.info(
        "LLM-labeled training dataset integrity validated."
    )


# ---------------------------------------------------------------------------
# Audit validation helpers
# ---------------------------------------------------------------------------

def validate_audit_columns(
    audit_dataframe: pd.DataFrame,
) -> None:
    """Verify that the hierarchical audit contains all required fields."""
    missing_columns = (
        set(AUDIT_REQUIRED_COLUMNS)
        - set(audit_dataframe.columns)
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise DatasetValidationError(
            "The audit file is incompatible with the hierarchical "
            f"labeling pipeline and is missing columns: {missing}. "
            "Delete the previous flat-label audit and regenerate labels."
        )


def validate_audit_status(
    audit_dataframe: pd.DataFrame,
) -> None:
    """Verify that every audit record completed successfully."""
    unsuccessful_rows = audit_dataframe[
        audit_dataframe["status"] != "success"
    ]

    if not unsuccessful_rows.empty:
        status_counts = (
            unsuccessful_rows["status"]
            .value_counts(dropna=False)
            .to_dict()
        )

        raise DatasetValidationError(
            f"The audit file contains {len(unsuccessful_rows)} "
            f"unsuccessful hierarchical labeling records. "
            f"Statuses: {status_counts}"
        )


def validate_audit_request_metadata(
    audit_dataframe: pd.DataFrame,
) -> None:
    """Validate hierarchical request metadata stored in the audit."""
    required_non_empty_columns = (
        "llm_model",
        "prompt_version",
        "raw_category_response",
        "raw_section_response",
        "generated_at_utc",
    )

    for column in required_non_empty_columns:
        if audit_dataframe[column].isna().any():
            missing_count = int(
                audit_dataframe[column].isna().sum()
            )

            raise DatasetValidationError(
                f"The audit file contains {missing_count} missing "
                f"values in {column}."
            )

        blank_count = int(
            audit_dataframe[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if blank_count > 0:
            raise DatasetValidationError(
                f"The audit file contains {blank_count} blank "
                f"values in {column}."
            )

    attempt_columns = (
        "category_request_attempts",
        "section_request_attempts",
    )

    for column in attempt_columns:
        numeric_values = pd.to_numeric(
            audit_dataframe[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            raise DatasetValidationError(
                f"The audit file contains non-numeric values in {column}."
            )

        if (numeric_values < 1).any():
            raise DatasetValidationError(
                f"The audit file contains values below 1 in {column}."
            )

    time_columns = (
        "category_processing_time_seconds",
        "section_processing_time_seconds",
        "total_processing_time_seconds",
    )

    for column in time_columns:
        numeric_values = pd.to_numeric(
            audit_dataframe[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            raise DatasetValidationError(
                f"The audit file contains non-numeric values in {column}."
            )

        if (numeric_values < 0).any():
            raise DatasetValidationError(
                f"The audit file contains negative values in {column}."
            )


def validate_audit_label_hierarchy(
    audit_dataframe: pd.DataFrame,
) -> None:
    """Verify that every audit Category-Section pair is hierarchical."""
    audit_pairs = audit_dataframe[
        [
            ID_COLUMN,
            "llm_category",
            "llm_section",
        ]
    ].rename(
        columns={
            "llm_category": CATEGORY_COLUMN,
            "llm_section": SECTION_COLUMN,
        }
    )

    validate_dataset_hierarchy(
        dataframe=audit_pairs,
        dataset_name="LLM audit dataset",
    )


# ---------------------------------------------------------------------------
# Full audit validation
# ---------------------------------------------------------------------------

def validate_audit_file(
    original_training: pd.DataFrame,
    llm_training: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
) -> None:
    """
    Verify that the hierarchical audit matches both training datasets.
    """
    validate_audit_columns(
        audit_dataframe
    )

    validate_unique_identifiers(
        audit_dataframe,
        "LLM audit dataset",
    )

    if len(audit_dataframe) != len(original_training):
        raise DatasetValidationError(
            "The audit file row count differs from the training row count."
        )

    validate_audit_status(
        audit_dataframe
    )

    original_ids = set(
        original_training[ID_COLUMN]
    )

    audit_ids = set(
        audit_dataframe[ID_COLUMN]
    )

    if audit_ids != original_ids:
        missing_ids = original_ids - audit_ids
        unexpected_ids = audit_ids - original_ids

        raise DatasetValidationError(
            "The audit file identifiers do not match the training set. "
            f"Missing identifiers: {len(missing_ids)}; "
            f"unexpected identifiers: {len(unexpected_ids)}."
        )

    original_lookup = original_training.set_index(
        ID_COLUMN
    )

    llm_lookup = llm_training.set_index(
        ID_COLUMN
    )

    audit_lookup = audit_dataframe.set_index(
        ID_COLUMN
    )

    expected_original_categories = normalize_label_series(
        original_lookup.loc[
            audit_lookup.index,
            CATEGORY_COLUMN,
        ]
    )

    expected_original_sections = normalize_label_series(
        original_lookup.loc[
            audit_lookup.index,
            SECTION_COLUMN,
        ]
    )

    expected_llm_categories = normalize_label_series(
        llm_lookup.loc[
            audit_lookup.index,
            CATEGORY_COLUMN,
        ]
    )

    expected_llm_sections = normalize_label_series(
        llm_lookup.loc[
            audit_lookup.index,
            SECTION_COLUMN,
        ]
    )

    actual_original_categories = normalize_label_series(
        audit_lookup["original_category"]
    )

    actual_original_sections = normalize_label_series(
        audit_lookup["original_section"]
    )

    actual_llm_categories = normalize_label_series(
        audit_lookup["llm_category"]
    )

    actual_llm_sections = normalize_label_series(
        audit_lookup["llm_section"]
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

    validate_audit_label_hierarchy(
        audit_dataframe
    )

    validate_audit_request_metadata(
        audit_dataframe
    )

    logger.info(
        "Hierarchical LLM audit file validated successfully."
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def log_hierarchy_distribution(
    llm_dataframe: pd.DataFrame,
) -> None:
    """Log generated Section counts within each generated Category."""
    normalized_dataframe = llm_dataframe.copy()

    normalized_dataframe[CATEGORY_COLUMN] = (
        normalize_label_series(
            normalized_dataframe[CATEGORY_COLUMN]
        )
    )

    normalized_dataframe[SECTION_COLUMN] = (
        normalize_label_series(
            normalized_dataframe[SECTION_COLUMN]
        )
    )

    for category in VALID_CATEGORIES:
        category_rows = normalized_dataframe[
            normalized_dataframe[CATEGORY_COLUMN]
            == category
        ]

        section_counts = (
            category_rows[SECTION_COLUMN]
            .value_counts()
            .to_dict()
        )

        logger.info(
            "Generated hierarchy for Category '%s': %s",
            category,
            section_counts,
        )


def log_validation_summary(
    original: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    llm_dataframe: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
) -> None:
    """Log a concise summary of all validated datasets."""
    original_categories = normalize_label_series(
        train_dataframe[CATEGORY_COLUMN]
    )

    generated_categories = normalize_label_series(
        llm_dataframe[CATEGORY_COLUMN]
    )

    original_sections = normalize_label_series(
        train_dataframe[SECTION_COLUMN]
    )

    generated_sections = normalize_label_series(
        llm_dataframe[SECTION_COLUMN]
    )

    category_agreement = (
        original_categories
        == generated_categories
    ).mean()

    section_agreement = (
        original_sections
        == generated_sections
    ).mean()

    exact_pair_agreement = (
        (original_categories == generated_categories)
        & (original_sections == generated_sections)
    ).mean()

    category_processing_time = pd.to_numeric(
        audit_dataframe[
            "category_processing_time_seconds"
        ],
        errors="coerce",
    ).sum()

    section_processing_time = pd.to_numeric(
        audit_dataframe[
            "section_processing_time_seconds"
        ],
        errors="coerce",
    ).sum()

    total_processing_time = pd.to_numeric(
        audit_dataframe[
            "total_processing_time_seconds"
        ],
        errors="coerce",
    ).sum()

    logger.info(
        "Validation summary:"
        "\nOriginal rows: %d"
        "\nTraining rows: %d"
        "\nTesting rows: %d"
        "\nLLM-labeled rows: %d"
        "\nDirect Category agreement: %.4f"
        "\nDirect Section agreement: %.4f"
        "\nExact Category-Section pair agreement: %.4f"
        "\nCategory-generation time: %.2f seconds"
        "\nSection-generation time: %.2f seconds"
        "\nTotal hierarchical labeling time: %.2f seconds",
        len(original),
        len(train_dataframe),
        len(test_dataframe),
        len(llm_dataframe),
        category_agreement,
        section_agreement,
        exact_pair_agreement,
        category_processing_time,
        section_processing_time,
        total_processing_time,
    )

    log_hierarchy_distribution(
        llm_dataframe
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute all dataset validation checks."""
    logger.info(
        "Starting hierarchical dataset validation."
    )

    config = load_config(
        CONFIG_PATH
    )

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
        "Hierarchical LLM-labeled training dataset",
    )

    audit_dataframe = load_required_csv(
        audit_path,
        "Hierarchical LLM label audit dataset",
    )

    validate_dataset_hierarchy(
        dataframe=original_dataframe,
        dataset_name="Original dataset",
    )

    validate_dataset_hierarchy(
        dataframe=train_dataframe,
        dataset_name="Original-label training dataset",
    )

    validate_dataset_hierarchy(
        dataframe=test_dataframe,
        dataset_name="Original-label testing dataset",
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
        audit_dataframe=audit_dataframe,
    )

    logger.info(
        "All hierarchical dataset validation checks completed successfully."
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