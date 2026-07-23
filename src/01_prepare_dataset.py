"""
01_prepare_dataset.py

Prepare the CNN Articles dataset for the classification experiments.

This script:
1. Loads the original CNN dataset.
2. Validates its structure and target labels.
3. Creates a reproducible train/test split stratified by Category.
4. Saves the labeled training and testing datasets.
5. Creates a full unlabeled copy by removing Category and Section.
6. Verifies that no rows or non-label values were changed.

Generated files:
- data/raw/cnn_unlabeled.csv
- data/interim/train_original.csv
- data/interim/test_original.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


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

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Index",
    "Author",
    "Date published",
    "Category",
    "Section",
    "Url",
    "Headline",
    "Description",
    "Keywords",
    "Second headline",
    "Article text",
)

LABEL_COLUMNS: tuple[str, ...] = (
    "Category",
    "Section",
)


class DatasetPreparationError(RuntimeError):
    """Raised when the dataset cannot be prepared safely."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate the project YAML configuration.

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
    DatasetPreparationError
        If the configuration file is missing or invalid.
    """
    if not config_path.exists():
        raise DatasetPreparationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise DatasetPreparationError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise DatasetPreparationError(
            "The configuration file must contain a YAML mapping."
        )

    project_config = config.get("project")
    dataset_config = config.get("dataset")

    if not isinstance(project_config, dict):
        raise DatasetPreparationError(
            "The configuration must contain a 'project' section."
        )

    if not isinstance(dataset_config, dict):
        raise DatasetPreparationError(
            "The configuration must contain a 'dataset' section."
        )

    required_project_settings = {
        "random_seed",
    }

    required_dataset_settings = {
        "original",
        "unlabeled",
        "train_file",
        "test_file",
        "train_ratio",
    }

    missing_project_settings = (
        required_project_settings - set(project_config)
    )

    missing_dataset_settings = (
        required_dataset_settings - set(dataset_config)
    )

    if missing_project_settings:
        missing = ", ".join(sorted(missing_project_settings))
        raise DatasetPreparationError(
            f"Missing project settings in config.yaml: {missing}"
        )

    if missing_dataset_settings:
        missing = ", ".join(sorted(missing_dataset_settings))
        raise DatasetPreparationError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """
    Resolve a configured path relative to the project root.

    Absolute paths are returned unchanged.
    """
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_split_settings(
    train_ratio: float,
    random_seed: int,
) -> None:
    """Validate the configured train/test split settings."""
    if not 0.0 < train_ratio < 1.0:
        raise DatasetPreparationError(
            "dataset.train_ratio must be greater than 0 and less than 1."
        )

    if not isinstance(random_seed, int):
        raise DatasetPreparationError(
            "project.random_seed must be an integer."
        )


def validate_original_dataset(dataframe: pd.DataFrame) -> None:
    """
    Validate the structure and labels of the original CNN dataset.

    Raises
    ------
    DatasetPreparationError
        If required columns are missing, the dataset is empty, identifiers
        are invalid, or target labels are missing.
    """
    if dataframe.empty:
        raise DatasetPreparationError(
            "The original dataset contains no rows."
        )

    missing_columns = set(REQUIRED_COLUMNS) - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DatasetPreparationError(
            f"The original dataset is missing required columns: {missing}"
        )

    additional_columns = set(dataframe.columns) - set(REQUIRED_COLUMNS)

    if additional_columns:
        additional = ", ".join(sorted(additional_columns))
        logger.warning(
            "Additional dataset columns will be preserved: %s",
            additional,
        )

    if dataframe["Index"].isna().any():
        raise DatasetPreparationError(
            "The Index column contains missing values."
        )

    if not dataframe["Index"].is_unique:
        duplicate_count = int(
            dataframe["Index"].duplicated(keep=False).sum()
        )

        raise DatasetPreparationError(
            f"The Index column contains {duplicate_count} rows with "
            "duplicated identifiers."
        )

    for label_column in LABEL_COLUMNS:
        missing_count = int(dataframe[label_column].isna().sum())

        if missing_count > 0:
            raise DatasetPreparationError(
                f"The {label_column} column contains "
                f"{missing_count} missing values."
            )

        blank_mask = (
            dataframe[label_column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        blank_count = int(blank_mask.sum())

        if blank_count > 0:
            raise DatasetPreparationError(
                f"The {label_column} column contains "
                f"{blank_count} blank values."
            )


def report_label_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Log the Category and Section distributions of a dataset."""
    logger.info("%s rows: %d", dataset_name, len(dataframe))

    category_counts = (
        dataframe["Category"]
        .value_counts()
        .sort_index()
    )

    logger.info(
        "%s Category distribution:\n%s",
        dataset_name,
        category_counts.to_string(),
    )

    section_counts = (
        dataframe["Section"]
        .value_counts()
        .sort_values()
    )

    logger.info(
        "%s contains %d unique Section labels.",
        dataset_name,
        dataframe["Section"].nunique(),
    )

    rare_sections = section_counts[section_counts < 2]

    if not rare_sections.empty:
        logger.warning(
            "%s has Section labels with fewer than two examples. "
            "These labels cannot be guaranteed to appear in both splits:\n%s",
            dataset_name,
            rare_sections.to_string(),
        )


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_train_test_split(
    dataframe: pd.DataFrame,
    train_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a reproducible train/test split stratified by Category.

    Section is not used for stratification because some Section labels have
    too few examples to appear in both training and testing sets.
    """
    train_dataframe, test_dataframe = train_test_split(
        dataframe,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=dataframe["Category"],
        shuffle=True,
    )

    train_dataframe = train_dataframe.reset_index(drop=True)
    test_dataframe = test_dataframe.reset_index(drop=True)

    return train_dataframe, test_dataframe


def create_unlabeled_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a copy of the complete dataset without Category or Section.

    All other columns, rows, values, and row ordering are preserved.
    """
    return dataframe.drop(
        columns=list(LABEL_COLUMNS)
    ).copy()


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def verify_unlabeled_dataset(
    original: pd.DataFrame,
    unlabeled: pd.DataFrame,
) -> None:
    """
    Verify that only Category and Section were removed.

    The remaining data must be exactly equal to the corresponding columns
    in the original dataset.
    """
    expected_columns = [
        column
        for column in original.columns
        if column not in LABEL_COLUMNS
    ]

    if list(unlabeled.columns) != expected_columns:
        raise DatasetPreparationError(
            "The unlabeled dataset has unexpected columns or column order."
        )

    if len(unlabeled) != len(original):
        raise DatasetPreparationError(
            "The unlabeled dataset row count differs from the original."
        )

    expected_dataframe = (
        original.loc[:, expected_columns]
        .reset_index(drop=True)
    )

    actual_dataframe = unlabeled.reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            expected_dataframe,
            actual_dataframe,
            check_dtype=True,
            check_like=False,
        )

    except AssertionError as error:
        raise DatasetPreparationError(
            "The unlabeled dataset differs from the original beyond "
            "removing Category and Section."
        ) from error


def verify_split_integrity(
    original: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """
    Verify that every original article appears exactly once across the splits.
    """
    original_ids = set(original["Index"])
    train_ids = set(train_dataframe["Index"])
    test_ids = set(test_dataframe["Index"])

    overlapping_ids = train_ids.intersection(test_ids)

    if overlapping_ids:
        raise DatasetPreparationError(
            f"The train and test sets share "
            f"{len(overlapping_ids)} Index values."
        )

    combined_ids = train_ids.union(test_ids)

    missing_ids = original_ids - combined_ids
    unexpected_ids = combined_ids - original_ids

    if missing_ids or unexpected_ids:
        raise DatasetPreparationError(
            "The generated splits do not reproduce the original dataset. "
            f"Missing identifiers: {len(missing_ids)}; "
            f"unexpected identifiers: {len(unexpected_ids)}."
        )

    if len(train_dataframe) + len(test_dataframe) != len(original):
        raise DatasetPreparationError(
            "The train and test row counts do not sum to the original "
            "dataset row count."
        )


def verify_written_csv(
    expected_dataframe: pd.DataFrame,
    output_path: Path,
    output_name: str,
) -> None:
    """
    Reload a generated CSV and verify its values.

    This catches writing or serialization problems before the script exits.
    """
    reloaded_dataframe = read_csv(output_path)

    try:
        pd.testing.assert_frame_equal(
            expected_dataframe.reset_index(drop=True),
            reloaded_dataframe.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )

    except AssertionError as error:
        raise DatasetPreparationError(
            f"The written {output_name} file does not match the "
            "DataFrame that was saved."
        ) from error


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the dataset preparation pipeline."""
    logger.info("Starting dataset preparation.")

    config = load_config(CONFIG_PATH)

    project_config = config["project"]
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

    train_ratio = float(dataset_config["train_ratio"])
    random_seed = project_config["random_seed"]

    validate_split_settings(
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    if not original_path.exists():
        raise DatasetPreparationError(
            f"Original dataset not found: {original_path}"
        )

    logger.info(
        "Loading the original dataset from %s.",
        original_path,
    )

    original_dataframe = read_csv(original_path)

    validate_original_dataset(original_dataframe)

    logger.info(
        "Original dataset successfully validated: "
        "%d rows and %d columns.",
        len(original_dataframe),
        len(original_dataframe.columns),
    )

    report_label_distribution(
        original_dataframe,
        "Original dataset",
    )

    train_dataframe, test_dataframe = create_train_test_split(
        dataframe=original_dataframe,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    verify_split_integrity(
        original=original_dataframe,
        train_dataframe=train_dataframe,
        test_dataframe=test_dataframe,
    )

    unlabeled_dataframe = create_unlabeled_dataset(
        original_dataframe
    )

    verify_unlabeled_dataset(
        original=original_dataframe,
        unlabeled=unlabeled_dataframe,
    )

    report_label_distribution(
        train_dataframe,
        "Training dataset",
    )

    report_label_distribution(
        test_dataframe,
        "Testing dataset",
    )

    logger.info(
        "Saving the unlabeled dataset to %s.",
        unlabeled_path,
    )
    write_csv(unlabeled_dataframe, unlabeled_path)

    logger.info(
        "Saving the original-label training dataset to %s.",
        train_path,
    )
    write_csv(train_dataframe, train_path)

    logger.info(
        "Saving the original-label testing dataset to %s.",
        test_path,
    )
    write_csv(test_dataframe, test_path)

    verify_written_csv(
        expected_dataframe=unlabeled_dataframe,
        output_path=unlabeled_path,
        output_name="unlabeled dataset",
    )

    verify_written_csv(
        expected_dataframe=train_dataframe,
        output_path=train_path,
        output_name="training dataset",
    )

    verify_written_csv(
        expected_dataframe=test_dataframe,
        output_path=test_path,
        output_name="testing dataset",
    )

    logger.info(
        "Dataset preparation completed successfully. "
        "Training rows: %d; testing rows: %d.",
        len(train_dataframe),
        len(test_dataframe),
    )


if __name__ == "__main__":
    try:
        main()

    except DatasetPreparationError as error:
        logger.error("Dataset preparation failed: %s", error)
        raise SystemExit(1) from error

    except Exception:
        logger.exception(
            "Dataset preparation failed because of an unexpected error."
        )
        raise SystemExit(1)