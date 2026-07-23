"""
02_generate_llm_labels.py

Generate LLM-assisted Category and Section labels for the CNN training set.

This script:
1. Loads project settings from config/config.yaml.
2. Loads the original-label training dataset.
3. Loads the classification prompt.
4. Removes the real labels from each article before prompting the LLM.
5. Uses the configured Ollama model to generate Category and Section labels.
6. Validates every generated response.
7. Retries invalid responses.
8. Saves progress periodically so interrupted runs can resume.
9. Creates:
   - data/interim/train_llm_labeled.csv
   - data/interim/llm_label_audit.csv

The LLM-labeled training file uses the same structure as train_original.csv,
but its Category and Section columns contain the LLM-generated labels.

The audit file retains both the original and generated labels for analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time
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

from llm.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaResponse,
)
from llm.validate_labels import (
    LabelValidationError,
    ValidatedLabels,
    get_label_response_schema,
    parse_label_response,
)
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

ID_COLUMN = "Index"
CATEGORY_COLUMN = "Category"
SECTION_COLUMN = "Section"

TITLE_COLUMN = "Headline"
DESCRIPTION_COLUMN = "Description"
ARTICLE_COLUMN = "Article text"

REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    CATEGORY_COLUMN,
    SECTION_COLUMN,
    TITLE_COLUMN,
    DESCRIPTION_COLUMN,
    ARTICLE_COLUMN,
)

AUDIT_COLUMNS: tuple[str, ...] = (
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


class LLMLabelGenerationError(RuntimeError):
    """Raised when the LLM-labeling pipeline cannot continue safely."""


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
        Parsed configuration dictionary.

    Raises
    ------
    LLMLabelGenerationError
        If the configuration is missing or malformed.
    """
    if not config_path.exists():
        raise LLMLabelGenerationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as error:
        raise LLMLabelGenerationError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise LLMLabelGenerationError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    llm_config = config.get("llm")

    if not isinstance(dataset_config, dict):
        raise LLMLabelGenerationError(
            "The configuration must contain a 'dataset' section."
        )

    if not isinstance(llm_config, dict):
        raise LLMLabelGenerationError(
            "The configuration must contain an 'llm' section."
        )

    required_dataset_settings = {
        "train_file",
        "llm_train_file",
        "audit_file",
    }

    required_llm_settings = {
        "model",
        "temperature",
        "prompt_file",
    }

    missing_dataset_settings = (
        required_dataset_settings - set(dataset_config)
    )

    missing_llm_settings = required_llm_settings - set(llm_config)

    if missing_dataset_settings:
        missing = ", ".join(sorted(missing_dataset_settings))
        raise LLMLabelGenerationError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    if missing_llm_settings:
        missing = ", ".join(sorted(missing_llm_settings))
        raise LLMLabelGenerationError(
            f"Missing LLM settings in config.yaml: {missing}"
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def validate_training_dataset(dataframe: pd.DataFrame) -> None:
    """
    Validate the original-label training dataset.

    Raises
    ------
    LLMLabelGenerationError
        If required columns or identifiers are invalid.
    """
    if dataframe.empty:
        raise LLMLabelGenerationError(
            "The training dataset contains no rows."
        )

    missing_columns = set(REQUIRED_COLUMNS) - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise LLMLabelGenerationError(
            f"The training dataset is missing required columns: {missing}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise LLMLabelGenerationError(
            f"The {ID_COLUMN} column contains missing values."
        )

    if not dataframe[ID_COLUMN].is_unique:
        raise LLMLabelGenerationError(
            f"The {ID_COLUMN} column must contain unique values."
        )

    for label_column in (CATEGORY_COLUMN, SECTION_COLUMN):
        if dataframe[label_column].isna().any():
            raise LLMLabelGenerationError(
                f"The {label_column} column contains missing values."
            )


def load_prompt(prompt_path: Path) -> str:
    """
    Load the article-classification prompt template.

    The prompt must contain:
    - {TITLE}
    - {DESCRIPTION}
    - {ARTICLE}
    """
    if not prompt_path.exists():
        raise LLMLabelGenerationError(
            f"Prompt file not found: {prompt_path}"
        )

    prompt_template = prompt_path.read_text(encoding="utf-8")

    required_placeholders = (
        "{TITLE}",
        "{DESCRIPTION}",
        "{ARTICLE}",
    )

    missing_placeholders = [
        placeholder
        for placeholder in required_placeholders
        if placeholder not in prompt_template
    ]

    if missing_placeholders:
        missing = ", ".join(missing_placeholders)
        raise LLMLabelGenerationError(
            f"The prompt file is missing placeholders: {missing}"
        )

    return prompt_template


# ---------------------------------------------------------------------------
# Prompt preparation
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """
    Convert a dataset value into safe prompt text.

    Missing values become an empty string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def truncate_text(text: str, maximum_characters: int | None) -> str:
    """
    Truncate text to a fixed number of characters.

    A value of None means no truncation.
    """
    if maximum_characters is None:
        return text

    if maximum_characters <= 0:
        raise LLMLabelGenerationError(
            "llm.max_article_characters must be greater than zero."
        )

    if len(text) <= maximum_characters:
        return text

    return text[:maximum_characters].rstrip()


def build_prompt(
    prompt_template: str,
    row: pd.Series,
    maximum_article_characters: int | None,
) -> str:
    """
    Build the prompt for one article.

    String replacement is used instead of str.format() because the prompt
    contains literal JSON braces.
    """
    title = clean_text(row[TITLE_COLUMN])
    description = clean_text(row[DESCRIPTION_COLUMN])
    article = clean_text(row[ARTICLE_COLUMN])

    article = truncate_text(
        article,
        maximum_characters=maximum_article_characters,
    )

    return (
        prompt_template
        .replace("{TITLE}", title)
        .replace("{DESCRIPTION}", description)
        .replace("{ARTICLE}", article)
    )


# ---------------------------------------------------------------------------
# Existing progress
# ---------------------------------------------------------------------------

def create_empty_audit_dataframe() -> pd.DataFrame:
    """Create an empty audit DataFrame with the expected column order."""
    return pd.DataFrame(columns=list(AUDIT_COLUMNS))


def load_existing_audit(audit_path: Path) -> pd.DataFrame:
    """
    Load a previous audit file so the labeling run can resume.

    Only rows with status='success' are considered completed.
    """
    if not audit_path.exists():
        return create_empty_audit_dataframe()

    logger.info(
        "Loading existing labeling audit from %s.",
        audit_path,
    )

    audit_dataframe = read_csv(audit_path)

    missing_columns = set(AUDIT_COLUMNS) - set(audit_dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise LLMLabelGenerationError(
            f"The existing audit file is missing columns: {missing}"
        )

    if audit_dataframe[ID_COLUMN].duplicated().any():
        duplicate_count = int(
            audit_dataframe[ID_COLUMN].duplicated(keep=False).sum()
        )

        raise LLMLabelGenerationError(
            f"The audit file contains {duplicate_count} rows with "
            f"duplicated {ID_COLUMN} values."
        )

    return audit_dataframe.loc[:, list(AUDIT_COLUMNS)].copy()


def get_completed_ids(audit_dataframe: pd.DataFrame) -> set[Any]:
    """Return identifiers that already have successful LLM labels."""
    if audit_dataframe.empty:
        return set()

    successful_rows = audit_dataframe[
        audit_dataframe["status"] == "success"
    ]

    return set(successful_rows[ID_COLUMN])


# ---------------------------------------------------------------------------
# LLM request and validation
# ---------------------------------------------------------------------------

def classify_article(
    client: OllamaClient,
    prompt: str,
    maximum_validation_attempts: int,
) -> tuple[
    ValidatedLabels | None,
    OllamaResponse | None,
    str,
    int,
]:
    """
    Generate and validate labels for one article.

    Parameters
    ----------
    client:
        Configured Ollama client.

    prompt:
        Complete article-classification prompt.

    maximum_validation_attempts:
        Number of times to request a new response when output validation
        fails.

    Returns
    -------
    tuple
        Validated labels or None, latest response or None, validation error,
        and number of requests made.
    """
    if maximum_validation_attempts < 1:
        raise ValueError(
            "maximum_validation_attempts must be at least 1."
        )

    response_schema = get_label_response_schema()
    latest_response: OllamaResponse | None = None
    latest_error = ""

    for request_attempt in range(
        1,
        maximum_validation_attempts + 1,
    ):
        latest_response = client.generate(
            prompt=prompt,
            response_schema=response_schema,
        )

        try:
            labels = parse_label_response(
                latest_response.content
            )

            return (
                labels,
                latest_response,
                "",
                request_attempt,
            )

        except LabelValidationError as error:
            latest_error = str(error)

            logger.warning(
                "LLM response failed label validation on attempt "
                "%d of %d: %s",
                request_attempt,
                maximum_validation_attempts,
                error,
            )

    return (
        None,
        latest_response,
        latest_error,
        maximum_validation_attempts,
    )


# ---------------------------------------------------------------------------
# Audit and output creation
# ---------------------------------------------------------------------------

def create_audit_record(
    row: pd.Series,
    model_name: str,
    prompt_version: str,
    labels: ValidatedLabels | None,
    response: OllamaResponse | None,
    status: str,
    validation_error: str,
    request_attempts: int,
    processing_time_seconds: float,
) -> dict[str, Any]:
    """Create one audit record for an article-labeling attempt."""
    return {
        ID_COLUMN: row[ID_COLUMN],
        "original_category": row[CATEGORY_COLUMN],
        "original_section": row[SECTION_COLUMN],
        "llm_category": (
            labels.category if labels is not None else ""
        ),
        "llm_section": (
            labels.section if labels is not None else ""
        ),
        "llm_model": (
            response.model if response is not None else model_name
        ),
        "prompt_version": prompt_version,
        "raw_response": (
            response.content if response is not None else ""
        ),
        "status": status,
        "validation_error": validation_error,
        "request_attempts": request_attempts,
        "processing_time_seconds": round(
            processing_time_seconds,
            4,
        ),
        "prompt_token_count": (
            response.prompt_eval_count
            if response is not None
            else None
        ),
        "response_token_count": (
            response.eval_count
            if response is not None
            else None
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def upsert_audit_record(
    audit_dataframe: pd.DataFrame,
    audit_record: dict[str, Any],
) -> pd.DataFrame:
    """
    Insert or replace an audit record by article identifier.

    Failed records can therefore be replaced by successful records if the
    script is run again.
    """
    article_id = audit_record[ID_COLUMN]

    if not audit_dataframe.empty:
        audit_dataframe = audit_dataframe[
            audit_dataframe[ID_COLUMN] != article_id
        ]

    new_record_dataframe = pd.DataFrame(
        [audit_record],
        columns=list(AUDIT_COLUMNS),
    )

    return pd.concat(
        [audit_dataframe, new_record_dataframe],
        ignore_index=True,
    )


def create_llm_labeled_dataset(
    original_training_dataframe: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the training dataset whose target columns use LLM labels.

    Only successfully labeled rows are included. Before the final pipeline
    proceeds, the row count should equal the original training row count.
    """
    successful_audit = audit_dataframe[
        audit_dataframe["status"] == "success"
    ].copy()

    label_lookup = successful_audit.set_index(ID_COLUMN)[
        ["llm_category", "llm_section"]
    ]

    llm_labeled_dataframe = original_training_dataframe.copy()

    llm_labeled_dataframe[CATEGORY_COLUMN] = (
        llm_labeled_dataframe[ID_COLUMN]
        .map(label_lookup["llm_category"])
    )

    llm_labeled_dataframe[SECTION_COLUMN] = (
        llm_labeled_dataframe[ID_COLUMN]
        .map(label_lookup["llm_section"])
    )

    # During checkpointing, unsuccessful or incomplete rows are omitted.
    completed_mask = (
        llm_labeled_dataframe[CATEGORY_COLUMN].notna()
        & llm_labeled_dataframe[SECTION_COLUMN].notna()
    )

    return (
        llm_labeled_dataframe[completed_mask]
        .reset_index(drop=True)
    )


def save_progress(
    original_training_dataframe: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
    llm_output_path: Path,
    audit_path: Path,
) -> None:
    """Save the current audit and LLM-labeled training datasets."""
    audit_dataframe = (
        audit_dataframe
        .sort_values(by=ID_COLUMN)
        .reset_index(drop=True)
    )

    llm_labeled_dataframe = create_llm_labeled_dataset(
        original_training_dataframe=original_training_dataframe,
        audit_dataframe=audit_dataframe,
    )

    write_csv(
        dataframe=audit_dataframe,
        file_path=audit_path,
    )

    write_csv(
        dataframe=llm_labeled_dataframe,
        file_path=llm_output_path,
    )


def verify_final_output(
    original_training_dataframe: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
    llm_labeled_dataframe: pd.DataFrame,
) -> None:
    """
    Verify that all training rows received valid LLM labels.

    Raises
    ------
    LLMLabelGenerationError
        If labeling is incomplete or article data changed.
    """
    successful_audit = audit_dataframe[
        audit_dataframe["status"] == "success"
    ]

    if len(successful_audit) != len(original_training_dataframe):
        raise LLMLabelGenerationError(
            "The LLM labeling run is incomplete. "
            f"Successful rows: {len(successful_audit)}; "
            f"expected rows: {len(original_training_dataframe)}."
        )

    if len(llm_labeled_dataframe) != len(
        original_training_dataframe
    ):
        raise LLMLabelGenerationError(
            "The final LLM-labeled dataset has an unexpected row count."
        )

    non_label_columns = [
        column
        for column in original_training_dataframe.columns
        if column not in (CATEGORY_COLUMN, SECTION_COLUMN)
    ]

    expected_data = (
        original_training_dataframe[non_label_columns]
        .reset_index(drop=True)
    )

    actual_data = (
        llm_labeled_dataframe[non_label_columns]
        .reset_index(drop=True)
    )

    try:
        pd.testing.assert_frame_equal(
            expected_data,
            actual_data,
            check_dtype=True,
            check_like=False,
        )

    except AssertionError as error:
        raise LLMLabelGenerationError(
            "Non-label values changed while creating the "
            "LLM-labeled dataset."
        ) from error


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the LLM-assisted labeling pipeline."""
    logger.info("Starting LLM-assisted label generation.")

    config = load_config(CONFIG_PATH)

    dataset_config = config["dataset"]
    llm_config = config["llm"]

    training_path = resolve_project_path(
        str(dataset_config["train_file"])
    )

    llm_output_path = resolve_project_path(
        str(dataset_config["llm_train_file"])
    )

    audit_path = resolve_project_path(
        str(dataset_config["audit_file"])
    )

    prompt_path = resolve_project_path(
        str(llm_config["prompt_file"])
    )

    model_name = str(llm_config["model"])
    host = str(
        llm_config.get(
            "host",
            "http://localhost:11434",
        )
    )

    temperature = float(
        llm_config.get("temperature", 0.0)
    )

    timeout_seconds = float(
        llm_config.get("timeout_seconds", 300.0)
    )

    connection_retries = int(
        llm_config.get("connection_retries", 2)
    )

    retry_delay_seconds = float(
        llm_config.get("retry_delay_seconds", 2.0)
    )

    validation_attempts = int(
        llm_config.get("validation_attempts", 3)
    )

    checkpoint_interval = int(
        llm_config.get("checkpoint_interval", 10)
    )

    maximum_article_characters_value = llm_config.get(
        "max_article_characters",
        3000,
    )

    maximum_article_characters = (
        None
        if maximum_article_characters_value is None
        else int(maximum_article_characters_value)
    )

    prompt_version = str(
        llm_config.get("prompt_version", "1.0")
    )

    if checkpoint_interval < 1:
        raise LLMLabelGenerationError(
            "llm.checkpoint_interval must be at least 1."
        )

    if not training_path.exists():
        raise LLMLabelGenerationError(
            f"Training dataset not found: {training_path}. "
            "Run 01_prepare_dataset.py first."
        )

    logger.info(
        "Loading training dataset from %s.",
        training_path,
    )

    training_dataframe = read_csv(training_path)
    validate_training_dataset(training_dataframe)

    prompt_template = load_prompt(prompt_path)
    audit_dataframe = load_existing_audit(audit_path)
    completed_ids = get_completed_ids(audit_dataframe)

    pending_dataframe = training_dataframe[
        ~training_dataframe[ID_COLUMN].isin(completed_ids)
    ]

    logger.info(
        "Training rows: %d; already completed: %d; remaining: %d.",
        len(training_dataframe),
        len(completed_ids),
        len(pending_dataframe),
    )

    with OllamaClient(
        model=model_name,
        host=host,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=connection_retries,
        retry_delay_seconds=retry_delay_seconds,
    ) as client:
        client.verify_connection()
        client.verify_model()

        processed_since_checkpoint = 0

        for sequence_number, (_, row) in enumerate(
            pending_dataframe.iterrows(),
            start=1,
        ):
            article_id = row[ID_COLUMN]

            logger.info(
                "Labeling article %d of %d. Index=%s",
                sequence_number,
                len(pending_dataframe),
                article_id,
            )

            prompt = build_prompt(
                prompt_template=prompt_template,
                row=row,
                maximum_article_characters=(
                    maximum_article_characters
                ),
            )

            start_time = time.perf_counter()

            try:
                (
                    labels,
                    response,
                    validation_error,
                    request_attempts,
                ) = classify_article(
                    client=client,
                    prompt=prompt,
                    maximum_validation_attempts=(
                        validation_attempts
                    ),
                )

                if labels is None:
                    status = "validation_failed"
                else:
                    status = "success"

            except OllamaClientError as error:
                labels = None
                response = None
                validation_error = str(error)
                request_attempts = 1
                status = "request_failed"

                logger.error(
                    "Ollama failed for article Index=%s: %s",
                    article_id,
                    error,
                )

            processing_time = (
                time.perf_counter() - start_time
            )

            audit_record = create_audit_record(
                row=row,
                model_name=model_name,
                prompt_version=prompt_version,
                labels=labels,
                response=response,
                status=status,
                validation_error=validation_error,
                request_attempts=request_attempts,
                processing_time_seconds=processing_time,
            )

            audit_dataframe = upsert_audit_record(
                audit_dataframe=audit_dataframe,
                audit_record=audit_record,
            )

            processed_since_checkpoint += 1

            if (
                processed_since_checkpoint
                >= checkpoint_interval
            ):
                logger.info(
                    "Saving labeling checkpoint after %d rows.",
                    processed_since_checkpoint,
                )

                save_progress(
                    original_training_dataframe=(
                        training_dataframe
                    ),
                    audit_dataframe=audit_dataframe,
                    llm_output_path=llm_output_path,
                    audit_path=audit_path,
                )

                processed_since_checkpoint = 0

    # Always save after the loop, including when no rows were pending.
    save_progress(
        original_training_dataframe=training_dataframe,
        audit_dataframe=audit_dataframe,
        llm_output_path=llm_output_path,
        audit_path=audit_path,
    )

    final_audit_dataframe = read_csv(audit_path)
    final_llm_dataframe = read_csv(llm_output_path)

    failed_rows = final_audit_dataframe[
        final_audit_dataframe["status"] != "success"
    ]

    if not failed_rows.empty:
        logger.warning(
            "LLM labeling completed with %d unsuccessful rows. "
            "Run the script again to retry them.",
            len(failed_rows),
        )

        raise LLMLabelGenerationError(
            "Some articles did not receive valid LLM labels. "
            "Review the audit file and rerun the script."
        )

    verify_final_output(
        original_training_dataframe=training_dataframe,
        audit_dataframe=final_audit_dataframe,
        llm_labeled_dataframe=final_llm_dataframe,
    )

    logger.info(
        "LLM-assisted label generation completed successfully. "
        "Generated labels for %d articles.",
        len(final_llm_dataframe),
    )


if __name__ == "__main__":
    try:
        main()

    except LLMLabelGenerationError as error:
        logger.error(
            "LLM-assisted label generation failed: %s",
            error,
        )
        raise SystemExit(1) from error

    except KeyboardInterrupt:
        logger.warning(
            "LLM-assisted label generation was interrupted by the user."
        )
        raise SystemExit(130)

    except Exception:
        logger.exception(
            "LLM-assisted label generation failed because of an "
            "unexpected error."
        )
        raise SystemExit(1)