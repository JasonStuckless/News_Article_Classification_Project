"""
02_generate_llm_labels.py

Generate hierarchical LLM-assisted Category and Section labels for the CNN
training set.

For each article, the script performs two constrained LLM requests:

1. Predict one broad CNN Category.
2. Predict one Section only from the Sections belonging to that Category.

The final Category-Section pair is validated against the dataset-derived CNN
hierarchy before it is saved.

Outputs:
- data/interim/train_llm_labeled.csv
- data/interim/llm_label_audit.csv

The LLM-labeled training file preserves every non-label value from
train_original.csv. Only Category and Section are replaced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Callable, TypeVar

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
    ValidatedCategory,
    ValidatedLabels,
    ValidatedSection,
    format_valid_sections,
    get_category_response_schema,
    get_section_response_schema,
    parse_category_response,
    parse_section_response,
    validate_label_pair,
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


SECTION_EXAMPLES: dict[str, str] = {
    "business": """Examples of distinctions within the business Category:\n\n"
    "- Stock-market or investment analysis -> investing\n"
    "- Company or general commercial reporting -> business\n"
    "- Inflation, jobs, trade, or macroeconomic policy -> economy\n"
    "- Oil, gas, electricity, or energy companies -> energy\n"
    "- Technology companies, products, or digital services -> tech\n"
    "- Automobile companies or vehicles -> cars\n"
    "- Food companies, restaurants, or the food industry -> business-food\n"
    "- Personal finance, spending, income, or money management -> business-money\n"
    "- Journalism, broadcasting, publishing, or media companies -> media\n"
    "- Housing or real estate -> homes\n"
    "- Careers, leadership, entrepreneurship, or professional success -> success\n"
    "- First-person business analysis or viewpoint pieces -> perspectives""",
    "entertainment": """Examples of distinctions within the entertainment Category:\n\n"
    "- General entertainment-industry reporting -> entertainment\n"
    "- Film releases, reviews, or the movie industry -> movies\n"
    "- Celebrity-focused reporting or interviews -> celebrities""",
    "health": """The only valid Section for the health Category is health.""",
    "news": """Examples of distinctions within the news Category:\n\n"
    "- United States domestic reporting -> us\n"
    "- United Kingdom reporting -> uk\n"
    "- European reporting outside the UK -> europe\n"
    "- African regional reporting -> africa\n"
    "- Asian regional reporting without a more specific country label -> asia\n"
    "- China-specific reporting -> china\n"
    "- India-specific reporting -> india\n"
    "- Australian reporting -> australia\n"
    "- Middle East reporting -> middleeast\n"
    "- North, Central, or South American reporting outside the US -> americas\n"
    "- Broad international reporting without one dominant region -> world\n"
    "- CNN international-world editorial coverage -> intl_world\n"
    "- Weather events or forecasts -> weather\n"
    "- Opinion columns or explicit editorial argument -> opinions\n"
    "- Lifestyle and everyday-living reporting -> living""",
    "politics": """The only valid Section for the politics Category is politics.""",
    "sport": """Examples of distinctions within the sport Category:\n\n"
    "- Association football or soccer -> football\n"
    "- Golf -> golf\n"
    "- Motor racing -> motorsport\n"
    "- Tennis -> tennis\n"
    "- General sports reporting without a more specific listed sport -> sport""",
}


class LLMLabelGenerationError(RuntimeError):
    """Raised when the hierarchical labeling pipeline cannot continue."""


TValidated = TypeVar(
    "TValidated",
    ValidatedCategory,
    ValidatedSection,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate config/config.yaml."""
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
        "category_prompt_file",
        "section_prompt_file",
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
    """Validate the original-label training dataset."""
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


def load_prompt(
    prompt_path: Path,
    required_placeholders: tuple[str, ...],
    prompt_name: str,
) -> str:
    """Load one prompt template and validate its placeholders."""
    if not prompt_path.exists():
        raise LLMLabelGenerationError(
            f"{prompt_name} prompt file not found: {prompt_path}"
        )

    prompt_template = prompt_path.read_text(encoding="utf-8")

    missing_placeholders = [
        placeholder
        for placeholder in required_placeholders
        if placeholder not in prompt_template
    ]

    if missing_placeholders:
        missing = ", ".join(missing_placeholders)
        raise LLMLabelGenerationError(
            f"The {prompt_name} prompt is missing placeholders: {missing}"
        )

    return prompt_template


# ---------------------------------------------------------------------------
# Prompt preparation
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """Convert a dataset value into safe prompt text."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def truncate_text(text: str, maximum_characters: int | None) -> str:
    """Truncate text to a fixed number of characters."""
    if maximum_characters is None:
        return text

    if maximum_characters <= 0:
        raise LLMLabelGenerationError(
            "llm.max_article_characters must be greater than zero."
        )

    if len(text) <= maximum_characters:
        return text

    return text[:maximum_characters].rstrip()


def get_article_prompt_values(
    row: pd.Series,
    maximum_article_characters: int | None,
) -> tuple[str, str, str]:
    """Return cleaned title, description, and truncated article text."""
    title = clean_text(row[TITLE_COLUMN])
    description = clean_text(row[DESCRIPTION_COLUMN])
    article = truncate_text(
        clean_text(row[ARTICLE_COLUMN]),
        maximum_characters=maximum_article_characters,
    )

    return title, description, article


def build_category_prompt(
    prompt_template: str,
    row: pd.Series,
    maximum_article_characters: int | None,
) -> str:
    """Build the Category-stage prompt for one article."""
    title, description, article = get_article_prompt_values(
        row=row,
        maximum_article_characters=maximum_article_characters,
    )

    return (
        prompt_template
        .replace("{TITLE}", title)
        .replace("{DESCRIPTION}", description)
        .replace("{ARTICLE}", article)
    )


def build_section_prompt(
    prompt_template: str,
    row: pd.Series,
    category: str,
    maximum_article_characters: int | None,
) -> str:
    """Build the Category-conditioned Section prompt for one article."""
    title, description, article = get_article_prompt_values(
        row=row,
        maximum_article_characters=maximum_article_characters,
    )

    examples = SECTION_EXAMPLES.get(
        category,
        "Choose the most appropriate valid Section for this Category.",
    )

    return (
        prompt_template
        .replace("{CATEGORY}", category)
        .replace("{VALID_SECTIONS}", format_valid_sections(category))
        .replace("{EXAMPLES}", examples)
        .replace("{TITLE}", title)
        .replace("{DESCRIPTION}", description)
        .replace("{ARTICLE}", article)
    )


# ---------------------------------------------------------------------------
# Existing progress
# ---------------------------------------------------------------------------

def create_empty_audit_dataframe() -> pd.DataFrame:
    """Create an empty audit DataFrame with the expected columns."""
    return pd.DataFrame(columns=list(AUDIT_COLUMNS))


def load_existing_audit(audit_path: Path) -> pd.DataFrame:
    """Load a compatible audit file so labeling can resume."""
    if not audit_path.exists():
        return create_empty_audit_dataframe()

    logger.info(
        "Loading existing hierarchical labeling audit from %s.",
        audit_path,
    )

    audit_dataframe = read_csv(audit_path)
    missing_columns = set(AUDIT_COLUMNS) - set(audit_dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise LLMLabelGenerationError(
            "The existing audit file is incompatible with hierarchical "
            f"prompt version 4.0 and is missing columns: {missing}. "
            "Delete the previous llm_label_audit.csv and "
            "train_llm_labeled.csv before starting the new run."
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
    """Return article identifiers with successful hierarchical labels."""
    if audit_dataframe.empty:
        return set()

    successful_rows = audit_dataframe[
        audit_dataframe["status"] == "success"
    ]

    return set(successful_rows[ID_COLUMN])


# ---------------------------------------------------------------------------
# LLM request and validation
# ---------------------------------------------------------------------------

def request_and_validate(
    client: OllamaClient,
    prompt: str,
    response_schema: dict[str, Any],
    parser: Callable[[str], TValidated],
    maximum_validation_attempts: int,
    stage_name: str,
) -> tuple[
    TValidated | None,
    OllamaResponse | None,
    str,
    int,
]:
    """Request one constrained response and retry validation failures."""
    if maximum_validation_attempts < 1:
        raise ValueError(
            f"{stage_name} validation attempts must be at least 1."
        )

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
            validated_value = parser(latest_response.content)

            return (
                validated_value,
                latest_response,
                "",
                request_attempt,
            )

        except LabelValidationError as error:
            latest_error = str(error)

            logger.warning(
                "%s response failed validation on attempt %d of %d: %s",
                stage_name,
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


def classify_category(
    client: OllamaClient,
    prompt: str,
    maximum_validation_attempts: int,
) -> tuple[
    ValidatedCategory | None,
    OllamaResponse | None,
    str,
    int,
]:
    """Generate and validate the broad Category for one article."""
    return request_and_validate(
        client=client,
        prompt=prompt,
        response_schema=get_category_response_schema(),
        parser=parse_category_response,
        maximum_validation_attempts=maximum_validation_attempts,
        stage_name="Category",
    )


def classify_section(
    client: OllamaClient,
    prompt: str,
    category: str,
    maximum_validation_attempts: int,
) -> tuple[
    ValidatedSection | None,
    OllamaResponse | None,
    str,
    int,
]:
    """Generate and validate a Section conditioned on the Category."""
    return request_and_validate(
        client=client,
        prompt=prompt,
        response_schema=get_section_response_schema(category),
        parser=lambda content: parse_section_response(
            response_content=content,
            category=category,
        ),
        maximum_validation_attempts=maximum_validation_attempts,
        stage_name="Section",
    )


# ---------------------------------------------------------------------------
# Audit and output creation
# ---------------------------------------------------------------------------

def get_response_model(
    category_response: OllamaResponse | None,
    section_response: OllamaResponse | None,
    configured_model: str,
) -> str:
    """Return the model name reported by Ollama when available."""
    if section_response is not None:
        return section_response.model

    if category_response is not None:
        return category_response.model

    return configured_model


def create_audit_record(
    row: pd.Series,
    model_name: str,
    prompt_version: str,
    labels: ValidatedLabels | None,
    category_response: OllamaResponse | None,
    section_response: OllamaResponse | None,
    status: str,
    category_validation_error: str,
    section_validation_error: str,
    category_request_attempts: int,
    section_request_attempts: int,
    category_processing_time_seconds: float,
    section_processing_time_seconds: float,
) -> dict[str, Any]:
    """Create one hierarchical audit record."""
    total_processing_time = (
        category_processing_time_seconds
        + section_processing_time_seconds
    )

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
        "llm_model": get_response_model(
            category_response=category_response,
            section_response=section_response,
            configured_model=model_name,
        ),
        "prompt_version": prompt_version,
        "raw_category_response": (
            category_response.content
            if category_response is not None
            else ""
        ),
        "raw_section_response": (
            section_response.content
            if section_response is not None
            else ""
        ),
        "status": status,
        "category_validation_error": category_validation_error,
        "section_validation_error": section_validation_error,
        "category_request_attempts": category_request_attempts,
        "section_request_attempts": section_request_attempts,
        "category_processing_time_seconds": round(
            category_processing_time_seconds,
            4,
        ),
        "section_processing_time_seconds": round(
            section_processing_time_seconds,
            4,
        ),
        "total_processing_time_seconds": round(
            total_processing_time,
            4,
        ),
        "category_prompt_token_count": (
            category_response.prompt_eval_count
            if category_response is not None
            else None
        ),
        "category_response_token_count": (
            category_response.eval_count
            if category_response is not None
            else None
        ),
        "section_prompt_token_count": (
            section_response.prompt_eval_count
            if section_response is not None
            else None
        ),
        "section_response_token_count": (
            section_response.eval_count
            if section_response is not None
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
    """Insert or replace an audit record by article identifier."""
    article_id = audit_record[ID_COLUMN]

    if audit_dataframe.empty:
        return pd.DataFrame(
            [audit_record],
            columns=list(AUDIT_COLUMNS),
        )

    remaining_records = audit_dataframe[
        audit_dataframe[ID_COLUMN] != article_id
    ]

    if remaining_records.empty:
        return pd.DataFrame(
            [audit_record],
            columns=list(AUDIT_COLUMNS),
        )

    new_record_dataframe = pd.DataFrame(
        [audit_record],
        columns=list(AUDIT_COLUMNS),
    )

    return pd.concat(
        [remaining_records, new_record_dataframe],
        ignore_index=True,
    )


def create_llm_labeled_dataset(
    original_training_dataframe: pd.DataFrame,
    audit_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create the training dataset whose targets use generated labels."""
    successful_audit = audit_dataframe[
        audit_dataframe["status"] == "success"
    ].copy()

    if successful_audit.empty:
        return original_training_dataframe.iloc[0:0].copy()

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
    """Save the current audit and generated training dataset."""
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
    """Verify completeness, hierarchy validity, and data preservation."""
    successful_audit = audit_dataframe[
        audit_dataframe["status"] == "success"
    ]

    if len(successful_audit) != len(original_training_dataframe):
        raise LLMLabelGenerationError(
            "The hierarchical LLM labeling run is incomplete. "
            f"Successful rows: {len(successful_audit)}; "
            f"expected rows: {len(original_training_dataframe)}."
        )

    if len(llm_labeled_dataframe) != len(
        original_training_dataframe
    ):
        raise LLMLabelGenerationError(
            "The final LLM-labeled dataset has an unexpected row count."
        )

    for _, audit_row in successful_audit.iterrows():
        try:
            validate_label_pair(
                category=str(audit_row["llm_category"]),
                section=str(audit_row["llm_section"]),
            )

        except LabelValidationError as error:
            raise LLMLabelGenerationError(
                "The audit contains an invalid hierarchical pair for "
                f"Index={audit_row[ID_COLUMN]}: {error}"
            ) from error

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
    """Run hierarchical LLM-assisted label generation."""
    logger.info(
        "Starting hierarchical LLM-assisted label generation."
    )

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
    category_prompt_path = resolve_project_path(
        str(llm_config["category_prompt_file"])
    )
    section_prompt_path = resolve_project_path(
        str(llm_config["section_prompt_file"])
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
    category_validation_attempts = int(
        llm_config.get("category_validation_attempts", 3)
    )
    section_validation_attempts = int(
        llm_config.get("section_validation_attempts", 3)
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
        llm_config.get("prompt_version", "4.0")
    )

    if checkpoint_interval < 1:
        raise LLMLabelGenerationError(
            "llm.checkpoint_interval must be at least 1."
        )

    if category_validation_attempts < 1:
        raise LLMLabelGenerationError(
            "llm.category_validation_attempts must be at least 1."
        )

    if section_validation_attempts < 1:
        raise LLMLabelGenerationError(
            "llm.section_validation_attempts must be at least 1."
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

    category_prompt_template = load_prompt(
        prompt_path=category_prompt_path,
        required_placeholders=(
            "{TITLE}",
            "{DESCRIPTION}",
            "{ARTICLE}",
        ),
        prompt_name="Category",
    )

    section_prompt_template = load_prompt(
        prompt_path=section_prompt_path,
        required_placeholders=(
            "{CATEGORY}",
            "{VALID_SECTIONS}",
            "{EXAMPLES}",
            "{TITLE}",
            "{DESCRIPTION}",
            "{ARTICLE}",
        ),
        prompt_name="Section",
    )

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
                "Hierarchically labeling article %d of %d. Index=%s",
                sequence_number,
                len(pending_dataframe),
                article_id,
            )

            labels: ValidatedLabels | None = None
            category_response: OllamaResponse | None = None
            section_response: OllamaResponse | None = None
            category_validation_error = ""
            section_validation_error = ""
            category_request_attempts = 0
            section_request_attempts = 0
            category_processing_time = 0.0
            section_processing_time = 0.0
            status = "request_failed"

            category_prompt = build_category_prompt(
                prompt_template=category_prompt_template,
                row=row,
                maximum_article_characters=(
                    maximum_article_characters
                ),
            )

            try:
                category_start_time = time.perf_counter()

                (
                    validated_category,
                    category_response,
                    category_validation_error,
                    category_request_attempts,
                ) = classify_category(
                    client=client,
                    prompt=category_prompt,
                    maximum_validation_attempts=(
                        category_validation_attempts
                    ),
                )

                category_processing_time = (
                    time.perf_counter() - category_start_time
                )

                if validated_category is None:
                    status = "category_validation_failed"

                else:
                    section_prompt = build_section_prompt(
                        prompt_template=section_prompt_template,
                        row=row,
                        category=validated_category.category,
                        maximum_article_characters=(
                            maximum_article_characters
                        ),
                    )

                    section_start_time = time.perf_counter()

                    (
                        validated_section,
                        section_response,
                        section_validation_error,
                        section_request_attempts,
                    ) = classify_section(
                        client=client,
                        prompt=section_prompt,
                        category=validated_category.category,
                        maximum_validation_attempts=(
                            section_validation_attempts
                        ),
                    )

                    section_processing_time = (
                        time.perf_counter() - section_start_time
                    )

                    if validated_section is None:
                        status = "section_validation_failed"

                    else:
                        labels = validate_label_pair(
                            category=validated_category.category,
                            section=validated_section.section,
                        )
                        status = "success"

            except OllamaClientError as error:
                if category_response is None:
                    category_validation_error = str(error)
                else:
                    section_validation_error = str(error)

                logger.error(
                    "Ollama failed for article Index=%s: %s",
                    article_id,
                    error,
                )

            except LabelValidationError as error:
                section_validation_error = str(error)
                status = "pair_validation_failed"

                logger.error(
                    "Final hierarchical pair validation failed for "
                    "Index=%s: %s",
                    article_id,
                    error,
                )

            audit_record = create_audit_record(
                row=row,
                model_name=model_name,
                prompt_version=prompt_version,
                labels=labels,
                category_response=category_response,
                section_response=section_response,
                status=status,
                category_validation_error=(
                    category_validation_error
                ),
                section_validation_error=(
                    section_validation_error
                ),
                category_request_attempts=(
                    category_request_attempts
                ),
                section_request_attempts=(
                    section_request_attempts
                ),
                category_processing_time_seconds=(
                    category_processing_time
                ),
                section_processing_time_seconds=(
                    section_processing_time
                ),
            )

            audit_dataframe = upsert_audit_record(
                audit_dataframe=audit_dataframe,
                audit_record=audit_record,
            )

            processed_since_checkpoint += 1

            if processed_since_checkpoint >= checkpoint_interval:
                logger.info(
                    "Saving hierarchical labeling checkpoint after %d rows.",
                    processed_since_checkpoint,
                )

                save_progress(
                    original_training_dataframe=training_dataframe,
                    audit_dataframe=audit_dataframe,
                    llm_output_path=llm_output_path,
                    audit_path=audit_path,
                )

                processed_since_checkpoint = 0

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
            "Hierarchical labeling completed with %d unsuccessful rows. "
            "Run the script again to retry them.",
            len(failed_rows),
        )

        raise LLMLabelGenerationError(
            "Some articles did not receive valid hierarchical labels. "
            "Review the audit file and rerun the script."
        )

    verify_final_output(
        original_training_dataframe=training_dataframe,
        audit_dataframe=final_audit_dataframe,
        llm_labeled_dataframe=final_llm_dataframe,
    )

    logger.info(
        "Hierarchical LLM-assisted label generation completed "
        "successfully. Generated labels for %d articles.",
        len(final_llm_dataframe),
    )


if __name__ == "__main__":
    try:
        main()

    except LLMLabelGenerationError as error:
        logger.error(
            "Hierarchical LLM label generation failed: %s",
            error,
        )
        raise SystemExit(1) from error

    except KeyboardInterrupt:
        logger.warning(
            "Hierarchical LLM label generation was interrupted by the user."
        )
        raise SystemExit(130)

    except Exception:
        logger.exception(
            "Hierarchical LLM label generation failed because of an "
            "unexpected error."
        )
        raise SystemExit(1)
