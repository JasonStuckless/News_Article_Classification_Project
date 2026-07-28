"""
validate_labels.py

Validation utilities for hierarchical Category and Section labels generated
by the LLM.

The CNN labeling task is hierarchical:

1. The LLM predicts one broad Category.
2. The LLM predicts one Section from the valid Sections belonging to that
   Category.
3. The completed Category-Section pair is validated against the hierarchy.

This module is responsible for:

- Defining the permitted CNN Category and Section labels.
- Defining the dataset-derived Category-to-Section hierarchy.
- Providing separate JSON schemas for Category and Section generation.
- Parsing and validating Category-only responses.
- Parsing and validating Section-only responses.
- Validating completed Category-Section pairs.
- Normalizing minor formatting differences in generated labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# CNN label taxonomy
# ---------------------------------------------------------------------------

VALID_CATEGORIES: tuple[str, ...] = (
    "news",
    "business",
    "health",
    "entertainment",
    "sport",
    "politics",
)


CATEGORY_SECTION_MAP: dict[str, tuple[str, ...]] = {
    "business": (
        "business",
        "business-food",
        "business-money",
        "cars",
        "economy",
        "energy",
        "homes",
        "investing",
        "media",
        "perspectives",
        "success",
        "tech",
    ),
    "entertainment": (
        "entertainment",
        "celebrities",
        "movies",
    ),
    "health": (
        "health",
    ),
    "news": (
        "africa",
        "americas",
        "asia",
        "australia",
        "china",
        "europe",
        "india",
        "intl_world",
        "living",
        "middleeast",
        "opinions",
        "uk",
        "us",
        "weather",
        "world",
    ),
    "politics": (
        "politics",
    ),
    "sport": (
        "sport",
        "football",
        "golf",
        "motorsport",
        "tennis",
    ),
}


VALID_SECTIONS: tuple[str, ...] = tuple(
    section
    for category in VALID_CATEGORIES
    for section in CATEGORY_SECTION_MAP[category]
)


# ---------------------------------------------------------------------------
# Exceptions and validated response objects
# ---------------------------------------------------------------------------

class LabelValidationError(ValueError):
    """Raised when an LLM response does not contain valid labels."""


@dataclass(frozen=True)
class ValidatedCategory:
    """
    A validated CNN Category label.

    Attributes
    ----------
    category:
        Valid CNN Category label.
    """

    category: str


@dataclass(frozen=True)
class ValidatedSection:
    """
    A validated CNN Section label.

    Attributes
    ----------
    section:
        Valid CNN Section label belonging to the supplied Category.
    """

    section: str


@dataclass(frozen=True)
class ValidatedLabels:
    """
    A validated hierarchical Category-Section pair.

    Attributes
    ----------
    category:
        Valid CNN Category label.

    section:
        Valid CNN Section label belonging to the Category.
    """

    category: str
    section: str


# ---------------------------------------------------------------------------
# Taxonomy access
# ---------------------------------------------------------------------------

def get_valid_sections(category: str) -> tuple[str, ...]:
    """
    Return the valid Sections belonging to a Category.

    Parameters
    ----------
    category:
        Category label to look up.

    Returns
    -------
    tuple[str, ...]
        Sections that occur under the normalized Category.

    Raises
    ------
    LabelValidationError
        If the Category is invalid.
    """
    normalized_category = normalize_label(category)

    if normalized_category not in CATEGORY_SECTION_MAP:
        raise LabelValidationError(
            f"Invalid Category label: '{category}'. "
            f"Expected one of: {', '.join(VALID_CATEGORIES)}"
        )

    return CATEGORY_SECTION_MAP[normalized_category]


def get_category_for_section(section: str) -> str:
    """
    Return the Category that owns a Section.

    The original CNN dataset defines a strict hierarchy in which every
    Section belongs to exactly one Category.

    Parameters
    ----------
    section:
        Section label to look up.

    Returns
    -------
    str
        Category containing the Section.

    Raises
    ------
    LabelValidationError
        If the Section is invalid.
    """
    normalized_section = normalize_label(section)

    for category, sections in CATEGORY_SECTION_MAP.items():
        if normalized_section in sections:
            return category

    raise LabelValidationError(
        f"Invalid Section label: '{section}'. "
        f"Expected one of: {', '.join(VALID_SECTIONS)}"
    )


def format_valid_sections(category: str) -> str:
    """
    Format a Category's valid Sections as a prompt-ready bullet list.

    Parameters
    ----------
    category:
        Category whose Sections should be formatted.

    Returns
    -------
    str
        Newline-separated bullet list.
    """
    return "\n".join(
        f"- {section}"
        for section in get_valid_sections(category)
    )


# ---------------------------------------------------------------------------
# Ollama JSON schemas
# ---------------------------------------------------------------------------

def get_category_response_schema() -> dict[str, Any]:
    """
    Return the JSON schema for the Category-generation stage.

    Returns
    -------
    dict[str, Any]
        Schema requiring exactly one valid Category.
    """
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(VALID_CATEGORIES),
            },
        },
        "required": [
            "category",
        ],
        "additionalProperties": False,
    }


def get_section_response_schema(
    category: str,
) -> dict[str, Any]:
    """
    Return the JSON schema for the Section-generation stage.

    The schema restricts the Section enum to the children of the supplied
    Category.

    Parameters
    ----------
    category:
        Previously generated and validated Category.

    Returns
    -------
    dict[str, Any]
        Schema requiring exactly one Section belonging to the Category.
    """
    valid_sections = get_valid_sections(category)

    return {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "enum": list(valid_sections),
            },
        },
        "required": [
            "section",
        ],
        "additionalProperties": False,
    }


def get_label_response_schema() -> dict[str, Any]:
    """
    Return a backward-compatible combined-label JSON schema.

    This schema constrains Category and Section to their global valid sets,
    but JSON Schema alone does not enforce the hierarchical relationship.
    Call ``parse_label_response`` or ``validate_label_pair`` to enforce the
    Category-to-Section hierarchy.

    Returns
    -------
    dict[str, Any]
        Schema requiring one valid Category and one valid Section.
    """
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(VALID_CATEGORIES),
            },
            "section": {
                "type": "string",
                "enum": list(VALID_SECTIONS),
            },
        },
        "required": [
            "category",
            "section",
        ],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# General parsing helpers
# ---------------------------------------------------------------------------

def normalize_label(value: str) -> str:
    """
    Normalize minor formatting differences in a generated label.

    Normalization includes:

    - Removing leading and trailing whitespace.
    - Converting text to lowercase.
    - Converting spaces to hyphens.

    Parameters
    ----------
    value:
        Label returned by the LLM.

    Returns
    -------
    str
        Normalized label.
    """
    return value.strip().lower().replace(" ", "-")


def _parse_json_object(
    response_content: str,
) -> dict[str, Any]:
    """
    Parse an LLM response and require a JSON object.

    Parameters
    ----------
    response_content:
        Raw response returned by Ollama.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    LabelValidationError
        If the response is empty, malformed, or not an object.
    """
    if not response_content or not response_content.strip():
        raise LabelValidationError(
            "The LLM response is empty."
        )

    try:
        parsed_response = json.loads(response_content)

    except json.JSONDecodeError as error:
        raise LabelValidationError(
            f"The LLM response is not valid JSON: {error.msg}"
        ) from error

    if not isinstance(parsed_response, dict):
        raise LabelValidationError(
            "The LLM response must be a JSON object."
        )

    return parsed_response


def _validate_exact_fields(
    parsed_response: Mapping[str, Any],
    required_fields: Sequence[str],
) -> None:
    """
    Require exactly the specified response fields.

    Parameters
    ----------
    parsed_response:
        Parsed JSON object.

    required_fields:
        Exact field names permitted in the response.

    Raises
    ------
    LabelValidationError
        If required fields are missing or unexpected fields are present.
    """
    required_field_set = set(required_fields)
    response_field_set = set(parsed_response)

    unexpected_fields = (
        response_field_set - required_field_set
    )

    if unexpected_fields:
        unexpected = ", ".join(
            sorted(unexpected_fields)
        )

        raise LabelValidationError(
            "The LLM response contains unexpected fields: "
            f"{unexpected}"
        )

    missing_fields = (
        required_field_set - response_field_set
    )

    if missing_fields:
        missing = ", ".join(
            sorted(missing_fields)
        )

        raise LabelValidationError(
            "The LLM response is missing required fields: "
            f"{missing}"
        )


def _require_string(
    value: Any,
    field_name: str,
) -> str:
    """
    Require and normalize a string response field.

    Parameters
    ----------
    value:
        Parsed field value.

    field_name:
        Human-readable field name used in error messages.

    Returns
    -------
    str
        Normalized field value.

    Raises
    ------
    LabelValidationError
        If the value is not a non-empty string.
    """
    if not isinstance(value, str):
        raise LabelValidationError(
            f"The {field_name} label must be a string."
        )

    normalized_value = normalize_label(value)

    if not normalized_value:
        raise LabelValidationError(
            f"The {field_name} label cannot be blank."
        )

    return normalized_value


# ---------------------------------------------------------------------------
# Hierarchical response parsing
# ---------------------------------------------------------------------------

def parse_category_response(
    response_content: str,
) -> ValidatedCategory:
    """
    Parse and validate a Category-only LLM response.

    Expected response:

    ``{"category": "<category>"}``

    Parameters
    ----------
    response_content:
        Raw JSON response returned by Ollama.

    Returns
    -------
    ValidatedCategory
        Validated and normalized Category.

    Raises
    ------
    LabelValidationError
        If the response is malformed or contains an invalid Category.
    """
    parsed_response = _parse_json_object(
        response_content
    )

    _validate_exact_fields(
        parsed_response=parsed_response,
        required_fields=("category",),
    )

    category_value = parsed_response["category"]

    category = _require_string(
        value=category_value,
        field_name="Category",
    )

    if category not in VALID_CATEGORIES:
        raise LabelValidationError(
            f"Invalid Category label: '{category_value}'. "
            f"Expected one of: {', '.join(VALID_CATEGORIES)}"
        )

    return ValidatedCategory(
        category=category,
    )


def parse_section_response(
    response_content: str,
    category: str,
) -> ValidatedSection:
    """
    Parse and validate a Section-only LLM response.

    The Section must belong to the previously validated Category.

    Expected response:

    ``{"section": "<section>"}``

    Parameters
    ----------
    response_content:
        Raw JSON response returned by Ollama.

    category:
        Previously generated Category used to constrain valid Sections.

    Returns
    -------
    ValidatedSection
        Validated and normalized Section.

    Raises
    ------
    LabelValidationError
        If the response is malformed, the Category is invalid, or the
        Section does not belong to the Category.
    """
    normalized_category = normalize_label(category)

    valid_sections = get_valid_sections(
        normalized_category
    )

    parsed_response = _parse_json_object(
        response_content
    )

    _validate_exact_fields(
        parsed_response=parsed_response,
        required_fields=("section",),
    )

    section_value = parsed_response["section"]

    section = _require_string(
        value=section_value,
        field_name="Section",
    )

    if section not in VALID_SECTIONS:
        raise LabelValidationError(
            f"Invalid Section label: '{section_value}'. "
            f"Expected one of: {', '.join(VALID_SECTIONS)}"
        )

    if section not in valid_sections:
        raise LabelValidationError(
            f"Section '{section}' does not belong to Category "
            f"'{normalized_category}'. Valid Sections for "
            f"'{normalized_category}' are: "
            f"{', '.join(valid_sections)}"
        )

    return ValidatedSection(
        section=section,
    )


def validate_label_pair(
    category: str,
    section: str,
) -> ValidatedLabels:
    """
    Validate a completed Category-Section pair.

    Parameters
    ----------
    category:
        Generated Category label.

    section:
        Generated Section label.

    Returns
    -------
    ValidatedLabels
        Validated hierarchical pair.

    Raises
    ------
    LabelValidationError
        If either label is invalid or the Section does not belong to the
        Category.
    """
    normalized_category = normalize_label(category)
    normalized_section = normalize_label(section)

    if normalized_category not in VALID_CATEGORIES:
        raise LabelValidationError(
            f"Invalid Category label: '{category}'. "
            f"Expected one of: {', '.join(VALID_CATEGORIES)}"
        )

    if normalized_section not in VALID_SECTIONS:
        raise LabelValidationError(
            f"Invalid Section label: '{section}'. "
            f"Expected one of: {', '.join(VALID_SECTIONS)}"
        )

    valid_sections = get_valid_sections(
        normalized_category
    )

    if normalized_section not in valid_sections:
        actual_category = get_category_for_section(
            normalized_section
        )

        raise LabelValidationError(
            f"Invalid Category-Section pair: "
            f"'{normalized_category}' -> '{normalized_section}'. "
            f"Section '{normalized_section}' belongs to Category "
            f"'{actual_category}', not '{normalized_category}'."
        )

    return ValidatedLabels(
        category=normalized_category,
        section=normalized_section,
    )


# ---------------------------------------------------------------------------
# Backward-compatible combined response parsing
# ---------------------------------------------------------------------------

def parse_label_response(
    response_content: str,
) -> ValidatedLabels:
    """
    Parse and validate a combined Category-and-Section response.

    This function is retained for compatibility with the previous flat
    labeling implementation. It now also validates the hierarchical
    relationship between the labels.

    Expected response:

    ``{"category": "<category>", "section": "<section>"}``

    Parameters
    ----------
    response_content:
        Raw JSON response returned by Ollama.

    Returns
    -------
    ValidatedLabels
        Validated Category-Section pair.

    Raises
    ------
    LabelValidationError
        If the response is malformed, contains invalid labels, or contains
        a Section that does not belong to the Category.
    """
    parsed_response = _parse_json_object(
        response_content
    )

    _validate_exact_fields(
        parsed_response=parsed_response,
        required_fields=(
            "category",
            "section",
        ),
    )

    category = _require_string(
        value=parsed_response["category"],
        field_name="Category",
    )

    section = _require_string(
        value=parsed_response["section"],
        field_name="Section",
    )

    return validate_label_pair(
        category=category,
        section=section,
    )


# ---------------------------------------------------------------------------
# Convenience Boolean validation functions
# ---------------------------------------------------------------------------

def category_is_valid(
    response_content: str,
) -> bool:
    """
    Return whether a response contains one valid Category.
    """
    try:
        parse_category_response(response_content)
        return True

    except LabelValidationError:
        return False


def section_is_valid(
    response_content: str,
    category: str,
) -> bool:
    """
    Return whether a response contains a valid Section for a Category.
    """
    try:
        parse_section_response(
            response_content=response_content,
            category=category,
        )
        return True

    except LabelValidationError:
        return False


def label_pair_is_valid(
    category: str,
    section: str,
) -> bool:
    """
    Return whether Category and Section form a valid hierarchical pair.
    """
    try:
        validate_label_pair(
            category=category,
            section=section,
        )
        return True

    except LabelValidationError:
        return False


def labels_are_valid(
    response_content: str,
) -> bool:
    """
    Return whether a combined response contains a valid hierarchical pair.

    This function is retained for compatibility with the previous flat
    labeling implementation.
    """
    try:
        parse_label_response(response_content)
        return True

    except LabelValidationError:
        return False