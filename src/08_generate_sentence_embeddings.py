"""
08_generate_sentence_embeddings.py

Generate normalized Sentence-BERT embeddings for the CNN News Articles dataset.

This stage adapts the Sentence-BERT methodology developed in the project
notebook and integrates it into the reproducible Python pipeline. It applies the
same enhanced text cleaning used in the notebook, loads
"sentence-transformers/all-MiniLM-L6-v2", and generates normalized dense
embeddings in batches.

Inputs:
- data/raw/cnn_original.csv
- data/interim/train_original.csv
- data/interim/test_original.csv
- data/interim/train_llm_labeled.csv

Outputs:
- data/processed/sentence_embeddings/train_embeddings.npy
- data/processed/sentence_embeddings/test_embeddings.npy
- data/processed/sentence_embeddings/full_embeddings.npy
- data/processed/sentence_embeddings/train_metadata.csv
- data/processed/sentence_embeddings/test_metadata.csv
- data/processed/sentence_embeddings/full_metadata.csv
- data/processed/sentence_embeddings/embedding_manifest.json

The article embeddings are independent of the label source. The same training
embedding matrix is therefore reused later for models trained with original CNN
labels and LLM-generated labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDING_DIRECTORY = PROCESSED_DATA_DIR / "sentence_embeddings"

TRAIN_EMBEDDINGS_PATH = EMBEDDING_DIRECTORY / "train_embeddings.npy"
TEST_EMBEDDINGS_PATH = EMBEDDING_DIRECTORY / "test_embeddings.npy"
FULL_EMBEDDINGS_PATH = EMBEDDING_DIRECTORY / "full_embeddings.npy"

TRAIN_METADATA_PATH = EMBEDDING_DIRECTORY / "train_metadata.csv"
TEST_METADATA_PATH = EMBEDDING_DIRECTORY / "test_metadata.csv"
FULL_METADATA_PATH = EMBEDDING_DIRECTORY / "full_metadata.csv"
MANIFEST_PATH = EMBEDDING_DIRECTORY / "embedding_manifest.json"

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

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NORMALIZE_EMBEDDINGS = True
DEFAULT_SHOW_PROGRESS_BAR = True
DEFAULT_DEVICE = "auto"
DEFAULT_OVERWRITE = False


class SentenceEmbeddingGenerationError(RuntimeError):
    """Raised when Sentence-BERT embeddings cannot be generated safely."""


@dataclass(frozen=True)
class EmbeddingSettings:
    """Resolved Sentence-BERT settings for this pipeline stage."""

    model_name: str
    batch_size: int
    normalize_embeddings: bool
    show_progress_bar: bool
    device: str | None
    overwrite: bool


# ---------------------------------------------------------------------------
# Configuration and paths
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the configuration required by this stage."""
    if not config_path.exists():
        raise SentenceEmbeddingGenerationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise SentenceEmbeddingGenerationError(
            f"Could not parse the configuration file: {error}"
        ) from error

    if not isinstance(config, dict):
        raise SentenceEmbeddingGenerationError(
            "The configuration file must contain a YAML mapping."
        )

    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise SentenceEmbeddingGenerationError(
            "The configuration must contain a 'dataset' section."
        )

    required_settings = {
        "original",
        "train_file",
        "test_file",
        "llm_train_file",
    }
    missing_settings = required_settings - set(dataset_config)

    if missing_settings:
        missing = ", ".join(sorted(missing_settings))
        raise SentenceEmbeddingGenerationError(
            f"Missing dataset settings in config.yaml: {missing}"
        )

    return config


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_embedding_settings(config: dict[str, Any]) -> EmbeddingSettings:
    """Resolve Sentence-BERT settings, using notebook-compatible defaults."""
    raw_settings = config.get("sentence_bert", {})

    if raw_settings is None:
        raw_settings = {}
    if not isinstance(raw_settings, dict):
        raise SentenceEmbeddingGenerationError(
            "sentence_bert must be a YAML mapping when provided."
        )

    model_name = str(
        raw_settings.get("model_name", DEFAULT_MODEL_NAME)
    ).strip()
    batch_size = raw_settings.get("batch_size", DEFAULT_BATCH_SIZE)
    normalize_embeddings = raw_settings.get(
        "normalize_embeddings",
        DEFAULT_NORMALIZE_EMBEDDINGS,
    )
    show_progress_bar = raw_settings.get(
        "show_progress_bar",
        DEFAULT_SHOW_PROGRESS_BAR,
    )
    configured_device = str(
        raw_settings.get("device", DEFAULT_DEVICE)
    ).strip()
    overwrite = raw_settings.get("overwrite_embeddings", DEFAULT_OVERWRITE)

    if not model_name:
        raise SentenceEmbeddingGenerationError(
            "sentence_bert.model_name cannot be empty."
        )
    if not isinstance(batch_size, int) or batch_size < 1:
        raise SentenceEmbeddingGenerationError(
            "sentence_bert.batch_size must be a positive integer."
        )
    if not isinstance(normalize_embeddings, bool):
        raise SentenceEmbeddingGenerationError(
            "sentence_bert.normalize_embeddings must be true or false."
        )
    if not isinstance(show_progress_bar, bool):
        raise SentenceEmbeddingGenerationError(
            "sentence_bert.show_progress_bar must be true or false."
        )
    if not isinstance(overwrite, bool):
        raise SentenceEmbeddingGenerationError(
            "sentence_bert.overwrite_embeddings must be true or false."
        )

    device = None if configured_device.lower() == "auto" else configured_device

    return EmbeddingSettings(
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        device=device,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(dataframe: pd.DataFrame, dataset_name: str) -> None:
    """Validate the structure needed for embedding generation."""
    if dataframe.empty:
        raise SentenceEmbeddingGenerationError(
            f"The {dataset_name} contains no rows."
        )

    missing_columns = set(REQUIRED_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SentenceEmbeddingGenerationError(
            f"The {dataset_name} is missing required columns: {missing}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise SentenceEmbeddingGenerationError(
            f"The {dataset_name} contains missing {ID_COLUMN} values."
        )

    if dataframe[ID_COLUMN].duplicated().any():
        duplicate_count = int(dataframe[ID_COLUMN].duplicated().sum())
        raise SentenceEmbeddingGenerationError(
            f"The {dataset_name} contains {duplicate_count} duplicated "
            f"{ID_COLUMN} values."
        )


def validate_training_alignment(
    original_training: pd.DataFrame,
    llm_training: pd.DataFrame,
) -> None:
    """Verify that original-label and LLM-label rows describe the same articles."""
    if len(original_training) != len(llm_training):
        raise SentenceEmbeddingGenerationError(
            "The original-label and LLM-label training datasets have different "
            "row counts."
        )

    original_ids = original_training[ID_COLUMN].reset_index(drop=True)
    llm_ids = llm_training[ID_COLUMN].reset_index(drop=True)

    if not original_ids.equals(llm_ids):
        raise SentenceEmbeddingGenerationError(
            "The original-label and LLM-label training datasets are not in the "
            "same row order."
        )

    for column in TEXT_COLUMNS:
        original_values = (
            original_training[column].fillna("").astype(str).reset_index(drop=True)
        )
        llm_values = (
            llm_training[column].fillna("").astype(str).reset_index(drop=True)
        )

        if not original_values.equals(llm_values):
            raise SentenceEmbeddingGenerationError(
                f"Training article content differs between label sources in "
                f"column '{column}'."
            )


def validate_split_membership(
    full_dataset: pd.DataFrame,
    training_dataset: pd.DataFrame,
    testing_dataset: pd.DataFrame,
) -> None:
    """Verify that the prepared split exactly partitions the original dataset."""
    full_ids = set(full_dataset[ID_COLUMN].tolist())
    training_ids = set(training_dataset[ID_COLUMN].tolist())
    testing_ids = set(testing_dataset[ID_COLUMN].tolist())

    overlap = training_ids & testing_ids
    if overlap:
        raise SentenceEmbeddingGenerationError(
            f"The train and test sets overlap on {len(overlap)} article IDs."
        )

    combined_ids = training_ids | testing_ids
    if combined_ids != full_ids:
        missing_from_split = full_ids - combined_ids
        unexpected_in_split = combined_ids - full_ids
        raise SentenceEmbeddingGenerationError(
            "The train/test split does not exactly partition the original "
            f"dataset. Missing IDs: {len(missing_from_split)}; unexpected IDs: "
            f"{len(unexpected_in_split)}."
        )


# ---------------------------------------------------------------------------
# Enhanced text processing adapted from the project notebook
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """Normalize Unicode, markup, URLs, and repeated whitespace."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_headline(value: Any) -> str:
    """Normalize a headline and remove a trailing CNN attribution."""
    text = normalize_text(value)
    return re.sub(
        r"\s*-\s*CNN\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def clean_article_text(value: Any) -> str:
    """Normalize article text and remove a leading CNN dateline when present."""
    text = normalize_text(value)
    text = re.sub(
        r"^[A-Z][A-Za-z .'-]+\s*\(CNN\)\s*[-—:]?\s*",
        "",
        text,
    )
    return text.strip()


def create_enhanced_text(dataframe: pd.DataFrame) -> pd.Series:
    """Combine cleaned headline, description, and article text."""
    headline = dataframe["Headline"].apply(clean_headline)
    description = dataframe["Description"].apply(normalize_text)
    article = dataframe["Article text"].apply(clean_article_text)

    components = pd.DataFrame(
        {
            "headline": headline,
            "description": description,
            "article": article,
        },
        index=dataframe.index,
    )

    enhanced_text = components.apply(
        lambda row: ". ".join(value for value in row if value),
        axis=1,
    )

    if (enhanced_text.str.len() == 0).any():
        empty_count = int((enhanced_text.str.len() == 0).sum())
        raise SentenceEmbeddingGenerationError(
            f"Enhanced text construction produced {empty_count} empty articles."
        )

    return enhanced_text


# ---------------------------------------------------------------------------
# Reuse and integrity helpers
# ---------------------------------------------------------------------------

def hash_texts(texts: pd.Series) -> str:
    """Create a stable SHA-256 digest for an ordered text collection."""
    digest = hashlib.sha256()
    for text in texts.astype(str):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def expected_output_paths() -> tuple[Path, ...]:
    """Return every artifact required for a complete embedding stage."""
    return (
        TRAIN_EMBEDDINGS_PATH,
        TEST_EMBEDDINGS_PATH,
        FULL_EMBEDDINGS_PATH,
        TRAIN_METADATA_PATH,
        TEST_METADATA_PATH,
        FULL_METADATA_PATH,
        MANIFEST_PATH,
    )


def can_reuse_existing_outputs(
    settings: EmbeddingSettings,
    train_text_hash: str,
    test_text_hash: str,
    full_text_hash: str,
) -> bool:
    """Return whether complete existing outputs match the current inputs."""
    if settings.overwrite:
        return False

    if not all(path.exists() for path in expected_output_paths()):
        return False

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    expected = {
        "model_name": settings.model_name,
        "normalize_embeddings": settings.normalize_embeddings,
        "train_text_sha256": train_text_hash,
        "test_text_sha256": test_text_hash,
        "full_text_sha256": full_text_hash,
    }

    if any(manifest.get(key) != value for key, value in expected.items()):
        return False

    try:
        train_embeddings = np.load(TRAIN_EMBEDDINGS_PATH, mmap_mode="r")
        test_embeddings = np.load(TEST_EMBEDDINGS_PATH, mmap_mode="r")
        full_embeddings = np.load(FULL_EMBEDDINGS_PATH, mmap_mode="r")
    except (OSError, ValueError):
        return False

    return (
        train_embeddings.ndim == 2
        and test_embeddings.ndim == 2
        and full_embeddings.ndim == 2
        and train_embeddings.shape[0] == manifest.get("train_rows")
        and test_embeddings.shape[0] == manifest.get("test_rows")
        and full_embeddings.shape[0] == manifest.get("full_rows")
        and train_embeddings.shape[1] == manifest.get("embedding_dimension")
        and test_embeddings.shape[1] == manifest.get("embedding_dimension")
        and full_embeddings.shape[1] == manifest.get("embedding_dimension")
    )


# ---------------------------------------------------------------------------
# Sentence-BERT generation
# ---------------------------------------------------------------------------

def load_sentence_transformer(model_name: str, device: str | None) -> Any:
    """Load SentenceTransformer with a clear dependency error."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SentenceEmbeddingGenerationError(
            "The 'sentence-transformers' package is required. Install the "
            "project requirements before running this stage."
        ) from error

    logger.info("Loading Sentence-BERT model: %s", model_name)

    try:
        if device is None:
            return SentenceTransformer(model_name)
        return SentenceTransformer(model_name, device=device)
    except Exception as error:
        raise SentenceEmbeddingGenerationError(
            f"Could not load Sentence-BERT model '{model_name}': {error}"
        ) from error


def encode_texts(
    model: Any,
    texts: pd.Series,
    dataset_name: str,
    settings: EmbeddingSettings,
) -> np.ndarray:
    """Encode one ordered text collection and validate the returned matrix."""
    logger.info(
        "Generating %s embeddings for %d articles.",
        dataset_name,
        len(texts),
    )

    try:
        embeddings = model.encode(
            texts.tolist(),
            batch_size=settings.batch_size,
            show_progress_bar=settings.show_progress_bar,
            normalize_embeddings=settings.normalize_embeddings,
            convert_to_numpy=True,
        )
    except Exception as error:
        raise SentenceEmbeddingGenerationError(
            f"Could not generate {dataset_name} embeddings: {error}"
        ) from error

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise SentenceEmbeddingGenerationError(
            f"The {dataset_name} embeddings must be a two-dimensional matrix."
        )
    if embeddings.shape[0] != len(texts):
        raise SentenceEmbeddingGenerationError(
            f"The {dataset_name} embedding row count does not match the dataset."
        )
    if not np.isfinite(embeddings).all():
        raise SentenceEmbeddingGenerationError(
            f"The {dataset_name} embeddings contain non-finite values."
        )

    if settings.normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise SentenceEmbeddingGenerationError(
                f"The {dataset_name} embeddings were requested as normalized, "
                "but some row norms differ materially from 1.0."
            )

    logger.info(
        "%s embedding matrix shape: %s",
        dataset_name.capitalize(),
        embeddings.shape,
    )
    return embeddings


def create_metadata(
    dataframe: pd.DataFrame,
    enhanced_text: pd.Series,
) -> pd.DataFrame:
    """Create row-aligned metadata used by downstream semantic analyses."""
    metadata = dataframe.loc[
        :,
        [ID_COLUMN, CATEGORY_COLUMN, SECTION_COLUMN, "Headline"],
    ].copy()
    metadata.insert(1, "embedding_row", np.arange(len(metadata), dtype=int))
    metadata["enhanced_text_characters"] = enhanced_text.str.len().to_numpy()
    metadata["enhanced_text_words"] = (
        enhanced_text.str.split().str.len().to_numpy()
    )
    return metadata.reset_index(drop=True)


def save_numpy_array(array: np.ndarray, path: Path) -> None:
    """Save an array atomically to reduce the chance of partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("wb") as file:
            np.save(file, array, allow_pickle=False)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Save the embedding manifest as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_embedding_generation(config_path: Path = CONFIG_PATH) -> None:
    """Run the complete Sentence-BERT embedding-generation stage."""
    logger.info("Starting Sentence-BERT embedding generation.")

    config = load_config(config_path)
    settings = get_embedding_settings(config)
    dataset_config = config["dataset"]

    original_path = resolve_project_path(dataset_config["original"])
    train_path = resolve_project_path(dataset_config["train_file"])
    test_path = resolve_project_path(dataset_config["test_file"])
    llm_train_path = resolve_project_path(dataset_config["llm_train_file"])

    full_original = read_csv(original_path)
    train_original = read_csv(train_path)
    test_original = read_csv(test_path)
    train_llm = read_csv(llm_train_path)

    validate_dataset(full_original, "original dataset")
    validate_dataset(train_original, "original-label training dataset")
    validate_dataset(test_original, "original testing dataset")
    validate_dataset(train_llm, "LLM-label training dataset")

    validate_training_alignment(train_original, train_llm)
    validate_split_membership(full_original, train_original, test_original)

    logger.info("Applying notebook-compatible enhanced text preprocessing.")
    train_text = create_enhanced_text(train_original)
    test_text = create_enhanced_text(test_original)
    full_text = create_enhanced_text(full_original)

    train_text_hash = hash_texts(train_text)
    test_text_hash = hash_texts(test_text)
    full_text_hash = hash_texts(full_text)

    if can_reuse_existing_outputs(
        settings=settings,
        train_text_hash=train_text_hash,
        test_text_hash=test_text_hash,
        full_text_hash=full_text_hash,
    ):
        logger.info(
            "Existing Sentence-BERT artifacts match the current model and "
            "datasets. Skipping regeneration."
        )
        return

    EMBEDDING_DIRECTORY.mkdir(parents=True, exist_ok=True)

    model = load_sentence_transformer(
        model_name=settings.model_name,
        device=settings.device,
    )

    train_embeddings = encode_texts(
        model=model,
        texts=train_text,
        dataset_name="training",
        settings=settings,
    )
    test_embeddings = encode_texts(
        model=model,
        texts=test_text,
        dataset_name="testing",
        settings=settings,
    )
    full_embeddings = encode_texts(
        model=model,
        texts=full_text,
        dataset_name="full-dataset",
        settings=settings,
    )

    dimensions = {
        train_embeddings.shape[1],
        test_embeddings.shape[1],
        full_embeddings.shape[1],
    }
    if len(dimensions) != 1:
        raise SentenceEmbeddingGenerationError(
            "The generated embedding matrices have inconsistent dimensions."
        )

    train_metadata = create_metadata(train_original, train_text)
    test_metadata = create_metadata(test_original, test_text)
    full_metadata = create_metadata(full_original, full_text)

    save_numpy_array(train_embeddings, TRAIN_EMBEDDINGS_PATH)
    save_numpy_array(test_embeddings, TEST_EMBEDDINGS_PATH)
    save_numpy_array(full_embeddings, FULL_EMBEDDINGS_PATH)

    write_csv(train_metadata, TRAIN_METADATA_PATH)
    write_csv(test_metadata, TEST_METADATA_PATH)
    write_csv(full_metadata, FULL_METADATA_PATH)

    embedding_dimension = train_embeddings.shape[1]
    manifest: dict[str, Any] = {
        **asdict(settings),
        "resolved_device": str(getattr(model, "device", settings.device)),
        "embedding_dimension": int(embedding_dimension),
        "embedding_dtype": str(train_embeddings.dtype),
        "train_rows": int(len(train_original)),
        "test_rows": int(len(test_original)),
        "full_rows": int(len(full_original)),
        "train_text_sha256": train_text_hash,
        "test_text_sha256": test_text_hash,
        "full_text_sha256": full_text_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_fields": list(TEXT_COLUMNS),
        "text_preprocessing": "notebook_enhanced_text",
        "output_files": {
            "train_embeddings": str(TRAIN_EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)),
            "test_embeddings": str(TEST_EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)),
            "full_embeddings": str(FULL_EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)),
            "train_metadata": str(TRAIN_METADATA_PATH.relative_to(PROJECT_ROOT)),
            "test_metadata": str(TEST_METADATA_PATH.relative_to(PROJECT_ROOT)),
            "full_metadata": str(FULL_METADATA_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    save_manifest(manifest, MANIFEST_PATH)

    logger.info(
        "Saved Sentence-BERT embeddings with dimension %d to %s.",
        embedding_dimension,
        EMBEDDING_DIRECTORY,
    )
    logger.info("Sentence-BERT embedding generation completed successfully.")


def main() -> None:
    """Command-line entry point."""
    try:
        run_embedding_generation()
    except SentenceEmbeddingGenerationError:
        logger.exception("Sentence-BERT embedding generation failed.")
        raise
    except Exception:
        logger.exception(
            "Sentence-BERT embedding generation failed unexpectedly."
        )
        raise


if __name__ == "__main__":
    main()
