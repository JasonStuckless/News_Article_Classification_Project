#!/usr/bin/env python3
"""
Run only the newly added analysis and Sentence-BERT pipeline stages.

This script intentionally skips the original LLM-label-generation workflow,
including:

    01_prepare_dataset.py
    02_generate_llm_labels.py
    03_validate_dataset.py
    04_vectorize_dataset.py
    05_train_models.py
    06_evaluate_models.py
    07_visualize_results.py

It runs only:

    01b_exploratory_data_analysis.py
    03b_lda_topic_analysis.py
    08_generate_sentence_embeddings.py
    09_semantic_neighbor_analysis.py
    10_train_embedding_models.py
    11_evaluate_embedding_models.py
    12_compare_representations.py

The existing outputs from the original pipeline must already be present,
especially the prepared train/test datasets, LLM-labelled training dataset,
and TF-IDF evaluation results used by the final comparison stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Stage:
    """A pipeline stage and its corresponding script."""

    stage_id: str
    script_name: str
    description: str


STAGES: tuple[Stage, ...] = (
    Stage(
        "01b",
        "01b_exploratory_data_analysis.py",
        "Run exploratory data analysis",
    ),
    Stage(
        "03b",
        "03b_lda_topic_analysis.py",
        "Run LDA topic analysis",
    ),
    Stage(
        "08",
        "08_generate_sentence_embeddings.py",
        "Generate Sentence-BERT embeddings",
    ),
    Stage(
        "09",
        "09_semantic_neighbor_analysis.py",
        "Run semantic-neighbour analysis",
    ),
    Stage(
        "10",
        "10_train_embedding_models.py",
        "Train embedding-based classifiers",
    ),
    Stage(
        "11",
        "11_evaluate_embedding_models.py",
        "Evaluate embedding-based classifiers",
    ),
    Stage(
        "12",
        "12_compare_representations.py",
        "Compare TF-IDF and Sentence-BERT results",
    ),
)


def parse_stage_set(values: Iterable[str] | None) -> set[str]:
    """Convert repeated/comma-separated stage arguments into a stage-ID set."""

    if not values:
        return set()

    parsed: set[str] = set()

    for value in values:
        for item in value.split(","):
            item = item.strip()

            if item:
                parsed.add(item)

    return parsed


def validate_stage_ids(stage_ids: set[str]) -> None:
    """Reject unknown stage identifiers."""

    valid_ids = {stage.stage_id for stage in STAGES}
    invalid_ids = sorted(stage_ids - valid_ids)

    if invalid_ids:
        valid_text = ", ".join(stage.stage_id for stage in STAGES)
        invalid_text = ", ".join(invalid_ids)

        raise ValueError(
            f"Unknown stage ID(s): {invalid_text}. "
            f"Valid new-stage IDs are: {valid_text}"
        )


def select_stages(
    start_stage: str | None,
    end_stage: str | None,
    skipped_stages: set[str],
) -> list[Stage]:
    """Select an inclusive stage range and remove explicitly skipped stages."""

    stage_ids = [stage.stage_id for stage in STAGES]

    if start_stage is not None and start_stage not in stage_ids:
        raise ValueError(f"Unknown start stage: {start_stage}")

    if end_stage is not None and end_stage not in stage_ids:
        raise ValueError(f"Unknown end stage: {end_stage}")

    start_index = stage_ids.index(start_stage) if start_stage else 0
    end_index = stage_ids.index(end_stage) if end_stage else len(STAGES) - 1

    if start_index > end_index:
        raise ValueError(
            f"Start stage {start_stage} occurs after end stage {end_stage}."
        )

    return [
        stage
        for stage in STAGES[start_index : end_index + 1]
        if stage.stage_id not in skipped_stages
    ]


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""

    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_stage(stage: Stage, src_directory: Path) -> float:
    """Run one stage in an isolated Python process."""

    script_path = src_directory / stage.script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required script was not found: {script_path}"
        )

    print()
    print("=" * 79)
    print(f"Stage {stage.stage_id}: {stage.description}")
    print(f"Script: {script_path.name}")
    print("=" * 79)

    started_at = time.perf_counter()

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=src_directory.parent,
        check=False,
    )

    elapsed = time.perf_counter() - started_at

    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage {stage.stage_id} failed with exit code "
            f"{completed.returncode} after {format_duration(elapsed)}."
        )

    print(
        f"Completed stage {stage.stage_id} in "
        f"{format_duration(elapsed)}."
    )

    return elapsed


def print_stage_list() -> None:
    """Print the stages managed by this script."""

    print("New pipeline stages:")
    print()

    for stage in STAGES:
        print(
            f"  {stage.stage_id:<3} "
            f"{stage.script_name:<43} "
            f"{stage.description}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run only the newly added EDA, LDA, Sentence-BERT, semantic "
            "analysis, embedding-model, and representation-comparison stages."
        )
    )

    parser.add_argument(
        "--start",
        choices=[stage.stage_id for stage in STAGES],
        help="Begin with this stage.",
    )

    parser.add_argument(
        "--end",
        choices=[stage.stage_id for stage in STAGES],
        help="Stop after this stage.",
    )

    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="STAGE",
        help=(
            "Skip one or more stages. This option may be repeated, or stage "
            "IDs may be comma-separated."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List the stages and exit.",
    )

    return parser


def main() -> int:
    """Run the selected new pipeline stages."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.list:
        print_stage_list()
        return 0

    try:
        skipped_stages = parse_stage_set(arguments.skip)
        validate_stage_ids(skipped_stages)

        selected_stages = select_stages(
            start_stage=arguments.start,
            end_stage=arguments.end,
            skipped_stages=skipped_stages,
        )
    except ValueError as error:
        parser.error(str(error))

    if not selected_stages:
        print("No stages were selected.")
        return 0

    current_file = Path(__file__).resolve()
    src_directory = current_file.parent
    project_root = src_directory.parent

    print("New-stage pipeline")
    print(f"Project root: {project_root}")
    print(
        "Selected stages: "
        + ", ".join(stage.stage_id for stage in selected_stages)
    )
    print(
        "The original LLM-label-generation stage will not be run."
    )

    pipeline_started_at = time.perf_counter()
    completed_stages: list[tuple[Stage, float]] = []

    try:
        for stage in selected_stages:
            elapsed = run_stage(stage, src_directory)
            completed_stages.append((stage, elapsed))

    except (FileNotFoundError, RuntimeError) as error:
        total_elapsed = time.perf_counter() - pipeline_started_at

        print()
        print("Pipeline stopped.")
        print(str(error))
        print(f"Elapsed time: {format_duration(total_elapsed)}")

        return 1

    total_elapsed = time.perf_counter() - pipeline_started_at

    print()
    print("=" * 79)
    print("New-stage pipeline completed successfully")
    print("=" * 79)

    for stage, elapsed in completed_stages:
        print(
            f"Stage {stage.stage_id:<3} "
            f"{format_duration(elapsed)}  "
            f"{stage.description}"
        )

    print("-" * 79)
    print(f"Total runtime: {format_duration(total_elapsed)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
