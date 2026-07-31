"""
run_pipeline.py

Run the complete CNN news classification pipeline in sequence.

The pipeline stops immediately if any stage fails and reports:
- the stage currently running,
- the runtime of each completed stage,
- the total pipeline runtime.

Usage:
    python run_pipeline.py

Optional arguments:
    python run_pipeline.py --start 04
    python run_pipeline.py --end 07
    python run_pipeline.py --start 08 --end 12
    python run_pipeline.py --skip 02 08
    python run_pipeline.py --list
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"


@dataclass(frozen=True)
class PipelineStage:
    """Definition of one pipeline stage."""

    identifier: str
    script_name: str
    description: str

    @property
    def script_path(self) -> Path:
        return SRC_DIRECTORY / self.script_name


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        identifier="01",
        script_name="01_prepare_dataset.py",
        description="Prepare and split the CNN dataset",
    ),
    PipelineStage(
        identifier="01b",
        script_name="01b_exploratory_data_analysis.py",
        description="Perform exploratory data analysis",
    ),
    PipelineStage(
        identifier="02",
        script_name="02_generate_llm_labels.py",
        description="Generate hierarchical LLM labels",
    ),
    PipelineStage(
        identifier="03",
        script_name="03_validate_dataset.py",
        description="Validate generated labels and dataset hierarchy",
    ),
    PipelineStage(
        identifier="03b",
        script_name="03b_lda_topic_analysis.py",
        description="Perform LDA topic analysis",
    ),
    PipelineStage(
        identifier="04",
        script_name="04_vectorize_dataset.py",
        description="Generate TF-IDF representations",
    ),
    PipelineStage(
        identifier="05",
        script_name="05_train_models.py",
        description="Train TF-IDF classification models",
    ),
    PipelineStage(
        identifier="06",
        script_name="06_evaluate_models.py",
        description="Evaluate TF-IDF classification models",
    ),
    PipelineStage(
        identifier="07",
        script_name="07_visualize_results.py",
        description="Generate TF-IDF result visualizations",
    ),
    PipelineStage(
        identifier="08",
        script_name="08_generate_sentence_embeddings.py",
        description="Generate Sentence-BERT embeddings",
    ),
    PipelineStage(
        identifier="09",
        script_name="09_semantic_neighbor_analysis.py",
        description="Perform semantic nearest-neighbour analysis",
    ),
    PipelineStage(
        identifier="10",
        script_name="10_train_embedding_models.py",
        description="Train Sentence-BERT embedding classifiers",
    ),
    PipelineStage(
        identifier="11",
        script_name="11_evaluate_embedding_models.py",
        description="Evaluate Sentence-BERT embedding classifiers",
    ),
    PipelineStage(
        identifier="12",
        script_name="12_compare_representations.py",
        description="Compare TF-IDF and Sentence-BERT representations",
    ),
)


def format_duration(seconds: float) -> str:
    """Format a duration as seconds or hours, minutes, and seconds."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    components: list[str] = []

    if hours:
        components.append(f"{hours} hour{'s' if hours != 1 else ''}")

    if minutes:
        components.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

    if seconds or not components:
        components.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return ", ".join(components)


def normalize_stage_identifier(value: str) -> str:
    """Normalize a stage identifier supplied on the command line."""
    normalized = value.strip().lower()

    if normalized.endswith(".py"):
        normalized = Path(normalized).stem

    for stage in PIPELINE_STAGES:
        if normalized == stage.identifier.lower():
            return stage.identifier

        if normalized == Path(stage.script_name).stem.lower():
            return stage.identifier

    valid_values = ", ".join(stage.identifier for stage in PIPELINE_STAGES)
    raise argparse.ArgumentTypeError(
        f"Unknown stage '{value}'. Valid stage identifiers: {valid_values}"
    )


def validate_pipeline_files(stages: Sequence[PipelineStage]) -> None:
    """Ensure that every selected pipeline script exists."""
    if not SRC_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SRC_DIRECTORY}"
        )

    missing_scripts = [
        stage.script_path
        for stage in stages
        if not stage.script_path.is_file()
    ]

    if missing_scripts:
        missing_text = "\n".join(
            f"  - {path}" for path in missing_scripts
        )
        raise FileNotFoundError(
            "The following pipeline scripts were not found:\n"
            f"{missing_text}"
        )


def select_stages(
    start_identifier: str | None,
    end_identifier: str | None,
    skipped_identifiers: set[str],
) -> list[PipelineStage]:
    """Select the requested inclusive stage range and remove skipped stages."""
    identifiers = [stage.identifier for stage in PIPELINE_STAGES]

    start_index = 0
    end_index = len(PIPELINE_STAGES) - 1

    if start_identifier is not None:
        start_index = identifiers.index(start_identifier)

    if end_identifier is not None:
        end_index = identifiers.index(end_identifier)

    if start_index > end_index:
        raise ValueError(
            f"Start stage '{start_identifier}' occurs after "
            f"end stage '{end_identifier}'."
        )

    selected = [
        stage
        for stage in PIPELINE_STAGES[start_index : end_index + 1]
        if stage.identifier not in skipped_identifiers
    ]

    if not selected:
        raise ValueError(
            "No pipeline stages remain after applying the selected range "
            "and skip arguments."
        )

    return selected


def print_pipeline_stages(stages: Sequence[PipelineStage]) -> None:
    """Display a pipeline stage list."""
    for stage in stages:
        print(
            f"{stage.identifier:>3}  "
            f"{stage.script_name:<42} "
            f"{stage.description}"
        )


def run_stage(
    stage: PipelineStage,
    stage_number: int,
    total_stages: int,
) -> float:
    """Run one stage as an isolated Python process."""
    separator = "=" * 79

    print()
    print(separator)
    print(
        f"Stage {stage_number} of {total_stages}: "
        f"{stage.identifier} - {stage.description}"
    )
    print(f"Script: {stage.script_name}")
    print(separator)
    print()

    start_time = time.perf_counter()

    try:
        subprocess.run(
            [sys.executable, str(stage.script_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except KeyboardInterrupt:
        raise
    except subprocess.CalledProcessError as error:
        elapsed = time.perf_counter() - start_time
        raise RuntimeError(
            f"Stage {stage.identifier} failed with exit code "
            f"{error.returncode} after {format_duration(elapsed)}."
        ) from error
    except OSError as error:
        elapsed = time.perf_counter() - start_time
        raise RuntimeError(
            f"Could not start stage {stage.identifier} after "
            f"{format_duration(elapsed)}: {error}"
        ) from error

    elapsed = time.perf_counter() - start_time

    print()
    print(
        f"Completed stage {stage.identifier} in "
        f"{format_duration(elapsed)}."
    )

    return elapsed


def run_pipeline(stages: Sequence[PipelineStage]) -> int:
    """Run all selected stages and return a process exit code."""
    validate_pipeline_files(stages)

    total_start_time = time.perf_counter()
    completed_stages: list[tuple[PipelineStage, float]] = []

    print()
    print("CNN News Classification Pipeline")
    print("=" * 79)
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")
    print(f"Selected stages: {len(stages)}")
    print()
    print_pipeline_stages(stages)

    try:
        for stage_number, stage in enumerate(stages, start=1):
            elapsed = run_stage(
                stage=stage,
                stage_number=stage_number,
                total_stages=len(stages),
            )
            completed_stages.append((stage, elapsed))

    except KeyboardInterrupt:
        total_elapsed = time.perf_counter() - total_start_time

        print()
        print("=" * 79)
        print("Pipeline interrupted by the user.")
        print(f"Elapsed time: {format_duration(total_elapsed)}")
        print("=" * 79)

        return 130

    except Exception as error:
        total_elapsed = time.perf_counter() - total_start_time

        print()
        print("=" * 79)
        print("PIPELINE FAILED")
        print("=" * 79)
        print(str(error))
        print()
        print(
            f"Completed stages: "
            f"{len(completed_stages)} of {len(stages)}"
        )
        print(f"Total elapsed time: {format_duration(total_elapsed)}")

        if completed_stages:
            print()
            print("Completed stage runtimes:")
            for completed_stage, elapsed in completed_stages:
                print(
                    f"  {completed_stage.identifier:>3}  "
                    f"{completed_stage.description:<55} "
                    f"{format_duration(elapsed)}"
                )

        print("=" * 79)
        return 1

    total_elapsed = time.perf_counter() - total_start_time

    print()
    print("=" * 79)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 79)

    for completed_stage, elapsed in completed_stages:
        print(
            f"  {completed_stage.identifier:>3}  "
            f"{completed_stage.description:<55} "
            f"{format_duration(elapsed)}"
        )

    print("-" * 79)
    print(f"Total runtime: {format_duration(total_elapsed)}")
    print("=" * 79)

    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the CNN news classification pipeline or a selected "
            "inclusive range of stages."
        )
    )

    parser.add_argument(
        "--start",
        type=normalize_stage_identifier,
        default=None,
        metavar="STAGE",
        help=(
            "First stage to run. Examples: 04, 08, "
            "08_generate_sentence_embeddings."
        ),
    )

    parser.add_argument(
        "--end",
        type=normalize_stage_identifier,
        default=None,
        metavar="STAGE",
        help=(
            "Last stage to run, inclusive. Examples: 07, 12, "
            "12_compare_representations."
        ),
    )

    parser.add_argument(
        "--skip",
        type=normalize_stage_identifier,
        nargs="*",
        default=[],
        metavar="STAGE",
        help=(
            "One or more stages to skip. Example: --skip 01b 03b 09"
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all pipeline stages and exit.",
    )

    return parser


def main() -> int:
    """Parse command-line arguments and run the pipeline."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.list:
        print_pipeline_stages(PIPELINE_STAGES)
        return 0

    try:
        selected_stages = select_stages(
            start_identifier=arguments.start,
            end_identifier=arguments.end,
            skipped_identifiers=set(arguments.skip),
        )
    except ValueError as error:
        parser.error(str(error))

    return run_pipeline(selected_stages)


if __name__ == "__main__":
    raise SystemExit(main())