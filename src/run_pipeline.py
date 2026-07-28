"""
run_pipeline.py

Execute the complete news article classification pipeline.

The pipeline can begin at any stage by changing START_STEP.

Stages:
1. Prepare dataset
2. Generate LLM labels
3. Validate datasets
4. Vectorize text
5. Train models
6. Evaluate models
7. Generate figures
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


SRC_DIRECTORY = Path(__file__).resolve().parent

PIPELINE = [
    ("Prepare Dataset", "01_prepare_dataset.py"),
    ("Generate LLM Labels", "02_generate_llm_labels.py"),
    ("Validate Dataset", "03_validate_dataset.py"),
    ("Vectorize Dataset", "04_vectorize_dataset.py"),
    ("Train Models", "05_train_models.py"),
    ("Evaluate Models", "06_evaluate_models.py"),
    ("Generate Figures", "07_generate_figures.py"),
]

# Change this value to choose where the pipeline begins.
# Valid values are 1 through 7.

START_STEP = 2


def validate_start_step(start_step: int) -> None:
    """Validate the requested starting stage."""
    if not isinstance(start_step, int):
        raise TypeError("START_STEP must be an integer.")

    if not 1 <= start_step <= len(PIPELINE):
        raise ValueError(
            f"START_STEP must be between 1 and {len(PIPELINE)}."
        )


def main() -> None:
    """Run the pipeline beginning at START_STEP."""
    validate_start_step(START_STEP)

    start_time = time.perf_counter()

    selected_pipeline = PIPELINE[START_STEP - 1:]

    print("=" * 70)
    print("News Article Classification Pipeline")
    print(f"Starting at step {START_STEP} of {len(PIPELINE)}")
    print("=" * 70)

    for stage_number, (stage_name, script_name) in enumerate(
        selected_pipeline,
        start=START_STEP,
    ):
        script_path = SRC_DIRECTORY / script_name

        print()
        print(
            f"[{stage_number}/{len(PIPELINE)}] "
            f"{stage_name}"
        )
        print("-" * 70)

        if not script_path.exists():
            raise FileNotFoundError(
                f"Script not found:\n{script_path}"
            )

        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
        )

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 70)
    print("Pipeline completed successfully.")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()

    except subprocess.CalledProcessError as error:
        print()
        print("=" * 70)
        print("Pipeline failed.")
        print(f"Exit code: {error.returncode}")
        print("=" * 70)
        raise SystemExit(error.returncode)

    except KeyboardInterrupt:
        print()
        print("Pipeline cancelled by user.")
        raise SystemExit(130)

    except Exception as error:
        print()
        print("=" * 70)
        print("Unexpected error:")
        print(error)
        print("=" * 70)
        raise SystemExit(1)