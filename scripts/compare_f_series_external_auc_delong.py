#!/usr/bin/env python3
"""Compare all 12 F-series Youyi external-validation AUCs with DeLong tests.

Example (run from the ml-cvnets repository):

    python3 scripts/compare_f_series_external_auc_delong.py \
        --model-dir F1=/path/to/results/F1/external_run \
        --model-dir F2a=/path/to/results/F2a/external_run \
        ... \
        --output-dir /path/to/delong_output

Each --model-dir must identify one specific external-evaluation run containing
external_eval_predictions.csv directly. The predictions must come from model
checkpoints fixed before Youyi evaluation.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from scripts.compare_validation_auc_delong import (
        capture_input_hashes,
        write_results,
    )
    from scripts.statistical_tests.validation_auc import ModelInput, align_predictions
except ModuleNotFoundError:
    from compare_validation_auc_delong import capture_input_hashes, write_results
    from statistical_tests.validation_auc import ModelInput, align_predictions


MODEL_NAMES: tuple[str, ...] = (
    "F1",
    "F2a",
    "F2b",
    "F3_esophagus_liver",
    "F3_esophagus_liver_portal",
    "F3_esophagus_liver_spleen",
    "F3_esophagus_liver_spleen_portal",
    "F3_esophagus_portal_venous_system",
    "F3_esophagus_spleen",
    "F3_esophagus_spleen_portal_venous_system",
    "F4",
    "F5",
)
PREDICTIONS_FILENAME = "external_eval_predictions.csv"
EXTERNAL_STAGE = "youyi_external"


def parse_model_directories(values: Sequence[str]) -> dict[str, Path]:
    """Require one explicitly provided external-run directory per model."""
    directories: dict[str, Path] = {}
    for value in values:
        name, separator, directory_text = value.partition("=")
        name = name.strip()
        directory_text = directory_text.strip()
        if not separator or not name or not directory_text:
            raise ValueError(f"--model-dir must use NAME=DIR: {value!r}")
        if name not in MODEL_NAMES:
            raise ValueError(f"unknown F-series model: {name!r}")
        if name in directories:
            raise ValueError(f"duplicate --model-dir for {name}")
        directories[name] = Path(directory_text).expanduser().resolve()

    missing = [name for name in MODEL_NAMES if name not in directories]
    if missing:
        raise ValueError(f"missing --model-dir for: {', '.join(missing)}")
    return directories


def resolve_external_run_directory(name: str, directory: Path) -> Path:
    """Require an exact external run containing the case-level predictions."""
    if not directory.is_dir():
        raise FileNotFoundError(f"directory for {name} not found: {directory}")
    predictions_path = directory / PREDICTIONS_FILENAME
    if predictions_path.is_file():
        return directory
    raise FileNotFoundError(
        f"--model-dir for {name} must point to a specific external-evaluation "
        f"run, not its parent directory: {directory}; missing directly: "
        f"{PREDICTIONS_FILENAME}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Specific external-evaluation run directory; repeat for all 12 models.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify paired Youyi cases and write DeLong result tables."""
    args = build_parser().parse_args(argv)
    try:
        directories = parse_model_directories(args.model_dir)
        model_inputs = []
        for name in MODEL_NAMES:
            run_directory = resolve_external_run_directory(name, directories[name])
            model_inputs.append(
                ModelInput(
                    name=name,
                    predictions_path=run_directory / PREDICTIONS_FILENAME,
                    metrics_path=None,
                    epoch=None,
                    epoch_selection="external_fixed_checkpoint",
                )
            )

        input_hashes = capture_input_hashes(model_inputs)
        case_ids, targets, scores = align_predictions(
            model_inputs=model_inputs,
            stage=EXTERNAL_STAGE,
        )
        write_results(
            output_dir=args.output_dir.expanduser().resolve(),
            model_inputs=model_inputs,
            stage=EXTERNAL_STAGE,
            case_ids=case_ids,
            targets=targets,
            scores=scores,
            alpha=0.05,
            input_hashes=input_hashes,
            analysis_set="external",
            entrypoint_path=Path(__file__),
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    sys.stdout.write(
        f"F-series external DeLong analysis written to: {args.output_dir}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
