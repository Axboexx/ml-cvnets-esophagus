#!/usr/bin/env python3
"""Compare all 12 F-series internal-validation AUCs with paired DeLong tests.

Example (run from the ml-cvnets repository):

    python3 scripts/compare_f_series_validation_auc_delong.py \
        --model-dir F1=/path/to/results/F1/run_label \
        --model-dir F2a=/path/to/results/F2a/run_label \
        ... \
        --output-dir /path/to/delong_output

Each --model-dir must identify one specific training run containing both
validation_predictions.csv and validation_confusion_matrix.jsonl directly.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

try:
    from scripts.compare_validation_auc_delong import (
        capture_input_hashes,
        write_results,
    )
    from scripts.statistical_tests.delong import delong_auc_covariance
    from scripts.statistical_tests.validation_auc import ModelInput, align_predictions
except ModuleNotFoundError:
    from compare_validation_auc_delong import capture_input_hashes, write_results
    from statistical_tests.delong import delong_auc_covariance
    from statistical_tests.validation_auc import ModelInput, align_predictions


MODEL_NAMES: Tuple[str, ...] = (
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
PREDICTIONS_FILENAME = "validation_predictions.csv"
METRICS_FILENAME = "validation_confusion_matrix.jsonl"


def parse_model_directories(values: Sequence[str]) -> Dict[str, Path]:
    """Require one explicitly provided directory for every F-series model."""
    directories: Dict[str, Path] = {}
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


def resolve_run_directory(name: str, directory: Path) -> Path:
    """Require the exact directory of one training run."""
    if not directory.is_dir():
        raise FileNotFoundError(f"directory for {name} not found: {directory}")
    required = (PREDICTIONS_FILENAME, METRICS_FILENAME)
    missing = [filename for filename in required if not (directory / filename).is_file()]
    if not missing:
        return directory
    raise FileNotFoundError(
        f"--model-dir for {name} must point to a specific training run, not its "
        f"parent directory: {directory}; missing directly: {', '.join(missing)}"
    )


def select_auc_best_epoch(metrics_path: Path) -> Tuple[int, float, str]:
    """Select highest recorded validation AUC, breaking ties by earlier epoch."""
    contents = metrics_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    try:
        lines = contents.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"metrics file is not UTF-8: {metrics_path}") from error

    candidates: Dict[int, float] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {metrics_path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {metrics_path}:{line_number}")
        if row.get("stage") != "validation":
            continue
        epoch = row.get("epoch")
        auc = row.get("auc")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"invalid epoch at {metrics_path}:{line_number}")
        if isinstance(auc, bool) or not isinstance(auc, (int, float)):
            raise ValueError(f"missing/invalid AUC at {metrics_path}:{line_number}")
        auc = float(auc)
        if not math.isfinite(auc) or not 0.0 <= auc <= 1.0:
            raise ValueError(f"AUC outside [0, 1] at {metrics_path}:{line_number}")
        if epoch in candidates:
            raise ValueError(f"duplicate validation epoch {epoch} in {metrics_path}")
        candidates[epoch] = auc

    if not candidates:
        raise ValueError(f"no validation AUC records in {metrics_path}")
    best_epoch = min(candidates, key=lambda epoch: (-candidates[epoch], epoch))
    return best_epoch, candidates[best_epoch], digest


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Specific training-run directory; repeat for all 12 models.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Select epochs, verify paired cases, and write DeLong result tables."""
    args = build_parser().parse_args(argv)
    try:
        directories = parse_model_directories(args.model_dir)
        model_inputs = []
        recorded_aucs = []
        for name in MODEL_NAMES:
            run_directory = resolve_run_directory(name, directories[name])
            metrics_path = run_directory / METRICS_FILENAME
            epoch, auc, digest = select_auc_best_epoch(metrics_path)
            model_inputs.append(
                ModelInput(
                    name=name,
                    predictions_path=run_directory / PREDICTIONS_FILENAME,
                    metrics_path=metrics_path,
                    epoch=epoch,
                    epoch_selection="max_recorded_validation_auc_earliest_tie",
                    metrics_sha256_at_selection=digest,
                )
            )
            recorded_aucs.append(auc)

        input_hashes = capture_input_hashes(model_inputs)
        case_ids, targets, scores = align_predictions(model_inputs, "validation")
        calculated_aucs, _ = delong_auc_covariance(targets, scores)
        for model_input, recorded_auc, calculated_auc in zip(
            model_inputs, recorded_aucs, calculated_aucs
        ):
            if not math.isclose(recorded_auc, float(calculated_auc), abs_tol=1e-8):
                raise ValueError(
                    f"recorded AUC does not match selected predictions for "
                    f"{model_input.name} at epoch {model_input.epoch}: "
                    f"recorded={recorded_auc}, calculated={calculated_auc}"
                )

        write_results(
            output_dir=args.output_dir.expanduser().resolve(),
            model_inputs=model_inputs,
            stage="validation",
            case_ids=case_ids,
            targets=targets,
            scores=scores,
            alpha=0.05,
            input_hashes=input_hashes,
            entrypoint_path=Path(__file__),
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    sys.stdout.write(f"F-series DeLong analysis written to: {args.output_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
