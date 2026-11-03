#!/usr/bin/env python3
"""Pairwise DeLong tests for all G1-G3 models, internal and external.

Example (run from the ml-cvnets repository on the server):

    python3 scripts/compare_g_series_auc_delong.py \
        --results-root /home/mw/work/results \
        --output-dir /home/mw/work/results/G_series_delong

Writes ``internal_validation/`` (Tongji validation predictions at each model's
AUC-best epoch) and ``external_validation/`` (Youyi predictions of the same
checkpoints). Each holds ``model_auc.csv``, ``pairwise_delong.csv`` (all 45
pairs, Holm-adjusted within that set), ``analysis_metadata.json`` and
``README.md``. Run directories follow ``summarize_g_series_results.py``.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from scripts.compare_f_series_validation_auc_delong import select_auc_best_epoch
    from scripts.compare_validation_auc_delong import (
        capture_input_hashes,
        write_results,
    )
    from scripts.statistical_tests.delong import delong_auc_covariance
    from scripts.statistical_tests.validation_auc import ModelInput, align_predictions
    from scripts.summarize_g_series_results import (
        EXTERNAL_STAGE,
        EXTERNAL_SUFFIX,
        G_SERIES_RUNS,
        VALIDATION_STAGE,
        top_auc_checkpoint,
    )
except ModuleNotFoundError:
    from compare_f_series_validation_auc_delong import select_auc_best_epoch
    from compare_validation_auc_delong import capture_input_hashes, write_results
    from statistical_tests.delong import delong_auc_covariance
    from statistical_tests.validation_auc import ModelInput, align_predictions
    from summarize_g_series_results import (
        EXTERNAL_STAGE,
        EXTERNAL_SUFFIX,
        G_SERIES_RUNS,
        VALIDATION_STAGE,
        top_auc_checkpoint,
    )


def build_inputs(results_root: Path) -> tuple[list[ModelInput], list[ModelInput]]:
    """Resolve the AUC-best epoch per model and both prediction sources."""
    internal, external = [], []
    for name, subdir, label in G_SERIES_RUNS:
        run_dir = results_root / subdir / label
        external_dir = results_root / subdir / f"{label}{EXTERNAL_SUFFIX}"
        metrics_path = run_dir / "validation_confusion_matrix.jsonl"
        validation_path = run_dir / "validation_predictions.csv"
        external_path = external_dir / "external_eval_predictions.csv"
        for path in (metrics_path, validation_path, external_path):
            if not path.is_file():
                raise FileNotFoundError(f"{name}: file not found: {path}")

        epoch, _, digest = select_auc_best_epoch(metrics_path)
        checkpoint_epoch, checkpoint_path = top_auc_checkpoint(run_dir)
        if checkpoint_epoch != epoch:
            raise ValueError(
                f"{name}: AUC-best epoch {epoch} differs from the top AUC "
                f"checkpoint {checkpoint_path.name}"
            )
        internal.append(
            ModelInput(
                name=name,
                predictions_path=validation_path,
                metrics_path=metrics_path,
                epoch=epoch,
                epoch_selection="max_recorded_validation_auc_earliest_tie",
                metrics_sha256_at_selection=digest,
            )
        )
        external.append(
            ModelInput(
                name=name,
                predictions_path=external_path,
                metrics_path=None,
                epoch=None,
                epoch_selection=f"checkpoint_auc_best.pt (epoch {epoch})",
            )
        )
    return internal, external


def check_recorded_aucs(model_inputs: Sequence[ModelInput], scores_aucs) -> None:
    """Recomputed validation AUCs must equal the logged AUC of each epoch."""
    for item, calculated in zip(model_inputs, scores_aucs):
        _, recorded, _ = select_auc_best_epoch(item.metrics_path)
        if not math.isclose(recorded, float(calculated), abs_tol=1e-8):
            raise ValueError(
                f"{item.name}: recorded AUC {recorded} differs from recomputed "
                f"{calculated} at epoch {item.epoch}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Must not already exist"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        internal, external = build_inputs(results_root)
        for analysis_set, stage, model_inputs, subdir in (
            ("validation", VALIDATION_STAGE, internal, "internal_validation"),
            ("external", EXTERNAL_STAGE, external, "external_validation"),
        ):
            input_hashes = capture_input_hashes(model_inputs)
            case_ids, targets, scores = align_predictions(model_inputs, stage)
            if analysis_set == "validation":
                aucs, _ = delong_auc_covariance(targets, scores)
                check_recorded_aucs(model_inputs, aucs)
            write_results(
                output_dir=output_dir / subdir,
                model_inputs=model_inputs,
                stage=stage,
                case_ids=case_ids,
                targets=targets,
                scores=scores,
                alpha=0.05,
                input_hashes=input_hashes,
                analysis_set=analysis_set,
                entrypoint_path=Path(__file__),
            )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    for subdir in ("internal_validation", "external_validation"):
        sys.stdout.write((output_dir / subdir / "README.md").read_text("utf-8"))
        sys.stdout.write("\n")
    sys.stdout.write(f"G-series DeLong analysis written to: {output_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
