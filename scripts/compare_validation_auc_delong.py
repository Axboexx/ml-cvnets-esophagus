#!/usr/bin/env python3
"""Compare validation or external ROC AUCs with paired DeLong tests."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from scripts.statistical_tests.delong import (
        delong_auc_covariance,
        delong_estimates,
        pairwise_delong_tests,
    )
    from scripts.statistical_tests.validation_auc import (
        ModelInput,
        align_predictions,
        holm_adjust,
        resolve_model_inputs,
    )
except ModuleNotFoundError:
    from statistical_tests.delong import (
        delong_auc_covariance,
        delong_estimates,
        pairwise_delong_tests,
    )
    from statistical_tests.validation_auc import (
        ModelInput,
        align_predictions,
        holm_adjust,
        resolve_model_inputs,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_input_hashes(model_inputs: Sequence[ModelInput]) -> Dict[Path, str]:
    """Hash every input before parsing so later mutations can be detected."""
    paths = []
    for item in model_inputs:
        paths.append(item.predictions_path)
        if item.metrics_path is not None:
            paths.append(item.metrics_path)
    hashes = {path: _sha256(path) for path in paths}
    for item in model_inputs:
        if (
            item.metrics_path is not None
            and item.metrics_sha256_at_selection is not None
        ):
            hashes[item.metrics_path] = item.metrics_sha256_at_selection
    return hashes


def verify_input_hashes(input_hashes: Mapping[Path, str]) -> None:
    """Fail if an input changed while the analysis was running."""
    changed = [
        str(path) for path, digest in input_hashes.items() if _sha256(path) != digest
    ]
    if changed:
        raise ValueError(f"input files changed during analysis: {changed}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_summary(
    model_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    analysis_set: str,
) -> str:
    title = "Validation" if analysis_set == "validation" else "External validation"
    if analysis_set == "validation":
        interpretation = (
            "These are internal validation comparisons after validation-based "
            "checkpoint selection and should be interpreted as exploratory."
        )
    else:
        interpretation = (
            "These are external validation comparisons of checkpoints fixed before "
            "external evaluation and quantify independent generalization evidence."
        )
    lines = [
        f"# {title} AUC paired DeLong analysis",
        "",
        "## Model AUC",
        "",
        "| Model | Epoch | AUC (95% CI) |",
        "|---|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['epoch']} | {float(row['auc']):.4f} "
            f"({float(row['ci_95_lower']):.4f}–"
            f"{float(row['ci_95_upper']):.4f}) |"
        )
    lines.extend(
        [
            "",
            "## Pairwise paired DeLong tests",
            "",
            "| Comparison | AUC difference | Raw P | Holm P |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['model_a']} − {row['model_b']} | "
            f"{float(row['auc_difference_a_minus_b']):.4f} | "
            f"{float(row['p_value_two_sided']):.6g} | "
            f"{float(row['p_value_holm']):.6g} |"
        )
    lines.extend(
        [
            "",
            "P values are two-sided paired DeLong tests and are Holm-adjusted "
            "across all listed comparisons.",
            "",
            interpretation,
            "",
        ]
    )
    return "\n".join(lines)


def write_results(
    output_dir: Path,
    model_inputs: Sequence[ModelInput],
    stage: str,
    case_ids: Sequence[str],
    targets: NDArray[np.int64],
    scores: NDArray[np.float64],
    alpha: float,
    input_hashes: Mapping[Path, str],
    analysis_set: str = "validation",
) -> None:
    """Compute results in a temporary directory, then atomically publish them."""
    if alpha != 0.05:
        raise ValueError("only alpha=0.05 is currently supported")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    aucs, covariance = delong_auc_covariance(targets=targets, scores=scores)
    estimates = delong_estimates(targets=targets, scores=scores)
    comparisons = pairwise_delong_tests(targets=targets, scores=scores)
    adjusted_p_values = holm_adjust([item.p_value for item in comparisons])
    model_rows = [
        {
            "model": item.name,
            "epoch": item.epoch,
            "epoch_selection": item.epoch_selection,
            "stage": stage,
            "n_cases": len(case_ids),
            "n_positive": int(targets.sum()),
            "n_negative": int(len(targets) - targets.sum()),
            "auc": estimate.auc,
            "standard_error": estimate.standard_error,
            "ci_95_lower": estimate.ci_lower,
            "ci_95_upper": estimate.ci_upper,
        }
        for item, estimate in zip(model_inputs, estimates)
    ]
    pair_rows = []
    normal_95 = 1.959963984540054
    for result, adjusted_p_value in zip(comparisons, adjusted_p_values):
        first = model_inputs[result.first_index]
        second = model_inputs[result.second_index]
        pair_rows.append(
            {
                "model_a": first.name,
                "model_b": second.name,
                "auc_a": estimates[result.first_index].auc,
                "auc_b": estimates[result.second_index].auc,
                "auc_difference_a_minus_b": result.auc_difference,
                "standard_error": result.standard_error,
                "difference_ci_95_lower": (
                    result.auc_difference - normal_95 * result.standard_error
                ),
                "difference_ci_95_upper": (
                    result.auc_difference + normal_95 * result.standard_error
                ),
                "z_statistic": result.z_statistic,
                "p_value_two_sided": result.p_value,
                "p_value_holm": adjusted_p_value,
                "significant_holm_0_05": adjusted_p_value < alpha,
            }
        )

    verify_input_hashes(input_hashes)
    metadata = {
        "method": "paired DeLong test for correlated ROC AUCs",
        "confidence_interval": "two-sided 95% normal-approximation DeLong CI",
        "multiple_testing": "Holm correction over all pairwise comparisons",
        "analysis_set": analysis_set,
        "stage": stage,
        "alpha": alpha,
        "n_cases": len(case_ids),
        "n_positive": int(targets.sum()),
        "n_negative": int(len(targets) - targets.sum()),
        "case_id_set_sha256": hashlib.sha256(
            "\n".join(case_ids).encode("utf-8")
        ).hexdigest(),
        "auc_order": [item.name for item in model_inputs],
        "aucs": aucs.tolist(),
        "delong_covariance": covariance.tolist(),
        "models": [
            {
                **asdict(item),
                "predictions_path": str(item.predictions_path),
                "metrics_path": (
                    str(item.metrics_path) if item.metrics_path is not None else None
                ),
                "predictions_sha256": input_hashes[item.predictions_path],
                "metrics_sha256": (
                    input_hashes[item.metrics_path]
                    if item.metrics_path is not None
                    else None
                ),
            }
            for item in model_inputs
        ],
        "implementation": {
            "python": sys.version,
            "numpy": np.__version__,
            "cli_sha256": _sha256(Path(__file__).resolve()),
            "delong_sha256": _sha256(
                Path(__file__).resolve().parent / "statistical_tests" / "delong.py"
            ),
        },
        "interpretation_boundary": (
            "Internal validation inference after validation-based checkpoint "
            "selection; treat as exploratory, not external generalization."
            if analysis_set == "validation"
            else "External validation inference for checkpoints fixed before "
            "external evaluation; do not use these results for model, checkpoint, "
            "hyperparameter, or threshold selection."
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=str(output_dir.parent),
        )
    )
    try:
        _write_csv(temporary_dir / "model_auc.csv", model_rows)
        _write_csv(temporary_dir / "pairwise_delong.csv", pair_rows)
        with (temporary_dir / "analysis_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        (temporary_dir / "README.md").write_text(
            _markdown_summary(model_rows, pair_rows, analysis_set), encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Model name and case-level prediction CSV; repeat per model.",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Optional training_metrics.csv override; repeat as needed.",
    )
    parser.add_argument(
        "--epoch",
        action="append",
        default=[],
        metavar="NAME=EPOCH",
        help="Optional selected-epoch override; repeat as needed.",
    )
    parser.add_argument(
        "--analysis-set",
        choices=("validation", "external"),
        default="validation",
        help="Evaluation role; external mode accepts prediction CSVs with blank epoch.",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Exact stage value in prediction CSV (default: validation or evaluation).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stage = args.stage or (
            "validation" if args.analysis_set == "validation" else "evaluation"
        )
        model_inputs = resolve_model_inputs(
            model_values=args.model,
            metrics_values=args.metrics,
            epoch_values=args.epoch,
            analysis_set=args.analysis_set,
        )
        input_hashes = capture_input_hashes(model_inputs)
        case_ids, targets, scores = align_predictions(
            model_inputs=model_inputs,
            stage=stage,
        )
        write_results(
            output_dir=args.output_dir.expanduser().resolve(),
            model_inputs=model_inputs,
            stage=stage,
            case_ids=case_ids,
            targets=targets,
            scores=scores,
            alpha=args.alpha,
            input_hashes=input_hashes,
            analysis_set=args.analysis_set,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    sys.stdout.write(f"DeLong analysis written to: {args.output_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
