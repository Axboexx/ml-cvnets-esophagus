#!/usr/bin/env python3
"""Summarize G1-G3 Tongji internal validation and Youyi external validation.

Example (run from the ml-cvnets repository on the server):

    python3 scripts/summarize_g_series_results.py \
        --results-root /home/mw/work/results \
        --output-dir /home/mw/work/results/G_series_summary

Selection protocol (same as the 2026-09-17 F-series lock):

- checkpoint: the epoch with the highest validation AUC in
  ``validation_confusion_matrix.jsonl``; ties keep the earliest epoch. The
  epoch must match the top ``checkpoint_auc_epoch_*`` file and
  ``checkpoint_auc_best.pt`` must be an identical copy of it.
- cutoff: Youden J on that epoch's ``validation_predictions.csv``. Candidates
  are every distinct positive probability plus an all-negative endpoint; J
  ties prefer higher sensitivity, then the lower cutoff. A case is positive
  when ``positive_probability >= cutoff``.
- external: the locked cutoff is applied once to
  ``external_eval_predictions.csv``; Youyi plays no part in any selection.

Internal threshold metrics are optimistic because the same 60 validation cases
select both the checkpoint and the cutoff.
"""

import argparse
import csv
import filecmp
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

try:
    from scripts.statistical_tests.delong import delong_estimates
except ModuleNotFoundError:
    from statistical_tests.delong import delong_estimates


# (model name, results-loc subdirectory, training run-label)
G_SERIES_RUNS: tuple[tuple[str, str, str], ...] = (
    ("G1", "G1", "g1_esophagus_masked_tongji82_4cm_seed42"),
    ("G2a", "G2a", "g2a_esophagus_masked_tongji82_4cm_4seg_seed42"),
    ("G2b", "G2b", "g2b_esophagus_masked_tongji82_4cm_4seg_seed42"),
    (
        "G3 E+L",
        "G3_esophagus_liver",
        # Rerun of the first E+L training; replaces ..._seed42 for every G analysis.
        "g3_esophagus_liver_masked_tongji82_4cm_seed42_2",
    ),
    (
        "G3 E+L+PV",
        "G3_esophagus_liver_portal",
        "g3_esophagus_liver_portal_masked_tongji82_4cm_seed42",
    ),
    (
        "G3 E+L+S",
        "G3_esophagus_liver_spleen",
        "g3_esophagus_liver_spleen_masked_tongji82_4cm_seed42",
    ),
    (
        "G3 E+L+S+PV",
        "G3_esophagus_liver_spleen_portal",
        "g3_esophagus_liver_spleen_portal_masked_tongji82_4cm_seed42",
    ),
    (
        "G3 E+PV",
        "G3_esophagus_portal",
        "g3_esophagus_portal_masked_tongji82_4cm_seed42",
    ),
    (
        "G3 E+S",
        "G3_esophagus_spleen",
        "g3_esophagus_spleen_masked_tongji82_4cm_seed42",
    ),
    (
        "G3 E+S+PV",
        "G3_esophagus_spleen_portal",
        "g3_esophagus_spleen_portal_masked_tongji82_4cm_seed42",
    ),
)
EXTERNAL_SUFFIX = "_youyi_external_auc_best"
VALIDATION_STAGE = "validation"
EXTERNAL_STAGE = "youyi_external"
AUC_CHECKPOINT_PATTERN = re.compile(
    r"checkpoint_auc_epoch_(\d+)_score_([0-9]+(?:\.[0-9]+)?)\.pt"
)
AUC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Case:
    case_id: str
    target: int
    probability: float


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def metrics(self) -> dict[str, float | None]:
        total = self.tn + self.fp + self.fn + self.tp
        return {
            "accuracy": _ratio(self.tp + self.tn, total),
            "sensitivity": _ratio(self.tp, self.tp + self.fn),
            "specificity": _ratio(self.tn, self.tn + self.fp),
            "ppv": _ratio(self.tp, self.tp + self.fp),
            "npv": _ratio(self.tn, self.tn + self.fn),
            "f1": _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def operating_point(cases: Sequence[Case], threshold: float) -> OperatingPoint:
    tn = fp = fn = tp = 0
    for case in cases:
        predicted = case.probability >= threshold
        if case.target == 1:
            tp += predicted
            fn += not predicted
        else:
            fp += predicted
            tn += not predicted
    return OperatingPoint(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp)


def youden_cutoff(cases: Sequence[Case]) -> OperatingPoint:
    """Exact Youden J; ties prefer higher sensitivity, then a lower cutoff."""
    positives = sum(case.target for case in cases)
    negatives = len(cases) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Youden cutoff needs both classes")
    candidates = sorted({case.probability for case in cases}) + [math.inf]
    best: OperatingPoint | None = None
    best_key: tuple[Fraction, Fraction, float] | None = None
    for threshold in candidates:
        point = operating_point(cases, threshold)
        sensitivity = Fraction(point.tp, positives)
        specificity = Fraction(point.tn, negatives)
        key = (sensitivity + specificity - 1, sensitivity, -threshold)
        if best_key is None or key > best_key:
            best, best_key = point, key
    assert best is not None
    return best


def read_validation_auc_by_epoch(path: Path) -> dict[int, float]:
    """Latest validation AUC per epoch (a resumed run may repeat epochs)."""
    aucs: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("stage") != VALIDATION_STAGE or record.get("auc") is None:
                continue
            epoch = int(record["epoch"])
            if epoch in aucs and abs(aucs[epoch] - float(record["auc"])) > 0:
                print(
                    f"warning: {path}:{line_number} repeats epoch {epoch} with a "
                    "different AUC; the latest record is used",
                    file=sys.stderr,
                )
            aucs[epoch] = float(record["auc"])
    if not aucs:
        raise ValueError(f"no validation AUC records in {path}")
    return aucs


def select_auc_best_epoch(aucs: dict[int, float]) -> int:
    return max(aucs, key=lambda epoch: (aucs[epoch], -epoch))


def top_auc_checkpoint(run_dir: Path) -> tuple[int, Path]:
    matches = []
    for path in run_dir.glob("checkpoint_auc_epoch_*_score_*.pt"):
        match = AUC_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            matches.append((float(match.group(2)), -int(match.group(1)), path))
    if not matches:
        raise FileNotFoundError(f"no checkpoint_auc_epoch_* file in {run_dir}")
    _, negative_epoch, path = max(matches)
    return -negative_epoch, path


def read_predictions(path: Path, stage: str, epoch: int | None) -> list[Case]:
    by_case: dict[str, Case] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stage"] != stage:
                continue
            if epoch is not None and int(row["epoch"]) != epoch:
                continue
            case = Case(
                case_id=row["case_id"],
                target=int(row["target"]),
                probability=float(row["positive_probability"]),
            )
            if case.target not in (0, 1) or not math.isfinite(case.probability):
                raise ValueError(f"invalid prediction row in {path}: {row}")
            previous = by_case.get(case.case_id)
            if previous is not None and previous != case:
                raise ValueError(f"conflicting rows for case {case.case_id} in {path}")
            by_case[case.case_id] = case
    if not by_case:
        raise ValueError(f"no {stage} predictions (epoch={epoch}) in {path}")
    return sorted(by_case.values(), key=lambda case: case.case_id)


def auc_with_ci(cases: Sequence[Case]) -> tuple[float, float, float]:
    estimate = delong_estimates(
        targets=[case.target for case in cases],
        scores=[[case.probability for case in cases]],
    )[0]
    return estimate.auc, estimate.ci_lower, estimate.ci_upper


def cohort_signature(cases: Sequence[Case]) -> tuple[tuple[str, int], ...]:
    return tuple((case.case_id, case.target) for case in cases)


def summarize_run(results_root: Path, name: str, subdir: str, label: str) -> dict:
    run_dir = results_root / subdir / label
    external_dir = results_root / subdir / f"{label}{EXTERNAL_SUFFIX}"
    for directory in (run_dir, external_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name}: directory not found: {directory}")

    aucs = read_validation_auc_by_epoch(run_dir / "validation_confusion_matrix.jsonl")
    best_epoch = select_auc_best_epoch(aucs)
    tie_epochs = sorted(e for e, auc in aucs.items() if auc == aucs[best_epoch])

    checkpoint_epoch, checkpoint_path = top_auc_checkpoint(run_dir)
    if checkpoint_epoch != best_epoch:
        raise ValueError(
            f"{name}: AUC-best epoch {best_epoch} from the jsonl differs from the "
            f"top AUC checkpoint {checkpoint_path.name}"
        )
    best_copy = run_dir / "checkpoint_auc_best.pt"
    if not best_copy.is_file() or not filecmp.cmp(
        best_copy, checkpoint_path, shallow=False
    ):
        raise ValueError(f"{name}: {best_copy} is not a copy of {checkpoint_path}")

    validation = read_predictions(
        run_dir / "validation_predictions.csv", VALIDATION_STAGE, best_epoch
    )
    val_auc, val_lower, val_upper = auc_with_ci(validation)
    if abs(val_auc - aucs[best_epoch]) > AUC_TOLERANCE:
        raise ValueError(
            f"{name}: recomputed validation AUC {val_auc} differs from the logged "
            f"AUC {aucs[best_epoch]} at epoch {best_epoch}"
        )
    cutoff = youden_cutoff(validation)

    external = read_predictions(
        external_dir / "external_eval_predictions.csv", EXTERNAL_STAGE, None
    )
    ext_auc, ext_lower, ext_upper = auc_with_ci(external)
    ext_cutoff = operating_point(external, cutoff.threshold)
    ext_half = operating_point(external, 0.5)

    row: dict = {
        "model": name,
        "epochs_logged": max(aucs),
        "auc_best_epoch": best_epoch,
        "auc_tie_epochs": ";".join(str(e) for e in tie_epochs),
        "val_n": len(validation),
        "val_hr": sum(case.target for case in validation),
        "val_auc": val_auc,
        "val_auc_ci_lower": val_lower,
        "val_auc_ci_upper": val_upper,
        "youden_cutoff": cutoff.threshold,
        "ext_n": len(external),
        "ext_hr": sum(case.target for case in external),
        "ext_auc": ext_auc,
        "ext_auc_ci_lower": ext_lower,
        "ext_auc_ci_upper": ext_upper,
    }
    for prefix, point in (
        ("val_cutoff", cutoff),
        ("ext_cutoff", ext_cutoff),
        ("ext_0.5", ext_half),
    ):
        row.update(
            {f"{prefix}_{k}": getattr(point, k) for k in ("tn", "fp", "fn", "tp")}
        )
        row.update({f"{prefix}_{k}": v for k, v in point.metrics().items()})
    row["_cohorts"] = (cohort_signature(validation), cohort_signature(external))
    row["_run_dir"] = str(run_dir)
    row["_external_dir"] = str(external_dir)
    return row


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _ci(row: dict, prefix: str) -> str:
    return (
        f"{_fmt(row[f'{prefix}_auc'])} "
        f"({_fmt(row[f'{prefix}_auc_ci_lower'])}-{_fmt(row[f'{prefix}_auc_ci_upper'])})"
    )


def _counts(row: dict, prefix: str) -> str:
    return "/".join(str(row[f"{prefix}_{k}"]) for k in ("tn", "fp", "fn", "tp"))


def _threshold_cells(row: dict, prefix: str) -> list[str]:
    keys = ("accuracy", "sensitivity", "specificity", "ppv", "npv", "f1")
    return [_fmt(row[f"{prefix}_{key}"]) for key in keys] + [_counts(row, prefix)]


def markdown_report(rows: Sequence[dict]) -> str:
    metric_header = "| ACC | Sens | Spec | PPV | NPV | F1 | TN/FP/FN/TP |"
    metric_rule = "|---:|---:|---:|---:|---:|---:|---|"
    lines = [
        "# G1-G3 internal and external validation",
        "",
        "Checkpoint: highest Tongji validation AUC (ties keep the earliest epoch).",
        "Cutoff: Youden J on that epoch's validation predictions; positive when",
        "`positive_probability >= cutoff`. AUC 95% CI: DeLong.",
        "",
        "## Tongji internal validation (selection set, optimistic)",
        "",
        "| Model | Epochs | Best epoch | AUC ties | n (HR) | AUC (95% CI) | Cutoff "
        + metric_header,
        "|---|---:|---:|---|---|---|---:" + metric_rule,
    ]
    for row in rows:
        cells = [
            row["model"],
            str(row["epochs_logged"]),
            str(row["auc_best_epoch"]),
            row["auc_tie_epochs"],
            f"{row['val_n']} ({row['val_hr']})",
            _ci(row, "val"),
            f"{row['youden_cutoff']:.8f}",
            *_threshold_cells(row, "val_cutoff"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    for title, prefix in (
        ("## Youyi external validation at the locked Youden cutoff", "ext_cutoff"),
        ("## Youyi external validation at 0.5 (reference only)", "ext_0.5"),
    ):
        lines += [
            "",
            title,
            "",
            "| Model | n (HR) | AUC (95% CI) " + metric_header,
            "|---|---|---" + metric_rule,
        ]
        for row in rows:
            cells = [
                row["model"],
                f"{row['ext_n']} ({row['ext_hr']})",
                _ci(row, "ext"),
                *_threshold_cells(row, prefix),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def check_shared_cohorts(rows: Sequence[dict]) -> list[str]:
    warnings = []
    for index, label in ((0, "validation"), (1, "external")):
        reference = rows[0]["_cohorts"][index]
        for row in rows[1:]:
            if row["_cohorts"][index] != reference:
                warnings.append(
                    f"{row['model']}: {label} cases or labels differ from "
                    f"{rows[0]['model']}"
                )
    return warnings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Must not already exist"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"use a fresh output directory: {output_dir}")

    rows = [summarize_run(results_root, *run) for run in G_SERIES_RUNS]
    warnings = check_shared_cohorts(rows)

    output_dir.mkdir(parents=True)
    fields = [key for key in rows[0] if not key.startswith("_")]
    with (output_dir / "g_series_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    report = markdown_report(rows)
    if warnings:
        report += "\n## Warnings\n\n" + "".join(f"- {w}\n" for w in warnings)
    (output_dir / "g_series_summary.md").write_text(report, encoding="utf-8")
    sources = {
        row["model"]: {"run_dir": row["_run_dir"], "external_dir": row["_external_dir"]}
        for row in rows
    }
    (output_dir / "sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(report)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
