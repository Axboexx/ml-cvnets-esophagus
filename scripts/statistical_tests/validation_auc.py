"""Input validation and case alignment for paired AUC comparisons."""

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

REQUIRED_PREDICTION_FIELDS = {
    "epoch",
    "stage",
    "case_id",
    "target",
    "positive_probability",
}


@dataclass(frozen=True)
class ModelInput:
    """Resolved input files and selected partition for one model."""

    name: str
    predictions_path: Path
    metrics_path: Optional[Path]
    epoch: Optional[int]
    epoch_selection: str = "manual_override"
    metrics_sha256_at_selection: Optional[str] = None


@dataclass(frozen=True)
class CasePrediction:
    """Binary target and positive-class score for one case."""

    target: int
    score: float


def _parse_named_values(values: Sequence[str], option: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        name = name.strip()
        item = item.strip()
        if not separator or not name or not item:
            raise ValueError(f"{option} must use NAME=VALUE: {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate model name in {option}: {name}")
        parsed[name] = item
    return parsed


def _parse_integer(value: str, context: str) -> int:
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"invalid integer for {context}: {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"value must be an integer for {context}: {value!r}")
    return int(numeric)


def _parse_boolean(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean at {context}: {value!r}")


def _select_best_epoch_snapshot(metrics_path: Path) -> Tuple[int, str]:
    """Select the best epoch and hash the exact bytes used for selection."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"training metrics not found: {metrics_path}")
    contents = metrics_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"training metrics is not UTF-8: {metrics_path}") from error
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"epoch", "is_best"} - fields
        if missing:
            raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")
        best_rows = []
        for row_number, row in enumerate(reader, start=2):
            if _parse_boolean(row.get("is_best", ""), f"{metrics_path}:{row_number}"):
                best_rows.append((row_number, row))
    if not best_rows:
        raise ValueError(f"no row has is_best=True in {metrics_path}")

    row_number, best_row = best_rows[-1]
    metric_source = best_row.get("best_metric_source", "").strip()
    if "best_metric_source" in fields and metric_source not in {
        "validation",
        "validation_loss",
    }:
        raise ValueError(
            f"best epoch in {metrics_path} was selected from "
            f"{metric_source or 'an empty metric source'}; use a validation-selected "
            "checkpoint or provide an explicit --epoch override"
        )
    epoch = _parse_integer(best_row["epoch"], f"{metrics_path}:{row_number}")
    return epoch, digest


def select_best_epoch(metrics_path: Path) -> int:
    """Return the last epoch marked is_best, matching checkpoint replacement."""
    epoch, _ = _select_best_epoch_snapshot(metrics_path)
    return epoch


def resolve_model_inputs(
    model_values: Sequence[str],
    metrics_values: Sequence[str],
    epoch_values: Sequence[str],
    analysis_set: str = "validation",
) -> List[ModelInput]:
    """Resolve paths and epochs while preserving command-line model order."""
    if analysis_set not in {"validation", "external"}:
        raise ValueError(f"unsupported analysis set: {analysis_set!r}")
    models = _parse_named_values(model_values, "--model")
    metrics = _parse_named_values(metrics_values, "--metrics")
    epochs = _parse_named_values(epoch_values, "--epoch")
    if len(models) < 2:
        raise ValueError("at least two --model inputs are required")

    unknown = (set(metrics) | set(epochs)) - set(models)
    if unknown:
        raise ValueError(f"options reference unknown models: {sorted(unknown)}")

    resolved = []
    for name, path_value in models.items():
        predictions_path = Path(path_value).expanduser().resolve()
        if not predictions_path.is_file():
            raise FileNotFoundError(
                f"prediction CSV for {name} not found: {predictions_path}"
            )
        metrics_path = None
        if name in metrics:
            metrics_path = Path(metrics[name]).expanduser().resolve()
        elif analysis_set == "validation":
            metrics_path = (predictions_path.parent / "training_metrics.csv").resolve()
        if name in metrics and metrics_path is not None and not metrics_path.is_file():
            raise FileNotFoundError(
                f"explicit training metrics for {name} not found: {metrics_path}"
            )

        if name in epochs:
            epoch = _parse_integer(epochs[name], name)
            optional_metrics_path = (
                metrics_path
                if metrics_path is not None and metrics_path.is_file()
                else None
            )
            epoch_selection = "manual_override"
            metrics_sha256_at_selection = None
        elif analysis_set == "external":
            epoch = None
            optional_metrics_path = (
                metrics_path
                if metrics_path is not None and metrics_path.is_file()
                else None
            )
            epoch_selection = "external_fixed_checkpoint"
            metrics_sha256_at_selection = None
        else:
            if metrics_path is None:
                raise ValueError(f"training metrics are required for {name}")
            epoch, metrics_sha256_at_selection = _select_best_epoch_snapshot(
                metrics_path
            )
            optional_metrics_path = metrics_path
            epoch_selection = "auto_last_is_best"
        resolved.append(
            ModelInput(
                name=name,
                predictions_path=predictions_path,
                metrics_path=optional_metrics_path,
                epoch=epoch,
                epoch_selection=epoch_selection,
                metrics_sha256_at_selection=metrics_sha256_at_selection,
            )
        )
    return resolved


def load_prediction_partition(
    path: Path,
    epoch: Optional[int],
    stage: str,
) -> Dict[str, CasePrediction]:
    """Load one epoch/stage partition and enforce one valid row per case."""
    predictions: Dict[str, CasePrediction] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_PREDICTION_FIELDS - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if row.get("stage", "").strip() != stage:
                continue
            if epoch is not None and (
                _parse_integer(row.get("epoch", ""), f"{path}:{row_number}")
                != epoch
            ):
                continue
            case_id = row.get("case_id", "").strip()
            if not case_id:
                raise ValueError(f"empty case_id at {path}:{row_number}")
            if case_id in predictions:
                raise ValueError(
                    f"duplicate case_id={case_id!r} in epoch {epoch}, stage {stage}"
                )
            target = _parse_integer(row.get("target", ""), f"{path}:{row_number}")
            if target not in (0, 1):
                raise ValueError(f"target must be 0 or 1 at {path}:{row_number}")
            try:
                score = float(row.get("positive_probability", ""))
            except ValueError as error:
                raise ValueError(
                    f"invalid positive_probability at {path}:{row_number}"
                ) from error
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"positive_probability must be finite and in [0, 1] "
                    f"at {path}:{row_number}"
                )
            predictions[case_id] = CasePrediction(target=target, score=score)
    if not predictions:
        partition = f"stage={stage!r}"
        if epoch is not None:
            partition = f"epoch={epoch}, {partition}"
        raise ValueError(f"no rows found for {partition} in {path}")
    return predictions


def align_predictions(
    model_inputs: Sequence[ModelInput],
    stage: str,
) -> Tuple[List[str], NDArray[np.int64], NDArray[np.float64]]:
    """Strictly align models on identical case IDs and targets."""
    by_model = [
        load_prediction_partition(item.predictions_path, item.epoch, stage)
        for item in model_inputs
    ]
    reference_ids = set(by_model[0])
    for item, predictions in zip(model_inputs[1:], by_model[1:]):
        current_ids = set(predictions)
        if current_ids != reference_ids:
            missing = sorted(reference_ids - current_ids)[:5]
            extra = sorted(current_ids - reference_ids)[:5]
            raise ValueError(
                f"case_id cohort mismatch for {item.name}: "
                f"missing={missing}, extra={extra}"
            )

    case_ids = sorted(reference_ids)
    targets = np.asarray(
        [by_model[0][case_id].target for case_id in case_ids], dtype=np.int64
    )
    for item, predictions in zip(model_inputs[1:], by_model[1:]):
        mismatched = [
            case_id
            for case_id in case_ids
            if predictions[case_id].target != by_model[0][case_id].target
        ]
        if mismatched:
            raise ValueError(
                f"target mismatch for {item.name}; examples={mismatched[:5]}"
            )
    scores = np.asarray(
        [
            [predictions[case_id].score for case_id in case_ids]
            for predictions in by_model
        ],
        dtype=np.float64,
    )
    return case_ids, targets, scores


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Return Holm step-down family-wise-error adjusted P values."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted
