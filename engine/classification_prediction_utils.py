#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import csv
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


PREDICTION_CSV_FIELDS = [
    "epoch",
    "stage",
    "sample_index",
    "case_id",
    "target",
    "prediction",
    "positive_probability",
    "negative_probability",
    "positive_logit",
    "negative_logit",
    "is_correct",
]


def build_binary_prediction_rows(
    prediction: Tensor,
    target: Tensor,
    sample_ids: Tensor,
    case_id_resolver: Callable[[int], str],
    stage: str,
    epoch: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convert binary-classification logits into case-level prediction rows."""
    if prediction.ndim != 2 or target.ndim != 1:
        return []
    if prediction.shape[0] != target.shape[0]:
        return []
    if prediction.shape[1] not in (1, 2):
        return []

    sample_ids = torch.as_tensor(sample_ids).reshape(-1)
    if sample_ids.numel() != target.numel():
        return []

    logits = prediction.detach().float().cpu()
    labels = target.detach().long().cpu()
    if not torch.all((labels == 0) | (labels == 1)):
        return []

    if logits.shape[1] == 2:
        probabilities = torch.softmax(logits, dim=1)
        negative_probabilities = probabilities[:, 0]
        positive_probabilities = probabilities[:, 1]
        negative_logits: Sequence[Optional[float]] = logits[:, 0].tolist()
        positive_logits = logits[:, 1].tolist()
    else:
        positive_probabilities = torch.sigmoid(logits[:, 0])
        negative_probabilities = 1.0 - positive_probabilities
        negative_logits = [None] * logits.shape[0]
        positive_logits = logits[:, 0].tolist()

    rows = []
    for values in zip(
        sample_ids.detach().cpu().tolist(),
        labels.tolist(),
        positive_probabilities.tolist(),
        negative_probabilities.tolist(),
        positive_logits,
        negative_logits,
    ):
        (
            sample_index,
            label,
            positive_probability,
            negative_probability,
            positive_logit,
            negative_logit,
        ) = values
        sample_index = int(sample_index)
        predicted_label = int(float(positive_probability) >= 0.5)
        rows.append(
            {
                "epoch": epoch,
                "stage": stage,
                "sample_index": sample_index,
                "case_id": case_id_resolver(sample_index),
                "target": int(label),
                "prediction": predicted_label,
                "positive_probability": float(positive_probability),
                "negative_probability": float(negative_probability),
                "positive_logit": float(positive_logit),
                "negative_logit": (
                    None if negative_logit is None else float(negative_logit)
                ),
                "is_correct": int(label) == predicted_label,
            }
        )
    return rows


def _same_sample_identity(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """Check metadata that must agree for sampler-generated duplicates."""
    identity_fields = ("epoch", "stage", "sample_index", "case_id", "target")
    return all(first.get(field) == second.get(field) for field in identity_fields)


def normalize_prediction_rows(
    rows: Sequence[Dict[str, Any]], expected_count: int
) -> Tuple[List[Dict[str, Any]], int]:
    """Validate rows and remove sampler padding by dataset sample index.

    Distributed validation samplers may repeat a sample both to equalize rank lengths
    and to fill the final per-rank batch. Repeated forwards can differ slightly under
    mixed-precision CUDA kernels, so numerical predictions are deliberately excluded
    from duplicate identity checks. The first occurrence is retained exactly once.
    """
    unique_by_sample_index: Dict[int, Dict[str, Any]] = {}
    sample_index_by_case: Dict[str, int] = {}
    duplicate_count = 0
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError("Prediction row is missing case_id")
        sample_index = int(row["sample_index"])

        positive_probability = float(row["positive_probability"])
        negative_probability = float(row["negative_probability"])
        if not (
            math.isfinite(positive_probability)
            and math.isfinite(negative_probability)
            and 0.0 <= positive_probability <= 1.0
            and 0.0 <= negative_probability <= 1.0
            and math.isclose(
                positive_probability + negative_probability,
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(f"Invalid probabilities for case_id={case_id}")

        previous = unique_by_sample_index.get(sample_index)
        if previous is not None:
            if not _same_sample_identity(previous, row):
                raise ValueError(
                    "Conflicting metadata for duplicate prediction: "
                    f"sample_index={sample_index}, case_id={case_id}"
                )
            duplicate_count += 1
            continue

        previous_sample_index = sample_index_by_case.get(case_id)
        if previous_sample_index is not None:
            raise ValueError(
                "Duplicate case_id maps to different dataset samples: "
                f"case_id={case_id}, sample_indices="
                f"{previous_sample_index},{sample_index}"
            )

        unique_by_sample_index[sample_index] = dict(row)
        sample_index_by_case[case_id] = sample_index

    normalized = sorted(
        unique_by_sample_index.values(),
        key=lambda row: (int(row["sample_index"]), str(row["case_id"])),
    )
    if len(normalized) != expected_count:
        raise ValueError(
            "Prediction count does not match dataset size: "
            f"expected={expected_count}, unique={len(normalized)}, raw={len(rows)}"
        )
    return normalized, duplicate_count


def write_prediction_rows_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    preserve_other_partitions: bool,
) -> None:
    """Atomically write predictions, replacing matching epoch/stage partitions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    retained_rows: List[Dict[str, Any]] = []
    partitions = {(str(row.get("epoch")), str(row.get("stage"))) for row in rows}

    if preserve_other_partitions and path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PREDICTION_CSV_FIELDS:
                raise ValueError(f"Unexpected prediction CSV schema: {path}")
            for row in reader:
                partition = (str(row.get("epoch")), str(row.get("stage")))
                if partition not in partitions:
                    retained_rows.append(row)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_CSV_FIELDS)
        writer.writeheader()
        for row in [*retained_rows, *rows]:
            writer.writerow({field: row.get(field) for field in PREDICTION_CSV_FIELDS})
    temporary_path.replace(path)
