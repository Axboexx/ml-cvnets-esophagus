import csv

import pytest
import torch

from engine.classification_prediction_utils import (
    build_binary_prediction_rows,
    normalize_prediction_rows,
    write_prediction_rows_csv,
)


def _prediction_rows(epoch: int = 1) -> list:
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1])
    sample_ids = torch.tensor([0, 1])
    return build_binary_prediction_rows(
        prediction=logits,
        target=targets,
        sample_ids=sample_ids,
        case_id_resolver=lambda index: f"case-{index}",
        stage="validation",
        epoch=epoch,
    )


def test_build_binary_prediction_rows_exports_every_case() -> None:
    rows = _prediction_rows()

    assert [row["case_id"] for row in rows] == ["case-0", "case-1"]
    assert [row["prediction"] for row in rows] == [0, 1]
    assert all(row["is_correct"] for row in rows)
    assert rows[0]["positive_probability"] == pytest.approx(0.26894143)
    assert rows[0]["negative_probability"] == pytest.approx(0.7310586)
    assert rows[0]["positive_logit"] == 1.0
    assert rows[0]["negative_logit"] == 2.0


def test_normalize_prediction_rows_collapses_ddp_padding() -> None:
    rows = _prediction_rows()

    normalized, duplicate_count = normalize_prediction_rows(
        rows=[*rows, dict(rows[0])], expected_count=2
    )

    assert len(normalized) == 2
    assert duplicate_count == 1


def test_normalize_prediction_rows_collapses_numerically_different_padding() -> None:
    rows = _prediction_rows()
    padded_row = dict(
        rows[0],
        prediction=1,
        positive_probability=0.500001,
        negative_probability=0.499999,
        positive_logit=1.00001,
    )

    normalized, duplicate_count = normalize_prediction_rows(
        rows=[*rows, padded_row], expected_count=2
    )

    assert duplicate_count == 1
    assert normalized == rows


def test_normalize_prediction_rows_rejects_case_id_reuse() -> None:
    rows = _prediction_rows()
    conflicting_row = dict(rows[1], case_id=rows[0]["case_id"])

    with pytest.raises(ValueError, match="case_id maps to different dataset samples"):
        normalize_prediction_rows(rows=[rows[0], conflicting_row], expected_count=2)


def test_normalize_prediction_rows_checks_dataset_size() -> None:
    with pytest.raises(ValueError, match="expected=3, unique=2"):
        normalize_prediction_rows(rows=_prediction_rows(), expected_count=3)


def test_prediction_csv_replaces_only_matching_epoch_stage(tmp_path) -> None:
    path = tmp_path / "validation_predictions.csv"
    epoch_one_rows = _prediction_rows(epoch=1)
    epoch_two_rows = _prediction_rows(epoch=2)
    write_prediction_rows_csv(
        path=path,
        rows=epoch_one_rows,
        preserve_other_partitions=True,
    )
    write_prediction_rows_csv(
        path=path,
        rows=epoch_two_rows,
        preserve_other_partitions=True,
    )
    replacement = [dict(epoch_one_rows[0], positive_probability=0.25)]
    write_prediction_rows_csv(
        path=path,
        rows=replacement,
        preserve_other_partitions=True,
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert sum(row["epoch"] == "1" for row in rows) == 1
    assert sum(row["epoch"] == "2" for row in rows) == 2
