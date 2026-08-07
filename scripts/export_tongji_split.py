"""Export the exact Tongji train/validation split used by the datasets."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Make repository packages importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets.classification.esophagus_risk_roi_3d import (
    CaseRecord,
    _discover_records,
    _filter_records_by_esophagus_height,
    _split_train_val_records,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitEntry:
    split: str
    record: CaseRecord
    esophagus_height_mm: float | None = None


def build_split_entries(
    dataset_root: Path,
    val_ratio: float,
    split_seed: int,
    esophagus_label_id: int,
    min_esophagus_height_mm: float,
) -> list[SplitEntry]:
    """Discover, filter, and split cases exactly as the training datasets do."""
    records = _discover_records(dataset_root)
    included, excluded = _filter_records_by_esophagus_height(
        records=records,
        label_id=esophagus_label_id,
        min_height_mm=min_esophagus_height_mm,
    )
    train_records, val_records = _split_train_val_records(
        records=included,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    entries = [
        *(SplitEntry(split="train", record=record) for record in train_records),
        *(SplitEntry(split="validation", record=record) for record in val_records),
        *(
            SplitEntry(
                split="excluded_height",
                record=record,
                esophagus_height_mm=height_mm,
            )
            for record, height_mm in excluded
        ),
    ]
    split_order = {"train": 0, "validation": 1, "excluded_height": 2}
    return sorted(
        entries,
        key=lambda entry: (split_order[entry.split], entry.record.case_id),
    )


def write_split_csv(entries: Sequence[SplitEntry], output_path: Path) -> None:
    """Write split membership and source paths to a UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "split",
                "case_id",
                "label",
                "label_name",
                "ct_path",
                "mask_path",
                "esophagus_height_mm",
            ),
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "split": entry.split,
                    "case_id": entry.record.case_id,
                    "label": entry.record.label,
                    "label_name": "hr" if entry.record.label == 1 else "nhr",
                    "ct_path": str(entry.record.ct_path),
                    "mask_path": str(entry.record.mask_path),
                    "esophagus_height_mm": (
                        f"{entry.esophagus_height_mm:.6f}"
                        if entry.esophagus_height_mm is not None
                        else ""
                    ),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the stratified Tongji train/validation split after applying "
            "the minimum esophagus-height filter."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Tongji dataset root containing the hr and nhr directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: tongji_split_seed<SPLIT_SEED>.csv).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--esophagus-label-id", type=int, default=15)
    parser.add_argument("--min-esophagus-height-mm", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = build_split_entries(
        dataset_root=args.dataset_root.expanduser(),
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        esophagus_label_id=args.esophagus_label_id,
        min_esophagus_height_mm=args.min_esophagus_height_mm,
    )
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else Path(f"tongji_split_seed{args.split_seed}.csv")
    )
    write_split_csv(entries=entries, output_path=output_path)

    counts = {
        split: sum(entry.split == split for entry in entries)
        for split in ("train", "validation", "excluded_height")
    }
    LOGGER.info(
        "Saved split to %s (train=%d, validation=%d, excluded_height=%d)",
        output_path.resolve(),
        counts["train"],
        counts["validation"],
        counts["excluded_height"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
