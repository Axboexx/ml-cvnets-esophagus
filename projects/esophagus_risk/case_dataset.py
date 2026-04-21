from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class CaseSample:
    case_id: str
    label: int
    image_paths: tuple[Path, ...]


def load_case_samples(
    data_dir: Path,
    split: str,
) -> tuple[list[CaseSample], int]:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    target_num_slices = int(metadata["num_slices_per_case"])

    grouped: dict[str, list[tuple[int, Path, int]]] = {}
    manifest_path = data_dir / "export_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] != split:
                continue
            case_id = row["case_id"]
            grouped.setdefault(case_id, []).append(
                (
                    int(row["slice_index"]),
                    data_dir / row["png_path"],
                    int(row["risk_label"]),
                )
            )

    samples: list[CaseSample] = []
    for case_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item[0])
        label = ordered[0][2]
        image_paths = tuple(path for _, path, _ in ordered)
        samples.append(CaseSample(case_id=case_id, label=label, image_paths=image_paths))

    if not samples:
        raise ValueError(f"No case samples found for split={split} in {manifest_path}")
    return samples, target_num_slices


class EsophagusRoiCaseDataset(Dataset[dict[str, Tensor | int | str]]):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        image_size: int,
        target_num_slices: int | None = None,
        random_slice_dropout: int = 0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples, default_num_slices = load_case_samples(self.data_dir, split=split)
        self.target_num_slices = target_num_slices or default_num_slices
        self.random_slice_dropout = max(0, random_slice_dropout)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _select_image_paths(self, image_paths: tuple[Path, ...]) -> tuple[list[Path], int]:
        selected = list(image_paths)
        if self.split == "train" and self.random_slice_dropout > 0 and len(selected) > 1:
            max_drop = min(self.random_slice_dropout, len(selected) - 1)
            drop_count = random.randint(0, max_drop)
            keep_count = len(selected) - drop_count
            chosen_indices = sorted(random.sample(range(len(selected)), k=keep_count))
            selected = [selected[index] for index in chosen_indices]

        valid_slices = min(len(selected), self.target_num_slices)
        selected = selected[: self.target_num_slices]
        while len(selected) < self.target_num_slices:
            selected.append(selected[-1])
        return selected, valid_slices

    def __getitem__(self, index: int) -> dict[str, Tensor | int | str]:
        sample = self.samples[index]
        image_paths, valid_slices = self._select_image_paths(sample.image_paths)

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self.transform(image.convert("RGB")))

        images = torch.stack(tensors, dim=0)
        slice_mask = torch.zeros(self.target_num_slices, dtype=torch.float32)
        slice_mask[:valid_slices] = 1.0
        label = torch.tensor(sample.label, dtype=torch.long)
        return {
            "images": images,
            "label": label,
            "slice_mask": slice_mask,
            "case_id": sample.case_id,
        }
