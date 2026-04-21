from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from projects.esophagus_risk.case_dataset import EsophagusRoiCaseDataset
from projects.esophagus_risk.case_model import RoiCropCaseClassifier
from projects.esophagus_risk.metrics import BinaryMetrics, compute_binary_metrics


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: BinaryMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a case-level Esophagus ROI Crop classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="ROI export directory with metadata.json")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=("resnet18", "resnet34"))
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained torchvision weights")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-slices", type=int, default=None, help="Override slices per case from metadata")
    parser.add_argument("--random-slice-dropout", type=int, default=0, help="Drop up to N tail slices during training")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(
    data_dir: Path,
    split: str,
    image_size: int,
    num_slices: int | None,
    random_slice_dropout: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = EsophagusRoiCaseDataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size,
        target_num_slices=num_slices,
        random_slice_dropout=random_slice_dropout if split == "train" else 0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def maybe_collect_pip_freeze() -> list[str]:
    try:
        result = subprocess.run(
            ["python3", "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0
    for batch in loader:
        images = batch["images"].to(device=device, non_blocking=True)
        labels = batch["label"].to(device=device, non_blocking=True)
        slice_mask = batch["slice_mask"].to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images, slice_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size
    return running_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []

    for batch in loader:
        images = batch["images"].to(device=device, non_blocking=True)
        labels = batch["label"].to(device=device, non_blocking=True)
        slice_mask = batch["slice_mask"].to(device=device, non_blocking=True)

        logits = model(images, slice_mask)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

        labels_list.append(labels.cpu().numpy())
        probs_list.append(probabilities.cpu().numpy())

    labels_np = np.concatenate(labels_list, axis=0)
    probs_np = np.concatenate(probs_list, axis=0)
    return EpochResult(
        loss=running_loss / max(sample_count, 1),
        metrics=compute_binary_metrics(labels=labels_np, probabilities=probs_np),
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    epoch: int,
    best_auc: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_auc": best_auc,
        "args": vars(args),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    save_json(results_dir / "config.json", vars(args))
    save_json(
        results_dir / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "pip_freeze": maybe_collect_pip_freeze(),
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and torch.cuda.is_available()

    train_loader = build_loader(
        data_dir=data_dir,
        split="train",
        image_size=args.image_size,
        num_slices=args.num_slices,
        random_slice_dropout=args.random_slice_dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        data_dir=data_dir,
        split="val",
        image_size=args.image_size,
        num_slices=args.num_slices,
        random_slice_dropout=0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = RoiCropCaseClassifier(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_result = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            **asdict(val_result.metrics),
        }
        history.append(epoch_summary)
        save_json(results_dir / "history.json", {"epochs": history})

        latest_path = results_dir / "checkpoint_latest.pt"
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_auc=best_auc,
                args=args,
            ),
            latest_path,
        )

        if val_result.metrics.auc >= best_auc:
            best_auc = val_result.metrics.auc
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_auc=best_auc,
                    args=args,
                ),
                results_dir / "best_model.pt",
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_result.loss, 6),
                    "auc": round(val_result.metrics.auc, 6),
                    "accuracy": round(val_result.metrics.accuracy, 6),
                    "f1": round(val_result.metrics.f1, 6),
                    "sensitivity": round(val_result.metrics.sensitivity, 6),
                    "specificity": round(val_result.metrics.specificity, 6),
                },
                ensure_ascii=False,
            )
        )

    save_json(
        results_dir / "summary.json",
        {
            "best_auc": best_auc,
            "best_epoch": max(history, key=lambda item: item["auc"])["epoch"],
            "final_epoch": history[-1]["epoch"],
        },
    )


if __name__ == "__main__":
    main()
