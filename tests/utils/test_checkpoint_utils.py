from pathlib import Path

import pytest
import torch

from utils.checkpoint_utils import save_top_k_auc_checkpoints


def _load_scalar_weight(checkpoint_path: Path) -> float:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    return float(state_dict["weight"].item())


def test_auc_checkpoints_keep_global_top_five_without_touching_existing_files(
    tmp_path: Path,
) -> None:
    existing_checkpoints = {
        "checkpoint_best.pt": {"sentinel": torch.tensor([1.0])},
        "checkpoint_last.pt": {"sentinel": torch.tensor([2.0])},
        "training_checkpoint_last.pt": {"sentinel": torch.tensor([3.0])},
        "checkpoint_score_0.7500.pt": {"sentinel": torch.tensor([4.0])},
        "checkpoint_avg.pt": {"sentinel": torch.tensor([5.0])},
    }
    for file_name, payload in existing_checkpoints.items():
        torch.save(payload, tmp_path / file_name)

    auc_values = [0.61, 0.83, 0.72, 0.91, 0.77, 0.88, 0.69]
    for epoch, auc in enumerate(auc_values):
        save_top_k_auc_checkpoints(
            model_state={"weight": torch.tensor([float(epoch)])},
            auc=auc,
            epoch=epoch,
            save_dir=str(tmp_path),
            k_best_checkpoints=5,
        )

    auc_checkpoints = sorted(tmp_path.glob("checkpoint_auc_epoch_*_score_*.pt"))
    assert len(auc_checkpoints) == 5
    retained_auc_values = sorted(
        float(path.stem.rsplit("_", maxsplit=1)[-1]) for path in auc_checkpoints
    )
    assert retained_auc_values == pytest.approx([0.72, 0.77, 0.83, 0.88, 0.91])
    assert not list(tmp_path.glob("checkpoint_auc*avg*.pt"))

    for file_name, payload in existing_checkpoints.items():
        saved_payload = torch.load(tmp_path / file_name, map_location="cpu")
        assert torch.equal(saved_payload["sentinel"], payload["sentinel"])


def test_auc_checkpoint_replaces_same_epoch_without_creating_duplicate(
    tmp_path: Path,
) -> None:
    save_top_k_auc_checkpoints(
        model_state={"weight": torch.tensor([1.0])},
        auc=0.7,
        epoch=3,
        save_dir=str(tmp_path),
        k_best_checkpoints=5,
    )
    save_top_k_auc_checkpoints(
        model_state={"weight": torch.tensor([2.0])},
        auc=0.8,
        epoch=3,
        save_dir=str(tmp_path),
        k_best_checkpoints=5,
    )

    auc_checkpoints = list(tmp_path.glob("checkpoint_auc_epoch_*_score_*.pt"))
    assert len(auc_checkpoints) == 1
    assert auc_checkpoints[0].name == "checkpoint_auc_epoch_0003_score_0.80000000.pt"
    assert _load_scalar_weight(auc_checkpoints[0]) == pytest.approx(2.0)
