from pathlib import Path

import pytest
import torch

from utils.checkpoint_utils import save_top_k_auc_checkpoints


def _load_scalar_weight(checkpoint_path: Path) -> float:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    return float(state_dict["weight"].item())


def _auc_checkpoint_names(save_dir: Path) -> list:
    return sorted(path.name for path in save_dir.glob("checkpoint_auc_epoch_*_score_*.pt"))


def _save_auc_sequence(save_dir: Path, auc_values: list, k: int = 5) -> None:
    for epoch, auc in enumerate(auc_values):
        save_top_k_auc_checkpoints(
            model_state={"weight": torch.tensor([float(epoch)])},
            auc=auc,
            epoch=epoch,
            save_dir=str(save_dir),
            k_best_checkpoints=k,
        )


def test_auc_checkpoints_save_only_new_records_without_touching_existing_files(
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

    # Epochs 2, 4, 6 are below the running record and must not be saved,
    # even though 0.88 (epoch 5) would be in a global top five.
    _save_auc_sequence(tmp_path, [0.61, 0.83, 0.72, 0.91, 0.77, 0.88, 0.69])

    assert _auc_checkpoint_names(tmp_path) == [
        "checkpoint_auc_epoch_0000_score_0.61000000.pt",
        "checkpoint_auc_epoch_0001_score_0.83000000.pt",
        "checkpoint_auc_epoch_0003_score_0.91000000.pt",
    ]
    assert _load_scalar_weight(tmp_path / "checkpoint_auc_best.pt") == pytest.approx(3.0)
    assert not list(tmp_path.glob("checkpoint_auc*avg*.pt"))

    for file_name, payload in existing_checkpoints.items():
        saved_payload = torch.load(tmp_path / file_name, map_location="cpu")
        assert torch.equal(saved_payload["sentinel"], payload["sentinel"])


def test_auc_checkpoint_ties_keep_earliest_epoch(tmp_path: Path) -> None:
    _save_auc_sequence(tmp_path, [0.8, 0.9, 0.9, 0.85, 0.9])

    assert _auc_checkpoint_names(tmp_path) == [
        "checkpoint_auc_epoch_0000_score_0.80000000.pt",
        "checkpoint_auc_epoch_0001_score_0.90000000.pt",
    ]
    assert _load_scalar_weight(tmp_path / "checkpoint_auc_best.pt") == pytest.approx(1.0)


def test_auc_checkpoints_retain_latest_k_records(tmp_path: Path) -> None:
    _save_auc_sequence(tmp_path, [0.5, 0.6, 0.7, 0.8], k=2)

    assert _auc_checkpoint_names(tmp_path) == [
        "checkpoint_auc_epoch_0002_score_0.70000000.pt",
        "checkpoint_auc_epoch_0003_score_0.80000000.pt",
    ]
    assert _load_scalar_weight(tmp_path / "checkpoint_auc_best.pt") == pytest.approx(3.0)


def test_auc_record_survives_resume_from_saved_files(tmp_path: Path) -> None:
    _save_auc_sequence(tmp_path, [0.7, 0.9])
    # A resumed run starts from epoch 2 and must compare against the saved 0.9.
    save_top_k_auc_checkpoints(
        model_state={"weight": torch.tensor([2.0])},
        auc=0.85,
        epoch=2,
        save_dir=str(tmp_path),
        k_best_checkpoints=5,
    )

    assert _auc_checkpoint_names(tmp_path) == [
        "checkpoint_auc_epoch_0000_score_0.70000000.pt",
        "checkpoint_auc_epoch_0001_score_0.90000000.pt",
    ]
    assert _load_scalar_weight(tmp_path / "checkpoint_auc_best.pt") == pytest.approx(1.0)


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
    assert _load_scalar_weight(tmp_path / "checkpoint_auc_best.pt") == pytest.approx(2.0)
