from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_NEGATIVE_LABEL_NAMES = {"nohr", "no-hr", "nonhr", "non-hr", "nhr"}
_POSITIVE_LABEL_NAMES = {"hr"}
_MASK_KEYWORDS = ("mask", "seg", "label")
_NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass(frozen=True)
class CaseInput:
    dataset: str
    case_id: str
    label: int | None
    ct_path: Path | None
    mask_path: Path | None
    discovery_error: str = ""


def _class_label(name: str) -> int | None:
    normalized = name.strip().lower().replace(" ", "").replace("_", "-")
    if normalized in _POSITIVE_LABEL_NAMES:
        return 1
    if normalized in _NEGATIVE_LABEL_NAMES:
        return 0
    return None


def _is_nifti_file(path: Path) -> bool:
    return path.is_file() and str(path).lower().endswith(_NIFTI_SUFFIXES)


def _contains_keyword(path: Path, keywords: Sequence[str]) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    return any(keyword in name or keyword in stem for keyword in keywords)


def _prefer_shorter_path(paths: Iterable[Path]) -> Path:
    return sorted(paths, key=lambda path: (len(path.parts), len(path.name), str(path)))[0]


def _resolve_tongji_case_files(case_dir: Path) -> tuple[Path, Path]:
    nifti_files = sorted(path for path in case_dir.rglob("*") if _is_nifti_file(path))
    if len(nifti_files) < 2:
        raise ValueError(f"Expected at least 2 NIfTI files, got {len(nifti_files)}")

    mask_candidates = [path for path in nifti_files if _contains_keyword(path, _MASK_KEYWORDS)]
    if not mask_candidates:
        raise ValueError("Unable to identify mask file")
    mask_path = _prefer_shorter_path(mask_candidates)
    ct_candidates = [
        path
        for path in nifti_files
        if path != mask_path and not _contains_keyword(path, _MASK_KEYWORDS)
    ]
    ct_candidates = ct_candidates or [path for path in nifti_files if path != mask_path]
    if not ct_candidates:
        raise ValueError("Unable to identify CT file")
    return _prefer_shorter_path(ct_candidates), mask_path


def discover_tongji_cases(root: Path) -> list[CaseInput]:
    cases: list[CaseInput] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = _class_label(class_dir.name)
        if label is None:
            continue
        for case_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            case_id = f"{class_dir.name}/{case_dir.name}"
            try:
                ct_path, mask_path = _resolve_tongji_case_files(case_dir)
                cases.append(CaseInput("tongji", case_id, label, ct_path, mask_path))
            except (OSError, ValueError) as error:
                cases.append(CaseInput("tongji", case_id, label, None, None, str(error)))
    return cases


def _find_matching_dir(parent: Path, name: str) -> Path | None:
    exact_match = parent / name
    if exact_match.is_dir():
        return exact_match
    normalized_name = name.strip().lower()
    return next(
        (
            child
            for child in sorted(path for path in parent.iterdir() if path.is_dir())
            if child.name.strip().lower() == normalized_name
        ),
        None,
    )


def _resolve_single_nifti(case_dir: Path, *, prefer_mask: bool) -> Path:
    nifti_files = sorted(path for path in case_dir.rglob("*") if _is_nifti_file(path))
    if not nifti_files:
        raise ValueError("No NIfTI file found")
    preferred = [
        path
        for path in nifti_files
        if _contains_keyword(path, _MASK_KEYWORDS) == prefer_mask
    ]
    return _prefer_shorter_path(preferred or nifti_files)


def discover_youyi_cases(root: Path, nii_dir_name: str, mask_dir_name: str) -> list[CaseInput]:
    nii_root = root / nii_dir_name
    mask_root = root / mask_dir_name
    if not nii_root.is_dir():
        raise FileNotFoundError(f"Youyi CT root does not exist: {nii_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Youyi mask root does not exist: {mask_root}")

    cases: list[CaseInput] = []
    for class_dir in sorted(path for path in nii_root.iterdir() if path.is_dir()):
        label = _class_label(class_dir.name)
        if label is None:
            continue
        mask_class_dir = _find_matching_dir(mask_root, class_dir.name)
        for case_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            case_id = f"{class_dir.name}/{case_dir.name}"
            try:
                if mask_class_dir is None:
                    raise FileNotFoundError(f"Missing mask class directory for {class_dir.name}")
                mask_case_dir = _find_matching_dir(mask_class_dir, case_dir.name)
                if mask_case_dir is None:
                    raise FileNotFoundError("Missing matching mask case directory")
                cases.append(
                    CaseInput(
                        "youyi",
                        case_id,
                        label,
                        _resolve_single_nifti(case_dir, prefer_mask=False),
                        _resolve_single_nifti(mask_case_dir, prefer_mask=True),
                    )
                )
            except (OSError, ValueError) as error:
                cases.append(CaseInput("youyi", case_id, label, None, None, str(error)))
    return cases
