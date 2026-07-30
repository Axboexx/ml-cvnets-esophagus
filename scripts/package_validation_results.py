#!/usr/bin/env python3
"""Package F1/F2a/F2b runs and DeLong outputs without checkpoints."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveEntry:
    """A source file and its destination path inside the ZIP archive."""

    source: Path
    archive_path: Path
    device: int
    inode: int
    size: int
    modified_time_ns: int
    mode: int


def _expand_path(value: str) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    return Path(os.path.abspath(expanded))


def _relative_to(path: Path, root: Path, description: str) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{description} must be inside results root {root}: {path}"
        ) from error


def _validate_directories(
    results_root: Path,
    input_directories: Sequence[Path],
) -> None:
    if not results_root.is_dir():
        raise NotADirectoryError(f"results root not found: {results_root}")

    for directory in input_directories:
        if not directory.is_dir():
            raise NotADirectoryError(f"input directory not found: {directory}")
        _relative_to(directory, results_root, "input directory")

    if len(set(input_directories)) != len(input_directories):
        raise ValueError("input directories must be distinct")

    for index, first in enumerate(input_directories):
        for second in input_directories[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError(
                    "input directories must not contain one another: "
                    f"{first}, {second}"
                )


def collect_archive_entries(
    results_root: Path,
    input_directories: Sequence[Path],
) -> tuple[list[ArchiveEntry], list[Path]]:
    """Collect files while preserving paths relative to the results root."""
    resolved_root = results_root.expanduser().resolve()
    expanded_inputs = [
        directory.expanduser().absolute() for directory in input_directories
    ]
    for directory in expanded_inputs:
        if directory.is_symlink():
            raise ValueError(f"symbolic links are not supported: {directory}")
    resolved_inputs = [directory.resolve() for directory in expanded_inputs]
    _validate_directories(resolved_root, resolved_inputs)

    entries = []
    excluded_checkpoints = []
    seen_archive_paths = set()
    for directory in resolved_inputs:
        for path in sorted(directory.rglob("*")):
            path_stat = path.lstat()
            if stat.S_ISLNK(path_stat.st_mode):
                raise ValueError(f"symbolic links are not supported: {path}")
            if not stat.S_ISREG(path_stat.st_mode):
                continue
            if "checkpoint" in path.name.lower():
                excluded_checkpoints.append(path)
                continue

            archive_path = _relative_to(path, resolved_root, "input file")
            if archive_path in seen_archive_paths:
                raise ValueError(f"duplicate ZIP path: {archive_path}")
            seen_archive_paths.add(archive_path)
            entries.append(
                ArchiveEntry(
                    source=path,
                    archive_path=archive_path,
                    device=path_stat.st_dev,
                    inode=path_stat.st_ino,
                    size=path_stat.st_size,
                    modified_time_ns=path_stat.st_mtime_ns,
                    mode=path_stat.st_mode,
                )
            )

    if not entries:
        raise ValueError("no non-checkpoint files found to package")
    return entries, excluded_checkpoints


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def _write_archive_entry(
    archive: zipfile.ZipFile,
    entry: ArchiveEntry,
) -> None:
    expected_identity = (
        entry.device,
        entry.inode,
        entry.size,
        entry.modified_time_ns,
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(entry.source, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if _file_identity(opened_stat) != expected_identity:
            raise ValueError(f"input file changed during packaging: {entry.source}")

        archive_info = zipfile.ZipInfo(entry.archive_path.as_posix())
        archive_info.compress_type = zipfile.ZIP_DEFLATED
        archive_info.external_attr = (stat.S_IMODE(entry.mode) & 0xFFFF) << 16
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            with archive.open(archive_info, mode="w", force_zip64=True) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            final_stat = os.fstat(source.fileno())

        if _file_identity(final_stat) != expected_identity:
            raise ValueError(f"input file changed during packaging: {entry.source}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def create_archive(
    results_root: Path,
    input_directories: Sequence[Path],
    output_path: Path,
) -> tuple[int, int]:
    """Create a ZIP atomically and return included and excluded file counts."""
    resolved_output = output_path.expanduser().resolve()
    if resolved_output.exists():
        raise FileExistsError(f"output ZIP already exists: {resolved_output}")

    expanded_inputs = [
        directory.expanduser().absolute() for directory in input_directories
    ]
    for directory in expanded_inputs:
        if directory.is_symlink():
            raise ValueError(f"symbolic links are not supported: {directory}")
    resolved_inputs = [directory.resolve() for directory in expanded_inputs]
    for directory in resolved_inputs:
        if directory == resolved_output or directory in resolved_output.parents:
            raise ValueError(
                f"output ZIP must not be inside an input directory: {directory}"
            )

    entries, excluded_checkpoints = collect_archive_entries(
        results_root=results_root,
        input_directories=expanded_inputs,
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{resolved_output.name}.",
            suffix=".tmp",
            dir=resolved_output.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for entry in entries:
                _write_archive_entry(archive, entry)

        os.link(temporary_path, resolved_output)
        published = True
        try:
            temporary_path.unlink()
        except OSError as error:
            LOGGER.warning(
                "could not remove temporary ZIP link %s: %s", temporary_path, error
            )
    except Exception:
        if temporary_path is not None and not published:
            temporary_path.unlink(missing_ok=True)
        raise

    return len(entries), len(excluded_checkpoints)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package F1, F2a, F2b and paired DeLong result directories while "
            "excluding files whose names contain 'checkpoint'."
        )
    )
    parser.add_argument(
        "--results-root",
        default="$HOME/work/results",
        help="Root used to preserve ZIP paths (default: $HOME/work/results).",
    )
    parser.add_argument(
        "--f1-dir",
        required=True,
        help="F1 results directory containing all F1 experiment folders.",
    )
    parser.add_argument(
        "--f2a-dir",
        required=True,
        help="F2a results directory containing all F2a experiment folders.",
    )
    parser.add_argument(
        "--f2b-dir",
        required=True,
        help="F2b results directory containing all F2b experiment folders.",
    )
    parser.add_argument(
        "--delong-dir",
        required=True,
        help="Directory produced by the paired DeLong analysis.",
    )
    parser.add_argument("--output", required=True, help="Destination .zip path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_path = _expand_path(arguments.output)
    if output_path.suffix.lower() != ".zip":
        parser.error("--output must end with .zip")

    try:
        included_count, excluded_count = create_archive(
            results_root=_expand_path(arguments.results_root),
            input_directories=[
                _expand_path(arguments.f1_dir),
                _expand_path(arguments.f2a_dir),
                _expand_path(arguments.f2b_dir),
                _expand_path(arguments.delong_dir),
            ],
            output_path=output_path,
        )
    except (FileExistsError, NotADirectoryError, OSError, ValueError) as error:
        parser.error(str(error))

    LOGGER.info("created ZIP: %s", output_path)
    LOGGER.info("included files: %d", included_count)
    LOGGER.info("excluded checkpoint files: %d", excluded_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
