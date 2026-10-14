#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

"""Small, dependency-free reader for the F4 clinical XLSX workbooks."""

from __future__ import annotations

import math
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from xml.etree import ElementTree


CLINICAL_FEATURE_NAMES = (
    "sex",
    "age",
    "hct",
    "hb",
    "plt",
    "pt",
    "inr",
    "ast",
    "alt",
    "alb",
    "tb",
    "tp",
    "cr",
    "child",
    "hepatitis",
)
_ID_COLUMN = "id"
_LABEL_COLUMN = "hrev"
_CELL_REFERENCE = re.compile(r"^([A-Za-z]+)[0-9]+$")
_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass(frozen=True)
class ClinicalRecord:
    """Validated clinical values for one case."""

    case_id: str
    label_for_qc: int
    features: tuple[float, ...]


def normalize_clinical_case_id(value: object) -> str:
    """Normalize an Excel or directory identifier without losing leading zeroes."""
    if value is None or isinstance(value, bool):
        raise ValueError("Clinical case ID must be a non-empty string or integer.")
    if isinstance(value, int):
        normalized = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"Clinical case ID must be integer-like, got {value!r}.")
        normalized = str(int(value))
    else:
        normalized = str(value).strip()
        if normalized.endswith(".0") and normalized[:-2].isdigit():
            normalized = normalized[:-2]

    if not normalized:
        raise ValueError("Clinical case ID must not be empty.")
    return normalized.casefold()


def case_id_from_record(case_id: str) -> str:
    """Return the clinical identifier from a ``class/case`` dataset ID."""
    basename = str(case_id).replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    lower_basename = basename.casefold()
    for suffix in (".nii.gz", ".nii"):
        if lower_basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            break
    if basename.casefold().endswith("s"):
        basename = basename[:-1]
    return normalize_clinical_case_id(basename)


def load_clinical_records(
    workbook_path: str | Path,
    feature_names: Sequence[str] = CLINICAL_FEATURE_NAMES,
) -> dict[str, ClinicalRecord]:
    """Load and validate the first worksheet of an F4 workbook."""
    workbook = Path(workbook_path).expanduser()
    if not workbook.is_file():
        raise FileNotFoundError(f"Clinical workbook does not exist: {workbook}")

    rows = _read_first_worksheet(workbook)
    if not rows:
        raise ValueError(f"Clinical workbook is empty: {workbook}")

    headers = [str(value).strip().casefold() if value is not None else "" for value in rows[0]]
    required_columns = (_ID_COLUMN, _LABEL_COLUMN, *feature_names)
    missing_columns = [name for name in required_columns if name not in headers]
    if missing_columns:
        raise ValueError(
            f"Clinical workbook is missing columns {missing_columns}: {workbook}"
        )
    duplicate_columns = [name for name in required_columns if headers.count(name) > 1]
    if duplicate_columns:
        raise ValueError(
            f"Clinical workbook has duplicate columns {duplicate_columns}: {workbook}"
        )
    column_index = {name: headers.index(name) for name in required_columns}

    records: dict[str, ClinicalRecord] = {}
    for row_number, row in enumerate(rows[1:], start=2):
        if not any(value not in (None, "") for value in row):
            continue
        case_id = normalize_clinical_case_id(_row_value(row, column_index[_ID_COLUMN]))
        if case_id in records:
            raise ValueError(
                f"Duplicate clinical case ID {case_id!r} at row {row_number}: {workbook}"
            )

        label_value = _to_finite_float(
            _row_value(row, column_index[_LABEL_COLUMN]),
            column=_LABEL_COLUMN,
            row_number=row_number,
        )
        if label_value not in (0.0, 1.0):
            raise ValueError(
                f"Column {_LABEL_COLUMN!r} must be 0 or 1 at row {row_number}."
            )
        features = tuple(
            _to_finite_float(
                _row_value(row, column_index[name]),
                column=name,
                row_number=row_number,
            )
            for name in feature_names
        )
        records[case_id] = ClinicalRecord(
            case_id=case_id,
            label_for_qc=int(label_value),
            features=features,
        )
    if not records:
        raise ValueError(f"Clinical workbook has no data rows: {workbook}")
    return records


def validate_clinical_matches(
    case_ids_and_labels: Sequence[tuple[str, int]],
    clinical_records: Mapping[str, ClinicalRecord],
) -> None:
    """Fail fast when an eligible image case is missing or label-inconsistent."""
    missing: list[str] = []
    mismatched: list[str] = []
    seen: set[str] = set()
    for case_id, image_label in case_ids_and_labels:
        normalized_id = case_id_from_record(case_id)
        if normalized_id in seen:
            raise ValueError(f"Duplicate normalized image case ID: {normalized_id!r}")
        seen.add(normalized_id)
        clinical_record = clinical_records.get(normalized_id)
        if clinical_record is None:
            missing.append(case_id)
        elif clinical_record.label_for_qc != int(image_label):
            mismatched.append(
                f"{case_id}: image={image_label}, clinical={clinical_record.label_for_qc}"
            )
    if missing:
        raise ValueError(
            f"Missing clinical records for {len(missing)} eligible image cases. "
            f"Examples: {', '.join(missing[:5])}"
        )
    if mismatched:
        raise ValueError(
            f"Clinical/image label mismatch for {len(mismatched)} cases. "
            f"Examples: {', '.join(mismatched[:5])}"
        )


def _row_value(row: Sequence[object], index: int) -> object:
    return row[index] if index < len(row) else None


def _to_finite_float(value: object, *, column: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Column {column!r} must be numeric at row {row_number}, got {value!r}."
        ) from error
    if not math.isfinite(number):
        raise ValueError(
            f"Column {column!r} must be finite at row {row_number}, got {value!r}."
        )
    return number


def _column_index(cell_reference: str) -> int:
    match = _CELL_REFERENCE.match(cell_reference)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference!r}")
    index = 0
    for character in match.group(1).upper():
        index = (index * 26) + ord(character) - ord("A") + 1
    return index - 1


def _read_first_worksheet(workbook: Path) -> list[list[object]]:
    try:
        with zipfile.ZipFile(workbook) as archive:
            worksheet_path = _first_worksheet_path(archive)
            shared_strings = _shared_strings(archive)
            worksheet_root = ElementTree.fromstring(archive.read(worksheet_path))
    except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as error:
        raise ValueError(f"Unable to read XLSX workbook {workbook}: {error}") from error

    namespace = {"s": _SPREADSHEET_NS}
    rows: list[list[object]] = []
    for row_element in worksheet_root.findall(".//s:sheetData/s:row", namespace):
        values: list[object] = []
        for cell in row_element.findall("s:c", namespace):
            index = _column_index(cell.attrib["r"])
            while len(values) <= index:
                values.append(None)
            values[index] = _cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def _first_worksheet_path(archive: zipfile.ZipFile) -> str:
    workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    namespace = {"s": _SPREADSHEET_NS}
    sheet = workbook_root.find(".//s:sheets/s:sheet", namespace)
    if sheet is None:
        raise ValueError("XLSX workbook contains no worksheets.")
    relationship_id = sheet.attrib[f"{{{_RELATIONSHIP_NS}}}id"]

    relationships_root = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relationship in relationships_root:
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            if target.startswith("xl/"):
                return posixpath.normpath(target)
            return posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(f"Unable to resolve worksheet relationship {relationship_id!r}.")


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    namespace = {"s": _SPREADSHEET_NS}
    return [
        "".join(text.text or "" for text in item.findall(".//s:t", namespace))
        for item in root.findall("s:si", namespace)
    ]


def _cell_value(cell: ElementTree.Element, shared_strings: Sequence[str]) -> object:
    namespace = {"s": _SPREADSHEET_NS}
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(
            text.text or "" for text in cell.findall(".//s:is/s:t", namespace)
        )
    value_element = cell.find("s:v", namespace)
    if value_element is None or value_element.text is None:
        return None
    value = value_element.text
    if cell_type == "s":
        return shared_strings[int(value)]
    if cell_type in {"str", "e"}:
        return value
    if cell_type == "b":
        return value == "1"
    number = float(value)
    return int(number) if number.is_integer() else number
