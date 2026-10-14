#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026.
#

from __future__ import annotations

import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from projects.esophagus_risk.clinical_data import (
    CLINICAL_FEATURE_NAMES,
    case_id_from_record,
    load_clinical_records,
    normalize_clinical_case_id,
    validate_clinical_matches,
)


def _write_workbook(path: Path, rows: list[list[object]]) -> None:
    shared_strings: list[str] = []

    def cell_xml(column: int, row: int, value: object) -> str:
        letters = ""
        number = column + 1
        while number:
            number, remainder = divmod(number - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        reference = f"{letters}{row}"
        if isinstance(value, str):
            shared_strings.append(value)
            return f'<c r="{reference}" t="s"><v>{len(shared_strings) - 1}</v></c>'
        return f'<c r="{reference}"><v>{value}</v></c>'

    row_xml = []
    for row_number, row in enumerate(rows, start=1):
        cells = "".join(
            cell_xml(column, row_number, value)
            for column, value in enumerate(row)
        )
        row_xml.append(f'<row r="{row_number}">{cells}</row>')

    workbook_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    relationships_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    worksheet_xml = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <sheetData>{}</sheetData>
</worksheet>""".format("".join(row_xml))
    strings_xml = """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">{}</sst>""".format(
        "".join(f"<si><t>{value}</t></si>" for value in shared_strings)
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
        archive.writestr("xl/sharedStrings.xml", strings_xml)


class ClinicalDataTest(TestCase):
    def test_loads_fixed_feature_order_and_preserves_leading_zero(self) -> None:
        headers = ["hrev", "id", *reversed(CLINICAL_FEATURE_NAMES)]
        feature_values = {
            name: float(index + 1)
            for index, name in enumerate(CLINICAL_FEATURE_NAMES)
        }
        row = [
            1,
            " 03093247 ",
            *(feature_values[name] for name in reversed(CLINICAL_FEATURE_NAMES)),
        ]
        with TemporaryDirectory() as directory:
            workbook = Path(directory) / "clinical.xlsx"
            _write_workbook(workbook, [headers, row])
            records = load_clinical_records(workbook)

        record = records["03093247"]
        self.assertEqual(record.label_for_qc, 1)
        self.assertEqual(record.features, tuple(feature_values.values()))

    def test_rejects_duplicate_normalized_ids(self) -> None:
        headers = ["hrev", "id", *CLINICAL_FEATURE_NAMES]
        features = [1.0] * len(CLINICAL_FEATURE_NAMES)
        with TemporaryDirectory() as directory:
            workbook = Path(directory) / "clinical.xlsx"
            _write_workbook(
                workbook,
                [headers, [0, "case-1", *features], [0, " CASE-1 ", *features]],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate clinical case ID"):
                load_clinical_records(workbook)

    def test_rejects_duplicate_required_headers(self) -> None:
        headers = ["hrev", "id", *CLINICAL_FEATURE_NAMES, "age"]
        features = [1.0] * (len(CLINICAL_FEATURE_NAMES) + 1)
        with TemporaryDirectory() as directory:
            workbook = Path(directory) / "clinical.xlsx"
            _write_workbook(workbook, [headers, [1, "case-1", *features]])
            with self.assertRaisesRegex(ValueError, "duplicate columns"):
                load_clinical_records(workbook)

    def test_validates_image_match_and_label(self) -> None:
        headers = ["hrev", "id", *CLINICAL_FEATURE_NAMES]
        features = [1.0] * len(CLINICAL_FEATURE_NAMES)
        with TemporaryDirectory() as directory:
            workbook = Path(directory) / "clinical.xlsx"
            _write_workbook(workbook, [headers, [1, "case-1", *features]])
            records = load_clinical_records(workbook)

        validate_clinical_matches([("HR/case-1", 1)], records)
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            validate_clinical_matches([("NHR/case-1", 0)], records)
        with self.assertRaisesRegex(ValueError, "Missing clinical records"):
            validate_clinical_matches([("HR/case-2", 1)], records)

    def test_normalizes_supported_identifier_types(self) -> None:
        self.assertEqual(normalize_clinical_case_id(123.0), "123")
        self.assertEqual(normalize_clinical_case_id("00123"), "00123")
        self.assertEqual(case_id_from_record("HR/00123"), "00123")
        with self.assertRaises(ValueError):
            normalize_clinical_case_id(123.5)


if __name__ == "__main__":
    main()
