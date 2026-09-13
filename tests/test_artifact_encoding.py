"""Committed text artifacts must be readable, and the code must not gamble on a locale.

`results/classification_report.txt` was committed carrying byte 0x97, a cp1252 em dash,
which makes the file invalid UTF-8: reading it with `encoding="utf-8"` raises. It got
there because `src/pipeline.py` opened the file with `open(path, "w")` and no encoding, so
Python used the platform default, cp1252 on Windows and UTF-8 on Linux. The same run
produced different bytes on different machines.

The header that contained it has since been changed to plain ASCII, but the artifact was
never regenerated, so the corrupted file stayed in the repository. Nothing noticed:
`src/pipeline.py` writes nine of the committed artifacts and no CI job runs it.

Two tests, for the two halves of that. The artifacts must decode, and no module may open a
text file without saying in what encoding.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Text artifacts that are committed and therefore read by whoever clones this.
TEXT_ARTIFACTS = (
    "results/classification_report.txt",
    "results/benchmark.json",
    "results/model_results.json",
    "results/summary/dataset_summary.json",
    "results/summary/feature_summary.json",
    "results/summary/model_metrics.json",
    "results/summary/error_summary.json",
)


class CommittedArtifactsDecode(unittest.TestCase):
    def test_every_text_artifact_is_valid_utf8(self):
        for name in TEXT_ARTIFACTS:
            with self.subTest(artifact=name):
                path = ROOT / name
                self.assertTrue(path.is_file(), f"{name} is missing")
                try:
                    path.read_text(encoding="utf-8")
                except UnicodeDecodeError as error:
                    self.fail(f"{name} is not valid UTF-8: {error}")

    def test_every_json_artifact_parses(self):
        for name in TEXT_ARTIFACTS:
            if not name.endswith(".json"):
                continue
            with self.subTest(artifact=name):
                json.loads((ROOT / name).read_text(encoding="utf-8"))

    def test_the_classification_report_header_is_what_the_code_writes(self):
        """The stale copy said "Classification Report" with an em dash before it."""
        report = (ROOT / "results/classification_report.txt").read_text(encoding="utf-8")
        first = report.splitlines()[0]
        self.assertEqual(first, "XGBoost (Tuned) classification report")
        self.assertTrue(first.isascii(), f"non-ASCII in the header: {first!r}")

    def test_no_committed_artifact_carries_a_stray_control_byte(self):
        for name in TEXT_ARTIFACTS:
            with self.subTest(artifact=name):
                raw = (ROOT / name).read_bytes()
                stray = sorted({b for b in raw if b > 127})
                self.assertEqual(stray, [], f"{name} has non-ASCII bytes {stray}")


class TextIsWrittenWithAnExplicitEncoding(unittest.TestCase):
    """A locale-dependent write is how the corrupted byte arrived in the first place."""

    def modules(self):
        for directory in ("src", "scripts", "tests"):
            yield from sorted((ROOT / directory).rglob("*.py"))

    def test_no_text_open_without_an_encoding(self):
        offenders = []
        for path in self.modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "id", None) != "open":
                    continue
                mode = ""
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                if "b" in mode:
                    continue                      # binary needs no encoding
                if "encoding" in {keyword.arg for keyword in node.keywords}:
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], f"text opened without an encoding: {offenders}")

    def test_no_pathlib_text_io_without_an_encoding(self):
        offenders = []
        for path in self.modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "attr", None) not in {"read_text", "write_text"}:
                    continue
                if "encoding" in {keyword.arg for keyword in node.keywords}:
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], f"pathlib text IO without an encoding: {offenders}")
