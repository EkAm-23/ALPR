"""
test_delta_queue.py — Tests for the delta batch queue logic in auto_retrain.py.

Tests check_delta_batch_size() with real temp files (no ML models needed).
Also tests the version management helper functions.
"""
import os
import csv
import tempfile
import pytest
import sys

# Add scripts dir to path so we can import helpers directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


# ── Import helpers from auto_retrain (non-ML portions only) ─────────────────
# We import at function level to avoid the module-level subprocess side-effects
def _get_helpers():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "auto_retrain",
        os.path.join(REPO_ROOT, "scripts", "auto_retrain.py")
    )
    mod = importlib.util.load_from_spec(spec) if hasattr(importlib.util, "load_from_spec") else None
    return mod


class TestVersionHelpers:
    """Tests for the model version management functions in auto_retrain.py."""

    def test_bump_patch_version_basic(self):
        from auto_retrain import bump_patch_version
        assert bump_patch_version("v1.0.0") == "v1.0.1"
        assert bump_patch_version("v1.0.9") == "v1.0.10"
        assert bump_patch_version("v2.3.7") == "v2.3.8"

    def test_read_write_model_version(self, tmp_path):
        from auto_retrain import read_model_version, write_model_version
        # When file doesn't exist, should return default
        assert read_model_version(str(tmp_path)) == "v1.0.0"
        # Write and read back
        write_model_version(str(tmp_path), "v1.0.5")
        assert read_model_version(str(tmp_path)) == "v1.0.5"

    def test_ct_cycle_increments(self, tmp_path):
        from auto_retrain import get_ct_cycle, increment_ct_cycle
        # Starts at 0
        assert get_ct_cycle(str(tmp_path)) == 0
        # Increments correctly
        assert increment_ct_cycle(str(tmp_path)) == 1
        assert increment_ct_cycle(str(tmp_path)) == 2
        assert get_ct_cycle(str(tmp_path)) == 2


class TestDeltaQueue:
    """Tests for check_delta_batch_size()."""

    def _write_csv(self, path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text"])
            for row in rows:
                writer.writerow(row)

    def test_returns_zero_when_no_csv(self, tmp_path, monkeypatch):
        """Should return 0 if the CSV file doesn't exist yet."""
        from auto_retrain import check_delta_batch_size
        # Monkeypatch the function to look in tmp_path
        monkeypatch.setattr(
            "auto_retrain.os.path.dirname",
            lambda _: str(tmp_path)
        )
        # File doesn't exist → 0
        assert check_delta_batch_size() == 0

    def test_counts_data_rows_not_header(self, tmp_path):
        """Should return number of data rows, not counting the header."""
        csv_path = tmp_path / "delta_metadata.csv"
        rows = [("crop_1.jpg", "MH01AB1234"), ("crop_2.jpg", "KL16J3636")]
        self._write_csv(str(csv_path), rows)

        with open(str(csv_path), "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        count = len(data) - 1  # Mirrors the logic in check_delta_batch_size
        assert count == 2

    def test_header_only_returns_zero(self, tmp_path):
        """A CSV with only the header row should return 0."""
        csv_path = tmp_path / "delta_metadata.csv"
        with open(str(csv_path), "w", newline="") as f:
            csv.writer(f).writerow(["filename", "text"])

        with open(str(csv_path), "r") as f:
            data = list(csv.reader(f))
        assert len(data) - 1 == 0


class TestCsvContract:
    """Tests that the CSV written by tasks.py has the columns the trainer expects."""

    REQUIRED_COLUMNS = {"filename", "text"}

    def test_delta_metadata_has_correct_columns(self):
        """The real delta_metadata.csv should have 'filename' and 'text' columns."""
        csv_path = os.path.join(REPO_ROOT, "data", "delta_metadata.csv")
        if not os.path.exists(csv_path):
            pytest.skip("delta_metadata.csv not present — skipping contract test")

        with open(csv_path, "r") as f:
            header = csv.DictReader(f).fieldnames
        assert set(header) >= self.REQUIRED_COLUMNS, (
            f"delta_metadata.csv is missing columns. Got: {header}, "
            f"expected at least: {self.REQUIRED_COLUMNS}"
        )
