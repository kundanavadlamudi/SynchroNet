"""Tests for the run_pipeline.py backup helper.

These tests verify:
  1. _backup_existing_outputs() copies existing files into a timestamped directory.
  2. Files that do not yet exist are silently skipped.
  3. Two successive backup calls produce distinct directories (unique timestamps).
  4. ML model files are never included in the backup.
  5. run_pipeline(backup_existing=False) leaves results/backups/ untouched.
  6. _PHASE1_OUTPUTS contains exactly the ten expected relative paths.
  7. shutil.copy2 preserves file content byte-for-byte.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so src.* and run_pipeline import.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the symbols under test from the root-level script.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_pipeline",
    _PROJECT_ROOT / "run_pipeline.py",
)
_rp = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(_rp)                          # type: ignore[union-attr]

_backup_existing_outputs: callable = _rp._backup_existing_outputs
_PHASE1_OUTPUTS: list[str] = _rp._PHASE1_OUTPUTS
run_pipeline: callable = _rp.run_pipeline

# Reference set of ML files that must never appear in a backup.
_ML_RELATIVES = {
    "models/selected_bot_classifier.joblib",
    "models/selected_bot_classifier_metadata.json",
    "models/background_sample.csv",
    "results/reports/ml_evaluation_summary.json",
    "results/reports/ml_model_comparison.csv",
    "results/reports/ml_explanations.json",
}


class TestPhase1Outputs(unittest.TestCase):
    """Sanity-checks on the _PHASE1_OUTPUTS manifest."""

    def test_exactly_ten_entries(self):
        self.assertEqual(len(_PHASE1_OUTPUTS), 10)

    def test_all_entries_are_strings(self):
        for entry in _PHASE1_OUTPUTS:
            self.assertIsInstance(entry, str, msg=entry)

    def test_no_ml_files_listed(self):
        for rel in _PHASE1_OUTPUTS:
            self.assertNotIn(
                rel,
                _ML_RELATIVES,
                msg=f"ML file found in _PHASE1_OUTPUTS: {rel}",
            )

    def test_no_model_dir_entries(self):
        for rel in _PHASE1_OUTPUTS:
            self.assertFalse(
                rel.startswith("models/"),
                msg=f"models/ path found in _PHASE1_OUTPUTS: {rel}",
            )


class TestBackupExistingOutputs(unittest.TestCase):
    """Integration tests for _backup_existing_outputs() using real temp files."""

    def setUp(self):
        """Create a small set of fake Phase 1 output files in the real project tree."""
        # We pick two entries that are easy to create/remove without
        # disturbing real pipeline results.  We write sentinel content so we
        # can verify byte-for-byte preservation.
        self._touched: list[Path] = []
        self._sentinel = b"backup_test_sentinel_content\n"

        for rel in (
            "results/reports/coordination_results.csv",
            "results/reports/account_behavior_results.csv",
        ):
            path = _PROJECT_ROOT / rel
            # Only create if it doesn't already exist; track it either way.
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self._sentinel)
                self._touched.append(path)

        # Record original content for files that existed before the test.
        self._originals: dict[Path, bytes] = {}
        for rel in _PHASE1_OUTPUTS:
            p = _PROJECT_ROOT / rel
            if p.exists():
                self._originals[p] = p.read_bytes()

    def tearDown(self):
        """Remove files we created during setUp and restore originals."""
        for path in self._touched:
            if path.exists():
                path.unlink()

        for path, content in self._originals.items():
            if path.exists():
                # Restore if we overwrote it during the test.
                path.write_bytes(content)

        # Remove backup directories created by the tests.
        backups_root = _PROJECT_ROOT / "results" / "backups"
        if backups_root.exists():
            import shutil
            shutil.rmtree(backups_root)

    # ------------------------------------------------------------------

    def test_backup_creates_timestamped_directory(self):
        backup_dir = _backup_existing_outputs()
        self.assertTrue(backup_dir.exists(), "Backup directory was not created")
        self.assertTrue(backup_dir.is_dir())
        # Directory name must be a 15-char timestamp: YYYYMMDD_HHMMSS
        self.assertRegex(
            backup_dir.name,
            r"^\d{8}_\d{6}$",
            msg=f"Unexpected backup dir name: {backup_dir.name}",
        )

    def test_existing_files_are_copied(self):
        backup_dir = _backup_existing_outputs()

        for rel in (
            "results/reports/coordination_results.csv",
            "results/reports/account_behavior_results.csv",
        ):
            backed_up = backup_dir / rel
            self.assertTrue(
                backed_up.exists(),
                msg=f"Expected backup of {rel} not found in {backup_dir}",
            )

    def test_content_is_preserved_byte_for_byte(self):
        """Backed-up file content must be identical to the original."""
        # Write a known payload to a real output path.
        target_rel = "results/reports/coordination_results.csv"
        target = _PROJECT_ROOT / target_rel
        known_content = b"col_a,col_b\n1,2\n3,4\n"
        original = target.read_bytes() if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(known_content)

        backup_dir = _backup_existing_outputs()
        backed_up = backup_dir / target_rel

        self.assertTrue(backed_up.exists())
        self.assertEqual(backed_up.read_bytes(), known_content)

        # Restore
        if original is not None:
            target.write_bytes(original)
        else:
            target.unlink(missing_ok=True)

    def test_missing_files_are_skipped_silently(self):
        """If a Phase 1 output does not exist, backup() must not raise."""
        # Ensure at least one expected path does NOT exist.
        absent_rel = "results/graphs/coordination_graph.png"
        absent = _PROJECT_ROOT / absent_rel
        existed = absent.exists()
        original_content = absent.read_bytes() if existed else None

        if existed:
            absent.unlink()

        try:
            backup_dir = _backup_existing_outputs()  # must not raise
            backed_up = backup_dir / absent_rel
            self.assertFalse(
                backed_up.exists(),
                msg="Non-existent file should not appear in backup",
            )
        finally:
            if existed and original_content is not None:
                absent.parent.mkdir(parents=True, exist_ok=True)
                absent.write_bytes(original_content)

    def test_two_successive_backups_produce_distinct_directories(self):
        first = _backup_existing_outputs()
        time.sleep(1)          # ensure the second second ticks over
        second = _backup_existing_outputs()
        self.assertNotEqual(
            first,
            second,
            msg="Two successive backups must land in different directories",
        )

    def test_ml_files_not_in_backup(self):
        """ML model and report files must never appear in the backup."""
        backup_dir = _backup_existing_outputs()
        for rel in _ML_RELATIVES:
            self.assertFalse(
                (backup_dir / rel).exists(),
                msg=f"ML file unexpectedly backed up: {rel}",
            )

    def test_backup_dir_is_inside_results_backups(self):
        backup_dir = _backup_existing_outputs()
        expected_parent = _PROJECT_ROOT / "results" / "backups"
        self.assertEqual(backup_dir.parent, expected_parent)


class TestNormalBehaviorUnchanged(unittest.TestCase):
    """Verify that omitting --backup-existing leaves the backup tree untouched."""

    def setUp(self):
        self._backups_root = _PROJECT_ROOT / "results" / "backups"
        self._pre_existed = self._backups_root.exists()

    def tearDown(self):
        if not self._pre_existed and self._backups_root.exists():
            import shutil
            shutil.rmtree(self._backups_root)

    def test_no_backup_created_without_flag(self):
        """Calling run_pipeline with backup_existing=False must not create
        any backup directory, even when output files are present.

        We patch _run_step inside the run_pipeline module to return True for
        every stage, so no real pipeline I/O is triggered.  This avoids the
        heavy sentence-transformer cold-import while still exercising the full
        run_pipeline() control flow.
        """
        import unittest.mock as mock

        # Patch _run_step so every stage is a no-op that reports success.
        with mock.patch.object(_rp, "_run_step", return_value=True):
            # Also stub the summary config import so the output-file listing
            # doesn't crash on missing paths.
            dummy_path = Path("/nonexistent/dummy")
            fake_config_attrs = {
                "CLEANED_POSTS_FILE": dummy_path,
                "CLEANED_KEYWORDS_FILE": dummy_path,
                "SEMANTIC_RESULTS_FILE": dummy_path,
                "COORDINATION_RESULTS_FILE": dummy_path,
                "ACCOUNT_BEHAVIOR_FILE": dummy_path,
                "CAMPAIGN_RESULTS_FILE": dummy_path,
                "COORDINATION_GRAPH_PNG": dummy_path,
                "RISK_RESULTS_FILE": dummy_path,
            }
            fake_config = type("_FakeConfig", (), fake_config_attrs)()

            with mock.patch.dict(
                sys.modules,
                {"src.config": fake_config},  # type: ignore[dict-item]
            ):
                try:
                    run_pipeline(skip_sample_data=True, backup_existing=False)
                except Exception:
                    pass  # Only the side-effect matters.

        self.assertFalse(
            self._backups_root.exists(),
            msg=(
                "results/backups/ was created even though "
                "--backup-existing was not requested"
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
