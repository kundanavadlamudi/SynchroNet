"""NexTrace — Phase 1 coordination-analysis pipeline orchestrator.

Runs the full Phase 1 pipeline in the correct dependency order:

    1. create_sample_data    — generate synthetic posts (if needed)
    2. data_preprocessing    — clean and validate posts
    3. keyword_loader        — process hashtag/keyword files
    4. semantic_similarity   — compute pairwise text embeddings
    5. coordination_detection — flag coordinated account pairs
    6. account_analysis      — summarise per-account behaviour
    7. campaign_analysis     — group connected accounts into campaigns
    8. graph_analysis        — build and render the coordination graph
    9. risk_scoring          — assign investigation-priority scores

Usage
-----
    # From the project root with the activated virtual environment:
    python run_pipeline.py

    # Skip the sample-data generation step if you already have data:
    python run_pipeline.py --skip-sample-data

    # Back up all existing output files before overwriting them:
    python run_pipeline.py --backup-existing

    # Combine flags:
    python run_pipeline.py --skip-sample-data --backup-existing

    # Show available options:
    python run_pipeline.py --help

Notes
-----
* This script does NOT retrain the ML model. Bot classification is a
  separate Phase 2 pipeline (see run_ml_pipeline.py).
* Each step is run as an in-process function call, not a subprocess, so
  Python import paths work correctly.
* If a required input file is missing the step will print a clear error
  message and the pipeline will stop.
* When --backup-existing is supplied, any Phase 1 output file that
  already exists is copied to results/backups/<YYYYMMDD_HHMMSS>/ before
  being overwritten. The ML model and ML report files are never touched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path so ``src.*`` imports resolve
# regardless of where the user launches the script from.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Backup helper
# ---------------------------------------------------------------------------

# All Phase 1 output files that the pipeline can overwrite.
# ML model/report files are deliberately excluded.
_PHASE1_OUTPUTS = [
    # Stage 1 — sample data
    "data/processed/sample_social_media_posts.csv",
    # Stage 2 — preprocessing
    "data/processed/cleaned_social_media_posts.csv",
    # Stage 3 — keyword loader
    "data/processed/cleaned_keywords.csv",
    # Stage 4 — semantic similarity
    "results/reports/semantic_similarity_results.csv",
    # Stage 5 — coordination detection
    "results/reports/coordination_results.csv",
    # Stage 6 — account analysis
    "results/reports/account_behavior_results.csv",
    # Stage 7 — campaign analysis
    "results/reports/campaign_analysis_results.csv",
    # Stage 8 — graph analysis
    "results/graphs/coordination_graph.png",
    "results/graphs/coordination_graph.graphml",
    # Stage 9 — risk scoring
    "results/reports/explainable_risk_results.csv",
]


def _backup_existing_outputs() -> Path:
    """Copy every existing Phase 1 output file into a timestamped backup
    directory and return the directory path.

    Backup location::

        results/backups/<YYYYMMDD_HHMMSS>/

    Each file is stored under the same relative sub-path it holds inside
    the project root, so the backup directory is self-contained and easy
    to restore from.

    Only files that *already exist* are copied; missing files are skipped
    silently.  The ML model and ML report files are never included.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = _PROJECT_ROOT / "results" / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    backed_up: list[str] = []
    for rel in _PHASE1_OUTPUTS:
        src = _PROJECT_ROOT / rel
        if src.exists():
            dest = backup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            backed_up.append(rel)

    print(f"\n  Backup directory : {backup_dir.relative_to(_PROJECT_ROOT)}")
    if backed_up:
        print(f"  Files backed up  : {len(backed_up)}")
        for rel in backed_up:
            print(f"    {rel}")
    else:
        print("  No existing output files found — nothing to back up.")

    return backup_dir


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def _run_step(step_name: str, fn) -> bool:
    """Execute *fn*, print timing, and return True on success."""
    print(f"\n{'='*60}")
    print(f"  Step: {step_name}")
    print(f"{'='*60}")
    start = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - start
        print(f"\n  [OK] {step_name} completed in {elapsed:.1f}s")
        return True
    except SystemExit as exc:
        # Some scripts call sys.exit(); treat non-zero as failure.
        if exc.code not in (None, 0):
            print(f"\n  [FAIL] {step_name} exited with code {exc.code}")
            return False
        return True
    except Exception:
        elapsed = time.perf_counter() - start
        print(f"\n  [FAIL] {step_name} failed after {elapsed:.1f}s")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    skip_sample_data: bool = False,
    backup_existing: bool = False,
) -> int:
    """Run all Phase 1 steps in order.  Returns 0 on success, 1 on failure."""

    total_start = time.perf_counter()
    print("\nNexTrace — Phase 1 Coordination Analysis Pipeline")
    print("=" * 60)
    print("This pipeline does NOT retrain the ML bot classifier.")
    print("Run  run_ml_pipeline.py  for Phase 2 (bot classification).")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Optional backup of existing output files
    # ------------------------------------------------------------------
    if backup_existing:
        print("\n  --backup-existing: copying existing output files …")
        _backup_existing_outputs()

    # ------------------------------------------------------------------
    # Step 1 — Sample data generation
    # ------------------------------------------------------------------
    if not skip_sample_data:
        def _create_sample_data():
            from src.create_sample_data import main
            main()

        if not _run_step("1 / 9  Create sample data", _create_sample_data):
            return 1
    else:
        print("\n  Skipping sample-data generation (--skip-sample-data).")

    # ------------------------------------------------------------------
    # Step 2 — Data preprocessing
    # ------------------------------------------------------------------
    def _preprocess():
        from src.data_preprocessing import main
        main()

    if not _run_step("2 / 9  Data preprocessing", _preprocess):
        return 1

    # ------------------------------------------------------------------
    # Step 3 — Keyword loader
    # ------------------------------------------------------------------
    def _load_keywords():
        from src.keyword_loader import main
        main()

    if not _run_step("3 / 9  Keyword loader", _load_keywords):
        return 1

    # ------------------------------------------------------------------
    # Step 4 — Semantic similarity
    # ------------------------------------------------------------------
    def _semantic_similarity():
        from src.semantic_similarity import main
        main()

    if not _run_step("4 / 9  Semantic similarity", _semantic_similarity):
        return 1

    # ------------------------------------------------------------------
    # Step 5 — Coordination detection
    # ------------------------------------------------------------------
    def _coordination_detection():
        from src.coordination_detection import main
        main()

    if not _run_step("5 / 9  Coordination detection", _coordination_detection):
        return 1

    # ------------------------------------------------------------------
    # Step 6 — Account analysis
    # ------------------------------------------------------------------
    def _account_analysis():
        from src.account_analysis import main
        main()

    if not _run_step("6 / 9  Account analysis", _account_analysis):
        return 1

    # ------------------------------------------------------------------
    # Step 7 — Campaign analysis
    # ------------------------------------------------------------------
    def _campaign_analysis():
        from src.campaign_analysis import main
        main()

    if not _run_step("7 / 9  Campaign analysis", _campaign_analysis):
        return 1

    # ------------------------------------------------------------------
    # Step 8 — Graph analysis
    # ------------------------------------------------------------------
    def _graph_analysis():
        from src.graph_analysis import main
        main()

    if not _run_step("8 / 9  Graph analysis", _graph_analysis):
        return 1

    # ------------------------------------------------------------------
    # Step 9 — Risk scoring
    # ------------------------------------------------------------------
    def _risk_scoring():
        from src.risk_scoring import main
        main()

    if not _run_step("9 / 9  Risk scoring", _risk_scoring):
        return 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline completed successfully in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nOutput files:")
    from src.config import (
        CLEANED_POSTS_FILE,
        CLEANED_KEYWORDS_FILE,
        SEMANTIC_RESULTS_FILE,
        COORDINATION_RESULTS_FILE,
        ACCOUNT_BEHAVIOR_FILE,
        CAMPAIGN_RESULTS_FILE,
        COORDINATION_GRAPH_PNG,
        RISK_RESULTS_FILE,
    )
    output_files = [
        ("Cleaned posts",         CLEANED_POSTS_FILE),
        ("Keywords",              CLEANED_KEYWORDS_FILE),
        ("Semantic similarity",   SEMANTIC_RESULTS_FILE),
        ("Coordination results",  COORDINATION_RESULTS_FILE),
        ("Account behaviour",     ACCOUNT_BEHAVIOR_FILE),
        ("Campaign results",      CAMPAIGN_RESULTS_FILE),
        ("Coordination graph",    COORDINATION_GRAPH_PNG),
        ("Risk scores",           RISK_RESULTS_FILE),
    ]
    for label, path in output_files:
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status}  {label:25s}  {path.relative_to(_PROJECT_ROOT)}")

    print("\nTo launch the investigation dashboard:")
    print("  streamlit run dashboard/app.py")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NexTrace Phase 1 coordination-analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-sample-data",
        action="store_true",
        default=False,
        help=(
            "Skip the sample-data generation step. Use this when you already "
            "have data in data/processed/sample_social_media_posts.csv."
        ),
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        default=False,
        help=(
            "Before running the pipeline, copy every existing Phase 1 output "
            "file to results/backups/<YYYYMMDD_HHMMSS>/. "
            "Useful when you want to preserve results from a previous run "
            "before overwriting them. The ML model is never included."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(
        run_pipeline(
            skip_sample_data=args.skip_sample_data,
            backup_existing=args.backup_existing,
        )
    )
