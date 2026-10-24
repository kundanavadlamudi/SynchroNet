"""NexTrace — Phase 1 coordination-analysis pipeline orchestrator.

Runs the full Phase 1 pipeline in the correct dependency order:

    1. Ingestion / Sample data   — load posts from synthetic, file, or API provider
    2. Data preprocessing       — validate and normalize post records
    3. Keyword loader           — process hashtag/keyword files
    4. Semantic similarity      — compute pairwise text similarity (neural or TF-IDF fallback)
    5. Coordination detection   — flag coordinated account pairs
    6. Account analysis & ML    — summarize behavior and invoke existing bot classifier
    7. Campaign analysis        — group connected accounts into campaign candidates
    8. Graph analysis           — build and render the coordination graph
    9. Risk scoring             — assign transparent, explainable investigation-priority scores

Usage
-----
    # Standard demo mode:
    python run_pipeline.py

    # Import authorized CSV/JSON data:
    python run_pipeline.py --source file --posts path/to/posts.csv

    # Back up existing outputs before running:
    python run_pipeline.py --backup-existing

    # Dry-run validation only (does not write outputs):
    python run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so ``src.*`` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Backup helper — exactly 10 Phase 1 files preserved for backward compatibility
# ---------------------------------------------------------------------------

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

_EXTENDED_OUTPUTS = [
    "results/reports/account_classification_results.csv",
    "results/reports/pipeline_run_metadata.json",
    "results/reports/osint_enrichment_results.json",
]


def _backup_existing_outputs() -> Path:
    """Copy every existing output file into a timestamped backup directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = _PROJECT_ROOT / "results" / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    backed_up: list[str] = []
    for rel in _PHASE1_OUTPUTS + _EXTENDED_OUTPUTS:
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
# Step runner
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
    source: str = "demo",
    posts_path: str | None = None,
    accounts_path: str | None = None,
    query: str = "disinformation",
    enrich_osint: bool = True,
    dry_run: bool = False,
) -> int:
    """Run all Phase 1 and inference steps in order. Returns 0 on success, 1 on failure."""

    total_start = time.perf_counter()
    run_timestamp = datetime.now(timezone.utc).isoformat()

    print("\nNexTrace — Coordinated Disinformation Detection Pipeline")
    print("=" * 60)
    print(f"Run started at: {run_timestamp}")
    print(f"Data source   : {source}")
    if source in ("x", "reddit"):
        print(f"Search query  : {query}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Optional backup of existing output files
    # ------------------------------------------------------------------
    if backup_existing and not dry_run:
        print("\n  --backup-existing: copying existing output files …")
        _backup_existing_outputs()

    # ------------------------------------------------------------------
    # Step 1 — Ingestion / Data Loading
    # ------------------------------------------------------------------
    from src.config import SAMPLE_POSTS_FILE, CLEANED_POSTS_FILE
    from src.ingestion import get_data_provider, IngestionError, AuthenticationError

    def _ingest_data():
        if source in ("file", "csv", "json") and posts_path:
            provider = get_data_provider("file", posts_path=posts_path, accounts_path=accounts_path)
            posts_df, meta = provider.fetch_posts()
            print(f"Loaded {len(posts_df)} posts from {posts_path}")
            if not dry_run:
                posts_df.to_csv(SAMPLE_POSTS_FILE, index=False)
        elif source in ("api", "rest", "feed"):
            provider = get_data_provider("api")
            posts_df, meta = provider.fetch_posts()
            print(f"Fetched {len(posts_df)} posts from API")
            if not dry_run:
                posts_df.to_csv(SAMPLE_POSTS_FILE, index=False)
        elif source in ("x", "twitter"):
            provider = get_data_provider("x")
            status = provider.check_status()
            if status.get("status") == "BLOCKED_MISSING_CREDENTIALS":
                print(f"[BLOCKED] {status.get('message')}")
                raise IngestionError(status.get("message"))
            posts_df, accounts_df = provider.ingest(query=query, max_results=20)
            print(f"Fetched {len(posts_df)} posts and {len(accounts_df)} accounts from X API")
            if not dry_run:
                posts_df.to_csv(SAMPLE_POSTS_FILE, index=False)
        elif source in ("reddit", "reddit_api"):
            provider = get_data_provider("reddit")
            posts_df, accounts_df = provider.ingest(query=query, limit=25)
            print(f"Fetched {len(posts_df)} posts and {len(accounts_df)} accounts from Reddit API")
            if not dry_run:
                posts_df.to_csv(SAMPLE_POSTS_FILE, index=False)
        else:
            if not skip_sample_data:
                from src.create_sample_data import main as create_sample
                create_sample()
            else:
                print("Skipping sample-data generation (--skip-sample-data).")

    if not _run_step("1 / 9  Data Ingestion / Sourcing", _ingest_data):
        return 1

    if dry_run:
        print("\n[DRY RUN] Completed ingestion validation. Exiting without writing full pipeline results.")
        return 0

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
    # Step 6 — Account analysis & ML bot classification
    # ------------------------------------------------------------------
    def _account_analysis():
        from src.account_analysis import main
        main()

    if not _run_step("6 / 9  Account analysis & ML classification", _account_analysis):
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
    # Extended Step — OSINT narrative & evidence enrichment
    # ------------------------------------------------------------------
    if enrich_osint:
        def _enrich_osint_step():
            import pandas as pd
            from src.config import CAMPAIGN_RESULTS_FILE, CLEANED_POSTS_FILE, OSINT_ENRICHMENT_FILE
            from src.osint import OSINTEnricher

            posts_df = pd.read_csv(CLEANED_POSTS_FILE) if CLEANED_POSTS_FILE.exists() else pd.DataFrame()
            camps_df = pd.read_csv(CAMPAIGN_RESULTS_FILE) if CAMPAIGN_RESULTS_FILE.exists() else pd.DataFrame()
            enricher = OSINTEnricher()
            enricher.enrich_dataset(posts_df, camps_df, save_path=OSINT_ENRICHMENT_FILE)

        _run_step("OSINT Narrative & Evidence Enrichment", _enrich_osint_step)

    # ------------------------------------------------------------------
    # Write Pipeline Run Metadata
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start
    metadata_file = _PROJECT_ROOT / "results" / "reports" / "pipeline_run_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(
            {
                "run_timestamp": run_timestamp,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(total_elapsed, 2),
                "source": source,
                "is_synthetic": source == "demo",
                "posts_file": posts_path or str(SAMPLE_POSTS_FILE),
                "osint_enriched": enrich_osint,
                "pipeline_version": "NexTrace-v1.1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

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
        OSINT_ENRICHMENT_FILE,
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
        ("OSINT enrichment",      OSINT_ENRICHMENT_FILE),
        ("Run metadata",          metadata_file),
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
        description="NexTrace coordinated disinformation detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["demo", "file", "api", "x", "reddit"],
        default="demo",
        help="Data source mode: 'demo' (default), 'file', 'api', 'x', or 'reddit'.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="disinformation",
        help="Search query when fetching from live API sources (X or Reddit).",
    )
    parser.add_argument(
        "--no-osint",
        action="store_true",
        default=False,
        help="Disable OSINT narrative and evidence enrichment step.",
    )
    parser.add_argument(
        "--posts",
        type=str,
        default=None,
        help="Path to CSV or JSON file containing posts (required when --source file).",
    )
    parser.add_argument(
        "--accounts",
        type=str,
        default=None,
        help="Optional path to CSV or JSON file containing account metadata.",
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
            "Useful when you want to preserve results from a previous run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate source inputs and data contracts without overwriting pipeline results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(
        run_pipeline(
            skip_sample_data=args.skip_sample_data,
            backup_existing=args.backup_existing,
            source=args.source,
            posts_path=args.posts,
            accounts_path=args.accounts,
            query=args.query,
            enrich_osint=not args.no_osint,
            dry_run=args.dry_run,
        )
    )
