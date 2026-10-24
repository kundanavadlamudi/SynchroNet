"""NexTrace — Phase 2 ML bot-classification pipeline reference script.

This script describes the Phase 2 pipeline and provides a safe way to
inspect the trained model status WITHOUT retraining.

IMPORTANT
---------
The trained Random Forest model (models/selected_bot_classifier.joblib)
was trained on 16,318 labeled accounts from the Cresci-2015 and
Cresci-2017 research datasets and achieves:

    Test F1 = 0.989 | Test ROC-AUC = 0.998

**Do not retrain the model unless the training dataset has changed.**

Phase 2 pipeline order (run only when needed)
---------------------------------------------
    1. prepare_dataset        src/ml/prepare_dataset.py
       Loads Cresci-2015/2017 CSVs from data/raw/ and writes the
       combined labeled feature table to data/processed/ml_dataset.csv.

    2. train_models           src/ml/train_models.py
       Trains Logistic Regression, Decision Tree, and Random Forest
       classifiers. Selects the best model by validation F1 and saves
       it to models/selected_bot_classifier.joblib.
       Also saves SHAP/LIME explanations and confusion-matrix plots.

    3. evaluate_models        (called automatically by train_models.py)
       Writes results/reports/ml_evaluation_summary.json and
       results/reports/ml_model_comparison.csv.

    4. explain_predictions    src/ml/explain_predictions.py
       Generates per-account SHAP and LIME explanations for sample
       accounts. Writes results/reports/ml_explanations.json.

    5. predict_account        src/ml/predict_account.py
       Load the trained model and score new account feature rows.

Usage — inspect model status (safe, no retraining)
---------------------------------------------------
    python run_ml_pipeline.py --status

Usage — retrain from scratch (only if datasets changed)
--------------------------------------------------------
    python run_ml_pipeline.py --retrain
    # You will be prompted to confirm before training starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Status check (safe — no retraining)
# ---------------------------------------------------------------------------

def show_status() -> None:
    """Print current model and dataset status without retraining."""
    from src.config import (
        SELECTED_MODEL_FILE,
        MODEL_METADATA_FILE,
        BACKGROUND_SAMPLE_FILE,
        ML_DATASET_FILE,
        ML_EVALUATION_SUMMARY_FILE,
        ML_MODEL_COMPARISON_FILE,
        ML_EXPLANATIONS_FILE,
    )

    print("\nNexTrace — Phase 2 ML Pipeline Status")
    print("=" * 60)
    print("NOTE: This command is read-only. No model is retrained.\n")

    # Model files
    print("Model artifacts:")
    model_files = [
        ("Trained model (.joblib)",  SELECTED_MODEL_FILE),
        ("Model metadata (.json)",   MODEL_METADATA_FILE),
        ("SHAP background sample",   BACKGROUND_SAMPLE_FILE),
    ]
    for label, path in model_files:
        status = "[OK]    " if path.exists() else "[MISSING]"
        size = f"  ({path.stat().st_size / 1_048_576:.1f} MB)" if path.exists() else ""
        print(f"  {status}  {label:30s}  {path.relative_to(_PROJECT_ROOT)}{size}")

    # Metadata summary
    if MODEL_METADATA_FILE.exists():
        meta = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
        print(f"\nSelected model : {meta.get('model_name')}")
        print(f"Split strategy : {meta.get('split_strategy')}")
        print(f"Training size  : {meta.get('dataset_provenance', {}).get('n_accounts')} accounts")
        print(f"Source datasets: {', '.join(meta.get('dataset_provenance', {}).get('source_datasets', []))}")
        tm = meta.get("test_metrics", {})
        print(f"\nTest metrics (held-out set):")
        print(f"  Accuracy  : {tm.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision : {tm.get('precision', 'N/A'):.4f}")
        print(f"  Recall    : {tm.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score  : {tm.get('f1_score', 'N/A'):.4f}")
        print(f"  ROC-AUC   : {tm.get('roc_auc', 'N/A'):.4f}")
        print(f"\nCreated at     : {meta.get('created_at', 'unknown')}")

    # Dataset and report files
    print("\nDataset & report files:")
    report_files = [
        ("ML training dataset",       ML_DATASET_FILE),
        ("Evaluation summary",        ML_EVALUATION_SUMMARY_FILE),
        ("Model comparison table",    ML_MODEL_COMPARISON_FILE),
        ("SHAP/LIME explanations",    ML_EXPLANATIONS_FILE),
    ]
    for label, path in report_files:
        status = "[OK]    " if path.exists() else "[MISSING]"
        print(f"  {status}  {label:30s}  {path.relative_to(_PROJECT_ROOT)}")

    print("\nTo retrain (only if datasets changed):")
    print("  python run_ml_pipeline.py --retrain")


# ---------------------------------------------------------------------------
# Retrain (requires explicit confirmation)
# ---------------------------------------------------------------------------

def retrain() -> int:
    """Retrain the model after explicit user confirmation."""
    print("\n" + "!" * 60)
    print("  WARNING: This will overwrite the trained model.")
    print("  The existing model achieves F1=0.989 on held-out data.")
    print("  Only proceed if the training dataset has changed.")
    print("!" * 60)

    answer = input("\n  Type YES to confirm retraining: ").strip()
    if answer != "YES":
        print("  Aborted. No changes made.")
        return 0

    print("\nStep 1/3 — Preparing dataset …")
    try:
        from src.ml.prepare_dataset import prepare_ml_dataset, describe_missing_datasets
        result = prepare_ml_dataset()
        if result.dataset.empty:
            print(describe_missing_datasets())
            return 1
        print(f"  Dataset prepared: {len(result.dataset)} accounts")
    except Exception as exc:
        print(f"  Dataset preparation failed: {exc}")
        return 1

    print("\nStep 2/3 — Training models …")
    try:
        from src.ml.train_models import load_prepared_dataset, train_and_evaluate
        dataset = load_prepared_dataset()
        outcome = train_and_evaluate(dataset)
        print(f"  Selected model: {outcome['selected_model']}")
        tm = outcome["test_results"]
        print(
            f"  Test: F1={tm['f1_score']:.4f}  "
            f"Accuracy={tm['accuracy']:.4f}  "
            f"ROC-AUC={tm['roc_auc']}"
        )
        print(f"  Model saved to: {outcome['model_file']}")
    except Exception as exc:
        print(f"  Training failed: {exc}")
        return 1

    print("\nStep 3/3 — Generating explanations …")
    try:
        from src.ml.explain_predictions import main as explain_main
        explain_main()
    except Exception as exc:
        print(f"  Explanation generation failed (non-fatal): {exc}")

    print("\nRetraining complete.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NexTrace Phase 2 ML pipeline — status and retraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--status",
        action="store_true",
        default=True,
        help="[Default] Show model status without retraining.",
    )
    group.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="Retrain the model (prompts for confirmation first).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.retrain:
        sys.exit(retrain())
    else:
        show_status()
        sys.exit(0)
