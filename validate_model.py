
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

model_path = ROOT / "models" / "selected_bot_classifier.joblib"
comparison_path = ROOT / "results" / "reports" / "ml_model_comparison.csv"
summary_path = ROOT / "results" / "reports" / "ml_evaluation_summary.json"

print("\n========== MODEL CHECK ==========")

try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    print("Model type:", type(model).__name__)

    if hasattr(model, "named_steps"):
        print("Pipeline steps:", list(model.named_steps.keys()))

    if hasattr(model, "feature_names_in_"):
        print("Expected features:", list(model.feature_names_in_))

except Exception as e:
    print("Model loading error:", repr(e))

print("\n========== MODEL COMPARISON ==========")

try:
    comparison = pd.read_csv(comparison_path)
    print("Columns:", comparison.columns.tolist())
    print(comparison.to_string(index=False))
except Exception as e:
    print("Comparison error:", repr(e))

print("\n========== EVALUATION SUMMARY ==========")

try:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    print(json.dumps(summary, indent=2))
except Exception as e:
    print("Summary error:", repr(e))

print("\n========== CHECK COMPLETE ==========")

