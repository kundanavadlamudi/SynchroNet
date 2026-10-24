from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent

dataset_path = ROOT / "data" / "processed" / "ml_dataset.csv"
status_path = ROOT / "data" / "processed" / "ml_dataset_status.json"

if not dataset_path.exists():
    print("ERROR: ML dataset not found:", dataset_path)
    raise SystemExit(1)

df = pd.read_csv(dataset_path, low_memory=False)

print("\n========== DATASET VALIDATION ==========")
print("Dataset path:", dataset_path)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:", df.duplicated().sum())

for col in df.columns:
    if any(term in col.lower() for term in ("label", "target", "class")):
        print(f"\nPossible label column: {col}")
        print(df[col].value_counts(dropna=False).head(20))

for col in df.columns:
    if any(term in col.lower() for term in ("source", "category", "dataset", "group")):
        print(f"\nPossible provenance column: {col}")
        print(df[col].value_counts(dropna=False).head(20))

if status_path.exists():
    print("\nDataset preparation status:")
    try:
        print(json.dumps(json.loads(status_path.read_text()), indent=2))
    except (json.JSONDecodeError, OSError) as exc:
        print("Could not read status JSON:", exc)

print("\n========== END ==========")

