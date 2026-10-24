
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "ml_dataset.csv"

df = pd.read_csv(DATA_FILE, low_memory=False)

print("\n========== DATASET CHECK ==========")
print("Total accounts:", len(df))
print("Duplicate account IDs:", df["account_id"].duplicated().sum())

if "source_group" not in df.columns:
    print("ERROR: source_group column not found.")
    raise SystemExit(1)

if "label" not in df.columns:
    print("ERROR: label column not found.")
    raise SystemExit(1)

print("\nSource groups and labels:")
print(pd.crosstab(df["source_group"], df["label"]))

print("\n========== EXISTING SPLIT STRATEGY ==========")

try:
    from src.ml.train_models import split_account_dataset

    splits = split_account_dataset(df)

    for split_name in ("train", "validation", "test"):
        split_df = splits[split_name]
        print(f"\n{split_name.upper()}: {len(split_df)} accounts")
        print("Labels:", split_df["label"].value_counts().to_dict())
        print("Source groups:", sorted(split_df["source_group"].unique()))

    train_groups = set(splits["train"]["source_group"])
    val_groups = set(splits["validation"]["source_group"])
    test_groups = set(splits["test"]["source_group"])

    print("\n========== GROUP OVERLAP ==========")
    print("Train / validation:", sorted(train_groups & val_groups))
    print("Train / test:", sorted(train_groups & test_groups))
    print("Validation / test:", sorted(val_groups & test_groups))
    print("Strategy used:", splits["strategy"])

except Exception as exc:
    print("Split inspection failed:", repr(exc))

