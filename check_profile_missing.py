from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data" / "processed" / "ml_dataset.csv"

df = pd.read_csv(FILE, low_memory=False)

features = [
    "followers_count",
    "friends_count",
    "favourites_count",
    "statuses_count",
    "description_length",
    "friends_followers_ratio",
    "account_age_days",
    "total_posts",
]

features = [c for c in features if c in df.columns]

print("\nMISSING VALUES BY SOURCE GROUP")

print(
    df.groupby("source_group")[features]
    .agg(lambda x: x.isna().sum())
    .to_string()
)

print("\nFEATURE MEDIANS BY SOURCE GROUP")

print(
    df.groupby("source_group")[features]
    .median()
    .round(2)
    .to_string()
)

print("\nDATASET SHAPE:", df.shape)