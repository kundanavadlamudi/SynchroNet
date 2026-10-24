from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

FILE = ROOT / "data" / "processed" / "ml_dataset.csv"

df = pd.read_csv(FILE, low_memory=False)

columns = [
    "total_posts",
    "posts_with_urls",
    "unique_texts",
    "duplicate_text_posts",
    "average_text_length",
    "average_word_count",
    "average_time_between_posts_minutes",
]

columns = [c for c in columns if c in df.columns]

print("\nMISSING VALUES BY SOURCE GROUP")

print(
    df.groupby("source_group")[columns]
    .agg(lambda x: (x == 0).sum())
    .to_string()
)

print("\nTOTAL POSTS DISTRIBUTION")

print(
    df.groupby(["source_group", "label"])["total_posts"]
    .describe()
    .round(2)
    .to_string()
)

print("\nZERO-POST ACCOUNTS BY SOURCE")

print(
    df.assign(
        zero_posts=df["total_posts"].eq(0)
    )
    .groupby("source_group")["zero_posts"]
    .agg(["sum", "count", "mean"])
    .round(3)
    .to_string()
)