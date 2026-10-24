from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data" / "processed" / "ml_dataset.csv"

df = pd.read_csv(FILE, low_memory=False)

print("Dataset shape:", df.shape)

print("\nTweet-data availability by source:")
print(
    df.groupby("source_group")["tweet_data_available"]
    .agg(["min", "max", "count"])
    .to_string()
)

print("\nMissing tweet features by source:")
features = [
    "total_posts",
    "average_text_length",
    "average_word_count",
    "posts_with_urls",
    "unique_texts",
    "duplicate_text_posts",
    "url_post_ratio",
    "duplicate_text_ratio",
]

features = [c for c in features if c in df.columns]

print(
    df.groupby("source_group")[features]
    .agg(lambda x: x.isna().sum())
    .to_string()
)