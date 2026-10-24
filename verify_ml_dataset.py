from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data" / "processed" / "ml_dataset.csv"

df = pd.read_csv(FILE, low_memory=False)

tweet_features = [
    "total_posts",
    "average_text_length",
    "average_word_count",
    "posts_with_urls",
    "unique_texts",
    "duplicate_text_posts",
    "url_post_ratio",
    "duplicate_text_ratio",
]

print("Dataset shape:", df.shape)
print("Duplicate account IDs:", df["account_id"].duplicated().sum())

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nMissing values by label:")
print(df.groupby("label")[tweet_features].apply(
    lambda group: group.isna().sum()
))

print("\nMissing tweet features by source:")
print(df.groupby("source_group")[tweet_features].apply(
    lambda group: group.isna().sum()
))

print("\nTweet availability by source:")
print(df.groupby("source_group")["tweet_data_available"].agg(
    ["min", "max", "count"]
))