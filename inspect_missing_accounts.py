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

print("Missing tweet features by source:")
print(
    df.groupby("source_group")[tweet_features]
    .apply(lambda x: x.isna().all(axis=1).sum())
    .to_string()
)

print("\nSample accounts with missing tweet features:")
print(
    df.loc[
        df[tweet_features].isna().all(axis=1),
        ["source_group", "label", "tweet_data_available",
         "statuses_count", "followers_count", "friends_count"]
    ].head(15).to_string(index=False)
)