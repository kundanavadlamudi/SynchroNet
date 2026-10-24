from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data" / "processed" / "ml_dataset.csv"

df = pd.read_csv(FILE, low_memory=False)

features = [
    "total_posts",
    "posts_with_urls",
    "unique_texts",
    "duplicate_text_posts",
    "average_text_length",
    "average_word_count",
    "average_time_between_posts_minutes",
    "url_post_ratio",
    "duplicate_text_ratio",
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
    "description_length",
    "friends_followers_ratio",
]

features = [c for c in features if c in df.columns]

summary = (
    df.groupby(["source_group", "label"])[features]
    .median()
    .round(2)
)

output = ROOT / "results" / "reports" / "source_feature_medians.csv"
output.parent.mkdir(parents=True, exist_ok=True)

summary.to_csv(output)

print(summary.to_string())
print(f"\nSaved results to: {output}")