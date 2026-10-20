import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_social_media_posts.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "account_behavior_results.csv"


def analyze_accounts(df):
    """Calculate behavioral features for each account."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    df["text"] = df["text"].fillna("").astype(str)

    df["url"] = df["url"].fillna("").astype(str)

    # Count posts per account
    summary = df.groupby("account_id").agg(
        total_posts=("post_id", "count"),
        unique_texts=("text", "nunique"),
        urls_shared=("url", lambda values:
                     values.str.strip().ne("").sum())
    ).reset_index()

    # Calculate exact duplicate-text counts
    duplicate_counts = (
        df.groupby("account_id")["text"]
        .apply(lambda values: values.duplicated().sum())
        .reset_index(name="duplicate_text_posts")
    )

    summary = summary.merge(
        duplicate_counts,
        on="account_id",
        how="left"
    )

    # Calculate average time between consecutive posts
    valid_posts = df.dropna(
        subset=["timestamp"]
    ).sort_values(
        ["account_id", "timestamp"]
    )

    valid_posts["time_gap_minutes"] = (
        valid_posts.groupby("account_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    average_gaps = (
        valid_posts.groupby("account_id")["time_gap_minutes"]
        .mean()
        .reset_index(name="average_time_between_posts_minutes")
    )

    summary = summary.merge(
        average_gaps,
        on="account_id",
        how="left"
    )

    # A single post has no time gap.
    # Missing averages are therefore expected for such accounts.

    return summary


def main():

    if not INPUT_FILE.exists():
        print("Cleaned dataset not found.")
        print("Run: python src/data_preprocessing.py")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "post_id",
        "account_id",
        "text",
        "timestamp",
        "url"
    }

    missing = required_columns - set(df.columns)

    if missing:
        print(f"Missing columns: {sorted(missing)}")
        return

    results = analyze_accounts(df)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAccount behavior analysis completed!")
    print(f"Accounts analyzed: {len(results)}")

    print("\nAccount behavior summary:")
    print(results.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()