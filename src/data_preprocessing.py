import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "sample_social_media_posts.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "cleaned_social_media_posts.csv"


def preprocess_data(df):
    """Validate and clean social media post data."""

    required_columns = [
        "post_id",
        "account_id",
        "text",
        "timestamp",
        "url"
    ]

    # Check required columns
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Work on a copy of the input data
    df = df[required_columns].copy()

    # Normalize text fields
    for column in ["post_id", "account_id", "text", "url"]:
        df[column] = df[column].astype("string").str.strip()

    # Convert blank strings to missing values
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Remove rows missing essential identifiers or post text
    df = df.dropna(
        subset=["post_id", "account_id", "text"]
    )

    # Remove duplicate post IDs
    df = df.drop_duplicates(
        subset=["post_id"],
        keep="first"
    )

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    # Remove records with invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # Normalize URLs
    df["url"] = df["url"].fillna("")

    # Remove leading and trailing whitespace from text
    df["text"] = df["text"].str.strip()

    # Reset row numbering
    df = df.reset_index(drop=True)

    return df


def main():
    if not INPUT_FILE.exists():
        print("Input dataset not found.")
        print("Run: python src/create_sample_data.py")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(INPUT_FILE)

    cleaned_df = preprocess_data(original_df)

    cleaned_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nData preprocessing completed!")
    print(f"Original records: {len(original_df)}")
    print(f"Cleaned records: {len(cleaned_df)}")
    print(
        f"Removed records: "
        f"{len(original_df) - len(cleaned_df)}"
    )
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()