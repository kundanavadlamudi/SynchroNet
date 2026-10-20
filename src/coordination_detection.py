import pandas as pd
from itertools import combinations
from pathlib import Path
from difflib import SequenceMatcher

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "sample_social_media_posts.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "coordination_results.csv"


def text_similarity(text1, text2):
    """Calculate lexical similarity between two text strings."""
    return SequenceMatcher(
        None,
        str(text1).lower(),
        str(text2).lower()
    ).ratio()


def detect_coordination(df):
    """Identify potentially coordinated pairs of accounts."""

    results = []

    # Ensure timestamps are in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Compare every pair of posts
    for (_, post1), (_, post2) in combinations(df.iterrows(), 2):

        # Ignore posts without valid timestamps
        if pd.isna(post1["timestamp"]) or pd.isna(post2["timestamp"]):
            continue

        # Ignore posts from the same account
        if post1["account_id"] == post2["account_id"]:
            continue

        similarity = text_similarity(post1["text"], post2["text"])

        time_difference = abs(
            (post1["timestamp"] - post2["timestamp"]).total_seconds()
        ) / 60

        # Check coordination signals
        similar_text = similarity >= 0.85
        close_in_time = time_difference <= 10

        url1 = str(post1.get("url", "")).strip()
        url2 = str(post2.get("url", "")).strip()

        shared_url = (
            url1 != ""
            and url2 != ""
            and url1.lower() != "nan"
            and url1 == url2
        )

        # Score: text similarity 50%, timing 25%, shared URL 25%
        score = (
            0.50 * float(similar_text)
            + 0.25 * float(close_in_time)
            + 0.25 * float(shared_url)
        )

        # Record pairs supported by at least two signals
        signal_count = sum([
            similar_text,
            close_in_time,
            shared_url
        ])

        if signal_count >= 2:
            results.append({
                "account_1": post1["account_id"],
                "account_2": post2["account_id"],
                "post_1": post1["post_id"],
                "post_2": post2["post_id"],
                "text_similarity": round(similarity, 3),
                "time_difference_minutes": round(time_difference, 2),
                "shared_url": shared_url,
                "coordination_score": round(score, 3),
                "evidence_signals": signal_count
            })

    return pd.DataFrame(results)


def main():
    if not INPUT_FILE.exists():
        print("Sample dataset not found.")
        print("Run: python src/create_sample_data.py")
        return

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
        print(f"Missing required columns: {sorted(missing)}")
        return

    results = detect_coordination(df)

    if results.empty:
        print("No potentially coordinated account pairs detected.")
    else:
        results = results.sort_values(
            by="coordination_score",
            ascending=False
        )

        print("\nPotentially coordinated account pairs:")
        print(results.to_string(index=False))

    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nTotal flagged account pairs: {len(results)}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()