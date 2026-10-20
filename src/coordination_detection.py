import pandas as pd
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_social_media_posts.csv"
)

SEMANTIC_FILE = (
    BASE_DIR
    / "results"
    / "reports"
    / "semantic_similarity_results.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "coordination_results.csv"


# ==========================================
# 2. DETECTION SETTINGS
# ==========================================

TIME_WINDOW_MINUTES = 10
SEMANTIC_THRESHOLD = 0.70


# ==========================================
# 3. COORDINATION DETECTION
# ==========================================

def detect_coordination(posts, semantic_results):
    """Identify potentially coordinated account pairs."""

    posts = posts.copy()

    posts["timestamp"] = pd.to_datetime(
        posts["timestamp"],
        errors="coerce",
        utc=True
    )

    # Store semantic similarity scores by post pair.
    semantic_scores = {}

    if not semantic_results.empty:

        for _, row in semantic_results.iterrows():

            pair = frozenset([
                str(row["post_1"]),
                str(row["post_2"])
            ])

            semantic_scores[pair] = float(
                row["text_similarity"]
            )

    results = []

    # Compare every pair of posts.
    for i in range(len(posts)):

        for j in range(i + 1, len(posts)):

            post1 = posts.iloc[i]
            post2 = posts.iloc[j]

            # Skip posts from the same account.
            if post1["account_id"] == post2["account_id"]:
                continue

            # Skip posts with invalid timestamps.
            if (
                pd.isna(post1["timestamp"])
                or pd.isna(post2["timestamp"])
            ):
                continue

            # Find semantic similarity for this pair.
            post_pair = frozenset([
                str(post1["post_id"]),
                str(post2["post_id"])
            ])

            semantic_similarity = semantic_scores.get(
                post_pair, 0.0
            )

            # Calculate the time difference in minutes.
            time_difference = abs(
                (
                    post1["timestamp"]
                    - post2["timestamp"]
                ).total_seconds()
            ) / 60

            close_in_time = (
                time_difference <= TIME_WINDOW_MINUTES
            )

            # Check whether the posts share a URL.
            url1 = str(post1.get("url", "")).strip()
            url2 = str(post2.get("url", "")).strip()

            shared_url = (
                bool(url1)
                and bool(url2)
                and url1.lower() != "nan"
                and url2.lower() != "nan"
                and url1 == url2
            )

            # Check semantic similarity threshold.
            similar_text = (
                semantic_similarity >= SEMANTIC_THRESHOLD
            )

            # Calculate the coordination score.
            coordination_score = (
                0.50 * semantic_similarity
                + 0.25 * float(close_in_time)
                + 0.25 * float(shared_url)
            )

            # Count the supporting signals.
            signal_count = sum([
                similar_text,
                close_in_time,
                shared_url
            ])

            # Require similar content plus at least one
            # additional coordination signal.
            if similar_text and signal_count >= 2:

                results.append({
                    "account_1": post1["account_id"],
                    "account_2": post2["account_id"],
                    "post_1": post1["post_id"],
                    "post_2": post2["post_id"],
                    "semantic_similarity": round(
                        semantic_similarity, 4
                    ),
                    "time_difference_minutes": round(
                        time_difference, 2
                    ),
                    "shared_url": shared_url,
                    "coordination_score": round(
                        coordination_score, 4
                    ),
                    "evidence_signals": signal_count
                })

    return pd.DataFrame(results)


# ==========================================
# 4. MAIN FUNCTION
# ==========================================

def main():

    if not INPUT_FILE.exists():

        print("Cleaned dataset not found.")
        print("Run: python src/data_preprocessing.py")
        return

    if not SEMANTIC_FILE.exists():

        print("Semantic similarity results not found.")
        print("Run: python src/semantic_similarity.py")
        return

    posts = pd.read_csv(INPUT_FILE)

    semantic_results = pd.read_csv(SEMANTIC_FILE)

    results = detect_coordination(
        posts,
        semantic_results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCoordination detection completed!")

    if results.empty:

        print("No account pairs met the detection criteria.")

    else:

        results = results.sort_values(
            "coordination_score",
            ascending=False
        )

        print(f"Flagged account pairs: {len(results)}")

        print(
            results.to_string(index=False)
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()