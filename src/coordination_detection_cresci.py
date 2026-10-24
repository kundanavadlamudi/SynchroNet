import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    TIME_WINDOW_MINUTES,
    COORDINATION_WEIGHT_SEMANTIC,
    COORDINATION_WEIGHT_TIME,
    COORDINATION_WEIGHT_URL,
    MIN_SIGNAL_COUNT,
)

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "cresci_posts_multisource_cleaned.csv"
)

SEMANTIC_FILE = (
    ROOT / "results" / "reports"
    / "cresci_semantic_similarity_results.csv"
)

OUTPUT_FILE = (
    ROOT / "results" / "reports"
    / "cresci_coordination_results.csv"
)


def detect_coordination(posts, semantic_results):
    posts = posts.copy()

    posts["post_id"] = posts["post_id"].astype(str)
    posts["account_id"] = posts["account_id"].astype(str)

    posts["timestamp"] = pd.to_datetime(
        posts["timestamp"],
        errors="coerce",
        utc=True,
    )

    # Index posts by ID for efficient candidate lookup.
    posts = posts.drop_duplicates("post_id").set_index("post_id")

    results = []

    for row in semantic_results.itertuples(index=False):
        post1_id = str(row.post_1)
        post2_id = str(row.post_2)

        if post1_id not in posts.index or post2_id not in posts.index:
            continue

        post1 = posts.loc[post1_id]
        post2 = posts.loc[post2_id]

        account1 = str(post1["account_id"])
        account2 = str(post2["account_id"])

        if account1 == account2:
            continue

        time1 = post1["timestamp"]
        time2 = post2["timestamp"]

        if pd.isna(time1) or pd.isna(time2):
            continue

        time_difference = abs(
            (time1 - time2).total_seconds()
        ) / 60.0

        close_in_time = time_difference <= TIME_WINDOW_MINUTES

        url1 = str(post1.get("url", "") or "").strip()
        url2 = str(post2.get("url", "") or "").strip()

        shared_url = (
            bool(url1)
            and bool(url2)
            and url1.lower() != "nan"
            and url2.lower() != "nan"
            and url1 == url2
        )

        similarity = float(row.text_similarity)

        similar_text = similarity >= 0.70

        signal_count = sum([
            similar_text,
            close_in_time,
            shared_url,
        ])

        score = (
            COORDINATION_WEIGHT_SEMANTIC * similarity
            + COORDINATION_WEIGHT_TIME * float(close_in_time)
            + COORDINATION_WEIGHT_URL * float(shared_url)
        )

        if similar_text and signal_count >= MIN_SIGNAL_COUNT:
            results.append({
                "account_1": account1,
                "account_2": account2,
                "post_1": post1_id,
                "post_2": post2_id,
                "semantic_similarity": round(similarity, 4),
                "time_difference_minutes": round(time_difference, 2),
                "shared_url": shared_url,
                "coordination_score": round(score, 4),
                "evidence_signals": signal_count,
                "source_group_1": post1.get("source_group", ""),
                "source_group_2": post2.get("source_group", ""),
            })

    columns = [
        "account_1",
        "account_2",
        "post_1",
        "post_2",
        "semantic_similarity",
        "time_difference_minutes",
        "shared_url",
        "coordination_score",
        "evidence_signals",
        "source_group_1",
        "source_group_2",
    ]

    return pd.DataFrame(results, columns=columns)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    if not SEMANTIC_FILE.exists():
        raise FileNotFoundError(
            f"Similarity results not found: {SEMANTIC_FILE}"
        )

    posts = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": str, "account_id": str},
    )

    semantic_results = pd.read_csv(
        SEMANTIC_FILE,
        dtype={
            "post_1": str,
            "account_1": str,
            "post_2": str,
            "account_2": str,
        },
    )

    results = detect_coordination(posts, semantic_results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nCresci coordination analysis completed.")
    print("Input posts:", len(posts))
    print("Semantic candidate pairs:", len(semantic_results))
    print("Flagged coordination pairs:", len(results))
    print("Output:", OUTPUT_FILE)

    if not results.empty:
        print("\nTop coordination candidates:")
        print(
            results.sort_values(
                "coordination_score",
                ascending=False,
            ).head(20).to_string(index=False)
        )


if __name__ == "__main__":
    main()