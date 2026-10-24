import sys
from pathlib import Path

# Add the project root to Python's import path
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import SEMANTIC_THRESHOLD

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "cresci_posts_multisource_cleaned.csv"
)

OUTPUT_FILE = (
    ROOT / "results" / "reports"
    / "cresci_semantic_similarity_results.csv"
)

TOP_K = 20
BATCH_SIZE = 256
def calculate_similarity(df):
    columns = [
        "post_1",
        "account_1",
        "post_2",
        "account_2",
        "text_similarity",
        "similarity_type",
    ]

    if len(df) < 2:
        return pd.DataFrame(columns=columns)

    df = df.reset_index(drop=True).copy()

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Sparse TF-IDF representations avoid a dense N x N matrix.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=columns)

    # Retrieve only the nearest candidate texts.
    k = min(TOP_K + 1, len(df))

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(matrix)

    results = []
    seen_pairs = set()

    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))

        distances, indices = nn.kneighbors(
            matrix[start:end],
            n_neighbors=k,
        )

        for local_i, (row_distances, row_indices) in enumerate(
            zip(distances, indices)
        ):
            i = start + local_i

            for distance, j in zip(row_distances, row_indices):
                j = int(j)

                if i == j:
                    continue

                if str(df.at[i, "account_id"]) == str(
                    df.at[j, "account_id"]
                ):
                    continue

                pair = (min(i, j), max(i, j))

                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)

                similarity = max(0.0, min(1.0, 1.0 - float(distance)))

                if similarity < SEMANTIC_THRESHOLD:
                    continue

                results.append({
                    "post_1": str(df.at[pair[0], "post_id"]),
                    "account_1": str(df.at[pair[0], "account_id"]),
                    "post_2": str(df.at[pair[1], "post_id"]),
                    "account_2": str(df.at[pair[1], "account_id"]),
                    "text_similarity": round(similarity, 4),
                    "similarity_type": "TF-IDF Nearest Neighbors",
                })

    return pd.DataFrame(results, columns=columns)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"post_id": str, "account_id": str},
    )

    required = {"post_id", "account_id", "text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    results = calculate_similarity(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Cresci similarity analysis completed.")
    print("Input posts:", len(df))
    print("Candidate pairs retained:", len(results))
    print("Threshold:", SEMANTIC_THRESHOLD)
    print("Output:", OUTPUT_FILE)

    if not results.empty:
        print("\nTop results:")
        print(
            results.sort_values(
                "text_similarity", ascending=False
            ).head(10).to_string(index=False)
        )


if __name__ == "__main__":
    main()