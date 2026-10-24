import pandas as pd
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SEMANTIC_THRESHOLD, SENTENCE_TRANSFORMER_MODEL


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "cleaned_social_media_posts.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "semantic_similarity_results.csv"


# Pretrained sentence embedding model — configured in src/config.py.
MODEL_NAME = SENTENCE_TRANSFORMER_MODEL

# Similarity threshold — configured in src/config.py.
SIMILARITY_THRESHOLD = SEMANTIC_THRESHOLD


def calculate_semantic_similarity(df):
    """Calculate semantic similarity between social media posts."""

    if df.empty:
        return pd.DataFrame()

    texts = df["text"].fillna("").astype(str).tolist()

    # Load the pretrained model
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    # Convert posts into numerical embeddings
    print("Generating text embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    results = []

    # Compare each unique pair
    for i in range(len(df)):
        for j in range(i + 1, len(df)):

            similarity_score = float(similarity_matrix[i, j])

            # Ignore posts from the same account
            if df.iloc[i]["account_id"] == df.iloc[j]["account_id"]:
                continue

            # Keep pairs above the similarity threshold
            if similarity_score >= SIMILARITY_THRESHOLD:
                results.append({
                    "post_1": df.iloc[i]["post_id"],
                    "account_1": df.iloc[i]["account_id"],
                    "post_2": df.iloc[j]["post_id"],
                    "account_2": df.iloc[j]["account_id"],
                    "text_similarity": round(
                        similarity_score, 4
                    ),
                    "similarity_type": "Semantic"
                })

    return pd.DataFrame(results)


def main():

    if not INPUT_FILE.exists():
        print("Cleaned dataset not found.")
        print("Run: python src/data_preprocessing.py")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "post_id",
        "account_id",
        "text"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"Missing columns: {sorted(missing_columns)}")
        return

    results = calculate_semantic_similarity(df)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSemantic similarity analysis completed!")

    if results.empty:
        print("No similar post pairs found above the threshold.")
    else:
        print(f"Similar post pairs found: {len(results)}")
        print(
            results.sort_values(
                "text_similarity",
                ascending=False
            ).to_string(index=False)
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()