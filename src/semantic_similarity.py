import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import SEMANTIC_THRESHOLD, SENTENCE_TRANSFORMER_MODEL

logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = _PROJECT_ROOT

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


def _compute_embeddings_sentence_transformer(texts: list[str]) -> np.ndarray | None:
    """Attempt embedding calculation using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading Sentence Transformer model ({MODEL_NAME})...")
        model = SentenceTransformer(MODEL_NAME)
        print("Generating neural text embeddings...")
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
    except Exception as exc:
        print(f"Notice: SentenceTransformer not available ({exc}). Falling back to TF-IDF n-gram vectorizer.")
        return None


def _compute_similarity_tfidf(texts: list[str]) -> np.ndarray:
    """Fallback cosine similarity using scikit-learn TfidfVectorizer."""
    print("Generating TF-IDF word & character n-gram text representations...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    # Handle case where all texts are empty
    valid_texts = [t if t.strip() else " " for t in texts]
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        return cosine_similarity(tfidf_matrix)
    except Exception:
        return np.zeros((len(texts), len(texts)))


def calculate_semantic_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise text similarity between social media posts.

    Uses SentenceTransformer if available, or graceful TF-IDF cosine fallback.
    Excludes pairs from the same account and pairs below SIMILARITY_THRESHOLD.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=[
            "post_1", "account_1", "post_2", "account_2",
            "text_similarity", "similarity_type"
        ])

    texts = [str(t).strip() if pd.notna(t) else "" for t in df["text"]]

    # Try neural embeddings first, fallback to TF-IDF
    embeddings = _compute_embeddings_sentence_transformer(texts)
    if embeddings is not None:
        similarity_matrix = cosine_similarity(embeddings)
        sim_type = "Semantic"
    else:
        similarity_matrix = _compute_similarity_tfidf(texts)
        sim_type = "TF-IDF Fallback"

    results = []

    # Compare each unique pair
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            account_i = str(df.iloc[i]["account_id"])
            account_j = str(df.iloc[j]["account_id"])

            # Ignore posts from the same account
            if account_i == account_j:
                continue

            similarity_score = float(similarity_matrix[i, j])

            # Keep pairs at or above the threshold
            if similarity_score >= SIMILARITY_THRESHOLD:
                results.append({
                    "post_1": df.iloc[i]["post_id"],
                    "account_1": account_i,
                    "post_2": df.iloc[j]["post_id"],
                    "account_2": account_j,
                    "text_similarity": round(similarity_score, 4),
                    "similarity_type": sim_type,
                })

    return pd.DataFrame(results, columns=[
        "post_1", "account_1", "post_2", "account_2",
        "text_similarity", "similarity_type"
    ])


def main():
    if not INPUT_FILE.exists():
        print("Cleaned dataset not found.")
        print("Run: python src/data_preprocessing.py")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"post_id", "account_id", "text"}
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