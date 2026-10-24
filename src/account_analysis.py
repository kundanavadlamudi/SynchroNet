import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = _PROJECT_ROOT

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_social_media_posts.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "account_behavior_results.csv"
CLASSIFICATION_OUTPUT_FILE = OUTPUT_DIR / "account_classification_results.csv"


def analyze_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate behavioral features for each account."""
    if df.empty:
        return pd.DataFrame(columns=[
            "account_id",
            "total_posts",
            "unique_texts",
            "urls_shared",
            "duplicate_text_posts",
            "average_time_between_posts_minutes",
        ])

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

    if not valid_posts.empty:
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
    else:
        summary["average_time_between_posts_minutes"] = pd.NA

    # A single post has no time gap.
    # Missing averages are therefore expected for such accounts.
    return summary


def classify_accounts_with_ml(
    posts_df: pd.DataFrame,
    accounts_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Safely apply the trained bot classifier to analyzed accounts.

    Uses the existing model bundle without modifying the artifact.
    Gracefully handles missing model or missing profile columns.
    """
    if posts_df.empty:
        return pd.DataFrame(columns=[
            "account_id", "predicted_label", "predicted_label_name",
            "bot_probability", "genuine_probability", "model_status"
        ])

    try:
        from src.ml.features import extract_account_features, add_profile_features
        from src.ml.predict_account import load_model_bundle, predict_accounts, ModelNotAvailableError

        bundle = load_model_bundle()
        expected_features = bundle.get("feature_names", [])

        # 1. Extract post-level features
        features_df = extract_account_features(posts_df)

        # 2. Add profile features if available
        if accounts_df is not None and not accounts_df.empty:
            features_df = add_profile_features(features_df, accounts_df)

        # 3. Supply neutral defaults for any missing features so model can score
        for col in expected_features:
            if col not in features_df.columns:
                if col == "tweet_data_available":
                    features_df[col] = 1
                else:
                    features_df[col] = 0

        # 4. Predict probabilities
        predictions = predict_accounts(features_df)
        predictions["model_status"] = "Scored with trained Random Forest classifier"
        predictions["ml_limitation"] = "Resemblance indicator based on Cresci benchmark; not proof of inauthentic intent"
        return predictions

    except Exception as exc:
        print(f"Notice: ML account classification skipped: {exc}")
        accounts = sorted(posts_df["account_id"].dropna().unique())
        return pd.DataFrame({
            "account_id": accounts,
            "predicted_label": [pd.NA] * len(accounts),
            "predicted_label_name": ["Not Scored"] * len(accounts),
            "bot_probability": [pd.NA] * len(accounts),
            "genuine_probability": [pd.NA] * len(accounts),
            "model_status": [f"Unavailable: {exc}"] * len(accounts),
            "ml_limitation": ["Model prediction was not run"] * len(accounts),
        })


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

    # Also compute ML predictions and attach bot_probability if available
    ml_results = classify_accounts_with_ml(df)
    if not ml_results.empty and "bot_probability" in ml_results.columns:
        results = results.merge(
            ml_results[["account_id", "predicted_label_name", "bot_probability"]],
            on="account_id",
            how="left",
        )
        # Save separate detailed classification report
        ml_results.to_csv(
            CLASSIFICATION_OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Saved account ML classifications to: {CLASSIFICATION_OUTPUT_FILE}")

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