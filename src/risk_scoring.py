import pandas as pd
from pathlib import Path

from src.config import (
    SEMANTIC_THRESHOLD,
    TIME_WINDOW_MINUTES,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "reports"
    / "coordination_results.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "explainable_risk_results.csv"


# ==========================================
# 2. RISK SCORING
# ==========================================

def calculate_risk(df):
    """Create an explainable investigation-priority score."""

    df = df.copy()

    if df.empty:
        return df

    required_columns = {
        "account_1",
        "account_2",
        "semantic_similarity",
        "time_difference_minutes",
        "shared_url",
        "coordination_score"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Convert values to their expected types.
    df["semantic_similarity"] = pd.to_numeric(
        df["semantic_similarity"],
        errors="coerce"
    ).fillna(0).clip(0, 1)

    df["time_difference_minutes"] = pd.to_numeric(
        df["time_difference_minutes"],
        errors="coerce"
    )

    df["coordination_score"] = pd.to_numeric(
        df["coordination_score"],
        errors="coerce"
    ).fillna(0).clip(0, 1)

    # Convert shared URL values safely.
    df["shared_url"] = (
        df["shared_url"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # Identify whether the posts were published
    # within the configured time window.
    df["close_in_time"] = (
        df["time_difference_minutes"] <= TIME_WINDOW_MINUTES
    )

    # Explain the evidence for each pair.
    df["evidence"] = df.apply(
        lambda row: "; ".join([
            "High semantic similarity"
            if row["semantic_similarity"] >= SEMANTIC_THRESHOLD
            else "Lower semantic similarity",

            f"Posts published within {TIME_WINDOW_MINUTES} minutes"
            if row["close_in_time"]
            else f"Posts published more than {TIME_WINDOW_MINUTES} minutes apart",

            "Shared URL detected"
            if row["shared_url"]
            else "No shared URL detected"
        ]),
        axis=1
    )

    # Convert the coordination score into an
    # investigation-priority category.
    def priority(score):
        if score >= RISK_HIGH_THRESHOLD:
            return "High"
        elif score >= RISK_MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "Low"

    df["investigation_priority"] = (
        df["coordination_score"].apply(priority)
    )

    # Add a clear interpretation warning.
    df["interpretation"] = (
        "Requires human review; not proof of "
        "malicious coordination or disinformation"
    )

    return df


# ==========================================
# 3. MAIN FUNCTION
# ==========================================

def main():

    if not INPUT_FILE.exists():
        print("Coordination results not found.")
        print("Run: python src/coordination_detection.py")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(INPUT_FILE)

    results = calculate_risk(df)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nExplainable risk scoring completed!")

    if results.empty:
        print("No account pairs are available to score.")
    else:
        print(
            results[
                [
                    "account_1",
                    "account_2",
                    "coordination_score",
                    "investigation_priority",
                    "evidence"
                ]
            ].to_string(index=False)
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()