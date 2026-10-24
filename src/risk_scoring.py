import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.config import (
    SEMANTIC_THRESHOLD,
    TIME_WINDOW_MINUTES,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = _PROJECT_ROOT

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "reports"
    / "coordination_results.csv"
)

ACCOUNT_BEHAVIOR_FILE = (
    BASE_DIR
    / "results"
    / "reports"
    / "account_behavior_results.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "explainable_risk_results.csv"


# ==========================================
# 2. RISK SCORING
# ==========================================

def calculate_risk(
    df: pd.DataFrame,
    account_behavior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create an explainable investigation-priority score.

    Keeps model classification probability separate from coordination scores.
    Transparently details contributing evidence signals and limitations.
    """
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

    # Look up bot probabilities if account behavior data is available
    bot_prob_lookup: dict[str, float] = {}
    if account_behavior is not None and not account_behavior.empty:
        if "account_id" in account_behavior.columns and "bot_probability" in account_behavior.columns:
            for _, r in account_behavior.iterrows():
                if pd.notna(r.get("bot_probability")):
                    try:
                        bot_prob_lookup[str(r["account_id"])] = float(r["bot_probability"])
                    except (ValueError, TypeError):
                        pass

    # Attach account bot probabilities if present
    if bot_prob_lookup:
        df["account_1_bot_probability"] = df["account_1"].astype(str).map(bot_prob_lookup)
        df["account_2_bot_probability"] = df["account_2"].astype(str).map(bot_prob_lookup)

    # Explain the evidence for each pair.
    def build_evidence(row: pd.Series) -> str:
        signals = [
            "High semantic similarity"
            if row["semantic_similarity"] >= SEMANTIC_THRESHOLD
            else "Lower semantic similarity",

            f"Posts published within {TIME_WINDOW_MINUTES} minutes"
            if row["close_in_time"]
            else f"Posts published more than {TIME_WINDOW_MINUTES} minutes apart",

            "Shared URL detected"
            if row["shared_url"]
            else "No shared URL detected"
        ]

        # Add ML bot signal context if available, clearly separated from coordination score
        acc1 = str(row["account_1"])
        acc2 = str(row["account_2"])
        p1 = bot_prob_lookup.get(acc1)
        p2 = bot_prob_lookup.get(acc2)
        if p1 is not None and p2 is not None:
            signals.append(
                f"Bot resemblance signals: {acc1}={p1:.1%}, {acc2}={p2:.1%}"
            )

        return "; ".join(signals)

    df["evidence"] = df.apply(build_evidence, axis=1)

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

    # Add detector version and explicit limitations
    df["detector_version"] = "NexTrace-v1.1"
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

    # Load account behavior with bot probability if available
    account_behavior_df = None
    if ACCOUNT_BEHAVIOR_FILE.exists():
        try:
            account_behavior_df = pd.read_csv(ACCOUNT_BEHAVIOR_FILE)
        except Exception:
            pass

    results = calculate_risk(df, account_behavior=account_behavior_df)

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