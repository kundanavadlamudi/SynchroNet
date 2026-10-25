
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "cresci_coordination_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "cresci_explainable_risk_results.csv"
)


def calculate_risk(df):
    df = df.copy()

    required_columns = [
        "account_1",
        "account_2",
        "coordination_score",
        "semantic_similarity",
        "time_difference_minutes",
        "shared_url",
        "evidence_signals",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["coordination_score"] = pd.to_numeric(
        df["coordination_score"], errors="coerce"
    ).fillna(0).clip(0, 1)

    df["semantic_similarity"] = pd.to_numeric(
        df["semantic_similarity"], errors="coerce"
    ).fillna(0).clip(0, 1)

    df["time_difference_minutes"] = pd.to_numeric(
        df["time_difference_minutes"], errors="coerce"
    )

    df["shared_url"] = (
        df["shared_url"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df["evidence_signals"] = pd.to_numeric(
        df["evidence_signals"], errors="coerce"
    ).fillna(0).astype(int)

    df["close_in_time"] = (
        df["time_difference_minutes"].between(0, 10, inclusive="both")
    )

    def priority(score):
        if score >= 0.80:
            return "High"
        elif score >= 0.60:
            return "Medium"
        return "Low"

    df["investigation_priority"] = (
        df["coordination_score"].apply(priority)
    )

    def evidence(row):
        signals = []

        if row["semantic_similarity"] >= 0.70:
            signals.append("Similar text")

        if row["close_in_time"]:
            signals.append("Posts published close together")

        if row["shared_url"]:
            signals.append("Shared URL")

        return "; ".join(signals) if signals else "Insufficient evidence"

    df["evidence"] = df.apply(evidence, axis=1)

    df["detector_version"] = "cresci-coordination-v1"

    df["interpretation"] = df["investigation_priority"].map(
        {
            "High": (
                "High-priority coordination signal; "
                "review the evidence before drawing conclusions."
            ),
            "Medium": (
                "Moderate coordination signal; "
                "manual investigation is recommended."
            ),
            "Low": (
                "Low coordination signal; "
                "additional evidence may be needed."
            ),
        }
    )

    return df


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Coordination results not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)
    risk_df = calculate_risk(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(OUTPUT_FILE, index=False)

    print("Cresci risk scoring completed.")
    print(f"Input coordination pairs: {len(df)}")
    print(f"Risk records saved: {len(risk_df)}")
    print(f"High priority: {(risk_df['investigation_priority'] == 'High').sum()}")
    print(f"Medium priority: {(risk_df['investigation_priority'] == 'Medium').sum()}")
    print(f"Low priority: {(risk_df['investigation_priority'] == 'Low').sum()}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
