
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "results" / "reports"

COORDINATION_FILE = REPORTS / "cresci_coordination_results.csv"
CAMPAIGNS_FILE = REPORTS / "cresci_campaign_results.csv"
MEMBERS_FILE = REPORTS / "cresci_campaign_members.csv"
OUTPUT_FILE = REPORTS / "cresci_campaign_validation.csv"


def main():
    for file in [COORDINATION_FILE, CAMPAIGNS_FILE, MEMBERS_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found: {file}")

    edges = pd.read_csv(COORDINATION_FILE)
    campaigns = pd.read_csv(CAMPAIGNS_FILE)
    members = pd.read_csv(MEMBERS_FILE)

    required_edge_columns = {
        "account_1", "account_2", "coordination_score",
        "evidence_signals"
    }

    if not required_edge_columns.issubset(edges.columns):
        raise ValueError(
            f"Coordination file must contain: {required_edge_columns}"
        )

    if campaigns.empty:
        print("No candidate campaigns found.")
        return

    results = []

    for _, campaign in campaigns.iterrows():
        campaign_id = campaign.get(
            "campaign_id",
            campaign.get("campaign", None)
        )

        if campaign_id is None:
            raise ValueError(
                "Could not identify campaign ID column. "
                "Check cresci_campaign_results.csv."
            )

        campaign_members = members[
            members["campaign_id"].astype(str)
            == str(campaign_id)
        ]

        member_ids = set(
            campaign_members["account_id"].astype(str)
        )

        campaign_edges = edges[
            edges["account_1"].astype(str).isin(member_ids)
            & edges["account_2"].astype(str).isin(member_ids)
        ]

        signals = []

        if not campaign_edges.empty:
            for value in campaign_edges["evidence_signals"].dropna():
                signals.extend(
                    signal.strip()
                    for signal in str(value).split(",")
                    if signal.strip()
                )

        results.append({
            "campaign_id": campaign_id,
            "member_count": len(member_ids),
            "supporting_post_pairs": len(campaign_edges),
            "unique_account_edges": campaign_edges[
                ["account_1", "account_2"]
            ].apply(
                lambda row: tuple(sorted(
                    [str(row["account_1"]), str(row["account_2"])]
                )),
                axis=1
            ).nunique() if not campaign_edges.empty else 0,
            "average_coordination_score": (
                campaign_edges["coordination_score"].mean()
                if not campaign_edges.empty else 0
            ),
            "evidence_signals": ", ".join(sorted(set(signals))),
            "has_members": len(member_ids) > 0,
            "has_supporting_edges": not campaign_edges.empty,
            "validation_status": (
                "Needs review"
                if len(member_ids) > 0 and not campaign_edges.empty
                else "Insufficient supporting data"
            )
        })

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\nCresci campaign validation completed.")
    print(f"Campaigns checked: {len(report)}")
    print(f"Campaigns with members: {report['has_members'].sum()}")
    print(
        "Campaigns with supporting edges: "
        f"{report['has_supporting_edges'].sum()}"
    )
    print(f"Validation report: {OUTPUT_FILE}")

    print("\nValidation summary:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

