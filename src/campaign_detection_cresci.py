import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import networkx as nx

PAIRS_FILE = (
    ROOT / "results" / "reports"
    / "cresci_coordination_results.csv"
)

ACCOUNTS_FILE = (
    ROOT / "results" / "reports"
    / "cresci_account_analysis.csv"
)

CAMPAIGNS_FILE = (
    ROOT / "results" / "reports"
    / "cresci_campaign_results.csv"
)

MEMBERS_FILE = (
    ROOT / "results" / "reports"
    / "cresci_campaign_members.csv"
)


def main():
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PAIRS_FILE}")

    if not ACCOUNTS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ACCOUNTS_FILE}")

    pairs = pd.read_csv(
        PAIRS_FILE,
        dtype={"account_1": str, "account_2": str},
    )

    accounts = pd.read_csv(
        ACCOUNTS_FILE,
        dtype={"account_id": str},
    )

    graph = nx.Graph()

    # Include all accounts, including accounts with no detected edges.
    graph.add_nodes_from(accounts["account_id"].dropna().astype(str))

    # Each flagged coordination pair becomes an edge.
    for row in pairs.itertuples(index=False):
        if row.account_1 == row.account_2:
            continue

        graph.add_edge(
            str(row.account_1),
            str(row.account_2),
            coordination_score=float(row.coordination_score),
            semantic_similarity=float(row.semantic_similarity),
            evidence_signals=int(row.evidence_signals),
        )

    account_lookup = accounts.set_index("account_id").to_dict("index")

    campaign_rows = []
    member_rows = []

    components = sorted(
        nx.connected_components(graph),
        key=lambda component: (
            -len(component),
            sorted(component)[0],
        ),
    )

    campaign_number = 0

    for component in components:
        # Single accounts with no coordination edges are not campaigns.
        if len(component) < 2:
            continue

        subgraph = graph.subgraph(component)

        campaign_number += 1
        campaign_id = f"CRESCI-CAMPAIGN-{campaign_number:03d}"

        edge_scores = [
            data["coordination_score"]
            for _, _, data in subgraph.edges(data=True)
        ]

        campaign_rows.append({
            "campaign_id": campaign_id,
            "account_count": subgraph.number_of_nodes(),
            "coordination_edge_count": subgraph.number_of_edges(),
            "average_coordination_score": round(
                sum(edge_scores) / len(edge_scores), 4
            ) if edge_scores else 0,
            "max_coordination_score": max(edge_scores, default=0),
            "source_groups": ", ".join(sorted({
                str(account_lookup[node].get("source_group", "unknown"))
                for node in component
                if node in account_lookup
            })),
            "interpretation": "Candidate coordination cluster; requires review",
        })

        for account_id in sorted(component):
            account = account_lookup.get(account_id, {})

            member_rows.append({
                "campaign_id": campaign_id,
                "account_id": account_id,
                "source_group": account.get("source_group", "unknown"),
                "source_label": account.get("source_label", "unknown"),
                "flagged_pair_count": account.get("flagged_pair_count", 0),
                "distinct_counterpart_accounts": account.get(
                    "distinct_counterpart_accounts", 0
                ),
            })

    campaign_df = pd.DataFrame(
        campaign_rows,
        columns=[
            "campaign_id",
            "account_count",
            "coordination_edge_count",
            "average_coordination_score",
            "max_coordination_score",
            "source_groups",
            "interpretation",
        ],
    )

    member_df = pd.DataFrame(
        member_rows,
        columns=[
            "campaign_id",
            "account_id",
            "source_group",
            "source_label",
            "flagged_pair_count",
            "distinct_counterpart_accounts",
        ],
    )

    CAMPAIGNS_FILE.parent.mkdir(parents=True, exist_ok=True)

    campaign_df.to_csv(
        CAMPAIGNS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    member_df.to_csv(
        MEMBERS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nCresci campaign analysis completed.")
    print("Accounts in graph:", graph.number_of_nodes())
    print("Coordination edges:", graph.number_of_edges())
    print("Candidate clusters:", len(campaign_df))
    print("Accounts in candidate clusters:", len(member_df))
    print("Campaign results:", CAMPAIGNS_FILE)
    print("Campaign members:", MEMBERS_FILE)

    if not campaign_df.empty:
        print("\nCandidate campaigns:")
        print(campaign_df.to_string(index=False))


if __name__ == "__main__":
    main()