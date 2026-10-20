import pandas as pd
import networkx as nx
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

COORDINATION_FILE = (
    BASE_DIR
    / "results"
    / "reports"
    / "coordination_results.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "reports"

OUTPUT_FILE = OUTPUT_DIR / "campaign_analysis_results.csv"


# ==========================================
# 2. CAMPAIGN ANALYSIS
# ==========================================

def analyze_campaigns(df):
    """Group connected accounts and summarize evidence."""

    graph = nx.Graph()

    if df.empty:
        return pd.DataFrame()

    required_columns = {
        "account_1",
        "account_2",
        "coordination_score"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Create an account graph.
    for _, row in df.iterrows():

        account1 = str(row["account_1"])
        account2 = str(row["account_2"])

        score = float(row["coordination_score"])

        if account1 == account2:
            continue

        graph.add_edge(
            account1,
            account2,
            weight=score
        )

    campaign_results = []

    # Each connected component is a candidate group.
    components = list(
        nx.connected_components(graph)
    )

    for campaign_number, accounts in enumerate(
        components,
        start=1
    ):

        accounts = sorted(accounts)

        # Find edges within this candidate group.
        subgraph = graph.subgraph(accounts)

        scores = [
            attributes["weight"]
            for _, _, attributes in subgraph.edges(data=True)
        ]

        campaign_results.append({
            "campaign_id": f"CAMP_{campaign_number:03d}",
            "accounts": ", ".join(accounts),
            "number_of_accounts": len(accounts),
            "number_of_connections": subgraph.number_of_edges(),
            "average_coordination_score": round(
                sum(scores) / len(scores), 4
            ) if scores else 0.0,
            "maximum_coordination_score": round(
                max(scores), 4
            ) if scores else 0.0
        })

    return pd.DataFrame(campaign_results)


# ==========================================
# 3. MAIN FUNCTION
# ==========================================

def main():

    if not COORDINATION_FILE.exists():

        print("Coordination results not found.")
        print("Run: python src/coordination_detection.py")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(COORDINATION_FILE)

    results = analyze_campaigns(df)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCampaign analysis completed!")

    if results.empty:

        print(
            "No candidate groups were found in the "
            "coordination results."
        )

    else:

        print(f"Candidate groups found: {len(results)}")
        print(results.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()