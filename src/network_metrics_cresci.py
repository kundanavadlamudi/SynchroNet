
from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "results" / "reports"
OUTPUT = REPORTS / "cresci_network_metrics.csv"

EDGES_FILE = REPORTS / "cresci_coordination_results.csv"
ACCOUNTS_FILE = REPORTS / "cresci_account_analysis.csv"


def main():
    edges = pd.read_csv(EDGES_FILE)
    accounts = pd.read_csv(ACCOUNTS_FILE)

    G = nx.Graph()

    # Add all account nodes
    for _, row in accounts.iterrows():
        account_id = str(row["account_id"])
        G.add_node(account_id)

    # Add coordination edges
    for _, row in edges.iterrows():
        a = str(row["account_1"])
        b = str(row["account_2"])

        if a != b:
            G.add_edge(
                a,
                b,
                coordination_score=float(row["coordination_score"])
            )

    # Calculate network metrics
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    components = list(nx.connected_components(G))
    component_map = {}

    for number, component in enumerate(components, start=1):
        for account in component:
            component_map[account] = number

    results = []

    for node in G.nodes():
        results.append({
            "account_id": node,
            "degree": G.degree(node),
            "degree_centrality": degree[node],
            "betweenness_centrality": betweenness[node],
            "component_id": component_map[node],
            "component_size": len(
                next(c for c in components if node in c)
            ),
            "is_connected": G.degree(node) > 0
        })

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT, index=False)

    print("\nCresci network analysis completed.")
    print(f"Total accounts: {G.number_of_nodes()}")
    print(f"Unique coordination edges: {G.number_of_edges()}")
    print(f"Connected accounts: {sum(d > 0 for _, d in G.degree())}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Output: {OUTPUT}")

    print("\nTop 10 accounts by degree centrality:")
    print(
        report.sort_values(
            "degree_centrality", ascending=False
        ).head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()

