import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

PAIRS_FILE = (
    ROOT / "results" / "reports"
    / "cresci_coordination_results.csv"
)

ACCOUNTS_FILE = (
    ROOT / "results" / "reports"
    / "cresci_account_analysis.csv"
)

GRAPH_DIR = ROOT / "results" / "graphs"

PNG_FILE = GRAPH_DIR / "cresci_coordination_graph.png"
GRAPHML_FILE = GRAPH_DIR / "cresci_coordination_graph.graphml"


def main():
    pairs = pd.read_csv(
        PAIRS_FILE,
        dtype={"account_1": str, "account_2": str},
    )

    accounts = pd.read_csv(
        ACCOUNTS_FILE,
        dtype={"account_id": str},
    )

    graph = nx.Graph()

    # Add accounts with their metadata.
    for row in accounts.to_dict("records"):
        account_id = str(row["account_id"])

        graph.add_node(
            account_id,
            source_group=str(row.get("source_group", "unknown")),
            source_label=str(row.get("source_label", "unknown")),
            total_posts=int(row.get("total_posts", 0)),
            flagged_pair_count=int(row.get("flagged_pair_count", 0)),
        )

    # Add unique coordination relationships.
    for row in pairs.to_dict("records"):
        account1 = str(row["account_1"])
        account2 = str(row["account_2"])

        if account1 == account2:
            continue

        graph.add_edge(
            account1,
            account2,
            coordination_score=float(row["coordination_score"]),
            semantic_similarity=float(row["semantic_similarity"]),
            evidence_signals=int(row["evidence_signals"]),
        )

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Export graph data.
    nx.write_graphml(graph, GRAPHML_FILE)

    # Visualize only the connected accounts for readability.
    active_nodes = [
        node for node in graph.nodes
        if graph.degree(node) > 0
    ]

    visible_graph = graph.subgraph(active_nodes).copy()

    plt.figure(figsize=(14, 10))

    if visible_graph.number_of_nodes() > 0:
        positions = nx.spring_layout(
            visible_graph,
            seed=42,
            k=1.0,
        )

        sizes = [
            500 + 150 * visible_graph.degree(node)
            for node in visible_graph.nodes
        ]

        nx.draw_networkx_nodes(
            visible_graph,
            positions,
            node_size=sizes,
            alpha=0.85,
        )

        nx.draw_networkx_edges(
            visible_graph,
            positions,
            width=1.5,
            alpha=0.6,
        )

        nx.draw_networkx_labels(
            visible_graph,
            positions,
            font_size=8,
        )

        edge_labels = {
            (u, v): f"{data['coordination_score']:.2f}"
            for u, v, data in visible_graph.edges(data=True)
        }

        nx.draw_networkx_edge_labels(
            visible_graph,
            positions,
            edge_labels=edge_labels,
            font_size=7,
        )

        plt.title("NexTrace — Candidate Coordination Network")
        plt.axis("off")
    else:
        plt.text(
            0.5, 0.5,
            "No coordination relationships detected",
            ha="center",
            va="center",
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nCresci graph analysis completed.")
    print("All account nodes:", graph.number_of_nodes())
    print("Unique coordination edges:", graph.number_of_edges())
    print("Connected accounts:", len(active_nodes))
    print("Graph components:", nx.number_connected_components(graph))
    print("PNG:", PNG_FILE)
    print("GraphML:", GRAPHML_FILE)


if __name__ == "__main__":
    main()