import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "results" / "reports" / "coordination_results.csv"

GRAPH_DIR = BASE_DIR / "results" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_IMAGE = GRAPH_DIR / "coordination_graph.png"
GRAPH_DATA = GRAPH_DIR / "coordination_graph.graphml"


def build_graph(df):
    """Build an account coordination graph from detected pairs."""

    graph = nx.Graph()

    for _, row in df.iterrows():
        account1 = str(row["account_1"])
        account2 = str(row["account_2"])

        score = float(row["coordination_score"])

        graph.add_edge(
            account1,
            account2,
            weight=score,
            text_similarity=float(row["text_similarity"]),
            time_difference_minutes=float(
                row["time_difference_minutes"]
            ),
            shared_url=bool(row["shared_url"])
        )

    return graph


def draw_graph(graph):
    """Visualize the coordination graph."""

    plt.figure(figsize=(12, 8))

    if graph.number_of_nodes() == 0:
        print("No account connections are available to visualize.")
        plt.close()
        return

    positions = nx.spring_layout(graph, seed=42)

    # Larger nodes indicate accounts with more connections
    degrees = dict(graph.degree())
    node_sizes = [
        500 + 250 * degrees[node]
        for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        width=[
            1 + 3 * graph[u][v]["weight"]
            for u, v in graph.edges()
        ],
        alpha=0.7
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9
    )

    plt.title("NexTrace - Potential Account Coordination Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_IMAGE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Graph image saved to: {GRAPH_IMAGE}")


def main():
    if not INPUT_FILE.exists():
        print("Coordination results not found.")
        print("Run: python src/coordination_detection.py")
        return

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "account_1",
        "account_2",
        "coordination_score",
        "text_similarity",
        "time_difference_minutes",
        "shared_url"
    }

    missing = required_columns - set(df.columns)

    if missing:
        print(f"Missing required columns: {sorted(missing)}")
        return

    graph = build_graph(df)

    # Save graph in a portable format
    nx.write_graphml(graph, GRAPH_DATA)

    print("\nCoordination graph summary")
    print("--------------------------")
    print(f"Accounts (nodes): {graph.number_of_nodes()}")
    print(f"Connections (edges): {graph.number_of_edges()}")
    print(f"Connected groups: {nx.number_connected_components(graph)}")

    draw_graph(graph)


if __name__ == "__main__":
    main()