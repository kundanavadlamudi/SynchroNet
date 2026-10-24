import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

POSTS_FILE = (
    ROOT / "data" / "processed"
    / "cresci_posts_multisource_cleaned.csv"
)

PAIRS_FILE = (
    ROOT / "results" / "reports"
    / "cresci_coordination_results.csv"
)

OUTPUT_FILE = (
    ROOT / "results" / "reports"
    / "cresci_account_analysis.csv"
)


def main():
    posts = pd.read_csv(
        POSTS_FILE,
        dtype={"post_id": str, "account_id": str},
    )

    pairs = pd.read_csv(
        PAIRS_FILE,
        dtype={"account_1": str, "account_2": str},
    )

    # Count posts and summarize activity for each account.
    account_stats = (
        posts.groupby("account_id")
        .agg(
            total_posts=("post_id", "nunique"),
            source_groups=("source_group", "nunique"),
            source_group=("source_group", "first"),
            source_label=("source_label", "first"),
        )
        .reset_index()
    )

    # Count each account's appearances in flagged relationships.
    appearances = pd.concat([
        pairs[["account_1"]].rename(
            columns={"account_1": "account_id"}
        ),
        pairs[["account_2"]].rename(
            columns={"account_2": "account_id"}
        ),
    ])

    coordination_counts = (
        appearances.groupby("account_id")
        .size()
        .reset_index(name="flagged_pair_count")
    )

    # Count distinct counterpart accounts.
    counterparts = pd.concat([
        pairs[["account_1", "account_2"]].rename(
            columns={
                "account_1": "account_id",
                "account_2": "counterpart",
            }
        ),
        pairs[["account_2", "account_1"]].rename(
            columns={
                "account_2": "account_id",
                "account_1": "counterpart",
            }
        ),
    ])

    distinct_counterparts = (
        counterparts.groupby("account_id")["counterpart"]
        .nunique()
        .reset_index(name="distinct_counterpart_accounts")
    )

    results = (
        account_stats
        .merge(coordination_counts, on="account_id", how="left")
        .merge(distinct_counterparts, on="account_id", how="left")
    )

    results["flagged_pair_count"] = (
        results["flagged_pair_count"].fillna(0).astype(int)
    )

    results["distinct_counterpart_accounts"] = (
        results["distinct_counterpart_accounts"]
        .fillna(0)
        .astype(int)
    )

    results = results.sort_values(
        ["flagged_pair_count", "distinct_counterpart_accounts"],
        ascending=False,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Account analysis completed.")
    print("Accounts analysed:", len(results))
    print(
        "Accounts in flagged pairs:",
        int((results["flagged_pair_count"] > 0).sum()),
    )
    print("Output:", OUTPUT_FILE)

    print("\nTop accounts by flagged pair count:")
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()