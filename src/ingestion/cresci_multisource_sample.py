
from pathlib import Path
import pandas as pd

from cresci_importer import find_tweet_files, SOURCE_GROUPS

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "archieves"
OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "cresci_posts_multisource.csv"
)

POSTS_PER_GROUP = 200
MAX_POSTS_PER_ACCOUNT = 5
CHUNK_SIZE = 5000


def main():
    collected = []
    seen_ids = set()
    counts = {}
    groups_found = set()

    files = find_tweet_files(ROOT)

    for file in files:
        group = file.parent.name.removesuffix(".csv")

        if group not in SOURCE_GROUPS:
            for part in reversed(file.parts):
                candidate = part.removesuffix(".csv")
                if candidate in SOURCE_GROUPS:
                    group = candidate
                    break

        if group not in SOURCE_GROUPS:
            continue

        # Collect a limited number from each source group.
        if counts.get(group, 0) >= POSTS_PER_GROUP:
            continue

        label = SOURCE_GROUPS[group]

        try:
            reader = pd.read_csv(
                file,
                encoding="latin1",
                chunksize=CHUNK_SIZE,
                dtype=str,
                usecols=lambda c: c in {
                    "id", "user_id", "text", "timestamp", "created_at"
                },
                on_bad_lines="skip",
            )

            account_counts = {}

            for chunk in reader:
                if counts.get(group, 0) >= POSTS_PER_GROUP:
                    break

                if not {"id", "user_id", "text"}.issubset(chunk.columns):
                    break

                time_col = (
                    "timestamp" if "timestamp" in chunk.columns
                    else "created_at" if "created_at" in chunk.columns
                    else None
                )

                if time_col is None:
                    break

                chunk["timestamp"] = pd.to_datetime(
                    chunk[time_col], errors="coerce", utc=True
                )

                chunk = chunk.dropna(
                    subset=["id", "user_id", "text", "timestamp"]
                )

                for row in chunk.itertuples(index=False, name=None):
                    record = dict(zip(chunk.columns, row))

                    post_id = str(record["id"]).strip()
                    account_id = str(record["user_id"]).strip()
                    text = str(record["text"]).strip()

                    if not post_id or not account_id or not text:
                        continue

                    if post_id in seen_ids:
                        continue

                    if account_counts.get(account_id, 0) >= MAX_POSTS_PER_ACCOUNT:
                        continue

                    collected.append({
                        "post_id": post_id,
                        "account_id": account_id,
                        "text": text,
                        "timestamp": record["timestamp"],
                        "url": "",
                        "source_group": group,
                        "source_label": label,
                    })

                    seen_ids.add(post_id)
                    account_counts[account_id] = (
                        account_counts.get(account_id, 0) + 1
                    )
                    counts[group] = counts.get(group, 0) + 1
                    groups_found.add(group)

                    if counts[group] >= POSTS_PER_GROUP:
                        break

        except Exception as exc:
            print(f"Could not read {file}: {exc}")

    if not collected:
        raise RuntimeError("No posts were collected.")

    df = pd.DataFrame(collected)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\nMULTI-SOURCE SAMPLE CREATED")
    print("Total posts:", len(df))
    print("Unique accounts:", df["account_id"].nunique())
    print("Source groups:", df["source_group"].nunique())
    print("\nPosts by source:")
    print(df["source_group"].value_counts().to_string())
    print("\nOutput:", OUTPUT)


if __name__ == "__main__":
    main()

