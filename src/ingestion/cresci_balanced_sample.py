
from pathlib import Path
import pandas as pd

from cresci_importer import find_tweet_files, SOURCE_GROUPS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "raw" / "archieves"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "cresci_posts_balanced.csv"


def main():
    target_posts = 1000
    max_posts_per_account = 5
    chunk_size = 5000

    selected = []
    seen_post_ids = set()
    account_counts = {}

    files = find_tweet_files(ROOT)

    for file in files:
        group = file.parent.name.removesuffix(".csv")

        if group not in SOURCE_GROUPS:
            for part in reversed(file.parts):
                candidate = part.removesuffix(".csv")
                if candidate in SOURCE_GROUPS:
                    group = candidate
                    break

        label = SOURCE_GROUPS.get(group, "unknown")

        try:
            for chunk in pd.read_csv(
                file,
                encoding="latin1",
                chunksize=chunk_size,
                dtype=str,
                usecols=lambda c: c in {
                    "id", "user_id", "text", "timestamp", "created_at"
                },
                on_bad_lines="skip",
            ):
                time_col = (
                    "timestamp" if "timestamp" in chunk.columns
                    else "created_at" if "created_at" in chunk.columns
                    else None
                )

                if not {"id", "user_id", "text"}.issubset(chunk.columns):
                    break

                if time_col is None:
                    break

                chunk["timestamp"] = pd.to_datetime(
                    chunk[time_col], errors="coerce", utc=True
                )

                chunk = chunk.dropna(
                    subset=["id", "user_id", "text", "timestamp"]
                )

                for row in chunk.itertuples(index=False, name=None):
                    if len(selected) >= target_posts:
                        break

                    record = dict(zip(chunk.columns, row))
                    post_id = str(record["id"]).strip()
                    account_id = str(record["user_id"]).strip()
                    text = str(record["text"]).strip()

                    if not post_id or not account_id or not text:
                        continue

                    if post_id in seen_post_ids:
                        continue

                    if account_counts.get(account_id, 0) >= max_posts_per_account:
                        continue

                    selected.append({
                        "post_id": post_id,
                        "account_id": account_id,
                        "text": text,
                        "timestamp": record["timestamp"],
                        "url": "",
                        "source_group": group,
                        "source_label": label,
                    })

                    seen_post_ids.add(post_id)
                    account_counts[account_id] = (
                        account_counts.get(account_id, 0) + 1
                    )

                if len(selected) >= target_posts:
                    break

        except Exception as exc:
            print(f"Skipped {file}: {exc}")

        if len(selected) >= target_posts:
            break

    if not selected:
        raise RuntimeError("No valid posts were collected.")

    df = pd.DataFrame(selected)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("Posts:", len(df))
    print("Unique accounts:", df["account_id"].nunique())
    print("Source groups:", df["source_group"].nunique())
    print("Output:", OUTPUT)
    print("\nPosts per account:")
    print(df.groupby("account_id").size().describe())


if __name__ == "__main__":
    main()

