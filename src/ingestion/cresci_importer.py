
"""Memory-conscious importer for local Cresci tweet datasets."""

from pathlib import Path
import argparse
import csv

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw" / "archieves"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cresci_posts_sample.csv"

SOURCE_GROUPS = {
    "E13": "bot",
    "FSF": "bot",
    "INT": "bot",
    "TFP": "genuine",
    "TWT": "bot",
    "fake_followers": "bot",
    "social_spambots_1": "bot",
    "social_spambots_2": "bot",
    "social_spambots_3": "bot",
    "traditional_spambots_1": "bot",
    "traditional_spambots_2": "bot",
    "traditional_spambots_3": "bot",
    "traditional_spambots_4": "bot",
    "genuine_accounts": "genuine",
}


def find_tweet_files(root: Path) -> list[Path]:
    """Find actual tweets.csv files, excluding macOS metadata."""
    return sorted(
        path for path in root.rglob("tweets.csv")
        if "__MACOSX" not in path.parts
    )


def import_cresci(
    root: Path,
    output: Path,
    limit: int = 1000,
    chunksize: int = 5000,
) -> int:
    """Import at most `limit` rows into NexTrace's post schema."""
    if limit < 1 or chunksize < 1:
        raise ValueError("limit and chunksize must be positive.")

    files = find_tweet_files(root)
    if not files:
        raise FileNotFoundError(f"No tweets.csv files found under {root}")

    output.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    seen_ids = set()
    first_write = True

    output_columns = [
        "post_id", "account_id", "text", "timestamp", "url",
        "source_group", "source_label",
    ]

    for file in files:
        if total_written >= limit:
            break

        # Identify the dataset group from the path.
        group = file.parent.name
        group = group.removesuffix(".csv")

        # Some Cresci-2017 folders contain an extra nested directory.
        if group not in SOURCE_GROUPS:
            for part in reversed(file.parts):
                candidate = part.removesuffix(".csv")
                if candidate in SOURCE_GROUPS:
                    group = candidate
                    break

        label = SOURCE_GROUPS.get(group, "unknown")

        # Use the CSV parser's supported encoding and skip malformed rows.
        try:
            reader = pd.read_csv(
                file,
                encoding="latin1",
                chunksize=chunksize,
                usecols=lambda column: column in {
                    "id", "user_id", "text", "timestamp", "created_at"
                },
                dtype=str,
                on_bad_lines="skip",
                quoting=csv.QUOTE_MINIMAL,
            )

            for chunk in reader:
                if total_written >= limit:
                    break

                if "id" not in chunk or "user_id" not in chunk or "text" not in chunk:
                    print(f"Skipping unsupported file: {file}")
                    break

                time_column = (
                    "timestamp" if "timestamp" in chunk.columns
                    else "created_at" if "created_at" in chunk.columns
                    else None
                )

                if time_column is None:
                    print(f"Skipping file without timestamp: {file}")
                    break

                frame = pd.DataFrame({
                    "post_id": chunk["id"],
                    "account_id": chunk["user_id"],
                    "text": chunk["text"],
                    "timestamp": pd.to_datetime(
                        chunk[time_column], errors="coerce", utc=True
                    ),
                    "url": "",
                    "source_group": group,
                    "source_label": label,
                })

                frame = frame.dropna(
                    subset=["post_id", "account_id", "text", "timestamp"]
                )

                frame["post_id"] = frame["post_id"].str.strip()
                frame["account_id"] = frame["account_id"].str.strip()
                frame["text"] = frame["text"].str.strip()

                frame = frame[
                    (frame["post_id"] != "")
                    & (frame["account_id"] != "")
                    & (frame["text"] != "")
                ]

                # Avoid duplicate IDs in the imported sample.
                frame = frame[~frame["post_id"].isin(seen_ids)]
                if frame.empty:
                    continue

                remaining = limit - total_written
                frame = frame.head(remaining)

                seen_ids.update(frame["post_id"].tolist())

                frame.to_csv(
                    output,
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False,
                    encoding="utf-8-sig" if first_write else "utf-8",
                )

                first_write = False
                total_written += len(frame)

                print(
                    f"Imported {total_written:,}/{limit:,} rows; "
                    f"current source: {group}"
                )

        except Exception as exc:
            print(f"Could not read {file}: {exc}")

    if total_written == 0:
        raise RuntimeError("No valid posts were imported.")

    print(f"\nImported posts: {total_written:,}")
    print(f"Output: {output}")
    print("This is an imported sample, not the full Cresci dataset.")

    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Import a manageable sample of local Cresci tweets."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=5000)
    args = parser.parse_args()

    import_cresci(
        root=args.root,
        output=args.output,
        limit=args.limit,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()

