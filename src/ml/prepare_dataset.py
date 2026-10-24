"""Load labeled account datasets and write a model-ready table.

Bot classification uses documented Genuine/Bot labels from supported
research datasets. It does not use Phase 1 sample posts, coordination
scores, or invented labels.

Supported sources when present on disk:
- Cresci-2015 (The Fake Project)
- Cresci-2017 (MIB / Cresci et al.)

Expected layout (any of these roots is searched recursively):
    data/raw/
    data/raw/archives/
    data/raw/archieves/
    data/external/
    data/datasets/

Category folders may be named TFP.csv or genuine_accounts.csv even when they
are directories. Nested folders such as fake_followers.csv/fake_followers.csv/
are discovered from the users.csv/tweets.csv parent directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.ml.features import (
    CORE_FEATURE_COLUMNS,
    add_profile_features,
    empty_account_features,
    extract_account_features,
    extract_account_features_from_tweets_csv,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent

SEARCH_ROOTS = [
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "raw" / "archives",
    BASE_DIR / "data" / "raw" / "archieves",
    BASE_DIR / "data" / "external",
    BASE_DIR / "data" / "datasets",
]

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "ml_dataset.csv"
STATUS_FILE = OUTPUT_DIR / "ml_dataset_status.json"

LABEL_GENUINE = 0
LABEL_BOT = 1
LABEL_MAPPING = {"Genuine": LABEL_GENUINE, "Bot": LABEL_BOT}

# Cresci et al., "Fame for sale: Efficient detection of fake Twitter
# followers", Decision Support Systems, 2015.
# E13 / TFP are certified genuine accounts. FSF / INT / TWT are fake
# follower accounts sold by vendors.
CRESCI_2015_GROUPS = {
    "e13": LABEL_GENUINE,
    "tfp": LABEL_GENUINE,
    "fsf": LABEL_BOT,
    "int": LABEL_BOT,
    "twt": LABEL_BOT,
}

# Cresci et al., "The paradigm-shift of social spambots", WWW Companion
# 2017 / IEEE. genuine_accounts are certified humans. social_spambots_*,
# traditional_spambots_*, and fake_followers are automated/fake accounts.
CRESCI_2017_GROUPS = {
    "genuine": LABEL_GENUINE,
    "genuine_accounts": LABEL_GENUINE,
    "social_spambots_1": LABEL_BOT,
    "social_spambots_2": LABEL_BOT,
    "social_spambots_3": LABEL_BOT,
    "traditional_spambots_1": LABEL_BOT,
    "traditional_spambots_2": LABEL_BOT,
    "traditional_spambots_3": LABEL_BOT,
    "traditional_spambots_4": LABEL_BOT,
    "fake_followers": LABEL_BOT,
}

TWEET_COLUMN_CANDIDATES = {
    "account_id": ["user_id", "userid", "account_id", "user"],
    "post_id": ["id", "tweet_id", "post_id"],
    "text": ["text", "tweet", "status"],
    "timestamp": ["timestamp", "created_at"],
    "num_urls": ["num_urls", "urls_count"],
    "url": ["url"],
}

USER_COLUMN_CANDIDATES = {
    "id": ["id", "user_id", "userid"],
    "screen_name": ["screen_name", "username"],
    "name": ["name"],
    "statuses_count": ["statuses_count"],
    "followers_count": ["followers_count"],
    "friends_count": ["friends_count"],
    "favourites_count": ["favourites_count", "favorites_count"],
    "listed_count": ["listed_count"],
    "created_at": ["created_at"],
    "description": ["description"],
    "default_profile": ["default_profile"],
    "geo_enabled": ["geo_enabled"],
    "verified": ["verified"],
    "protected": ["protected"],
}


@dataclass
class DatasetSource:
    group_name: str
    source_dataset: str
    label: int
    directory: Path
    users_path: Path | None
    tweets_path: Path | None


@dataclass
class PrepareResult:
    dataset: pd.DataFrame
    provenance: dict
    missing_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    discovered_sources: list[str] = field(default_factory=list)


class LabeledDatasetNotFoundError(FileNotFoundError):
    """Raised when no documented labeled dataset is available."""


def normalize_group_name(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "_")
    if cleaned.endswith(".csv"):
        cleaned = cleaned[:-4]
    if cleaned.endswith(".zip"):
        cleaned = cleaned[:-4]
    return cleaned


def label_name_for(label: int) -> str:
    if label == LABEL_GENUINE:
        return "Genuine"
    if label == LABEL_BOT:
        return "Bot"
    raise ValueError(f"Unsupported label value: {label}")


def map_group_to_label(group_name: str) -> tuple[str, int] | None:
    """Return (source_dataset, label) only for documented groups."""
    group = normalize_group_name(group_name)
    if group in CRESCI_2015_GROUPS:
        return "cresci-2015", CRESCI_2015_GROUPS[group]
    if group in CRESCI_2017_GROUPS:
        return "cresci-2017", CRESCI_2017_GROUPS[group]
    return None


def _read_csv(path: Path, usecols=None) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding)
            available = list(header.columns)
            selected = None
            if usecols:
                selected = [column for column in usecols if column in available]
                if not selected:
                    return pd.DataFrame()
            return pd.read_csv(
                path,
                encoding=encoding,
                usecols=selected,
                low_memory=False,
            )
        except UnicodeDecodeError as error:
            last_error = error
            continue
    raise last_error or UnicodeDecodeError(
        "unknown", b"", 0, 1, f"Could not decode {path}"
    )


def _rename_columns(df: pd.DataFrame, candidates: dict[str, list[str]]) -> pd.DataFrame:
    renamed = df.copy()
    mapping = build_rename_map(renamed.columns, candidates)
    if mapping:
        renamed = renamed.rename(columns=mapping)
    return renamed


def build_rename_map(columns, candidates: dict[str, list[str]]) -> dict[str, str]:
    lower_map = {str(column).strip().lower(): column for column in columns}
    mapping: dict[str, str] = {}
    used_targets = set()
    used_actual = set()
    for target, options in candidates.items():
        if target in used_targets:
            continue
        for option in options:
            actual = lower_map.get(option.lower())
            if actual is not None and actual not in used_actual:
                mapping[actual] = target
                used_targets.add(target)
                used_actual.add(actual)
                break
    return mapping


def detect_csv_header(path: Path) -> tuple[list[str], str]:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding)
            return list(header.columns), encoding
        except UnicodeDecodeError as error:
            last_error = error
            continue
    raise last_error or UnicodeDecodeError(
        "unknown", b"", 0, 1, f"Could not decode {path}"
    )


def discover_labeled_sources(search_roots: list[Path] | None = None) -> list[DatasetSource]:
    roots = search_roots or SEARCH_ROOTS
    discovered: dict[str, DatasetSource] = {}

    for root in roots:
        if root is None or not Path(root).exists():
            continue
        root = Path(root)
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__macosx" in str(path).lower():
                continue
            if path.name.lower() not in {"users.csv", "tweets.csv"}:
                continue

            group_dir = path.parent
            mapped = map_group_to_label(group_dir.name)
            if mapped is None:
                continue
            source_dataset, label = mapped
            key = f"{source_dataset}:{normalize_group_name(group_dir.name)}"
            users_path = group_dir / "users.csv"
            tweets_path = group_dir / "tweets.csv"
            discovered[key] = DatasetSource(
                group_name=normalize_group_name(group_dir.name),
                source_dataset=source_dataset,
                label=label,
                directory=group_dir,
                users_path=users_path if users_path.exists() else None,
                tweets_path=tweets_path if tweets_path.exists() else None,
            )

    return sorted(
        discovered.values(),
        key=lambda source: (source.source_dataset, source.group_name),
    )


def _load_tweet_features(source: DatasetSource) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if source.tweets_path is None:
        return pd.DataFrame(), [
            f"{source.group_name}: tweets.csv not found; post features set to 0"
        ]

    try:
        columns, encoding = detect_csv_header(source.tweets_path)
    except Exception as error:
        return pd.DataFrame(), [
            f"{source.group_name}: could not read tweets.csv header ({error})"
        ]

    rename_map = build_rename_map(columns, TWEET_COLUMN_CANDIDATES)
    if "account_id" not in rename_map.values():
        return pd.DataFrame(), [
            f"{source.group_name}: tweets.csv has no user id column; "
            f"columns={columns}"
        ]

    encodings = [encoding]
    for candidate in ("utf-8", "utf-8-sig", "latin-1"):
        if candidate not in encodings:
            encodings.append(candidate)

    last_error = None
    for current_encoding in encodings:
        try:
            print(
                f"  Streaming tweets for {source.source_dataset}/"
                f"{source.group_name} ({source.tweets_path})"
            )
            features = extract_account_features_from_tweets_csv(
                source.tweets_path,
                rename_map,
                encoding=current_encoding,
            )
            return features, warnings
        except (UnicodeDecodeError, ValueError, OSError, pd.errors.ParserError) as error:
            last_error = error
            continue

    return pd.DataFrame(), [
        f"{source.group_name}: failed to stream tweets.csv ({last_error})"
    ]


def load_source_accounts(source: DatasetSource) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    users = pd.DataFrame()

    if source.users_path is not None:
        users = _rename_columns(_read_csv(source.users_path), USER_COLUMN_CANDIDATES)
        if "id" in users.columns:
            users["account_id"] = users["id"].astype("string").str.strip()
        elif "user_id" in users.columns:
            users["account_id"] = users["user_id"].astype("string").str.strip()

    tweet_features, tweet_warnings = _load_tweet_features(source)
    warnings.extend(tweet_warnings)
    tweet_data_available = not tweet_features.empty

    if tweet_features.empty and users.empty:
        warnings.append(f"{source.group_name}: no usable users or tweets")
        return pd.DataFrame(), warnings

    if tweet_features.empty:
        account_ids = []
        if "account_id" in users.columns:
            account_ids = (
                users["account_id"].dropna().astype("string").str.strip().unique()
            )
        accounts = empty_account_features(account_ids)
    else:
        accounts = tweet_features

    if not users.empty and "account_id" in users.columns:
        accounts = add_profile_features(accounts, users)
        extra_ids = users.loc[
            ~users["account_id"].isin(accounts["account_id"]),
            "account_id",
        ].dropna().astype("string").str.strip()
        extra_ids = extra_ids[extra_ids.ne("")]
        if not extra_ids.empty:
            extra = empty_account_features(extra_ids.unique())
            extra = add_profile_features(extra, users)
            accounts = pd.concat([accounts, extra], ignore_index=True)

    
    accounts = accounts[
        accounts["account_id"].notna() & accounts["account_id"].ne("")
    ]

    if not tweet_features.empty:
        accounts["tweet_data_available"] = (
            accounts["account_id"]
            .isin(tweet_features["account_id"])
            .astype(int)
        )
    else:
        accounts["tweet_data_available"] = 0

    for column in CORE_FEATURE_COLUMNS:
        if column in accounts.columns:
            accounts[column] = pd.to_numeric(
                accounts[column], errors="coerce"
            )
            accounts.loc[
                accounts["tweet_data_available"] == 0,
                column
            ] = pd.NA

    accounts["label"] = source.label
    accounts["label_name"] = label_name_for(source.label)
    accounts["source_dataset"] = source.source_dataset
    accounts["source_group"] = source.group_name
    accounts["dataset_provenance"] = (
        f"{source.source_dataset}/{source.group_name}"
    )
    return accounts, warnings



def combine_account_tables(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if not frames:
        return pd.DataFrame(), warnings

    combined = pd.concat(frames, ignore_index=True)
    combined["account_id"] = combined["account_id"].astype("string").str.strip()
    combined = combined[combined["account_id"].notna() & combined["account_id"].ne("")]
    n_loaded = len(combined)

    label_nunique = combined.groupby("account_id")["label"].nunique()
    conflicts = label_nunique[label_nunique > 1].index
    if len(conflicts) > 0:
        warnings.append(
            f"Dropped {len(conflicts)} accounts with conflicting labels"
        )
        combined = combined[~combined["account_id"].isin(conflicts)]

    n_duplicate_rows = int(combined["account_id"].duplicated().sum())
    if n_duplicate_rows > 0:
        duplicate_groups = (
            combined.loc[combined["account_id"].duplicated(keep=False), "dataset_provenance"]
            if "dataset_provenance" in combined.columns
            else pd.Series(dtype="string")
        )
        extra = ""
        if not duplicate_groups.empty:
            extra = " across " + ", ".join(
                sorted(duplicate_groups.dropna().astype(str).unique())
            )
        warnings.append(
            f"Dropped {n_duplicate_rows} duplicate account rows with the same "
            f"label{extra}. Kept the first occurrence."
        )

    combined = combined.drop_duplicates(subset=["account_id"], keep="first")
    combined = combined.reset_index(drop=True)
    warnings.append(
        f"Combined {n_loaded} labeled account rows into {len(combined)} unique accounts."
    )
    return combined, warnings


def prepare_ml_dataset(
    search_roots: list[Path] | None = None,
    output_file: Path | None = None,
    status_file: Path | None = None,
) -> PrepareResult:
    sources = discover_labeled_sources(search_roots)
    missing = []
    if not any(source.source_dataset == "cresci-2015" for source in sources):
        missing.append("cresci-2015")
    if not any(source.source_dataset == "cresci-2017" for source in sources):
        missing.append("cresci-2017")

    frames = []
    warnings: list[str] = []
    for source in sources:
        accounts, source_warnings = load_source_accounts(source)
        warnings.extend(source_warnings)
        if not accounts.empty:
            frames.append(accounts)

    dataset, combine_warnings = combine_account_tables(frames)
    warnings.extend(combine_warnings)

    provenance = {
        "label_mapping": LABEL_MAPPING,
        "n_accounts": int(len(dataset)),
        "n_genuine": int((dataset["label"] == LABEL_GENUINE).sum()) if not dataset.empty else 0,
        "n_bot": int((dataset["label"] == LABEL_BOT).sum()) if not dataset.empty else 0,
        "class_counts_by_source": class_counts_by_source(dataset),
        "input_files": input_file_inventory(sources),
        "skipped_files": summarize_skipped_files(search_roots or SEARCH_ROOTS),
        "sources": [
            {
                "source_dataset": source.source_dataset,
                "source_group": source.group_name,
                "label": source.label,
                "label_name": label_name_for(source.label),
                "directory": str(source.directory),
                "users_csv": str(source.users_path) if source.users_path else None,
                "tweets_csv": str(source.tweets_path) if source.tweets_path else None,
            }
            for source in sources
        ],
        "missing_sources": missing,
        "excluded_phase1_sample": str(
            BASE_DIR / "data" / "processed" / "sample_social_media_posts.csv"
        ),
        "note": (
            "Phase 1 sample posts are unlabeled synthetic demonstration "
            "data and are not used as training labels."
        ),
    }

    result = PrepareResult(
        dataset=dataset,
        provenance=provenance,
        missing_sources=missing,
        warnings=warnings,
        discovered_sources=[
            f"{source.source_dataset}/{source.group_name}" for source in sources
        ],
    )

    status_path = status_file or STATUS_FILE
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "ready_for_training": bool(
                    not dataset.empty
                    and provenance["n_genuine"] > 0
                    and provenance["n_bot"] > 0
                ),
                "provenance": provenance,
                "warnings": warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if dataset.empty:
        return result

    output_path = output_file or OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    provenance["output_file"] = str(output_path)
    return result


def _file_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return int(path.stat().st_size)


def input_file_inventory(sources: list[DatasetSource]) -> list[dict]:
    inventory = []
    for source in sources:
        inventory.append(
            {
                "source_dataset": source.source_dataset,
                "source_group": source.group_name,
                "users_csv": str(source.users_path) if source.users_path else None,
                "users_csv_bytes": _file_size(source.users_path),
                "tweets_csv": str(source.tweets_path) if source.tweets_path else None,
                "tweets_csv_bytes": _file_size(source.tweets_path),
                "profile_only": source.tweets_path is None,
            }
        )
    return inventory


def class_counts_by_source(dataset: pd.DataFrame) -> list[dict]:
    if dataset.empty or "source_group" not in dataset.columns:
        return []
    grouped = (
        dataset.groupby(["source_dataset", "source_group", "label_name"], dropna=False)
        .size()
        .reset_index(name="n_accounts")
    )
    return grouped.to_dict(orient="records")


def summarize_skipped_files(search_roots: list[Path]) -> dict:
    macosx = 0
    unmapped: list[str] = []
    companion: list[str] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lowered = str(path).lower()
            if "__macosx" in lowered or path.name.startswith("._"):
                macosx += 1
                continue
            if path.name.lower() in {"users.csv", "tweets.csv"}:
                if map_group_to_label(path.parent.name) is None:
                    unmapped.append(str(path))
                continue
            if path.suffix.lower() in {".csv", ".txt", ".md"} or path.name.upper() in {
                "READ.ME",
                "README",
            }:
                companion.append(str(path))
    return {
        "macosx_metadata_files": macosx,
        "unmapped_users_or_tweets": unmapped,
        "companion_or_documentation_files": companion,
    }


def describe_missing_datasets() -> str:
    return (
        "No documented labeled account datasets were found.\n"
        "Place Cresci-2015 and/or Cresci-2017 group folders under "
        "data/raw, data/external, or data/datasets.\n\n"
        "Cresci-2015 groups: TFP and E13 (Genuine=0); "
        "FSF, INT, TWT (Bot=1).\n"
        "Cresci-2017 groups: genuine_accounts (Genuine=0); "
        "social_spambots_1/2/3, traditional_spambots_1/2/3/4, "
        "fake_followers (Bot=1).\n"
        "Each group folder should contain users.csv and tweets.csv.\n"
        "Unlabeled Phase 1 sample posts are not used for training."
    )


def main() -> None:
    result = prepare_ml_dataset()
    print("NexTrace ML dataset preparation")
    print("Bot classification is separate from coordination detection.")

    if result.discovered_sources:
        print("Discovered labeled sources:")
        for name in result.discovered_sources:
            print(f"  - {name}")
    else:
        print("No labeled Cresci group folders were discovered.")

    if result.missing_sources:
        print("Missing datasets:", ", ".join(result.missing_sources))

    for warning in result.warnings:
        print(f"Warning: {warning}")

    if result.dataset.empty:
        print(describe_missing_datasets())
        print(f"Status written to: {STATUS_FILE}")
        return

    print(f"Accounts prepared: {len(result.dataset)}")
    print(f"Genuine: {result.provenance['n_genuine']}")
    print(f"Bot: {result.provenance['n_bot']}")
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Status written to: {STATUS_FILE}")


if __name__ == "__main__":
    main()
