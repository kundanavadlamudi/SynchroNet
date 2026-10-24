"""Account-level feature engineering for bot classification."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


IDENTIFIER_COLUMNS = {
    "account_id",
    "user_id",
    "id",
    "username",
    "screen_name",
    "name",
    "label",
    "label_name",
    "source_dataset",
    "source_group",
    "dataset_provenance",
    "profile_image_url",
    "location",
    "description",
    "created_at",
    "lang",
    "time_zone",
    "url",
    "test_set_1",
    "test_set_2",
    "dataset",
    "updated",
    "crawled_at",
}

CORE_FEATURE_COLUMNS = [
    "total_posts",
    "average_text_length",
    "average_word_count",
    "posts_with_urls",
    "unique_texts",
    "duplicate_text_posts",
    "url_post_ratio",
    "duplicate_text_ratio",
]

OPTIONAL_PROFILE_COLUMNS = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
    "friends_followers_ratio",
    "description_length",
    "default_profile",
    "geo_enabled",
    "verified",
    "protected",
    "average_time_between_posts_minutes",
]

URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


def _safe_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _row_has_url(row: pd.Series) -> bool:
    url_value = _safe_text(row.get("url", ""))
    if url_value:
        return True

    if "num_urls" in row.index and pd.notna(row.get("num_urls")):
        try:
            if float(row.get("num_urls")) > 0:
                return True
        except (TypeError, ValueError):
            pass

    return bool(URL_PATTERN.search(_safe_text(row.get("text", ""))))


def _has_url_series(df: pd.DataFrame) -> pd.Series:
    has_url = pd.Series(False, index=df.index)
    if "url" in df.columns:
        url_text = df["url"].map(_safe_text)
        has_url = has_url | url_text.ne("")
    if "num_urls" in df.columns:
        has_url = has_url | (pd.to_numeric(df["num_urls"], errors="coerce").fillna(0) > 0)
    if "text" in df.columns:
        has_url = has_url | df["text"].fillna("").astype(str).str.contains(
            URL_PATTERN, na=False
        )
    return has_url


def _finalize_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    summary["total_posts"] = pd.to_numeric(summary["total_posts"], errors="coerce").fillna(0)
    summary["posts_with_urls"] = pd.to_numeric(
        summary["posts_with_urls"], errors="coerce"
    ).fillna(0)
    summary["unique_texts"] = pd.to_numeric(
        summary["unique_texts"], errors="coerce"
    ).fillna(0)
    summary["duplicate_text_posts"] = pd.to_numeric(
        summary["duplicate_text_posts"], errors="coerce"
    ).fillna(0)
    summary["average_text_length"] = pd.to_numeric(
        summary["average_text_length"], errors="coerce"
    ).fillna(0)
    summary["average_word_count"] = pd.to_numeric(
        summary["average_word_count"], errors="coerce"
    ).fillna(0)

    nonzero_posts = summary["total_posts"].replace(0, pd.NA)
    summary["url_post_ratio"] = (summary["posts_with_urls"] / nonzero_posts).fillna(0)
    summary["duplicate_text_ratio"] = (
        summary["duplicate_text_posts"] / nonzero_posts
    ).fillna(0)
    return summary


def empty_account_features(account_ids: Iterable[str] | None = None) -> pd.DataFrame:
    ids = list(account_ids or [])
    frame = pd.DataFrame({"account_id": pd.Series(ids, dtype="string")})

    for column in CORE_FEATURE_COLUMNS:
        frame[column] = pd.NA

    return frame


def extract_account_features(posts: pd.DataFrame) -> pd.DataFrame:
    """Build reusable account-level posting features.

    Expected columns: account_id, text.
    Optional columns: url, num_urls, timestamp, post_id.
    """
    if posts is None or posts.empty:
        return pd.DataFrame(columns=["account_id", *CORE_FEATURE_COLUMNS])

    if "account_id" not in posts.columns:
        raise ValueError("Posts must include an account_id column.")

    df = posts.copy()
    df["account_id"] = df["account_id"].astype("string").str.strip()
    df = df[df["account_id"].notna() & df["account_id"].ne("")]
    if "text" in df.columns:
        df["text"] = df["text"].map(_safe_text)
    else:
        df["text"] = ""
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].map(_word_count)
    df["has_url"] = _has_url_series(df)

    grouped = df.groupby("account_id", dropna=False)

    summary = grouped.agg(
        total_posts=("account_id", "size"),
        average_text_length=("text_length", "mean"),
        average_word_count=("word_count", "mean"),
        posts_with_urls=("has_url", "sum"),
        unique_texts=("text", "nunique"),
    ).reset_index()
    summary["duplicate_text_posts"] = (
        summary["total_posts"] - summary["unique_texts"]
    )
    summary = _finalize_ratios(summary)

    if "timestamp" in df.columns:
        timed = df.copy()
        timed["timestamp"] = pd.to_datetime(
            timed["timestamp"],
            errors="coerce",
            utc=True,
        )
        timed = timed.dropna(subset=["timestamp"])
        time_bounds = timed.groupby("account_id")["timestamp"].agg(["min", "max", "size"])
        duration_minutes = (
            (time_bounds["max"] - time_bounds["min"]).dt.total_seconds() / 60
        )
        gaps = duration_minutes / (time_bounds["size"] - 1).replace(0, pd.NA)
        average_gaps = gaps.rename("average_time_between_posts_minutes").reset_index()
        summary = summary.merge(average_gaps, on="account_id", how="left")

    return summary


def extract_account_features_from_tweets_csv(
    path: Path,
    rename_map: dict[str, str],
    chunksize: int = 80_000,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Stream a large tweets.csv file into account-level features."""
    usecols = list(rename_map.keys())
    dtype = {}
    for original, target in rename_map.items():
        if target in {"account_id", "post_id", "text", "url"}:
            dtype[original] = "string"

    aggregations = []
    unique_parts = []
    reader = pd.read_csv(
        path,
        encoding=encoding,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=True,
        on_bad_lines="skip",
    )

    for chunk in reader:
        chunk = chunk.rename(columns=rename_map)
        if "account_id" not in chunk.columns:
            continue
        if "timestamp" not in chunk.columns and "created_at" in chunk.columns:
            chunk["timestamp"] = chunk["created_at"]
        annotated = chunk.copy()
        annotated["account_id"] = annotated["account_id"].astype("string").str.strip()
        annotated = annotated[
            annotated["account_id"].notna() & annotated["account_id"].ne("")
        ]
        if annotated.empty:
            continue
        if "text" in annotated.columns:
            annotated["text"] = annotated["text"].map(_safe_text)
        else:
            annotated["text"] = ""
        annotated["text_length"] = annotated["text"].str.len()
        annotated["word_count"] = annotated["text"].str.split().str.len()
        annotated["has_url"] = _has_url_series(annotated)
        annotated["text_hash"] = pd.util.hash_pandas_object(
            annotated["text"], index=False
        )
        if "timestamp" in annotated.columns:
            annotated["timestamp"] = pd.to_datetime(
                annotated["timestamp"], errors="coerce", utc=True
            )

        grouped = annotated.groupby("account_id", dropna=False)
        agg = grouped.agg(
            total_posts=("account_id", "size"),
            text_length_sum=("text_length", "sum"),
            word_count_sum=("word_count", "sum"),
            posts_with_urls=("has_url", "sum"),
        )
        if "timestamp" in annotated.columns:
            time_bounds = grouped["timestamp"].agg(["min", "max"])
            agg = agg.join(time_bounds)
        aggregations.append(agg.reset_index())
        unique_parts.append(
            annotated[["account_id", "text_hash"]].drop_duplicates()
        )

    if not aggregations:
        return pd.DataFrame(columns=["account_id", *CORE_FEATURE_COLUMNS])

    combined = pd.concat(aggregations, ignore_index=True)
    grouped = combined.groupby("account_id", dropna=False)
    summary = grouped.agg(
        total_posts=("total_posts", "sum"),
        text_length_sum=("text_length_sum", "sum"),
        word_count_sum=("word_count_sum", "sum"),
        posts_with_urls=("posts_with_urls", "sum"),
    )
    if "min" in combined.columns:
        summary["timestamp_min"] = grouped["min"].min()
        summary["timestamp_max"] = grouped["max"].max()
    summary = summary.reset_index()

    unique_texts = (
        pd.concat(unique_parts, ignore_index=True)
        .drop_duplicates()
        .groupby("account_id")
        .size()
        .rename("unique_texts")
        .reset_index()
    )
    summary = summary.merge(unique_texts, on="account_id", how="left")
    summary["unique_texts"] = summary["unique_texts"].fillna(0)
    summary["duplicate_text_posts"] = (
        summary["total_posts"] - summary["unique_texts"]
    ).clip(lower=0)
    nonzero = summary["total_posts"].replace(0, pd.NA)
    summary["average_text_length"] = (
        summary["text_length_sum"] / nonzero
    ).fillna(0)
    summary["average_word_count"] = (summary["word_count_sum"] / nonzero).fillna(0)
    if "timestamp_min" in summary.columns:
        duration = (
            summary["timestamp_max"] - summary["timestamp_min"]
        ).dt.total_seconds() / 60
        summary["average_time_between_posts_minutes"] = (
            duration / (summary["total_posts"] - 1).replace(0, pd.NA)
        )
        summary = summary.drop(columns=["timestamp_min", "timestamp_max"])
    summary = summary.drop(columns=["text_length_sum", "word_count_sum"])
    return _finalize_ratios(summary)


def add_profile_features(
    accounts: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Merge optional profile features when a users table is available."""
    if users is None or users.empty:
        return accounts

    profile = users.copy()
    if "account_id" not in profile.columns:
        if "id" in profile.columns:
            profile["account_id"] = profile["id"]
        elif "user_id" in profile.columns:
            profile["account_id"] = profile["user_id"]
        else:
            return accounts

    profile["account_id"] = profile["account_id"].astype("string").str.strip()

    keep = ["account_id"]
    numeric_source_columns = [
        "statuses_count",
        "followers_count",
        "friends_count",
        "favourites_count",
        "listed_count",
    ]
    boolean_source_columns = [
        "default_profile",
        "geo_enabled",
        "verified",
        "protected",
    ]

    for column in numeric_source_columns + boolean_source_columns:
        if column in profile.columns:
            keep.append(column)

    if "description" in profile.columns:
        profile["description_length"] = profile["description"].map(
            lambda value: len(_safe_text(value))
        )
        keep.append("description_length")

    profile = profile[keep].drop_duplicates(subset=["account_id"], keep="first")

    for column in numeric_source_columns:
        if column in profile.columns:
            profile[column] = pd.to_numeric(profile[column], errors="coerce")

    for column in boolean_source_columns:
        if column in profile.columns:
            profile[column] = pd.to_numeric(
                profile[column].map(_coerce_boolean),
                errors="coerce",
            )

    if (
        "friends_count" in profile.columns
        and "followers_count" in profile.columns
    ):
        followers = profile["followers_count"].fillna(0).clip(lower=0)
        friends = profile["friends_count"].fillna(0)
        profile["friends_followers_ratio"] = friends / followers.replace(0, pd.NA)
        profile["friends_followers_ratio"] = profile[
            "friends_followers_ratio"
        ].fillna(0)

    merged = accounts.merge(profile, on="account_id", how="left")
    return merged


def _coerce_boolean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return 1
    if text in {"0", "false", "f", "no"}:
        return 0
    try:
        return int(bool(int(float(text))))
    except (TypeError, ValueError):
        return pd.NA


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding identifiers and labels."""
    excluded = {column.lower() for column in IDENTIFIER_COLUMNS}
    feature_columns = []
    for column in df.columns:
        if column.lower() in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)
    return feature_columns


def select_model_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    columns = list(feature_columns or model_feature_columns(df))
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    features = df[columns].copy()
    for column in columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features
