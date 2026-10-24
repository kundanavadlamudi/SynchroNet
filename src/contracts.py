"""NexTrace — Unified Data Contracts and Schema Validation.

Defines schemas and normalization boundaries for:
1. Posts (post_id, account_id, text, timestamp, url, source_platform, provenance, collection_time, engagement)
2. Accounts (account_id, source_platform, metadata, behavioral signals, provenance)
3. Analysis Results (subjects, scores, detector version, signals used, evidence, limitations)
4. Campaign Candidates (campaign_id, member accounts, connections, scores, time window, review_status)

Designed to handle malformed input gracefully without crashing the pipeline.
Missing values are safely defaulted and never silently converted into high-risk signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import pandas as pd


@dataclass
class PostRecord:
    post_id: str
    account_id: str
    text: str
    timestamp: str  # ISO 8601 UTC string
    url: str = ""
    source_platform: str = "unknown"
    provenance: str = "unknown"
    collection_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    engagement_likes: int = 0
    engagement_shares: int = 0
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AccountRecord:
    account_id: str
    source_platform: str = "unknown"
    provenance: str = "unknown"
    collection_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    followers_count: int = 0
    friends_count: int = 0
    statuses_count: int = 0
    favourites_count: int = 0
    listed_count: int = 0
    default_profile: int = 0
    geo_enabled: int = 0
    verified: int = 0
    protected: int = 0
    description_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResultRecord:
    subject_1: str
    subject_2: str
    post_1: str
    post_2: str
    detector_version: str
    signals_used: str
    coordination_score: float
    investigation_priority: str
    evidence: str
    confidence_level: str
    created_at: str
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignRecord:
    campaign_id: str
    accounts: str
    number_of_accounts: int
    number_of_connections: int
    average_coordination_score: float
    maximum_coordination_score: float
    time_window_start: str | None = None
    time_window_end: str | None = None
    review_status: str = "Requires Human Review"
    limitations: str = "Connected component heuristic; does not prove coordinated intent"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NewsArticleRecord:
    url: str
    title: str
    seen_date: str = ""
    domain: str = ""
    language: str = "en"
    source_country: str = ""
    query_term: str = ""
    collection_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provenance: str = "GDELT DOC 2.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactCheckRecord:
    query_term: str
    claim_text: str
    claimant: str = ""
    claim_date: str = ""
    publisher_name: str = ""
    publisher_site: str = ""
    review_url: str = ""
    review_title: str = ""
    review_date: str = ""
    textual_rating: str = ""
    collection_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provenance: str = "Google Fact Check Tools API"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomainInvestigationRecord:
    domain: str
    registrar: str = ""
    registration_date: str = ""
    last_changed_date: str = ""
    status: list[str] = field(default_factory=list)
    nameservers: list[str] = field(default_factory=list)
    collection_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provenance: str = "RDAP"
    evidence_note: str = (
        "Contextual domain registration data only; does not establish malicious attribution or intent."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OSINTEnrichmentBundle:
    target_id: str  # post_id or campaign_id
    target_type: str  # "post" or "campaign"
    matched_news: list[dict[str, Any]] = field(default_factory=list)
    matched_fact_checks: list[dict[str, Any]] = field(default_factory=list)
    domain_investigations: list[dict[str, Any]] = field(default_factory=list)
    ocr_extracted_text: str = ""
    enrichment_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    limitations: str = (
        "OSINT evidence provides external context and public reporting; absence of fact-checks does "
        "not imply truthfulness, and external ratings remain the sole attribution of original publishers."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_timestamp(val: Any) -> pd.Timestamp | None:
    """Normalize any date/time string or object to a UTC Timestamp, or None if invalid."""
    if pd.isna(val):
        return None
    try:
        ts = pd.to_datetime(val, utc=True)
        return ts if not pd.isna(ts) else None
    except Exception:
        return None


def validate_and_normalize_posts(
    df: pd.DataFrame,
    default_source: str = "unknown",
    default_provenance: str = "unspecified",
) -> tuple[pd.DataFrame, list[str]]:
    """Validate, clean, and normalize post records into the canonical contract.

    Returns:
        tuple of (cleaned_df, validation_warnings)
    """
    warnings: list[str] = []
    if df.empty:
        warnings.append("Input DataFrame is empty.")
        empty_df = pd.DataFrame(columns=[
            "post_id", "account_id", "text", "timestamp", "url",
            "source_platform", "provenance", "collection_time"
        ])
        return empty_df, warnings

    df = df.copy()

    # Required identifiers and core fields
    core_cols = ["post_id", "account_id", "text", "timestamp"]
    missing_core = [col for col in core_cols if col not in df.columns]
    if missing_core:
        raise ValueError(f"Input records missing mandatory columns: {missing_core}")

    initial_count = len(df)

    # Standardize string fields
    for col in ["post_id", "account_id", "text"]:
        df[col] = df[col].astype(str).str.strip()

    # Drop blank or NaN essential identifiers
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(subset=["post_id", "account_id", "text"])

    dropped_missing = initial_count - len(df)
    if dropped_missing > 0:
        warnings.append(f"Dropped {dropped_missing} rows missing post_id, account_id, or text.")

    # Deduplicate post_id
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["post_id"], keep="first")
    if len(df) < pre_dedup:
        warnings.append(f"Deduplicated {pre_dedup - len(df)} duplicate post IDs.")

    # Parse and validate timestamps
    parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    invalid_ts_count = parsed_ts.isna().sum()
    if invalid_ts_count > 0:
        warnings.append(f"Dropped {invalid_ts_count} rows with invalid timestamps.")
        df = df[parsed_ts.notna()].copy()
        df["timestamp"] = parsed_ts[parsed_ts.notna()]
    else:
        df["timestamp"] = parsed_ts

    # Standardize optional fields
    if "url" not in df.columns:
        df["url"] = ""
    else:
        df["url"] = df["url"].fillna("").astype(str).str.strip()
        df.loc[df["url"].str.lower() == "nan", "url"] = ""

    if "source_platform" not in df.columns:
        df["source_platform"] = default_source
    else:
        df["source_platform"] = df["source_platform"].fillna(default_source).astype(str)

    if "provenance" not in df.columns:
        df["provenance"] = default_provenance
    else:
        df["provenance"] = df["provenance"].fillna(default_provenance).astype(str)

    if "collection_time" not in df.columns:
        df["collection_time"] = datetime.now(timezone.utc).isoformat()

    df = df.reset_index(drop=True)
    return df, warnings


def validate_and_normalize_accounts(
    df: pd.DataFrame,
    default_source: str = "unknown",
    default_provenance: str = "unspecified",
) -> tuple[pd.DataFrame, list[str]]:
    """Validate, clean, and normalize account profile records into canonical schema.

    Returns:
        tuple of (cleaned_df, validation_warnings)
    """
    warnings: list[str] = []
    if df.empty:
        warnings.append("Accounts DataFrame is empty.")
        empty_cols = list(AccountRecord.__annotations__.keys())
        return pd.DataFrame(columns=empty_cols), warnings

    df = df.copy()
    if "account_id" not in df.columns:
        raise ValueError("Missing mandatory column 'account_id' in account records.")

    initial_count = len(df)
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(subset=["account_id"])

    dropped = initial_count - len(df)
    if dropped > 0:
        warnings.append(f"Dropped {dropped} rows missing account_id.")

    df = df.drop_duplicates(subset=["account_id"], keep="first")

    # Numeric features for ML model compatibility
    numeric_defaults = {
        "followers_count": 0,
        "friends_count": 0,
        "statuses_count": 0,
        "favourites_count": 0,
        "listed_count": 0,
        "default_profile": 0,
        "geo_enabled": 0,
        "verified": 0,
        "protected": 0,
        "description_length": 0,
    }

    for col, default_val in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val).astype(int)

    if "source_platform" not in df.columns:
        df["source_platform"] = default_source
    if "provenance" not in df.columns:
        df["provenance"] = default_provenance
    if "collection_time" not in df.columns:
        df["collection_time"] = datetime.now(timezone.utc).isoformat()

    df = df.reset_index(drop=True)
    return df, warnings
