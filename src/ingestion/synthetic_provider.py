"""NexTrace — Synthetic Demo Data Provider.

Generates clearly marked synthetic demonstration data.
Never presents synthetic fixtures as live social media intelligence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import pandas as pd

from src.contracts import validate_and_normalize_accounts, validate_and_normalize_posts
from src.create_sample_data import _SAMPLE_DATA
from src.ingestion.base import BaseDataProvider


class SyntheticDataProvider(BaseDataProvider):
    """Provides synthetic posts and account profiles for demonstration and testing."""

    @property
    def name(self) -> str:
        return "synthetic_demo"

    @property
    def is_synthetic(self) -> bool:
        return True

    def check_status(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "status": "READY",
            "is_synthetic": True,
            "message": "Synthetic demo mode available. Data does not represent real persons or entities.",
            "configured": True,
        }

    def fetch_posts(self, limit: int | None = None, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        raw_posts = [dict(p) for p in _SAMPLE_DATA]
        for p in raw_posts:
            p["source_platform"] = "synthetic"
            p["provenance"] = "synthetic_demo"

        if limit is not None and limit > 0:
            raw_posts = raw_posts[:limit]

        raw_df = pd.DataFrame(raw_posts)
        cleaned_df, warnings = validate_and_normalize_posts(
            raw_df, default_source="synthetic", default_provenance="synthetic_demo"
        )

        metadata = {
            "provider": self.name,
            "is_synthetic": True,
            "record_count": len(cleaned_df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "notice": "DEMO DATA: Synthetic records generated for testing and demonstration only."
        }
        return cleaned_df, metadata

    def fetch_accounts(self, account_ids: list[str] | None = None, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        # Generate baseline synthetic profile accounts
        posts_df, _ = self.fetch_posts()
        unique_accounts = sorted(posts_df["account_id"].unique())
        if account_ids:
            unique_accounts = [acc for acc in unique_accounts if acc in account_ids]

        account_rows = []
        for i, acc in enumerate(unique_accounts):
            # Accounts A101, A102, A103 simulate higher bot-signal profiles
            is_suspicious = acc in ("A101", "A102", "A103", "A106", "A109")
            account_rows.append({
                "account_id": acc,
                "source_platform": "synthetic",
                "provenance": "synthetic_demo",
                "followers_count": 12 + i if is_suspicious else 450 + (i * 50),
                "friends_count": 890 + i if is_suspicious else 320 + (i * 20),
                "statuses_count": 3400 + i if is_suspicious else 800 + (i * 40),
                "favourites_count": 5 if is_suspicious else 230,
                "listed_count": 0 if is_suspicious else 5,
                "default_profile": 1 if is_suspicious else 0,
                "geo_enabled": 0,
                "verified": 0,
                "protected": 0,
                "description_length": 10 if is_suspicious else 75,
            })

        accounts_df, warnings = validate_and_normalize_accounts(
            pd.DataFrame(account_rows), default_source="synthetic", default_provenance="synthetic_demo"
        )
        metadata = {
            "provider": self.name,
            "is_synthetic": True,
            "record_count": len(accounts_df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
            "notice": "DEMO DATA: Synthetic profile records."
        }
        return accounts_df, metadata
