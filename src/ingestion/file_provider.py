"""NexTrace — Local File Data Provider (CSV & JSON).

Allows importing authorized datasets from CSV or JSON files.
Supports both single-file posts and combined posts + accounts.
Handles malformed records, missing fields, and date parsing errors safely.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from src.contracts import validate_and_normalize_accounts, validate_and_normalize_posts
from src.ingestion.base import BaseDataProvider, IngestionError


class FileDataProvider(BaseDataProvider):
    """Loads posts and accounts from local CSV or JSON files."""

    def __init__(
        self,
        posts_path: Path | str,
        accounts_path: Path | str | None = None,
        source_platform: str = "file_import",
    ) -> None:
        self.posts_path = Path(posts_path)
        self.accounts_path = Path(accounts_path) if accounts_path else None
        self.source_platform = source_platform

    @property
    def name(self) -> str:
        return "file_import"

    @property
    def is_synthetic(self) -> bool:
        return False

    def check_status(self) -> dict[str, Any]:
        posts_exists = self.posts_path.exists()
        accounts_exists = self.accounts_path.exists() if self.accounts_path else False
        status = "READY" if posts_exists else "ERROR_FILE_NOT_FOUND"

        return {
            "provider": self.name,
            "status": status,
            "is_synthetic": False,
            "posts_file": str(self.posts_path),
            "posts_file_exists": posts_exists,
            "accounts_file": str(self.accounts_path) if self.accounts_path else None,
            "accounts_file_exists": accounts_exists,
        }

    def _read_file_to_df(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise IngestionError(f"File not found: {path}")

        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                return pd.read_csv(path, low_memory=False)
            elif suffix in (".json", ".jsonl"):
                # Try reading as standard JSON array or dict, then JSON lines
                try:
                    content = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(content, list):
                        return pd.DataFrame(content)
                    elif isinstance(content, dict) and "posts" in content:
                        return pd.DataFrame(content["posts"])
                    elif isinstance(content, dict) and "accounts" in content:
                        return pd.DataFrame(content["accounts"])
                    elif isinstance(content, dict):
                        return pd.DataFrame([content])
                except Exception:
                    # Fallback to json lines
                    return pd.read_json(path, lines=True)
            else:
                raise IngestionError(f"Unsupported file format '{suffix}'. Supported: .csv, .json, .jsonl")
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Error reading file {path.name}: {exc}") from exc

    def fetch_posts(self, limit: int | None = None, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        raw_df = self._read_file_to_df(self.posts_path)
        if limit is not None and limit > 0:
            raw_df = raw_df.head(limit)

        provenance = f"file:{self.posts_path.name}"
        cleaned_df, warnings = validate_and_normalize_posts(
            raw_df, default_source=self.source_platform, default_provenance=provenance
        )

        metadata = {
            "provider": self.name,
            "is_synthetic": False,
            "source_path": str(self.posts_path),
            "raw_count": len(raw_df),
            "record_count": len(cleaned_df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }
        return cleaned_df, metadata

    def fetch_accounts(self, account_ids: list[str] | None = None, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        if self.accounts_path and self.accounts_path.exists():
            raw_df = self._read_file_to_df(self.accounts_path)
            provenance = f"file:{self.accounts_path.name}"
            cleaned_df, warnings = validate_and_normalize_accounts(
                raw_df, default_source=self.source_platform, default_provenance=provenance
            )
            if account_ids:
                cleaned_df = cleaned_df[cleaned_df["account_id"].isin(account_ids)].reset_index(drop=True)

            metadata = {
                "provider": self.name,
                "is_synthetic": False,
                "source_path": str(self.accounts_path),
                "record_count": len(cleaned_df),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "warnings": warnings,
            }
            return cleaned_df, metadata

        # If no separate accounts file is provided, synthesize default profiles from distinct accounts in posts
        posts_df, _ = self.fetch_posts()
        distinct_accounts = sorted(posts_df["account_id"].unique())
        if account_ids:
            distinct_accounts = [a for a in distinct_accounts if a in account_ids]

        default_rows = [
            {
                "account_id": a,
                "source_platform": self.source_platform,
                "provenance": f"inferred_from:{self.posts_path.name}",
            }
            for a in distinct_accounts
        ]
        cleaned_df, warnings = validate_and_normalize_accounts(
            pd.DataFrame(default_rows),
            default_source=self.source_platform,
            default_provenance=f"inferred_from:{self.posts_path.name}",
        )
        metadata = {
            "provider": self.name,
            "is_synthetic": False,
            "source_path": "inferred_from_posts",
            "record_count": len(cleaned_df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings + ["Account profiles were inferred from post records without external metadata."],
        }
        return cleaned_df, metadata
