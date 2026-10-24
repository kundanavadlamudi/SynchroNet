"""NexTrace — Generic REST / Feed API Provider Adapter.

Implements an authenticated provider abstraction for authorized social feeds or platform APIs.
Features:
- Configured via environment variables (NEXTRACE_API_BASE_URL, NEXTRACE_API_BEARER_TOKEN).
- Token protection: credentials remain server-side and are NEVER printed to logs.
- Bounded exponential backoff with retry logic for transient errors (429, 500, 502, 503, 504).
- Rate limit handling with Retry-After header inspection.
- Pagination controls with bounded maximum page requests.
- Clear blocked status reporting when credentials or platform endpoints are not configured.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
import pandas as pd
import requests

from src.contracts import validate_and_normalize_accounts, validate_and_normalize_posts
from src.ingestion.base import (
    AuthenticationError,
    BaseDataProvider,
    IngestionError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


def mask_token(token: str | None) -> str:
    """Safely mask API tokens for logging."""
    if not token:
        return "[UNSET]"
    if len(token) <= 8:
        return "****"
    return f"{token[:3]}****{token[-3:]}"


class GenericAPIProvider(BaseDataProvider):
    """Adapter for official REST APIs and authorized feed endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        bearer_token: str | None = None,
        source_platform: str = "generic_api",
        timeout_seconds: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("NEXTRACE_API_BASE_URL", "").strip()
        )
        self.bearer_token = (
            bearer_token
            or os.getenv("NEXTRACE_API_BEARER_TOKEN", "").strip()
        )
        self.source_platform = source_platform
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = session or requests.Session()

    @property
    def name(self) -> str:
        return "api_provider"

    @property
    def is_synthetic(self) -> bool:
        return False

    def check_status(self) -> dict[str, Any]:
        """Check whether live credentials and platform endpoints are configured."""
        if not self.base_url:
            return {
                "provider": self.name,
                "status": "BLOCKED_MISSING_ENDPOINT",
                "is_synthetic": False,
                "message": (
                    "No API base URL configured. Set NEXTRACE_API_BASE_URL in your "
                    "environment or .env file with your authorized endpoint URL."
                ),
                "configured": False,
            }
        if not self.bearer_token:
            return {
                "provider": self.name,
                "status": "BLOCKED_MISSING_CREDENTIALS",
                "is_synthetic": False,
                "endpoint": self.base_url,
                "message": (
                    "No API bearer token configured. Set NEXTRACE_API_BEARER_TOKEN in "
                    "your environment or .env file with your authorized API token."
                ),
                "configured": False,
            }
        return {
            "provider": self.name,
            "status": "CONFIGURED",
            "is_synthetic": False,
            "endpoint": self.base_url,
            "token_status": f"Configured ({mask_token(self.bearer_token)})",
            "configured": True,
        }

    def _execute_request_with_retries(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute HTTP request with bounded backoff and rate-limit handling."""
        status = self.check_status()
        if not status["configured"]:
            raise AuthenticationError(status["message"])

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "User-Agent": "NexTrace-Intelligence/1.0",
        }

        attempts = 0
        current_backoff = self.backoff_factor

        while attempts < self.max_retries:
            attempts += 1
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )
                # Auth error checks
                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        f"Authentication failed for {endpoint} (HTTP {response.status_code}). "
                        "Check your API credentials and permission scopes."
                    )
                # Rate limit checks
                if response.status_code == 429:
                    retry_after_hdr = response.headers.get("Retry-After")
                    retry_after = int(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else int(current_backoff)
                    if attempts >= self.max_retries:
                        raise RateLimitError(
                            f"Rate limit exceeded (HTTP 429). Retry after {retry_after}s.",
                            retry_after=retry_after,
                        )
                    time.sleep(retry_after)
                    current_backoff *= 2
                    continue

                if response.status_code >= 500:
                    if attempts >= self.max_retries:
                        raise IngestionError(
                            f"Remote server error from {endpoint} (HTTP {response.status_code})."
                        )
                    time.sleep(current_backoff)
                    current_backoff *= 2
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as exc:
                if attempts >= self.max_retries:
                    raise IngestionError(f"Request timeout connecting to {url} after {self.timeout}s.") from exc
                time.sleep(current_backoff)
                current_backoff *= 2
            except requests.exceptions.RequestException as exc:
                if isinstance(exc, (AuthenticationError, RateLimitError, IngestionError)):
                    raise
                if attempts >= self.max_retries:
                    raise IngestionError(f"Network request error: {exc}") from exc
                time.sleep(current_backoff)
                current_backoff *= 2

        raise IngestionError(f"Max retries ({self.max_retries}) exceeded for {endpoint}")

    def fetch_posts(
        self,
        limit: int | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        status = self.check_status()
        if not status["configured"]:
            raise AuthenticationError(status["message"])

        params: dict[str, Any] = {"limit": min(limit or 100, 100)}
        if query:
            params["q"] = query

        all_records: list[dict[str, Any]] = []
        page_count = 0
        max_pages = 5

        while page_count < max_pages:
            page_count += 1
            payload = self._execute_request_with_retries("posts", params=params)

            data_chunk = payload.get("data") or payload.get("posts") or []
            if not isinstance(data_chunk, list):
                break

            all_records.extend(data_chunk)
            if limit and len(all_records) >= limit:
                all_records = all_records[:limit]
                break

            # Handle pagination
            next_token = payload.get("next_token") or payload.get("pagination", {}).get("next_cursor")
            if not next_token:
                break
            params["cursor"] = next_token

        raw_df = pd.DataFrame(all_records)
        provenance = f"api:{self.base_url}"
        cleaned_df, warnings = validate_and_normalize_posts(
            raw_df, default_source=self.source_platform, default_provenance=provenance
        )

        metadata = {
            "provider": self.name,
            "is_synthetic": False,
            "endpoint": self.base_url,
            "record_count": len(cleaned_df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }
        return cleaned_df, metadata

    def fetch_accounts(
        self,
        account_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        status = self.check_status()
        if not status["configured"]:
            raise AuthenticationError(status["message"])

        params: dict[str, Any] = {}
        if account_ids:
            params["ids"] = ",".join(account_ids[:100])

        payload = self._execute_request_with_retries("accounts", params=params)
        data = payload.get("data") or payload.get("accounts") or []
        raw_df = pd.DataFrame(data if isinstance(data, list) else [])

        provenance = f"api:{self.base_url}"
        cleaned_df, warnings = validate_and_normalize_accounts(
            raw_df, default_source=self.source_platform, default_provenance=provenance
        )
        metadata = {
            "provider": self.name,
            "is_synthetic": False,
            "endpoint": self.base_url,
            "record_count": len(cleaned_df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }
        return cleaned_df, metadata
