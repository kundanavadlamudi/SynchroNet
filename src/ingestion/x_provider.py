"""NexTrace — X (formerly Twitter) API v2 Provider.

Integrates with the official Twitter/X API v2 Recent Search endpoint:
https://api.twitter.com/2/tweets/search/recent

Features:
- Bearer token authentication via X_BEARER_TOKEN or TWITTER_BEARER_TOKEN environment variables.
- Token masking: API secrets remain protected server-side and are NEVER output in logs.
- Bounded retries with exponential backoff for transient errors (429, 500, 502, 503, 504).
- Rate-limit handling reading x-rate-limit-reset header.
- Multi-page pagination support using meta.next_token.
- Response extraction:
  - Tweets: id, text, created_at, public_metrics (like_count, retweet_count), lang.
  - Users: author_id expanded from includes.users with follower counts and profile metrics.
- Canonical normalization through validate_and_normalize_posts and validate_and_normalize_accounts.
- Explicit status check: returns BLOCKED_MISSING_CREDENTIALS when credentials are unset.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
import pandas as pd
import requests

from src.config import DEFAULT_BACKOFF_FACTOR, DEFAULT_MAX_RETRIES, DEFAULT_REQUEST_TIMEOUT, X_API_RECENT_SEARCH_URL
from src.contracts import validate_and_normalize_accounts, validate_and_normalize_posts
from src.ingestion.api_adapter import mask_token
from src.ingestion.base import (
    AuthenticationError,
    BaseDataProvider,
    IngestionError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class XAPIProvider(BaseDataProvider):
    """Data provider for authorized Twitter / X API v2 access."""

    def __init__(
        self,
        bearer_token: str | None = None,
        base_url: str = X_API_RECENT_SEARCH_URL,
        timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        session: requests.Session | None = None,
    ) -> None:
        self.bearer_token = (
            bearer_token
            or os.getenv("X_BEARER_TOKEN", "").strip()
            or os.getenv("TWITTER_BEARER_TOKEN", "").strip()
            or os.getenv("NEXTRACE_API_BEARER_TOKEN", "").strip()
        )
        self.base_url = base_url
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = session or requests.Session()

    @property
    def name(self) -> str:
        return "x_api_v2"

    @property
    def is_synthetic(self) -> bool:
        return False

    def check_status(self) -> dict[str, Any]:
        """Check whether live credentials and platform endpoints are operational."""
        if not self.bearer_token:
            return {
                "provider": self.name,
                "status": "BLOCKED_MISSING_CREDENTIALS",
                "is_synthetic": False,
                "endpoint": self.base_url,
                "token_masked": mask_token(None),
                "configured": False,
                "message": (
                    "No X Bearer Token configured. Set X_BEARER_TOKEN in your environment "
                    "or .env file with your authorized developer portal credentials."
                ),
            }

        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "NexTrace-Disinfo-Research/1.0",
        }

        try:
            # Lightweight probe query: max_results=10 is minimum allowed by X v2
            resp = self.session.get(
                self.base_url,
                headers=headers,
                params={"query": "python", "max_results": 10},
                timeout=min(self.timeout, 5),
            )

            if resp.status_code == 200:
                return {
                    "provider": self.name,
                    "status": "OPERATIONAL",
                    "is_synthetic": False,
                    "endpoint": self.base_url,
                    "token_masked": mask_token(self.bearer_token),
                    "configured": True,
                }
            elif resp.status_code == 401:
                return {
                    "provider": self.name,
                    "status": "AUTHENTICATION_FAILED",
                    "is_synthetic": False,
                    "error_code": 401,
                    "token_masked": mask_token(self.bearer_token),
                    "message": "Invalid or revoked X Bearer Token.",
                    "configured": True,
                }
            elif resp.status_code == 403:
                return {
                    "provider": self.name,
                    "status": "FORBIDDEN_SUBSCRIPTION_REQUIRED",
                    "is_synthetic": False,
                    "error_code": 403,
                    "token_masked": mask_token(self.bearer_token),
                    "message": "Access forbidden: Current X subscription tier lacks recent search permissions.",
                    "configured": True,
                }
            elif resp.status_code == 429:
                return {
                    "provider": self.name,
                    "status": "RATE_LIMITED",
                    "is_synthetic": False,
                    "error_code": 429,
                    "token_masked": mask_token(self.bearer_token),
                    "message": "X API rate limit reached.",
                    "configured": True,
                }
            else:
                return {
                    "provider": self.name,
                    "status": f"HTTP_{resp.status_code}",
                    "is_synthetic": False,
                    "configured": True,
                    "token_masked": mask_token(self.bearer_token),
                }

        except Exception as e:
            return {
                "provider": self.name,
                "status": "UNREACHABLE",
                "is_synthetic": False,
                "configured": True,
                "error": str(e),
                "token_masked": mask_token(self.bearer_token),
            }

    def _execute_request_with_backoff(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute HTTP request with exponential backoff and rate-limit handling."""
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "NexTrace-Disinfo-Research/1.0",
        }

        attempt = 0
        while attempt <= self.max_retries:
            try:
                logger.info(
                    "Executing X API query to %s (attempt %d/%d, token=%s)",
                    self.base_url,
                    attempt + 1,
                    self.max_retries + 1,
                    mask_token(self.bearer_token),
                )
                resp = self.session.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (401, 403):
                    msg = f"X API authentication failure (HTTP {resp.status_code}): {resp.text}"
                    logger.error(msg)
                    raise AuthenticationError(msg)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("x-rate-limit-reset") or resp.headers.get("Retry-After")
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    if retry_after:
                        try:
                            # If epoch timestamp
                            reset_epoch = float(retry_after)
                            sleep_time = max(1.0, min(reset_epoch - time.time(), 60.0))
                        except ValueError:
                            pass
                    logger.warning("X API rate limit (429). Backing off for %.1fs...", sleep_time)
                    if attempt == self.max_retries:
                        raise RateLimitError(f"X API rate limit exceeded after {attempt + 1} attempts.")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue

                if resp.status_code in (500, 502, 503, 504):
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    logger.warning("Transient X API server error %d. Retrying in %.1fs...", resp.status_code, sleep_time)
                    if attempt == self.max_retries:
                        raise IngestionError(f"X API server error {resp.status_code} persisted after retries.")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue

                raise IngestionError(f"X API error (HTTP {resp.status_code}): {resp.text}")

            except requests.exceptions.Timeout as e:
                logger.warning("Timeout connecting to X API on attempt %d: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    raise IngestionError(f"X API request timed out after {attempt + 1} attempts: {e}") from e
                time.sleep(self.backoff_factor * (2 ** attempt))
                attempt += 1
            except requests.exceptions.RequestException as e:
                logger.error("Network error connecting to X API: %s", e)
                raise IngestionError(f"Network error connecting to X API: {e}") from e

        raise IngestionError("X API request failed after all retry attempts.")

    def ingest(
        self,
        query: str = "disinformation",
        max_results: int = 10,
        max_pages: int = 1,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ingest live tweets and author profiles from the X API v2 Recent Search endpoint.

        Args:
            query: Search query string (must follow X v2 query syntax).
            max_results: Tweets per page (between 10 and 100).
            max_pages: Maximum pages to paginate.

        Returns:
            Tuple of (posts_df, accounts_df) normalized into canonical contracts.
        """
        if not self.bearer_token:
            raise IngestionError(
                "Cannot ingest from X API: X_BEARER_TOKEN is not configured. "
                "Set X_BEARER_TOKEN in your environment or use --source demo."
            )

        clamped_max_results = min(max(10, max_results), 100)
        params: dict[str, Any] = {
            "query": query,
            "max_results": clamped_max_results,
            "tweet.fields": "created_at,public_metrics,lang,conversation_id",
            "expansions": "author_id",
            "user.fields": "created_at,description,public_metrics,verified",
        }

        raw_tweets: list[dict[str, Any]] = []
        user_map: dict[str, dict[str, Any]] = {}

        next_token: str | None = None
        pages_retrieved = 0

        while pages_retrieved < max_pages:
            current_params = dict(params)
            if next_token:
                current_params["next_token"] = next_token

            data = self._execute_request_with_backoff(current_params)
            tweets = data.get("data", [])
            raw_tweets.extend(tweets)

            # Extract user objects from includes
            users = data.get("includes", {}).get("users", [])
            for u in users:
                uid = str(u.get("id", ""))
                if uid:
                    user_map[uid] = u

            meta = data.get("meta", {})
            next_token = meta.get("next_token")
            pages_retrieved += 1

            if not next_token:
                break

        if not raw_tweets:
            logger.info("X API returned 0 matching tweets for query %r", query)
            empty_posts, _ = validate_and_normalize_posts(
                pd.DataFrame(columns=["post_id", "account_id", "text", "timestamp"]),
                default_source="x",
                default_provenance="x_api_v2",
            )
            empty_accounts, _ = validate_and_normalize_accounts(
                pd.DataFrame(columns=["account_id"]),
                default_source="x",
                default_provenance="x_api_v2",
            )
            return empty_posts, empty_accounts

        # Normalize tweets into post records
        post_rows: list[dict[str, Any]] = []
        for t in raw_tweets:
            tid = str(t.get("id", ""))
            author_id = str(t.get("author_id", "unknown_author"))
            metrics = t.get("public_metrics", {})
            like_count = int(metrics.get("like_count", 0)) if isinstance(metrics, dict) else 0
            retweet_count = int(metrics.get("retweet_count", 0)) if isinstance(metrics, dict) else 0

            # Construct canonical tweet URL
            tweet_url = f"https://x.com/i/web/status/{tid}" if tid else ""

            post_rows.append({
                "post_id": tid,
                "account_id": author_id,
                "text": str(t.get("text", "")),
                "timestamp": str(t.get("created_at", "")),
                "url": tweet_url,
                "source_platform": "x",
                "provenance": "x_api_v2",
                "collection_time": datetime.now(timezone.utc).isoformat(),
                "engagement_likes": like_count,
                "engagement_shares": retweet_count,
                "language": str(t.get("lang") or ""),
            })

        posts_df, post_warnings = validate_and_normalize_posts(
            pd.DataFrame(post_rows),
            default_source="x",
            default_provenance="x_api_v2",
        )
        for w in post_warnings:
            logger.warning("X post validation warning: %s", w)

        # Normalize account profiles
        account_rows: list[dict[str, Any]] = []
        # Ensure every author in posts has an account record
        unique_authors = set(posts_df["account_id"].unique()) if not posts_df.empty else set(user_map.keys())

        for uid in unique_authors:
            u_obj = user_map.get(uid, {})
            u_metrics = u_obj.get("public_metrics", {})
            followers = int(u_metrics.get("followers_count", 0)) if isinstance(u_metrics, dict) else 0
            following = int(u_metrics.get("following_count", 0)) if isinstance(u_metrics, dict) else 0
            statuses = int(u_metrics.get("tweet_count", 0)) if isinstance(u_metrics, dict) else 0
            listed = int(u_metrics.get("listed_count", 0)) if isinstance(u_metrics, dict) else 0
            desc = str(u_obj.get("description", ""))
            is_verified = 1 if u_obj.get("verified", False) else 0

            account_rows.append({
                "account_id": uid,
                "source_platform": "x",
                "provenance": "x_api_v2",
                "followers_count": followers,
                "friends_count": following,
                "statuses_count": statuses,
                "favourites_count": 0,
                "listed_count": listed,
                "default_profile": 0,
                "geo_enabled": 0,
                "verified": is_verified,
                "protected": 0,
                "description_length": len(desc),
                "collection_time": datetime.now(timezone.utc).isoformat(),
            })

        accounts_df, account_warnings = validate_and_normalize_accounts(
            pd.DataFrame(account_rows),
            default_source="x",
            default_provenance="x_api_v2",
        )
        for w in account_warnings:
            logger.warning("X account validation warning: %s", w)

        return posts_df, accounts_df

    def fetch_posts(
        self,
        limit: int | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch and return normalized posts DataFrame and provenance metadata."""
        query = kwargs.get("query", "disinformation")
        max_results = limit or kwargs.get("max_results", 20)
        posts_df, accounts_df = self.ingest(query=query, max_results=max_results, **kwargs)
        self._cached_accounts = accounts_df
        metadata = {
            "source": self.name,
            "query": query,
            "count": len(posts_df),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return posts_df, metadata

    def fetch_accounts(
        self,
        account_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch and return normalized accounts DataFrame and metadata."""
        if hasattr(self, "_cached_accounts") and self._cached_accounts is not None:
            acc_df = self._cached_accounts
        else:
            _, acc_df = self.ingest(query=kwargs.get("query", "disinformation"), **kwargs)
        if account_ids:
            acc_df = acc_df[acc_df["account_id"].isin(account_ids)]
        return acc_df, {"source": self.name, "count": len(acc_df)}
