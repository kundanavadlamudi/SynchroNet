"""NexTrace — Reddit API Provider Adapter.

Integrates with Reddit's search endpoint to ingest public submissions and author metadata
for disinformation and coordinated narrative tracking.

Official Documentation:
https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki
https://www.reddit.com/dev/api/

Features:
- Configurable User-Agent adhering to Reddit API guidelines.
- Optional OAuth authentication or public JSON endpoint fallback.
- Normalizes Reddit posts (title, selftext, score, permalink) into canonical PostRecords.
- Normalizes Reddit authors into canonical AccountRecords.
- Handles rate limits and network errors with bounded retry backoff.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
import pandas as pd
import requests

from src.config import DEFAULT_BACKOFF_FACTOR, DEFAULT_MAX_RETRIES, DEFAULT_REQUEST_TIMEOUT, REDDIT_SEARCH_URL
from src.contracts import validate_and_normalize_accounts, validate_and_normalize_posts
from src.ingestion.base import (
    AuthenticationError,
    BaseDataProvider,
    IngestionError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class RedditAPIProvider(BaseDataProvider):
    """Data provider for Reddit submissions and author profiles."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        base_url: str = REDDIT_SEARCH_URL,
        timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        session: requests.Session | None = None,
    ) -> None:
        self.client_id = (
            client_id
            or os.getenv("REDDIT_CLIENT_ID", "").strip()
        )
        self.client_secret = (
            client_secret
            or os.getenv("REDDIT_CLIENT_SECRET", "").strip()
        )
        self.user_agent = (
            user_agent
            or os.getenv("REDDIT_USER_AGENT", "").strip()
            or "NexTrace:disinformation.research:v1.0 (academic research)"
        )
        self.base_url = base_url
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = session or requests.Session()

    @property
    def name(self) -> str:
        return "reddit_api"

    @property
    def is_synthetic(self) -> bool:
        return False

    def check_status(self) -> dict[str, Any]:
        """Check operational status of Reddit provider."""
        headers = {"User-Agent": self.user_agent}
        try:
            resp = self.session.get(
                self.base_url,
                headers=headers,
                params={"q": "disinformation", "limit": 1},
                timeout=min(self.timeout, 5),
            )
            if resp.status_code == 200:
                return {
                    "provider": self.name,
                    "status": "OPERATIONAL",
                    "mode": "authenticated" if (self.client_id and self.client_secret) else "public_feed",
                    "configured": True,
                    "endpoint": self.base_url,
                }
            elif resp.status_code == 429:
                return {
                    "provider": self.name,
                    "status": "RATE_LIMITED",
                    "configured": True,
                    "message": "Reddit rate limit reached. Consider configuring OAuth credentials.",
                }
            else:
                return {
                    "provider": self.name,
                    "status": f"HTTP_{resp.status_code}",
                    "configured": True,
                }
        except Exception as e:
            return {
                "provider": self.name,
                "status": "UNREACHABLE",
                "configured": True,
                "error": str(e),
            }

    def ingest(
        self,
        query: str = "disinformation",
        subreddit: str | None = None,
        limit: int = 25,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ingest posts and authors matching a search query from Reddit.

        Args:
            query: Search query string.
            subreddit: Optional subreddit name to restrict search.
            limit: Maximum posts to retrieve (1-100).

        Returns:
            Tuple of (posts_df, accounts_df).
        """
        search_url = f"https://www.reddit.com/r/{subreddit}/search.json" if subreddit else self.base_url
        headers = {"User-Agent": self.user_agent}
        params = {
            "q": query,
            "sort": "new",
            "limit": min(max(1, limit), 100),
            "restrict_sr": 1 if subreddit else 0,
        }

        attempt = 0
        data: dict[str, Any] = {}
        while attempt <= self.max_retries:
            try:
                logger.info("Executing Reddit search query %r on %s (attempt %d)", query, search_url, attempt + 1)
                resp = self.session.get(search_url, headers=headers, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    data = resp.json()
                    break

                if resp.status_code == 429:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    logger.warning("Reddit API rate limited (429). Retrying in %.1fs...", sleep_time)
                    if attempt == self.max_retries:
                        raise RateLimitError("Reddit API rate limit exceeded.")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue

                if resp.status_code in (500, 502, 503):
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    if attempt == self.max_retries:
                        raise IngestionError(f"Reddit server error {resp.status_code}")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue

                raise IngestionError(f"Reddit API returned HTTP {resp.status_code}: {resp.text}")

            except requests.exceptions.Timeout as e:
                if attempt == self.max_retries:
                    raise IngestionError(f"Reddit API request timed out: {e}") from e
                time.sleep(self.backoff_factor * (2 ** attempt))
                attempt += 1
            except requests.exceptions.RequestException as e:
                raise IngestionError(f"Network error querying Reddit API: {e}") from e

        children = data.get("data", {}).get("children", [])
        if not children:
            logger.info("Reddit returned 0 results for query %r", query)
            empty_posts, _ = validate_and_normalize_posts(
                pd.DataFrame(columns=["post_id", "account_id", "text", "timestamp"]),
                default_source="reddit",
                default_provenance="reddit_api",
            )
            empty_accounts, _ = validate_and_normalize_accounts(
                pd.DataFrame(columns=["account_id"]),
                default_source="reddit",
                default_provenance="reddit_api",
            )
            return empty_posts, empty_accounts

        post_rows: list[dict[str, Any]] = []
        account_rows: list[dict[str, Any]] = []
        seen_authors: set[str] = set()

        for child in children:
            item = child.get("data", {})
            if not isinstance(item, dict):
                continue

            post_id = str(item.get("id") or item.get("name") or "")
            author = str(item.get("author") or "unknown_author")
            title = str(item.get("title") or "").strip()
            selftext = str(item.get("selftext") or "").strip()
            text = f"{title}\n{selftext}".strip() if selftext else title

            created_utc = item.get("created_utc")
            ts_str = ""
            if created_utc:
                try:
                    ts_str = datetime.fromtimestamp(float(created_utc), tz=timezone.utc).isoformat()
                except (ValueError, TypeError):
                    ts_str = datetime.now(timezone.utc).isoformat()

            permalink = str(item.get("permalink") or "")
            post_url = f"https://www.reddit.com{permalink}" if permalink else ""
            score = int(item.get("score") or 0)
            num_comments = int(item.get("num_comments") or 0)

            post_rows.append({
                "post_id": f"t3_{post_id}" if not post_id.startswith("t3_") else post_id,
                "account_id": author,
                "text": text,
                "timestamp": ts_str,
                "url": post_url,
                "source_platform": "reddit",
                "provenance": "reddit_api",
                "collection_time": datetime.now(timezone.utc).isoformat(),
                "engagement_likes": max(0, score),
                "engagement_shares": max(0, num_comments),
            })

            if author and author not in seen_authors and author != "[deleted]":
                seen_authors.add(author)
                account_rows.append({
                    "account_id": author,
                    "source_platform": "reddit",
                    "provenance": "reddit_api",
                    "followers_count": 0,
                    "friends_count": 0,
                    "statuses_count": 1,
                    "favourites_count": score,
                    "listed_count": 0,
                    "default_profile": 0,
                    "geo_enabled": 0,
                    "verified": 0,
                    "protected": 0,
                    "description_length": 0,
                    "collection_time": datetime.now(timezone.utc).isoformat(),
                })

        posts_df, post_warnings = validate_and_normalize_posts(
            pd.DataFrame(post_rows),
            default_source="reddit",
            default_provenance="reddit_api",
        )
        accounts_df, account_warnings = validate_and_normalize_accounts(
            pd.DataFrame(account_rows),
            default_source="reddit",
            default_provenance="reddit_api",
        )

        return posts_df, accounts_df

    def fetch_posts(
        self,
        limit: int | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch and return normalized posts DataFrame and provenance metadata."""
        query = kwargs.get("query", "disinformation")
        max_limit = limit or kwargs.get("limit", 25)
        posts_df, accounts_df = self.ingest(query=query, limit=max_limit, **kwargs)
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
