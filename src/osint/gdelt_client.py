"""NexTrace — GDELT DOC 2.0 API Client for Open Source News Intelligence.

Integrates with the GDELT 2.0 Document API to retrieve relevant global news coverage,
article metadata, publication timestamps, and source domains.

Official Documentation:
https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Key Characteristics:
- No API key required for standard ArtList queries.
- Query parameters: query, mode=ArtList, maxrecords, format=json, timespan, sort=DateDesc.
- Normalizes raw article objects into canonical NewsArticleRecord instances.
- Handles empty results, malformed JSON, and service timeouts gracefully.
"""

from __future__ import annotations

import logging
import urllib.parse
from datetime import datetime, timezone
from typing import Any
import requests

from src.config import DEFAULT_REQUEST_TIMEOUT, GDELT_BASE_URL
from src.contracts import NewsArticleRecord

logger = logging.getLogger(__name__)


class GDELTClient:
    """Client for querying the GDELT DOC 2.0 API."""

    def __init__(
        self,
        base_url: str = GDELT_BASE_URL,
        timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout_seconds
        self.session = session or requests.Session()
        self._cache: dict[str, list[NewsArticleRecord]] = {}

    def check_status(self) -> dict[str, Any]:
        """Check availability of the GDELT DOC API service."""
        try:
            # Simple probe query for service health check
            resp = self.session.get(
                self.base_url,
                params={"query": "test", "mode": "ArtList", "maxrecords": 1, "format": "json"},
                timeout=min(self.timeout, 5),
            )
            return {
                "service": "GDELT DOC 2.0 API",
                "status": "OPERATIONAL" if resp.status_code == 200 else f"HTTP_{resp.status_code}",
                "requires_auth": False,
                "endpoint": self.base_url,
                "configured": True,
            }
        except Exception as e:
            return {
                "service": "GDELT DOC 2.0 API",
                "status": "UNREACHABLE",
                "requires_auth": False,
                "endpoint": self.base_url,
                "configured": True,
                "error": str(e),
            }

    def search_news(
        self,
        query: str,
        max_records: int = 10,
        timespan: str = "7d",
        language: str | None = None,
        sort: str = "DateDesc",
        use_cache: bool = True,
    ) -> list[NewsArticleRecord]:
        """Search global news coverage matching a keyword query.

        Args:
            query: Keyword search string (e.g. topic, narrative terms).
            max_records: Maximum number of articles to return (1-250).
            timespan: Time window for coverage (e.g. '24h', '7d', '1m').
            language: Optional language filter (e.g. 'English', 'Spanish').
            sort: Sorting order ('DateDesc', 'ToneAsc', 'ToneDesc').
            use_cache: Whether to return cached results if available.

        Returns:
            List of normalized NewsArticleRecord instances.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        full_query = clean_query
        if language:
            full_query = f"{clean_query} sourcelang:{language}"

        cache_key = f"{full_query}::{max_records}::{timespan}::{sort}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "query": full_query,
            "mode": "ArtList",
            "maxrecords": min(max(1, max_records), 250),
            "format": "json",
            "timespan": timespan,
            "sort": sort,
        }

        try:
            logger.info("Querying GDELT DOC API: query=%r timespan=%s", full_query, timespan)
            resp = self.session.get(self.base_url, params=params, timeout=self.timeout)

            if resp.status_code != 200:
                logger.warning("GDELT API returned status %d: %s", resp.status_code, resp.text[:200])
                return []

            # GDELT returns empty string or HTML on no matches
            text = resp.text.strip()
            if not text or text.startswith("<") or text.startswith("<!DOCTYPE"):
                logger.info("GDELT returned empty or non-JSON body for query %r", query)
                return []

            data = resp.json()
            raw_articles = data.get("articles", [])
            records: list[NewsArticleRecord] = []

            for item in raw_articles:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or "").strip()
                if not url or not title:
                    continue

                domain = str(item.get("domain") or "").strip()
                if not domain and url:
                    parsed = urllib.parse.urlparse(url)
                    domain = parsed.netloc

                seen_date = str(item.get("seendate") or "").strip()
                # Format seendate (often YYYYMMDDTHHMMSSZ) into readable ISO
                formatted_date = seen_date
                if len(seen_date) == 16 and seen_date.endswith("Z"):
                    try:
                        dt = datetime.strptime(seen_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                        formatted_date = dt.isoformat()
                    except ValueError:
                        formatted_date = seen_date

                rec = NewsArticleRecord(
                    url=url,
                    title=title,
                    seen_date=formatted_date,
                    domain=domain,
                    language=str(item.get("language") or "en"),
                    source_country=str(item.get("sourcecountry") or ""),
                    query_term=clean_query,
                    collection_time=datetime.now(timezone.utc).isoformat(),
                    provenance="GDELT DOC 2.0 API",
                )
                records.append(rec)

            if use_cache:
                self._cache[cache_key] = records

            return records

        except requests.exceptions.Timeout:
            logger.warning("GDELT DOC API request timed out after %ds", self.timeout)
            return []
        except requests.exceptions.RequestException as e:
            logger.warning("GDELT DOC API request failed: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error parsing GDELT response: %s", e)
            return []
