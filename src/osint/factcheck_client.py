"""NexTrace — Google Fact Check Tools API Client for Claim Verification.

Integrates with the Google Fact Check Tools API to query published fact-checks
from certified IFCN signatories and independent journalism organizations.

Official Documentation:
https://developers.google.com/fact-check/tools/api

Critical Interpretation Rules:
1. No matching fact-check does NOT mean a claim is true or verified.
2. No matching fact-check does NOT mean a claim is false.
3. Third-party publisher ratings belong exclusively to the source publisher
   and are never represented as NexTrace's independent conclusions.
4. Results are retained with full publisher attribution and review URLs.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
import requests

from src.config import DEFAULT_REQUEST_TIMEOUT, GOOGLE_FACT_CHECK_BASE_URL
from src.contracts import FactCheckRecord

logger = logging.getLogger(__name__)


class GoogleFactCheckClient:
    """Client for querying the Google Fact Check Tools API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = GOOGLE_FACT_CHECK_BASE_URL,
        timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
        )
        self.base_url = base_url
        self.timeout = timeout_seconds
        self.session = session or requests.Session()
        self._cache: dict[str, list[FactCheckRecord]] = {}

    def check_status(self) -> dict[str, Any]:
        """Check whether the API key is configured and verify connectivity."""
        if not self.api_key:
            return {
                "service": "Google Fact Check Tools API",
                "status": "BLOCKED_MISSING_API_KEY",
                "configured": False,
                "message": (
                    "No Google Fact Check API key configured. Set "
                    "GOOGLE_FACT_CHECK_API_KEY in your environment or .env file."
                ),
            }

        try:
            resp = self.session.get(
                self.base_url,
                params={"query": "health", "key": self.api_key, "pageSize": 1},
                timeout=min(self.timeout, 5),
            )
            if resp.status_code == 200:
                return {
                    "service": "Google Fact Check Tools API",
                    "status": "OPERATIONAL",
                    "configured": True,
                    "endpoint": self.base_url,
                }
            elif resp.status_code in (400, 403):
                return {
                    "service": "Google Fact Check Tools API",
                    "status": "AUTHENTICATION_FAILED",
                    "configured": True,
                    "error_code": resp.status_code,
                    "message": "Invalid API key or Fact Check API is not enabled in Google Cloud project.",
                }
            else:
                return {
                    "service": "Google Fact Check Tools API",
                    "status": f"HTTP_{resp.status_code}",
                    "configured": True,
                }
        except Exception as e:
            return {
                "service": "Google Fact Check Tools API",
                "status": "UNREACHABLE",
                "configured": True,
                "error": str(e),
            }

    def search_claims(
        self,
        query: str,
        language_code: str = "en",
        page_size: int = 10,
        use_cache: bool = True,
    ) -> list[FactCheckRecord]:
        """Search published fact-checks for a specific claim or narrative phrase.

        Args:
            query: Claim text or narrative keywords to search.
            language_code: BCP-47 language code (default 'en').
            page_size: Maximum claims to retrieve.
            use_cache: Whether to use cached results.

        Returns:
            List of normalized FactCheckRecord objects.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        if not self.api_key:
            logger.info("Fact check search skipped: GOOGLE_FACT_CHECK_API_KEY is not configured.")
            return []

        cache_key = f"{clean_query}::{language_code}::{page_size}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "query": clean_query,
            "languageCode": language_code,
            "pageSize": page_size,
            "key": self.api_key,
        }

        try:
            logger.info("Querying Google Fact Check API for query=%r", clean_query)
            resp = self.session.get(self.base_url, params=params, timeout=self.timeout)

            if resp.status_code == 403:
                logger.warning("Google Fact Check API 403 Forbidden: check API key permissions.")
                return []
            if resp.status_code != 200:
                logger.warning("Google Fact Check API returned %d: %s", resp.status_code, resp.text[:200])
                return []

            data = resp.json()
            raw_claims = data.get("claims", [])
            records: list[FactCheckRecord] = []

            for claim_obj in raw_claims:
                if not isinstance(claim_obj, dict):
                    continue

                claim_text = str(claim_obj.get("text") or "").strip()
                claimant = str(claim_obj.get("claimant") or "").strip()
                claim_date = str(claim_obj.get("claimDate") or "").strip()

                reviews = claim_obj.get("claimReview", [])
                if not reviews:
                    # Record the claim even without a review if present
                    records.append(
                        FactCheckRecord(
                            query_term=clean_query,
                            claim_text=claim_text,
                            claimant=claimant,
                            claim_date=claim_date,
                            collection_time=datetime.now(timezone.utc).isoformat(),
                        )
                    )
                    continue

                for rev in reviews:
                    if not isinstance(rev, dict):
                        continue
                    pub = rev.get("publisher") or {}
                    pub_name = str(pub.get("name") or "").strip() if isinstance(pub, dict) else ""
                    pub_site = str(pub.get("site") or "").strip() if isinstance(pub, dict) else ""

                    rec = FactCheckRecord(
                        query_term=clean_query,
                        claim_text=claim_text,
                        claimant=claimant,
                        claim_date=claim_date,
                        publisher_name=pub_name,
                        publisher_site=pub_site,
                        review_url=str(rev.get("url") or "").strip(),
                        review_title=str(rev.get("title") or "").strip(),
                        review_date=str(rev.get("reviewDate") or "").strip(),
                        textual_rating=str(rev.get("textualRating") or "").strip(),
                        collection_time=datetime.now(timezone.utc).isoformat(),
                        provenance="Google Fact Check Tools API",
                    )
                    records.append(rec)

            if use_cache:
                self._cache[cache_key] = records

            return records

        except requests.exceptions.Timeout:
            logger.warning("Google Fact Check API request timed out after %ds", self.timeout)
            return []
        except requests.exceptions.RequestException as e:
            logger.warning("Google Fact Check API request failed: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error in Google Fact Check client: %s", e)
            return []
