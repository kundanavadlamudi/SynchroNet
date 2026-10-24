"""NexTrace — OSINT Narrative & Evidence Enrichment Orchestrator.

Combines open-source news intelligence (GDELT), third-party claim verifications (Google Fact Check),
and domain infrastructure intelligence (RDAP) into a unified narrative enrichment workflow.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from src.config import OSINT_ENRICHMENT_FILE
from src.contracts import (
    DomainInvestigationRecord,
    FactCheckRecord,
    NewsArticleRecord,
    OSINTEnrichmentBundle,
)
from src.osint.factcheck_client import GoogleFactCheckClient
from src.osint.gdelt_client import GDELTClient
from src.osint.rdap_client import RDAPClient, extract_domain
from src.osint.vision_ocr import VisionOCRClient

logger = logging.getLogger(__name__)


def extract_keywords_from_text(text: str, max_words: int = 4) -> str:
    """Extract a concise query phrase from text for external search."""
    # Remove URLs, mentions, hashtags symbols
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"[@#]", "", cleaned)
    # Remove punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    words = [w for w in cleaned.split() if len(w) > 3 and not w.lower() in {
        "this", "that", "with", "from", "have", "been", "were", "what", "when",
        "where", "will", "would", "about", "there", "their", "they"
    }]
    if not words:
        return ""
    return " ".join(words[:max_words])


class OSINTEnricher:
    """Coordinates OSINT queries and associates external intelligence with posts and campaigns."""

    def __init__(
        self,
        gdelt_client: GDELTClient | None = None,
        factcheck_client: GoogleFactCheckClient | None = None,
        rdap_client: RDAPClient | None = None,
        vision_client: VisionOCRClient | None = None,
    ) -> None:
        self.gdelt = gdelt_client or GDELTClient()
        self.factcheck = factcheck_client or GoogleFactCheckClient()
        self.rdap = rdap_client or RDAPClient()
        self.vision = vision_client or VisionOCRClient()

    def get_service_statuses(self) -> dict[str, Any]:
        """Aggregate health and authentication statuses across all OSINT providers."""
        return {
            "gdelt": self.gdelt.check_status(),
            "fact_check": self.factcheck.check_status(),
            "rdap": self.rdap.check_status(),
            "ocr": self.vision.check_status(),
        }

    def enrich_post(self, post_row: dict[str, Any] | pd.Series) -> OSINTEnrichmentBundle:
        """Enrich an individual post with news context, fact-checks, and domain info."""
        post_id = str(post_row.get("post_id", ""))
        text = str(post_row.get("text", "")).strip()
        url = str(post_row.get("url", "")).strip()

        matched_news: list[dict[str, Any]] = []
        matched_fact_checks: list[dict[str, Any]] = []
        domain_investigations: list[dict[str, Any]] = []

        # 1. Domain investigation if URL is present
        if url and url.lower() != "nan":
            domain = extract_domain(url)
            if domain:
                dom_rec = self.rdap.investigate_domain(domain)
                if dom_rec:
                    domain_investigations.append(dom_rec.to_dict())

        # 2. News and Fact Checks query phrase
        query_phrase = extract_keywords_from_text(text)
        if query_phrase:
            # GDELT News
            news_recs = self.gdelt.search_news(query_phrase, max_records=3)
            matched_news.extend([n.to_dict() for n in news_recs])

            # Fact Check
            fc_recs = self.factcheck.search_claims(query_phrase, page_size=2)
            matched_fact_checks.extend([fc.to_dict() for fc in fc_recs])

        return OSINTEnrichmentBundle(
            target_id=post_id,
            target_type="post",
            matched_news=matched_news,
            matched_fact_checks=matched_fact_checks,
            domain_investigations=domain_investigations,
        )

    def enrich_campaign(
        self,
        campaign_row: dict[str, Any] | pd.Series,
        posts_df: pd.DataFrame,
    ) -> OSINTEnrichmentBundle:
        """Enrich a candidate campaign cluster with aggregated narrative and domain intelligence."""
        camp_id = str(campaign_row.get("campaign_id", "CAMPAIGN_UNKNOWN"))
        raw_accounts = campaign_row.get("accounts", "")

        # Parse account list
        if isinstance(raw_accounts, str):
            account_list = [a.strip() for a in raw_accounts.split(",") if a.strip()]
        elif isinstance(raw_accounts, list):
            account_list = [str(a).strip() for a in raw_accounts]
        else:
            account_list = []

        # Filter member posts
        member_posts = posts_df[posts_df["account_id"].isin(account_list)] if not posts_df.empty else pd.DataFrame()

        matched_news: list[dict[str, Any]] = []
        matched_fact_checks: list[dict[str, Any]] = []
        domain_investigations: list[dict[str, Any]] = []

        # Extract all URLs across member posts
        if not member_posts.empty and "url" in member_posts.columns:
            urls = member_posts["url"].dropna().unique()
            seen_domains = set()
            for u in urls:
                if u and str(u).lower() != "nan":
                    d = extract_domain(str(u))
                    if d and d not in seen_domains:
                        seen_domains.add(d)
                        d_rec = self.rdap.investigate_domain(d)
                        if d_rec:
                            domain_investigations.append(d_rec.to_dict())

        # Extract top collective text keywords
        if not member_posts.empty and "text" in member_posts.columns:
            combined_text = " ".join(member_posts["text"].dropna().astype(str))
            query_phrase = extract_keywords_from_text(combined_text, max_words=4)
            if query_phrase:
                news_recs = self.gdelt.search_news(query_phrase, max_records=5)
                matched_news.extend([n.to_dict() for n in news_recs])

                fc_recs = self.factcheck.search_claims(query_phrase, page_size=3)
                matched_fact_checks.extend([fc.to_dict() for fc in fc_recs])

        return OSINTEnrichmentBundle(
            target_id=camp_id,
            target_type="campaign",
            matched_news=matched_news,
            matched_fact_checks=matched_fact_checks,
            domain_investigations=domain_investigations,
        )

    def enrich_dataset(
        self,
        posts_df: pd.DataFrame,
        campaigns_df: pd.DataFrame,
        max_posts_to_enrich: int = 5,
        save_path: Path | str = OSINT_ENRICHMENT_FILE,
    ) -> dict[str, Any]:
        """Perform batch OSINT enrichment across top posts and campaigns, saving results to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        enriched_posts: list[dict[str, Any]] = []
        enriched_campaigns: list[dict[str, Any]] = []

        # Enrich sample posts (prioritizing posts that share URLs or top engagement)
        if not posts_df.empty:
            sample_posts = posts_df.head(max_posts_to_enrich)
            for _, row in sample_posts.iterrows():
                bundle = self.enrich_post(row)
                enriched_posts.append(bundle.to_dict())

        # Enrich candidate campaigns
        if not campaigns_df.empty:
            for _, row in campaigns_df.iterrows():
                bundle = self.enrich_campaign(row, posts_df)
                enriched_campaigns.append(bundle.to_dict())

        results = {
            "enriched_posts": enriched_posts,
            "enriched_campaigns": enriched_campaigns,
            "provider_statuses": self.get_service_statuses(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.1.0",
            "methodology_disclaimer": (
                "OSINT enrichment aggregates external news, third-party claim reviews, and domain registries. "
                "These sources provide context and do not represent internal model predictions or legal attributions."
            ),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("OSINT enrichment results saved to %s", save_path)
        return results
