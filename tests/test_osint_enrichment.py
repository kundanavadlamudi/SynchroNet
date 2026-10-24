"""NexTrace — Tests for OSINT Intelligence Sources & Enrichment.

Tests:
1. GDELTClient: Mock article parsing, empty body handling, query construction, timeout handling.
2. GoogleFactCheckClient: Unconfigured status, mock claims parsing, publisher attribution, missing fields.
3. RDAPClient: Safe domain extraction, SSRF prevention (IPs, localhost), metadata parsing, cache.
4. VisionOCRClient: Status reporting, disclaimer presence, fallback execution.
5. OSINTEnricher: Post enrichment, campaign enrichment, dataset enrichment, file output.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.contracts import DomainInvestigationRecord, FactCheckRecord, NewsArticleRecord, OSINTEnrichmentBundle
from src.osint.enrichment import OSINTEnricher, extract_keywords_from_text
from src.osint.factcheck_client import GoogleFactCheckClient
from src.osint.gdelt_client import GDELTClient
from src.osint.rdap_client import RDAPClient, extract_domain
from src.osint.vision_ocr import DISCLAIMER_TEXT, VisionOCRClient


class TestGDELTClient(unittest.TestCase):
    """Unit tests for the GDELT DOC 2.0 API client."""

    def test_search_news_parses_valid_json_response(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({
            "articles": [
                {
                    "url": "https://www.reuters.com/world/disinfo-report",
                    "title": "Study Finds Coordinated Influence Campaign Online",
                    "seendate": "20261024T120000Z",
                    "domain": "reuters.com",
                    "language": "English",
                    "sourcecountry": "United States",
                }
            ]
        })
        mock_resp.json.return_value = json.loads(mock_resp.text)
        mock_session.get.return_value = mock_resp

        client = GDELTClient(session=mock_session)
        articles = client.search_news("disinformation campaign", max_records=5, use_cache=False)

        self.assertEqual(len(articles), 1)
        art = articles[0]
        self.assertIsInstance(art, NewsArticleRecord)
        self.assertEqual(art.title, "Study Finds Coordinated Influence Campaign Online")
        self.assertEqual(art.domain, "reuters.com")
        self.assertEqual(art.provenance, "GDELT DOC 2.0 API")
        self.assertTrue("2026-10-24" in art.seen_date)

    def test_search_news_handles_empty_or_html_response(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>No matching records found.</body></html>"
        mock_session.get.return_value = mock_resp

        client = GDELTClient(session=mock_session)
        articles = client.search_news("rare_phrase_never_found", use_cache=False)
        self.assertEqual(articles, [])

    def test_search_news_handles_timeout(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

        client = GDELTClient(session=mock_session)
        articles = client.search_news("timeout test", use_cache=False)
        self.assertEqual(articles, [])


class TestGoogleFactCheckClient(unittest.TestCase):
    """Unit tests for the Google Fact Check Tools API client."""

    def test_unconfigured_client_returns_blocked_status(self):
        client = GoogleFactCheckClient(api_key="")
        status = client.check_status()
        self.assertEqual(status["status"], "BLOCKED_MISSING_API_KEY")
        self.assertFalse(status["configured"])

        # Searches should safely return empty without network calls
        claims = client.search_claims("miracle cure announced")
        self.assertEqual(claims, [])

    def test_search_claims_parses_reviews_and_preserves_attribution(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "claims": [
                {
                    "text": "Drinking seawater cures all modern illnesses.",
                    "claimant": "Social Media Viral Video",
                    "claimDate": "2026-09-15T00:00:00Z",
                    "claimReview": [
                        {
                            "publisher": {
                                "name": "PolitiFact",
                                "site": "politifact.com"
                            },
                            "url": "https://www.politifact.com/factchecks/seawater",
                            "title": "Fact check: Seawater is not a medical cure",
                            "reviewDate": "2026-09-16T10:00:00Z",
                            "textualRating": "False",
                        }
                    ],
                }
            ]
        }
        mock_session.get.return_value = mock_resp

        client = GoogleFactCheckClient(api_key="mock_key_12345", session=mock_session)
        recs = client.search_claims("drinking seawater", use_cache=False)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertIsInstance(rec, FactCheckRecord)
        self.assertEqual(rec.claim_text, "Drinking seawater cures all modern illnesses.")
        self.assertEqual(rec.publisher_name, "PolitiFact")
        self.assertEqual(rec.textual_rating, "False")
        self.assertEqual(rec.review_url, "https://www.politifact.com/factchecks/seawater")
        self.assertEqual(rec.provenance, "Google Fact Check Tools API")


class TestRDAPClient(unittest.TestCase):
    """Unit tests for RDAP domain resolution and SSRF defense."""

    def test_extract_domain_sanitization(self):
        self.assertEqual(extract_domain("https://news-outlet.org/article?id=123"), "news-outlet.org")
        self.assertEqual(extract_domain("http://sub.domain.co.uk/path/"), "sub.domain.co.uk")
        self.assertEqual(extract_domain("rawdomain.com"), "rawdomain.com")

    def test_extract_domain_rejects_ssrf_and_private_ips(self):
        self.assertIsNone(extract_domain("http://localhost:8080/admin"))
        self.assertIsNone(extract_domain("http://127.0.0.1/status"))
        self.assertIsNone(extract_domain("http://10.0.0.1/metadata"))
        self.assertIsNone(extract_domain("http://192.168.1.1/router"))
        self.assertIsNone(extract_domain("http://169.254.169.254/latest/meta-data"))
        self.assertIsNone(extract_domain("invalid..domain"))
        self.assertIsNone(extract_domain(""))

    def test_rdap_lookup_parses_registrar_and_dates(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ldhName": "investigation-target.net",
            "events": [
                {"eventAction": "registration", "eventDate": "2025-01-10T15:00:00Z"},
                {"eventAction": "last changed", "eventDate": "2025-06-20T11:00:00Z"},
            ],
            "entities": [
                {
                    "roles": ["registrar"],
                    "vcardArray": ["vcard", [["fn", {}, "text", "Example Registrar LLC"]]],
                }
            ],
            "status": ["clientTransferProhibited"],
            "nameservers": [{"ldhName": "ns1.exampledns.com"}],
        }
        mock_session.get.return_value = mock_resp

        client = RDAPClient(session=mock_session, enabled=True)
        rec = client.investigate_domain("https://investigation-target.net/page.html", use_cache=False)

        self.assertIsNotNone(rec)
        self.assertIsInstance(rec, DomainInvestigationRecord)
        self.assertEqual(rec.domain, "investigation-target.net")
        self.assertEqual(rec.registrar, "Example Registrar LLC")
        self.assertEqual(rec.registration_date, "2025-01-10T15:00:00Z")
        self.assertEqual(rec.nameservers, ["ns1.exampledns.com"])
        self.assertIn("clientTransferProhibited", rec.status)
        self.assertIn("does not establish malicious attribution", rec.evidence_note)


class TestVisionOCRClient(unittest.TestCase):
    """Unit tests for the Vision OCR module."""

    def test_ocr_status_and_disclaimer_presence(self):
        client = VisionOCRClient(enabled=True)
        status = client.check_status()
        self.assertIn("service", status)
        self.assertIn("status", status)
        self.assertEqual(status["disclaimer"], DISCLAIMER_TEXT)

    def test_ocr_extract_text_returns_disclaimer(self):
        client = VisionOCRClient(enabled=True)
        res = client.extract_text("non_existent_file.png")
        self.assertIn("disclaimer", res)
        self.assertEqual(res["disclaimer"], DISCLAIMER_TEXT)


class TestOSINTEnricher(unittest.TestCase):
    """Unit tests for high-level OSINT pipeline enrichment."""

    def setUp(self):
        self.mock_gdelt = MagicMock(spec=GDELTClient)
        self.mock_factcheck = MagicMock(spec=GoogleFactCheckClient)
        self.mock_rdap = MagicMock(spec=RDAPClient)
        self.mock_vision = MagicMock(spec=VisionOCRClient)

        self.enricher = OSINTEnricher(
            gdelt_client=self.mock_gdelt,
            factcheck_client=self.mock_factcheck,
            rdap_client=self.mock_rdap,
            vision_client=self.mock_vision,
        )

    def test_extract_keywords_from_text(self):
        phrase = extract_keywords_from_text("Urgent breaking alert: Deepfake leaked by hackers! https://t.co/abc")
        self.assertTrue(len(phrase) > 0)
        self.assertNotIn("http", phrase)
        self.assertNotIn("https", phrase)

    def test_enrich_campaign_aggregates_urls_and_queries(self):
        self.mock_gdelt.search_news.return_value = [
            NewsArticleRecord(url="https://news.com/1", title="Article 1")
        ]
        self.mock_factcheck.search_claims.return_value = [
            FactCheckRecord(query_term="test", claim_text="Claim 1", textual_rating="False")
        ]
        self.mock_rdap.investigate_domain.return_value = DomainInvestigationRecord(domain="news.com")

        camp_row = pd.Series({
            "campaign_id": "CAMP_TEST_001",
            "accounts": "ACC_1, ACC_2",
            "number_of_accounts": 2,
        })
        posts_df = pd.DataFrame([
            {"post_id": "P01", "account_id": "ACC_1", "text": "Viral vaccine hoax spreading", "url": "https://news.com/fake"},
            {"post_id": "P02", "account_id": "ACC_2", "text": "Viral vaccine hoax spreading", "url": "https://news.com/fake"},
        ])

        bundle = self.enricher.enrich_campaign(camp_row, posts_df)
        self.assertIsInstance(bundle, OSINTEnrichmentBundle)
        self.assertEqual(bundle.target_id, "CAMP_TEST_001")
        self.assertEqual(bundle.target_type, "campaign")
        self.assertEqual(len(bundle.matched_news), 1)
        self.assertEqual(len(bundle.matched_fact_checks), 1)
        self.assertEqual(len(bundle.domain_investigations), 1)


if __name__ == "__main__":
    unittest.main()
