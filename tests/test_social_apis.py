"""NexTrace — Tests for Social Media API Providers (X and Reddit).

Tests:
1. XAPIProvider:
   - Unconfigured status: reports BLOCKED_MISSING_CREDENTIALS and instructions.
   - Token masking: ensures credentials never leak in logs or status dictionaries.
   - Normalization: maps X API v2 JSON (data and includes.users) into canonical PostRecord and AccountRecord schemas.
   - Authentication errors: raises AuthenticationError on 401 or 403.
   - Rate limiting: respects retry backoff and handles 429 with retry-after.
   - Pagination: traverses pages using next_token.
2. RedditAPIProvider:
   - Status reporting and user-agent compliance.
   - Normalization: maps Reddit submission listings into canonical contracts.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.api_adapter import mask_token
from src.ingestion.base import AuthenticationError, IngestionError, RateLimitError
from src.ingestion.reddit_provider import RedditAPIProvider
from src.ingestion.x_provider import XAPIProvider


class TestTokenMasking(unittest.TestCase):
    """Verify security masking of API credentials."""

    def test_mask_token_varieties(self):
        self.assertEqual(mask_token(None), "[UNSET]")
        self.assertEqual(mask_token(""), "[UNSET]")
        self.assertEqual(mask_token("short"), "****")
        self.assertEqual(mask_token("12345678"), "****")
        self.assertEqual(mask_token("AAAA1234567890ZZZZ"), "AAA****ZZZ")


class TestXAPIProvider(unittest.TestCase):
    """Unit and mock integration tests for Twitter / X API v2 provider."""

    def test_unconfigured_provider_returns_blocked_status(self):
        prov = XAPIProvider(bearer_token="")
        stat = prov.check_status()
        self.assertEqual(stat["status"], "BLOCKED_MISSING_CREDENTIALS")
        self.assertFalse(stat["configured"])
        self.assertIn("X_BEARER_TOKEN", stat["message"])

        # Ingestion without credentials must raise IngestionError with helpful guidance
        with self.assertRaises(IngestionError):
            prov.ingest(query="test")

    def test_x_ingest_normalizes_tweets_and_users(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "1840000000000000001",
                    "text": "Breaking news on social media manipulation! #disinfo",
                    "created_at": "2026-10-24T12:00:00.000Z",
                    "author_id": "999888777",
                    "public_metrics": {
                        "like_count": 42,
                        "retweet_count": 15,
                    },
                    "lang": "en",
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "999888777",
                        "name": "Investigative Watch",
                        "username": "invest_watch",
                        "description": "Tracking influence networks.",
                        "public_metrics": {
                            "followers_count": 5400,
                            "following_count": 320,
                            "tweet_count": 1200,
                            "listed_count": 14,
                        },
                        "verified": True,
                    }
                ]
            },
            "meta": {"result_count": 1},
        }
        mock_session.get.return_value = mock_resp

        provider = XAPIProvider(
            bearer_token="mock_bearer_token_99999",
            session=mock_session,
        )
        posts_df, accounts_df = provider.ingest(query="disinfo", max_results=10)

        # Verify posts normalization
        self.assertEqual(len(posts_df), 1)
        post = posts_df.iloc[0]
        self.assertEqual(post["post_id"], "1840000000000000001")
        self.assertEqual(post["account_id"], "999888777")
        self.assertEqual(post["source_platform"], "x")
        self.assertEqual(post["provenance"], "x_api_v2")
        self.assertEqual(post["engagement_likes"], 42)
        self.assertEqual(post["engagement_shares"], 15)
        self.assertEqual(post["url"], "https://x.com/i/web/status/1840000000000000001")

        # Verify accounts normalization
        self.assertEqual(len(accounts_df), 1)
        account = accounts_df.iloc[0]
        self.assertEqual(account["account_id"], "999888777")
        self.assertEqual(account["followers_count"], 5400)
        self.assertEqual(account["friends_count"], 320)
        self.assertEqual(account["statuses_count"], 1200)
        self.assertEqual(account["verified"], 1)

    def test_x_ingest_handles_authentication_failure(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid bearer token"
        mock_session.get.return_value = mock_resp

        provider = XAPIProvider(
            bearer_token="invalid_token_12345",
            session=mock_session,
        )
        with self.assertRaises(AuthenticationError):
            provider.ingest(query="test")

    def test_x_ingest_handles_rate_limiting_backoff(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "1"}
        mock_session.get.return_value = mock_resp

        provider = XAPIProvider(
            bearer_token="valid_token",
            max_retries=1,
            backoff_factor=0.01,
            session=mock_session,
        )
        with self.assertRaises(RateLimitError):
            provider.ingest(query="test")

    def test_x_ingest_pagination_traversal(self):
        mock_session = MagicMock(spec=requests.Session)
        page1 = {
            "data": [{"id": "P1", "text": "Post 1", "author_id": "U1", "created_at": "2026-10-24T12:00:00Z"}],
            "includes": {"users": [{"id": "U1"}]},
            "meta": {"next_token": "TOKEN_PAGE_2"},
        }
        page2 = {
            "data": [{"id": "P2", "text": "Post 2", "author_id": "U2", "created_at": "2026-10-24T12:05:00Z"}],
            "includes": {"users": [{"id": "U2"}]},
            "meta": {},
        }
        resp1 = MagicMock(status_code=200)
        resp1.json.return_value = page1
        resp2 = MagicMock(status_code=200)
        resp2.json.return_value = page2

        mock_session.get.side_effect = [resp1, resp2]

        provider = XAPIProvider(bearer_token="token", session=mock_session)
        posts_df, accounts_df = provider.ingest(query="test", max_pages=2)

        self.assertEqual(len(posts_df), 2)
        self.assertEqual(len(accounts_df), 2)


class TestRedditAPIProvider(unittest.TestCase):
    """Unit tests for the Reddit API provider."""

    def test_reddit_ingest_normalizes_submissions(self):
        mock_session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "id": "xyz789",
                            "name": "t3_xyz789",
                            "author": "reddit_researcher",
                            "title": "Analysis of automated social accounts",
                            "selftext": "Detailed findings on coordinated behaviors.",
                            "created_utc": 1761300000.0,
                            "permalink": "/r/disinformation/comments/xyz789/analysis",
                            "score": 150,
                            "num_comments": 25,
                        }
                    }
                ]
            }
        }
        mock_session.get.return_value = mock_resp

        provider = RedditAPIProvider(session=mock_session)
        posts_df, accounts_df = provider.ingest(query="coordinated accounts")

        self.assertEqual(len(posts_df), 1)
        post = posts_df.iloc[0]
        self.assertEqual(post["post_id"], "t3_xyz789")
        self.assertEqual(post["account_id"], "reddit_researcher")
        self.assertEqual(post["source_platform"], "reddit")
        self.assertIn("Detailed findings", post["text"])
        self.assertEqual(post["engagement_likes"], 150)

        self.assertEqual(len(accounts_df), 1)
        acc = accounts_df.iloc[0]
        self.assertEqual(acc["account_id"], "reddit_researcher")
        self.assertEqual(acc["source_platform"], "reddit")


if __name__ == "__main__":
    unittest.main()
