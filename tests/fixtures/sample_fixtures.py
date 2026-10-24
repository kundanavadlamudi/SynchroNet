"""Synthetic test fixtures for NexTrace.

All fixtures are strictly synthetic and isolated from real-world accounts or campaigns.
Used for deterministic testing of schema validation, ingestion, and detection algorithms.
"""

from __future__ import annotations

import pandas as pd


def get_synthetic_posts_fixture() -> pd.DataFrame:
    """Returns a deterministic DataFrame of 10 synthetic posts simulating:
    - A coordinated cluster: accounts A101, A102, A103 posting identical text and URL within 2 minutes
    - An isolated benign post: account A104 posting security advice
    - Other diverse benign posts
    """
    return pd.DataFrame([
        {
            "post_id": "P001",
            "account_id": "A101",
            "text": "Breaking news about a major cyber security incident #CyberSecurity",
            "timestamp": "2026-10-20 09:00:00",
            "url": "https://example.com/news1",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P002",
            "account_id": "A102",
            "text": "Breaking news about a major cyber security incident #CyberSecurity",
            "timestamp": "2026-10-20 09:01:00",
            "url": "https://example.com/news1",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P003",
            "account_id": "A103",
            "text": "Breaking news about a major cyber security incident #CyberSecurity",
            "timestamp": "2026-10-20 09:02:00",
            "url": "https://example.com/news1",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P004",
            "account_id": "A104",
            "text": "Remember to use strong passwords and enable two factor authentication.",
            "timestamp": "2026-10-20 10:30:00",
            "url": "https://example.com/security",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P005",
            "account_id": "A105",
            "text": "Heavy rainfall reported in several areas. Please follow official weather updates.",
            "timestamp": "2026-10-20 11:15:00",
            "url": "https://example.com/weather",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P006",
            "account_id": "A106",
            "text": "Breaking news about a major cyber security incident #CyberSecurity",
            "timestamp": "2026-10-20 09:03:00",
            "url": "https://example.com/news1",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P007",
            "account_id": "A107",
            "text": "New government recruitment notification announced. Check the official website.",
            "timestamp": "2026-10-20 12:00:00",
            "url": "https://example.com/recruitment",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P008",
            "account_id": "A108",
            "text": "Stay alert against suspicious links and online payment fraud.",
            "timestamp": "2026-10-20 12:15:00",
            "url": "https://example.com/fraud",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P009",
            "account_id": "A109",
            "text": "Breaking news about a major cyber security incident #CyberSecurity",
            "timestamp": "2026-10-20 09:04:00",
            "url": "https://example.com/news1",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "post_id": "P010",
            "account_id": "A110",
            "text": "Always verify news through reliable sources before sharing it.",
            "timestamp": "2026-10-20 13:00:00",
            "url": "https://example.com/media-literacy",
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        }
    ])


def get_malformed_posts_fixture() -> pd.DataFrame:
    """Returns posts containing duplicates, invalid timestamps, missing text and ids."""
    return pd.DataFrame([
        {"post_id": "M001", "account_id": "ACC1", "text": "Valid text", "timestamp": "2026-10-20 09:00:00", "url": ""},
        {"post_id": "M001", "account_id": "ACC1", "text": "Duplicate post_id", "timestamp": "2026-10-20 09:01:00", "url": ""},
        {"post_id": "M002", "account_id": "ACC2", "text": "Bad timestamp", "timestamp": "invalid-time-format", "url": ""},
        {"post_id": "M003", "account_id": "", "text": "Missing account id", "timestamp": "2026-10-20 09:00:00", "url": ""},
        {"post_id": "M004", "account_id": "ACC4", "text": "   ", "timestamp": "2026-10-20 09:00:00", "url": ""},
        {"post_id": "", "account_id": "ACC5", "text": "Missing post id", "timestamp": "2026-10-20 09:00:00", "url": ""},
        {"post_id": "M006", "account_id": "ACC6", "text": "Valid post 2", "timestamp": "2026-10-20 09:05:00", "url": "https://example.com"},
    ])


def get_synthetic_accounts_fixture() -> pd.DataFrame:
    """Returns synthetic account metadata for testing bot classification integration."""
    return pd.DataFrame([
        {
            "account_id": "A101",
            "followers_count": 15,
            "friends_count": 850,
            "statuses_count": 2200,
            "favourites_count": 10,
            "listed_count": 0,
            "default_profile": 1,
            "geo_enabled": 0,
            "verified": 0,
            "protected": 0,
            "description_length": 5,
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        },
        {
            "account_id": "A104",
            "followers_count": 1250,
            "friends_count": 420,
            "statuses_count": 3100,
            "favourites_count": 890,
            "listed_count": 12,
            "default_profile": 0,
            "geo_enabled": 1,
            "verified": 0,
            "protected": 0,
            "description_length": 84,
            "source_platform": "synthetic_demo",
            "provenance": "fixture_test"
        }
    ])
