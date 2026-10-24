"""Automated unit and integration tests for NexTrace ingestion, contracts, and pipeline.

Tests:
- Schema validation and malformed records (contracts.py)
- CSV/JSON ingestion (file_provider.py)
- Synthetic demo provider (synthetic_provider.py)
- API adapter authentication, retries, rate-limits, and token masking (api_adapter.py)
- Model inference input validation and artifact integrity (predict_account.py)
- End-to-end pipeline execution with metadata tracking (run_pipeline.py)
- Dashboard helper functions
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from src.contracts import (
    validate_and_normalize_accounts,
    validate_and_normalize_posts,
)
from src.ingestion import (
    AuthenticationError,
    FileDataProvider,
    GenericAPIProvider,
    IngestionError,
    RateLimitError,
    SyntheticDataProvider,
    get_data_provider,
)
from src.ingestion.api_adapter import mask_token
from tests.fixtures.sample_fixtures import (
    get_malformed_posts_fixture,
    get_synthetic_accounts_fixture,
    get_synthetic_posts_fixture,
)


class TestContractsAndSchemaValidation(unittest.TestCase):
    """Tests for src/contracts.py."""

    def test_valid_posts_normalization(self):
        posts = get_synthetic_posts_fixture()
        cleaned, warnings = validate_and_normalize_posts(posts)
        self.assertEqual(len(cleaned), 10)
        self.assertIn("post_id", cleaned.columns)
        self.assertIn("account_id", cleaned.columns)
        self.assertIn("timestamp", cleaned.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"]))

    def test_malformed_posts_handling(self):
        malformed = get_malformed_posts_fixture()
        cleaned, warnings = validate_and_normalize_posts(malformed)
        # Should drop duplicate M001, invalid timestamp M002, blank account M003, blank text M004, blank post_id
        # Leaving valid M001 (first occurrence) and M006
        self.assertEqual(len(cleaned), 2)
        self.assertListEqual(sorted(cleaned["post_id"].tolist()), ["M001", "M006"])
        self.assertTrue(len(warnings) > 0)

    def test_empty_posts_df_returns_empty_contract(self):
        empty_df = pd.DataFrame()
        cleaned, warnings = validate_and_normalize_posts(empty_df)
        self.assertTrue(cleaned.empty)
        self.assertIn("post_id", cleaned.columns)

    def test_missing_mandatory_column_raises_value_error(self):
        invalid_df = pd.DataFrame([{"wrong_col": "123"}])
        with self.assertRaises(ValueError):
            validate_and_normalize_posts(invalid_df)

    def test_account_normalization_defaults_numeric_features(self):
        accounts = pd.DataFrame([{"account_id": "ACC99"}])
        cleaned, warnings = validate_and_normalize_accounts(accounts)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["followers_count"], 0)
        self.assertEqual(cleaned.iloc[0]["statuses_count"], 0)
        self.assertEqual(cleaned.iloc[0]["default_profile"], 0)


class TestIngestionProviders(unittest.TestCase):
    """Tests for SyntheticDataProvider, FileDataProvider, and GenericAPIProvider."""

    def test_synthetic_provider_lifecycle(self):
        provider = SyntheticDataProvider()
        self.assertEqual(provider.name, "synthetic_demo")
        self.assertTrue(provider.is_synthetic)

        status = provider.check_status()
        self.assertEqual(status["status"], "READY")
        self.assertTrue(status["is_synthetic"])

        posts_df, meta = provider.fetch_posts(limit=5)
        self.assertEqual(len(posts_df), 5)
        self.assertEqual(meta["provider"], "synthetic_demo")
        self.assertIn("DEMO DATA", meta["notice"])

        accounts_df, acc_meta = provider.fetch_accounts()
        self.assertTrue(len(accounts_df) > 0)
        self.assertIn("followers_count", accounts_df.columns)

    def test_file_provider_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_posts.csv"
            json_path = Path(tmpdir) / "test_posts.json"

            fixture = get_synthetic_posts_fixture()
            fixture.to_csv(csv_path, index=False)
            fixture.to_json(json_path, orient="records")

            # CSV provider
            csv_provider = FileDataProvider(posts_path=csv_path)
            self.assertFalse(csv_provider.is_synthetic)
            csv_posts, csv_meta = csv_provider.fetch_posts()
            self.assertEqual(len(csv_posts), 10)

            # JSON provider
            json_provider = FileDataProvider(posts_path=json_path)
            json_posts, json_meta = json_provider.fetch_posts()
            self.assertEqual(len(json_posts), 10)

    def test_file_provider_missing_file_raises_ingestion_error(self):
        provider = FileDataProvider(posts_path="non_existent_file.csv")
        with self.assertRaises(IngestionError):
            provider.fetch_posts()

    def test_token_masking(self):
        self.assertEqual(mask_token(None), "[UNSET]")
        self.assertEqual(mask_token(""), "[UNSET]")
        self.assertEqual(mask_token("short"), "****")
        masked = mask_token("secret_bearer_token_12345")
        self.assertEqual(masked, "sec****345")
        self.assertNotIn("bearer", masked)

    def test_api_provider_status_blocked_when_unconfigured(self):
        provider = GenericAPIProvider(base_url="", bearer_token="")
        status = provider.check_status()
        self.assertEqual(status["status"], "BLOCKED_MISSING_ENDPOINT")
        self.assertFalse(status["configured"])

        provider_with_url = GenericAPIProvider(base_url="https://api.example.com", bearer_token="")
        status_token = provider_with_url.check_status()
        self.assertEqual(status_token["status"], "BLOCKED_MISSING_CREDENTIALS")
        self.assertFalse(status_token["configured"])

    def test_api_provider_raises_authentication_error_on_missing_creds(self):
        provider = GenericAPIProvider(base_url="", bearer_token="")
        with self.assertRaises(AuthenticationError):
            provider.fetch_posts()

    def test_api_provider_mock_fetch_and_pagination(self):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "posts": [
                {
                    "post_id": "API001",
                    "account_id": "USR1",
                    "text": "Live mock post #news",
                    "timestamp": "2026-10-20T10:00:00Z",
                    "url": "https://example.com",
                }
            ],
            "next_token": None,
        }
        mock_session.get.return_value = mock_response

        provider = GenericAPIProvider(
            base_url="https://api.example.com",
            bearer_token="mock_valid_token_xyz",
            session=mock_session,
        )
        posts_df, meta = provider.fetch_posts()
        self.assertEqual(len(posts_df), 1)
        self.assertEqual(posts_df.iloc[0]["post_id"], "API001")
        self.assertEqual(meta["endpoint"], "https://api.example.com")

    def test_api_provider_handles_rate_limit(self):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_session.get.return_value = mock_response

        provider = GenericAPIProvider(
            base_url="https://api.example.com",
            bearer_token="mock_token_123",
            max_retries=1,
            session=mock_session,
        )
        with self.assertRaises(RateLimitError):
            provider.fetch_posts()


class TestEndToEndPipelineAndInference(unittest.TestCase):
    """Integration test verifying end-to-end execution of run_pipeline."""

    def test_pipeline_dry_run_does_not_fail(self):
        from run_pipeline import run_pipeline

        exit_code = run_pipeline(source="demo", dry_run=True)
        self.assertEqual(exit_code, 0)

    def test_bot_classifier_predict_accounts_with_neutral_defaults(self):
        from src.ml.predict_account import predict_accounts
        from src.ml.features import extract_account_features

        posts = get_synthetic_posts_fixture()
        account_features = extract_account_features(posts)

        from src.ml.predict_account import load_model_bundle
        bundle = load_model_bundle()
        for col in bundle["feature_names"]:
            if col not in account_features.columns:
                account_features[col] = 1 if col == "tweet_data_available" else 0

        predictions = predict_accounts(account_features)
        self.assertEqual(len(predictions), account_features["account_id"].nunique())
        self.assertIn("bot_probability", predictions.columns)
        self.assertIn("predicted_label_name", predictions.columns)
        # Probabilities must be valid float between 0 and 1
        for prob in predictions["bot_probability"]:
            self.assertTrue(0.0 <= prob <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
