"""NexTrace — Ingestion package.

Exports data providers and ingestion helpers.
"""

from __future__ import annotations

from typing import Any

from src.ingestion.api_adapter import GenericAPIProvider
from src.ingestion.base import (
    AuthenticationError,
    BaseDataProvider,
    IngestionError,
    RateLimitError,
)
from src.ingestion.file_provider import FileDataProvider
from src.ingestion.reddit_provider import RedditAPIProvider
from src.ingestion.synthetic_provider import SyntheticDataProvider
from src.ingestion.x_provider import XAPIProvider


def get_data_provider(
    provider_type: str = "synthetic",
    **kwargs: Any,
) -> BaseDataProvider:
    """Factory function to instantiate configured data provider."""
    provider_type = provider_type.lower().strip()
    if provider_type in ("synthetic", "demo"):
        return SyntheticDataProvider()
    elif provider_type in ("file", "csv", "json", "local"):
        posts_path = kwargs.get("posts_path")
        if not posts_path:
            raise ValueError("posts_path is required for FileDataProvider.")
        accounts_path = kwargs.get("accounts_path")
        source_platform = kwargs.get("source_platform", "file_import")
        return FileDataProvider(
            posts_path=posts_path,
            accounts_path=accounts_path,
            source_platform=source_platform,
        )
    elif provider_type in ("api", "rest", "feed", "live"):
        return GenericAPIProvider(
            base_url=kwargs.get("base_url"),
            bearer_token=kwargs.get("bearer_token"),
            source_platform=kwargs.get("source_platform", "generic_api"),
        )
    elif provider_type in ("x", "twitter", "x_api"):
        return XAPIProvider(
            bearer_token=kwargs.get("bearer_token"),
        )
    elif provider_type in ("reddit", "reddit_api"):
        return RedditAPIProvider(
            client_id=kwargs.get("client_id"),
            client_secret=kwargs.get("client_secret"),
            user_agent=kwargs.get("user_agent"),
        )
    else:
        raise ValueError(
            f"Unknown provider type '{provider_type}'. "
            "Supported types: 'synthetic', 'file', 'api', 'x', 'reddit'"
        )


__all__ = [
    "AuthenticationError",
    "BaseDataProvider",
    "FileDataProvider",
    "GenericAPIProvider",
    "IngestionError",
    "RateLimitError",
    "RedditAPIProvider",
    "SyntheticDataProvider",
    "XAPIProvider",
    "get_data_provider",
]
