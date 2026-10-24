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
from src.ingestion.synthetic_provider import SyntheticDataProvider


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
    else:
        raise ValueError(
            f"Unknown provider type '{provider_type}'. "
            "Supported types: 'synthetic', 'file', 'api'"
        )


__all__ = [
    "AuthenticationError",
    "BaseDataProvider",
    "FileDataProvider",
    "GenericAPIProvider",
    "IngestionError",
    "RateLimitError",
    "SyntheticDataProvider",
    "get_data_provider",
]
