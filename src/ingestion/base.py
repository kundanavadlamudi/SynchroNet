"""NexTrace — Base Data Provider Interface.

Defines the contract for all data ingestion sources (synthetic, file-based, or API).
All providers must return validated DataFrames conforming to the data contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class IngestionError(Exception):
    """Base exception for ingestion failures."""


class AuthenticationError(IngestionError):
    """Raised when provider authentication fails or credentials are invalid."""


class RateLimitError(IngestionError):
    """Raised when an API rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BaseDataProvider(ABC):
    """Abstract base class for data ingestion providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the data provider."""

    @property
    @abstractmethod
    def is_synthetic(self) -> bool:
        """True if the provider returns synthetic demo data."""

    @abstractmethod
    def check_status(self) -> dict[str, Any]:
        """Check provider configuration, availability, and credential status."""

    @abstractmethod
    def fetch_posts(self, limit: int | None = None, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch and return validated posts DataFrame and metadata dict.

        Returns:
            tuple of (posts_dataframe, metadata)
        """

    @abstractmethod
    def fetch_accounts(self, account_ids: list[str] | None = None, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch and return validated accounts DataFrame and metadata dict.

        Returns:
            tuple of (accounts_dataframe, metadata)
        """
