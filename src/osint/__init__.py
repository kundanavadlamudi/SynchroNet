"""NexTrace — OSINT and Narrative Intelligence Package.

Exports:
- GDELTClient: Global news intelligence via GDELT DOC 2.0 API.
- GoogleFactCheckClient: Claim verification via Google Fact Check Tools API.
- RDAPClient: Domain registration investigation via RDAP.
- VisionOCRClient: Computer vision and text extraction with explicit disclaimers.
- OSINTEnricher: Pipeline enrichment orchestrator.
"""

from __future__ import annotations

from src.osint.enrichment import OSINTEnricher, extract_keywords_from_text
from src.osint.factcheck_client import GoogleFactCheckClient
from src.osint.gdelt_client import GDELTClient
from src.osint.rdap_client import RDAPClient, extract_domain
from src.osint.vision_ocr import VisionOCRClient

__all__ = [
    "GDELTClient",
    "GoogleFactCheckClient",
    "OSINTEnricher",
    "RDAPClient",
    "VisionOCRClient",
    "extract_domain",
    "extract_keywords_from_text",
]
