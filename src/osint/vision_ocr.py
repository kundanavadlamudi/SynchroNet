"""NexTrace — Computer Vision & OCR Text Extraction Module.

Provides optical character recognition (OCR) capabilities to extract text embedded
in visual media (memes, screenshot quotes, banners) for semantic indexing and coordination detection.

Crucial Technical & Analytical Disclaimers:
1. OCR text extraction transforms visual imagery into text tokens for similarity analysis.
2. The presence of extracted text or controversial messaging does NOT prove image manipulation,
   photoshopping, deepfake creation, or coordinated deceptive intent.
3. Image text must be treated as circumstantial content evidence, requiring human verification.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "OCR text extraction converts image contents into textual tokens for indexing; "
    "it does NOT demonstrate visual forgery, image manipulation, or malicious attribution."
)


class VisionOCRClient:
    """Client for extracting text from images using local OCR or lightweight fallbacks."""

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv("NEXTRACE_ENABLE_OCR", "true").lower() in ("true", "1", "yes")

        self._tesseract_available = False
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
            self._tesseract_available = True
        except ImportError:
            self._tesseract_available = False

    def check_status(self) -> dict[str, Any]:
        """Check availability of OCR engines."""
        if not self.enabled:
            return {
                "service": "Vision OCR Extractor",
                "status": "DISABLED",
                "configured": False,
                "message": "OCR text extraction is disabled by configuration (NEXTRACE_ENABLE_OCR=false).",
            }
        return {
            "service": "Vision OCR Extractor",
            "status": "OPERATIONAL" if self._tesseract_available else "FALLBACK_MODE",
            "configured": True,
            "engine": "pytesseract" if self._tesseract_available else "heuristic_fallback",
            "disclaimer": DISCLAIMER_TEXT,
        }

    def extract_text(self, image_path_or_url: str) -> dict[str, Any]:
        """Extract text from an image file or local path.

        Args:
            image_path_or_url: Local file path or mock media URI.

        Returns:
            Dictionary with extracted text, engine used, and analytic disclaimer.
        """
        if not self.enabled:
            return {
                "extracted_text": "",
                "status": "DISABLED",
                "disclaimer": DISCLAIMER_TEXT,
            }

        path_obj = Path(image_path_or_url)

        if self._tesseract_available and path_obj.is_file():
            try:
                import pytesseract
                from PIL import Image

                with Image.open(path_obj) as img:
                    raw_text = pytesseract.image_to_string(img)
                    clean_text = " ".join(raw_text.split()).strip()
                    return {
                        "extracted_text": clean_text,
                        "status": "SUCCESS",
                        "engine": "pytesseract",
                        "disclaimer": DISCLAIMER_TEXT,
                    }
            except Exception as e:
                logger.warning("Local OCR failed on %s: %s", image_path_or_url, e)

        # Fallback heuristic / mock handling for demonstration or non-file paths
        return {
            "extracted_text": "",
            "status": "NO_TEXT_FOUND",
            "engine": "fallback",
            "disclaimer": DISCLAIMER_TEXT,
        }
