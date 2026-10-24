"""NexTrace — RDAP Domain Investigation Client.

Performs domain registration and infrastructure lookups via Registration Data Access Protocol (RDAP).
Used to enrich URLs shared within suspected disinformation campaigns.

Official Documentation:
RFC 7484 / RFC 9083 / https://about.rdap.org/

Key Constraints:
1. URL extraction and validation prevents SSRF (rejects local/private IP ranges and reserved names).
2. Domain registration info provides contextual infrastructure data; it does NOT constitute proof of malicious activity.
3. Redacted or privacy-protected records are handled gracefully.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any
import requests

from src.config import DEFAULT_REQUEST_TIMEOUT, RDAP_BASE_URL
from src.contracts import DomainInvestigationRecord

logger = logging.getLogger(__name__)

# Basic domain regex matching valid TLDs
_DOMAIN_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


def extract_domain(url_or_domain: str) -> str | None:
    """Safely extract and sanitize domain name from a URL or raw domain string.

    Prevents SSRF and internal-network requests.
    """
    raw = url_or_domain.strip().lower()
    if not raw:
        return None

    # Prepend scheme if missing so urlparse handles it correctly
    if not (raw.startswith("http://") or raw.startswith("https://")):
        raw = "http://" + raw

    try:
        parsed = urllib.parse.urlparse(raw)
        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            return None

        # Check for localhost / loopback
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return None

        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return None
            # Do not perform RDAP domain queries on raw IP addresses via domain endpoint
            return None
        except ValueError:
            pass  # Normal domain name

        # Validate against domain regex
        if _DOMAIN_REGEX.match(hostname):
            return hostname
        return None
    except Exception:
        return None


class RDAPClient:
    """Client for retrieving domain registration metadata via RDAP."""

    def __init__(
        self,
        base_url: str = RDAP_BASE_URL,
        timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
        session: requests.Session | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.session = session or requests.Session()
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv("NEXTRACE_ENABLE_RDAP", "true").lower() in ("true", "1", "yes")
        self._cache: dict[str, DomainInvestigationRecord | None] = {}

    def check_status(self) -> dict[str, Any]:
        """Check status and operational availability of the RDAP resolver."""
        if not self.enabled:
            return {
                "service": "RDAP Domain Resolver",
                "status": "DISABLED",
                "configured": False,
                "message": "RDAP resolution is disabled by configuration (NEXTRACE_ENABLE_RDAP=false).",
            }
        try:
            resp = self.session.get(f"{self.base_url}/example.com", timeout=min(self.timeout, 5))
            return {
                "service": "RDAP Domain Resolver",
                "status": "OPERATIONAL" if resp.status_code == 200 else f"HTTP_{resp.status_code}",
                "configured": True,
                "endpoint": self.base_url,
            }
        except Exception as e:
            return {
                "service": "RDAP Domain Resolver",
                "status": "UNREACHABLE",
                "configured": True,
                "error": str(e),
            }

    def investigate_domain(
        self,
        url_or_domain: str,
        use_cache: bool = True,
    ) -> DomainInvestigationRecord | None:
        """Lookup registration metadata for a target domain.

        Args:
            url_or_domain: Target URL or domain name.
            use_cache: Whether to use cached lookup.

        Returns:
            DomainInvestigationRecord or None if unavailable/disabled.
        """
        if not self.enabled:
            return None

        domain = extract_domain(url_or_domain)
        if not domain:
            return None

        if use_cache and domain in self._cache:
            return self._cache[domain]

        lookup_url = f"{self.base_url}/{domain}"
        try:
            logger.info("Performing RDAP lookup for domain: %s", domain)
            resp = self.session.get(
                lookup_url,
                headers={"Accept": "application/rdap+json, application/json"},
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                logger.info("RDAP lookup for %s returned status %d", domain, resp.status_code)
                self._cache[domain] = None
                return None

            data = resp.json()
            if not isinstance(data, dict):
                self._cache[domain] = None
                return None

            # Extract Registrar
            registrar = ""
            entities = data.get("entities", [])
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                roles = ent.get("roles", [])
                if "registrar" in roles:
                    vcard = ent.get("vcardArray", [])
                    if len(vcard) > 1 and isinstance(vcard[1], list):
                        for item in vcard[1]:
                            if isinstance(item, list) and len(item) > 3 and item[0] == "fn":
                                registrar = str(item[3])
                                break
                    if not registrar:
                        registrar = str(ent.get("handle") or "")
                    break

            # Extract Registration & Last Changed Dates from events
            registration_date = ""
            last_changed_date = ""
            events = data.get("events", [])
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                action = str(ev.get("eventAction") or "").lower()
                date_val = str(ev.get("eventDate") or "")
                if action == "registration" and not registration_date:
                    registration_date = date_val
                elif action in ("last changed", "last update") and not last_changed_date:
                    last_changed_date = date_val

            # Extract Nameservers
            nameservers: list[str] = []
            ns_list = data.get("nameservers", [])
            for ns in ns_list:
                if isinstance(ns, dict):
                    ldh_name = ns.get("ldhName")
                    if ldh_name:
                        nameservers.append(str(ldh_name).lower())

            # Extract Status
            statuses: list[str] = []
            status_field = data.get("status", [])
            if isinstance(status_field, list):
                statuses = [str(s) for s in status_field]

            record = DomainInvestigationRecord(
                domain=domain,
                registrar=registrar,
                registration_date=registration_date,
                last_changed_date=last_changed_date,
                status=statuses,
                nameservers=nameservers,
                collection_time=datetime.now(timezone.utc).isoformat(),
                provenance="RDAP",
                evidence_note=(
                    "Contextual domain registration data only; does not establish malicious attribution or intent."
                ),
            )

            self._cache[domain] = record
            return record

        except requests.exceptions.Timeout:
            logger.warning("RDAP lookup for %s timed out", domain)
            self._cache[domain] = None
            return None
        except requests.exceptions.RequestException as e:
            logger.warning("RDAP lookup for %s failed: %s", domain, e)
            self._cache[domain] = None
            return None
        except Exception as e:
            logger.error("Unexpected error in RDAP lookup for %s: %s", domain, e)
            self._cache[domain] = None
            return None
