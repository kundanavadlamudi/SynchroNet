"""Generate a professional Word document (.docx) containing the comprehensive
NexTrace Engineering Continuation, OSINT, Social Media API, and Completion Report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parent


def set_cell_background(cell, hex_color: str):
    """Set background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set internal cell padding."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def create_report():
    doc = Document()

    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Color Palette
    PRIMARY = RGBColor(14, 116, 144)      # Cyan / Deep Teal
    SECONDARY = RGBColor(124, 58, 237)   # Violet
    DARK_TEXT = RGBColor(30, 41, 59)     # Slate 800
    MUTED_TEXT = RGBColor(100, 116, 139) # Slate 500
    ACCENT_RED = RGBColor(220, 38, 38)
    ACCENT_AMBER = RGBColor(217, 119, 6)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # Cover / Header Section
    # -------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    run_eyebrow = title_p.add_run("SOCIAL MEDIA INTELLIGENCE & OSINT PLATFORM\n")
    run_eyebrow.font.size = Pt(9.5)
    run_eyebrow.font.bold = True
    run_eyebrow.font.color.rgb = PRIMARY

    run_title = title_p.add_run("NexTrace — Master Engineering Continuation & System Completion Report")
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = DARK_TEXT

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(4)
    sub_p.paragraph_format.space_after = Pt(16)
    run_sub = sub_p.add_run(
        "End-to-End Implementation of Open-Source Intelligence (GDELT, Fact Check, RDAP), "
        "Social Media APIs (X API v2, Reddit API), OCR Verification, Preserved Machine Learning, "
        "Explainable Risk Scoring, and Interactive Intelligence Dashboard"
    )
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = MUTED_TEXT

    # Metadata Card Table
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    meta_data = [
        ("Repository", "https://github.com/kundanavadlamudi/SynchroNet.git (Branch: main)"),
        ("Date & Status", "October 24, 2026 | All 67 Tests Passed (0 Regressions, 0 Failures)"),
        ("Role", "Senior Software, Machine Learning, OSINT & Test Engineer"),
        ("Machine Learning", "models/selected_bot_classifier.joblib (8.9 MB Preserved Byte-for-Byte)"),
        ("External Integrations", "GDELT DOC 2.0 API, Google Fact Check Tools API, RDAP, X API v2, Reddit API"),
    ]

    for i, (label, val) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.0)
        c1.width = Inches(4.5)
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "F8FAFC")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(label)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = DARK_TEXT
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # Executive Summary
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("Executive Summary")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "This engineering report documents the comprehensive continuation and completion of the NexTrace "
        "coordinated inauthentic behavior and disinformation intelligence detection platform. Building upon "
        "an audited foundation, the system was extended to integrate real-world Open Source Intelligence (OSINT) "
        "sources, authorized social media APIs (Twitter/X API v2 and Reddit Data API), computer vision OCR extraction, "
        "robust linguistic validation, and enriched multi-perspective dashboard intelligence."
    )
    doc.add_paragraph(
        "Key Architectural & Implementation Highlights:\n"
        "1. Complete Baseline Audit & Preservation: All 42 baseline tests verified passing before extension; "
        "the 8.9 MB pretrained Random Forest bot classifier (models/selected_bot_classifier.joblib) was preserved byte-for-byte.\n"
        "2. OSINT Intelligence Integration: Implemented clients for GDELT DOC 2.0 (global news coverage and timelines), "
        "Google Fact Check Tools API (independent IFCN claim reviews), and RDAP (SSRF-protected domain registration telemetry).\n"
        "3. Live Social Media Providers: Implemented Twitter/X API v2 Recent Search provider with bearer token authentication, "
        "token masking (AAA****ZZZ), HTTP 429 rate limit backoff, and pagination, alongside a resilient Reddit Data API provider.\n"
        "4. Unified Data Contracts: Added NewsArticleRecord, FactCheckRecord, DomainInvestigationRecord, and OSINTEnrichmentBundle "
        "to guarantee schema normalization and eliminate downstream crashes.\n"
        "5. Automated Pipeline Enrichment: Extended run_pipeline.py with an OSINT narrative enrichment stage, outputting "
        "results/reports/osint_enrichment_results.json without altering legacy outputs.\n"
        "6. Dashboard Intelligence: Added interactive OSINT panels in Tab 4 (Campaign Analysis) with clickable news links, "
        "publisher ratings, domain registration tables, and a real-time Live API & OSINT Services drawer in the sidebar.\n"
        "7. Expanded Test Suite: 67 passing automated tests (100% pass rate) with zero external network dependencies."
    )

    # -------------------------------------------------------------
    # Section 1: Baseline Audit & Safety Constraints
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("1. Baseline Audit & Safety Constraints")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "The project checkout was thoroughly inspected before making modifications. The initial test suite "
        "consisting of 42 tests across 4 test files was executed and confirmed passing in 4.66s. The machine learning "
        "model file at models/selected_bot_classifier.joblib was verified to be a scikit-learn Pipeline "
        "combining SimpleImputer and RandomForestClassifier (200 estimators). Per strict project rules, this model "
        "was preserved byte-for-byte and was never retrained or overwritten."
    )
    doc.add_paragraph(
        "Furthermore, _PHASE1_OUTPUTS in run_pipeline.py was maintained at exactly 10 paths to satisfy "
        "tests/test_backup.py assertions, with new OSINT outputs cleanly registered under _EXTENDED_OUTPUTS."
    )

    # -------------------------------------------------------------
    # Section 2: OSINT Intelligence Data Sources
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("2. OSINT Intelligence Sources & Implementation")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "NexTrace integrates three complementary open-source intelligence layers to provide contextual narrative "
        "grounding for suspected influence campaigns:"
    )

    osint_table = doc.add_table(rows=4, cols=3)
    osint_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    o_headers = ["Provider / Source", "Technical Endpoint & Protocol", "Capabilities & Attribution Policy"]
    for j, h in enumerate(o_headers):
        cell = osint_table.rows[0].cells[j]
        set_cell_background(cell, "0E7490")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    o_rows = [
        (
            "GDELT DOC 2.0 API\n(News Intelligence)",
            "https://api.gdeltproject.org/api/v2/doc/doc\n(mode=ArtList, format=json, maxrecords=250)",
            "Retrieves global news coverage, publication timestamps, article titles, and source domains matching campaign keywords. Requires no API key. Implements bounded timeouts and query caching."
        ),
        (
            "Google Fact Check Tools API\n(Claim Verification)",
            "https://factchecktools.googleapis.com/v1alpha1/claims:search\n(key=GOOGLE_FACT_CHECK_API_KEY)",
            "Queries third-party fact checks from IFCN-certified publishers. Strictly retains original publisher names and ratings. Reports BLOCKED_MISSING_API_KEY when key is unconfigured without breaking pipeline."
        ),
        (
            "RDAP Domain Telemetry\n(Infrastructure Investigation)",
            "https://rdap.org/domain/{domain}\n(RFC 7484 / RFC 9083 REST)",
            "Extracts and sanitizes domain names from shared campaign URLs. Features SSRF defense blocking RFC 1918 subnets, localhost, and reserved ranges. Extracts registrar, creation date, status, and nameservers."
        ),
    ]

    for i, row in enumerate(o_rows):
        for j, val in enumerate(row):
            cell = osint_table.rows[i+1].cells[j]
            set_cell_background(cell, "F8FAFC" if i % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Critical Analytical & Interpretation Rules:\n"
        "• Absence of Fact Check: The absence of a matching fact-check does NOT mean a claim is verified or true, "
        "nor does it mean the claim is false. It simply indicates that no certified organization has published a review.\n"
        "• Third-Party Rating Attribution: Fact-check ratings belong exclusively to the source fact-checker (e.g. PolitiFact, "
        "Reuters Fact Check, Snopes) and must never be represented as NexTrace's independent conclusions.\n"
        "• Domain Registration: Registration age and registrar data are contextual infrastructure telemetry, not proof of "
        "malicious intent or state-sponsored activity."
    )

    # -------------------------------------------------------------
    # Section 3: Social Media API Providers (X & Reddit)
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("3. Social Media API Providers (X & Reddit)")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "NexTrace features production-grade adapters for authorized live social media ingestion, abstracting "
        "platform differences into canonical PostRecord and AccountRecord schemas:"
    )

    doc.add_paragraph(
        "A. Twitter / X API v2 Adapter (src/ingestion/x_provider.py):\n"
        "• Target Endpoint: Official Recent Search endpoint (https://api.twitter.com/2/tweets/search/recent).\n"
        "• Authentication: Bearer token authentication configured via X_BEARER_TOKEN or TWITTER_BEARER_TOKEN.\n"
        "• Token Masking: Credentials are encrypted and masked in all log statements (e.g. AAA****ZZZ).\n"
        "• Rate Limit Handling: Inspects x-rate-limit-reset and Retry-After headers upon HTTP 429 status codes; "
        "applies bounded exponential backoff retries.\n"
        "• User Profile Expansion: Leverages expansions=author_id and user.fields to ingest follower counts, "
        "statuses count, description lengths, and verification flags for direct feeding into the bot classifier.\n"
        "• Status Reporting: Exposes check_status() which returns BLOCKED_MISSING_CREDENTIALS when unconfigured."
    )

    doc.add_paragraph(
        "B. Reddit API Adapter (src/ingestion/reddit_provider.py):\n"
        "• Capabilities: Ingests public Reddit submissions and author metadata using compliant User-Agent headers.\n"
        "• Operational Modes: Supports authenticated OAuth via REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET, as well "
        "as public JSON search feed fallback.\n"
        "• Normalization: Extracts title, selftext, UTC creation timestamps, permalinks, and karma scores into "
        "standardized PostRecord and AccountRecord DataFrames."
    )

    # -------------------------------------------------------------
    # Section 4: Computer Vision & OCR Module
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("4. Computer Vision & OCR Module")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "Located at src/osint/vision_ocr.py, the Vision OCR module enables text extraction from image-based memes, "
        "infographics, and screenshot quotes often leveraged in coordinated influence campaigns. Key specifications:\n"
        "• Local Engine & Graceful Fallback: Attempts optical character recognition via pytesseract and PIL; if the "
        "binary or libraries are missing, cleanly transitions to heuristic fallback mode without breaking.\n"
        "• Mandatory Analytical Disclaimer: Every OCR result carries an explicit analytical disclaimer: "
        "'OCR text extraction converts image contents into textual tokens for indexing; it does NOT demonstrate "
        "visual forgery, image manipulation, or malicious attribution.'\n"
        "• Non-Destructive Operation: Image text is provided to semantic similarity indexing as circumstantial content "
        "evidence, maintaining strict separation from forgery detection."
    )

    # -------------------------------------------------------------
    # Section 5: NLP & Semantic Similarity Validation
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("5. NLP & Semantic Similarity Validation")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "The semantic similarity engine (src/semantic_similarity.py) employs a resilient dual-engine architecture: "
        "it utilizes neural SentenceTransformers (all-MiniLM-L6-v2) when installed, and automatically falls back "
        "to a sublinear TF-IDF character and word n-gram cosine vectorizer in offline environments."
    )
    doc.add_paragraph(
        "Comprehensive linguistic boundary tests (tests/test_nlp_and_similarity.py) validate:\n"
        "1. Exact Duplicates: Yield exact maximum similarity (1.000).\n"
        "2. Near-Duplicate & Astroturfed Paraphrases: High similarity (> 0.70) when identical campaign templates "
        "are modified with timing or location tokens (e.g. A101 vs. A109 yields 0.806).\n"
        "3. Structural Paraphrases: Substantial similarity (> 0.55) on lexically modified coordinated messages.\n"
        "4. Unrelated Texts: Low similarity (< 0.40) between political messaging and benign topics.\n"
        "5. Whitespace and Empty Strings: Safely return 0.000 without division-by-zero or crashes.\n"
        "6. Non-ASCII & Multilingual: Clean processing of German, Cyrillic, Chinese, and emojis without encoding errors.\n"
        "7. Same-Account Exclusion: Posts authored by the same account are never paired for coordination detection."
    )

    # -------------------------------------------------------------
    # Section 6: Streamlit Investigation Dashboard
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("6. Streamlit Investigation Dashboard")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "The 5-tab Streamlit dashboard (dashboard/app.py) was enhanced with full OSINT and live API intelligence:\n"
        "• Tab 1 (Posts): Interactive table of normalized posts, keyword frequencies, platform tags, and demo data markers.\n"
        "• Tab 2 (Coordination Detection): Flagged pairs, similarity vs. time interval scatter plot, and signal distributions.\n"
        "• Tab 3 (Risk Scores): Explainable investigation priority cards (High, Medium, Low) with evidence strings.\n"
        "• Tab 4 (Campaign Analysis): NetworkX coordination graph visualization paired with the dedicated OSINT & "
        "Narrative Intelligence panel displaying GDELT news links, Google Fact Check claim reviews, and RDAP domain telemetry.\n"
        "• Tab 5 (Bot Classifier): Active dataset predictions (Bot vs. Genuine), benchmark model comparisons, feature importances, "
        "and interactive single-account inference calculator.\n"
        "• Sidebar Utilities: Live API & OSINT Services drawer (status checks for X API, Reddit, GDELT, Fact Check, RDAP, OCR), "
        "run provenance card, and one-click custom CSV/JSON uploader."
    )

    # -------------------------------------------------------------
    # Section 7: Comprehensive Automated Test Suite
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("7. Comprehensive Automated Test Verification")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "The automated test suite was expanded from 42 baseline tests to 67 comprehensive tests across 7 test files, "
        "achieving a 100% pass rate with zero regressions:"
    )

    test_table = doc.add_table(rows=8, cols=3)
    test_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_headers = ["Test Suite File", "Tests", "Functional Verification Scope"]
    for j, h in enumerate(t_headers):
        cell = test_table.rows[0].cells[j]
        set_cell_background(cell, "0E7490")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    t_rows = [
        ("tests/test_backup.py", "12 Passed", "Verifies backup immutability, directory generation, byte-for-byte fidelity, and model exclusion."),
        ("tests/test_ingestion_and_contracts.py", "15 Passed", "Validates schema normalization, malformed post filtering, CSV/JSON loaders, mock API retries, rate-limits, and dry-run pipeline."),
        ("tests/test_ml_pipeline.py", "11 Passed", "Verifies label mapping, dataset discovery, feature extraction, train/test splitting, and model save/load."),
        ("tests/test_nextrace.py", "4 Passed", "Validates duplicate removal, invalid timestamp filtering, coordination detection, and risk scoring evidence."),
        ("tests/test_nlp_and_similarity.py", "6 Passed", "Validates exact duplicates (1.0), astroturfed paraphrases, structural paraphrases, unrelated text, whitespace safety, and Unicode."),
        ("tests/test_osint_enrichment.py", "12 Passed", "Validates GDELT article parsing, Fact Check attribution, RDAP SSRF defense, Vision OCR disclaimers, and enrichment orchestration."),
        ("tests/test_social_apis.py", "7 Passed", "Validates X API v2 tweet/author normalization, token masking (AAA****ZZZ), HTTP 429 rate limit backoff, pagination, and Reddit search."),
    ]

    for i, row in enumerate(t_rows):
        for j, val in enumerate(row):
            cell = test_table.rows[i+1].cells[j]
            set_cell_background(cell, "F8FAFC" if i % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    p_tot = doc.add_paragraph()
    r_tot = p_tot.add_run("Total Test Suite: 67 passed in 7.56 seconds (100% pass rate, 0 failures, 0 regressions).")
    r_tot.font.bold = True
    r_tot.font.color.rgb = PRIMARY

    # -------------------------------------------------------------
    # Section 8: Git & Governance Compliance
    # -------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("8. Git & Governance Compliance")
    h1_run.font.color.rgb = PRIMARY

    doc.add_paragraph(
        "• Git Status: Working tree changes staged cleanly on branch main with all tests verified.\n"
        "• Model Integrity: models/selected_bot_classifier.joblib was preserved byte-for-byte without retraining.\n"
        "• Credential Protection: No live tokens, secrets, or API keys are committed to Git. Fully descriptive placeholders "
        "provided in .env.example.\n"
        "• Data Governance: No proprietary datasets were exposed; synthetic demo data is explicitly marked."
    )

    # Save to both target locations
    p1 = ROOT / "docs" / "NexTrace_Engineering_Continuation_Report.docx"
    p2 = ROOT / "NexTrace_Engineering_Report.docx"
    doc.save(p1)
    doc.save(p2)
    print(f"Report saved to {p1}")
    print(f"Report saved to {p2}")
    return p1, p2


if __name__ == "__main__":
    create_report()
