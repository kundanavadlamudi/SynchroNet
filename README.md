# NexTrace

**NexTrace** is an Explainable Social Media Coordinated Inauthentic Behavior, OSINT & Disinformation Detection Intelligence System. It identifies coordinated posting patterns, enriches narrative themes with open-source intelligence (GDELT DOC 2.0 API, Google Fact Check Tools API, and RDAP domain registration telemetry), and classifies accounts using a benchmark-trained Random Forest classifier.

> ⚠️ **Important Ethical & Operational Disclaimer**  
> NexTrace generates investigation leads, **not definitive proof** of disinformation, malicious intent, or coordinated manipulation. A high coordination score reflects synchronization across semantic, temporal, and shared URL signals. Third-party fact-check ratings belong exclusively to the publishing organizations and are never presented as NexTrace's independent conclusions. The absence of a fact check does NOT mean a claim is true or false. OCR text extraction provides textual tokens for indexing and does NOT prove visual forgery or photoshopping. Every flagged account pair or campaign candidate requires human investigator review.

---

## Table of Contents

1. [Verified System Status](#1-verified-system-status)
2. [Technology Stack](#2-technology-stack)
3. [Directory Layout](#3-directory-layout)
4. [Prerequisites & Environment Setup](#4-prerequisites--environment-setup)
5. [Data Contracts & Schemas](#5-data-contracts--schemas)
6. [OSINT Data Sources & Verification](#6-osint-data-sources--verification)
7. [Social Media API Ingestion (X & Reddit)](#7-social-media-api-ingestion-x--reddit)
8. [Running the Analysis Pipeline](#8-running-the-analysis-pipeline)
9. [Trained Bot Classifier & Responsible ML](#9-trained-bot-classifier--responsible-ml)
10. [Investigation Dashboard (Streamlit)](#10-investigation-dashboard-streamlit)
11. [Running Automated Tests](#11-running-automated-tests)
12. [Configuration & Environment Variables](#12-configuration--environment-variables)
13. [Troubleshooting & Known Limitations](#13-troubleshooting--known-limitations)

---

## 1. Verified System Status

As of October 2026:
- **Test Suite:** **67 passing automated unit and integration tests** (0 failures).
- **Model Integrity:** `models/selected_bot_classifier.joblib` (8.9 MB) preserved byte-for-byte and verified.
- **OSINT Enrichment:** Automated correlation of campaign narratives with GDELT 2.0 global news, Google Fact Check claim reviews, and RDAP domain telemetry.
- **Real Social APIs:** Full X API v2 Recent Search adapter (`/2/tweets/search/recent`) and Reddit Data API search adapter with bounded exponential backoff and rate limit handling.
- **Resilient NLP:** Dual-engine similarity using `SentenceTransformer` (`all-MiniLM-L6-v2`) when available, and sublinear TF-IDF character/word n-gram cosine similarity fallback.
- **End-to-End Orchestration:** Ingested post accounts are automatically evaluated with the trained Random Forest classifier, producing unified behavior, coordination, OSINT intelligence, and bot classification reports.

---

## 2. Technology Stack

| Component | Libraries | Role |
|---|---|---|
| Core Data Science | `pandas`, `numpy`, `scipy` | Data manipulation, contracts, cleaning |
| Machine Learning | `scikit-learn`, `joblib` | Random Forest bot classifier, imputer |
| Graph Analysis | `networkx` | Account coordination network & connected components |
| Text Similarity & NLP | `sentence-transformers`, `scikit-learn` | Neural embeddings & TF-IDF cosine similarity |
| News Intelligence | GDELT DOC 2.0 API (`requests`) | Global news coverage & publication timelines |
| Claim Verification | Google Fact Check Tools API (`requests`) | IFCN-certified third-party fact checks |
| Domain Telemetry | RDAP Protocol (`requests`, `ipaddress`) | SSRF-safe domain registration intelligence |
| Computer Vision & OCR | `pytesseract`, `PIL` / Heuristic | Image text extraction with disclaimers |
| Social Media APIs | Twitter/X API v2, Reddit Data API | Public tweet & submission search adapters |
| Web Dashboard | `streamlit`, `plotly` | 5-tab intelligence analysis interface |
| Testing | `pytest`, `unittest` | Automated regression & integration test suite |

---

## 3. Directory Layout

```
NexTrace/
├── dashboard/
│   └── app.py                             # 5-tab Streamlit intelligence dashboard with OSINT panels
├── data/
│   ├── raw/                               # Hashtags text files
│   └── processed/                         # Cleaned posts, sample data, keywords
├── docs/
│   ├── IMPLEMENTATION_AUDIT.md            # Comprehensive baseline repository audit
│   └── MODEL_EVALUATION_REPORT.md         # Responsible bot classifier audit & evaluation
├── models/
│   ├── selected_bot_classifier.joblib     # Pretrained Random Forest model bundle (8.9 MB)
│   ├── selected_bot_classifier_backup.joblib # Preserved model backup
│   ├── selected_bot_classifier_metadata.json # Provenance & validation metrics
│   └── background_sample.csv              # SHAP background sample
├── results/
│   ├── graphs/                            # coordination_graph.png, coordination_graph.graphml
│   ├── reports/                           # Output CSVs, JSON evaluation summaries, OSINT results
│   └── backups/                           # Timestamped run backups (--backup-existing)
├── src/
│   ├── config.py                          # Centralized thresholds, weights, endpoints, and paths
│   ├── contracts.py                       # Validated schemas for posts, accounts, results, and OSINT
│   ├── create_sample_data.py              # Synthetic demo fixture generator
│   ├── data_preprocessing.py              # Data cleaning and normalization
│   ├── keyword_loader.py                  # Hashtag parsing from data/raw/
│   ├── semantic_similarity.py             # Pairwise text similarity (neural or TF-IDF fallback)
│   ├── coordination_detection.py          # Multi-signal coordination scorer
│   ├── account_analysis.py                # Behavioral aggregations & ML inference
│   ├── campaign_analysis.py               # Connected components via NetworkX
│   ├── graph_analysis.py                  # GraphML export and Matplotlib rendering
│   ├── risk_scoring.py                    # Explainable investigation priority & evidence
│   ├── osint/                             # Open Source Intelligence Package
│   │   ├── gdelt_client.py                # GDELT DOC 2.0 API news search
│   │   ├── factcheck_client.py            # Google Fact Check Tools API claim search
│   │   ├── rdap_client.py                 # SSRF-defended RDAP domain investigator
│   │   ├── vision_ocr.py                  # Vision OCR text extractor with disclaimers
│   │   └── enrichment.py                  # Narrative & evidence enrichment orchestrator
│   ├── ingestion/                         # Data provider boundary package
│   │   ├── base.py                        # BaseDataProvider and IngestionError hierarchy
│   │   ├── synthetic_provider.py          # Isolated demo fixture provider
│   │   ├── file_provider.py               # Local CSV / JSON / JSONL importer
│   │   ├── api_adapter.py                 # Authenticated REST feed adapter
│   │   ├── x_provider.py                  # X API v2 Recent Search provider
│   │   └── reddit_provider.py             # Reddit API search provider
│   └── ml/                                # Bot classifier training & evaluation tools
│       ├── features.py                    # Account feature extraction (21 features)
│       ├── model_report.py                # Read-only evaluation report generator
│       ├── predict_account.py             # Model bundle loader & inference
│       ├── prepare_dataset.py             # Cresci-2015/2017 dataset parser
│       └── train_models.py                # Guarded model training script
├── tests/
│   ├── test_backup.py                     # 12 backup utility unit tests
│   ├── test_ingestion_and_contracts.py    # 15 tests for schemas, providers, mocks, & pipeline
│   ├── test_ml_pipeline.py                # 11 tests for ML preparation, features, & inference
│   ├── test_nextrace.py                   # 4 tests for preprocessing, coordination, & risk
│   ├── test_nlp_and_similarity.py         # 6 tests for NLP paraphrase, duplicate, & Unicode
│   ├── test_osint_enrichment.py           # 12 tests for GDELT, Fact Check, RDAP, OCR, & Enricher
│   └── test_social_apis.py                # 7 tests for X API v2, Reddit API, token masking, & rate limits
├── run_pipeline.py                        # Main pipeline CLI orchestrator
├── run_ml_pipeline.py                     # ML status checker and retrain guard
├── requirements.txt                       # Pinned Python dependencies
└── .env.example                           # Sanitized environment configuration template
```

---

## 4. Prerequisites & Environment Setup

- **Python:** Python 3.10 to 3.13 (Python 3.11 recommended).
- **Setup Commands:**

```bash
# Clone the repository
git clone https://github.com/kundanavadlamudi/SynchroNet.git
cd SynchroNet

# Create a clean virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Data Contracts & Schemas

NexTrace enforces formal data contracts (`src/contracts.py`) to prevent corrupt or malformed data from crashing the application:

- **Post Record (`PostRecord`):**
  - `post_id` (str, required, unique)
  - `account_id` (str, required)
  - `text` (str, required)
  - `timestamp` (ISO 8601 UTC timestamp)
  - `url` (str, optional shared link)
  - `source_platform` ("demo", "x", "reddit", "file_import")
  - `provenance` ("synthetic_demo", "x_api_v2", "reddit_api", file path)
  - `engagement_likes`, `engagement_shares`, `language`
- **Account Record (`AccountRecord`):**
  - `account_id` (str, required)
  - `followers_count`, `friends_count`, `statuses_count`, `favourites_count`, `listed_count`
  - Profile flags: `default_profile`, `geo_enabled`, `verified`, `protected`, `description_length`
- **News Article Record (`NewsArticleRecord`):**
  - `url`, `title`, `seen_date`, `domain`, `language`, `source_country`, `query_term`, `provenance`
- **Fact Check Record (`FactCheckRecord`):**
  - `query_term`, `claim_text`, `claimant`, `claim_date`, `publisher_name`, `publisher_site`, `review_url`, `review_title`, `review_date`, `textual_rating`, `provenance`
- **Domain Investigation Record (`DomainInvestigationRecord`):**
  - `domain`, `registrar`, `registration_date`, `last_changed_date`, `status`, `nameservers`, `evidence_note`
- **OSINT Enrichment Bundle (`OSINTEnrichmentBundle`):**
  - `target_id`, `target_type`, `matched_news`, `matched_fact_checks`, `domain_investigations`, `limitations`

---

## 6. OSINT Data Sources & Verification

### A. GDELT DOC 2.0 API (News Intelligence)
- Endpoint: `https://api.gdeltproject.org/api/v2/doc/doc`
- Queries global news coverage matching campaign narrative keywords.
- Extracts publication timestamps, article titles, source domains, and languages.
- Requires no API key; implements timeout, query validation, and in-memory caching.

### B. Google Fact Check Tools API (Claim Verification)
- Endpoint: `https://factchecktools.googleapis.com/v1alpha1/claims:search`
- Queries claim reviews from IFCN-certified independent fact-checking organizations.
- Retains original source publisher attribution, review URL, and textual rating.
- If unconfigured, reports `BLOCKED_MISSING_API_KEY` and fails gracefully without blocking the pipeline.

### C. RDAP Domain Telemetry
- Resolves registration metadata for URLs shared by coordinated accounts.
- Features strict SSRF defense: blocks loopback (`127.0.0.1`), private RFC1918 subnets, and malformed hosts.
- Retrieves registrar name, creation date, event history, status, and nameservers.
- Contextual evidence only: does not imply malicious attribution or intent.

### D. Computer Vision & OCR
- Extracts embedded text from meme and infographic URLs for indexing.
- Includes mandatory disclaimer: OCR does not demonstrate visual manipulation or deceptive intent.

---

## 7. Social Media API Ingestion (X & Reddit)

### A. Twitter / X API v2 Provider (`XAPIProvider`)
- Authenticates via `X_BEARER_TOKEN` with recent search endpoint: `/2/tweets/search/recent`.
- Retrieves tweet text, timestamps, author IDs, public metrics (`like_count`, `retweet_count`).
- Expands user profiles via `author_id` to extract follower/following counts, status counts, and verification badges.
- Automatically handles HTTP 429 rate limits using exponential backoff and `Retry-After` / `x-rate-limit-reset`.
- Masks API tokens in all log outputs (`AAA****ZZZ`).

### B. Reddit API Provider (`RedditAPIProvider`)
- Ingests public submissions and user karma using custom User-Agent headers.
- Supports both public JSON search feeds and authenticated OAuth workflows.
- Normalizes Reddit titles, post text, and engagement metrics into canonical contracts.

---

## 8. Running the Analysis Pipeline

The pipeline orchestrator (`run_pipeline.py`) runs all stages in sequence:

```bash
# 1. Run standard demo mode with automated OSINT enrichment:
python run_pipeline.py

# 2. Ingest from live X API v2 (requires X_BEARER_TOKEN in .env):
python run_pipeline.py --source x --query "disinformation"

# 3. Ingest from Reddit Data API:
python run_pipeline.py --source reddit --query "election security"

# 4. Ingest an authorized local CSV or JSON file of posts:
python run_pipeline.py --source file --posts path/to/posts.csv

# 5. Back up existing outputs before overwriting:
python run_pipeline.py --backup-existing

# 6. Dry-run mode (validates inputs and contracts without writing outputs):
python run_pipeline.py --dry-run
```

---

## 9. Trained Bot Classifier & Responsible ML

- **Preserved Model:** `models/selected_bot_classifier.joblib` (Random Forest, 200 trees, median imputer).
- **Benchmark Performance (Held-out test set, N = 2,448):**
  - Accuracy: `0.9853`
  - Precision: `0.9963`
  - Recall: `0.9816`
  - F1 Score: `0.9889`
  - ROC-AUC: `0.9985`
- **Inspect Status Safely:**
  ```bash
  python run_ml_pipeline.py --status
  ```
- **Audit & Evaluation Report:** See `docs/MODEL_EVALUATION_REPORT.md` for cross-group generalization analysis, leakage audits, and ethical limitations.

---

## 10. Investigation Dashboard (Streamlit)

Launch the interactive 5-tab dashboard from the repository root:

```bash
streamlit run dashboard/app.py
```

### Dashboard Tabs
1. **Posts Tab:** Displays ingested posts, keyword frequencies, and platform indicators. Clearly marks demo data.
2. **Coordination Detection Tab:** Displays flagged pairs, time difference vs. similarity scatter plots, and score distributions.
3. **Risk Scores Tab:** Shows investigation priority breakdown (High, Medium, Low) and transparent signal evidence.
4. **Campaign Analysis Tab:**
   - Interactive Plotly network visualization of connected account components.
   - **OSINT & Narrative Intelligence Panel:** Displays matched GDELT global news articles with direct links, Google Fact Check claim reviews with publisher attribution, and RDAP domain telemetry for shared links.
5. **Bot Classifier Tab:**
   - Displays automatic ML predictions for active dataset accounts (`account_classification_results.csv`).
   - View benchmark model comparison and confusion matrix.
   - Interactive single-account classification form with feature presets.
- **Sidebar Utilities:**
   - **Live API & OSINT Services drawer:** Real-time operational status for X API, Reddit API, GDELT, Google Fact Check, RDAP Resolver, and Vision OCR.
   - One-click CSV/JSON dataset uploader.

---

## 11. Running Automated Tests

Run the full automated test suite using `pytest`:

```bash
python -m pytest tests/ -v
```

All **67 tests** execute deterministically without external network dependencies:
- `tests/test_backup.py`: 12 tests for backup directory safety and immutability.
- `tests/test_ingestion_and_contracts.py`: 15 tests for schemas, malformed inputs, file providers, and pipeline integration.
- `tests/test_ml_pipeline.py`: 11 tests for dataset loading, feature extraction, train/test split safety, and model persistence.
- `tests/test_nextrace.py`: 4 tests for preprocessing, coordination scoring, and risk explanation.
- `tests/test_nlp_and_similarity.py`: 6 tests for near-duplicate paraphrasing, structural paraphrasing, unrelated sentences, empty text, and Unicode handling.
- `tests/test_osint_enrichment.py`: 12 tests for GDELT parsing, Google Fact Check attribution, RDAP SSRF defense, Vision OCR disclaimers, and enrichment orchestration.
- `tests/test_social_apis.py`: 7 tests for X API v2 normalization, token masking, authentication errors, rate limiting backoff, pagination, and Reddit search.

---

## 12. Configuration & Environment Variables

Copy `.env.example` to `.env` to configure settings:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `X_BEARER_TOKEN` | *(None)* | Authorized Twitter / X API v2 Bearer Token |
| `REDDIT_CLIENT_ID` | *(None)* | Reddit developer script client ID |
| `REDDIT_CLIENT_SECRET` | *(None)* | Reddit developer script secret |
| `REDDIT_USER_AGENT` | *NexTrace...* | Custom Reddit User-Agent string |
| `GOOGLE_FACT_CHECK_API_KEY` | *(None)* | Google Cloud API key for Fact Check Tools API |
| `NEXTRACE_ENABLE_RDAP` | `true` | Enable/disable RDAP domain registration lookups |
| `NEXTRACE_ENABLE_OCR` | `true` | Enable/disable image OCR text extraction |
| `NEXTRACE_SEMANTIC_THRESHOLD` | `0.70` | Pairwise text similarity threshold |
| `NEXTRACE_TIME_WINDOW_MINUTES` | `10` | Maximum minutes between co-posts |
| `NEXTRACE_API_BASE_URL` | *(None)* | Generic authorized REST API base URL |
| `NEXTRACE_API_BEARER_TOKEN` | *(None)* | Generic server-side API authentication token |

---

## 13. Troubleshooting & Known Limitations

1. **SentenceTransformers Fallback:** If `torch` or `sentence-transformers` is not installed, NexTrace automatically uses TF-IDF cosine similarity. To use neural embeddings, install sentence-transformers: `pip install sentence-transformers`.
2. **Platform Live Ingestion:** Direct live scraping without platform authorization is strictly prohibited by platform policies. Use official developer credentials configured in `.env`.
3. **Model Retraining:** Never retrain `models/selected_bot_classifier.joblib` without explicit justification and confirmation.