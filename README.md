# NexTrace

**NexTrace** is an Explainable Social Media Coordinated Inauthentic Behavior & Disinformation Detection Intelligence System. It identifies coordinated posting patterns by combining semantic text similarity, temporal synchronization, shared network relationships, and benchmark-trained Random Forest bot classification.

> ⚠️ **Important Ethical & Operational Disclaimer**  
> NexTrace generates investigation leads, **not definitive proof** of disinformation, malicious intent, or coordinated manipulation. A high coordination score or classifier probability reflects resemblance to historical benchmark patterns. Every flagged account pair or campaign candidate requires human investigator review before any conclusions are drawn.

---

## Table of Contents

1. [Verified System Status](#1-verified-system-status)
2. [Technology Stack](#2-technology-stack)
3. [Directory Layout](#3-directory-layout)
4. [Prerequisites & Environment Setup](#4-prerequisites--environment-setup)
5. [Data Contracts & Schemas](#5-data-contracts--schemas)
6. [Running the Analysis Pipeline](#6-running-the-analysis-pipeline)
7. [Data Ingestion Modes](#7-data-ingestion-modes)
8. [Trained Bot Classifier & Responsible ML](#8-trained-bot-classifier--responsible-ml)
9. [Investigation Dashboard (Streamlit)](#9-investigation-dashboard-streamlit)
10. [Running Automated Tests](#10-running-automated-tests)
11. [Configuration & Environment Variables](#11-configuration--environment-variables)
12. [Troubleshooting & Known Limitations](#12-troubleshooting--known-limitations)

---

## 1. Verified System Status

As of October 2026:
- **Test Suite:** **42 passing automated unit and integration tests** (0 failures).
- **Model Integrity:** `models/selected_bot_classifier.joblib` (8.9 MB) preserved intact and verified.
- **Resilient NLP:** `src/semantic_similarity.py` automatically uses `SentenceTransformer` (`all-MiniLM-L6-v2`) when available, and seamlessly falls back to scikit-learn TF-IDF n-gram cosine similarity in offline or CPU environments without PyTorch.
- **End-to-End Orchestration:** Ingested post accounts are automatically evaluated with the trained Random Forest classifier, producing unified behavior, coordination, and bot classification reports.

---

## 2. Technology Stack

| Component | Libraries | Role |
|---|---|---|
| Core Data Science | `pandas`, `numpy`, `scipy` | Data manipulation, contracts, cleaning |
| Machine Learning | `scikit-learn`, `joblib` | Random Forest bot classifier, imputer |
| Graph Analysis | `networkx` | Account coordination network & connected components |
| Text Similarity | `sentence-transformers`, `scikit-learn` | Neural embeddings & TF-IDF cosine similarity |
| Visualisation | `matplotlib`, `plotly` | Graph image export & interactive network plots |
| Web Dashboard | `streamlit` | 5-tab intelligence analysis interface |
| Ingestion & HTTP | `requests` | Generic API provider adapter with retry & backoff |
| Testing | `pytest`, `unittest` | Automated regression & integration test suite |

---

## 3. Directory Layout

```
NexTrace/
├── dashboard/
│   └── app.py                             # 5-tab Streamlit intelligence dashboard
├── data/
│   ├── raw/                               # Hashtags text files (Cresci datasets not in Git)
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
│   ├── reports/                           # Output CSVs, JSON evaluation summaries, run metadata
│   └── backups/                           # Timestamped run backups (--backup-existing)
├── src/
│   ├── config.py                          # Centralized thresholds, weights, and paths
│   ├── contracts.py                       # Validated schemas for posts, accounts, and results
│   ├── create_sample_data.py              # Synthetic demo fixture generator
│   ├── data_preprocessing.py              # Data cleaning and normalization
│   ├── keyword_loader.py                  # Hashtag parsing from data/raw/
│   ├── semantic_similarity.py             # Pairwise text similarity (neural or TF-IDF fallback)
│   ├── coordination_detection.py          # Multi-signal coordination scorer
│   ├── account_analysis.py                # Behavioral aggregations & ML inference
│   ├── campaign_analysis.py               # Connected components via NetworkX
│   ├── graph_analysis.py                  # GraphML export and Matplotlib rendering
│   ├── risk_scoring.py                    # Explainable investigation priority & evidence
│   ├── ingestion/                         # Data provider boundary package
│   │   ├── base.py                        # BaseDataProvider and IngestionError hierarchy
│   │   ├── synthetic_provider.py          # Isolated demo fixture provider
│   │   ├── file_provider.py               # Local CSV / JSON / JSONL importer
│   │   └── api_adapter.py                 # Authenticated REST feed adapter with retry & rate limits
│   └── ml/                                # Bot classifier training & evaluation tools
│       ├── features.py                    # Account feature extraction (21 features)
│       ├── model_report.py                # Read-only evaluation report generator
│       ├── predict_account.py             # Model bundle loader & inference
│       ├── prepare_dataset.py             # Cresci-2015/2017 dataset parser
│       └── train_models.py                # Model training (guarded against accidental rerun)
├── tests/
│   ├── test_backup.py                     # 12 backup utility unit tests
│   ├── test_ingestion_and_contracts.py    # 15 tests for schemas, providers, mocks, & pipeline
│   ├── test_ml_pipeline.py                # 11 tests for ML preparation, features, & inference
│   ├── test_nextrace.py                   # 4 tests for preprocessing, coordination, & risk
│   └── fixtures/
│       └── sample_fixtures.py             # Deterministic synthetic test fixtures
├── run_pipeline.py                        # Main pipeline CLI orchestrator
├── run_ml_pipeline.py                     # ML status checker and retrain guard
├── requirements.txt                       # Pinned Python dependencies
└── .env.example                           # Sanitized environment configuration template
```

---

## 4. Prerequisites & Environment Setup

- **Python:** Python 3.10 to 3.13 (Python 3.11 recommended).
- **Environment creation:**

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

NexTrace enforces formal data contracts (`src/contracts.py`) to prevent corrupt or malformed data from crashing the application or silently biasing risk scores:

- **Post Record:**
  - `post_id` (str, required, unique)
  - `account_id` (str, required)
  - `text` (str, required, stripped)
  - `timestamp` (ISO 8601 UTC timestamp, required)
  - `url` (str, optional shared link)
  - `source_platform` (str, e.g. "synthetic", "twitter", "mastodon", "reddit")
  - `provenance` (str, tracking file or API origin)
  - `collection_time` (ISO 8601 UTC timestamp)
- **Account Record:**
  - `account_id` (str, required)
  - Metadata: `followers_count`, `friends_count`, `statuses_count`, `favourites_count`, `listed_count`
  - Profile flags: `default_profile`, `geo_enabled`, `verified`, `protected`, `description_length`
- **Coordination & Risk Output:**
  - `account_1`, `account_2`, `post_1`, `post_2`
  - `semantic_similarity` (0.0 to 1.0)
  - `time_difference_minutes`
  - `shared_url` (bool)
  - `coordination_score` (0.0 to 1.0)
  - `investigation_priority` ("High", "Medium", "Low")
  - `evidence` (Transparent human-readable breakdown of contributing signals)

---

## 6. Running the Analysis Pipeline

The pipeline orchestrator (`run_pipeline.py`) runs all stages in sequence:

```bash
# 1. Run standard demo mode (generates and analyzes synthetic sample data):
python run_pipeline.py

# 2. Ingest an authorized local CSV or JSON file of posts:
python run_pipeline.py --source file --posts path/to/posts.csv

# 3. Ingest posts and separate account profile metadata:
python run_pipeline.py --source file --posts path/to/posts.csv --accounts path/to/accounts.csv

# 4. Back up existing outputs before overwriting:
python run_pipeline.py --backup-existing

# 5. Dry-run mode (validates inputs and contracts without writing outputs):
python run_pipeline.py --dry-run
```

---

## 7. Data Ingestion Modes

NexTrace cleanly separates data sources:
1. **Synthetic Demo Provider:** Built-in 10-post fixture containing a deterministic co-posting cluster (`A101`, `A102`, `A103`, `A106`, `A109`) and benign posts. Every record is explicitly marked `provenance="synthetic_demo"`.
2. **File Data Provider:** Imports local CSV or JSON/JSONL datasets. Missing fields are safely defaulted, invalid timestamps dropped with warnings, and duplicate IDs deduplicated.
3. **Generic REST API Provider:** Authenticated adapter for authorized platform APIs and public feeds. Configured via environment variables (`NEXTRACE_API_BASE_URL` and `NEXTRACE_API_BEARER_TOKEN`). Implements:
   - Server-side credential protection (tokens masked in all logs).
   - Bounded exponential backoff retry for HTTP 429 and 5xx errors.
   - Respects `Retry-After` headers and handles pagination safely.
   - If credentials are unset, reports `BLOCKED_MISSING_CREDENTIALS` with actionable instructions.

---

## 8. Trained Bot Classifier & Responsible ML

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

## 9. Investigation Dashboard (Streamlit)

Launch the interactive 5-tab dashboard from the repository root:

```bash
streamlit run dashboard/app.py
```

### Dashboard Tabs
1. **Posts Tab:** Displays ingested posts, keyword frequencies, and filters. Clearly marks demo data.
2. **Coordination Detection Tab:** Displays flagged pairs, time difference vs. similarity scatter plots, and score distributions.
3. **Risk Scores Tab:** Shows investigation priority breakdown (High, Medium, Low) and transparent signal evidence.
4. **Campaign Analysis Tab:** Interactive Plotly network visualization of connected account components and cluster metrics.
5. **Bot Classifier Tab:**
   - Displays automatic ML predictions for active dataset accounts (`account_classification_results.csv`).
   - View benchmark model comparison and confusion matrix.
   - Interactive single-account classification form with feature presets.
- **Sidebar Utilities:** Displays active data source, run timestamp, one-click CSV/JSON dataset uploader, and live API connection status.

---

## 10. Running Automated Tests

Run the full automated test suite using `pytest`:

```bash
python -m pytest tests/ -v
```

All **42 tests** execute deterministically without external network calls:
- `tests/test_backup.py`: 12 tests for backup directory safety and immutability.
- `tests/test_ingestion_and_contracts.py`: 15 tests for schema validation, malformed record handling, file providers, mock API retries, rate limits, and pipeline integration.
- `tests/test_ml_pipeline.py`: 11 tests for dataset loading, feature extraction, train/test split safety, and model persistence.
- `tests/test_nextrace.py`: 4 tests for preprocessing, coordination scoring, and risk explanation.

---

## 11. Configuration & Environment Variables

Copy `.env.example` to `.env` to configure settings:

```bash
# Copy template
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `NEXTRACE_API_BASE_URL` | *(None)* | Authorized REST API base URL |
| `NEXTRACE_API_BEARER_TOKEN` | *(None)* | Server-side API authentication token |
| `NEXTRACE_API_TIMEOUT_SECONDS` | `10` | HTTP request timeout |
| `NEXTRACE_API_MAX_RETRIES` | `3` | Max retry attempts for transient errors |
| `NEXTRACE_SEMANTIC_THRESHOLD` | `0.70` | Pairwise text similarity threshold |
| `NEXTRACE_TIME_WINDOW_MINUTES` | `10` | Maximum minutes between co-posts |

---

## 12. Troubleshooting & Known Limitations

1. **SentenceTransformers Fallback:** If `torch` or `sentence-transformers` is not installed, NexTrace automatically uses TF-IDF cosine similarity. To use neural embeddings, install sentence-transformers: `pip install sentence-transformers`.
2. **Platform Live Ingestion:** Direct live crawling without platform approval is strictly prohibited by platform terms of service. For live intelligence, obtain official developer credentials, configure them in `.env`, or use authorized data dumps.
3. **Model Retraining:** Never retrain `models/selected_bot_classifier.joblib` unless updated Cresci training datasets are provided. Retraining requires explicit confirmation via `python run_ml_pipeline.py --retrain`.