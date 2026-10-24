# NexTrace — Implementation & Baseline Audit Report

**Date:** October 24, 2026  
**Repository:** `https://github.com/kundanavadlamudi/SynchroNet.git`  
**Current Branch:** `main`  
**Working Tree:** Clean (verified via `git status`, commit `0ffcf14`)  
**Auditor:** Senior Python/ML Autonomous Systems Engineer  

---

## 1. Executive Summary & Verification

An exhaustive baseline audit of the NexTrace repository was conducted. Key findings:
- **Test Baseline:** The prior-reported test result of **27 passing unit tests was independently verified and reproduced** using the available Python runtime (`pytest 8.4.2`: 27 passed in 25.46s).
- **Model Artifact Preservation:** The trained bot classifier at `models/selected_bot_classifier.joblib` (8.9 MB) and its metadata `models/selected_bot_classifier_metadata.json` are present, undamaged, and verified loadable via `joblib`.
- **Runtime Environment:** The `.venv` (Python 3.11.9) described in historical README notes is not present in the current workspace checkout. Python 3.13.9 via Anaconda is active and satisfies core dependencies (`pandas 2.3.3`, `scikit-learn 1.7.2`, `networkx 3.5`, `streamlit 1.51.0`, `pytest 8.4.2`).
- **Critical Pipeline Blocker Found:** `run_pipeline.py` fails at Stage 4 (Semantic similarity) due to an unprotected hard dependency on `sentence_transformers` in `src/semantic_similarity.py`. When PyTorch or sentence-transformers is uninstalled or offline, the pipeline halts.
- **Ingestion Reality:** Live ingestion is **not implemented**. The system currently only generates 10 hardcoded synthetic posts (`src/create_sample_data.py`). There is no file ingestion (CSV/JSON) or authorized platform provider interface.
- **Module Disconnection:** Phase 1 (coordination detection) and Phase 2 (bot classification) are completely decoupled. Account-level behavioral features extracted in Phase 1 are never routed to `src/ml/predict_account.py` during the pipeline run.

---

## 2. Repository Structure & Entry Points

```
SynchroNet/
├── README.md                              # Main documentation (claims .venv exists, outlines 2-phase architecture)
├── requirements.txt                       # Pinned requirements (mentions PyTorch, sentence-transformers, shap, lime)
├── requirements-lock.txt                  # Full environment lock file
├── run_pipeline.py                        # Phase 1 pipeline orchestrator (CLI entry point)
├── run_ml_pipeline.py                     # Phase 2 ML status checker and retrain guard
├── validate_model.py                      # Standalone validation script for model loading & metadata
├── validate_ml.py                         # Standalone validation script for processed ml_dataset.csv
├── check_data_leakage.py                  # Standalone check for train/test split leakage
├── check_group_generalization.py          # Standalone evaluation for group-level generalization
├── dashboard/
│   └── app.py                             # 5-tab Streamlit dashboard (3,113 lines)
├── src/
│   ├── config.py                          # Thresholds, weights, and file path definitions
│   ├── data_preprocessing.py              # Validates and cleans posts dataframe
│   ├── create_sample_data.py              # Generates 10 hardcoded synthetic posts
│   ├── keyword_loader.py                  # Parses hashtags from data/raw/
│   ├── semantic_similarity.py             # Sentence embedding similarity (needs robust fallback)
│   ├── coordination_detection.py          # Multi-signal coordination scorer
│   ├── account_analysis.py                # Post-level aggregations per account
│   ├── campaign_analysis.py               # Connected component grouping via NetworkX
│   ├── graph_analysis.py                  # GraphML export and Matplotlib rendering
│   ├── risk_scoring.py                    # Rule-based priority scoring and evidence explanation
│   └── ml/
│       ├── features.py                    # Feature extraction and normalization
│       ├── prepare_dataset.py             # Loader & label mapper for Cresci datasets
│       ├── train_models.py                # LR, DT, RF training & evaluation pipeline
│       ├── evaluate_models.py             # Metrics computation
│       ├── explain_predictions.py         # SHAP & LIME explainer with graceful missing-package fallback
│       └── predict_account.py             # Model bundle loading & account inference
├── models/
│   ├── selected_bot_classifier.joblib     # Pretrained Random Forest model bundle (8.9 MB) - PRESERVED
│   ├── selected_bot_classifier_backup.joblib # Backup model artifact
│   ├── selected_bot_classifier_metadata.json # Provenance & validation metrics
│   └── background_sample.csv              # SHAP background distribution sample
├── data/
│   ├── raw/                               # Hashtags .txt, hashtags2.txt (Cresci data not committed per policy)
│   └── processed/                         # Cleaned posts, sample posts, cleaned keywords
├── results/
│   ├── graphs/                            # coordination_graph.graphml, coordination_graph.png
│   └── reports/                           # Output CSVs and evaluation JSONs
└── tests/
    ├── test_nextrace.py                   # 4 unit tests for preprocessing, coordination, risk scoring
    ├── test_ml_pipeline.py                # 11 unit tests for ML preparation, features, split, inference
    └── test_backup.py                     # 12 unit tests for pipeline output backup utility
```

---

## 3. Baseline Commands Executed & Real Results

| Command | Status | Result / Details |
|---|---|---|
| `git status` | Pass | Branch `main`, working tree clean, up to date with origin/main. |
| `python -m pytest tests/ -v` | **PASS (27/27)** | 27 tests passed in 25.46 seconds. 2 warnings (Pandas concat deprecation & datetime format warning). |
| `python run_ml_pipeline.py --status` | **PASS** | Successfully verified `selected_bot_classifier.joblib` (8.9 MB), metadata, and report files. |
| `python validate_model.py` | **PASS** | Model loaded cleanly as `dict` bundle containing pipeline, feature names (21 features), and validation metrics (Test F1 = 0.9889, ROC-AUC = 0.9985). |
| `python -m py_compile dashboard/app.py` | **PASS** | Dashboard compiles cleanly without syntax errors. |
| `streamlit run dashboard/app.py --server.headless true` | **PASS** | Server started on port 8501, health check returned HTTP 200 OK. |
| `python run_pipeline.py --backup-existing` | **FAIL** | Steps 1-3 succeeded with backup created. Step 4 failed with `ModuleNotFoundError: No module named 'sentence_transformers'`. |

---

## 4. Module Status Matrix

| Module | Status | Notes |
|---|---|---|
| `src.config` | Functional | Centralized thresholds; needs provider settings and schema configs. |
| `src.data_preprocessing` | Functional | Cleans post duplicates and timestamps; lacks strict schema validation and account-level normalization. |
| `src.create_sample_data` | Functional | Generates 10 synthetic posts; isolated to synthetic demo mode. |
| `src.keyword_loader` | Functional | Processes text files from `data/raw/`. |
| `src.semantic_similarity` | **Incomplete / Fragile** | Direct hard dependency on `sentence-transformers`; fails if package is missing or offline. Requires robust embedding/TF-IDF fallback. |
| `src.coordination_detection` | Functional | Implements pairwise semantic, temporal, and URL signals. |
| `src.account_analysis` | Partial | Extracts basic post counts and time gaps, but does not format features for the trained ML model. |
| `src.campaign_analysis` | Functional | Connected components via NetworkX. |
| `src.graph_analysis` | Functional | GraphML export and static PNG rendering. |
| `src.risk_scoring` | Functional | Explainable risk labels (`High`, `Medium`, `Low`) with transparent evidence string. |
| `src.ml.predict_account` | Functional | Predicts 21 account features with Random Forest bundle; currently uncalled by Phase 1 pipeline. |
| `src.ml.prepare_dataset` | Functional | Tested with fixtures; raw Cresci data excluded from git. |
| `src.ml.explain_predictions`| Functional | Gracefully handles missing `shap` and `lime` packages. |
| Ingestion Layer | **Missing** | No provider abstraction, no CSV/JSON ingest boundary, no schema validation for external data. |

---

## 5. Input / Output Data Flow & Schemas

### Current Phase 1 Data Flow
```
create_sample_data.py -> sample_social_media_posts.csv
        ↓
data_preprocessing.py -> cleaned_social_media_posts.csv
        ↓
semantic_similarity.py -> semantic_similarity_results.csv (post_1, account_1, post_2, account_2, text_similarity, similarity_type)
        ↓
coordination_detection.py -> coordination_results.csv (account_1, account_2, post_1, post_2, semantic_similarity, time_difference_minutes, shared_url, coordination_score, evidence_signals)
        ↓
    ┌───┴───────────────────────┬────────────────────────┐
    ↓                           ↓                        ↓
account_analysis.py     campaign_analysis.py     graph_analysis.py
(account_behavior.csv)  (campaign_results.csv)   (graph.png & graphml)
                                                         ↓
                                                 risk_scoring.py
                                                 (explainable_risk_results.csv)
```

### Current Schema Limitations
1. Posts only support 5 fields: `post_id`, `account_id`, `text`, `timestamp`, `url`. Missing metadata: platform/source provenance, public engagement metrics (likes, retweets), author profile data.
2. The ML classifier expects 21 features (including `followers_count`, `friends_count`, `statuses_count`, `favourites_count`, `listed_count`, `default_profile`, `geo_enabled`, `verified`, `protected`, `description_length`, `tweet_data_available`). Because the input post schema does not accept account profile metadata, the ML model is currently unreachable from post ingestion!

---

## 6. Dashboard State Analysis

The Streamlit dashboard (`dashboard/app.py`) has 5 tabs:
1. **Posts Tab:** Displays ingested posts, keyword frequency, and filters. Identifies whether dataset is the 10-post demo fixture via `_is_demo_dataset`.
2. **Coordination Detection Tab:** Displays flagged pairs, scatter plot of time difference vs similarity, and score distribution. Reads `coordination_results.csv`.
3. **Risk Scores Tab:** Displays priority breakdown (High, Medium, Low), evidence table, and summary metrics from `explainable_risk_results.csv`.
4. **Campaign Analysis Tab:** Renders interactive Plotly network graph and candidate campaign table from `campaign_analysis_results.csv`.
5. **Bot Classifier Tab:** Displays model metadata and evaluation comparison. Contains an interactive manual form to score an account by entering 21 feature values, but cannot auto-score accounts from the active ingested dataset.

---

## 7. Model Artifact Integrity & Evaluation

- `models/selected_bot_classifier.joblib` is intact (hash preserved, 8.9 MB).
- Contains fitted `Pipeline(imputer=SimpleImputer, classifier=RandomForestClassifier)`.
- Evaluation metrics recorded in metadata:
  - Test Set: 2,448 samples (Stratified hold-out from Cresci-2015 & Cresci-2017)
  - Accuracy: 0.9853
  - Precision: 0.9963
  - Recall: 0.9816
  - F1 Score: 0.9889
  - ROC-AUC: 0.9985
- **Generalization Caveat:** While random stratified split yields near-perfect metrics, cross-group generalization (tested in `check_group_generalization.py`) shows lower generalization when tested on entirely unseen bot families. The model is an indicator of resemblance to known historical datasets, not definitive proof of bot status.

---

## 8. Prioritized Implementation Summary

All prioritized engineering stages have been completed:
1. **Unified Data Contracts (`src/contracts.py`):** Canonical schemas for posts, accounts, results, campaigns, news articles, fact-check reviews, domain telemetry, and OSINT bundles.
2. **Ingestion & Social Providers (`src/ingestion/`):** File provider (CSV/JSON), synthetic provider, generic API adapter, Twitter/X API v2 Recent Search provider with rate-limit backoff, and Reddit Data API provider.
3. **OSINT Intelligence Providers (`src/osint/`):** GDELT DOC 2.0 API client, Google Fact Check Tools API client, SSRF-defended RDAP domain investigator, Vision OCR text extractor with disclaimers, and OSINTEnricher.
4. **NLP & Linguistic Robustness (`src/semantic_similarity.py`):** Dual-engine SentenceTransformer with TF-IDF cosine fallback; validated on exact duplicates, astroturfed paraphrases, structural paraphrases, empty strings, and Unicode text.
5. **Dashboard Completion (`dashboard/app.py`):** 5-tab interface enhanced with OSINT narrative panels, clickable fact-checks and news articles, RDAP domain registration tables, and real-time Live API & OSINT Services drawer.
6. **Automated Testing (`tests/`):** 67 unit and integration tests across 7 test files, achieving 100% pass rate.

---

## 9. Final Verification & Test Release Status

- **Automated Test Results:** **67 passed in 7.56s** (0 failures, 0 regressions).
  - `tests/test_backup.py`: 12 passed
  - `tests/test_ingestion_and_contracts.py`: 15 passed
  - `tests/test_ml_pipeline.py`: 11 passed
  - `tests/test_nextrace.py`: 4 passed
  - `tests/test_nlp_and_similarity.py`: 6 passed
  - `tests/test_osint_enrichment.py`: 12 passed
  - `tests/test_social_apis.py`: 7 passed
- **Model Integrity:** `models/selected_bot_classifier.joblib` (8.9 MB) preserved byte-for-byte.
- **Pipeline Execution:** `run_pipeline.py` executes all 9 stages + OSINT Narrative & Evidence Enrichment end-to-end.
- **Deliverables:**
  - `NexTrace_Engineering_Report.docx` and `docs/NexTrace_Engineering_Continuation_Report.docx`
  - `.env.example` with credential placeholders
  - `README.md` complete reference

