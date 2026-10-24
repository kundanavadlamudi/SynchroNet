# NexTrace

**NexTrace** is a Social Media Coordinated Disinformation Detection System.
It identifies potentially coordinated inauthentic behaviour by combining
semantic text similarity, temporal analysis, and machine learning–based
bot classification.

> ⚠️ **Important disclaimer**  
> NexTrace generates investigation leads, not proof of disinformation or
> malicious intent. Every flagged account pair requires human review before
> any conclusions are drawn.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Technology stack](#2-technology-stack)
3. [Directory layout](#3-directory-layout)
4. [Prerequisites](#4-prerequisites)
5. [Environment setup](#5-environment-setup)
6. [Running the Phase 1 pipeline](#6-running-the-phase-1-pipeline)
7. [Running the Phase 2 ML pipeline](#7-running-the-phase-2-ml-pipeline)
8. [Launching the dashboard](#8-launching-the-dashboard)
9. [Trained model information](#9-trained-model-information)
10. [Configuration](#10-configuration)
11. [Running tests](#11-running-tests)
12. [Output files](#12-output-files)
13. [Alerts directory](#13-alerts-directory)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project overview

NexTrace operates in two independent pipelines:

| Pipeline | Purpose |
|---|---|
| **Phase 1 — Coordination detection** | Detects synchronized posting behaviour between account pairs using semantic similarity, time-window analysis, and shared-URL signals |
| **Phase 2 — Bot classification (ML)** | Classifies individual accounts as Genuine or Bot using a trained Random Forest classifier |

The investigation dashboard (Streamlit) reads the output files produced
by these pipelines and presents them as interactive tables, charts, and
network visualisations.

---

## 2. Technology stack

| Component | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn |
| Graph analysis | NetworkX |
| Semantic similarity | sentence-transformers (`all-MiniLM-L6-v2`) |
| Explainability | SHAP, LIME |
| Visualisation | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Serialisation | joblib |

---

## 3. Directory layout

```
NexTrace/
├── src/
│   ├── config.py                   # Central thresholds and path constants
│   ├── create_sample_data.py       # Generates 10 synthetic posts
│   ├── data_preprocessing.py       # Cleans and validates post data
│   ├── keyword_loader.py           # Processes hashtag/keyword files
│   ├── semantic_similarity.py      # Text embedding + cosine similarity
│   ├── coordination_detection.py   # Multi-signal coordination scoring
│   ├── account_analysis.py         # Per-account behavioural features
│   ├── campaign_analysis.py        # Connected-component campaign grouping
│   ├── graph_analysis.py           # NetworkX graph + PNG output
│   ├── risk_scoring.py             # Investigation-priority scoring
│   └── ml/
│       ├── features.py             # Account-level feature engineering
│       ├── prepare_dataset.py      # Cresci-2015/2017 dataset loader
│       ├── train_models.py         # LR / DT / RF training + selection
│       ├── evaluate_models.py      # Metrics, confusion matrices
│       ├── explain_predictions.py  # SHAP + LIME explanations
│       └── predict_account.py      # Inference on new accounts
├── dashboard/
│   └── app.py                      # Streamlit investigation dashboard
├── data/
│   ├── raw/                        # Keyword/hashtag files; Cresci data (not committed)
│   └── processed/                  # Cleaned CSVs; ML dataset
├── models/
│   ├── selected_bot_classifier.joblib        # Trained Random Forest (~9.4 MB)
│   ├── selected_bot_classifier_backup.joblib # Backup model
│   ├── selected_bot_classifier_metadata.json # Model provenance and metrics
│   └── background_sample.csv                 # SHAP reference dataset
├── results/
│   ├── graphs/                     # Coordination graph images
│   ├── reports/                    # CSV and JSON analysis outputs
│   └── alerts/                     # Reserved for future alerting module (currently empty)
├── tests/
│   ├── test_nextrace.py            # Phase 1 unit tests
│   └── test_ml_pipeline.py        # Phase 2 ML unit tests
├── run_pipeline.py                 # Phase 1 orchestrator (run this first)
├── run_ml_pipeline.py              # Phase 2 status checker / retrain guard
├── requirements.txt                # Pinned Python dependencies
└── README.md
```

---

## 4. Prerequisites

- **Python 3.11.x** — the trained model was produced with Python 3.11.9
  and scikit-learn 1.9.1. Using a different Python or scikit-learn version
  may produce joblib deserialization warnings.
- **pip** 23+ (bundled with Python 3.11)
- Approximately **2 GB** of disk space for the model, Torch, and
  sentence-transformers cache.

---

## 5. Environment setup

### Which virtual environment to use

Two virtual environments are present in the project root:

| Directory | Python version | Purpose |
|---|---|---|
| `.venv` | **3.11.9** | ✅ **Primary environment** — matches the environment used to train the model |
| `.venv-1` | 3.14.3 | Experimental — created more recently; may have package differences |

**Always activate `.venv` for normal use.**

### Activating the environment

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### Installing dependencies into a fresh environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
```

---

## 6. Running the Phase 1 pipeline

The Phase 1 pipeline produces all coordination detection outputs.
Run it from the project root with the environment activated:

```bash
# Full pipeline (generates sample data + all analysis steps)
python run_pipeline.py

# Skip sample-data generation if you already have post data
python run_pipeline.py --skip-sample-data

# Back up existing outputs before the pipeline overwrites them
python run_pipeline.py --backup-existing

# Combine both flags
python run_pipeline.py --skip-sample-data --backup-existing

# Show help
python run_pipeline.py --help
```

### Backing up existing results (`--backup-existing`)

By default, every pipeline run unconditionally overwrites the ten Phase 1
output files. Passing `--backup-existing` copies any files that already
exist into a timestamped directory **before** they are overwritten:

```
results/backups/<YYYYMMDD_HHMMSS>/
    data/processed/sample_social_media_posts.csv
    data/processed/cleaned_social_media_posts.csv
    data/processed/cleaned_keywords.csv
    results/reports/semantic_similarity_results.csv
    results/reports/coordination_results.csv
    results/reports/account_behavior_results.csv
    results/reports/campaign_analysis_results.csv
    results/graphs/coordination_graph.png
    results/graphs/coordination_graph.graphml
    results/reports/explainable_risk_results.csv
```

Each backup is in its own timestamped subdirectory, so earlier backups
are never overwritten. The trained ML model (`models/`) and ML report
files are **never** included in the backup.

### Pipeline step order

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `src/create_sample_data.py` | — | `data/processed/sample_social_media_posts.csv` |
| 2 | `src/data_preprocessing.py` | sample_posts | `data/processed/cleaned_social_media_posts.csv` |
| 3 | `src/keyword_loader.py` | `data/raw/Hashtags .txt`, `hashtags2.txt` | `data/processed/cleaned_keywords.csv` |
| 4 | `src/semantic_similarity.py` | cleaned_posts | `results/reports/semantic_similarity_results.csv` |
| 5 | `src/coordination_detection.py` | cleaned_posts + semantic_results | `results/reports/coordination_results.csv` |
| 6 | `src/account_analysis.py` | cleaned_posts | `results/reports/account_behavior_results.csv` |
| 7 | `src/campaign_analysis.py` | coordination_results | `results/reports/campaign_analysis_results.csv` |
| 8 | `src/graph_analysis.py` | coordination_results | `results/graphs/coordination_graph.png` + `.graphml` |
| 9 | `src/risk_scoring.py` | coordination_results | `results/reports/explainable_risk_results.csv` |

> ⚠️ Step 4 (semantic similarity) downloads the `all-MiniLM-L6-v2`
> sentence-transformer model on first run (~90 MB). Subsequent runs
> use the local cache.

---

## 7. Running the Phase 2 ML pipeline

### Check model status (safe — no retraining)

```bash
python run_ml_pipeline.py
# or explicitly:
python run_ml_pipeline.py --status
```

This prints the model name, metrics, dataset provenance, and file
locations without changing anything.

### Retrain the model (only if datasets have changed)

> ⚠️ The trained model achieves **F1 = 0.989 / ROC-AUC = 0.998** on a
> held-out test set. **Do not retrain unless you have new Cresci data.**

```bash
python run_ml_pipeline.py --retrain
# You will be prompted to type YES to confirm.
```

Retraining requires the Cresci-2015 and/or Cresci-2017 CSV files to be
present under `data/raw/` (see `src/ml/prepare_dataset.py` for the
expected directory layout).

---

## 8. Launching the dashboard

From the project root, with the virtual environment active:

```bash
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. It reads the Phase 1 CSV
reports, the GraphML coordination network, and the Phase 2 model metadata;
run the pipeline at least once before opening it.

The interface has five tabs:

| Tab | Contents |
|---|---|
| Posts | Searchable post table, activity timeline, and per-account counts |
| Coordination Detection | Pair filters, score distribution, timing/similarity scatter, and the coordination network |
| Risk Scores | Priority filters, risk matrix, evidence cards, and explainable risk scores |
| Campaign Analysis | Campaign group overview and member-account detail |
| Bot Classifier | Model evaluation, pre-computed SHAP/LIME explanations, and an interactive account classifier |

The bundled dataset (`P001`–`P010` from accounts `A101`–`A110`) is a
**synthetic demonstration dataset**, labelled by a compact **DEMO DATA**
notice at the top of the dashboard. Scores and groupings illustrate how
NexTrace works; they are not real-world findings. The bot classifier is
evaluated separately on the Cresci-2015/2017 research datasets. All output
is an investigation lead that requires human review.

The two sidebar thresholds (coordination score and semantic similarity) are
**display-only filters**. They narrow the pair rows shown in the
Coordination Detection and Risk Scores tabs from the precomputed reports;
they do not re-run detection, rewrite reports, or change any score.
Selecting a semantic threshold below the pipeline value (`0.70`) cannot
reveal additional pairs, because those pairs were never written to the
report.

---

## 9. Trained model information

| Property | Value |
|---|---|
| Algorithm | Random Forest (200 estimators) |
| Training data | Cresci-2015 + Cresci-2017 (16,318 labeled accounts) |
| Split strategy | Stratified by label |
| Features | 21 account-level features (posting behaviour + profile stats) |
| Test accuracy | 0.9853 |
| Test precision | 0.9963 |
| Test recall | 0.9816 |
| Test F1 | 0.9889 |
| Test ROC-AUC | 0.9985 |
| Model file | `models/selected_bot_classifier.joblib` (~9.4 MB) |
| Metadata | `models/selected_bot_classifier_metadata.json` |
| Created | 2026-10-23 |

**The model classifies individual accounts as Genuine (0) or Bot (1).**
This is distinct from Phase 1 coordination detection, which measures
synchronized behaviour between account pairs.

---

## 10. Configuration

All shared thresholds and file paths are defined in
**`src/config.py`**. Edit this file to adjust detection sensitivity:

| Setting | Default | Description |
|---|---|---|
| `SEMANTIC_THRESHOLD` | `0.70` | Minimum cosine similarity for text to be considered "similar" |
| `TIME_WINDOW_MINUTES` | `10` | Maximum minutes between posts to be "close in time" |
| `RISK_HIGH_THRESHOLD` | `0.80` | Coordination score above which a pair is "High" priority |
| `RISK_MEDIUM_THRESHOLD` | `0.60` | Coordination score above which a pair is "Medium" priority |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | NLP model used for embeddings |

---

## 11. Running tests

```bash
# All tests
python -m pytest tests/ -v

# Phase 1 only
python -m pytest tests/test_nextrace.py -v

# Phase 2 ML only
python -m pytest tests/test_ml_pipeline.py -v
```

The test suite uses synthetic in-memory data and does not require the
trained model to be present (tests gracefully skip when the model is
absent).

---

## 12. Output files

| File | Description |
|---|---|
| `data/processed/cleaned_social_media_posts.csv` | Validated and normalised post data |
| `data/processed/cleaned_keywords.csv` | Deduplicated keyword/hashtag list |
| `results/reports/semantic_similarity_results.csv` | Post pairs with cosine similarity ≥ 0.70 |
| `results/reports/coordination_results.csv` | Flagged account pairs with coordination scores |
| `results/reports/account_behavior_results.csv` | Per-account posting behaviour summary |
| `results/reports/campaign_analysis_results.csv` | Connected-component campaign groups |
| `results/reports/explainable_risk_results.csv` | Risk-scored pairs with evidence text |
| `results/graphs/coordination_graph.png` | Coordination network visualisation |
| `results/graphs/coordination_graph.graphml` | Machine-readable graph file |
| `results/reports/ml_evaluation_summary.json` | Full model training/evaluation metrics |
| `results/reports/ml_model_comparison.csv` | LR / DT / RF comparison table |
| `results/reports/ml_explanations.json` | SHAP + LIME explanations for sample accounts |

---

## 13. Alerts directory

The `results/alerts/` directory is **currently empty and reserved for
future use**. No alerting module has been implemented yet. This directory
exists as a placeholder for a planned feature that would generate
notification records (e.g., JSON alerts or email triggers) when
high-priority coordination pairs are detected.

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run all scripts from the **project root** directory:

```bash
# Correct (from NexTrace/)
python run_pipeline.py

# Incorrect (from NexTrace/src/)
python run_pipeline.py   # will fail
```

### Streamlit reports "Sample dataset not found"

Run the Phase 1 pipeline first:

```bash
python run_pipeline.py
```

### SHAP or LIME errors

Both packages are optional. If they fail, the dashboard and pipeline
continue without explanations. Verify they are installed:

```bash
python -c "import shap; import lime; print('OK')"
```

### Model deserialization warning

If joblib warns about version mismatch when loading the trained model,
ensure you are using the `.venv` environment (Python 3.11.9,
scikit-learn 1.9.1) rather than `.venv-1` (Python 3.14.3).

### Sentence-transformer download is slow

The `all-MiniLM-L6-v2` model is downloaded from Hugging Face on first
use. Set the `TRANSFORMERS_CACHE` environment variable to a fast local
disk if needed.

### Pipeline step fails with "Cleaned dataset not found"

Steps 4–9 depend on the output of step 2. Always run the full pipeline
from step 1, or pass `--skip-sample-data` only after at least one full
run has completed.