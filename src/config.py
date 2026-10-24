"""NexTrace — Central configuration module.

All shared thresholds, directory paths, and runtime settings are defined
here. Individual analysis modules import from this file instead of
hardcoding values locally. This makes it straightforward to adjust
detection sensitivity without hunting through multiple files.

Usage
-----
    from src.config import SEMANTIC_THRESHOLD, TIME_WINDOW_MINUTES

Do NOT modify model paths or label mappings here — those are owned by
src/ml/prepare_dataset.py and src/ml/train_models.py respectively.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolved relative to this file so the config works
# regardless of the working directory the user launches scripts from).
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Phase 1 — Coordination detection thresholds
# ---------------------------------------------------------------------------

# Minimum cosine-similarity score (0–1) for two posts to be considered
# semantically similar.  Used by semantic_similarity.py and
# coordination_detection.py.
SEMANTIC_THRESHOLD: float = 0.70

# Maximum number of minutes between two posts for them to be considered
# "close in time" when scoring coordination.
TIME_WINDOW_MINUTES: int = 10

# Weights used when computing the composite coordination score:
#   score = W_SEMANTIC * similarity + W_TIME * close_in_time + W_URL * shared_url
COORDINATION_WEIGHT_SEMANTIC: float = 0.50
COORDINATION_WEIGHT_TIME: float = 0.25
COORDINATION_WEIGHT_URL: float = 0.25

# Minimum number of supporting signals (similarity, timing, shared URL)
# required before a pair is flagged as coordinated.  The similar-text
# signal must always be present; at least one additional signal is needed.
MIN_SIGNAL_COUNT: int = 2

# Risk-priority thresholds applied by risk_scoring.py.
RISK_HIGH_THRESHOLD: float = 0.80
RISK_MEDIUM_THRESHOLD: float = 0.60


# ---------------------------------------------------------------------------
# Sentence-Transformer model used for semantic similarity
# ---------------------------------------------------------------------------

# Name of the pretrained model loaded from Hugging Face / local cache.
SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

RESULTS_DIR: Path = BASE_DIR / "results"
REPORTS_DIR: Path = RESULTS_DIR / "reports"
GRAPHS_DIR: Path = RESULTS_DIR / "graphs"
ALERTS_DIR: Path = RESULTS_DIR / "alerts"

MODELS_DIR: Path = BASE_DIR / "models"


# ---------------------------------------------------------------------------
# Phase 1 — Data file paths
# ---------------------------------------------------------------------------

SAMPLE_POSTS_FILE: Path = DATA_PROCESSED_DIR / "sample_social_media_posts.csv"
CLEANED_POSTS_FILE: Path = DATA_PROCESSED_DIR / "cleaned_social_media_posts.csv"
CLEANED_KEYWORDS_FILE: Path = DATA_PROCESSED_DIR / "cleaned_keywords.csv"

SEMANTIC_RESULTS_FILE: Path = REPORTS_DIR / "semantic_similarity_results.csv"
COORDINATION_RESULTS_FILE: Path = REPORTS_DIR / "coordination_results.csv"
ACCOUNT_BEHAVIOR_FILE: Path = REPORTS_DIR / "account_behavior_results.csv"
CAMPAIGN_RESULTS_FILE: Path = REPORTS_DIR / "campaign_analysis_results.csv"
RISK_RESULTS_FILE: Path = REPORTS_DIR / "explainable_risk_results.csv"

COORDINATION_GRAPH_PNG: Path = GRAPHS_DIR / "coordination_graph.png"
COORDINATION_GRAPH_GRAPHML: Path = GRAPHS_DIR / "coordination_graph.graphml"


# ---------------------------------------------------------------------------
# Phase 2 — ML model file paths
# (These paths are also referenced in src/ml/train_models.py and
#  src/ml/predict_account.py.  They are listed here for documentation;
#  the ML modules retain their own path constants to avoid circular imports.)
# ---------------------------------------------------------------------------

SELECTED_MODEL_FILE: Path = MODELS_DIR / "selected_bot_classifier.joblib"
MODEL_METADATA_FILE: Path = MODELS_DIR / "selected_bot_classifier_metadata.json"
BACKGROUND_SAMPLE_FILE: Path = MODELS_DIR / "background_sample.csv"

ML_DATASET_FILE: Path = DATA_PROCESSED_DIR / "ml_dataset.csv"
ML_EVALUATION_SUMMARY_FILE: Path = REPORTS_DIR / "ml_evaluation_summary.json"
ML_MODEL_COMPARISON_FILE: Path = REPORTS_DIR / "ml_model_comparison.csv"
ML_EXPLANATIONS_FILE: Path = REPORTS_DIR / "ml_explanations.json"


# ---------------------------------------------------------------------------
# Phase 3 & 4 — OSINT & Social Media API configurations
# ---------------------------------------------------------------------------

GDELT_BASE_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_FACT_CHECK_BASE_URL: str = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
RDAP_BASE_URL: str = "https://rdap.org/domain"
X_API_RECENT_SEARCH_URL: str = "https://api.twitter.com/2/tweets/search/recent"
REDDIT_SEARCH_URL: str = "https://www.reddit.com/search.json"

OSINT_ENRICHMENT_FILE: Path = REPORTS_DIR / "osint_enrichment_results.json"

DEFAULT_REQUEST_TIMEOUT: int = 10
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 1.0

