"""NexTrace — coordinated disinformation detection dashboard.

Five-tab Streamlit interface reading the Phase 1 pipeline reports and the
Phase 2 trained bot classifier. Run from the project root:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import html
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap — ensure ``src.*`` imports resolve from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ACCOUNT_BEHAVIOR_FILE,
    CAMPAIGN_RESULTS_FILE,
    CLEANED_POSTS_FILE,
    COORDINATION_GRAPH_GRAPHML,
    COORDINATION_GRAPH_PNG,
    COORDINATION_RESULTS_FILE,
    COORDINATION_WEIGHT_SEMANTIC,
    COORDINATION_WEIGHT_TIME,
    COORDINATION_WEIGHT_URL,
    MIN_SIGNAL_COUNT,
    ML_EVALUATION_SUMMARY_FILE,
    ML_EXPLANATIONS_FILE,
    ML_MODEL_COMPARISON_FILE,
    MODEL_METADATA_FILE,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    RISK_RESULTS_FILE,
    SELECTED_MODEL_FILE,
    SEMANTIC_THRESHOLD,
    TIME_WINDOW_MINUTES,
    OSINT_ENRICHMENT_FILE,
    MANUAL_OSINT_FILE,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NexTrace | Disinformation Intelligence Dashboard",
    page_icon=":material/radar:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
CYAN = "#22d3ee"
CYAN_DEEP = "#0891b2"
VIOLET = "#a78bfa"
VIOLET_DEEP = "#7c3aed"
INK = "#e2e8f0"
MUTED = "#94a3b8"
FAINT = "#64748b"
PANEL = "rgba(15, 23, 42, 0.72)"

PRIORITY_COLORS = {
    "High": "#f87171",
    "Medium": "#fbbf24",
    "Low": "#34d399",
}
PRIORITY_ORDER = ["High", "Medium", "Low"]
SCORE_SCALE = [[0.0, "#0e7490"], [0.55, VIOLET_DEEP], [1.0, "#e879f9"]]

DEMO_POST_IDS = {f"P{index:03d}" for index in range(1, 11)}

# ---------------------------------------------------------------------------
# Bot-classifier feature metadata
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "total_posts": "Total posts",
    "posts_with_urls": "Posts with URLs",
    "unique_texts": "Unique texts",
    "duplicate_text_posts": "Duplicate-text posts",
    "average_text_length": "Average text length",
    "average_word_count": "Average word count",
    "average_time_between_posts_minutes": "Avg. time between posts (min)",
    "url_post_ratio": "URL post ratio",
    "duplicate_text_ratio": "Duplicate-text ratio",
    "statuses_count": "Statuses count",
    "followers_count": "Followers",
    "friends_count": "Following (friends)",
    "favourites_count": "Favourites / likes",
    "listed_count": "Listed count",
    "default_profile": "Default profile image",
    "geo_enabled": "Geo enabled",
    "verified": "Verified",
    "protected": "Protected account",
    "description_length": "Bio description length",
    "friends_followers_ratio": "Friends / followers ratio",
    "tweet_data_available": "Tweet data available",
}

FEATURE_HELP = {
    "total_posts": "Total number of posts analysed for the account.",
    "posts_with_urls": "Number of posts containing at least one URL.",
    "unique_texts": "Count of distinct post texts.",
    "duplicate_text_posts": "Count of posts whose text repeats an earlier post.",
    "average_text_length": "Mean character count per post.",
    "average_word_count": "Mean word count per post.",
    "average_time_between_posts_minutes": "Mean minutes between consecutive posts.",
    "url_post_ratio": "Fraction of posts containing a URL (0–1).",
    "duplicate_text_ratio": "Fraction of posts with duplicate text (0–1).",
    "statuses_count": "Total statuses/tweets on the profile.",
    "followers_count": "Number of followers.",
    "friends_count": "Number of accounts followed.",
    "favourites_count": "Number of liked posts or favourites.",
    "listed_count": "Number of public lists containing the account.",
    "default_profile": "1 = default profile image, 0 = custom image.",
    "geo_enabled": "1 = location sharing enabled.",
    "verified": "1 = platform-verified account.",
    "protected": "1 = protected/private account.",
    "description_length": "Character length of the profile bio.",
    "friends_followers_ratio": "friends_count ÷ max(followers_count, 1).",
    "tweet_data_available": "1 = post-level data available, 0 = profile-only record.",
}

BOOLEAN_FEATURES = {
    "default_profile",
    "geo_enabled",
    "verified",
    "protected",
    "tweet_data_available",
}
RATIO_FEATURES = {"url_post_ratio", "duplicate_text_ratio"}
COUNT_FEATURES = {
    "total_posts",
    "posts_with_urls",
    "unique_texts",
    "duplicate_text_posts",
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
    "description_length",
}

BOT_FEATURE_GROUPS = {
    "Posting behaviour": [
        "total_posts",
        "posts_with_urls",
        "unique_texts",
        "duplicate_text_posts",
        "average_text_length",
        "average_word_count",
        "average_time_between_posts_minutes",
        "url_post_ratio",
        "duplicate_text_ratio",
        "tweet_data_available",
    ],
    "Profile and account signals": [
        "statuses_count",
        "followers_count",
        "friends_count",
        "favourites_count",
        "listed_count",
        "default_profile",
        "geo_enabled",
        "verified",
        "protected",
        "description_length",
        "friends_followers_ratio",
    ],
}

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1100px 520px at 12% -8%, rgba(34, 211, 238, 0.09), transparent 60%),
            radial-gradient(900px 520px at 92% -2%, rgba(139, 92, 246, 0.11), transparent 58%),
            #0a0f1e;
    }
    [data-testid="stHeader"] {
        background: rgba(10, 15, 30, 0.88);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    }
    [data-testid="stAppDeployButton"] { display: none; }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1440px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1426 0%, #0a0f1e 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, 0.14); }

    /* ---- Hero ---------------------------------------------------------- */
    .nx-hero {
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 28px;
        padding: 30px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.82));
        border: 1px solid rgba(34, 211, 238, 0.18);
        box-shadow: 0 22px 55px rgba(2, 6, 23, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        overflow: hidden;
    }
    .nx-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #22d3ee, #8b5cf6 55%, transparent);
    }
    .nx-hero-copy { flex: 1 1 auto; min-width: 0; }
    .nx-eyebrow {
        color: #67e8f9;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .nx-title {
        color: #f1f5f9;
        font-size: clamp(2.05rem, 3.4vw, 2.4rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.05;
    }
    .nx-title span {
        background: linear-gradient(90deg, #22d3ee, #a78bfa);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .nx-subtitle {
        color: #94a3b8;
        font-size: 0.93rem;
        line-height: 1.6;
        max-width: 780px;
        margin-top: 11px;
    }
    .nx-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
    .nx-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
        background: rgba(15, 23, 42, 0.7);
    }
    .nx-pill-cyan { color: #a5f3fc; border-color: rgba(34, 211, 238, 0.35); background: rgba(8, 145, 178, 0.12); }
    .nx-pill-violet { color: #ddd6fe; border-color: rgba(167, 139, 250, 0.38); background: rgba(124, 58, 237, 0.13); }
    .nx-mark {
        flex: 0 0 auto;
        width: 78px;
        height: 78px;
        border-radius: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #ecfeff;
        background:
            radial-gradient(circle at 30% 22%, rgba(34, 211, 238, 0.45), transparent 58%),
            linear-gradient(145deg, #0e7490, #6d28d9);
        border: 1px solid rgba(167, 139, 250, 0.5);
        box-shadow: 0 0 46px rgba(34, 211, 238, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }
    @media (max-width: 720px) {
        .nx-hero { padding: 24px; gap: 18px; align-items: flex-start; }
        .nx-hero-copy { flex: 1 1 auto; }
        .nx-mark { width: 56px; height: 56px; border-radius: 16px; font-size: 1.45rem; }
        .nx-eyebrow { font-size: 0.6rem; letter-spacing: 0.14em; }
        .nx-title { font-size: 1.95rem; }
        .nx-subtitle { font-size: 0.88rem; }
    }

    /* ---- Compact source notice ---------------------------------------- */
    .nx-source-note {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px 10px;
        margin: 10px 0 4px;
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #94a3b8;
        font-size: 0.76rem;
        line-height: 1.45;
    }
    .nx-source-tag {
        display: inline-flex;
        align-items: center;
        flex: 0 0 auto;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.59rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.28);
        white-space: nowrap;
    }
    .nx-source-tag.is-pipeline {
        color: #a5f3fc;
        background: rgba(8, 145, 178, 0.12);
        border-color: rgba(34, 211, 238, 0.30);
    }
    .nx-source-tag.is-muted {
        color: #cbd5e1;
        background: rgba(100, 116, 139, 0.14);
        border-color: rgba(148, 163, 184, 0.24);
    }
    .nx-source-icon { display: inline-flex; flex: 0 0 auto; color: #64748b; }
    .nx-source-text { flex: 1 1 320px; min-width: 0; }
    .nx-source-text strong { color: #e2e8f0; font-weight: 650; }

    /* ---- KPI cards ----------------------------------------------------- */
    .nx-kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin: 2px 0 4px;
    }
    .nx-kpi {
        display: flex;
        flex-direction: column;
        gap: 9px;
        min-width: 0;
        min-height: 104px;
        padding: 13px 15px 12px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.85));
        position: relative;
        overflow: hidden;
    }
    .nx-kpi::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #22d3ee, #8b5cf6);
        opacity: 0.85;
    }
    .nx-kpi-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
    .nx-kpi-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.35;
        overflow-wrap: break-word;
    }
    .nx-kpi-help {
        flex: 0 0 auto;
        width: 16px;
        height: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #94a3b8;
        font-size: 0.6rem;
        font-weight: 700;
        line-height: 1;
        cursor: help;
        user-select: none;
    }
    .nx-kpi-help:hover { color: #e2e8f0; border-color: rgba(34, 211, 238, 0.55); }
    .nx-kpi-value {
        color: #f1f5f9;
        font-size: 1.65rem;
        font-weight: 750;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .nx-kpi-value.is-unavailable { color: #64748b; }
    .nx-kpi-foot { color: #64748b; font-size: 0.65rem; line-height: 1.35; margin-top: auto; }
    .nx-kpi.is-unavailable { border-color: rgba(245, 158, 11, 0.24); }
    .nx-kpi.is-unavailable .nx-kpi-foot { color: #f59e0b; }
    .nx-kpi-help:focus-visible {
        outline: 2px solid rgba(34, 211, 238, 0.65);
        outline-offset: 2px;
    }
    @media (max-width: 1180px) {
        .nx-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 780px) {
        .nx-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 520px) {
        .nx-kpi-grid { grid-template-columns: 1fr; }
    }

    /* ---- Metrics used inside tabs -------------------------------------- */
    div[data-testid="stMetric"], [data-testid="metric-container"] {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="stMetric"]::before, [data-testid="metric-container"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #22d3ee, #8b5cf6);
        opacity: 0.85;
    }
    div[data-testid="stMetricLabel"], [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0;
        text-transform: none;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] > div,
    [data-testid="metric-container"] [data-testid="stMetricLabel"] p,
    [data-testid="metric-container"] [data-testid="stMetricLabel"] > div {
        overflow: visible !important;
        white-space: normal !important;
        text-overflow: clip !important;
        line-height: 1.35 !important;
    }
    div[data-testid="stMetricValue"], [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.55rem !important;
        font-weight: 750 !important;
    }
    div[data-testid="stMetricValue"] > div { color: #f1f5f9; }

    /* ---- Tabs ---------------------------------------------------------- */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.65);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        overflow-x: auto;
        scrollbar-width: thin;
        scrollbar-color: rgba(100, 116, 139, 0.45) transparent;
    }
    [data-testid="stTabs"] [role="tab"] {
        height: auto;
        flex: 0 0 auto;
        padding: 9px 16px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.84rem;
        border-bottom: none;
        white-space: nowrap;
    }
    [data-testid="stTabs"] [role="tab"] p {
        color: inherit;
        font-size: inherit;
        font-weight: inherit;
        line-height: 1.35;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        color: #e2e8f0;
        background: rgba(30, 41, 59, 0.55);
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #ecfeff !important;
        background: linear-gradient(90deg, rgba(34, 211, 238, 0.16), rgba(139, 92, 246, 0.18));
        border-bottom: none !important;
    }
    [data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
    @media (max-width: 760px) {
        [data-testid="stTabs"] [role="tablist"] { gap: 0; padding: 3px; }
        [data-testid="stTabs"] [role="tab"] {
            padding: 7px 3px;
            font-size: 0.66rem;
            letter-spacing: -0.01em;
        }
    }

    /* ---- Section headings --------------------------------------------- */
    .nx-section {
        display: flex;
        align-items: stretch;
        gap: 12px;
        margin: 6px 0 4px;
    }
    .nx-section-bar {
        width: 4px;
        border-radius: 3px;
        background: linear-gradient(180deg, #22d3ee, #8b5cf6);
        flex: 0 0 auto;
    }
    .nx-section-title {
        color: #f1f5f9;
        font-size: 1.12rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.25;
    }
    .nx-section-sub {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 3px;
    }
    .nx-section-sub code, .nx-inline code {
        color: #a5f3fc;
        background: rgba(8, 145, 178, 0.12);
        border: 1px solid rgba(34, 211, 238, 0.16);
        border-radius: 5px;
        padding: 1px 5px;
        font-size: 0.74rem;
    }

    /* ---- Cards, expanders, alerts ------------------------------------- */
    div[data-testid="stExpander"] details {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] summary { color: #cbd5e1; font-weight: 600; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: rgba(148, 163, 184, 0.14) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stAlert"] { border-radius: 10px; }
    div[data-testid="stForm"] {
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 14px !important;
        background: rgba(15, 23, 42, 0.42);
    }

    /* ---- Buttons ------------------------------------------------------- */
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border-radius: 9px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #0891b2, #7c3aed);
        border: 1px solid rgba(167, 139, 250, 0.55);
        color: #ffffff;
        box-shadow: 0 8px 26px rgba(124, 58, 237, 0.25);
    }
    .stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
        border-color: rgba(34, 211, 238, 0.8);
        filter: brightness(1.08);
    }

    /* ---- Badges, chips, account pills --------------------------------- */
    .nx-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .nx-badge-high { background: rgba(220, 38, 38, 0.2); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.45); }
    .nx-badge-medium { background: rgba(217, 119, 6, 0.18); color: #fcd34d; border: 1px solid rgba(251, 191, 36, 0.42); }
    .nx-badge-low { background: rgba(22, 163, 74, 0.18); color: #86efac; border: 1px solid rgba(52, 211, 153, 0.4); }
    .nx-badge-violet { background: rgba(124, 58, 237, 0.2); color: #ddd6fe; border: 1px solid rgba(167, 139, 250, 0.45); }
    .nx-badge-cyan { background: rgba(8, 145, 178, 0.18); color: #a5f3fc; border: 1px solid rgba(34, 211, 238, 0.4); }
    .nx-badge-muted { background: rgba(100, 116, 139, 0.18); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
    .nx-account-list { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 2px; }
    .nx-account {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 6px;
        background: rgba(8, 145, 178, 0.10);
        border: 1px solid rgba(34, 211, 238, 0.20);
        color: #a5f3fc;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.74rem;
    }

    /* ---- Result banners ------------------------------------------------ */
    .nx-result {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 22px;
        border-radius: 14px;
        margin: 14px 0 6px;
    }
    .nx-result-bot {
        background: linear-gradient(90deg, rgba(109, 40, 217, 0.22), rgba(124, 58, 237, 0.08));
        border: 1px solid rgba(167, 139, 250, 0.5);
    }
    .nx-result-genuine {
        background: linear-gradient(90deg, rgba(8, 145, 178, 0.20), rgba(34, 211, 238, 0.06));
        border: 1px solid rgba(34, 211, 238, 0.42);
    }
    .nx-result-mark {
        width: 46px; height: 46px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem; font-weight: 800;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .nx-result-label { color: #cbd5e1; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
    .nx-result-value { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; }

    /* ---- Sidebar ------------------------------------------------------- */
    .nx-side-brand { text-align: center; padding: 2px 0 12px; }
    .nx-side-mark {
        width: 54px; height: 54px;
        margin: 0 auto 10px;
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem; font-weight: 800;
        color: #ecfeff;
        background: linear-gradient(145deg, #0e7490, #6d28d9);
        border: 1px solid rgba(167, 139, 250, 0.45);
        box-shadow: 0 0 30px rgba(34, 211, 238, 0.16);
    }
    .nx-side-name { color: #f1f5f9; font-size: 1.2rem; font-weight: 800; letter-spacing: 0.02em; }
    .nx-side-tag { color: #64748b; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px; }
    .nx-side-label {
        color: #67e8f9;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0 0 6px;
    }
    .nx-side-help {
        padding: 8px 10px;
        margin: 8px 0 8px;
        border: 1px solid rgba(34, 211, 238, 0.16);
        border-radius: 8px;
        background: rgba(8, 145, 178, 0.07);
        color: #94a3b8;
        font-size: 0.7rem;
        line-height: 1.55;
    }
    .nx-side-help strong { color: #a5f3fc; font-weight: 700; }
    .nx-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        color: #cbd5e1;
        font-size: 0.78rem;
    }
    .nx-status-note { margin-left: auto; color: #64748b; font-size: 0.68rem; }
    .nx-dot { width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto; }
    .nx-dot-ok { background: #34d399; box-shadow: 0 0 8px rgba(52, 211, 153, 0.55); }
    .nx-dot-bad { background: #f87171; box-shadow: 0 0 8px rgba(248, 113, 113, 0.45); }
    .nx-side-card {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.55);
        padding: 12px 14px;
        margin: 6px 0 10px;
    }
    .nx-side-note { color: #64748b; font-size: 0.7rem; line-height: 1.6; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #cbd5e1;
        font-size: 0.8rem;
        font-weight: 600;
        line-height: 1.4;
    }
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] { color: #64748b; }
    [data-testid="stSidebar"] [data-testid="stSlider"] { margin-bottom: 2px; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #64748b; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: 0.7rem; line-height: 1.5; }

    /* ---- Misc ---------------------------------------------------------- */
    .nx-input-group {
        color: #a5f3fc;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 14px 0 8px;
        padding-left: 9px;
        border-left: 3px solid #7c3aed;
    }
    .nx-footer {
        color: #475569;
        font-size: 0.72rem;
        line-height: 1.7;
        text-align: center;
        padding: 8px 0 4px;
    }
    .nx-divider { height: 1px; background: rgba(148, 163, 184, 0.14); margin: 18px 0 14px; }

    @media (max-width: 720px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


@st.cache_data(ttl=60, show_spinner=False)
def _load_csv(path_text: str, label: str) -> tuple[pd.DataFrame, str | None]:
    """Read a pipeline CSV and return ``(dataframe, error)``."""
    path = Path(path_text)
    if not path.exists():
        return (
            pd.DataFrame(),
            f"{label} not found. Expected `{_relative(path)}` — run "
            "`python run_pipeline.py` from the project root.",
        )
    try:
        return pd.read_csv(path, low_memory=False), None
    except Exception as exc:  # noqa: BLE001 - surface the real error to the user
        return pd.DataFrame(), f"Could not read `{_relative(path)}`: {type(exc).__name__}: {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def _load_json(path_text: str) -> tuple[object | None, str | None]:
    path = Path(path_text)
    if not path.exists():
        return None, f"`{_relative(path)}` not found."
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not parse `{_relative(path)}`: {type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def _load_model_bundle() -> tuple[dict | None, str | None]:
    try:
        from src.ml.predict_account import load_model_bundle

        return load_model_bundle(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _is_demo_dataset(posts: pd.DataFrame) -> bool:
    if posts.empty or "post_id" not in posts.columns or "account_id" not in posts.columns:
        return False
    ids = set(posts["post_id"].dropna().astype(str))
    return len(posts) == 10 and ids == DEMO_POST_IDS


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _fmt(value: object, digits: int = 3, fallback: str = "—") -> str:
    try:
        if value is None or pd.isna(value):
            return fallback
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def _source_line(path: Path) -> str:
    return f"Source: <code>{html.escape(_relative(path))}</code>"


def _section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="nx-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="nx-section"><span class="nx-section-bar"></span>'
        f'<div><div class="nx-section-title">{html.escape(title)}</div>'
        f"{subtitle_html}</div></div>",
        unsafe_allow_html=True,
    )


def _empty_state(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center;padding:46px 18px;color:#64748b;
                    border:1px dashed rgba(148,163,184,0.20);border-radius:12px;
                    background:rgba(15,23,42,0.35)">
            <div style="color:#94a3b8;font-size:1rem;font-weight:700;margin-bottom:6px">
                {html.escape(title)}
            </div>
            <div style="font-size:0.84rem;line-height:1.55">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _priority_badge(priority: object) -> str:
    label = str(priority).strip() or "Unknown"
    css = {
        "High": "nx-badge-high",
        "Medium": "nx-badge-medium",
        "Low": "nx-badge-low",
    }.get(label, "nx-badge-muted")
    return f'<span class="nx-badge {css}">{html.escape(label.upper())}</span>'


def _style_figure(
    figure: go.Figure,
    *,
    height: int = 330,
    show_legend: bool = True,
    margin_top: int = 44,
) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,21,39,0.55)",
        font=dict(color="#cbd5e1", family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=12, r=12, t=margin_top, b=12),
        height=height,
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#334155",
            font=dict(color="#e2e8f0", size=12),
        ),
        title=dict(font=dict(size=13.5, color="#e2e8f0"), x=0.01, xanchor="left"),
    )
    figure.update_xaxes(gridcolor="rgba(30,45,69,0.55)", zeroline=False, title_font=dict(size=11))
    figure.update_yaxes(gridcolor="rgba(30,45,69,0.55)", zeroline=False, title_font=dict(size=11))
    return figure


def _chart(figure: go.Figure, *, key: str | None = None) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        key=key,
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
    )


def _accounts_html(accounts_raw: object) -> str:
    accounts = [
        item.strip()
        for item in str(accounts_raw or "").split(",")
        if item.strip()
    ]
    if not accounts:
        return '<span style="color:#64748b">No member accounts listed.</span>'
    chips = "".join(
        f'<span class="nx-account">{html.escape(account)}</span>' for account in accounts
    )
    return f'<div class="nx-account-list">{chips}</div>'


def _render_data_error(message: str) -> None:
    st.error(message)


def _build_network_figure(pair_df: pd.DataFrame) -> go.Figure | None:
    """Build an interactive network from the pipeline GraphML, with a CSV fallback."""
    try:
        graph = nx.read_graphml(str(COORDINATION_GRAPH_GRAPHML))
    except Exception:  # noqa: BLE001 - GraphML is optional; fall back to the CSV
        graph = nx.Graph()

    if graph.number_of_nodes() == 0 and pair_df is not None and not pair_df.empty:
        for _, row in pair_df.iterrows():
            account_1 = str(row.get("account_1", "")).strip()
            account_2 = str(row.get("account_2", "")).strip()
            if not account_1 or not account_2 or account_1 == account_2:
                continue
            try:
                weight = float(row.get("coordination_score", 0.5))
            except (TypeError, ValueError):
                weight = 0.5
            graph.add_edge(account_1, account_2, weight=weight)

    if graph.number_of_nodes() == 0:
        return None

    try:
        positions = nx.spring_layout(graph, seed=42, iterations=120)
    except Exception:  # noqa: BLE001
        positions = nx.circular_layout(graph)

    figure = go.Figure()

    # Group edges by rounded strength so large graphs stay responsive.
    edge_groups: dict[float, list[tuple[str, str]]] = {}
    for node_a, node_b, attributes in graph.edges(data=True):
        try:
            weight = max(0.0, min(1.0, float(attributes.get("weight", 0.5))))
        except (TypeError, ValueError):
            weight = 0.5
        edge_groups.setdefault(round(weight * 4) / 4, []).append((node_a, node_b))

    for strength, edges in sorted(edge_groups.items()):
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for node_a, node_b in edges:
            x_a, y_a = positions[node_a]
            x_b, y_b = positions[node_b]
            edge_x.extend([x_a, x_b, None])
            edge_y.extend([y_a, y_b, None])
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=1.0 + 3.5 * strength,
                    color="rgba(148,163,184,0.30)",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    degrees = dict(graph.degree())
    nodes = list(graph.nodes())
    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    node_degrees = [degrees[node] for node in nodes]
    node_sizes = [18 + min(22, 4 * degree) for degree in node_degrees]
    node_hover = [
        f"<b>{html.escape(str(node))}</b><br>Connections: {degrees[node]}"
        for node in nodes
    ]
    show_labels = len(nodes) <= 40

    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if show_labels else "markers",
            text=[str(node) for node in nodes] if show_labels else None,
            textposition="top center",
            textfont=dict(color="#e2e8f0", size=11),
            hovertext=node_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=node_sizes,
                color=node_degrees,
                colorscale=[[0.0, CYAN_DEEP], [1.0, VIOLET]],
                line=dict(width=1, color="rgba(226,232,240,0.55)"),
                showscale=False,
            ),
            showlegend=False,
        )
    )

    figure.update_layout(
        title="Interactive coordination network",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    _style_figure(figure, height=440, show_legend=False)
    return figure


def _render_feature_group(
    title: str,
    features: list[str],
    preset_values: dict[str, float],
    preset_slug: str,
    available_features: list[str],
) -> dict[str, float]:
    selected_features = [feature for feature in features if feature in available_features]
    if not selected_features:
        return {}

    st.markdown(
        f'<div class="nx-input-group">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="medium")
    values: dict[str, float] = {}

    for index, feature in enumerate(selected_features):
        column = columns[index % 3]
        label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        help_text = FEATURE_HELP.get(feature, "")
        default = float(preset_values.get(feature, 0.0) or 0.0)
        widget_key = f"bot_{preset_slug}_{feature}"

        if feature in BOOLEAN_FEATURES:
            choice = column.selectbox(
                label,
                options=[0, 1],
                index=1 if default >= 0.5 else 0,
                format_func=lambda value: "Yes" if int(value) == 1 else "No",
                help=help_text,
                key=widget_key,
            )
            values[feature] = float(choice)
        elif feature in RATIO_FEATURES:
            values[feature] = float(
                column.number_input(
                    label,
                    min_value=0.0,
                    max_value=1.0,
                    value=max(0.0, min(1.0, default)),
                    step=0.01,
                    format="%.3f",
                    help=help_text,
                    key=widget_key,
                )
            )
        elif feature == "friends_followers_ratio":
            values[feature] = float(
                column.number_input(
                    label,
                    min_value=0.0,
                    value=max(0.0, default),
                    step=0.1,
                    format="%.2f",
                    help=help_text,
                    key=widget_key,
                )
            )
        elif feature in COUNT_FEATURES:
            values[feature] = float(
                column.number_input(
                    label,
                    min_value=0,
                    value=int(round(default)),
                    step=1,
                    format="%d",
                    help=help_text,
                    key=widget_key,
                )
            )
        else:
            values[feature] = float(
                column.number_input(
                    label,
                    min_value=0.0,
                    value=max(0.0, default),
                    step=0.1,
                    format="%.2f",
                    help=help_text,
                    key=widget_key,
                )
            )

    return values


def _render_prediction_result(stored_result: dict) -> None:
    prediction = stored_result.get("prediction", {})
    features = stored_result.get("features", {})
    created_at = stored_result.get("created_at", "")
    preset = stored_result.get("preset", "")

    try:
        label = int(prediction.get("predicted_label", -1))
    except (TypeError, ValueError):
        label = -1
    label_name = str(prediction.get("predicted_label_name", "Unknown"))
    try:
        bot_probability = (
            float(prediction["bot_probability"])
            if prediction.get("bot_probability") is not None
            else None
        )
    except (TypeError, ValueError):
        bot_probability = None
    try:
        genuine_probability = (
            float(prediction["genuine_probability"])
            if prediction.get("genuine_probability") is not None
            else None
        )
    except (TypeError, ValueError):
        genuine_probability = None

    is_bot = label == 1
    result_class = "nx-result-bot" if is_bot else "nx-result-genuine"
    accent = "#c4b5fd" if is_bot else "#67e8f9"
    mark = "B" if is_bot else "G"
    headline = "Bot-like pattern" if is_bot else "Genuine-like pattern"

    st.markdown(
        f"""
        <div class="nx-result {result_class}">
            <div class="nx-result-mark" style="color:{accent}">{mark}</div>
            <div>
                <div class="nx-result-label">Model prediction</div>
                <div class="nx-result-value" style="color:{accent}">{headline}</div>
                <div style="color:#94a3b8;font-size:0.76rem;margin-top:3px">
                    Predicted label: {html.escape(label_name)} ({label}) · {html.escape(str(preset))} · {html.escape(str(created_at))}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_columns = st.columns(3, gap="small")
    result_columns[0].metric("Predicted label", f"{label_name} ({label})")
    result_columns[1].metric(
        "Bot probability",
        _fmt(bot_probability, 4, "n/a"),
        help="Model-estimated probability for the Bot class.",
    )
    result_columns[2].metric(
        "Genuine probability",
        _fmt(genuine_probability, 4, "n/a"),
        help="Model-estimated probability for the Genuine class.",
    )

    if bot_probability is not None and genuine_probability is not None:
        probability_frame = pd.DataFrame(
            {
                "Class": ["Genuine", "Bot"],
                "Probability": [genuine_probability, bot_probability],
            }
        )
        probability_figure = px.bar(
            probability_frame,
            x="Probability",
            y="Class",
            orientation="h",
            color="Class",
            color_discrete_map={"Genuine": CYAN, "Bot": VIOLET},
            text=[f"{value:.4f}" for value in probability_frame["Probability"]],
            title="Class probabilities",
            template="plotly_dark",
        )
        probability_figure.update_layout(
            showlegend=False,
            xaxis=dict(range=[0, 1.05], title="Probability"),
            yaxis_title="",
        )
        probability_figure.update_traces(textposition="outside")
        _style_figure(probability_figure, height=210, show_legend=False)
        _chart(probability_figure, key="bot_probability_chart")

    with st.expander("Feature values submitted", expanded=False):
        submitted_frame = pd.DataFrame(
            [
                {
                    "Feature": FEATURE_LABELS.get(key, key.replace("_", " ").title()),
                    "Key": key,
                    "Value": value,
                }
                for key, value in features.items()
            ]
        )
        st.dataframe(
            submitted_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    payload = {
        "created_at": created_at,
        "preset": preset,
        "prediction": {
            "label": label,
            "label_name": label_name,
            "bot_probability": bot_probability,
            "genuine_probability": genuine_probability,
        },
        "features": {key: float(value) for key, value in features.items()},
    }
    st.download_button(
        "Download prediction (JSON)",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name="nextrace_prediction.json",
        mime="application/json",
        width="content",
    )
    st.caption(
        "This prediction is a statistical pattern match against labeled research accounts. "
        "It is not a determination about a real account and must be reviewed by a human."
    )


def _feature_importance_frame(bundle: dict) -> pd.DataFrame | None:
    pipeline = bundle.get("pipeline")
    feature_names = bundle.get("feature_names") or []
    if pipeline is None or not feature_names:
        return None

    importances = None
    for step in getattr(pipeline, "named_steps", {}).values():
        if hasattr(step, "feature_importances_"):
            importances = step.feature_importances_
            break
    if importances is None or len(importances) != len(feature_names):
        return None

    frame = pd.DataFrame(
        {
            "feature": [FEATURE_LABELS.get(name, name.replace("_", " ").title()) for name in feature_names],
            "importance": [float(value) for value in importances],
        }
    )
    return frame.sort_values("importance", ascending=False).head(12)


def _top_attributions(items: object, limit: int = 8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if isinstance(items, dict):
        items = list(items.items())
    if not isinstance(items, (list, tuple)):
        return pd.DataFrame(columns=["feature", "attribution"])

    for item in items:
        if isinstance(item, dict):
            feature = item.get("feature")
            value = item.get("attribution", item.get("value"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            feature, value = item[0], item[1]
        else:
            continue
        try:
            attribution = float(value)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "feature": FEATURE_LABELS.get(str(feature), str(feature).replace("_", " ").title()),
                "attribution": attribution,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "attribution"])
    frame = pd.DataFrame(rows)
    frame["_abs"] = frame["attribution"].abs()
    return (
        frame.sort_values("_abs", ascending=False)
        .head(limit)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )


def _attribution_figure(frame: pd.DataFrame, title: str) -> go.Figure | None:
    if frame is None or frame.empty:
        return None
    plot = frame.copy()
    plot["_abs"] = plot["attribution"].abs()
    plot = plot.sort_values("_abs", ascending=True)
    max_abs = max(float(plot["_abs"].max()), 1e-9)

    figure = px.bar(
        plot,
        x="attribution",
        y="feature",
        orientation="h",
        color="attribution",
        color_continuous_scale=[
            [0.0, CYAN],
            [0.5, "#334155"],
            [1.0, VIOLET],
        ],
        range_color=[-max_abs, max_abs],
        text=[f"{value:+.3f}" for value in plot["attribution"]],
        title=title,
        template="plotly_dark",
    )
    figure.add_vline(x=0, line_width=1, line_color="rgba(148,163,184,0.55)")
    figure.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Attribution toward bot class",
        yaxis_title="",
    )
    figure.update_traces(textposition="outside")
    _style_figure(figure, height=330, show_legend=False)
    return figure


# ---------------------------------------------------------------------------
# Load pipeline artifacts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
_CRESCI_POSTS_FILE = (
    _PROJECT_ROOT
    / "data"
    / "processed"
    / "cresci_posts_multisource_cleaned.csv"
)

_CRESCI_REPORT_DIR = _PROJECT_ROOT / "results" / "reports"

with st.sidebar:
    st.markdown(
        '<div class="nx-side-label">Dataset selection</div>',
        unsafe_allow_html=True,
    )

    _dataset_choice = st.radio(
        "Choose dataset",
        options=["Demo Dataset", "Cresci Research Dataset"],
        index=0,
        key="nx_dataset_choice",
        help="Switch between the synthetic demonstration data and the Cresci research sample.",
    )

_use_cresci = _dataset_choice == "Cresci Research Dataset"

# ---------------------------------------------------------------------------
# Load pipeline artifacts
# ---------------------------------------------------------------------------
with st.spinner("Loading NexTrace pipeline outputs and model metadata…"):
    if _use_cresci:
        _posts_df, _posts_err = _load_csv(
            str(_CRESCI_POSTS_FILE), "Cresci research posts"
        )

        _coord_df, _coord_err = _load_csv(
            str(_CRESCI_REPORT_DIR / "cresci_coordination_results.csv"),
            "Cresci coordination results",
        )

        _risk_df, _risk_err = _load_csv(
            str(_CRESCI_REPORT_DIR / "cresci_account_analysis.csv"),
            "Cresci account analysis",
        )

        _camp_df, _camp_err = _load_csv(
            str(_CRESCI_REPORT_DIR / "cresci_campaign_results.csv"),
            "Cresci campaign results",
        )

        _behavior_df, _behavior_err = _load_csv(
            str(_CRESCI_REPORT_DIR / "cresci_network_metrics.csv"),
            "Cresci network metrics",
        )

    else:
        _posts_df, _posts_err = _load_csv(
            str(CLEANED_POSTS_FILE), "Cleaned posts"
        )
        _coord_df, _coord_err = _load_csv(
            str(COORDINATION_RESULTS_FILE), "Coordination results"
        )
        _risk_df, _risk_err = _load_csv(
            str(RISK_RESULTS_FILE), "Risk results"
        )
        _camp_df, _camp_err = _load_csv(
            str(CAMPAIGN_RESULTS_FILE), "Campaign results"
        )
        _behavior_df, _behavior_err = _load_csv(
            str(ACCOUNT_BEHAVIOR_FILE), "Account behaviour results"
        )

    _comparison_df, _comparison_err = _load_csv(
        str(ML_MODEL_COMPARISON_FILE), "Model comparison report"
    )
    _metadata, _metadata_err = _load_json(str(MODEL_METADATA_FILE))
    _evaluation, _evaluation_err = _load_json(str(ML_EVALUATION_SUMMARY_FILE))
    _explanations, _explanations_err = _load_json(str(ML_EXPLANATIONS_FILE))
    _run_meta, _run_meta_err = _load_json(
        str(_PROJECT_ROOT / "results" / "reports" / "pipeline_run_metadata.json")
    )
    _classification_df, _classification_err = _load_csv(
        str(_PROJECT_ROOT / "results" / "reports" / "account_classification_results.csv"),
        "Account ML classifications",
    )

_metadata = _metadata if isinstance(_metadata, dict) else {}
_test_metrics = _metadata.get("test_metrics", {}) if isinstance(_metadata, dict) else {}
_run_meta = _run_meta if isinstance(_run_meta, dict) else {}

_demo_mode = not _use_cresci and (
    _is_demo_dataset(_posts_df)
    or _run_meta.get("is_synthetic", False)
    or _run_meta.get("source") == "demo"
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="nx-side-brand">
            <div class="nx-side-mark">N</div>
            <div class="nx-side-name">NexTrace</div>
            <div class="nx-side-tag">Intelligence Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nx-side-label">Display thresholds</div>', unsafe_allow_html=True)
    coord_threshold = st.slider(
        "Coordination score",
        min_value=0.0,
        max_value=1.0,
        value=float(RISK_MEDIUM_THRESHOLD),
        step=0.05,
        help=(
            "Minimum coordination score for pairs shown in Coordination Detection and "
            "Risk Scores. This filters precomputed results; it does not re-run detection."
        ),
    )
    sim_threshold = st.slider(
        "Semantic similarity",
        min_value=0.0,
        max_value=1.0,
        value=float(SEMANTIC_THRESHOLD),
        step=0.05,
        help=(
            "Minimum semantic similarity for the same pair-level tabs. "
            f"The pipeline flags pairs at {SEMANTIC_THRESHOLD:.2f}; lowering this filter "
            "below that value cannot reveal additional pairs."
        ),
    )
    st.markdown(
        '<div class="nx-side-help"><strong>Display-only filters.</strong> '
        "These thresholds narrow the precomputed pairs shown in Coordination Detection and "
        "Risk Scores. They do not re-run detection, rewrite reports, or alter scores.</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Pipeline defaults: semantic ≥ {SEMANTIC_THRESHOLD:.2f} · "
        f"time window ≤ {TIME_WINDOW_MINUTES} min."
    )

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-side-label">Pipeline status</div>', unsafe_allow_html=True)

    _status_items = [
        ("Posts", CLEANED_POSTS_FILE),
        ("Coordination", COORDINATION_RESULTS_FILE),
        ("Risk scores", RISK_RESULTS_FILE),
        ("Campaigns", CAMPAIGN_RESULTS_FILE),
        ("Network graph", COORDINATION_GRAPH_GRAPHML),
    ]
    for _label, _path in _status_items:
        _ok = _path.exists()
        st.markdown(
            f'<div class="nx-status"><span class="nx-dot '
            f'{"nx-dot-ok" if _ok else "nx-dot-bad"}"></span>{html.escape(_label)}'
            f'<span class="nx-status-note">{"ready" if _ok else "missing"}</span></div>',
            unsafe_allow_html=True,
        )

    _model_ok = SELECTED_MODEL_FILE.exists()
    st.markdown(
        f'<div class="nx-status"><span class="nx-dot '
        f'{"nx-dot-ok" if _model_ok else "nx-dot-bad"}"></span>Bot classifier'
        f'<span class="nx-status-note">{"ready" if _model_ok else "missing"}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-side-label">Trained model</div>', unsafe_allow_html=True)
    if _model_ok:
        _model_name = str(_metadata.get("model_name", "random_forest")).replace("_", " ").title()
        st.markdown(
            f"""
            <div class="nx-side-card">
                <div style="color:#e2e8f0;font-size:0.82rem;font-weight:700">{html.escape(_model_name)}</div>
                <div style="color:#64748b;font-size:0.68rem;margin:3px 0 9px">
                    {len(_metadata.get("feature_names", []))} features · Cresci-2015/2017
                </div>
                <div style="display:flex;gap:16px">
                    <div>
                        <div style="color:#64748b;font-size:0.62rem;letter-spacing:0.08em">TEST F1</div>
                        <div style="color:#22d3ee;font-size:1rem;font-weight:800">
                            {_fmt(_test_metrics.get("f1_score"), 4, "n/a")}
                        </div>
                    </div>
                    <div>
                        <div style="color:#64748b;font-size:0.62rem;letter-spacing:0.08em">ROC-AUC</div>
                        <div style="color:#a78bfa;font-size:1rem;font-weight:800">
                            {_fmt(_test_metrics.get("roc_auc"), 4, "n/a")}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Model file not found.", icon=":material/warning:")

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-side-label">Ingestion & Run Provenance</div>', unsafe_allow_html=True)
    _active_src = _run_meta.get("source", "demo")
    _run_time = _run_meta.get("run_timestamp", "")
    _time_display = _run_time[:19].replace("T", " ") if _run_time else "Baseline Run"

    st.markdown(
        f"""
        <div class="nx-side-card" style="margin-bottom:10px">
            <div style="color:#e2e8f0;font-size:0.78rem;font-weight:700">Source: <span style="color:#22d3ee">{html.escape(_active_src.upper())}</span></div>
            <div style="color:#64748b;font-size:0.68rem;margin-top:2px">Run: {_time_display} UTC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Import Custom Dataset", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload posts (CSV/JSON)",
            type=["csv", "json"],
            help="Upload posts containing post_id, account_id, text, timestamp.",
            key="dash_uploader",
        )
        if uploaded_file is not None:
            if st.button("Run Pipeline On Upload", type="primary"):
                try:
                    from src.contracts import validate_and_normalize_posts
                    if uploaded_file.name.endswith(".csv"):
                        up_df = pd.read_csv(uploaded_file)
                    else:
                        up_df = pd.read_json(uploaded_file)
                    cleaned_up, val_warn = validate_and_normalize_posts(
                        up_df, default_source="manual_upload", default_provenance=uploaded_file.name
                    )
                    from src.config import SAMPLE_POSTS_FILE
                    cleaned_up.to_csv(SAMPLE_POSTS_FILE, index=False)
                    from run_pipeline import run_pipeline
                    run_pipeline(source="file", posts_path=str(SAMPLE_POSTS_FILE), skip_sample_data=False)
                    st.cache_data.clear()
                    st.success("Analysis complete! Reloading...")
                    st.rerun()
                except Exception as up_err:
                    st.error(f"Upload error: {up_err}")

    with st.expander("Manual Post Input", expanded=False):
        m_account = st.text_input("Account ID / Handle", value="analyst_test", key="man_acc")
        m_text = st.text_area("Post Text", placeholder="Paste social media post text here...", key="man_text")
        m_url = st.text_input("Shared URL (optional)", placeholder="https://example.com/claim", key="man_url")
        if st.button("Analyze Manual Post", type="primary", key="btn_man_post"):
            if not m_text.strip():
                st.warning("Please enter post text to analyze.")
            else:
                try:
                    from datetime import datetime, timezone
                    import pandas as pd
                    from src.contracts import validate_and_normalize_posts
                    from src.config import SAMPLE_POSTS_FILE
                    from run_pipeline import run_pipeline

                    new_row = {
                        "post_id": f"MANUAL_{int(datetime.now(timezone.utc).timestamp())}",
                        "account_id": m_account.strip() or "anonymous",
                        "text": m_text.strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "url": m_url.strip(),
                        "source_platform": "manual_entry",
                        "provenance": "analyst_input",
                    }
                    # Append to existing sample posts if exists
                    if SAMPLE_POSTS_FILE.exists():
                        existing_df = pd.read_csv(SAMPLE_POSTS_FILE)
                        combined_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        combined_df = pd.DataFrame([new_row])

                    cleaned_df, _ = validate_and_normalize_posts(
                        combined_df, default_source="manual_entry", default_provenance="analyst_input"
                    )
                    cleaned_df.to_csv(SAMPLE_POSTS_FILE, index=False)
                    run_pipeline(source="file", posts_path=str(SAMPLE_POSTS_FILE), skip_sample_data=False)
                    st.cache_data.clear()
                    st.success("Post analyzed and pipeline executed! Reloading...")
                    st.rerun()
                except Exception as man_err:
                    st.error(f"Error analyzing post: {man_err}")

    with st.expander("Live API & OSINT Services", expanded=False):
        from src.ingestion.reddit_provider import RedditAPIProvider
        from src.ingestion.x_provider import XAPIProvider
        from src.osint import GDELTClient, GoogleFactCheckClient, RDAPClient, VisionOCRClient

        st.markdown("**Social Media Providers**")
        x_stat = XAPIProvider().check_status()
        x_badge = "🟢 OPERATIONAL" if x_stat.get("status") == "OPERATIONAL" else f"⚪ {x_stat.get('status')}"
        st.markdown(f"- **X API v2**: `{x_badge}`")
        if not x_stat.get("configured"):
            st.caption("Set X_BEARER_TOKEN in .env to enable live X ingestion.")

        red_stat = RedditAPIProvider().check_status()
        red_badge = "🟢 OPERATIONAL" if red_stat.get("status") == "OPERATIONAL" else f"🟡 {red_stat.get('status')}"
        st.markdown(f"- **Reddit API**: `{red_badge}`")

        st.markdown("**OSINT & Verification Services**")
        gdelt_stat = GDELTClient().check_status()
        gd_badge = "🟢 OPERATIONAL" if gdelt_stat.get("status") == "OPERATIONAL" else f"⚪ {gdelt_stat.get('status')}"
        st.markdown(f"- **GDELT DOC 2.0**: `{gd_badge}`")

        fc_stat = GoogleFactCheckClient().check_status()
        fc_badge = "🟢 OPERATIONAL" if fc_stat.get("status") == "OPERATIONAL" else f"⚪ {fc_stat.get('status')}"
        st.markdown(f"- **Google Fact Check**: `{fc_badge}`")
        if not fc_stat.get("configured"):
            st.caption("Set GOOGLE_FACT_CHECK_API_KEY in .env for claim verification.")

        rdap_stat = RDAPClient().check_status()
        rd_badge = "🟢 OPERATIONAL" if rdap_stat.get("status") == "OPERATIONAL" else f"⚪ {rdap_stat.get('status')}"
        st.markdown(f"- **RDAP Resolver**: `{rd_badge}`")

        ocr_stat = VisionOCRClient().check_status()
        st.markdown(f"- **Vision OCR**: `{ocr_stat.get('status')}` ({ocr_stat.get('engine')})")

        st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Execute Live API Search**")
        live_src = st.selectbox("API Source", options=["x", "reddit"], format_func=lambda s: "Twitter / X API v2" if s == "x" else "Reddit Data API")
        live_query = st.text_input("Search Keyword / Query", value="disinformation", key="live_api_query_input")
        if st.button("Fetch & Analyze from API", key="btn_run_live_api"):
            try:
                from run_pipeline import run_pipeline
                ret = run_pipeline(source=live_src, query=live_query, skip_sample_data=False)
                if ret == 0:
                    st.cache_data.clear()
                    st.success(f"Successfully ingested live records from {live_src}! Reloading...")
                    st.rerun()
                else:
                    st.error(f"Live API ingestion from {live_src} failed or was blocked. Check credentials in .env.")
            except Exception as api_err:
                st.error(f"Live API Error: {api_err}")

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nx-side-note">'
        "NexTrace is a research prototype. Scores are investigation leads, not proof "
        "of disinformation or malicious intent. Every flagged pair requires human review."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="nx-hero">
        <div class="nx-hero-copy">
            <div class="nx-eyebrow">SOCIAL MEDIA INTELLIGENCE PLATFORM</div>
            <div class="nx-title">Nex<span>Trace</span></div>
            <div class="nx-subtitle">
                Analyze coordinated social media activity through semantic similarity,
                temporal patterns, shared URLs, network relationships, and
                machine-learning-based account classification.
            </div>
            <div class="nx-pills">
                <span class="nx-pill nx-pill-cyan">Phase 1 · Coordination</span>
                <span class="nx-pill nx-pill-violet">Phase 2 · Bot classifier</span>
            </div>
        </div>
        <div class="nx-mark">N</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data-source notice
# ---------------------------------------------------------------------------
_source_accounts = (
    int(_posts_df["account_id"].nunique())
    if not _posts_df.empty and "account_id" in _posts_df.columns
    else 0
)
_source_icon = (
    '<svg class="nx-source-icon" width="13" height="13" viewBox="0 0 16 16" '
    'fill="none" aria-hidden="true" focusable="false">'
    '<circle cx="8" cy="8" r="6.4" stroke="currentColor" stroke-width="1.3"/>'
    '<path d="M8 7.2v3.4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
    '<circle cx="8" cy="4.9" r="0.8" fill="currentColor"/>'
    "</svg>"
)
if _demo_mode:
    _source_tag_class = ""
    _source_tag = "DEMO DATA"
    _source_message = (
        f"Showing <strong>{len(_posts_df):,} synthetic posts</strong> from "
        f"<strong>{_source_accounts:,} accounts</strong>. Results are illustrative and "
        "do not represent real-world findings."
    )
elif _posts_err is not None or _posts_df.empty:
    _source_tag_class = " is-muted"
    _source_tag = "NO DATA"
    _source_message = (
        "No post data is loaded. Run the pipeline to generate the report artifacts."
    )
else:
    _source_tag_class = " is-pipeline"
    _source_tag = "PIPELINE DATA"
    _post_noun = "post" if len(_posts_df) == 1 else "posts"
    _account_noun = "account" if _source_accounts == 1 else "accounts"
    _account_phrase = (
        f" from <strong>{_source_accounts:,} {_account_noun}</strong>"
        if "account_id" in _posts_df.columns
        else ""
    )
    _source_message = (
        f"Showing <strong>{len(_posts_df):,} {_post_noun}</strong>{_account_phrase} "
        "loaded from local pipeline reports. Results are investigation leads and "
        "require human review."
    )
st.markdown(
    f'<div class="nx-source-note">'
    f'<span class="nx-source-tag{_source_tag_class}">{_source_tag}</span>'
    f"{_source_icon}"
    f'<span class="nx-source-text">{_source_message}</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
_posts_ready = _posts_err is None
_accounts_ready = _posts_ready and "account_id" in _posts_df.columns
_pairs_ready = _coord_err is None and "coordination_score" in _coord_df.columns
_risk_ready = _risk_err is None and "investigation_priority" in _risk_df.columns
_campaigns_ready = _camp_err is None and "campaign_id" in _camp_df.columns

_post_count = len(_posts_df) if _posts_ready else 0
_account_count = int(_posts_df["account_id"].nunique()) if _accounts_ready else 0
_pair_count = len(_coord_df) if _pairs_ready else 0
_high_priority = (
    int((_risk_df["investigation_priority"].astype(str).str.lower() == "high").sum())
    if _risk_ready
    else 0
)
_campaign_count = len(_camp_df) if _campaigns_ready else 0


def _kpi_card(
    label: str,
    value: int | None,
    help_text: str,
    footnote: str,
    unavailable_note: str = "Report unavailable",
) -> str:
    label_safe = html.escape(label)
    help_safe = html.escape(help_text, quote=True)
    if value is None:
        tooltip = html.escape(f"{help_text} Unavailable: {unavailable_note}.", quote=True)
        return (
            '<div class="nx-kpi is-unavailable">'
            f'<div class="nx-kpi-head"><span class="nx-kpi-label">{label_safe}</span>'
            f'<span class="nx-kpi-help" role="img" tabindex="0" '
            f'aria-label="{tooltip}" title="{tooltip}">i</span></div>'
            '<div class="nx-kpi-value is-unavailable">—</div>'
            f'<div class="nx-kpi-foot">{html.escape(unavailable_note)}</div>'
            "</div>"
        )
    return (
        '<div class="nx-kpi">'
        f'<div class="nx-kpi-head"><span class="nx-kpi-label">{label_safe}</span>'
        f'<span class="nx-kpi-help" role="img" tabindex="0" '
        f'aria-label="{help_safe}" title="{help_safe}">i</span></div>'
        f'<div class="nx-kpi-value">{value:,}</div>'
        f'<div class="nx-kpi-foot">{html.escape(footnote)}</div>'
        "</div>"
    )


_kpi_html = "".join(
    [
        _kpi_card(
            "Posts analysed",
            _post_count if _posts_ready else None,
            "Rows in the cleaned Phase 1 post dataset.",
            "Cleaned Phase 1 dataset",
            "Post report unavailable",
        ),
        _kpi_card(
            "Accounts observed",
            _account_count if _accounts_ready else None,
            "Distinct account IDs in the post dataset.",
            "Distinct account IDs",
            "Account column unavailable",
        ),
        _kpi_card(
            "Flagged pairs",
            _pair_count if _pairs_ready else None,
            "Account pairs meeting the Phase 1 coordination criteria.",
            "Meeting Phase 1 criteria",
            "Coordination report unavailable",
        ),
        _kpi_card(
            "High-priority pairs",
            _high_priority if _risk_ready else None,
            "Pairs labelled High investigation priority in the risk report.",
            "High priority in risk report",
            "Risk report unavailable",
        ),
        _kpi_card(
            "Campaign groups",
            _campaign_count if _campaigns_ready else None,
            "Connected components in the coordination network.",
            "Connected components",
            "Campaign report unavailable",
        ),
    ]
)
st.markdown(f'<div class="nx-kpi-grid">{_kpi_html}</div>', unsafe_allow_html=True)

st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_posts, tab_coord, tab_risk, tab_campaign, tab_bot, tab_cresci = st.tabs(
    [
        "Posts",
        "Coordination Detection",
        "Risk Scores",
        "Campaign Analysis",
        "Bot Classifier",
        "Cresci Analysis",
    ]
)
# ===========================================================================
# TAB 1 — POSTS
# ===========================================================================
with tab_posts:
    _section_header(
        "Social media posts",
        _source_line(_CRESCI_POSTS_FILE if _use_cresci else CLEANED_POSTS_FILE)
        + " · Phase 1 input after validation and normalisation",
    )

    if _posts_err:
        _render_data_error(_posts_err)
    elif _posts_df.empty:
        _empty_state(
            "No posts available",
            "Run <code>python run_pipeline.py</code> from the project root to generate the "
            "cleaned post dataset.",
        )
    else:
        _posts = _posts_df.copy()
        for _required in ("post_id", "account_id", "text", "timestamp"):
            if _required not in _posts.columns:
                st.error(
                    f"The post dataset is missing the required column `{_required}`. "
                    "Re-run `python run_pipeline.py`."
                )
                break
        else:
            _timestamps = pd.to_datetime(_posts["timestamp"], errors="coerce", utc=True)
            _urls = _posts["url"].fillna("").astype(str).str.strip() if "url" in _posts.columns else pd.Series("", index=_posts.index)
            _has_url = (~_urls.isin({"", "nan", "None"})).fillna(False)

            with st.container(border=True):
                _filter_left, _filter_mid, _filter_right = st.columns([2.2, 1.6, 1.1], gap="medium")
                with _filter_left:
                    _search_text = st.text_input(
                        "Search post text",
                        placeholder="e.g. cyber security, rainfall, recruitment…",
                        key="posts_search_text",
                    )
                _account_values = sorted(_posts["account_id"].dropna().astype(str).unique().tolist())
                with _filter_mid:
                    _selected_accounts = st.multiselect(
                        "Accounts",
                        options=_account_values,
                        default=[],
                        placeholder="All accounts",
                        key="posts_account_filter",
                    )
                with _filter_right:
                    _url_only = st.checkbox(
                        "With URLs only",
                        value=False,
                        key="posts_url_only",
                    )

            _filtered = _posts.copy()
            _filtered["_ts"] = _timestamps
            _filtered["_has_url"] = _has_url

            if _search_text:
                _filtered = _filtered[
                    _filtered["text"].astype(str).str.contains(
                        _search_text, case=False, na=False, regex=False
                    )
                ]
            if _selected_accounts:
                _filtered = _filtered[
                    _filtered["account_id"].astype(str).isin(_selected_accounts)
                ]
            if _url_only:
                _filtered = _filtered[_filtered["_has_url"]]

            _visible_posts = len(_filtered)
            _visible_accounts = int(_filtered["account_id"].nunique())
            _visible_urls = int(_filtered["_has_url"].sum())
            _median_length = (
                _filtered["text"].fillna("").astype(str).str.len().median()
                if _visible_posts
                else 0
            )

            _pc1, _pc2, _pc3, _pc4 = st.columns(4, gap="small")
            _pc1.metric("Visible posts", _visible_posts)
            _pc2.metric("Visible accounts", _visible_accounts)
            _pc3.metric("Posts with URLs", _visible_urls)
            _pc4.metric("Median text length", f"{_median_length:.0f} chars")

            if _filtered.empty:
                _empty_state(
                    "No posts match these filters",
                    "Clear the search text or select fewer filters to see the dataset.",
                )
            else:
                _chart_col, _bar_col = st.columns([1.35, 1], gap="medium")

                with _chart_col:
                    _timeline = _filtered.dropna(subset=["_ts"]).copy()
                    if not _timeline.empty:
                        _timeline["_account"] = _timeline["account_id"].astype(str)
                        _timeline["_hover_text"] = _timeline["text"].astype(str).str.slice(0, 120)
                        _timeline_fig = px.scatter(
                            _timeline,
                            x="_ts",
                            y="_account",
                            color="_account",
                            hover_data={"_hover_text": True, "post_id": True, "_ts": False},
                            labels={"_ts": "Timestamp (UTC)", "_account": "Account", "_hover_text": "Post"},
                            title="Post activity timeline",
                            template="plotly_dark",
                            color_discrete_sequence=[
                                CYAN,
                                VIOLET,
                                "#f472b6",
                                "#38bdf8",
                                "#c084fc",
                                "#2dd4bf",
                                "#818cf8",
                                "#facc15",
                                "#fb7185",
                                "#4ade80",
                            ],
                        )
                        _timeline_fig.update_traces(
                            marker=dict(size=11, line=dict(width=1, color="#0a0f1e"))
                        )
                        _timeline_fig.update_layout(showlegend=False)
                        _style_figure(_timeline_fig, height=350, show_legend=False)
                        _chart(_timeline_fig, key="posts_timeline")
                    else:
                        st.warning("No valid timestamps to plot.", icon=":material/schedule:")

                with _bar_col:
                    _per_account = (
                        _filtered.groupby("account_id", as_index=False)
                        .size()
                        .rename(columns={"size": "posts"})
                        .sort_values("posts", ascending=True)
                    )
                    _account_fig = px.bar(
                        _per_account,
                        x="posts",
                        y="account_id",
                        orientation="h",
                        title="Posts per account",
                        template="plotly_dark",
                        text="posts",
                        color="posts",
                        color_continuous_scale=[[0.0, CYAN_DEEP], [1.0, VIOLET_DEEP]],
                    )
                    _account_fig.update_layout(
                        coloraxis_showscale=False,
                        xaxis_title="Posts",
                        yaxis_title="",
                        height=350,
                    )
                    _account_fig.update_traces(textposition="outside")
                    _style_figure(_account_fig, height=350, show_legend=False)
                    _chart(_account_fig, key="posts_per_account")

                _display = _filtered.copy()
                _display["timestamp"] = _display["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                _display["timestamp"] = _display["timestamp"].fillna("Invalid timestamp")
                _display["text_length"] = _display["text"].fillna("").astype(str).str.len()
                _display_cols = [
                    column
                    for column in ["post_id", "account_id", "text", "timestamp", "url", "text_length"]
                    if column in _display.columns
                ]
                st.dataframe(
                    _display[_display_cols].reset_index(drop=True),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "post_id": st.column_config.TextColumn("Post ID", width="small"),
                        "account_id": st.column_config.TextColumn("Account", width="small"),
                        "text": st.column_config.TextColumn("Post text", width="large"),
                        "url": st.column_config.LinkColumn("URL", width="medium"),
                        "timestamp": st.column_config.TextColumn("Timestamp (UTC)", width="medium"),
                        "text_length": st.column_config.NumberColumn("Chars", format="%d", width="small"),
                    },
                )

                _download_cols = [
                    column
                    for column in ["post_id", "account_id", "text", "timestamp", "url"]
                    if column in _display.columns
                ]
                st.download_button(
                    "Download filtered posts (CSV)",
                    data=_display[_download_cols].to_csv(index=False).encode("utf-8"),
                    file_name="nextrace_filtered_posts.csv",
                    mime="text/csv",
                    width="content",
                )

            if not _behavior_err and not _behavior_df.empty:
                with st.expander("Account behaviour summary (Phase 1)", expanded=False):
                    st.caption(
                        "Per-account posting statistics computed by `src/account_analysis.py`. "
                        "These are descriptive features, not risk determinations."
                    )
                    st.dataframe(
                        _behavior_df.reset_index(drop=True),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "account_id": st.column_config.TextColumn("Account", width="small"),
                            "total_posts": st.column_config.NumberColumn("Posts", format="%d"),
                            "unique_texts": st.column_config.NumberColumn("Unique texts", format="%d"),
                            "urls_shared": st.column_config.NumberColumn("URLs shared", format="%d"),
                            "duplicate_text_posts": st.column_config.NumberColumn("Duplicates", format="%d"),
                            "average_time_between_posts_minutes": st.column_config.NumberColumn(
                                "Avg. gap (min)", format="%.1f"
                            ),
                        },
                    )
                    if (
                        "total_posts" in _behavior_df.columns
                        and pd.to_numeric(_behavior_df["total_posts"], errors="coerce").fillna(0).le(1).all()
                    ):
                        st.caption(
                            "Each account has a single post in this demonstration dataset, so "
                            "time-between-posts is undefined (shown as blank)."
                        )
            elif _behavior_err:
                st.caption(_behavior_err)

# ===========================================================================
# TAB 2 — COORDINATION DETECTION
# ===========================================================================
with tab_coord:
    _section_header(
        "Coordination detection",
        _source_line(COORDINATION_RESULTS_FILE)
        + " · Account pairs meeting the Phase 1 signal criteria",
    )
    st.caption(
        "Coordination is inferred from semantic similarity, posting-time proximity, and shared URLs. "
        "A high score flags a pair for review; it does **not** confirm coordination or malicious intent."
    )

    with st.expander("How the coordination score is calculated", expanded=False):
        st.markdown(
            f"""
            A pair is flagged only when the posts are semantically similar
            (cosine similarity ≥ **{SEMANTIC_THRESHOLD:.2f}**) **and** at least
            **{MIN_SIGNAL_COUNT}** signals are present. The composite score is:

            `score = {COORDINATION_WEIGHT_SEMANTIC:.2f} × similarity
            + {COORDINATION_WEIGHT_TIME:.2f} × close_in_time
            + {COORDINATION_WEIGHT_URL:.2f} × shared_url`

            where *close in time* means the posts are at most
            **{TIME_WINDOW_MINUTES} minutes** apart. Priorities are assigned at
            **{RISK_MEDIUM_THRESHOLD:.2f}** (Medium) and **{RISK_HIGH_THRESHOLD:.2f}** (High).
            """
        )

    if _coord_err:
        _render_data_error(_coord_err)
    elif _coord_df.empty:
        _empty_state(
            "No coordination results",
            "Run <code>python run_pipeline.py</code> to generate the coordination report.",
        )
    elif "coordination_score" not in _coord_df.columns:
        st.error("The coordination report is missing `coordination_score`. Re-run the pipeline.")
    else:
        _coord = _coord_df.copy()
        _coord["_score"] = _numeric(_coord, "coordination_score").fillna(0.0)
        _coord["_similarity"] = _numeric(_coord, "semantic_similarity").fillna(0.0)
        _coord["_time_diff"] = _numeric(_coord, "time_difference_minutes")
        _coord["_signals"] = _numeric(_coord, "evidence_signals").fillna(0)
        _coord["_shared_url"] = (
            _as_bool(_coord["shared_url"]) if "shared_url" in _coord.columns else False
        )

        _coord = _coord[
            (_coord["_score"] >= coord_threshold)
            & (_coord["_similarity"] >= sim_threshold)
        ]

        with st.container(border=True):
            _cf1, _cf2, _cf3 = st.columns([1.5, 1.5, 1.2], gap="medium")
            _pair_accounts = sorted(
                set(_coord_df["account_1"].dropna().astype(str))
                | set(_coord_df["account_2"].dropna().astype(str))
            )
            with _cf1:
                _pair_account = st.selectbox(
                    "Account in pair",
                    options=["All accounts"] + _pair_accounts,
                    key="coord_account_filter",
                )
            _signal_options = sorted(
                int(value)
                for value in pd.to_numeric(
                    _coord_df.get("evidence_signals", pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna().unique()
            )
            with _cf2:
                _signal_filter = st.multiselect(
                    "Evidence signals",
                    options=_signal_options,
                    default=_signal_options,
                    key="coord_signal_filter",
                    help="Number of supporting signals (text, timing, shared URL).",
                )
            with _cf3:
                _shared_only = st.checkbox(
                    "Shared URL only",
                    value=False,
                    key="coord_shared_only",
                )

        if _pair_account != "All accounts":
            _coord = _coord[
                (_coord["account_1"].astype(str) == _pair_account)
                | (_coord["account_2"].astype(str) == _pair_account)
            ]
        if _signal_filter:
            _coord = _coord[_coord["_signals"].isin(_signal_filter)]
        if _shared_only:
            _coord = _coord[_coord["_shared_url"]]

        st.caption(
            f"Showing **{len(_coord)}** pair(s) · coordination ≥ **{coord_threshold:.2f}** "
            f"· semantic similarity ≥ **{sim_threshold:.2f}**"
        )

        if _coord.empty:
            _empty_state(
                "No pairs match these filters",
                "Lower the sidebar thresholds or widen the account/signal filters.",
            )
        else:
            _cc1, _cc2, _cc3, _cc4 = st.columns(4, gap="small")
            _cc1.metric("Pairs shown", len(_coord))
            _cc2.metric("Mean score", _fmt(_coord["_score"].mean(), 3))
            _cc3.metric("Max score", _fmt(_coord["_score"].max(), 3))
            _cc4.metric("Avg. signals", _fmt(_coord["_signals"].mean(), 1))

            _hist_col, _scatter_col = st.columns(2, gap="medium")

            with _hist_col:
                # Explicit 0.1 buckets keep the chart readable when all scores coincide.
                _score_bins = pd.cut(
                    _coord["_score"].astype(float).clip(lower=0.0, upper=1.0),
                    bins=[index / 10 for index in range(11)],
                    include_lowest=True,
                    right=True,
                )
                _score_counts = _score_bins.value_counts(sort=False)
                _hist = go.Figure(
                    go.Bar(
                        x=[
                            (interval.left + interval.right) / 2
                            for interval in _score_counts.index
                        ],
                        y=_score_counts.to_numpy(),
                        marker_color=CYAN,
                        marker_line_width=0,
                        width=0.085,
                        hovertemplate="Score %{x:.2f} · %{y} pair(s)<extra></extra>",
                    )
                )
                _hist.update_layout(
                    title="Coordination score distribution",
                    xaxis_title="Coordination score",
                    yaxis_title="Pairs",
                    xaxis_range=[0, 1.02],
                    bargap=0.08,
                )
                _style_figure(_hist, height=320, show_legend=False)
                _chart(_hist, key="coord_histogram")

            with _scatter_col:
                _scatter = _coord.copy()
                _scatter["_signals"] = _scatter["_signals"].clip(lower=1)
                _scatter_fig = px.scatter(
                    _scatter,
                    x="_time_diff",
                    y="_similarity",
                    size="_signals",
                    color="_score",
                    color_continuous_scale=SCORE_SCALE,
                    range_color=[0, 1],
                    hover_name="account_1",
                    hover_data={
                        "account_2": True,
                        "_score": ":.3f",
                        "_time_diff": ":.1f",
                        "_similarity": ":.3f",
                        "_signals": True,
                    },
                    labels={
                        "_time_diff": "Time difference (minutes)",
                        "_similarity": "Semantic similarity",
                        "_score": "Score",
                        "_signals": "Signals",
                    },
                    title="Semantic similarity vs. timing",
                    template="plotly_dark",
                )
                _scatter_fig.update_traces(
                    marker=dict(
                        line=dict(width=1, color="rgba(226,232,240,0.35)"),
                        sizemin=8,
                        sizemode="area",
                    )
                )
                _scatter_fig.update_layout(
                    coloraxis_colorbar=dict(
                        title=dict(text="Score", font=dict(size=10)),
                        thickness=10,
                        len=0.7,
                    ),
                    yaxis=dict(
                        range=[
                            max(0.0, float(_scatter["_similarity"].min()) - 0.05),
                            min(1.05, float(_scatter["_similarity"].max()) + 0.05),
                        ]
                    ),
                )
                _style_figure(_scatter_fig, height=320, show_legend=False)
                _chart(_scatter_fig, key="coord_scatter")

            _coord_table = _coord.sort_values("_score", ascending=False).copy()
            _coord_table["semantic_similarity"] = _coord_table["_similarity"]
            _coord_table["coordination_score"] = _coord_table["_score"]
            _coord_table["time_difference_minutes"] = _coord_table["_time_diff"]
            _coord_table["shared_url"] = _coord_table["_shared_url"]
            _coord_cols = [
                column
                for column in [
                    "account_1",
                    "account_2",
                    "post_1",
                    "post_2",
                    "semantic_similarity",
                    "time_difference_minutes",
                    "shared_url",
                    "coordination_score",
                    "evidence_signals",
                ]
                if column in _coord_table.columns
            ]
            st.dataframe(
                _coord_table[_coord_cols].reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "account_1": st.column_config.TextColumn("Account 1", width="small"),
                    "account_2": st.column_config.TextColumn("Account 2", width="small"),
                    "post_1": st.column_config.TextColumn("Post 1", width="small"),
                    "post_2": st.column_config.TextColumn("Post 2", width="small"),
                    "semantic_similarity": st.column_config.ProgressColumn(
                        "Semantic similarity", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "time_difference_minutes": st.column_config.NumberColumn(
                        "Time diff (min)", format="%.1f"
                    ),
                    "shared_url": st.column_config.CheckboxColumn("Shared URL"),
                    "coordination_score": st.column_config.ProgressColumn(
                        "Coordination score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "evidence_signals": st.column_config.NumberColumn("Signals", format="%d"),
                },
            )

            with st.expander("Account coordination network", expanded=False):
                _net_left, _net_right = st.columns([1, 1.15], gap="medium")
                with _net_left:
                    if COORDINATION_GRAPH_PNG.exists():
                        st.image(
                            str(COORDINATION_GRAPH_PNG),
                            caption="Pipeline-generated network (Matplotlib, 300 dpi)",
                            use_container_width=True,
                        )
                    else:
                        st.caption(
                            "Static graph image not found. Run `python run_pipeline.py` to generate it."
                        )
                with _net_right:
                    _network_fig = _build_network_figure(_coord_df)
                    if _network_fig is None:
                        st.caption("No network edges are available to draw.")
                    else:
                        _chart(_network_fig, key="coord_network")
                        st.caption(
                            "Interactive layout computed from the pipeline GraphML file; "
                            "node size and colour scale with the number of connections."
                        )

            st.download_button(
                "Download coordination results (CSV)",
                data=_coord_table[_coord_cols].to_csv(index=False).encode("utf-8"),
                file_name="nextrace_coordination_filtered.csv",
                mime="text/csv",
                width="content",
            )

            if _demo_mode:
                st.caption(
                    "Demonstration note: five synthetic accounts reuse the same text, URL, and "
                    "1–4 minute posting window, so every qualifying pair scores 1.000. This "
                    "illustrates the scoring mechanics, not a real-world finding."
                )

# ===========================================================================
# TAB 3 — RISK SCORES
# ===========================================================================
with tab_risk:
    _section_header(
        "Investigation risk scores",
        _source_line(RISK_RESULTS_FILE)
        + " · Explainable prioritisation of flagged pairs",
    )
    st.caption(
        "Risk scores translate coordination evidence into review priority. "
        "They indicate where an analyst should look first — **not evidence of wrongdoing**."
    )

    if _risk_err:
        _render_data_error(_risk_err)
    elif _risk_df.empty:
        _empty_state(
            "No risk results",
            "Run <code>python run_pipeline.py</code> to generate explainable risk scores.",
        )
    elif (
        "coordination_score" not in _risk_df.columns
        or "investigation_priority" not in _risk_df.columns
    ):
        st.error(
            "The risk report is missing required columns "
            "(`coordination_score`, `investigation_priority`). Re-run the pipeline."
        )
    else:
        _risk = _risk_df.copy()
        _risk["_score"] = _numeric(_risk, "coordination_score").fillna(0.0)
        _risk["_similarity"] = _numeric(_risk, "semantic_similarity").fillna(0.0)
        _risk["_time_diff"] = _numeric(_risk, "time_difference_minutes")
        _risk["_signals"] = _numeric(_risk, "evidence_signals").fillna(0)
        _risk["investigation_priority"] = _risk.get(
            "investigation_priority", pd.Series("Low", index=_risk.index)
        ).astype(str)

        _risk = _risk[
            (_risk["_score"] >= coord_threshold)
            & (_risk["_similarity"] >= sim_threshold)
        ]

        with st.container(border=True):
            _rf1, _rf2 = st.columns([1.4, 2], gap="medium")
            _available_priorities = [
                priority
                for priority in PRIORITY_ORDER
                if priority in set(_risk["investigation_priority"].dropna().astype(str))
            ]
            with _rf1:
                _selected_priorities = st.multiselect(
                    "Investigation priority",
                    options=_available_priorities,
                    default=_available_priorities,
                    key="risk_priority_filter",
                )
            with _rf2:
                _risk_account_search = st.text_input(
                    "Filter by account",
                    placeholder="Account ID contains…",
                    key="risk_account_search",
                )

        if _selected_priorities:
            _risk = _risk[_risk["investigation_priority"].isin(_selected_priorities)]
        else:
            _risk = _risk.iloc[0:0]
        if _risk_account_search:
            _needle = _risk_account_search.strip().lower()
            _risk = _risk[
                _risk["account_1"].astype(str).str.lower().str.contains(_needle, regex=False)
                | _risk["account_2"].astype(str).str.lower().str.contains(_needle, regex=False)
            ]

        st.caption(
            f"Showing **{len(_risk)}** pair(s) · coordination ≥ **{coord_threshold:.2f}** "
            f"· semantic similarity ≥ **{sim_threshold:.2f}**"
        )

        if _risk.empty:
            _empty_state(
                "No risk pairs match these filters",
                "Adjust the priority or account filters, or lower the sidebar thresholds.",
            )
        else:
            _rc1, _rc2, _rc3, _rc4 = st.columns(4, gap="small")
            _priority_counts = _risk["investigation_priority"].value_counts()
            _rc1.metric("High priority", int(_priority_counts.get("High", 0)))
            _rc2.metric("Medium priority", int(_priority_counts.get("Medium", 0)))
            _rc3.metric("Low priority", int(_priority_counts.get("Low", 0)))
            _rc4.metric("Pairs shown", len(_risk))

            _priority_col, _matrix_col = st.columns([1, 1.35], gap="medium")

            with _priority_col:
                _priority_frame = pd.DataFrame(
                    {
                        "Priority": PRIORITY_ORDER,
                        "Pairs": [int(_priority_counts.get(p, 0)) for p in PRIORITY_ORDER],
                    }
                )
                _priority_fig = px.bar(
                    _priority_frame,
                    x="Priority",
                    y="Pairs",
                    color="Priority",
                    color_discrete_map=PRIORITY_COLORS,
                    text="Pairs",
                    title="Pairs by investigation priority",
                    template="plotly_dark",
                )
                _priority_fig.update_traces(textposition="outside")
                _priority_fig.update_layout(
                    showlegend=False,
                    yaxis_title="Pairs",
                    xaxis_title="",
                )
                _style_figure(_priority_fig, height=320, show_legend=False)
                _chart(_priority_fig, key="risk_priority_bar")

            with _matrix_col:
                _matrix = _risk.copy()
                _matrix["_signals_plot"] = _matrix["_signals"].clip(lower=1)
                _matrix_fig = px.scatter(
                    _matrix,
                    x="_time_diff",
                    y="_score",
                    color="investigation_priority",
                    color_discrete_map=PRIORITY_COLORS,
                    size="_signals_plot",
                    category_orders={"investigation_priority": PRIORITY_ORDER},
                    hover_name="account_1",
                    hover_data={
                        "account_2": True,
                        "_score": ":.3f",
                        "_time_diff": ":.1f",
                        "investigation_priority": True,
                    },
                    labels={
                        "_time_diff": "Time difference (minutes)",
                        "_score": "Coordination score",
                        "investigation_priority": "Priority",
                    },
                    title="Risk matrix — score vs. timing",
                    template="plotly_dark",
                )
                _matrix_fig.add_hline(
                    y=RISK_HIGH_THRESHOLD,
                    line_dash="dot",
                    line_color="rgba(248,113,113,0.65)",
                    annotation_text=f"High ≥ {RISK_HIGH_THRESHOLD:.2f}",
                    annotation_font_color="#fca5a5",
                    annotation_font_size=10,
                )
                _matrix_fig.add_hline(
                    y=RISK_MEDIUM_THRESHOLD,
                    line_dash="dot",
                    line_color="rgba(251,191,36,0.55)",
                    annotation_text=f"Medium ≥ {RISK_MEDIUM_THRESHOLD:.2f}",
                    annotation_font_color="#fcd34d",
                    annotation_font_size=10,
                )
                _matrix_fig.update_traces(
                    marker=dict(line=dict(width=1, color="rgba(226,232,240,0.35)"), sizemin=8)
                )
                _matrix_fig.update_layout(yaxis_range=[0, 1.05])
                _style_figure(_matrix_fig, height=320, show_legend=True)
                _chart(_matrix_fig, key="risk_matrix")

            _risk_table = _risk.sort_values("_score", ascending=False).copy()
            _risk_table["coordination_score"] = _risk_table["_score"]
            _risk_table["semantic_similarity"] = _risk_table["_similarity"]
            _risk_table["time_difference_minutes"] = _risk_table["_time_diff"]
            if "shared_url" in _risk_table.columns:
                _risk_table["shared_url"] = _as_bool(_risk_table["shared_url"])
            _risk_cols = [
                column
                for column in [
                    "account_1",
                    "account_2",
                    "coordination_score",
                    "investigation_priority",
                    "semantic_similarity",
                    "time_difference_minutes",
                    "shared_url",
                    "evidence",
                    "interpretation",
                ]
                if column in _risk_table.columns
            ]
            st.dataframe(
                _risk_table[_risk_cols].reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "account_1": st.column_config.TextColumn("Account 1", width="small"),
                    "account_2": st.column_config.TextColumn("Account 2", width="small"),
                    "coordination_score": st.column_config.ProgressColumn(
                        "Coordination score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "investigation_priority": st.column_config.TextColumn("Priority", width="small"),
                    "semantic_similarity": st.column_config.ProgressColumn(
                        "Semantic similarity", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "time_difference_minutes": st.column_config.NumberColumn(
                        "Time diff (min)", format="%.1f"
                    ),
                    "shared_url": st.column_config.CheckboxColumn("Shared URL"),
                    "evidence": st.column_config.TextColumn("Evidence", width="large"),
                    "interpretation": st.column_config.TextColumn("Interpretation", width="large"),
                },
            )

            _detail_limit = 25
            _detail_rows = _risk_table.head(_detail_limit)
            _section_header(
                "Evidence cards",
                f"Expanded evidence for the top {len(_detail_rows)} pair(s), sorted by score",
            )
            if len(_risk_table) > _detail_limit:
                st.caption(
                    f"Showing the first {_detail_limit} of {len(_risk_table)} pairs. "
                    "Use the filters or download the CSV for the complete list."
                )

            for _, _row in _detail_rows.iterrows():
                _priority = str(_row.get("investigation_priority", "Unknown")).strip()
                _score_text = _fmt(_row.get("_score"), 3)
                with st.expander(
                    f"{_row.get('account_1', '?')} ↔ {_row.get('account_2', '?')}  ·  "
                    f"Score {_score_text}  ·  {_priority}",
                    expanded=False,
                ):
                    st.markdown(
                        f'<div style="margin-bottom:8px">{_priority_badge(_priority)}</div>',
                        unsafe_allow_html=True,
                    )
                    _ec1, _ec2, _ec3, _ec4 = st.columns(4, gap="small")
                    _ec1.metric("Coordination score", _score_text)
                    _ec2.metric("Semantic similarity", _fmt(_row.get("_similarity"), 3))
                    _ec3.metric("Time difference", f'{_fmt(_row.get("_time_diff"), 1)} min')
                    _ec4.metric("Evidence signals", f'{_fmt(_row.get("_signals"), 0)} / 3')

                    _shared = _row.get("shared_url", "")
                    _shared_text = "Yes" if str(_shared).strip().lower() in {"true", "1", "yes"} else "No"
                    st.markdown(
                        f'<div class="nx-inline" style="color:#cbd5e1;font-size:0.84rem;margin-top:4px">'
                        f"Shared URL: <strong>{_shared_text}</strong> &nbsp;·&nbsp; "
                        f"Posts: <code>{html.escape(str(_row.get('post_1', '—')))}</code> ↔ "
                        f"<code>{html.escape(str(_row.get('post_2', '—')))}</code>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    _evidence_text = str(_row.get("evidence", "") or "")
                    if _evidence_text:
                        st.markdown("**Supporting evidence**")
                        for _part in _evidence_text.split(";"):
                            _part = _part.strip()
                            if _part:
                                st.markdown(f"- {_part}")

                    _interpretation = str(_row.get("interpretation", "") or "")
                    if _interpretation:
                        st.caption(_interpretation)

            st.download_button(
                "Download risk scores (CSV)",
                data=_risk_table[_risk_cols].to_csv(index=False).encode("utf-8"),
                file_name="nextrace_risk_scores_filtered.csv",
                mime="text/csv",
                width="content",
            )

# ===========================================================================
# TAB 4 — CAMPAIGN ANALYSIS
# ===========================================================================
with tab_campaign:
    _section_header(
        "Campaign analysis",
        _source_line(CAMPAIGN_RESULTS_FILE)
        + " · Connected components in the coordination network",
    )
    st.caption(
        "A campaign group is a connected component of accounts linked by flagged coordination pairs. "
        "The term describes a network structure; it does **not** imply a malicious campaign."
    )

    if _camp_err:
        _render_data_error(_camp_err)
    elif _camp_df.empty:
        _empty_state(
            "No campaign groups",
            "Run <code>python run_pipeline.py</code> to generate campaign analysis results.",
        )
    elif "campaign_id" not in _camp_df.columns:
        st.error("The campaign report is missing `campaign_id`. Re-run the pipeline.")
    else:
        _campaigns = _camp_df.copy()
        _campaigns["_accounts"] = pd.to_numeric(
            _campaigns.get("number_of_accounts", pd.Series(0, index=_campaigns.index)),
            errors="coerce",
        ).fillna(0)
        _campaigns["_connections"] = pd.to_numeric(
            _campaigns.get("number_of_connections", pd.Series(0, index=_campaigns.index)),
            errors="coerce",
        ).fillna(0)
        _campaigns["_avg"] = pd.to_numeric(
            _campaigns.get("average_coordination_score", pd.Series(0, index=_campaigns.index)),
            errors="coerce",
        ).fillna(0)
        _campaigns["_max"] = pd.to_numeric(
            _campaigns.get("maximum_coordination_score", pd.Series(0, index=_campaigns.index)),
            errors="coerce",
        ).fillna(0)

        _group_accounts = int(_campaigns["_accounts"].sum())
        _group_connections = int(_campaigns["_connections"].sum())
        _group_avg = _campaigns["_avg"].mean() if not _campaigns.empty else 0.0

        _cm1, _cm2, _cm3, _cm4 = st.columns(4, gap="small")
        _cm1.metric("Campaign groups", len(_campaigns))
        _cm2.metric("Grouped accounts", _group_accounts)
        _cm3.metric("Connections", _group_connections)
        _cm4.metric("Mean campaign score", _fmt(_group_avg, 3))

        with st.container(border=True):
            _cf1, _cf2 = st.columns([1.6, 1.4], gap="medium")
            _campaign_ids = _campaigns.get(
                "campaign_id", pd.Series(_campaigns.index.astype(str), index=_campaigns.index)
            ).astype(str).tolist()
            with _cf1:
                _selected_campaigns = st.multiselect(
                    "Campaign groups",
                    options=_campaign_ids,
                    default=_campaign_ids,
                    key="campaign_filter",
                )
            with _cf2:
                _campaign_sort = st.selectbox(
                    "Sort by",
                    options=[
                        "Average score (high to low)",
                        "Maximum score (high to low)",
                        "Accounts (high to low)",
                        "Campaign ID",
                    ],
                    key="campaign_sort",
                )

        _campaigns = _campaigns[_campaigns["campaign_id"].astype(str).isin(_selected_campaigns)] if "campaign_id" in _campaigns.columns else _campaigns
        _sort_map = {
            "Average score (high to low)": ("_avg", False),
            "Maximum score (high to low)": ("_max", False),
            "Accounts (high to low)": ("_accounts", False),
            "Campaign ID": ("campaign_id", True),
        }
        _sort_column, _sort_ascending = _sort_map[_campaign_sort]
        if _sort_column in _campaigns.columns:
            _campaigns = _campaigns.sort_values(_sort_column, ascending=_sort_ascending)

        if _campaigns.empty:
            _empty_state(
                "No campaign groups selected",
                "Select at least one campaign group in the filter above.",
            )
        else:
            _campaign_chart_df = _campaigns.sort_values("_accounts", ascending=True)
            _campaign_fig = px.bar(
                _campaign_chart_df,
                x="_accounts",
                y="campaign_id",
                orientation="h",
                color="_avg",
                color_continuous_scale=SCORE_SCALE,
                range_color=[0, 1],
                text="_accounts",
                title="Accounts per campaign group",
                template="plotly_dark",
                labels={
                    "_accounts": "Accounts",
                    "campaign_id": "Campaign",
                    "_avg": "Avg. score",
                },
            )
            _campaign_fig.update_traces(textposition="outside")
            _campaign_fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(text="Avg. score", font=dict(size=10)),
                    thickness=10,
                    len=0.7,
                ),
                xaxis_title="Accounts",
                yaxis_title="",
                height=min(420, max(250, 70 * len(_campaign_chart_df) + 140)),
            )
            _style_figure(
                _campaign_fig,
                height=min(420, max(250, 70 * len(_campaign_chart_df) + 140)),
                show_legend=False,
            )
            _chart(_campaign_fig, key="campaign_bar")

            _campaign_cols = [
                column
                for column in [
                    "campaign_id",
                    "number_of_accounts",
                    "number_of_connections",
                    "average_coordination_score",
                    "maximum_coordination_score",
                    "accounts",
                ]
                if column in _campaigns.columns
            ]
            st.dataframe(
                _campaigns[_campaign_cols].reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "campaign_id": st.column_config.TextColumn("Campaign", width="small"),
                    "number_of_accounts": st.column_config.NumberColumn("Accounts", format="%d"),
                    "number_of_connections": st.column_config.NumberColumn("Connections", format="%d"),
                    "average_coordination_score": st.column_config.ProgressColumn(
                        "Avg. score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "maximum_coordination_score": st.column_config.ProgressColumn(
                        "Max. score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "accounts": st.column_config.TextColumn("Member accounts", width="large"),
                },
            )

            _section_header("Campaign detail", "Member accounts and connection strength")
            for _, _row in _campaigns.iterrows():
                _campaign_id = html.escape(str(_row.get("campaign_id", "?")))
                _accounts_count = int(_row.get("_accounts", 0))
                _avg_score = _fmt(_row.get("_avg"), 3)
                with st.container(border=True):
                    _header_left, _header_right = st.columns([2, 1], gap="medium")
                    with _header_left:
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
                            f'<span class="nx-badge nx-badge-violet">{_campaign_id}</span>'
                            f'<span style="color:#94a3b8;font-size:0.8rem">'
                            f'{_accounts_count} accounts · {int(_row.get("_connections", 0))} connections'
                            f"</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(_accounts_html(_row.get("accounts", "")), unsafe_allow_html=True)
                    with _header_right:
                        _dc1, _dc2 = st.columns(2, gap="small")
                        _dc1.metric("Avg. score", _avg_score)
                        _dc2.metric("Max. score", _fmt(_row.get("_max"), 3))
                    st.progress(
                        max(0.0, min(1.0, float(_row.get("_avg", 0.0)))),
                        text=f"Average coordination score {_avg_score}",
                    )
                    st.caption(
                        "Grouping reflects shared coordination signals and requires human review. "
                        "Membership is not evidence of coordinated disinformation."
                    )

            st.download_button(
                "Download campaign results (CSV)",
                data=_campaigns[_campaign_cols].to_csv(index=False).encode("utf-8"),
                file_name="nextrace_campaigns_filtered.csv",
                mime="text/csv",
                width="content",
            )

        # -------------------------------------------------------------------
        # OSINT Narrative & Evidence Intelligence Section
        # -------------------------------------------------------------------
        st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
        _section_header(
            "OSINT & Narrative Intelligence",
            "Open-source context: GDELT global news, Google Fact Check claim reviews, and RDAP domain telemetry",
        )
        _osint_data, _osint_err = _load_json(str(OSINT_ENRICHMENT_FILE))
        if isinstance(_osint_data, dict):
            _enriched_camps = _osint_data.get("enriched_campaigns", [])
            if _enriched_camps:
                for _c_idx, _c_bundle in enumerate(_enriched_camps, 1):
                    _cid = _c_bundle.get("target_id", f"Campaign {_c_idx}")
                    with st.expander(f"OSINT Evidence & External Attribution for {_cid}", expanded=True):
                        _oc1, _oc2 = st.columns(2, gap="medium")
                        with _oc1:
                            st.markdown("##### 📰 GDELT Global News Coverage")
                            _news = _c_bundle.get("matched_news", [])
                            if _news:
                                for _n in _news[:5]:
                                    _n_title = html.escape(_n.get("title", "Untitled"))
                                    _n_url = html.escape(_n.get("url", "#"))
                                    _n_dom = html.escape(_n.get("domain", "unknown"))
                                    _n_date = _n.get("seen_date", "")[:10]
                                    st.markdown(
                                        f"- <a href='{_n_url}' target='_blank' style='color:#22d3ee;font-weight:600'>"
                                        f"{_n_title}</a> <span style='color:#94a3b8;font-size:0.8rem'>({_n_dom} · {_n_date})</span>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("No matching global news articles found for campaign query keywords.")

                        with _oc2:
                            st.markdown("##### 🛡️ Third-Party Fact Checks")
                            _fc = _c_bundle.get("matched_fact_checks", [])
                            if _fc:
                                for _f in _fc[:5]:
                                    _f_claim = html.escape(_f.get("claim_text", ""))
                                    _f_pub = html.escape(_f.get("publisher_name", "Fact Checker"))
                                    _f_rating = html.escape(_f.get("textual_rating", "Reviewed"))
                                    _f_url = html.escape(_f.get("review_url", "#"))
                                    st.markdown(
                                        f"<div style='background:rgba(30,41,59,0.7);padding:8px;border-radius:6px;margin-bottom:6px;border-left:3px solid #fbbf24'>"
                                        f"<div style='font-size:0.82rem;color:#e2e8f0'><strong>Claim:</strong> {_f_claim}</div>"
                                        f"<div style='font-size:0.75rem;color:#94a3b8;margin-top:4px'>"
                                        f"Rating: <strong style='color:#fbbf24'>{_f_rating}</strong> by {_f_pub} · "
                                        f"<a href='{_f_url}' target='_blank' style='color:#22d3ee'>Read Review ↗</a>"
                                        f"</div></div>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("No published fact checks found for campaign narrative phrases.")

                        # Domains
                        _domains = _c_bundle.get("domain_investigations", [])
                        if _domains:
                            st.markdown("##### 🌐 RDAP Domain Infrastructure Intelligence")
                            _dom_rows = []
                            for _d in _domains:
                                _dom_rows.append({
                                    "Domain": _d.get("domain"),
                                    "Registrar": _d.get("registrar") or "Unknown / Redacted",
                                    "Registered": _d.get("registration_date")[:10] if _d.get("registration_date") else "Unknown",
                                    "Status": ", ".join(_d.get("status", []))[:30] if _d.get("status") else "Active",
                                })
                            st.dataframe(pd.DataFrame(_dom_rows), hide_index=True, use_container_width=True)

                        st.caption(
                            "Critical Interpretation Note: Absence of a fact-check does NOT indicate a claim is verified or true. "
                            "Third-party publisher ratings belong exclusively to the source fact-checker and are not NexTrace conclusions."
                        )
            else:
                st.info("No campaign-level OSINT bundles recorded in current pipeline run.")
        else:
            st.info(
                "OSINT enrichment results file not found. Re-run `python run_pipeline.py` to generate OSINT intelligence."
            )

        # -------------------------------------------------------------------
        # Manual OSINT Evidence Recording (Analyst Workspace - No APIs Required)
        # -------------------------------------------------------------------
        st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
        _section_header(
            "Manual OSINT Investigation & Evidence Recording",
            "Analyst workflow without APIs: Research claims on public sources and record verifiable evidence.",
        )

        _manual_records = []
        if MANUAL_OSINT_FILE.exists():
            try:
                with open(MANUAL_OSINT_FILE, "r", encoding="utf-8") as _mf:
                    _manual_records = json.load(_mf)
                if not isinstance(_manual_records, list):
                    _manual_records = []
            except Exception:
                _manual_records = []

        with st.expander("📝 Record New Manual OSINT Evidence", expanded=False):
            with st.form("manual_osint_form", clear_on_submit=True):
                _m_c1, _m_c2 = st.columns(2, gap="medium")
                with _m_c1:
                    _m_claim = st.text_input("Investigated Keyword or Claim*", placeholder="e.g. 5G towers cause radiation poisoning")
                    _m_title = st.text_input("Source Article / Fact-Check Title*", placeholder="e.g. Fact Check: False claims about 5G")
                    _m_url = st.text_input("Source URL*", placeholder="https://www.reuters.com/fact-check/...")
                with _m_c2:
                    _m_pub_date = st.text_input("Publication Date", placeholder="e.g. 2024-03-15")
                    _target_options = ["None"]
                    if not _camp_df.empty and "campaign_id" in _camp_df.columns:
                        _target_options += [f"Campaign: {cid}" for cid in _camp_df["campaign_id"].dropna().unique()]
                    if not _posts_df.empty and "post_id" in _posts_df.columns:
                        _target_options += [f"Post: {pid}" for pid in _posts_df["post_id"].dropna().unique()[:20]]
                    _m_assoc = st.selectbox("Associate Evidence with Target", options=_target_options)
                    _m_notes = st.text_area("Analyst Investigation Notes", placeholder="e.g. Originates from known disinfo cluster; debunked by IFCN signatory.", height=80)

                _submitted = st.form_submit_button("Save Manual Evidence", use_container_width=True)
                if _submitted:
                    if not _m_claim.strip() or not _m_url.strip() or not _m_title.strip():
                        st.error("Please provide at least the Claim, Source URL, and Source Title.")
                    else:
                        _new_rec = {
                            "query_claim": _m_claim.strip(),
                            "source_title": _m_title.strip(),
                            "source_url": _m_url.strip(),
                            "publication_date": _m_pub_date.strip(),
                            "collection_time": datetime.now(timezone.utc).isoformat(),
                            "relevant_notes": _m_notes.strip(),
                            "associated_target": _m_assoc,
                            "provenance": "Manual Analyst Investigation"
                        }
                        _manual_records.append(_new_rec)
                        MANUAL_OSINT_FILE.parent.mkdir(parents=True, exist_ok=True)
                        with open(MANUAL_OSINT_FILE, "w", encoding="utf-8") as _mf:
                            json.dump(_manual_records, _mf, indent=2)
                        st.success("Manual OSINT evidence saved successfully!")
                        st.rerun()

        if _manual_records:
            st.markdown(f"**Recorded Manual Evidence ({len(_manual_records)} records)**")
            _m_rows = []
            for _rec in reversed(_manual_records):
                _m_rows.append({
                    "Target": _rec.get("associated_target", "None"),
                    "Claim / Keyword": _rec.get("query_claim", ""),
                    "Source Title": _rec.get("source_title", ""),
                    "Source URL": _rec.get("source_url", ""),
                    "Pub Date": _rec.get("publication_date", ""),
                    "Collected (UTC)": _rec.get("collection_time", "")[:19].replace("T", " "),
                    "Analyst Notes": _rec.get("relevant_notes", ""),
                })
            st.dataframe(pd.DataFrame(_m_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No manual OSINT evidence recorded yet. Use the form above to record verifiable public evidence.")


# ===========================================================================
# TAB 5 — BOT CLASSIFIER
# ===========================================================================
with tab_bot:
    _section_header(
        "Bot / genuine account classifier",
        "Phase 2 · Trained Random Forest applied to account-level feature vectors",
    )
    st.caption(
        "This classifier is separate from Phase 1 pair coordination. It was trained on the "
        "Cresci-2015 and Cresci-2017 research datasets and identifies **bot-like or genuine-like "
        "feature patterns**; it does not determine an account's true nature. Human review is required."
    )

    _bundle, _bundle_err = _load_model_bundle()
    _bundle = _bundle if isinstance(_bundle, dict) else None
    _feature_names = _bundle.get("feature_names", []) if _bundle else []

    if _metadata_err:
        st.warning(f"Model metadata unavailable: {_metadata_err}", icon=":material/warning:")

    _training_accounts_raw = (
        _metadata.get("dataset_provenance", {}).get("n_accounts", 0) if _metadata else 0
    )
    try:
        _training_accounts = f"{int(_training_accounts_raw):,}"
    except (TypeError, ValueError):
        _training_accounts = "n/a"

    _model_cols = st.columns(4, gap="small")
    _model_cols[0].metric(
        "Algorithm",
        str(_metadata.get("model_name", "Random forest")).replace("_", " ").title(),
        help="Selected after validation comparison against logistic regression and decision tree.",
    )
    _model_cols[1].metric("Test F1", _fmt(_test_metrics.get("f1_score"), 4, "n/a"))
    _model_cols[2].metric("Test ROC-AUC", _fmt(_test_metrics.get("roc_auc"), 4, "n/a"))
    _model_cols[3].metric("Training accounts", _training_accounts)

    with st.expander("Model performance and evaluation", expanded=False):
        if _comparison_err:
            st.warning(_comparison_err, icon=":material/warning:")
        elif _comparison_df.empty:
            st.caption("No model comparison report is available.")
        else:
            _comparison = _comparison_df.copy()
            _selected_name = str(_metadata.get("selected_model", "random_forest"))
            _selected_colors = [
                VIOLET if str(name) == _selected_name else CYAN_DEEP
                for name in _comparison.get("model", [])
            ]
            _comparison_fig = go.Figure()
            _comparison_fig.add_bar(
                x=_comparison.get("model", []),
                y=pd.to_numeric(
                    _comparison.get("f1_score", pd.Series(dtype=float)), errors="coerce"
                ),
                name="F1 score",
                marker_color=_selected_colors,
                text=[
                    _fmt(value, 3)
                    for value in _comparison.get("f1_score", [])
                ],
                textposition="outside",
            )
            _comparison_fig.add_bar(
                x=_comparison.get("model", []),
                y=pd.to_numeric(
                    _comparison.get("roc_auc", pd.Series(dtype=float)), errors="coerce"
                ),
                name="ROC-AUC",
                marker_color="rgba(34,211,238,0.38)",
                marker_line=dict(color=CYAN, width=1),
                text=[
                    _fmt(value, 3)
                    for value in _comparison.get("roc_auc", [])
                ],
                textposition="outside",
            )
            _comparison_fig.update_layout(
                barmode="group",
                title="Validation metrics by candidate model",
                xaxis_title="",
                yaxis_title="Score",
                yaxis_range=[0, 1.12],
            )
            _style_figure(_comparison_fig, height=330, show_legend=True)
            _chart(_comparison_fig, key="ml_comparison_chart")

            _comparison_table = _comparison.rename(
                columns={
                    "model": "Model",
                    "split": "Split",
                    "n_samples": "Samples",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1",
                    "roc_auc": "ROC-AUC",
                }
            )
            st.dataframe(
                _comparison_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                    "Precision": st.column_config.NumberColumn(format="%.4f"),
                    "Recall": st.column_config.NumberColumn(format="%.4f"),
                    "F1": st.column_config.NumberColumn(format="%.4f"),
                    "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        if isinstance(_evaluation, dict):
            _test_results = _evaluation.get("test_results_selected_model", {})
            _confusion = _test_results.get("confusion_matrix")
            if isinstance(_confusion, list) and len(_confusion) == 2:
                _cm_col, _split_col = st.columns([1.1, 1], gap="medium")
                with _cm_col:
                    _annotations = []
                    for row_index, row_values in enumerate(_confusion):
                        for col_index, value in enumerate(row_values):
                            _annotations.append(
                                dict(
                                    x=col_index,
                                    y=row_index,
                                    text=f"{int(value):,}",
                                    showarrow=False,
                                    font=dict(color="#f1f5f9", size=15),
                                )
                            )
                    _cm_fig = go.Figure(
                        go.Heatmap(
                            z=_confusion,
                            x=["Genuine (0)", "Bot (1)"],
                            y=["Genuine (0)", "Bot (1)"],
                            colorscale=[
                                [0.0, "#0b1220"],
                                [0.45, "#0e7490"],
                                [1.0, VIOLET_DEEP],
                            ],
                            showscale=False,
                            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z:,}<extra></extra>",
                        )
                    )
                    _cm_fig.update_layout(
                        title="Held-out test confusion matrix",
                        xaxis_title="Predicted label",
                        yaxis_title="Actual label",
                        annotations=_annotations,
                    )
                    _style_figure(_cm_fig, height=320, show_legend=False)
                    _chart(_cm_fig, key="ml_confusion_matrix")
                with _split_col:
                    _split_frame = pd.DataFrame(
                        {
                            "Split": ["Train", "Validation", "Test"],
                            "Accounts": [
                                int(_evaluation.get("n_train", 0)),
                                int(_evaluation.get("n_validation", 0)),
                                int(_evaluation.get("n_test", 0)),
                            ],
                        }
                    )
                    _split_fig = px.bar(
                        _split_frame,
                        x="Split",
                        y="Accounts",
                        text="Accounts",
                        title="Held-out evaluation split",
                        template="plotly_dark",
                        color="Split",
                        color_discrete_sequence=[CYAN, VIOLET, "#e879f9"],
                    )
                    _split_fig.update_traces(textposition="outside")
                    _split_fig.update_layout(showlegend=False, yaxis_title="Accounts", xaxis_title="")
                    _style_figure(_split_fig, height=320, show_legend=False)
                    _chart(_split_fig, key="ml_split_chart")
                    st.caption(
                        f"Selection rule: {str(_evaluation.get('selection_rule', 'n/a')).replace('_', ' ')}. "
                        f"Test accuracy: {_fmt(_test_results.get('accuracy'), 4)}."
                    )

        if _bundle and _feature_names:
            _importance = _feature_importance_frame(_bundle)
            if _importance is not None:
                _importance_fig = px.bar(
                    _importance.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Random Forest feature importance (top 12)",
                    template="plotly_dark",
                    color="importance",
                    color_continuous_scale=[[0.0, CYAN_DEEP], [1.0, VIOLET]],
                    text=[f"{value:.3f}" for value in _importance.sort_values("importance", ascending=True)["importance"]],
                )
                _importance_fig.update_layout(
                    coloraxis_showscale=False,
                    xaxis_title="Relative importance",
                    yaxis_title="",
                )
                _importance_fig.update_traces(textposition="outside")
                _style_figure(_importance_fig, height=390, show_legend=False)
                _chart(_importance_fig, key="ml_feature_importance")
                st.caption(
                    "Impurity-based importances from the trained pipeline. Correlated profile "
                    "counters can share or split importance."
                )
            if isinstance(_explanations, list) and _explanations:
                _explanations_fig_data = _explanations
                _option_labels = [
                    f"{item.get('account_id', '?')} · true label "
                    f"{'Bot' if int(item.get('true_label', 0)) == 1 else 'Genuine'}"
                    f" · {item.get('source_group', 'unknown')}"
                    for item in _explanations_fig_data
                ]
                _selected_explanation = st.selectbox(
                    "Pre-computed explanation for a held-out test account",
                    options=_option_labels,
                    key="ml_explanation_account",
                )
                _explanation_index = _option_labels.index(_selected_explanation)
                _explanation = _explanations_fig_data[_explanation_index]
                _shap = _explanation.get("shap", {}) if isinstance(_explanation, dict) else {}
                _lime = _explanation.get("lime", {}) if isinstance(_explanation, dict) else {}

                _shap_col, _lime_col = st.columns(2, gap="medium")
                with _shap_col:
                    _shap_frame = _top_attributions(_shap.get("top_features"), limit=8)
                    if _shap_frame.empty:
                        st.caption("No SHAP attributions available for this account.")
                    else:
                        _shap_fig = _attribution_figure(
                            _shap_frame, "SHAP attribution (pre-computed)"
                        )
                        if _shap_fig is not None:
                            _chart(_shap_fig, key="ml_shap")
                with _lime_col:
                    _lime_frame = _top_attributions(
                        list((_lime.get("feature_attributions") or {}).items()), limit=8
                    )
                    if _lime_frame.empty:
                        st.caption("No LIME attributions available for this account.")
                    else:
                        _lime_fig = _attribution_figure(
                            _lime_frame, "LIME attribution (pre-computed)"
                        )
                        if _lime_fig is not None:
                            _chart(_lime_fig, key="ml_lime")
                st.caption(
                    "Explanations were pre-computed on held-out Cresci test accounts. They are "
                    "unrelated to the demonstration posts and do not describe any real person."
                )
    if not _classification_df.empty:
        st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
        _section_header(
            "Active Dataset Accounts — Bot Classification",
            "Automatic inference on accounts from the current pipeline run using the trained Random Forest model",
        )
        st.dataframe(
            _classification_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "⚠️ Resemblance indicator only: Scores reflect similarity to historical Cresci training accounts. "
            "A high bot probability is NOT proof of automated or malicious intent; human review is required."
        )

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
    _section_header(
        "Classify an account (Interactive)",
        "Provide a complete feature vector. Presets are illustrative examples, not real accounts.",
    )

    _presets: dict[str, dict[str, float] | None] = {
        "Custom values": None,
        "Genuine example": {
            "total_posts": 120,
            "posts_with_urls": 30,
            "unique_texts": 115,
            "duplicate_text_posts": 5,
            "average_text_length": 80,
            "average_word_count": 14,
            "average_time_between_posts_minutes": 720,
            "url_post_ratio": 0.25,
            "duplicate_text_ratio": 0.04,
            "statuses_count": 120,
            "followers_count": 350,
            "friends_count": 280,
            "favourites_count": 900,
            "listed_count": 4,
            "default_profile": 0,
            "geo_enabled": 1,
            "verified": 0,
            "protected": 0,
            "description_length": 85,
            "friends_followers_ratio": 0.80,
            "tweet_data_available": 1,
        },
        "Bot-like example": {
            "total_posts": 1800,
            "posts_with_urls": 1700,
            "unique_texts": 20,
            "duplicate_text_posts": 1780,
            "average_text_length": 30,
            "average_word_count": 5,
            "average_time_between_posts_minutes": 1.2,
            "url_post_ratio": 0.94,
            "duplicate_text_ratio": 0.99,
            "statuses_count": 1800,
            "followers_count": 12,
            "friends_count": 2000,
            "favourites_count": 5,
            "listed_count": 0,
            "default_profile": 1,
            "geo_enabled": 0,
            "verified": 0,
            "protected": 0,
            "description_length": 0,
            "friends_followers_ratio": 166.7,
            "tweet_data_available": 1,
        },
    }

    _preset_choice = st.selectbox(
        "Starting point",
        options=list(_presets.keys()),
        key="bot_preset",
        help="Selecting a preset changes the input defaults. You can edit any value before classifying.",
    )
    if st.session_state.get("nx_bot_last_preset") != _preset_choice:
        st.session_state["nx_bot_last_preset"] = _preset_choice
        st.session_state.pop("nx_bot_result", None)

    if _bundle is None:
        st.error(
            f"The trained model could not be loaded: {_bundle_err or 'unknown error'}",
            icon=":material/error:",
        )
        st.info(
            "Confirm that `models/selected_bot_classifier.joblib` exists and that the `.venv` "
            "environment is active.",
            icon=":material/info:",
        )
    elif not _feature_names:
        st.error(
            "The model bundle does not declare its feature names. Re-check "
            "`models/selected_bot_classifier.joblib` and its metadata."
        )
    else:
        _preset_values = _presets[_preset_choice] or {}
        _preset_slug = _preset_choice.lower().replace(" ", "_").replace("-", "_")
        _known_features = {
            feature
            for group in BOT_FEATURE_GROUPS.values()
            for feature in group
        }
        _extra_features = [
            feature for feature in _feature_names if feature not in _known_features
        ]
        _feature_values: dict[str, float] = {}

        with st.form("account_classifier_form", border=True):
            _posted_values = _render_feature_group(
                "Posting behaviour",
                BOT_FEATURE_GROUPS["Posting behaviour"],
                _preset_values,
                _preset_slug,
                _feature_names,
            )
            _profile_values = _render_feature_group(
                "Profile and account signals",
                BOT_FEATURE_GROUPS["Profile and account signals"],
                _preset_values,
                _preset_slug,
                _feature_names,
            )
            _extra_values = _render_feature_group(
                "Additional model features",
                _extra_features,
                _preset_values,
                _preset_slug,
                _feature_names,
            )
            _feature_values = {**_posted_values, **_profile_values, **_extra_values}
            _submitted = st.form_submit_button(
                "Classify account",
                type="primary",
                width="content",
            )

        if _submitted:
            if len(_feature_values) != len(_feature_names):
                st.error(
                    f"Expected {len(_feature_names)} feature values but collected "
                    f"{len(_feature_values)}. Refresh the page and try again."
                )
            else:
                with st.spinner("Scoring account features with the trained Random Forest…"):
                    try:
                        from src.ml.predict_account import predict_account

                        _prediction = predict_account(_feature_values)
                        st.session_state["nx_bot_result"] = {
                            "prediction": _prediction,
                            "features": dict(_feature_values),
                            "preset": _preset_choice,
                            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        }
                    except Exception as exc:  # noqa: BLE001
                        st.error(
                            f"Classification failed: {type(exc).__name__}: {exc}",
                            icon=":material/error:",
                        )
                        with st.expander("Error details", expanded=False):
                            st.code(traceback.format_exc(), language="python")

        _stored_result = st.session_state.get("nx_bot_result")
        if _stored_result and _stored_result.get("preset") == _preset_choice:
            _render_prediction_result(_stored_result)

    with st.expander("Reference: observed posting behaviour in the demonstration data", expanded=False):
        if _behavior_err:
            st.caption(_behavior_err)
        elif _behavior_df.empty:
            st.caption("No account behaviour report is available.")
        else:
            st.caption(
                "These six posting features come from the 10-post demonstration dataset. "
                "Profile fields (followers, statuses, favourites) are not present in the demo "
                "data, so no demo accounts are scored automatically."
            )
            st.dataframe(
                _behavior_df.reset_index(drop=True),
                width="stretch",
                hide_index=True,
                column_config={
                    "account_id": st.column_config.TextColumn("Account", width="small"),
                    "total_posts": st.column_config.NumberColumn("Posts", format="%d"),
                    "unique_texts": st.column_config.NumberColumn("Unique texts", format="%d"),
                    "urls_shared": st.column_config.NumberColumn("URLs shared", format="%d"),
                    "duplicate_text_posts": st.column_config.NumberColumn("Duplicates", format="%d"),
                    "average_time_between_posts_minutes": st.column_config.NumberColumn(
                        "Avg. gap (min)", format="%.1f"
                    ),
                },
            )

    st.caption(
        "Model disclaimer: predictions describe statistical similarity to labeled accounts in the "
        "Cresci research datasets. They are not determinations about any real person or account and "
        "must not be used as the sole basis for moderation, enforcement, or public claims."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)
_source_note = (
    f"Demonstration dataset: {_post_count} synthetic posts."
    if _demo_mode
    else f"Current dataset: {_post_count} posts."
)
st.markdown(
    f'<div class="nx-footer">NexTrace · Social Media Coordinated Disinformation Detection System<br>'
    f"{html.escape(_source_note)} · Investigation leads only · Human review required before any conclusions."
    f"</div>",
    unsafe_allow_html=True,
)
# ===========================================================================
# TAB 6 — CRESCI DATASET ANALYSIS
# ===========================================================================

with tab_cresci:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <h1 style="color: #22d3ee;">Cresci Dataset Analysis</h1>
            <p style="color: #94a3b8;">
                Research dataset analysis of account relationships, similar posts,
                candidate coordination clusters, and network structure.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cresci_report_dir = _PROJECT_ROOT / "results" / "reports"
    cresci_graph_path = (
        _PROJECT_ROOT / "results" / "graphs" / "cresci_coordination_graph.png"
    )

    cresci_files = {
        "Semantic Similarity": "cresci_semantic_similarity_results.csv",
        "Coordination Results": "cresci_coordination_results.csv",
        "Account Analysis": "cresci_account_analysis.csv",
        "Campaign Results": "cresci_campaign_results.csv",
        "Campaign Members": "cresci_campaign_members.csv",
        "Network Metrics": "cresci_network_metrics.csv",
        "Campaign Validation": "cresci_campaign_validation.csv",
    }

    @st.cache_data(show_spinner=False)
    def _load_cresci_report(path_string):
        path = Path(path_string)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    cresci_data = {}

    for report_name, filename in cresci_files.items():
        report_path = cresci_report_dir / filename
        try:
            cresci_data[report_name] = _load_cresci_report(str(report_path))
        except Exception as exc:
            cresci_data[report_name] = pd.DataFrame()
            st.warning(
                f"Could not load {filename}: {type(exc).__name__}: {exc}"
            )

    cresci_posts_path = (
        _PROJECT_ROOT
        / "data"
        / "processed"
        / "cresci_posts_multisource_cleaned.csv"
    )

    try:
        cresci_posts = _load_cresci_report(str(cresci_posts_path))
    except Exception as exc:
        cresci_posts = pd.DataFrame()
        st.warning(f"Could not load Cresci posts: {exc}")

    cresci_coord = cresci_data["Coordination Results"]
    cresci_accounts = cresci_data["Account Analysis"]
    cresci_campaigns = cresci_data["Campaign Results"]
    cresci_members = cresci_data["Campaign Members"]
    cresci_network = cresci_data["Network Metrics"]
    cresci_validation = cresci_data["Campaign Validation"]
    cresci_similarity = cresci_data["Semantic Similarity"]

    if cresci_posts.empty:
        st.warning(
            "The Cresci cleaned dataset was not found or is empty. "
            "Check data/processed/cresci_posts_multisource_cleaned.csv."
        )
    else:
        post_count = len(cresci_posts)
        account_count = (
            cresci_posts["account_id"].nunique()
            if "account_id" in cresci_posts.columns
            else 0
        )
        source_count = (
            cresci_posts["source_group"].nunique()
            if "source_group" in cresci_posts.columns
            else 0
        )

        st.markdown("### Dataset Overview")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Posts", f"{post_count:,}")
        k2.metric("Accounts", f"{account_count:,}")
        k3.metric("Source Groups", f"{source_count:,}")
        k4.metric("Similarity Candidates", f"{len(cresci_similarity):,}")

        st.caption(
            "These counts describe the currently loaded Cresci sample, "
            "not the complete original Cresci datasets."
        )

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Coordination Detection")

    flagged_pairs = len(cresci_coord)

    unique_edges = 0
    if not cresci_coord.empty and {
        "account_1",
        "account_2",
    }.issubset(cresci_coord.columns):
        unique_edges = cresci_coord.apply(
            lambda row: tuple(
                sorted(
                    [
                        str(row["account_1"]),
                        str(row["account_2"]),
                    ]
                )
            ),
            axis=1,
        ).nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged Post Pairs", f"{flagged_pairs:,}")
    c2.metric("Unique Account Connections", f"{unique_edges:,}")
    c3.metric("Candidate Campaigns", f"{len(cresci_campaigns):,}")

    if not cresci_coord.empty:
        display_coord = cresci_coord.copy()

        sort_column = (
            "coordination_score"
            if "coordination_score" in display_coord.columns
            else None
        )

        if sort_column:
            display_coord[sort_column] = pd.to_numeric(
                display_coord[sort_column], errors="coerce"
            )
            display_coord = display_coord.sort_values(
                sort_column, ascending=False
            )

        st.dataframe(
            display_coord,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No coordination results are available.")

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Candidate Campaign Analysis")

    if not cresci_campaigns.empty:
        st.dataframe(
            cresci_campaigns,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No campaign results are available.")

    st.markdown("### Campaign Validation")

    if not cresci_validation.empty:
        st.dataframe(
            cresci_validation,
            use_container_width=True,
            hide_index=True,
        )

        if "validation_status" in cresci_validation.columns:
            st.caption(
                "Validation statuses are evidence-review indicators. "
                "They do not establish that a campaign conducted disinformation."
            )
    else:
        st.info("No campaign validation report is available.")

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Account Analysis")

    if not cresci_accounts.empty:
        if "flagged_pair_count" in cresci_accounts.columns:
            account_display = cresci_accounts.sort_values(
                "flagged_pair_count", ascending=False
            )
        else:
            account_display = cresci_accounts

        st.dataframe(
            account_display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No account analysis report is available.")

    st.markdown("### Network Metrics")

    if not cresci_network.empty:
        if "degree_centrality" in cresci_network.columns:
            network_display = cresci_network.sort_values(
                "degree_centrality", ascending=False
            )
        else:
            network_display = cresci_network

        st.dataframe(
            network_display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No network metrics report is available.")

    st.markdown('<div class="nx-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Coordination Network Graph")

    if cresci_graph_path.exists():
        st.image(
            str(cresci_graph_path),
            caption=(
                "Cresci coordination network. Connections represent candidate "
                "coordination evidence, not proof of malicious activity."
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "Network graph image not found. Check "
            "results/graphs/cresci_coordination_graph.png."
        )

    st.markdown("### Export Reports")

    for report_name, filename in cresci_files.items():
        report_path = cresci_report_dir / filename

        if report_path.exists():
            try:
                report_bytes = report_path.read_bytes()

                st.download_button(
                    label=f"Download {report_name} CSV",
                    data=report_bytes,
                    file_name=filename,
                    mime="text/csv",
                    key=f"cresci_download_{filename}",
                )
            except OSError as exc:
                st.warning(f"Could not prepare {filename}: {exc}")

    if cresci_posts_path.exists():
        try:
            st.download_button(
                label="Download Cleaned Cresci Posts CSV",
                data=cresci_posts_path.read_bytes(),
                file_name="cresci_posts_multisource_cleaned.csv",
                mime="text/csv",
                key="cresci_download_cleaned_posts",
            )
        except OSError as exc:
            st.warning(f"Could not prepare cleaned Cresci posts: {exc}")

    st.info(
        "**Research disclaimer:** Similarity, coordination scores, graph connections, "
        "and classifier outputs are investigative signals. They are not proof of "
        "coordinated disinformation or the intent of any account. Human review "
        "and independent evidence are required."
    )
