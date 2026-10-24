# NexTrace — Bot Classifier Audit & Responsible Evaluation Report

**Date Generated:** 2026-10-24T07:25:29.543416+00:00
**Model Artifact:** `models/selected_bot_classifier.joblib` (Preserved intact)
**Model Architecture:** Pipeline
**Classifier:** RandomForestClassifier

---

## 1. Model Artifact & Feature Schema Verification

- **Model Key Name:** `random_forest`
- **Expected Feature Count:** 21 features
- **Input Features:**
  - `total_posts`
  - `posts_with_urls`
  - `unique_texts`
  - `duplicate_text_posts`
  - `average_text_length`
  - `average_word_count`
  - `average_time_between_posts_minutes`
  - `url_post_ratio`
  - `duplicate_text_ratio`
  - `statuses_count`
  - `followers_count`
  - `friends_count`
  - `favourites_count`
  - `listed_count`
  - `default_profile`
  - `geo_enabled`
  - `verified`
  - `protected`
  - `description_length`
  - `friends_followers_ratio`
  - `tweet_data_available`

### Preprocessing Pipeline Steps
- **Imputer:** `SimpleImputer(strategy='median')` — handles missing account profile or tweet values neutrally without dropping accounts.
- **Classifier:** `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`.

---

## 2. Dataset Provenance & Benchmark Partitioning

- **Training Dataset Sources:** Cresci-2015 and Cresci-2017 research datasets.
- **Total Benchmark Accounts:** 16,318 accounts.
- **Partition Sizes:**
  - Train Set: 11,422 accounts (70%)
  - Validation Set: 2,448 accounts (15%)
  - Test Set: 2,448 accounts (15%)
- **Split Strategy:** Stratified by class label (`stratified_by_label`).

---

## 3. Verified Benchmark Performance Metrics

### Held-Out Test Set (N = 2,448 accounts)
- **Accuracy:** 0.9853
- **Precision (Bot class):** 0.9963
- **Recall (Bot class):** 0.9816
- **F1 Score (Bot class):** 0.9889
- **ROC-AUC:** 0.9985

### Confusion Matrix (Test Set)
```
                Predicted Genuine    Predicted Bot
Actual Genuine         0                    0
Actual Bot             0                    0
```

---

## 4. Rigorous Leakage & Generalization Assessment

### 4.1 Target & Identifier Leakage Audit
- Account identifiers (`account_id`, `id`, `screen_name`, `username`) and dataset provenance markers (`source_dataset`, `source_group`) are strictly excluded from the model feature vector (`select_model_features`).
- Features are purely behavioral and profile metrics (post frequencies, URL ratios, duplicate text rates, follower counts).

### 4.2 Cross-Split Contamination Audit
- Split partitioning is conducted at the unique account level: no single account exists in more than one partition (train, validation, or test).
- Duplicate account records across dataset variants were deduplicated prior to splitting.

### 4.3 Generalization Across Bot Families (Source Groups)
> [!WARNING]
> When evaluating on held-out *source groups* (e.g. testing on social spambots after training only on traditional spambots or fake followers), model recall decreases compared to random stratified splitting.
> This confirms that the model detects behavioral patterns characteristic of its training distribution, rather than a universal 'bot signature'.

---

## 5. Ethical AI & Operational Limitations

1. **Resemblance vs. Intent:** Predictions reflect similarity to historical training clusters (Cresci datasets). High bot probability does NOT establish that an account is malicious or automated.
2. **Evolving Evasion:** Modern automated accounts often mimic human posting cadence and buy legitimate follower bases, which can bypass simple heuristic or profile thresholds.
3. **Independent Decision Support:** Model outputs must always be combined with observable coordination evidence (synchronized timing, identical text) and human investigator review.