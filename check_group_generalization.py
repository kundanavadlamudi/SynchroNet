
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "ml_dataset.csv"

df = pd.read_csv(DATA_FILE)

feature_columns = [
    "total_posts",
    "posts_with_urls",
    "unique_texts",
    "duplicate_text_posts",
    "average_text_length",
    "average_word_count",
    "average_time_between_posts_minutes",
    "url_post_ratio",
    "duplicate_text_ratio",
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
]

feature_columns = [
    col for col in feature_columns if col in df.columns
]

X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
X = X.drop(
    columns=["verified", "protected"],
    errors="ignore"
)
y = df["label"].astype(int)
groups = df["source_group"]

print("Accounts:", len(df))
print("Features:", len(feature_columns))
print("Source groups:", groups.nunique())
print("\nAccounts per source group:")
print(df.groupby(["source_group", "label"]).size())

if groups.nunique() < 3:
    raise SystemExit("Not enough source groups for grouped evaluation.")

# Every source group is kept in a single fold.
cv = GroupKFold(n_splits=min(5, groups.nunique()))

# Produce out-of-fold predictions; no model predicts accounts from a
# source group that it trained on in the corresponding fold.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

predictions = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    method="predict",
    n_jobs=1,
)

print("\n========== GROUPED EVALUATION ==========")
print("Accuracy:", round(accuracy_score(y, predictions), 4))
print("\nConfusion matrix (labels: Genuine=0, Bot=1):")
print(confusion_matrix(y, predictions, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y,
    predictions,
    labels=[0, 1],
    target_names=["Genuine", "Bot"],
    zero_division=0,
))

print("\nNote: These results evaluate generalization to held-out source groups.")
print("They are not directly comparable to the previous random-split scores.")

from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    train_groups = sorted(groups.iloc[train_idx].unique())
    test_groups = sorted(groups.iloc[test_idx].unique())

    print(f"\nFold {fold}")
    print("Training groups:", train_groups)
    print("Testing groups:", test_groups)
    print("Training labels:", y.iloc[train_idx].value_counts().to_dict())
    print("Testing labels:", y.iloc[test_idx].value_counts().to_dict())
