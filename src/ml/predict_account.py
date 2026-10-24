"""Score account-level features with a fitted bot classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.ml.features import select_model_features
from src.ml.prepare_dataset import LABEL_MAPPING, label_name_for


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "selected_bot_classifier.joblib"
METADATA_FILE = MODEL_DIR / "selected_bot_classifier_metadata.json"


class ModelNotAvailableError(FileNotFoundError):
    """Raised when no validated trained model is present."""


def load_model_bundle(model_dir: Path | None = None) -> dict:
    directory = Path(model_dir or MODEL_DIR)
    model_file = directory / MODEL_FILE.name
    metadata_file = directory / METADATA_FILE.name
    if not model_file.exists():
        raise ModelNotAvailableError(
            "No trained bot classifier found. Train a model after placing "
            "labeled Cresci-2015/2017 data under data/raw."
        )
    bundle = joblib.load(model_file)
    metadata = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    bundle["metadata"] = metadata
    bundle["model_file"] = str(model_file)
    return bundle


def validate_feature_columns(
    features: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    missing = [column for column in expected_columns if column not in features.columns]
    if missing:
        raise ValueError(
            "Incoming features are missing required columns: "
            + ", ".join(missing)
        )
    return select_model_features(features, expected_columns)


def predict_accounts(
    features: pd.DataFrame,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    bundle = load_model_bundle(model_dir)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    X = validate_feature_columns(features, feature_names)
    pred = pipeline.predict(X)
    result = pd.DataFrame(index=features.index)
    if "account_id" in features.columns:
        result["account_id"] = features["account_id"].values
    result["predicted_label"] = pred.astype(int)
    result["predicted_label_name"] = [
        label_name_for(int(value)) for value in pred
    ]
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if 1 in classes:
            result["bot_probability"] = proba[:, classes.index(1)]
        if 0 in classes:
            result["genuine_probability"] = proba[:, classes.index(0)]
    result["label_mapping"] = str(bundle.get("label_mapping") or LABEL_MAPPING)
    return result


def predict_account(feature_row: dict | pd.Series, model_dir: Path | None = None) -> dict:
    row = pd.DataFrame([feature_row])
    predicted = predict_accounts(row, model_dir=model_dir)
    return predicted.iloc[0].to_dict()


def main() -> None:
    try:
        bundle = load_model_bundle()
    except ModelNotAvailableError as error:
        print(str(error))
        return
    print(f"Loaded {bundle.get('model_name')} from {bundle['model_file']}")
    print(f"Features: {bundle['feature_names']}")
    print("Call predict_account(feature_row) with those numeric columns.")


if __name__ == "__main__":
    main()
