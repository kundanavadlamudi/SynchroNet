"""Optional SHAP and LIME explanations for a fitted bot classifier.

Explanations are generated only from a compatible trained model and the
account's real feature values. This module does not invent attribution
scores. Coordination detection is a separate Phase 1 task and is not
explained here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.predict_account import (
    ModelNotAvailableError,
    load_model_bundle,
    validate_feature_columns,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
BACKGROUND_FILE = MODEL_DIR / "background_sample.csv"


def _import_shap():
    try:
        import shap
    except ImportError:
        return None
    return shap


def _import_lime():
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return None
    return LimeTabularExplainer


def explanation_unavailable_reason(model_dir: Path | None = None) -> str | None:
    directory = Path(model_dir or MODEL_DIR)
    model_file = directory / "selected_bot_classifier.joblib"
    if not model_file.exists():
        return (
            "Explainable AI cannot run because no trained bot classifier "
            "is available. Place documented Cresci labeled data under "
            "data/raw, prepare the dataset, and train a model first."
        )
    return None


def explain_with_shap(
    feature_row: dict | pd.Series,
    model_dir: Path | None = None,
    background: pd.DataFrame | None = None,
) -> dict:
    shap = _import_shap()
    if shap is None:
        return {
            "method": "shap",
            "available": False,
            "reason": "The shap package is not installed.",
        }

    unavailable = explanation_unavailable_reason(model_dir)
    if unavailable:
        return {"method": "shap", "available": False, "reason": unavailable}

    bundle = load_model_bundle(model_dir)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    row = pd.DataFrame([feature_row])
    X = validate_feature_columns(row, feature_names)

    if background is None:
        background_path = Path(model_dir or MODEL_DIR) / BACKGROUND_FILE.name
        if background_path.exists():
            background = pd.read_csv(background_path)
        else:
            background = X
    background_X = validate_feature_columns(background, feature_names)

    transformed_background = pipeline[:-1].transform(background_X)
    transformed_row = pipeline[:-1].transform(X)
    classifier = pipeline.named_steps["clf"]
    model_name = str(bundle.get("model_name") or "")

    try:
        if model_name in {"random_forest", "decision_tree"}:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_row)
        else:
            explainer = shap.Explainer(classifier, transformed_background)
            shap_values = explainer(transformed_row).values
    except Exception as error:
        return {
            "method": "shap",
            "available": False,
            "reason": f"SHAP could not explain this model: {error}",
        }

    if isinstance(shap_values, list):
        positive_index = list(getattr(classifier, "classes_", [0, 1])).index(1) if hasattr(classifier, "classes_") and 1 in list(classifier.classes_) else -1
        values = shap_values[positive_index][0]
    else:
        values = shap_values[0]
        if getattr(values, "ndim", 1) > 1:
            values = values[:, -1]

    attributions = {
        feature_names[index]: float(values[index])
        for index in range(len(feature_names))
    }
    ranked = sorted(
        attributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return {
        "method": "shap",
        "available": True,
        "model_name": bundle.get("model_name"),
        "feature_attributions": attributions,
        "top_features": ranked[:10],
    }


def explain_with_lime(
    feature_row: dict | pd.Series,
    model_dir: Path | None = None,
    background: pd.DataFrame | None = None,
    num_features: int = 8,
) -> dict:
    LimeTabularExplainer = _import_lime()
    if LimeTabularExplainer is None:
        return {
            "method": "lime",
            "available": False,
            "reason": "The lime package is not installed.",
        }

    unavailable = explanation_unavailable_reason(model_dir)
    if unavailable:
        return {"method": "lime", "available": False, "reason": unavailable}

    bundle = load_model_bundle(model_dir)
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    row = pd.DataFrame([feature_row])
    X = validate_feature_columns(row, feature_names)

    if background is None:
        background_path = Path(model_dir or MODEL_DIR) / BACKGROUND_FILE.name
        if not background_path.exists():
            return {
                "method": "lime",
                "available": False,
                "reason": "No background sample is saved for LIME.",
            }
        background = pd.read_csv(background_path)
    background_X = validate_feature_columns(background, feature_names)
    fill_values = background_X.median(numeric_only=True)
    background_X = background_X.fillna(fill_values)
    X = X.fillna(fill_values)

    def predict_proba_df(data):
        frame = pd.DataFrame(data, columns=feature_names)
        return pipeline.predict_proba(frame)

    try:
        explainer = LimeTabularExplainer(
            background_X.to_numpy(dtype=float),
            feature_names=feature_names,
            class_names=["Genuine", "Bot"],
            discretize_continuous=False,
            random_state=42,
        )
        explanation = explainer.explain_instance(
            X.iloc[0].to_numpy(dtype=float),
            predict_proba_df,
            num_features=num_features,
            labels=(1,),
        )
    except Exception as error:
        return {
            "method": "lime",
            "available": False,
            "reason": f"LIME could not explain this instance: {error}",
        }
    return {
        "method": "lime",
        "available": True,
        "model_name": bundle.get("model_name"),
        "feature_attributions": dict(explanation.as_list(label=1)),
    }


def explain_prediction(
    feature_row: dict | pd.Series,
    model_dir: Path | None = None,
    method: str = "shap",
) -> dict:
    method = method.lower().strip()
    if method == "shap":
        return explain_with_shap(feature_row, model_dir=model_dir)
    if method == "lime":
        return explain_with_lime(feature_row, model_dir=model_dir)
    return {
        "method": method,
        "available": False,
        "reason": "Supported methods are shap and lime.",
    }


def explain_sample_accounts(
    accounts: pd.DataFrame,
    model_dir: Path | None = None,
    output_file: Path | None = None,
    n_per_class: int = 2,
) -> list[dict]:
    """Explain a few labeled accounts with SHAP and LIME after training."""
    unavailable = explanation_unavailable_reason(model_dir)
    if unavailable:
        return [{"available": False, "reason": unavailable}]

    if accounts is None or accounts.empty:
        return [{"available": False, "reason": "No accounts were provided."}]

    samples = []
    if "label" in accounts.columns:
        for label in (0, 1):
            subset = accounts[accounts["label"] == label]
            samples.append(subset.head(n_per_class))
        chosen = pd.concat(samples, ignore_index=True)
    else:
        chosen = accounts.head(n_per_class * 2)

    explanations = []
    for _, row in chosen.iterrows():
        payload = {
            "account_id": None if "account_id" not in row.index else str(row["account_id"]),
            "true_label": None if "label" not in row.index else int(row["label"]),
            "source_group": None if "source_group" not in row.index else str(row["source_group"]),
            "shap": explain_prediction(row, model_dir=model_dir, method="shap"),
            "lime": explain_prediction(row, model_dir=model_dir, method="lime"),
        }
        explanations.append(payload)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(explanations, indent=2, default=str),
            encoding="utf-8",
        )
    return explanations


def main() -> None:
    try:
        load_model_bundle()
    except ModelNotAvailableError as error:
        print(str(error))
        print(
            "SHAP/LIME are wired for later use. They will not fabricate "
            "explanations without a trained model."
        )
        return

    dataset_file = BASE_DIR / "data" / "processed" / "ml_dataset.csv"
    output_file = BASE_DIR / "results" / "reports" / "ml_explanations.json"
    if not dataset_file.exists():
        print("A trained model is available, but ml_dataset.csv was not found.")
        return
    accounts = pd.read_csv(dataset_file)
    explanations = explain_sample_accounts(
        accounts,
        output_file=output_file,
    )
    available = [
        item for item in explanations
        if item.get("shap", {}).get("available") or item.get("lime", {}).get("available")
    ]
    print(f"Wrote {len(explanations)} explanation records to {output_file}")
    print(f"Records with a successful SHAP or LIME result: {len(available)}")


if __name__ == "__main__":
    main()
