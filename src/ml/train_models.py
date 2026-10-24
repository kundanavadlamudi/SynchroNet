"""Train baseline Genuine vs Bot classifiers.

Training runs only on prepared labeled account features. Phase 1 sample
posts are never used as ground-truth bot labels.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.ml.evaluate_models import (
    comparison_row,
    evaluate_predictions,
    plot_confusion_matrix,
    save_evaluation_outputs,
)
from src.ml.explain_predictions import explain_sample_accounts
from src.ml.features import model_feature_columns, select_model_features
from src.ml.prepare_dataset import (
    LABEL_MAPPING,
    OUTPUT_FILE,
    LabeledDatasetNotFoundError,
    describe_missing_datasets,
    prepare_ml_dataset,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
GRAPH_DIR = BASE_DIR / "results" / "graphs"
SELECTED_MODEL_FILE = MODEL_DIR / "selected_bot_classifier.joblib"
METADATA_FILE = MODEL_DIR / "selected_bot_classifier_metadata.json"
BACKGROUND_FILE = MODEL_DIR / "background_sample.csv"
RANDOM_STATE = 42


class InsufficientLabeledDataError(ValueError):
    """Raised when labeled data cannot support a valid train/eval split."""


def load_prepared_dataset(dataset_file: Path | None = None) -> pd.DataFrame:
    path = Path(dataset_file or OUTPUT_FILE)
    if not path.exists():
        raise LabeledDatasetNotFoundError(describe_missing_datasets())
    df = pd.read_csv(path)
    if df.empty:
        raise LabeledDatasetNotFoundError(describe_missing_datasets())
    if "label" not in df.columns:
        raise ValueError("Prepared dataset is missing the label column.")
    return df


def _positive_class_proba(model, X) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return None


def split_account_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    group_column: str = "source_group",
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Create train/validation/test splits without account leakage.

    Each account appears in only one split. When both classes have at
    least two source groups, groups are kept together to reduce
    campaign-level leakage. Otherwise a stratified account split is used.
    """
    if df["account_id"].duplicated().any():
        raise ValueError("Split requires unique account_id rows.")

    if df[label_column].nunique() < 2:
        raise InsufficientLabeledDataError(
            "Training requires both Genuine (0) and Bot (1) accounts."
        )

    if len(df) < 12:
        raise InsufficientLabeledDataError(
            "At least 12 labeled accounts are required for train/validation/test."
        )

    labels = df[label_column]
    use_groups = (
        group_column in df.columns
        and df[group_column].nunique() >= 4
        and bool((df.groupby(label_column)[group_column].nunique() >= 2).all())
    )

    train_df = validation_df = test_df = None
    strategy = "stratified_by_label"

    if use_groups:
        try:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.30,
                random_state=random_state,
            )
            train_idx, temp_idx = next(
                splitter.split(df, labels, groups=df[group_column])
            )
            grouped_train = df.iloc[train_idx].copy()
            temp_df = df.iloc[temp_idx].copy()
            if (
                grouped_train[label_column].nunique() < 2
                or temp_df[label_column].nunique() < 2
                or temp_df[group_column].nunique() < 2
                or len(temp_df) < 4
            ):
                raise InsufficientLabeledDataError(
                    "Group split did not retain both classes."
                )
            val_idx, test_idx = next(
                GroupShuffleSplit(
                    n_splits=1,
                    test_size=0.50,
                    random_state=random_state,
                ).split(
                    temp_df,
                    temp_df[label_column],
                    groups=temp_df[group_column],
                )
            )
            grouped_val = temp_df.iloc[val_idx].copy()
            grouped_test = temp_df.iloc[test_idx].copy()
            if (
                grouped_val[label_column].nunique() < 2
                or grouped_test[label_column].nunique() < 2
            ):
                raise InsufficientLabeledDataError(
                    "Group split did not retain both classes."
                )
            train_df = grouped_train
            validation_df = grouped_val
            test_df = grouped_test
            strategy = "group_shuffle_by_source_group"
        except (InsufficientLabeledDataError, ValueError):
            train_df = validation_df = test_df = None
            strategy = "stratified_by_label"

    if train_df is None:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            stratify=labels,
            random_state=random_state,
        )
        validation_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df[label_column],
            random_state=random_state,
        )
        strategy = "stratified_by_label"

    overlap = (
        set(train_df["account_id"])
        & set(validation_df["account_id"])
        & set(test_df["account_id"])
    )
    pairwise_overlap = (
        set(train_df["account_id"]) & set(validation_df["account_id"])
        | set(train_df["account_id"]) & set(test_df["account_id"])
        | set(validation_df["account_id"]) & set(test_df["account_id"])
    )
    if overlap or pairwise_overlap:
        raise RuntimeError("Account leakage detected in data splits.")

    train_df.attrs["split_strategy"] = strategy
    return {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
        "strategy": strategy,
    }


def build_model_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(
    dataset: pd.DataFrame,
    model_dir: Path | None = None,
    graph_dir: Path | None = None,
    comparison_file: Path | None = None,
    summary_file: Path | None = None,
) -> dict:
    feature_columns = model_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for training.")

    splits = split_account_dataset(dataset)
    X_train = select_model_features(splits["train"], feature_columns)
    y_train = splits["train"]["label"].astype(int)
    X_val = select_model_features(splits["validation"], feature_columns)
    y_val = splits["validation"]["label"].astype(int)
    X_test = select_model_features(splits["test"], feature_columns)
    y_test = splits["test"]["label"].astype(int)

    pipelines = build_model_pipelines()
    graph_path = Path(graph_dir or GRAPH_DIR)
    graph_path.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    validation_details = {}
    fitted = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        val_pred = pipeline.predict(X_val)
        val_proba = _positive_class_proba(pipeline, X_val)
        metrics = evaluate_predictions(
            y_val, val_pred, val_proba, split_name="validation"
        )
        validation_details[name] = metrics
        comparison_rows.append(comparison_row(name, metrics))
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            graph_path / f"{name}_validation_confusion_matrix.png",
            title=f"{name} validation confusion matrix",
        )

    selected_name = max(
        validation_details,
        key=lambda name: (
            validation_details[name]["f1_score"],
            validation_details[name]["accuracy"],
        ),
    )
    selected_model = fitted[selected_name]

    test_pred = selected_model.predict(X_test)
    test_proba = _positive_class_proba(selected_model, X_test)
    test_metrics = evaluate_predictions(
        y_test, test_pred, test_proba, split_name="test"
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        graph_path / "selected_model_test_confusion_matrix.png",
        title=f"{selected_name} test confusion matrix",
    )

    summary = {
        "task": "account_bot_classification",
        "distinct_from": "coordination_detection",
        "label_mapping": LABEL_MAPPING,
        "feature_names": feature_columns,
        "split_strategy": splits["strategy"],
        "n_train": int(len(splits["train"])),
        "n_validation": int(len(splits["validation"])),
        "n_test": int(len(splits["test"])),
        "selected_model": selected_name,
        "selection_rule": "highest_validation_f1_then_accuracy",
        "validation_results": validation_details,
        "test_results_selected_model": test_metrics,
        "dataset_provenance": {
            "n_accounts": int(len(dataset)),
            "source_datasets": sorted(
                dataset["source_dataset"].dropna().astype(str).unique().tolist()
            ) if "source_dataset" in dataset.columns else [],
            "source_groups": sorted(
                dataset["source_group"].dropna().astype(str).unique().tolist()
            ) if "source_group" in dataset.columns else [],
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    comparison_path, summary_path = save_evaluation_outputs(
        comparison_rows,
        summary,
        comparison_file=comparison_file,
        summary_file=summary_file,
    )

    model_path = Path(model_dir or MODEL_DIR)
    model_path.mkdir(parents=True, exist_ok=True)
    selected_file = model_path / SELECTED_MODEL_FILE.name
    metadata_file = model_path / METADATA_FILE.name
    background_file = model_path / BACKGROUND_FILE.name

    joblib.dump(
        {
            "pipeline": selected_model,
            "feature_names": feature_columns,
            "label_mapping": LABEL_MAPPING,
            "model_name": selected_name,
        },
        selected_file,
    )

    background = X_train.head(100).copy()
    background.to_csv(background_file, index=False, encoding="utf-8-sig")

    metadata = {
        "model_name": selected_name,
        "feature_names": feature_columns,
        "label_mapping": LABEL_MAPPING,
        "split_strategy": splits["strategy"],
        "dataset_provenance": summary["dataset_provenance"],
        "validation_f1_score": validation_details[selected_name]["f1_score"],
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
        },
        "model_file": str(selected_file),
        "background_sample_file": str(background_file),
        "comparison_file": str(comparison_path),
        "summary_file": str(summary_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    explanation_file = BASE_DIR / "results" / "reports" / "ml_explanations.json"
    if summary_file is not None:
        explanation_file = Path(summary_file).parent / "ml_explanations.json"
    try:
        explanations = explain_sample_accounts(
            splits["test"],
            model_dir=model_path,
            output_file=explanation_file,
        )
    except Exception as error:
        explanations = [{
            "available": False,
            "reason": f"Explainable AI did not complete: {error}",
        }]
        explanation_file.write_text(
            json.dumps(explanations, indent=2, default=str),
            encoding="utf-8",
        )

    return {
        "selected_model": selected_name,
        "feature_names": feature_columns,
        "splits": splits,
        "validation_results": validation_details,
        "test_results": test_metrics,
        "summary": summary,
        "model_file": selected_file,
        "metadata_file": metadata_file,
        "comparison_file": comparison_path,
        "summary_file": summary_path,
        "explanation_file": explanation_file,
        "explanations": explanations,
    }


def main() -> None:
    print("NexTrace bot classification training")
    print(
        "This trains Genuine vs Bot account classifiers. "
        "It does not detect coordinated campaigns."
    )

    prepared = OUTPUT_FILE
    if not prepared.exists():
        print("Prepared dataset not found. Running dataset preparation.")
        result = prepare_ml_dataset()
        if result.dataset.empty:
            print(describe_missing_datasets())
            return

    try:
        dataset = load_prepared_dataset()
        outcome = train_and_evaluate(dataset)
    except (LabeledDatasetNotFoundError, InsufficientLabeledDataError) as error:
        print(str(error))
        return

    print(f"Selected model: {outcome['selected_model']}")
    print("Validation comparison:")
    for name, metrics in outcome["validation_results"].items():
        print(
            f"  {name}: F1={metrics['f1_score']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"ROC-AUC={metrics['roc_auc']}"
        )
    test = outcome["test_results"]
    print("Held-out test results for selected model:")
    print(
        f"  accuracy={test['accuracy']:.4f} "
        f"precision={test['precision']:.4f} "
        f"recall={test['recall']:.4f} "
        f"F1={test['f1_score']:.4f} "
        f"ROC-AUC={test['roc_auc']}"
    )
    print(f"Model saved to: {outcome['model_file']}")
    print(f"Comparison saved to: {outcome['comparison_file']}")
    print(f"Summary saved to: {outcome['summary_file']}")
    if outcome.get("explanation_file"):
        print(f"Explanations saved to: {outcome['explanation_file']}")


if __name__ == "__main__":
    main()
