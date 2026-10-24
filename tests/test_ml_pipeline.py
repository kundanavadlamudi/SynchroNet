import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ml.explain_predictions import explain_prediction, explanation_unavailable_reason
from src.ml.features import extract_account_features, model_feature_columns
from src.ml.predict_account import (
    ModelNotAvailableError,
    load_model_bundle,
    predict_account,
)
from src.ml.prepare_dataset import (
    LABEL_BOT,
    LABEL_GENUINE,
    combine_account_tables,
    map_group_to_label,
    prepare_ml_dataset,
)
from src.ml.train_models import (
    InsufficientLabeledDataError,
    split_account_dataset,
    train_and_evaluate,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_labeled_accounts(n_per_class: int = 20) -> pd.DataFrame:
    rows = []
    for index in range(n_per_class):
        rows.append(
            {
                "account_id": f"G{index}",
                "total_posts": 8 + (index % 3),
                "average_text_length": 40 + index,
                "average_word_count": 8 + (index % 4),
                "posts_with_urls": 1,
                "unique_texts": 8 + (index % 3),
                "duplicate_text_posts": 0,
                "url_post_ratio": 0.1,
                "duplicate_text_ratio": 0.0,
                "followers_count": 200 + index,
                "friends_count": 150 + index,
                "label": LABEL_GENUINE,
                "label_name": "Genuine",
                "source_dataset": "test-fixture",
                "source_group": "tfp" if index % 2 == 0 else "e13",
            }
        )
        rows.append(
            {
                "account_id": f"B{index}",
                "total_posts": 40 + index,
                "average_text_length": 12 + (index % 2),
                "average_word_count": 3,
                "posts_with_urls": 35 + (index % 5),
                "unique_texts": 4,
                "duplicate_text_posts": 30 + index,
                "url_post_ratio": 0.9,
                "duplicate_text_ratio": 0.85,
                "followers_count": 10 + index,
                "friends_count": 400 + index,
                "label": LABEL_BOT,
                "label_name": "Bot",
                "source_dataset": "test-fixture",
                "source_group": "fsf" if index % 2 == 0 else "social_spambots_1",
            }
        )
    return pd.DataFrame(rows)


class TestLabelMapping(unittest.TestCase):

    def test_documented_cresci_groups_map_to_genuine_or_bot(self):
        self.assertEqual(map_group_to_label("TFP"), ("cresci-2015", 0))
        self.assertEqual(map_group_to_label("E13.csv"), ("cresci-2015", 0))
        self.assertEqual(map_group_to_label("FSF"), ("cresci-2015", 1))
        self.assertEqual(map_group_to_label("INT"), ("cresci-2015", 1))
        self.assertEqual(map_group_to_label("TWT"), ("cresci-2015", 1))
        self.assertEqual(
            map_group_to_label("genuine_accounts"),
            ("cresci-2017", 0),
        )
        self.assertEqual(
            map_group_to_label("social_spambots_1"),
            ("cresci-2017", 1),
        )
        self.assertEqual(
            map_group_to_label("fake_followers"),
            ("cresci-2017", 1),
        )

    def test_unknown_groups_are_not_mapped(self):
        self.assertIsNone(map_group_to_label("mystery_accounts"))
        self.assertIsNone(map_group_to_label("sample_social_media_posts"))


class TestDatasetLoading(unittest.TestCase):

    def test_loader_discovers_cresci_layout_and_maps_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_csv(
                root / "cresci-2015" / "TFP" / "users.csv",
                [{"id": 1, "screen_name": "alice", "followers_count": 10}],
            )
            write_csv(
                root / "cresci-2015" / "TFP" / "tweets.csv",
                [
                    {
                        "id": 11,
                        "user_id": 1,
                        "text": "hello world",
                        "timestamp": "2020-01-01 00:00:00",
                        "num_urls": 0,
                    },
                    {
                        "id": 12,
                        "user_id": 1,
                        "text": "hello world",
                        "timestamp": "2020-01-01 00:10:00",
                        "num_urls": 0,
                    },
                ],
            )
            write_csv(
                root / "cresci-2017" / "social_spambots_1" / "users.csv",
                [{"id": 2, "screen_name": "bot", "followers_count": 3}],
            )
            write_csv(
                root / "cresci-2017" / "social_spambots_1" / "tweets.csv",
                [
                    {
                        "id": 21,
                        "user_id": 2,
                        "text": "buy now http://spam.example",
                        "timestamp": "2020-01-01 00:00:00",
                        "num_urls": 1,
                    }
                ],
            )
            write_csv(
                root / "ignored_group" / "users.csv",
                [{"id": 99, "screen_name": "unknown"}],
            )

            output = root / "ml_dataset.csv"
            status = root / "status.json"
            result = prepare_ml_dataset(
                search_roots=[root],
                output_file=output,
                status_file=status,
            )

            self.assertEqual(len(result.dataset), 2)
            genuine = result.dataset.set_index("account_id").loc["1"]
            bot = result.dataset.set_index("account_id").loc["2"]
            self.assertEqual(int(genuine["label"]), LABEL_GENUINE)
            self.assertEqual(int(bot["label"]), LABEL_BOT)
            self.assertTrue(output.exists())
            self.assertNotIn("99", set(result.dataset["account_id"].astype(str)))

    def test_conflicting_duplicate_accounts_are_dropped(self):
        frames = [
            pd.DataFrame([
                {"account_id": "1", "total_posts": 1, "label": 0},
                {"account_id": "2", "total_posts": 1, "label": 0},
            ]),
            pd.DataFrame([
                {"account_id": "1", "total_posts": 4, "label": 1},
            ]),
        ]
        combined, warnings = combine_account_tables(frames)
        self.assertNotIn("1", set(combined["account_id"].astype(str)))
        self.assertIn("2", set(combined["account_id"].astype(str)))
        self.assertTrue(any("conflicting" in warning for warning in warnings))


class TestFeatureExtraction(unittest.TestCase):

    def test_account_features_and_ratios(self):
        posts = pd.DataFrame([
            {
                "account_id": "A1",
                "text": "hello world",
                "url": "",
            },
            {
                "account_id": "A1",
                "text": "hello world",
                "url": "https://example.com",
            },
            {
                "account_id": "A1",
                "text": "another post with more words",
                "url": "",
            },
        ])
        features = extract_account_features(posts)
        row = features.iloc[0]
        self.assertEqual(int(row["total_posts"]), 3)
        self.assertEqual(int(row["unique_texts"]), 2)
        self.assertEqual(int(row["duplicate_text_posts"]), 1)
        self.assertEqual(int(row["posts_with_urls"]), 1)
        self.assertAlmostEqual(float(row["url_post_ratio"]), 1 / 3)
        self.assertAlmostEqual(float(row["duplicate_text_ratio"]), 1 / 3)
        self.assertGreater(float(row["average_word_count"]), 0)
        self.assertGreater(float(row["average_text_length"]), 0)

    def test_missing_values_and_zero_posts_do_not_divide_by_zero(self):
        posts = pd.DataFrame([
            {"account_id": "A2", "text": None, "url": None},
        ])
        features = extract_account_features(posts)
        self.assertEqual(int(features.iloc[0]["total_posts"]), 1)
        self.assertEqual(float(features.iloc[0]["url_post_ratio"]), 0.0)
        self.assertEqual(float(features.iloc[0]["average_text_length"]), 0.0)

        empty = extract_account_features(pd.DataFrame(columns=["account_id", "text"]))
        self.assertTrue(empty.empty)

    def test_identifiers_are_excluded_from_model_inputs(self):
        df = make_labeled_accounts(3)
        columns = model_feature_columns(df)
        self.assertNotIn("account_id", columns)
        self.assertNotIn("label", columns)
        self.assertNotIn("source_group", columns)
        self.assertIn("total_posts", columns)
        self.assertIn("url_post_ratio", columns)


class TestDataSplitting(unittest.TestCase):

    def test_splits_do_not_leak_accounts(self):
        dataset = make_labeled_accounts(20)
        splits = split_account_dataset(dataset)
        train_ids = set(splits["train"]["account_id"])
        val_ids = set(splits["validation"]["account_id"])
        test_ids = set(splits["test"]["account_id"])
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(
            train_ids | val_ids | test_ids,
            set(dataset["account_id"]),
        )
        for part in ("train", "validation", "test"):
            self.assertGreaterEqual(splits[part]["label"].nunique(), 2)

    def test_single_class_data_is_rejected(self):
        dataset = make_labeled_accounts(10)
        dataset = dataset[dataset["label"] == 0]
        with self.assertRaises(InsufficientLabeledDataError):
            split_account_dataset(dataset)


class TestModelTrainingEvaluationAndPersistence(unittest.TestCase):

    def test_training_evaluation_save_and_load(self):
        dataset = make_labeled_accounts(20)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outcome = train_and_evaluate(
                dataset,
                model_dir=root / "models",
                graph_dir=root / "graphs",
                comparison_file=root / "ml_model_comparison.csv",
                summary_file=root / "ml_evaluation_summary.json",
            )

            self.assertIn(
                outcome["selected_model"],
                {"logistic_regression", "decision_tree", "random_forest"},
            )
            self.assertTrue(outcome["model_file"].exists())
            self.assertTrue(outcome["comparison_file"].exists())
            self.assertTrue(outcome["summary_file"].exists())

            comparison = pd.read_csv(outcome["comparison_file"])
            self.assertEqual(len(comparison), 3)
            self.assertTrue(
                {"accuracy", "precision", "recall", "f1_score"}.issubset(
                    comparison.columns
                )
            )
            for value in comparison["accuracy"]:
                self.assertGreaterEqual(float(value), 0.0)
                self.assertLessEqual(float(value), 1.0)

            summary = json.loads(outcome["summary_file"].read_text(encoding="utf-8"))
            test_metrics = summary["test_results_selected_model"]
            self.assertIn("accuracy", test_metrics)
            self.assertIn("classification_report", test_metrics)
            self.assertIn("confusion_matrix", test_metrics)

            bundle = load_model_bundle(root / "models")
            sample = dataset.drop(columns=["label"]).iloc[0]
            predicted = predict_account(sample, model_dir=root / "models")
            self.assertIn(int(predicted["predicted_label"]), {0, 1})
            self.assertIn(
                predicted["predicted_label_name"],
                {"Genuine", "Bot"},
            )

            shap_result = explain_prediction(
                sample,
                model_dir=root / "models",
                method="shap",
            )
            lime_result = explain_prediction(
                sample,
                model_dir=root / "models",
                method="lime",
            )
            self.assertIn("available", shap_result)
            self.assertIn("available", lime_result)
            if shap_result["available"]:
                self.assertTrue(shap_result["feature_attributions"])
            if lime_result["available"]:
                self.assertTrue(lime_result["feature_attributions"])

            del bundle

    def test_predict_without_model_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ModelNotAvailableError):
                load_model_bundle(Path(temp_dir))
            reason = explanation_unavailable_reason(Path(temp_dir))
            self.assertIn("no trained bot classifier", reason.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
