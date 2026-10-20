import unittest
import pandas as pd

from src.data_preprocessing import preprocess_data
from src.coordination_detection import detect_coordination
from src.risk_scoring import calculate_risk


class TestNexTrace(unittest.TestCase):

    def setUp(self):

        self.posts = pd.DataFrame([
            {
                "post_id": "P001",
                "account_id": "A101",
                "text": "Heavy rainfall caused flooding",
                "timestamp": "2026-10-20 09:00:00",
                "url": "https://example.com/flood"
            },
            {
                "post_id": "P002",
                "account_id": "A102",
                "text": "Heavy rainfall caused flooding",
                "timestamp": "2026-10-20 09:02:00",
                "url": "https://example.com/flood"
            },
            {
                "post_id": "P003",
                "account_id": "A103",
                "text": "Practice safe password habits",
                "timestamp": "2026-10-20 12:00:00",
                "url": "https://example.com/security"
            }
        ])

    def test_preprocessing_removes_duplicate_ids(self):

        data = pd.concat(
            [self.posts, self.posts.iloc[[0]]],
            ignore_index=True
        )

        cleaned = preprocess_data(data)

        self.assertEqual(
            cleaned["post_id"].nunique(),
            len(cleaned)
        )

    def test_preprocessing_removes_invalid_timestamps(self):

        data = self.posts.copy()

        data.loc[0, "timestamp"] = "invalid timestamp"

        cleaned = preprocess_data(data)

        self.assertNotIn("P001", cleaned["post_id"].tolist())

    def test_coordination_detection_flags_supported_pair(self):

        semantic_results = pd.DataFrame([
            {
                "post_1": "P001",
                "account_1": "A101",
                "post_2": "P002",
                "account_2": "A102",
                "text_similarity": 0.95,
                "similarity_type": "Semantic"
            }
        ])

        results = detect_coordination(
            self.posts,
            semantic_results
        )

        self.assertTrue(
            (
                (results["account_1"] == "A101")
                & (results["account_2"] == "A102")
            ).any()
        )

    def test_risk_scoring_explains_evidence(self):

        coordination_data = pd.DataFrame([
            {
                "account_1": "A101",
                "account_2": "A102",
                "semantic_similarity": 0.95,
                "time_difference_minutes": 2,
                "shared_url": True,
                "coordination_score": 0.975,
                "evidence_signals": 3
            }
        ])

        results = calculate_risk(coordination_data)

        self.assertEqual(
            results.iloc[0]["investigation_priority"],
            "High"
        )

        self.assertIn(
            "Shared URL detected",
            results.iloc[0]["evidence"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)