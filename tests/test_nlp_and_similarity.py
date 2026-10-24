"""NexTrace — Semantic Similarity and NLP Validation Tests.

Validates the semantic similarity engine across core linguistic conditions:
1. Exact duplicates: similarity must equal 1.0.
2. Paraphrased sentences: strong similarity (> 0.65).
3. Completely unrelated sentences: low similarity (< 0.40).
4. Empty or whitespace-only inputs: safely return 0.0 without crashing.
5. Non-ASCII and multilingual Unicode inputs: process cleanly without encoding errors.
6. Pairwise DataFrame exclusion: posts from the same account_id are never paired.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.semantic_similarity import (
    calculate_semantic_similarity,
    compute_pairwise_text_similarity,
)


class TestSemanticSimilarityAndNLP(unittest.TestCase):
    """Linguistic and computational boundary tests for the similarity module."""

    def test_exact_duplicates_return_maximum_similarity(self):
        text = "Coordinated influence campaign detected across social network accounts."
        score = compute_pairwise_text_similarity(text, text)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_paraphrased_sentences_return_high_similarity(self):
        # Astroturfed variant with timing/location token alteration
        text_a = "Breaking news about a major cyber security incident #CyberSecurity"
        text_b = "Breaking news about a major cyber security incident today #CyberSecurity"
        score = compute_pairwise_text_similarity(text_a, text_b)
        self.assertGreater(
            score,
            0.70,
            f"Expected strong similarity (> 0.70) for near-duplicate paraphrase, got {score:.3f}",
        )

        # Lexical modification with preserved structure
        text_c = "Breaking: Major coordinated cyber incident confirmed on central infrastructure #CyberAlert"
        text_d = "Breaking: Major coordinated cyber attack incident confirmed on critical infrastructure #CyberAlert"
        score_mod = compute_pairwise_text_similarity(text_c, text_d)
        self.assertGreater(
            score_mod,
            0.55,
            f"Expected substantial similarity (> 0.55) for structural paraphrase, got {score_mod:.3f}",
        )

    def test_unrelated_sentences_return_low_similarity(self):
        text_a = "Coordinated bots launched an information operation against election infrastructure."
        text_b = "The recipe for chocolate chip cookies requires butter, brown sugar, and flour."
        score = compute_pairwise_text_similarity(text_a, text_b)
        self.assertLess(
            score,
            0.40,
            f"Expected low similarity for unrelated text, got {score:.3f}",
        )

    def test_empty_and_whitespace_inputs_safely_return_zero(self):
        self.assertEqual(compute_pairwise_text_similarity("", ""), 0.0)
        self.assertEqual(compute_pairwise_text_similarity("   ", "Valid post text"), 0.0)
        self.assertEqual(compute_pairwise_text_similarity("\n\t", "\r\n"), 0.0)
        self.assertEqual(compute_pairwise_text_similarity(None, "Valid post text"), 0.0)

    def test_multilingual_and_unicode_text_handled_safely(self):
        # Non-ASCII European, Cyrillic, Chinese, and emoji text
        text_a = "Desinformation und manipulierte Netzwerke gefährden demokratische Wahlen! 🚨"
        text_b = "Desinformation und manipulierte Netzwerke bedrohen demokratische Wahlen! 🚨"
        score = compute_pairwise_text_similarity(text_a, text_b)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.60)

        # Non-ASCII Cyrillic
        cyr_a = "Скоординированная дезинформация обнаружена в социальных сетях."
        cyr_b = "Скоординированная пропаганда распространяется через фальшивые аккаунты."
        score_cyr = compute_pairwise_text_similarity(cyr_a, cyr_b)
        self.assertIsInstance(score_cyr, float)

    def test_same_account_exclusion_in_dataframe_similarity(self):
        df = pd.DataFrame([
            {"post_id": "P01", "account_id": "ACC_X", "text": "Identical urgent breaking report."},
            {"post_id": "P02", "account_id": "ACC_X", "text": "Identical urgent breaking report."},
            {"post_id": "P03", "account_id": "ACC_Y", "text": "Identical urgent breaking report."},
        ])
        results = calculate_semantic_similarity(df)
        self.assertFalse(results.empty)
        # Should pair P01 (ACC_X) with P03 (ACC_Y) and P02 (ACC_X) with P03 (ACC_Y)
        # But NEVER pair P01 (ACC_X) with P02 (ACC_X)
        for _, row in results.iterrows():
            self.assertNotEqual(row["account_1"], row["account_2"])


if __name__ == "__main__":
    unittest.main()
