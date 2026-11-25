import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


BOOLEAN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOLEAN_DIR))

from boolean.boolean_benchmark_improved_2 import BooleanBenchmarkImproved


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def get_name(self):
        return "FakeLLM"

    def query_with_usage(self, prompt):
        self.prompts.append(prompt)
        return {
            "response": self.responses.pop(0),
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3
            },
            "cost": 0.0
        }


def make_sample(identifier, input_values, expected, ground_truths=None):
    return {
        "observation_set_id": identifier,
        "n_observations": 1,
        "observations": [{
            "inputs": input_values,
            "output": expected
        }],
        "ground_truth_expressions": ground_truths or []
    }


class Improved2FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_path = BOOLEAN_DIR / "datasets" / "boolean_2var.json"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.benchmark = BooleanBenchmarkImproved(str(dataset_path))

    def test_invalid_candidate_is_named_and_observation_match_has_priority(self):
        llm = FakeLLM([
            "Expression: x OR y",
            "Expression: y"
        ])
        sample = make_sample("repair", {"x": 1, "y": 0}, 0)

        result = self.benchmark.evaluate_single_observation_set(
            llm, sample, n_queries=2
        )

        second_prompt = llm.prompts[1]
        self.assertIn('Previous candidate: "x OR y"', second_prompt)
        self.assertIn("does not match observation 1 ((x=1, y=0) -> 0)", second_prompt)
        self.assertIn("Do not sacrifice validity for structural novelty.", second_prompt)
        self.assertEqual(result["valid_unique_archive"], ["y"])
        self.assertEqual(result["n_queries"], 2)
        self.assertEqual(result["n_valid"], 1)
        self.assertEqual(result["valid_rate"], 0.5)
        expected_fields = {
            "observation_set_id", "n_observations", "n_ground_truths", "n_queries",
            "n_valid", "n_unique_valid", "n_unique_all", "n_recovered_gts",
            "parse_success_rate", "in_space_rate", "valid_rate", "novelty_rate",
            "recovery_rate", "all_hypotheses", "valid_hypotheses",
            "unique_valid_expressions", "unique_all_expressions", "token_usage",
            "cost", "errors", "error_summary"
        }
        self.assertTrue(expected_fields.issubset(result))

    def test_valid_candidate_feedback_asks_for_unique_structure_only_if_still_valid(self):
        llm = FakeLLM([
            "Expression: x AND y",
            "Expression: x OR y",
            "Expression: y AND x",
            "Expression: x AND y"
        ])
        sample = make_sample("valid-diversity", {"x": 1, "y": 1}, 1)

        result = self.benchmark.evaluate_single_observation_set(
            llm, sample, n_queries=4
        )

        self.assertEqual(result["valid_unique_archive"], ["x AND y", "x OR y"])
        self.assertIn('Previous candidate: "x AND y"', llm.prompts[1])
        self.assertIn("Try another structure only if it still satisfies every observation.", llm.prompts[1])
        self.assertIn("duplicates a structure in the validated archive", llm.prompts[3])
        self.assertEqual(result["valid_rate"], 1.0)
        self.assertEqual(result["n_queries"], 4)
        self.assertEqual(len(result["raw_responses"]), 4)

    def test_ground_truth_reference_is_not_in_prompt_or_diagnostic(self):
        sentinel = "REFERENCE_ONLY_SENTINEL"
        llm = FakeLLM(["Expression: x OR y", "Expression: y"])
        sample = make_sample(
            "no-reference-leak",
            {"x": 1, "y": 0},
            0,
            ground_truths=[{
                "expression": sentinel,
                "mechanistic_key": None
            }]
        )

        result = self.benchmark.evaluate_single_observation_set(
            llm, sample, n_queries=2
        )

        visible_text = "\n".join(llm.prompts) + str(result["feedback_history"])
        self.assertNotIn(sentinel, visible_text)
        self.assertEqual(result["n_ground_truths"], 1)

    def test_run_metadata_and_checkpoint_are_isolated_as_improved_2(self):
        llm = FakeLLM(["Expression: x AND y"])
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.benchmark.run_benchmark(
                    llm=llm,
                    n_samples=1,
                    n_queries_per_sample=1,
                    seed=33550336,
                    verbose=False,
                    checkpoint_dir=checkpoint_dir,
                    run_id="offline-seed"
                )

            self.assertEqual(result["method"], "improved_2")
            self.assertEqual(result["run_id"], "improved_2_offline-seed")
            checkpoint_path = next(Path(checkpoint_dir).glob("checkpoint_*.json"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["method"], "improved_2")
            self.assertEqual(checkpoint["run_id"], "improved_2_offline-seed")


if __name__ == "__main__":
    unittest.main()
