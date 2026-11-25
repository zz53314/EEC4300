import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BOOLEAN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOLEAN_DIR))

from boolean.boolean_benchmark_improved_3 import BooleanBenchmarkImproved3
import boolean.boolean_benchmark_improved_3 as improved_3_module


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


def make_sample(identifier, observations, ground_truths=None):
    return {
        "observation_set_id": identifier,
        "n_observations": len(observations),
        "observations": observations,
        "ground_truth_expressions": ground_truths or []
    }


class Improved3FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_path = BOOLEAN_DIR / "datasets" / "boolean_2var.json"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.benchmark = BooleanBenchmarkImproved3(str(dataset_path))

    def test_feedback_lists_every_mismatching_observation(self):
        llm = FakeLLM([
            "Expression: x OR y",
            "Expression: x AND y"
        ])
        sample = make_sample("all-mismatches", [
            {"inputs": {"x": 1, "y": 0}, "output": 0},
            {"inputs": {"x": 0, "y": 1}, "output": 0}
        ])

        result = self.benchmark.evaluate_single_observation_set(
            llm, sample, n_queries=2
        )

        diagnostic = result["feedback_history"][0]["feedback"]
        self.assertIn("does not match these observations: 1", diagnostic)
        self.assertIn("2 ((x=0, y=1) -> 0)", diagnostic)
        self.assertIn("1 ((x=1, y=0) -> 0)", diagnostic)
        self.assertEqual(result["valid_unique_archive"], ["x AND y"])
        self.assertEqual(result["n_queries"], 2)
        self.assertEqual(result["valid_rate"], 0.5)
        self.assertEqual(len(result["raw_responses"]), 2)
        self.assertEqual(len(result["all_hypotheses"]), 2)
        self.assertEqual(result["n_unique_all"], 2)
        self.assertEqual(result["novelty_rate"], 1.0)
        expected_fields = {
            "observation_set_id", "n_observations", "n_ground_truths", "n_queries",
            "n_valid", "n_unique_valid", "n_unique_all", "n_recovered_gts",
            "parse_success_rate", "in_space_rate", "valid_rate", "novelty_rate",
            "recovery_rate", "all_hypotheses", "valid_hypotheses",
            "unique_valid_expressions", "unique_all_expressions", "token_usage",
            "cost", "errors", "error_summary", "raw_responses",
            "valid_unique_archive", "feedback_history"
        }
        self.assertTrue(expected_fields.issubset(result))

    def test_duplicate_structure_is_not_archived_and_feedback_preserves_validity(self):
        llm = FakeLLM([
            "Expression: x AND y",
            "Expression: x OR y",
            "Expression: y AND x"
        ])
        sample = make_sample("duplicate-structure", [
            {"inputs": {"x": 1, "y": 1}, "output": 1}
        ])

        result = self.benchmark.evaluate_single_observation_set(
            llm, sample, n_queries=3
        )

        self.assertEqual(
            result["valid_unique_archive"],
            ["x AND y", "x OR y"]
        )
        self.assertIn(
            "duplicates a structure in the validated archive",
            result["feedback_history"][2]["feedback"]
        )
        self.assertIn(
            "every observation satisfied",
            result["feedback_history"][2]["feedback"]
        )
        self.assertEqual(result["n_queries"], 3)
        self.assertEqual(result["valid_rate"], 1.0)
        self.assertEqual(result["n_unique_valid"], 2)

    def test_archive_is_isolated_between_observation_sets(self):
        llm = FakeLLM(["Expression: x AND y", "Expression: x OR y"])
        first = make_sample("first", [
            {"inputs": {"x": 1, "y": 1}, "output": 1}
        ])
        second = make_sample("second", [
            {"inputs": {"x": 1, "y": 1}, "output": 1}
        ])

        self.benchmark.evaluate_single_observation_set(llm, first, n_queries=1)
        self.benchmark.evaluate_single_observation_set(llm, second, n_queries=1)

        for prompt in llm.prompts:
            prior_block = prompt.split("Prior expressions generated", 1)[1].split(
                "Task:", 1
            )[0]
            self.assertIn("None", prior_block)

    def test_ground_truth_is_not_in_feedback_or_prompt(self):
        sentinel = "REFERENCE_ONLY_SENTINEL"
        llm = FakeLLM(["Expression: x OR y", "Expression: x AND y"])
        sample = make_sample(
            "no-reference-leak",
            [
                {"inputs": {"x": 1, "y": 0}, "output": 0},
                {"inputs": {"x": 0, "y": 1}, "output": 0}
            ],
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

    def test_checkpoint_and_result_metadata_are_isolated_as_improved_3(self):
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

            self.assertEqual(result["method"], "improved_3")
            self.assertEqual(result["run_id"], "improved_3_offline-seed")
            checkpoint_path = next(Path(checkpoint_dir).glob("checkpoint_*.json"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["method"], "improved_3")
            self.assertEqual(checkpoint["run_id"], "improved_3_offline-seed")

    def test_cli_writes_default_result_and_checkpoint_to_improved_3_directories(self):
        dataset_path = BOOLEAN_DIR / "datasets" / "boolean_2var.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "llm:\n"
                "  type: openai\n"
                "  models:\n"
                "    openai: deepseek-chat\n"
                "  api_keys:\n"
                "    openai: ''\n"
                "benchmark:\n"
                "  checkpoint: checkpoints\n"
                "  verbose: false\n",
                encoding="utf-8"
            )
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                with patch.object(sys, "argv", [
                    "boolean_benchmark_improved_3.py",
                    "--dataset", str(dataset_path),
                    "--config", str(config_path),
                    "--n-samples", "1",
                    "--n-queries", "1",
                    "--seed", "33550336",
                    "--run-id", "offline-cli"
                ]):
                    with patch.object(
                        improved_3_module,
                        "setup_llm",
                        return_value=FakeLLM(["Expression: x AND y"])
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            improved_3_module.main()
            finally:
                os.chdir(previous_cwd)

            output_path = (
                Path(temp_dir)
                / "results"
                / "boolean_2var_deepseek-chat_improved_3"
                / "boolean_2var_deepseek-chat_improved_3_seed33550336.json"
            )
            checkpoint_dir = Path(temp_dir) / "checkpoints" / "improved_3"
            checkpoint_path = next(checkpoint_dir.glob("checkpoint_*.json"))
            output = json.loads(output_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(output["method"], "improved_3")
            self.assertEqual(output["run_id"], "improved_3_offline-cli")
            self.assertEqual(checkpoint["method"], "improved_3")


if __name__ == "__main__":
    unittest.main()
