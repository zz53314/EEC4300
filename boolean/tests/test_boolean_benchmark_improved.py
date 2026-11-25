import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BOOLEAN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOOLEAN_DIR))

import boolean.boolean_benchmark_improved_1 as improved_module
from boolean.boolean_benchmark_improved_1 import BooleanBenchmarkImproved


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def get_name(self):
        return "FakeLLM"

    def query_with_usage(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return {
            "response": response,
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3
            },
            "cost": 0.0
        }


def observation_set(identifier, inputs, output, ground_truths=None):
    return {
        "observation_set_id": identifier,
        "n_observations": 1,
        "observations": [{"inputs": inputs, "output": output}],
        "ground_truth_expressions": ground_truths or []
    }


class ImprovedBooleanBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_path = BOOLEAN_DIR / "datasets" / "boolean_2var.json"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.benchmark = BooleanBenchmarkImproved(str(dataset_path))

    def test_only_valid_unique_candidates_are_archived_and_repeats_are_reported(self):
        llm = FakeLLM([
            "Expression: x AND y",
            "Expression: y AND x",
            "Expression: x OR y"
        ])
        sample = observation_set("archive-test", {"x": 1, "y": 1}, 1)

        result = self.benchmark.evaluate_single_observation_set(llm, sample, n_queries=3)

        self.assertEqual(result["valid_unique_archive"], ["x AND y", "x OR y"])
        self.assertIn("already archived candidate", result["feedback_history"][1]["feedback"])
        self.assertIn("Expression: x AND y", llm.prompts[1])
        self.assertIn("already archived candidate", llm.prompts[2])
        self.assertEqual(result["n_queries"], 3)
        self.assertEqual(result["n_valid"], 3)
        self.assertEqual(result["n_unique_valid"], 2)
        self.assertEqual(result["n_unique_all"], 2)
        self.assertEqual(result["valid_rate"], 1.0)
        self.assertAlmostEqual(result["novelty_rate"], 2 / 3)
        self.assertEqual(len(result["raw_responses"]), 3)
        original_result_fields = {
            "observation_set_id", "n_observations", "n_ground_truths", "n_queries",
            "n_valid", "n_unique_valid", "n_unique_all", "n_recovered_gts",
            "parse_success_rate", "in_space_rate", "valid_rate", "novelty_rate",
            "recovery_rate", "all_hypotheses", "valid_hypotheses",
            "unique_valid_expressions", "unique_all_expressions", "token_usage",
            "cost", "errors", "error_summary"
        }
        self.assertTrue(original_result_fields.issubset(result))
        prior_block = re.search(
            r"Prior expressions generated.*?:\s*(.*?)\n\s*Task:",
            llm.prompts[1],
            flags=re.DOTALL
        )
        self.assertIsNotNone(prior_block)
        self.assertEqual(prior_block.group(1).strip(), "Expression: x AND y")

    def test_invalid_candidate_is_recorded_not_archived_and_feedback_uses_observations(self):
        llm = FakeLLM([
            "Expression: x OR y",
            "Expression: y"
        ])
        sample = observation_set("invalid-test", {"x": 1, "y": 0}, 0)

        result = self.benchmark.evaluate_single_observation_set(llm, sample, n_queries=2)

        self.assertEqual(result["valid_unique_archive"], ["y"])
        self.assertEqual(result["all_hypotheses"], ["x OR y", "y"])
        self.assertIn("does not match observation 1", result["feedback_history"][0]["feedback"])
        self.assertIn("does not match observation 1", llm.prompts[1])
        self.assertIn("n_valid", result)
        self.assertIn("novelty_rate", result)
        self.assertIn("recovery_rate", result)
        self.assertAlmostEqual(result["valid_rate"], 0.5)
        self.assertAlmostEqual(result["novelty_rate"], 1.0)

    def test_operator_diagnostic_excludes_candidate_from_archive(self):
        llm = FakeLLM(["Expression: NOT x", "Expression: x AND y"])
        sample = observation_set("operator-test", {"x": 1, "y": 1}, 1)

        result = self.benchmark.evaluate_single_observation_set(llm, sample, n_queries=2)

        self.assertIn("operator outside the allowed", result["feedback_history"][0]["feedback"])
        self.assertEqual(result["valid_unique_archive"], ["x AND y"])

    def test_unparseable_response_is_recorded_and_feedback_is_used_for_retry(self):
        llm = FakeLLM(["???!", "Expression: x AND y"])
        sample = observation_set("retry-test", {"x": 1, "y": 1}, 1)

        result = self.benchmark.evaluate_single_observation_set(
            llm, sample, n_queries=1, max_retries=2
        )

        self.assertEqual(len(result["raw_responses"]), 2)
        self.assertEqual(result["raw_responses"][0]["response"], "???!")
        self.assertIn("parseable Boolean expression", llm.prompts[1])
        self.assertEqual(result["n_queries"], 1)
        self.assertEqual(result["valid_rate"], 1.0)

    def test_default_attempt_limit_records_failure_without_extra_api_retries(self):
        error = "Error querying OpenAI: simulated local failure"
        llm = FakeLLM([error, error])
        sample = observation_set("request-error-test", {"x": 1, "y": 1}, 1)

        result = self.benchmark.evaluate_single_observation_set(llm, sample, n_queries=1)

        self.assertEqual(len(result["raw_responses"]), 1)
        self.assertEqual(result["n_queries"], 1)
        self.assertEqual(result["valid_rate"], 0.0)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["error_summary"]["error_types"]["unknown_error"], 1)
        self.assertIn("[ERROR after 1 attempts]", result["all_hypotheses"][0])

    def test_openai_compatible_client_internal_retries_are_disabled(self):
        class FakeClient:
            def __init__(self):
                self.options = None

            def with_options(self, **kwargs):
                self.options = kwargs
                return self

        class FakeOpenAILLM:
            def __init__(self, **kwargs):
                self.client = FakeClient()

        with patch.object(improved_module, "OpenAILLM", FakeOpenAILLM):
            llm = improved_module.setup_llm(
                "openai",
                api_key="dummy",
                model="deepseek-chat",
                temperature=0.7,
                base_url="https://api.deepseek.com"
            )

        self.assertEqual(llm.client.options, {"max_retries": 0})

    def test_recovery_metric_still_uses_structural_ground_truth_keys_offline(self):
        expression = "x AND y"
        ground_truth_key = self.benchmark.get_expression_mechanistic_key(expression)
        llm = FakeLLM(["Expression: x AND y"])
        sample = observation_set(
            "recovery-test",
            {"x": 1, "y": 1},
            1,
            ground_truths=[{"mechanistic_key": ground_truth_key}]
        )

        result = self.benchmark.evaluate_single_observation_set(llm, sample, n_queries=1)

        self.assertEqual(result["n_recovered_gts"], 1)
        self.assertEqual(result["recovery_rate"], 1.0)

    def test_improved_method_is_marked_and_checkpoint_is_separate(self):
        llm = FakeLLM(["Expression: x AND y"])
        with tempfile.TemporaryDirectory() as temp_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.benchmark.run_benchmark(
                    llm=llm,
                    n_samples=1,
                    n_queries_per_sample=1,
                    seed=33550336,
                    verbose=False,
                    checkpoint_dir=temp_dir,
                    run_id="offline-test"
                )

            self.assertEqual(result["method"], "improved")
            self.assertEqual(result["run_id"], "improved_offline-test")
            checkpoint_paths = list(Path(temp_dir).glob("checkpoint_*.json"))
            self.assertEqual(len(checkpoint_paths), 1)

    def test_interrupted_run_resumes_from_last_saved_response_without_repeating_it(self):
        run_id = "resume-after-response"
        with tempfile.TemporaryDirectory() as temp_dir:
            first_llm = FakeLLM([
                "Expression: x AND y",
                RuntimeError("simulated interruption")
            ])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                interrupted_result = self.benchmark.run_benchmark(
                    llm=first_llm,
                    n_samples=1,
                    n_queries_per_sample=2,
                    seed=33550336,
                    verbose=False,
                    checkpoint_dir=temp_dir,
                    run_id=run_id
                )

            self.assertFalse(interrupted_result["run_complete"])
            checkpoint_path = next(Path(temp_dir).glob("checkpoint_*.json"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            progress = checkpoint["current_sample"]["progress"]
            self.assertEqual(progress["query_index"], 1)
            self.assertEqual(len(progress["raw_responses"]), 1)

            resumed_llm = FakeLLM(["Expression: x OR y"])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                resumed_result = self.benchmark.run_benchmark(
                    llm=resumed_llm,
                    n_samples=1,
                    n_queries_per_sample=2,
                    seed=33550336,
                    verbose=False,
                    checkpoint_dir=temp_dir,
                    run_id=run_id
                )

            self.assertTrue(resumed_result["run_complete"])
            sample_result = resumed_result["per_sample_results"][0]
            self.assertEqual(len(sample_result["raw_responses"]), 2)
            self.assertEqual(sample_result["raw_responses"][0]["response"], "Expression: x AND y")
            self.assertEqual(sample_result["raw_responses"][1]["response"], "Expression: x OR y")
            self.assertEqual(resumed_result["token_usage"]["total_tokens"], 6)
            self.assertEqual(len(resumed_llm.prompts), 1)

    def test_legacy_checkpoint_keeps_completed_samples_without_api_calls(self):
        seed = 33550336
        sample = self.benchmark.sample_observation_sets(1, seed)[0]
        result = self.benchmark.evaluate_single_observation_set(
            FakeLLM(["Expression: x AND y"]), sample, n_queries=1
        )
        run_id = "improved_legacy-run"

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_FakeLLM_improved_legacy-run.json"
            checkpoint_path.write_text(json.dumps({
                "run_id": run_id,
                "llm_name": "FakeLLM",
                "n_samples": 1,
                "n_queries_per_sample": 1,
                "query_multiplier": None,
                "seed": seed,
                "results": [result],
                "total_token_usage": result["token_usage"],
                "total_cost": result["cost"],
                "total_errors": result["error_summary"]["total_errors"],
                "error_types": result["error_summary"]["error_types"]
            }), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                resumed = self.benchmark.run_benchmark(
                    llm=FakeLLM([]),
                    n_samples=1,
                    n_queries_per_sample=1,
                    seed=seed,
                    verbose=False,
                    checkpoint_dir=temp_dir,
                    run_id=run_id
                )

        self.assertTrue(resumed["run_complete"])
        self.assertEqual(resumed["n_samples"], 1)
        self.assertEqual(resumed["token_usage"]["total_tokens"], 3)

    def test_archive_is_isolated_between_observation_sets(self):
        first_llm = FakeLLM(["Expression: x AND y"])
        second_llm = FakeLLM(["Expression: x OR y"])

        first_result = self.benchmark.evaluate_single_observation_set(
            first_llm,
            observation_set("first-question", {"x": 1, "y": 1}, 1),
            n_queries=1
        )
        second_result = self.benchmark.evaluate_single_observation_set(
            second_llm,
            observation_set("second-question", {"x": 1, "y": 1}, 1),
            n_queries=1
        )

        self.assertEqual(first_result["valid_unique_archive"], ["x AND y"])
        self.assertEqual(second_result["valid_unique_archive"], ["x OR y"])
        self.assertIn("Prior expressions generated", second_llm.prompts[0])
        prior_block = re.search(
            r"Prior expressions generated.*?:\s*(.*?)\n\s*Task:",
            second_llm.prompts[0],
            flags=re.DOTALL
        )
        self.assertIsNotNone(prior_block)
        self.assertEqual(prior_block.group(1).strip(), "None")

    def test_ground_truth_sentinel_is_not_used_in_prompts_or_feedback(self):
        sentinel = "GROUND_TRUTH_REFERENCE_SENTINEL"
        llm = FakeLLM(["Expression: x AND y"])
        sample = observation_set(
            "no-leak-test",
            {"x": 1, "y": 1},
            1,
            ground_truths=[{"expression": sentinel, "mechanistic_key": None}]
        )

        result = self.benchmark.evaluate_single_observation_set(llm, sample, n_queries=1)
        inspected = (
            "\n".join(llm.prompts)
            + "\n"
            + str(result["feedback_history"])
            + "\n"
            + str(result["valid_unique_archive"])
        )

        self.assertNotIn(sentinel, inspected)
        self.assertEqual(result["n_ground_truths"], 1)


if __name__ == "__main__":
    unittest.main()
