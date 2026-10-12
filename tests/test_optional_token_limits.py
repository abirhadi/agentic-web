"""Optional client limits and strict handling of server-truncated outputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
from agentic_web.client import LMStudioClient, ModelConfig
from agentic_web.reasoning import InferenceError, resolve_policy
from inference_mock import sdk as stream_sdk
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from mock_client import ROOT
from test_framework import stub_verifier


ROLES = ("planner", "ux", "architect", "coordinator", "implementer", "qa")


class OptionalTokenLimitsTests(unittest.TestCase):
    def request_for(self, config, role, finish_reason="stop"):
        sdk = stream_sdk(finish=finish_reason, usage={"prompt_tokens": 8, "completion_tokens": 13, "total_tokens": 21})
        role_config, planner = config.for_role(role)
        client = LMStudioClient()
        with patch.object(client, "_async_client", return_value=sdk):
            _, metrics = client.complete("System", "User", role_config, planner=planner)
        return sdk.chat.completions.create.call_args.kwargs, metrics

    def test_defaults_and_explicit_null_round_trip(self):
        config = ModelConfig()
        for field in ("planner_max_tokens", "review_max_tokens", "max_tokens"):
            self.assertIsNone(getattr(config, field))
        self.assertEqual(ModelConfig.model_validate_json(config.model_dump_json()), config)
        explicit = ModelConfig(planner_max_tokens=None, review_max_tokens=None, max_tokens=None)
        self.assertIsNone(explicit.max_tokens)

    def test_all_roles_apply_policy_caps_when_optional_override_unset(self):
        for role in ROLES:
            with self.subTest(role=role):
                request, metrics = self.request_for(ModelConfig(), role)
                self.assertGreater(request["max_tokens"], 0)
                self.assertNotIn("max_completion_tokens", request)
                self.assertEqual(metrics["max_tokens"], request["max_tokens"])
                self.assertEqual(metrics["usage"]["completion_tokens"], 13)
                self.assertEqual(metrics["finish_reason"], "stop")
                self.assertFalse(metrics["incomplete_generation"])

    def test_all_roles_preserve_explicit_positive_limits(self):
        config = ModelConfig(planner_max_tokens=256, review_max_tokens=768, max_tokens=1024)
        for role in ROLES:
            with self.subTest(role=role):
                expected = 256 if role == "planner" else 1024 if role == "implementer" else 768
                request, metrics = self.request_for(config, role)
                self.assertEqual(request["max_tokens"], expected)
                self.assertEqual(metrics["max_tokens"], expected)

    def test_unset_planner_and_reviews_do_not_inherit_implementation_limit(self):
        config = ModelConfig(max_tokens=512)
        for role in ROLES:
            request, _ = self.request_for(config, role)
            if role == "implementer":
                self.assertEqual(request["max_tokens"], 512)
            else:
                self.assertGreater(request["max_tokens"], 0)

    def test_nonpositive_limits_are_rejected(self):
        for field in ("planner_max_tokens", "review_max_tokens", "max_tokens"):
            for value in (0, -1):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    ModelConfig(**{field: value})

    def test_length_finish_reason_is_recorded_with_actual_usage(self):
        with self.assertRaises(InferenceError) as caught:
            self.request_for(ModelConfig(), "qa", "length")
        metrics = caught.exception.diagnostics
        self.assertTrue(metrics["incomplete_generation"])
        self.assertEqual(metrics["error_classification"], "output_budget_exhausted")
        self.assertEqual(metrics["finish_reason"], "length")
        self.assertEqual(metrics["usage"], {"prompt_tokens": 8, "completion_tokens": 13, "total_tokens": 21})

    def test_cli_optional_defaults_and_explicit_limits(self):
        from agentic_web.__main__ import main
        for workflow in ("baseline", "collaborative"):
            for flags, limits in [([], (None, None, None)),
                                  (["--planner-max-tokens", "100", "--review-max-tokens", "200", "--max-tokens", "300"], (100, 200, 300))]:
                with self.subTest(workflow=workflow, flags=flags):
                    graph_name = "app" if workflow == "baseline" else "collaborative_app"
                    command = ["agentic_web", "generate", "--task", str(ROOT / "tasks/todo.json"), "--workflow", workflow, *flags]
                    with patch("sys.argv", command), patch(f"agentic_web.__main__.{graph_name}") as graph, patch("builtins.print"):
                        graph.invoke.return_value = {"status": "passed", "run_dir": "mock-run"}
                        self.assertEqual(main(), 0)
                        config = graph.invoke.call_args.args[0]["model_config"]
                        self.assertEqual(tuple(config[field] for field in ("planner_max_tokens", "review_max_tokens", "max_tokens")), limits)


class TruncationArtifactTests(unittest.TestCase):
    def test_complete_looking_outputs_with_length_are_rejected_for_every_role(self):
        for target in ROLES:
            with self.subTest(role=target), tempfile.TemporaryDirectory() as directory:
                class LengthClient(CollaborationMockClient):
                    def complete(self, system, user, config, *, planner=False, response_schema=None):
                        role = json.loads(user).get("role", "planner" if planner else "implementer")
                        raw, metrics = super().complete(system, user, config, planner=planner, response_schema=response_schema)
                        if role == target:
                            metrics.update(finish_reason="length", incomplete_generation=True)
                        return raw, metrics
                client = LengthClient()
                result = build_graph(client, stub_verifier, workflow="collaborative").invoke({
                    "task_path": str(ROOT / "tasks/todo.json"), "output_root": directory, "execution_mode": "mocked"})
                root = Path(result["run_dir"])
                metrics = json.loads((root / f"{target}_metrics.json").read_text())
                self.assertIsNone(metrics["max_tokens"])
                self.assertEqual(metrics["finish_reason"], "length")
                self.assertTrue(metrics["incomplete_generation"])
                self.assertTrue((root / f"{target}_raw.txt").is_file())
                validated = {"planner": "plan.json", "ux": "ux_review.json", "architect": "architecture_review.json",
                             "coordinator": "coordination.json", "implementer": "application/index.html", "qa": "qa_review.json"}[target]
                self.assertFalse((root / validated).exists(), validated)
                self.assertEqual(client.roles.count(target), 1)
                if target == "qa":
                    self.assertEqual(result["qa_status"], "failed")
                    self.assertIn("Incomplete qa output", result["qa_error"])
                    self.assertTrue(result["acceptance"]["accepted"])
                else:
                    self.assertEqual(result["status"], "failed")
                    self.assertFalse(result["acceptance"]["accepted"])

    def test_baseline_planner_length_is_rejected_before_implementation(self):
        class LengthPlanner(CollaborationMockClient):
            def complete(self, *args, **kwargs):
                raw, metrics = super().complete(*args, **kwargs)
                metrics["finish_reason"] = "length"
                return raw, metrics
        with tempfile.TemporaryDirectory() as directory:
            client = LengthPlanner()
            result = build_graph(client, stub_verifier).invoke({"task_path": str(ROOT / "tasks/todo.json"), "output_root": directory})
            self.assertEqual(result["status"], "failed")
            self.assertIn("Incomplete planner output", result["error"])
            self.assertEqual(client.roles, ["planner"])
            self.assertFalse((Path(result["run_dir"]) / "plan.json").exists())
            metrics = json.loads((Path(result["run_dir"]) / "planner_metrics.json").read_text())
            self.assertTrue(metrics["incomplete_generation"])


if __name__ == "__main__":
    unittest.main()
