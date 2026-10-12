"""Console progress must preserve JSON output, failures and resource ownership."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_web.__main__ import main
from agentic_web.client import ModelConfig
from agentic_web.progress import logger, observed_node
from agentic_web.resource_scheduler import ResourceScheduler
from agentic_web.resource_schema import ResourceSettings
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from mock_client import MockClient, ROOT
from resource_mock_backend import SimulatedBackend
from test_framework import stub_verifier


class ProgressLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = logger.handlers[:], logger.level, logger.propagate
        def restore_logger():
            for handler in logger.handlers[:]:
                if handler not in previous[0]:
                    logger.removeHandler(handler)
                    handler.close()
            logger.handlers = previous[0]
            logger.setLevel(previous[1])
            logger.propagate = previous[2]
        self.addCleanup(restore_logger)

    def cli(self, client=None, workflow="baseline", extra=()):
        graph = build_graph(client or CollaborationMockClient(), stub_verifier, workflow=workflow)
        command = ["agentic_web", "generate", "--task", str(ROOT / "tasks/todo.json"),
                   "--output", self.temp.name, "--workflow", workflow, *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        graph_name = "collaborative_app" if workflow == "collaborative" else "app"
        with patch("sys.argv", command), patch(f"agentic_web.__main__.{graph_name}", graph), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = main()
        return code, json.loads(stdout.getvalue()), stderr.getvalue()

    def test_console_reports_stages_and_metrics_without_polluting_json(self):
        class UsageClient(CollaborationMockClient):
            def complete(self, *args, **kwargs):
                raw, metrics = super().complete(*args, **kwargs)
                metrics.update(elapsed_seconds=2.0,
                    usage={"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160})
                return raw, metrics
        for workflow in ("baseline", "collaborative"):
            with self.subTest(workflow=workflow):
                code, result, console = self.cli(UsageClient(), workflow)
                self.assertEqual(code, 0)
                self.assertEqual(result["status"], "passed")
                roles = ["task_loader", "planner", "implementer", "verifier"]
                if workflow == "collaborative":
                    roles += ["ux", "architect", "coordinator", "qa", "acceptance_gate"]
                for role in roles:
                    self.assertIn(f'stage="{role}" event="started"', console)
                self.assertIn('event="inference_finished"', console)
                self.assertIn("completion_tokens=40", console)
                self.assertIn("completion_tokens_per_second=20.000", console)
                self.assertIn('finish_reason="stop"', console)
                self.assertIn('max_tokens=6144', console)
                self.assertIn('max_tokens=16000', console)
                self.assertIn('reasoning_mode="auto"', console)
                self.assertIn('policy_hash=', console)
                self.assertIn('event="test_result"', console)
                self.assertIn('event="finished"', console)
                self.assertIn(Path(result["run_dir"]).name, console)
                self.assertNotIn("<!DOCTYPE html>", console)
                self.assertNotIn("Build an accessible responsive offline personal Task Tracker.", console)

    def test_error_log_level_suppresses_progress_for_a_successful_run(self):
        code, result, console = self.cli(extra=("--log-level", "ERROR"))
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(console, "")

    def test_qa_validation_failure_is_visible_without_changing_acceptance(self):
        code, result, console = self.cli(CollaborationMockClient(malformed="qa"), "collaborative")
        self.assertEqual(code, 0)
        self.assertEqual(result["qa_status"], "failed")
        self.assertTrue(result["acceptance"]["accepted"])
        self.assertIn('stage="qa" event="failed"', console)
        self.assertIn("ValidationError", console)
        self.assertIn('stage="acceptance_gate" event="accepted"', console)

    def test_truncated_response_logs_finish_reason_and_skipped_stages(self):
        class Truncated(MockClient):
            def complete(self, *args, **kwargs):
                raw, metrics = super().complete(*args, **kwargs)
                metrics["finish_reason"] = "length"
                return raw, metrics
        code, result, console = self.cli(Truncated())
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn('finish_reason="length"', console)
        self.assertIn('stage="planner" event="failed"', console)
        self.assertIn('stage="implementer" event="skipped"', console)
        self.assertIn('stage="verifier" event="skipped"', console)
        self.assertFalse(Path(result["run_dir"], "application/index.html").exists())

    def test_resource_progress_retains_owned_model_cleanup(self):
        backend = SimulatedBackend()
        scheduler = ResourceScheduler(backend, backend.hardware)
        settings = ResourceSettings(context_length=4096, gpu_safety_bytes=0, ram_safety_bytes=0,
                                    estimate_padding=0, prefetch_all_if_gpu_fit=False)
        with self.assertLogs("agentic_web", level="INFO") as captured:
            result = build_graph(MockClient(), stub_verifier, resource_scheduler=scheduler).invoke({
                "task_path": str(ROOT / "tasks/todo.json"), "output_root": self.temp.name,
                "model_config": ModelConfig(model="tiny").model_dump(), "resource_config": settings.model_dump()})
        console = "\n".join(captured.output)
        self.assertEqual(result["status"], "passed")
        for event in ("hardware", "load_started", "loaded", "reused", "prepared", "unloaded"):
            self.assertIn(f'event="{event}"', console)
        self.assertIn('device="0"', console)
        self.assertIn("ram_free_gib=", console)
        self.assertIn('stage="resource_cleanup" event="completed"', console)
        self.assertEqual(backend.instances, {})
        cleanup = json.loads(Path(result["run_dir"], "resources.cleanup.json").read_text())
        self.assertEqual(len(cleanup["unloaded_instances"]), 1)

    def test_instrumentation_preserves_uncaught_exception(self):
        error = RuntimeError("A diagnostic\nwith a second line")
        def fail(state):
            raise error
        with self.assertLogs("agentic_web", level="INFO") as captured, self.assertRaises(RuntimeError) as caught:
            observed_node("verifier", fail)({"run_dir": "run-example"})
        self.assertIs(caught.exception, error)
        self.assertIn('event="failed"', captured.output[-1])
        self.assertIn("diagnostic\\nwith", captured.output[-1])


if __name__ == "__main__":
    unittest.main()
