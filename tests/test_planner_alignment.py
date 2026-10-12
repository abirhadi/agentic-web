"""Mandatory coverage is strict; optional planning is genuinely optional."""
import json
from pathlib import Path
import tempfile
import unittest

from agentic_web.schema import load_task
from agentic_web.workflow import build_graph
from mock_client import MockClient, ROOT
from test_framework import stub_verifier


class PlannerAlignmentTests(unittest.TestCase):
    def run_plan(self, ids):
        class PlannerClient(MockClient):
            def complete(self, *args, planner=False, **kwargs):
                raw, metrics = super().complete(*args, planner=planner, **kwargs)
                if planner:
                    plan = json.loads(raw)
                    plan["requirement_ids"] = ids
                    raw = json.dumps(plan)
                return raw, metrics
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = PlannerClient()
        self.task = load_task(ROOT / "tasks/todo.json")
        result = build_graph(self.client, stub_verifier).invoke({
            "task_spec": self.task.model_dump(mode="json"), "output_root": self.temp.name})
        report = json.loads((Path(result["run_dir"]) / "planner_alignment.json").read_text())
        return result, report

    def test_optional_omission_proceeds_with_unchanged_task(self):
        ids = ["add", "complete", "delete", "responsive", "accessible"]
        result, report = self.run_plan(ids)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["plan"]["requirement_ids"], ids)
        self.assertEqual(result["task_spec"], self.task.model_dump(mode="json"))
        self.assertEqual(self.client.calls[1]["payload"]["task_spec"], self.task.model_dump(mode="json"))
        self.assertEqual(report["unplanned_optional_ids"], ["persistence"])
        self.assertEqual(self.client.calls[0]["payload"]["requirement_id_contract"], {
            "required": ids, "optional": ["persistence"]})

    def test_missing_mandatory_id_is_rejected(self):
        result, report = self.run_plan(["add", "complete", "delete", "responsive"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(report["missing_required_ids"], ["accessible"])
        self.assertIn("missing_required=['accessible']", result["error"])
        self.assertEqual(len(self.client.calls), 1)

    def test_unknown_id_is_rejected(self):
        result, report = self.run_plan(["add", "complete", "delete", "responsive", "accessible", "invented"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(report["unknown_ids"], ["invented"])

    def test_duplicate_id_is_rejected(self):
        result, report = self.run_plan(["add", "complete", "delete", "responsive", "accessible", "add"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(report["duplicate_ids"], ["add"])


if __name__ == "__main__":
    unittest.main()
