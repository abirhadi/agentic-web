"""Acceptance-contract extraction and prompt/artifact integration regressions."""
import json
from pathlib import Path
import tempfile
import unittest

from agentic_web.implementation_contract import derive_implementation_contract, render_implementation_contract
from agentic_web.schema import load_task, TaskSpec
from agentic_web.workflow import build_graph
from mock_client import MockClient, ROOT


class ImplementationContractTests(unittest.TestCase):
    def check_fixture(self, name, expected_ids):
        task = load_task(ROOT / f"tasks/{name}.json")
        before = task.model_dump(mode="json")
        contract = derive_implementation_contract(task)
        self.assertEqual(set(contract["exact_data_testids"]), expected_ids)
        self.assertEqual(len(contract["exact_data_testids"]), len(expected_ids))
        self.assertEqual(contract["required_ui_content"], task.verification.required_content)
        self.assertEqual(contract["acceptance_tests"], [test.model_dump(mode="json", exclude_none=True) for test in task.tests if test.required])
        self.assertEqual(task.model_dump(mode="json"), before)
        expected_selectors = {step.selector for test in task.tests if test.required and test.test_type == "browser"
                              for step in [*test.preconditions, *test.actions, *test.expected_results,
                                           *(assertion for action in test.actions for assertion in action.assert_after)]}
        self.assertEqual(set(contract["required_selectors"]), expected_selectors)
        return contract

    def test_eventhub_contract(self):
        contract = self.check_fixture("eventhub", {"event-card", "event-search", "category-filter", "add-event-form", "register-button", "registration-feedback"})
        self.assertIn('[data-testid="event-card"]:first-child [data-testid="register-button"]', contract["required_selectors"])
        added = next(test for test in contract["acceptance_tests"] if test["requirement_id"] == "add")
        self.assertEqual(added["preconditions"][0]["value"], 6)
        self.assertEqual(added["expected_results"][0]["value"], 7)
        self.assertIn('[name="title"]', contract["required_selectors"])

    def test_task_tracker_contract(self):
        contract = self.check_fixture("todo", {"task-title", "add-task", "task-item", "complete-task", "delete-task"})
        completion = next(test for test in contract["acceptance_tests"] if test["requirement_id"] == "complete")
        self.assertEqual(completion["actions"][-1], {"action": "check", "selector": '[data-testid="complete-task"]', "assert_after": []})
        self.assertEqual(completion["expected_results"][0], {"assertion": "checked", "selector": '[data-testid="complete-task"]'})
        self.assertEqual([r["id"] for r in contract["optional_requirements"]], ["persistence"])
        self.assertNotIn("persistence", [r["id"] for r in contract["required_requirements"]])

    def test_product_catalog_contract(self):
        contract = self.check_fixture("catalog", {"product-card", "product-search", "product-category"})
        filtered = next(test for test in contract["acceptance_tests"] if test["requirement_id"] == "filter")
        self.assertEqual(filtered["actions"][0]["value"], "Home")
        self.assertEqual(filtered["expected_results"][0]["value"], 2)
        self.assertIn('[data-testid="product-card"][data-category="Tech"]', contract["required_selectors"])

    def test_unrelated_selectors_quotes_and_css_escapes(self):
        spec = load_task(ROOT / "tasks/todo.json").model_dump(mode="json")
        spec["tests"] = [{"test_id": "custom", "requirement_id": "add", "actions": [
            {"action": "click", "selector": "section [data-testid='launch-control']"},
            {"action": "fill", "selector": "[data-testid=entry]", "value": "Example"}],
            "expected_results": [{"assertion": "visible", "selector": r'[data-testid="result\2d panel"]'}]}]
        contract = derive_implementation_contract(TaskSpec.model_validate(spec))
        self.assertEqual(contract["exact_data_testids"], ["launch-control", "entry", "result-panel"])

    def test_no_invented_selectors_and_optional_tests_stay_optional(self):
        spec = load_task(ROOT / "tasks/todo.json").model_dump(mode="json")
        spec["tests"] = [{"test_id": "optional", "requirement_id": "persistence", "required": False,
                          "expected_results": [{"assertion": "visible", "selector": '[data-testid="optional-only"]'}]}]
        contract = derive_implementation_contract(TaskSpec.model_validate(spec))
        self.assertEqual(contract["exact_data_testids"], [])
        self.assertEqual(contract["required_selectors"], [])
        self.assertEqual(contract["acceptance_tests"], [])
        self.assertEqual(contract["optional_acceptance_tests"][0]["test_id"], "optional")
        self.assertIn("add", [r["id"] for r in contract["required_requirements"]])

    def test_attribute_prefix_is_not_an_exact_test_id(self):
        spec = load_task(ROOT / "tasks/todo.json").model_dump(mode="json")
        spec["tests"] = [{"test_id": "prefix", "requirement_id": "add", "expected_results": [
            {"assertion": "visible", "selector": '[data-testid^="entry-"]'}]}]
        contract = derive_implementation_contract(TaskSpec.model_validate(spec))
        self.assertEqual(contract["exact_data_testids"], [])
        self.assertEqual(contract["required_selectors"], ['[data-testid^="entry-"]'])

    def test_prompt_is_prominent_and_artifacts_match(self):
        task = load_task(ROOT / "tasks/todo.json")
        client = MockClient()
        def verifier(path, task, output):
            return {"status": "passed"}
        with tempfile.TemporaryDirectory() as output:
            result = build_graph(client, verifier).invoke({"task_spec": task.model_dump(mode="json"), "output_root": output})
            directory = Path(result["run_dir"])
            contract = json.loads((directory / "implementation_contract.json").read_text(encoding="utf-8"))
            text = (directory / "implementation_contract.txt").read_text(encoding="utf-8")
            self.assertEqual(contract, derive_implementation_contract(task))
            self.assertEqual(text, render_implementation_contract(contract))
            system = client.calls[1]["system"]
            self.assertTrue(system.startswith(text + "\n\n"))
            for rule in ("dynamically generated", "EXACTLY", "CSS class or HTML id is not a substitute", "Do not append index suffixes", "SAME required data-testid", "data-item-id"):
                self.assertIn(rule, system)
            self.assertEqual(client.calls[1]["payload"]["original_requirements"], task.original_requirements)
            self.assertEqual(client.calls[1]["payload"]["task_spec"], task.model_dump(mode="json"))
            self.assertEqual(client.calls[1]["payload"]["plan"], result["plan"])
            self.assertFalse(client.calls[0]["system"].startswith("MANDATORY IMPLEMENTATION CONTRACT"))

    def test_contract_retained_on_implementation_failure(self):
        class FailedImplementer(MockClient):
            def complete(self, *args, planner=False, **kwargs):
                if planner:
                    return super().complete(*args, planner=True, **kwargs)
                raise ConnectionError("Implementer unavailable")
        with tempfile.TemporaryDirectory() as output:
            result = build_graph(FailedImplementer()).invoke({"task_path": str(ROOT / "tasks/todo.json"), "output_root": output})
            self.assertEqual(result["status"], "failed")
            self.assertTrue((Path(result["run_dir"]) / "implementation_contract.json").is_file())
            self.assertTrue((Path(result["run_dir"]) / "implementation_contract.txt").is_file())


if __name__ == "__main__":
    unittest.main()
