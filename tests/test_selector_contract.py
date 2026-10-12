"""Selector cardinality comes from approved tests, including intermediate states."""
import json
from pathlib import Path
import tempfile
import unittest

from agentic_web.implementation_contract import derive_implementation_contract, derive_selector_contract, render_implementation_contract
from agentic_web.schema import TaskSpec, load_task
from agentic_web.verification import verify_application
from mock_client import ROOT


class SelectorContractTests(unittest.TestCase):
    def test_fixture_controls_are_unique_at_use_and_collections_can_repeat(self):
        fixtures = [("eventhub", "event-search", "event-card"),
                    ("catalog", "product-search", "product-card"),
                    ("todo", "task-title", "task-item")]
        for name, control, collection in fixtures:
            with self.subTest(task=name):
                task = load_task(ROOT / f"tasks/{name}.json")
                original = task.model_dump(mode="json")
                entries = {entry["selector"]: entry for entry in derive_selector_contract(task)}
                target = entries[f'[data-testid="{control}"]']
                self.assertEqual(target["cardinality"], "single_at_use")
                self.assertIn("input, textarea or editable control", target["control_roles"])
                self.assertTrue(all(use["matches"] == 1 for use in target["single_element_uses"]))
                cards = entries[f'[data-testid="{collection}"]']
                self.assertEqual(cards["cardinality"], "single_at_use" if name == "todo" else "state_dependent")
                self.assertEqual(task.model_dump(mode="json"), original)

    def test_collection_cardinality_and_scoped_control_are_distinct(self):
        spec = load_task(ROOT / "tasks/todo.json").model_dump(mode="json")
        spec["tests"] = [{"test_id": "unrelated", "requirement_id": "add", "preconditions": [
            {"assertion": "count", "selector": '[data-testid="record"]', "value": 3}],
            "actions": [{"action": "check", "selector": '[data-testid="record"]:first-child [data-testid="flag"]',
                         "assert_after": [{"assertion": "count", "selector": '[data-testid="new-control"]', "value": 1}]}],
            "expected_results": [{"assertion": "count", "selector": '[data-testid="record"]', "value": 3}]}]
        contract = derive_implementation_contract(TaskSpec.model_validate(spec))
        entries = {entry["selector"]: entry for entry in contract["selector_contract"]}
        self.assertEqual(entries['[data-testid="record"]']["cardinality"], "multiple")
        self.assertEqual(entries['[data-testid="record"]:first-child [data-testid="flag"]']["cardinality"], "single_at_use")
        intermediate = entries['[data-testid="new-control"]']
        self.assertEqual(intermediate["count_expectations"][0]["phase"], "action[0].assert_after[0]")
        self.assertIn("new-control", contract["exact_data_testids"])

    def test_absent_and_hidden_selectors_do_not_gain_existence_requirements(self):
        spec = load_task(ROOT / "tasks/todo.json").model_dump(mode="json")
        spec["tests"] = [{"test_id": "empty", "requirement_id": "delete", "expected_results": [
            {"assertion": "count", "selector": ".empty", "value": 0},
            {"assertion": "hidden", "selector": ".notice"}]}]
        entries = {entry["selector"]: entry for entry in derive_selector_contract(TaskSpec.model_validate(spec))}
        self.assertEqual(entries[".empty"]["cardinality"], "count_only")
        self.assertEqual(entries[".notice"]["single_element_uses"][0]["matches"], "zero_or_one")

    def test_optional_selectors_are_not_promoted_to_mandatory(self):
        spec = load_task(ROOT / "tasks/todo.json").model_dump(mode="json")
        spec["tests"] = [{"test_id": "optional", "requirement_id": "persistence", "required": False,
                          "expected_results": [{"assertion": "visible", "selector": "#optional-control"}]}]
        self.assertEqual(derive_selector_contract(TaskSpec.model_validate(spec)), [])

    def test_prompt_prominently_describes_uniqueness_and_accessibility_lifecycle(self):
        contract = derive_implementation_contract(load_task(ROOT / "tasks/todo.json"))
        prompt = render_implementation_contract(contract)
        for instruction in ("SELECTOR UNIQUENESS", "COLLECTIONS", "ACCESSIBILITY LIFECYCLE", "initial page load AND after every DOM-changing interaction",
                            "Do not reuse a control's test ID on display text", "dynamically created controls", "EXACTLY"):
            self.assertIn(instruction, prompt[:3500])
        self.assertIn('"[data-testid=\\"task-title\\"]": single_at_use', prompt)


class DynamicControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.task = load_task(ROOT / "tasks/todo.json")
        self.task.verification.timeout_ms = 200
        self.task.verification.screenshots = False
        self.task.verification.required_content = []

    def verify_html(self, html, tests, suffix):
        spec = self.task.model_dump(mode="json")
        spec["tests"] = tests
        task = TaskSpec.model_validate(spec)
        root = Path(self.temp.name) / suffix
        root.mkdir()
        path = root / "index.html"
        path.write_text(html, encoding="utf-8")
        return verify_application(path, task, root / "report")

    def test_all_dynamic_control_types_require_names_after_interaction(self):
        for index, control in enumerate(['<input type="checkbox">', '<input>', '<select><option>Choice</option></select>', '<button></button>', '<input type="button">']):
            with self.subTest(control=control):
                html = '<!DOCTYPE html><html lang="en"><head><title>Controls</title></head><body><button id="add">Add control</button><section id="result"></section><script>'
                html += 'document.querySelector("#add").onclick=()=>{document.querySelector("#result").innerHTML=' + json.dumps(control) + ';};</script></body></html>'
                report = self.verify_html(html, [{"test_id": "dynamic", "requirement_id": "add", "actions": [
                    {"action": "click", "selector": "#add"}], "expected_results": [
                    {"assertion": "count", "selector": "#result > *", "value": 1}]}], f"dynamic-{index}")
                self.assertTrue(report["checks"]["desktop_basic_accessibility"])
                self.assertTrue(report["checks"]["mobile_basic_accessibility"])
                self.assertEqual(report["tests"][0]["status"], "failed", report)
                self.assertEqual(report["tests"][0]["phase"], "accessibility_after_action[0]")
                self.assertIn("missing accessible", report["tests"][0]["error"])

    def test_dynamic_named_controls_pass(self):
        controls = '<label>Choose<input type="checkbox"></label><label>Title<input></label><label>Category<select><option>Choice</option></select></label><button>Delete</button><input type="button" value="Save">'
        html = '<!DOCTYPE html><html lang="en"><head><title>Controls</title></head><body><button id="add">Add controls</button><section id="result"></section><script>'
        html += 'document.querySelector("#add").onclick=()=>{document.querySelector("#result").innerHTML=' + json.dumps(controls) + ';};</script></body></html>'
        report = self.verify_html(html, [{"test_id": "named", "requirement_id": "add", "actions": [
            {"action": "click", "selector": "#add"}], "expected_results": [
            {"assertion": "count", "selector": "#result input, #result select, #result button", "value": 5}]}], "named")
        self.assertEqual(report["tests"][0]["status"], "passed", report)

    def test_ambiguous_unique_input_locator_fails(self):
        html = '<!DOCTYPE html><html lang="en"><head><title>Editor</title></head><body><label>Title<input data-testid="entry"></label><span data-testid="entry">Display text</span></body></html>'
        report = self.verify_html(html, [{"test_id": "unique", "requirement_id": "add", "actions": [
            {"action": "fill", "selector": '[data-testid="entry"]', "value": "Record"}], "expected_results": [
            {"assertion": "value", "selector": '[data-testid="entry"]', "value": "Record"}]}], "duplicate")
        self.assertEqual(report["tests"][0]["status"], "failed")
        self.assertEqual(report["tests"][0]["phase"], "action[0]")
        self.assertIn("strict mode violation", report["tests"][0]["error"])


if __name__ == "__main__":
    unittest.main()
