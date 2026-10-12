"""Regression tests for false-positive deletion and inaccessible generated DOM."""
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError
from agentic_web.implementation_contract import derive_implementation_contract, render_implementation_contract
from agentic_web.schema import Action, Assertion, TaskSpec, load_task
from agentic_web.verification import verify_application
from mock_client import ROOT


class AcceptanceAdherenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.task = load_task(ROOT / "tasks/todo.json")
        self.task.verification.timeout_ms = 200
        self.task.verification.screenshots = False
        self.source = (ROOT / "tests/fixtures/todo.html").read_text(encoding="utf-8")
        self.sequence = 0

    def verify_mutation(self, old, new, test_id="delete-acceptance"):
        self.assertIn(old, self.source)
        self.task.tests = [test for test in self.task.tests if test.test_id == test_id]
        self.assertEqual(len(self.task.tests), 1)
        self.sequence += 1
        root = Path(self.temp.name) / str(self.sequence)
        root.mkdir()
        path = root / "index.html"
        path.write_text(self.source.replace(old, new), encoding="utf-8")
        report = verify_application(path, self.task, root / "verification")
        self.assertEqual(report["status"], "failed", report)
        self.assertEqual(report["tests"][0]["status"], "failed", report)
        return report, report["tests"][0]

    def test_missing_item_selector_cannot_pass_delete(self):
        report, result = self.verify_mutation("item.dataset.testid='task-item';", "item.className='task-item';")
        self.assertEqual(result["phase"], "action[1].assert_after[0]")
        self.assertIn("Actual value: 0", result["error"])
        self.assertEqual(report["coverage"]["delete"]["status"], "failed")

    def test_placeholder_is_not_an_accessible_label(self):
        report, result = self.verify_mutation('<label>Task title<input data-testid="task-title" required></label>',
            '<input data-testid="task-title" placeholder="Task title" required>', "add-acceptance")
        self.assertFalse(report["checks"]["desktop_basic_accessibility"])
        self.assertFalse(report["checks"]["mobile_basic_accessibility"])
        self.assertIn("placeholder is not a label", result["error"])
        self.assertEqual(result["phase"], "precondition[2]")

    def test_unnamed_dynamic_checkbox_fails_checkpoint(self):
        _, result = self.verify_mutation('label.append(box,document.createTextNode(input.value));',
            'label.append(box);item.append(document.createTextNode(input.value));')
        self.assertEqual(result["phase"], "action[1].assert_after[4]")
        self.assertIn("accessible label", result["error"])

    def test_dynamic_accessibility_is_checked_without_explicit_name_assertions(self):
        # Prove Layer A also inspects controls after interaction, independently
        # of the task-specific accessible_name assertions.
        add = next(test for test in self.task.tests if test.test_id == "add-acceptance")
        add.preconditions = []
        add.expected_results = [Assertion(assertion="count", selector='[data-testid="task-item"]', value=1)]
        report, result = self.verify_mutation('label.append(box,document.createTextNode(input.value));',
            'label.append(box);item.append(document.createTextNode(input.value));', "add-acceptance")
        self.assertTrue(report["checks"]["desktop_basic_accessibility"])
        self.assertEqual(result["phase"], "accessibility_after_action[1]")
        self.assertIn("missing accessible label", result["error"])

    def test_input_test_id_reused_on_display_text_is_rejected(self):
        _, result = self.verify_mutation('label.append(box,document.createTextNode(input.value));',
            "label.append(box);const span=document.createElement('span');span.dataset.testid='task-title';span.textContent=input.value;label.append(span);")
        self.assertEqual(result["phase"], "action[1].assert_after[2]")
        self.assertIn("Actual value: 2", result["error"])

    def test_indexed_dynamic_test_ids_are_rejected(self):
        for original, indexed, checkpoint in [("box.dataset.testid='complete-task';", "box.dataset.testid='complete-task-0';", 3),
                                                ("del.dataset.testid='delete-task';", "del.dataset.testid='delete-task-0';", 6)]:
            with self.subTest(selector=original):
                _, result = self.verify_mutation(original, indexed)
                self.assertEqual(result["phase"], f"action[1].assert_after[{checkpoint}]")

    def test_completion_requires_initial_unchecked_state(self):
        _, result = self.verify_mutation("box.type='checkbox';", "box.type='checkbox';box.checked=true;", "complete-acceptance")
        self.assertEqual(result["phase"], "action[1].assert_after[5]")

    def test_delete_noop_fails_after_establishing_item(self):
        _, result = self.verify_mutation('del.onclick=()=>item.remove();', 'del.onclick=()=>{};')
        self.assertEqual(result["phase"], "expected_result[0]")
        self.assertIn("Actual value: 1", result["error"])

    def test_checkpoint_schema_rejects_arbitrary_code(self):
        with self.assertRaises(ValidationError):
            Action(action="click", selector="button", assert_after=[{"assertion": "evaluate", "selector": "body", "value": "danger()"}])
        with self.assertRaises(ValidationError):
            Assertion(assertion="accessible_name", selector="input", value="")

    def test_not_checked_assertion_requires_existing_control(self):
        spec = self.task.model_dump(mode="json")
        spec["tests"] = [{"test_id": "absent-control", "requirement_id": "complete", "expected_results": [
            {"assertion": "not_checked", "selector": '[data-testid="missing-control"]'}]}]
        task = TaskSpec.model_validate(spec)
        path = Path(self.temp.name) / "index.html"
        path.write_text(self.source, encoding="utf-8")
        report = verify_application(path, task, Path(self.temp.name) / "verification")
        self.assertEqual(report["tests"][0]["status"], "failed")
        self.assertIn("Actual value: 0", report["tests"][0]["error"])

    def test_checkpoint_only_selectors_are_in_mandatory_contract(self):
        spec = self.task.model_dump(mode="json")
        spec["tests"] = [{"test_id": "unrelated", "requirement_id": "add", "actions": [
            {"action": "click", "selector": "button", "assert_after": [
                {"assertion": "visible", "selector": '[data-testid="new-record"]'},
                {"assertion": "accessible_name", "selector": '[data-testid="record-control"]'}]}],
            "expected_results": [{"assertion": "contains_text", "selector": "body", "value": "Saved"}]}]
        task = TaskSpec.model_validate(spec)
        before = task.model_dump(mode="json")
        contract = derive_implementation_contract(task)
        self.assertEqual(contract["exact_data_testids"], ["new-record", "record-control"])
        prompt = render_implementation_contract(contract)
        for instruction in ("PRIORITY", "Placeholders are hints, NOT accessible labels", "dynamically created controls", "Do not reuse", "assert_after"):
            self.assertIn(instruction, prompt)
        self.assertEqual(task.model_dump(mode="json"), before)


if __name__ == "__main__":
    unittest.main()
