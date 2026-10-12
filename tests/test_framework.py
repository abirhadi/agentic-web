import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from agentic_web.client import ModelConfig, LMStudioClient
from agentic_web.runtime import extract_html, get_adapter, UnsupportedRuntime
from agentic_web.schema import Action, Assertion, TaskSpec, TestCase, load_task
from agentic_web.verification import coverage, finish, verify_application
from agentic_web.workflow import app, build_graph, contract_hash
from mock_client import MockClient, ROOT


def stub_verifier(path, task, output):
    return finish(task, {"checks": {"mock_check": True}, "tests": [
        {"test_id": t.test_id, "status": "passed", "required": t.required} for t in task.tests], "errors": [], "artifacts": {}})


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.task = load_task(ROOT / "tasks/todo.json")

    def invoke(self, client=None, verifier=stub_verifier, **inputs):
        self.client = client or MockClient()
        state = {"task_spec": self.task.model_dump(mode="json"), "output_root": self.temp.name, "execution_mode": "mocked"}
        state.update(inputs)
        return build_graph(self.client, verifier).invoke(state)

    def test_load_three_manifests(self):
        for name in ("eventhub", "todo", "catalog"):
            self.assertEqual(load_task(ROOT / f"tasks/{name}.json").task_id, name)

    def test_invalid_input(self):
        spec = self.task.model_dump()
        spec["unknown"] = 1
        with self.assertRaises(ValidationError):
            TaskSpec.model_validate(spec)

    def test_duplicate_requirement(self):
        spec = self.task.model_dump()
        spec["features"].append(spec["features"][0])
        with self.assertRaises(ValidationError):
            TaskSpec.model_validate(spec)

    def test_invalid_test_reference(self):
        spec = self.task.model_dump()
        spec["tests"][0]["requirement_id"] = "invented"
        with self.assertRaises(ValidationError):
            TaskSpec.model_validate(spec)

    def test_required_test_cannot_be_downgraded(self):
        spec = self.task.model_dump()
        spec["tests"][0]["required"] = False
        with self.assertRaises(ValidationError):
            TaskSpec.model_validate(spec)

    def test_paths_confined(self):
        for entrypoint in ("../index.html", "C:/secret.html", "/index.html", "..\\secret.html"):
            spec = self.task.model_dump()
            spec["runtime"]["entrypoint"] = entrypoint
            with self.assertRaises(ValidationError):
                TaskSpec.model_validate(spec)

    def test_unsafe_test_code_rejected(self):
        for operation in ("evaluate", "python", "shell", "goto"):
            with self.assertRaises(ValidationError):
                Action(action=operation, selector="body", value="danger")
        with self.assertRaises(ValidationError):
            Assertion(assertion="count", selector="body", value="1")
        with self.assertRaises(ValidationError):
            TestCase(test_id="empty", requirement_id="add", expected_results=[])

    def test_same_graph_accepts_unrelated_specs(self):
        client = MockClient()
        graph = build_graph(client, stub_verifier)
        directories = []
        for name in ("eventhub", "todo", "catalog"):
            task = load_task(ROOT / f"tasks/{name}.json")
            original = task.model_dump(mode="json")
            result = graph.invoke({"task_spec": original, "output_root": self.temp.name})
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["task_spec"], original)
            self.assertEqual(result["requirements"], task.original_requirements)
            self.assertEqual(result["requirement_hash"], contract_hash(task))
            directories.append(result["run_dir"])
        self.assertEqual(len(set(directories)), 3)

    def test_agent_inputs_and_saved_artifacts(self):
        result = self.invoke(model_config={"model": "implementation-model", "planner_model": "planning-model", "seed": 5})
        planner, implementer = self.client.calls
        self.assertEqual(planner["payload"]["task_spec"], self.task.model_dump(mode="json"))
        self.assertEqual(implementer["payload"]["original_requirements"], self.task.original_requirements)
        self.assertEqual(implementer["payload"]["plan"], result["plan"])
        self.assertEqual(implementer["payload"]["runtime_constraints"], self.task.runtime.model_dump(mode="json"))
        for artifact in ("input.json", "requirements.txt", "task.json", "model_config.json", "planner_raw.txt", "plan.json",
                         "implementer_raw.txt", "planner_metrics.json", "implementer_metrics.json", "verification_plan.json",
                         "verification_results.json", "metadata.json", "application/index.html"):
            self.assertTrue((Path(result["run_dir"]) / artifact).is_file(), artifact)
        self.assertEqual(json.loads((Path(result["run_dir"]) / "verification_plan.json").read_text()), [t.model_dump() for t in self.task.tests])

    def test_failed_plan_preserved(self):
        class BadPlanner(MockClient):
            def complete(self, *args, **kwargs):
                return '{"invalid": true}', {"finish_reason": "stop"}
        result = self.invoke(BadPlanner())
        self.assertEqual(result["status"], "failed")
        self.assertTrue((Path(result["run_dir"]) / "planner_raw.txt").exists())
        self.assertNotIn("html_path", result)

    def test_plan_cannot_drop_requirements(self):
        class ChangedPlan(MockClient):
            def complete(self, *args, **kwargs):
                raw, metrics = super().complete(*args, **kwargs)
                plan = json.loads(raw)
                plan["requirement_ids"] = []
                return json.dumps(plan), metrics
        result = self.invoke(ChangedPlan())
        self.assertEqual(result["status"], "failed")

    def test_truncated_generation_preserved(self):
        class Truncated(MockClient):
            def complete(self, *args, planner=False, **kwargs):
                if planner:
                    return super().complete(*args, planner=planner, **kwargs)
                return "<!DOCTYPE html><html>", {"finish_reason": "length"}
        result = self.invoke(Truncated())
        self.assertEqual(result["status"], "failed")
        self.assertTrue((Path(result["run_dir"]) / "implementer_raw.txt").exists())

    def test_model_connection_failure_retained(self):
        class Broken(MockClient):
            def complete(self, *args, **kwargs):
                raise ConnectionError("Offline")
        result = self.invoke(Broken())
        self.assertEqual(result["status"], "failed")
        self.assertTrue((Path(result["run_dir"]) / "failure.json").exists())

    def test_unsupported_runtime_explicit(self):
        spec = self.task.model_dump()
        spec["runtime"]["kind"] = "full_stack"
        result = self.invoke(task_spec=spec)
        self.assertEqual(result["status"], "failed")
        self.assertIn("UnsupportedRuntime", result["error"])
        self.assertEqual(self.client.calls, [])

    def test_missing_coverage_is_unverified(self):
        self.task.tests = []
        result = self.invoke()
        self.assertEqual(result["status"], "unverified")
        self.assertTrue(all(c["status"] == "unverified" for c in result["verification"]["coverage"].values()))

    def test_optional_requirement_stays_unverified(self):
        result = self.invoke()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["verification"]["coverage"]["persistence"]["status"], "unverified")

    def test_verifier_failure_never_success(self):
        def failed(path, task, output):
            return finish(task, {"checks": {"loads": False}, "tests": [], "errors": ["Failed"], "artifacts": {}})
        self.assertEqual(self.invoke(verifier=failed)["status"], "failed")

    def test_conflicting_original_requirements_rejected(self):
        result = self.invoke(requirements="Conflicting original requirements")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.client.calls, [])

    def test_legacy_imports_no_inference(self):
        with patch.object(LMStudioClient, "complete", side_effect=AssertionError("Inference on import")):
            for module in ("test_langgraph", "lmstudio_pipeline"):
                imported = importlib.import_module(module)
                self.assertIs(imported.app, app)
        from web_task_spec import load_task as compatibility_load
        self.assertEqual(compatibility_load(ROOT / "tasks/todo.json"), self.task)

    def test_legacy_requirements_execute_without_invented_features(self):
        class FreeTextClient(MockClient):
            def complete(self, *args, planner=False, **kwargs):
                if planner:
                    return super().complete(*args, planner=planner, **kwargs)
                return (ROOT / "tests/fixtures/todo.html").read_text(), {"finish_reason": "stop"}
        requirements = "Build a personal website with the requested functionality"
        result = build_graph(FreeTextClient(), stub_verifier).invoke({"requirements": requirements, "output_root": self.temp.name})
        self.assertEqual(result["requirements"], requirements)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual([feature["id"] for feature in result["task_spec"]["features"]], ["user_request"])

    def test_studio_loads_all_configured_graphs(self):
        import importlib.util
        config = json.loads((ROOT / "langgraph.json").read_text())
        self.assertEqual(config["graphs"]["agentic_web_collaborative"], "agentic_web.workflow:collaborative_app")
        for name, path in config["graphs"].items():
            module_path, attribute = path.split(":")
            if module_path.endswith(".py"):
                spec = importlib.util.spec_from_file_location("studio_" + name, ROOT / module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(module_path)
            graph = getattr(module, attribute)
            expected = {"__start__", "task_decomposition", "task_loader", "planner", "implementer", "verifier", "__end__"}
            if name == "agentic_web_collaborative":
                expected |= {"requirements_prepare", "requirements_analyst", "requirements_review", "requirements_revision",
                             "requirements_finalize", "handoff_decomposer", "ux_agent", "architecture_agent", "coordinator", "qa_critic", "acceptance_gate"}
                self.assertIn("prompt", graph.get_input_jsonschema()["properties"])
                self.assertIn("analyze_only", graph.get_input_jsonschema()["properties"])
                self.assertIn("supporting_context", graph.get_input_jsonschema()["properties"])
                self.assertIn("analysis", graph.get_output_jsonschema()["properties"])
            self.assertEqual(set(graph.get_graph().nodes), expected)
            self.assertIn("task_path", graph.get_input_jsonschema()["properties"])
            self.assertIn("task_spec", graph.get_input_jsonschema()["properties"])
            self.assertNotIn("run_dir", graph.get_input_jsonschema()["properties"])

    def test_extract_html_compatibility(self):
        self.assertEqual(extract_html("```html\n<!DOCTYPE html><html></html>\n```"), "<!DOCTYPE html><html></html>\n")
        with self.assertRaises(ValueError):
            extract_html("<html>")


class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_real_browser_for_three_mocked_applications(self):
        for name in ("eventhub", "todo", "catalog"):
            with self.subTest(task=name):
                result = build_graph(MockClient()).invoke({"task_path": str(ROOT / f"tasks/{name}.json"),
                    "output_root": self.temp.name, "execution_mode": "mocked-model-real-browser"})
                self.assertEqual(result["status"], "passed", result.get("verification", result))
                self.assertTrue(all(t["status"] == "passed" for t in result["verification"]["tests"]))
                for label in ("desktop", "mobile"):
                    self.assertTrue(Path(result["verification"]["artifacts"][label + "_screenshot"]).exists())

    def test_failed_acceptance_has_details(self):
        task = load_task(ROOT / "tasks/todo.json")
        task.verification.timeout_ms = 200
        task.tests[0].expected_results[0].value = 999
        report = verify_application(ROOT / "tests/fixtures/todo.html", task, self.temp.name)
        self.assertEqual(report["status"], "failed")
        result = report["tests"][0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("expected_result", result["phase"])
        self.assertTrue(Path(result["screenshot"]).exists())

    def test_browser_exceptions_fail(self):
        task = load_task(ROOT / "tasks/todo.json")
        source = (ROOT / "tests/fixtures/todo.html").read_text()
        path = Path(self.temp.name) / "bad.html"
        path.write_text(source.replace("<script>", "<script>throw new Error('deliberate failure');"), encoding="utf-8")
        task.tests = []
        report = verify_application(path, task, Path(self.temp.name) / "report")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["desktop_no_javascript_errors"])

    def test_javascript_syntax_preflight_precedes_downstream_timeout(self):
        task = load_task(ROOT / "tasks/todo.json")
        task.verification.timeout_ms = 300
        task.verification.screenshots = False
        source = (ROOT / "tests/fixtures/todo.html").read_text()
        invalid = source.replace("del.textContent='Delete';", "del.textContent='Delete';\n\ndel aria-label='Delete';")
        expected_line = invalid[:invalid.index("del aria-label")].count("\n") + 1
        path = Path(self.temp.name) / "syntax-error.html"
        path.write_text(invalid, encoding="utf-8")
        report = verify_application(path, task, Path(self.temp.name) / "syntax-report")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["javascript_syntax"])
        diagnostic = report["javascript_diagnostics"][0]
        self.assertEqual(diagnostic["kind"], "syntax_error")
        self.assertIn("Unexpected identifier 'aria'", diagnostic["message"])
        self.assertEqual(diagnostic["html_line"], expected_line)
        failed = [test for test in report["tests"] if test["status"] == "failed"]
        self.assertTrue(any("TimeoutError: Locator.click" in test["error"] for test in failed))
        self.assertTrue(any("Unexpected identifier 'aria'" in " ".join(test.get("runtime_errors", [])) for test in failed))

    def test_network_and_files_blocked(self):
        task = load_task(ROOT / "tasks/todo.json")
        task.verification.total_timeout_seconds = 10
        task.tests = [TestCase(test_id="isolation", requirement_id="add", expected_results=[
            Assertion(assertion="text", selector="#network", value="blocked"),
            Assertion(assertion="text", selector="#file", value="blocked"),
            Assertion(assertion="text", selector="#socket", value="blocked"),
            Assertion(assertion="text", selector="#rtc", value="blocked")])]
        source = (ROOT / "tests/fixtures/todo.html").read_text()
        probe = '''<p id="network">pending</p><p id="file">pending</p><p id="socket">pending</p><p id="rtc">pending</p><script>
        fetch('https://example.com').then(()=>document.querySelector('#network').textContent='escaped').catch(()=>document.querySelector('#network').textContent='blocked');
        fetch('file:///C:/Windows/win.ini').then(()=>document.querySelector('#file').textContent='escaped').catch(()=>document.querySelector('#file').textContent='blocked');
        const socket=new WebSocket('wss://example.com');socket.onerror=()=>document.querySelector('#socket').textContent='blocked';socket.onopen=()=>document.querySelector('#socket').textContent='escaped';
        document.querySelector('#rtc').textContent=typeof RTCPeerConnection==='undefined'?'blocked':'escaped';
        </script>'''
        path = Path(self.temp.name) / "isolation.html"
        path.write_text(source.replace('</body>', probe + '</body>'), encoding="utf-8")
        report = verify_application(path, task, Path(self.temp.name) / "report")
        self.assertEqual(report["tests"][0]["status"], "passed", report)

    def test_legacy_smoke_verifier_preserved(self):
        from browser_verifier import verify_site
        path = Path(self.temp.name) / "legacy.html"
        path.write_text((ROOT / "tests/fixtures/todo.html").read_text(), encoding="utf-8")
        self.assertEqual(verify_site(path, timeout_ms=1000)["status"], "passed")

    def test_runaway_script_has_hard_timeout(self):
        task = load_task(ROOT / "tasks/todo.json")
        task.verification.total_timeout_seconds = 5
        path = Path(self.temp.name) / "runaway.html"
        path.write_text('<!DOCTYPE html><html><head><title>Task Tracker</title></head><body><script>while(true){}</script></body></html>', encoding="utf-8")
        report = verify_application(path, task, Path(self.temp.name) / "report")
        self.assertEqual(report["status"], "failed")
        self.assertIn("time budget", report["errors"][0])
        self.assertTrue(all(c["status"] == "unverified" for c in report["coverage"].values()))


if __name__ == "__main__":
    unittest.main()
