"""A3 frozen outcomes and DAGs; inference is mocked, existing graphs are reused."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_web.client import ModelConfig, StructuredOutputRequestError
from agentic_web.decomposition import decompose, load_decomposition, response_schema
from agentic_web.intake import build_intake_graph
from agentic_web.schema import TaskSpec, load_task
from agentic_web.workflow import build_graph, contract_hash
from test_intake import IntegratedIntakeClient, PROMPT, analysis_fixture
from test_documents import EvidenceClient
from test_framework import stub_verifier
from mock_client import ROOT


def tasks_for(task):
    tasks, previous = [], None
    for requirement in task.requirements:
        tasks.append({"task_id": requirement.id + "-outcome", "title": requirement.description,
            "description": "Accomplish the approved observable outcome without selecting a candidate layout",
            "requirement_ids": [requirement.id], "completion_criteria": requirement.acceptance_criteria,
            "dependencies": [previous] if previous and requirement.required else [],
            "priority": "high" if requirement.required else "low", "required": requirement.required})
        if requirement.required:
            previous = tasks[-1]["task_id"]
    return {"tasks": tasks}


class DecompositionClient(IntegratedIntakeClient):
    def __init__(self, *, override=None, raw=None, finish_reason="stop", error=None):
        super().__init__()
        self.override, self.raw, self.finish_reason, self.decomposition_error = override, raw, finish_reason, error
        self.decomposition_calls = []

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        payload = json.loads(user)
        if payload.get("role") != "task_decomposition":
            return super().complete(system, user, config, planner=planner, response_schema=response_schema)
        self.roles.append("task_decomposition")
        self.decomposition_calls.append({"system": system, "payload": payload, "config": config,
                                         "planner": planner, "schema": response_schema})
        if self.decomposition_error:
            raise self.decomposition_error
        response = self.override if self.override is not None else tasks_for(TaskSpec.model_validate(payload["task_spec"]))
        return self.raw if self.raw is not None else json.dumps(response), {
            "finish_reason": self.finish_reason, "elapsed_seconds": 0.1, "model": "mock-planner",
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}}


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.analysis = analysis_fixture()
        self.task = TaskSpec.model_validate(self.analysis["task_spec"])
        self.analysis["task_hash"] = contract_hash(self.task)

    def invoke(self, *, task=None, client=None, analysis=None, name="tasks.json", **extra):
        return decompose({"task_spec": (task or self.task).model_dump(mode="json"),
            "output_file": str(self.root / name), "model_config": {"model": "implementer-model", "planner_model": "planner-model", "seed": 1025},
            **({"analysis": analysis} if analysis is not None else {}), **extra}, client or DecompositionClient())

    def test_simple_specification_creates_frozen_traceable_tasks(self):
        client = DecompositionClient()
        original = self.task.model_dump(mode="json")
        result = self.invoke(client=client, analysis=self.analysis)
        self.assertEqual(result["status"], "decomposed", result)
        record = result["decomposition"]
        self.assertEqual(record["specification_hash"], contract_hash(self.task))
        self.assertEqual(record["requirement_registry"]["accent"]["origin"], "recommendation")
        self.assertEqual(record["requirement_registry"]["responsive"]["origin"], "inferred")
        self.assertFalse(next(item for item in record["tasks"] if item["requirement_ids"] == ["accent"])["required"])
        self.assertEqual(self.task.model_dump(mode="json"), original)
        self.assertEqual(load_decomposition(result["decomposition_file"], self.task), record)
        self.assertTrue((Path(result["decomposition_run_dir"]) / "ledger/task_decomposition.json").exists())
        self.assertEqual(client.decomposition_calls[0]["config"].planner_model, "planner-model")
        self.assertTrue(client.decomposition_calls[0]["planner"])

    def test_three_existing_applications_have_dynamic_outcomes_and_ordering(self):
        for filename in ("todo.json", "eventhub.json", "catalog.json"):
            with self.subTest(filename=filename):
                task = load_task(ROOT / "tasks" / filename)
                result = self.invoke(task=task, name=filename)
                self.assertEqual(result["status"], "decomposed", result)
                record = result["decomposition"]
                order = record["dependency_order"]
                for item in record["tasks"]:
                    for dependency in item["dependencies"]:
                        self.assertLess(order.index(dependency), order.index(item["task_id"]))
                self.assertEqual({ref for item in record["tasks"] for ref in item["requirement_ids"]}, {r.id for r in task.requirements})
                self.assertFalse(record["provenance_available"])

    def test_coarse_tasks_can_cover_multiple_mandatory_requirements(self):
        response = tasks_for(self.task)
        response["tasks"] = [response["tasks"][0], response["tasks"][1]]
        response["tasks"][0]["requirement_ids"].append("responsive")
        response["tasks"][0]["completion_criteria"].extend(self.task.nonfunctional_requirements[0].acceptance_criteria)
        result = self.invoke(client=DecompositionClient(override=response))
        self.assertEqual(result["status"], "decomposed", result)
        self.assertEqual(len(result["decomposition"]["tasks"]), 2)

    def test_invalid_ids_dependencies_coverage_and_optionality_fail_closed(self):
        for index, mode in enumerate(("unknown-requirement", "duplicate-requirement", "duplicate-task", "missing-dependency",
                                     "duplicate-dependency", "cycle", "self-cycle", "missing-coverage", "optional-promoted", "mandatory-demoted", "mixed", "optional-dependency")):
            with self.subTest(mode=mode):
                response = tasks_for(self.task)
                first, optional, last = response["tasks"]
                if mode == "unknown-requirement": first["requirement_ids"] = ["invented"]
                elif mode == "duplicate-requirement": first["requirement_ids"] *= 2
                elif mode == "duplicate-task": optional["task_id"] = first["task_id"]
                elif mode == "missing-dependency": first["dependencies"] = ["not_a_task"]
                elif mode == "duplicate-dependency": last["dependencies"] *= 2
                elif mode == "cycle": first["dependencies"] = [last["task_id"]]
                elif mode == "self-cycle": first["dependencies"] = [first["task_id"]]
                elif mode == "missing-coverage": response["tasks"].pop()
                elif mode == "optional-promoted": optional["required"] = True
                elif mode == "mandatory-demoted": first["required"] = False
                elif mode == "mixed": first["requirement_ids"].append("accent")
                else: first["dependencies"] = [optional["task_id"]]
                client = DecompositionClient(override=response)
                result = self.invoke(client=client, name=f"invalid{index}.json")
                self.assertEqual(result["status"], "failed", result)
                self.assertFalse((self.root / f"invalid{index}.json").exists())
                self.assertTrue((Path(result["decomposition_run_dir"]) / "decomposition_raw.txt").exists())
                self.assertEqual(len(client.decomposition_calls), 1)

    def test_blockers_propagate_through_dependencies_without_assuming_content(self):
        self.analysis["missing_information"] = [{"issue_id": "source", "description": "Approved source text is missing",
            "question": "Supply the source", "blocks_generation": True, "requirement_ids": ["notice"], "sources": []}]
        result = self.invoke(analysis=self.analysis)
        self.assertEqual(result["status"], "needs_review", result)
        record = result["decomposition"]
        self.assertFalse(record["ready_for_planning"])
        self.assertEqual({item["task_id"]: item["status"] for item in record["tasks"]},
            {"notice-outcome": "blocked", "accent-outcome": "ready", "responsive-outcome": "blocked"})
        self.assertEqual(record["requirement_registry"]["notice"]["blocked_by"][0]["issue_id"], "source")
        self.assertEqual(load_decomposition(result["decomposition_file"], self.task), record)

    def test_unsupported_runtime_preserves_requirements_and_status(self):
        task = self.task.model_copy(deep=True)
        task.runtime.kind = "full_stack"
        result = self.invoke(task=task)
        self.assertEqual(result["status"], "needs_review", result)
        self.assertEqual({item["status"] for item in result["decomposition"]["tasks"]}, {"unsupported"})
        self.assertEqual(task.runtime.kind, "full_stack")

    def test_wrong_analysis_hash_or_specification_is_rejected(self):
        for index, changed in enumerate(("hash", "task")):
            with self.subTest(changed=changed):
                analysis = deepcopy(self.analysis)
                if changed == "hash": analysis["task_hash"] = "wrong"
                else: analysis["task_spec"]["description"] = "Changed authority"
                client = DecompositionClient()
                result = self.invoke(analysis=analysis, client=client, name=f"mismatch{index}.json")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(client.decomposition_calls)

    def test_structure_hash_is_reproducible_for_the_same_mock_response(self):
        first = self.invoke(analysis=self.analysis)
        second = self.invoke(analysis=self.analysis, name="second.json")
        a, b = first["decomposition"], second["decomposition"]
        self.assertEqual(a["structure_hash"], b["structure_hash"])
        self.assertEqual(a["tasks"], b["tasks"])
        self.assertIsNone(a["inference_metrics"]["max_tokens"])
        self.assertEqual(a["inference_metrics"]["finish_reason"], "stop")
        self.assertEqual(a["inference_metrics"]["usage"]["total_tokens"], 300)

    def test_frozen_artifact_tampering_and_stale_spec_are_rejected(self):
        result = self.invoke(analysis=self.analysis)
        for index, mode in enumerate(("hash", "status", "readiness", "origin", "spec")):
            with self.subTest(mode=mode):
                record = deepcopy(result["decomposition"])
                task = self.task.model_copy(deep=True)
                if mode == "hash": record["structure_hash"] = "wrong"
                elif mode == "status": record["tasks"][0]["status"] = "blocked"
                elif mode == "readiness": record["ready_for_planning"] = False
                elif mode == "origin": record["requirement_registry"]["notice"]["origin"] = "recommendation"
                else: task.description = "Different canonical source"
                file = self.root / f"tampered{index}.json"
                file.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError): load_decomposition(file, task)

    def test_original_and_existing_output_are_preserved(self):
        path = self.root / "canonical.json"
        original = json.dumps(self.task.model_dump(mode="json")).encode()
        path.write_bytes(original)
        first = decompose({"task_path": str(path), "output_file": str(self.root / "tasks.json")}, DecompositionClient())
        self.assertEqual(first["status"], "decomposed")
        client = DecompositionClient()
        second = self.invoke(client=client)
        self.assertEqual(second["status"], "failed")
        self.assertFalse(client.decomposition_calls)
        self.assertEqual(path.read_bytes(), original)

    def test_malformed_wrapped_truncated_and_unsupported_outputs_are_not_repaired(self):
        clients = [DecompositionClient(raw="{malformed"), DecompositionClient(raw=json.dumps({"TaskGraph": tasks_for(self.task)})),
            DecompositionClient(finish_reason="length"), DecompositionClient(error=StructuredOutputRequestError("unsupported schema", {"kind": "unsupported"}))]
        for index, client in enumerate(clients):
            with self.subTest(index=index):
                result = self.invoke(client=client, name=f"failure{index}.json")
                self.assertEqual(result["status"], "failed", result)
                self.assertEqual(len(client.decomposition_calls), 1)
                self.assertFalse((self.root / f"failure{index}.json").exists())
                directory = Path(result["decomposition_run_dir"])
                self.assertTrue((directory / "decomposition_error.json").exists())
                if index == 2:
                    self.assertTrue(json.loads((directory / "decomposition_metrics.json").read_text())["incomplete_generation"])
                if index == 3:
                    self.assertTrue((directory / "decomposition_structured_output_error.json").exists())

    def test_cli_decomposes_without_website_or_intake(self):
        from agentic_web.__main__ import main
        client = DecompositionClient()
        task_file = self.root / "canonical.json"
        task_file.write_text(self.task.model_dump_json(), encoding="utf-8")
        with patch("agentic_web.decomposition.LMStudioClient", return_value=client), patch("sys.argv", ["agentic_web", "decompose",
            "--task", str(task_file), "--output", str(self.root / "tasks.json"), "--model", "planner-model", "--seed", "1025", "--max-tokens", "512"]), redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "decomposed")
        self.assertEqual(client.roles, ["task_decomposition"])
        self.assertEqual(client.decomposition_calls[0]["config"].planner_max_tokens, 512)

    def test_existing_planner_consumes_reusable_decomposition_in_both_graphs(self):
        frozen = self.invoke()
        for workflow in ("baseline", "collaborative"):
            with self.subTest(workflow=workflow):
                client = DecompositionClient()
                result = build_graph(client, stub_verifier, workflow=workflow).invoke({"task_spec": self.task.model_dump(mode="json"),
                    "decomposition_path": frozen["decomposition_file"], "output_root": str(self.root)})
                self.assertEqual(result["status"], "passed", result)
                self.assertNotIn("task_decomposition", client.roles)
                self.assertNotIn("requirements_analyst", client.roles)
                planner = next(call for call in client.calls if call["planner"])
                self.assertEqual(planner["payload"]["task_decomposition"], frozen["decomposition"])
                self.assertTrue((Path(result["run_dir"]) / "task_decomposition.json").exists())

    def test_natural_language_and_documents_decompose_before_candidate_planning(self):
        client = DecompositionClient()
        client.intake_client = EvidenceClient()
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({"prompt": PROMPT,
            "documents": ["tests/fixtures/documents/resume.docx"], "document_root": str(ROOT), "decompose": True,
            "output_root": str(self.root)})
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(client.roles[:3], ["requirements_analyst", "task_decomposition", "planner"])
        self.assertEqual(client.roles.count("requirements_analyst"), 1)
        self.assertIn("Alex Example", client.decomposition_calls[0]["payload"]["task_spec"]["supporting_context"])

    def test_standalone_intake_can_stop_after_decomposition(self):
        client = DecompositionClient()
        result = build_intake_graph(client).invoke({"prompt": PROMPT, "output_file": str(self.root / "canonical.json"), "decompose": True})
        self.assertEqual(result["status"], "decomposed", result)
        self.assertEqual(client.roles, ["requirements_analyst", "task_decomposition"])
        self.assertTrue((self.root / "canonical.tasks.json").exists())

    def test_graph_preview_and_blockers_never_start_candidate_agents(self):
        for index, blocked in enumerate((False, True)):
            with self.subTest(blocked=blocked):
                client = DecompositionClient()
                if blocked:
                    client.intake_client.response["missing_information"] = [{"issue_id": "source", "description": "Missing source",
                        "question": "Provide source", "blocks_generation": True, "requirement_ids": ["notice"], "sources": []}]
                result = build_graph(client, stub_verifier, workflow="collaborative").invoke({"prompt": PROMPT,
                    "decompose": True, "analyze_only": not blocked, "output_root": str(self.root / str(index))})
                self.assertEqual(result["status"], "needs_review" if blocked else "decomposed", result)
                self.assertEqual(client.roles, ["requirements_analyst", "task_decomposition"])
                self.assertNotIn("plan", result)

    def test_adjacent_a1_analysis_is_reused_without_reanalysis(self):
        client = DecompositionClient()
        intake = build_intake_graph(client).invoke({"prompt": PROMPT, "output_file": str(self.root / "canonical.json")})
        other = DecompositionClient()
        result = decompose({"task_path": intake["output_file"], "output_file": str(self.root / "tasks.json")}, other)
        self.assertEqual(result["status"], "decomposed", result)
        self.assertTrue(result["decomposition"]["provenance_available"])
        self.assertEqual(other.roles, ["task_decomposition"])

    def test_model_schema_and_instructions_preserve_design_freedom(self):
        from jsonschema_rs import Draft202012Validator
        schema = response_schema(self.task).model_json_schema()
        response = tasks_for(self.task)
        validator = Draft202012Validator(schema)
        self.assertTrue(validator.is_valid(response))
        response["tasks"][0]["requirement_ids"] = ["undeclared"]
        self.assertFalse(validator.is_valid(response))
        client = DecompositionClient()
        self.invoke(client=client)
        call = client.decomposition_calls[0]
        self.assertIn("WHAT", call["system"])
        self.assertIn("candidate-specific HOW", call["system"])
        self.assertEqual(call["payload"]["design_freedom"], self.task.design_freedom)

    def test_optional_only_unsupported_and_global_blockers_remain_not_ready(self):
        analysis = deepcopy(self.analysis)
        for requirement in analysis["task_spec"]["features"] + analysis["task_spec"]["nonfunctional_requirements"]:
            requirement["required"] = False
        for test in analysis["task_spec"]["tests"]:
            test["required"] = False
        task = TaskSpec.model_validate(analysis["task_spec"])
        analysis["task_hash"] = contract_hash(task)
        analysis["missing_information"] = [{"issue_id": "global", "description": "A global input is missing",
            "question": "Provide input", "blocks_generation": True, "requirement_ids": [], "sources": []}]
        first = self.invoke(task=task, analysis=analysis)
        self.assertEqual(first["status"], "needs_review", first)
        task.runtime.kind = "full_stack"
        second = self.invoke(task=task, name="unsupported.json")
        self.assertEqual(second["status"], "needs_review", second)

    def test_resource_admission_is_reused_and_external_models_are_preserved(self):
        from agentic_web.resource_scheduler import ResourceScheduler, select_placements
        from agentic_web.resource_schema import ResourceSettings
        from resource_mock_backend import SimulatedBackend
        backend = SimulatedBackend()
        settings = ResourceSettings(context_length=4096, gpu_safety_bytes=0, ram_safety_bytes=0,
            estimate_padding=0, prefetch_all_if_gpu_fit=False, resident_devices={"manual-resident": "0"})
        placement = select_placements([backend.estimate("light", settings, 1)], backend.hardware(), backend.capabilities(), settings)[0]["light"]
        backend.load(placement, "manual-resident")
        scheduler = ResourceScheduler(backend, backend.hardware)
        class AdmittedClient(DecompositionClient):
            def complete(self, system, user, config, *, planner=False, response_schema=None):
                if (config.planner_model if planner else config.model) not in backend.instances:
                    raise AssertionError("Inference bypassed resource admission")
                return super().complete(system, user, config, planner=planner, response_schema=response_schema)
        client = AdmittedClient()
        result = build_graph(client, stub_verifier, workflow="collaborative", resource_scheduler=scheduler).invoke({
            "task_spec": self.task.model_dump(mode="json"), "decompose": True, "output_root": str(self.root),
            "model_config": ModelConfig(model="tiny").model_dump(), "resource_config": settings.model_dump()})
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(client.roles[:2], ["task_decomposition", "planner"])
        self.assertEqual(set(backend.instances), {"manual-resident"})
        self.assertNotIn("manual-resident", backend.unloads)
        self.assertTrue((Path(result["decomposition_run_dir"]) / "resources.cleanup.json").exists())

    def test_frozen_blocked_decomposition_stops_before_planner(self):
        self.analysis["missing_information"] = [{"issue_id": "source", "description": "Missing source",
            "question": "Provide source", "blocks_generation": True, "requirement_ids": ["notice"], "sources": []}]
        frozen = self.invoke(analysis=self.analysis)
        client = DecompositionClient()
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({"task_spec": self.task.model_dump(mode="json"),
            "decomposition_path": frozen["decomposition_file"], "output_root": str(self.root)})
        self.assertEqual(result["status"], "needs_review", result)
        self.assertEqual(client.roles, [])

    def test_decompose_only_canonical_graph_does_not_run_intake_or_candidates(self):
        client = DecompositionClient()
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({"task_spec": self.task.model_dump(mode="json"),
            "decompose_only": True, "output_root": str(self.root)})
        self.assertEqual(result["status"], "decomposed", result)
        self.assertEqual(client.roles, ["task_decomposition"])
        self.assertNotIn("plan", result)
