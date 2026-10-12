import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from agentic_web.client import ModelConfig
from agentic_web.collaboration import evidence_registry, validate_diagnostic_hierarchy, validate_evidence
from agentic_web.collaboration_schema import EvidenceReference, QAReview, grounded_qa_review_schema
from agentic_web.provenance import record_stage
from agentic_web.schema import TaskSpec
from agentic_web.workflow import build_graph, collaborative_app
from collaboration_mock_client import CollaborationMockClient
from mock_client import ROOT
from test_framework import stub_verifier


def evidence_schema_variants(schema):
    items = schema["$defs"]["QAFinding"]["properties"]["evidence"]["items"]
    return [schema["$defs"][choice["$ref"].rsplit("/", 1)[-1]]["properties"]
            for choice in items.get("anyOf", [items])]


def literal_values(schema):
    return schema.get("enum", [schema.get("const")])


class CollaborationTests(unittest.TestCase):
    def test_specialist_generation_schemas_bind_exact_manifest_ids(self):
        from jsonschema_rs import Draft202012Validator
        from agentic_web.collaboration_schema import UXReview, ArchitectureReview, grounded_specialist_schema
        for name in ('eventhub', 'todo', 'catalog'):
            task = TaskSpec.model_validate_json((ROOT / f'tasks/{name}.json').read_text())
            ids = [r.id for r in task.requirements]
            for base in (UXReview, ArchitectureReview):
                schema = grounded_specialist_schema(task, base)
                document = schema.model_json_schema()
                self.assertEqual(schema.__name__, base.__name__)
                allowed = document['$defs']['Recommendation']['properties']['requirement_ids']['items']
                self.assertEqual(set(literal_values(allowed)), set(ids))
                validator = Draft202012Validator(document)
                value = {key: [] for key in base.model_fields}
                if base is ArchitectureReview:
                    value['runtime_feasibility'] = 'Supported'
                # Empty recommendations remain valid advisory output.
                schema.model_validate(value)
                self.assertTrue(validator.is_valid(value))
                recommendation = dict(requirement_ids=ids, recommendation='Preserve outcomes', rationale='Manifest')
                value['recommendations'] = [recommendation]
                schema.model_validate(value)
                self.assertTrue(validator.is_valid(value))
                for unknown in ('interaction_flows', 'architecture_variance', 'layout_variance',
                                'feat_nav-01', 'nf_accessibility-01', 'feat_registration_form-01'):
                    with self.subTest(task=name, role=base.__name__, invalid=unknown):
                        recommendation['requirement_ids'] = [unknown]
                        self.assertFalse(validator.is_valid(value))
                        with self.assertRaises(ValidationError):
                            schema.model_validate(value)

    def test_specialist_requests_use_grounded_schema_and_reject_invalid_backend_output(self):
        class InvalidSpecialist(CollaborationMockClient):
            def complete(self, system, user, config, **kwargs):
                raw, metrics = super().complete(system, user, config, **kwargs)
                if json.loads(user).get('role') == 'ux':
                    value = json.loads(raw)
                    value['recommendations'][0]['requirement_ids'] = ['layout_variance']
                    raw = json.dumps(value)
                return raw, metrics
        result = self.invoke(InvalidSpecialist())
        self.assertIn('ux_error', result)
        self.assertNotIn('implementer', self.client.roles)
        directory = Path(result['run_dir'])
        self.assertIn('layout_variance', (directory / 'ux_raw.txt').read_text())
        self.assertTrue((directory / 'ux_validation_error.json').exists())
        for role in ('ux', 'architect'):
            request = json.loads((directory / f'{role}_request.json').read_text())
            schema = request['response_format']['json_schema']['schema']
            self.assertEqual(json.loads(request['user'])['output_schema'], schema)
            self.assertIn('enum', schema['$defs']['Recommendation']['properties']['requirement_ids']['items'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = ModelConfig(model="build-model", planner_model="plan-model", ux_model="ux-model",
            architect_model="architecture-model", coordinator_model="coordinate-model", qa_model="qa-model", seed=20)

    def invoke(self, client=None, verifier=stub_verifier, name="todo"):
        self.client = client or CollaborationMockClient()
        return build_graph(self.client, verifier, workflow="collaborative").invoke({"task_path": str(ROOT / f"tasks/{name}.json"),
            "model_config": self.config.model_dump(), "output_root": self.temp.name, "execution_mode": "mocked"})

    def test_all_manifests_share_collaborative_graph(self):
        graph = build_graph(CollaborationMockClient(), stub_verifier, workflow="collaborative")
        for name in ("eventhub", "todo", "catalog"):
            result = graph.invoke({"task_path": str(ROOT / f"tasks/{name}.json"), "output_root": self.temp.name,
                                   "model_config": self.config.model_dump()})
            self.assertEqual(result["status"], "passed", result)
            self.assertTrue(result["acceptance"]["accepted"])

    def test_independence_join_coordination_and_implementation_inputs(self):
        result = self.invoke()
        calls = {call["payload"].get("role", "planner" if call["planner"] else "implementer"): call for call in self.client.calls}
        ux, architect = calls["ux"]["payload"], calls["architect"]["payload"]
        for payload in (ux, architect):
            for field in ("original_requirements", "task_spec", "plan", "runtime_constraints"):
                self.assertEqual(payload[field], ux[field])
            self.assertNotIn("ux_review", payload)
            self.assertNotIn("architecture_review", payload)
            self.assertNotIn("coordinated_specification", payload)
        self.assertEqual(calls["coordinator"]["payload"]["ux_review"], result["ux_review"])
        self.assertEqual(calls["coordinator"]["payload"]["architecture_review"], result["architecture_review"])
        self.assertEqual(self.client.roles.count("coordinator"), 1)
        self.assertLess(self.client.roles.index("ux"), self.client.roles.index("coordinator"))
        self.assertLess(self.client.roles.index("architect"), self.client.roles.index("coordinator"))
        self.assertEqual(len(result["coordinated_specification"]["conflict_resolutions"]), 1)
        implementation = calls["implementer"]["payload"]
        self.assertEqual(implementation["coordinated_specification"], result["coordinated_specification"])
        self.assertEqual(implementation["task_spec"], result["task_spec"])
        self.assertIn("selector_contract", implementation["mandatory_implementation_contract"])
        self.assertTrue(calls["implementer"]["system"].startswith("MANDATORY IMPLEMENTATION CONTRACT"))

    def test_role_models_seeds_and_artifacts(self):
        result = self.invoke()
        directory = Path(result["run_dir"])
        for role, seed in (("planner", 20), ("implementer", 21), ("ux", 22), ("architect", 23), ("coordinator", 24), ("qa", 25)):
            metrics = json.loads((directory / f"{role}_metrics.json").read_text())
            self.assertEqual(metrics["requested_model"], self.config.model_for(role))
            self.assertEqual(metrics["seed"], seed)
            self.assertEqual(metrics["role"], role)
        for name in ("ux_review.json", "architecture_review.json", "coordination.json", "qa_review.json", "acceptance.json"):
            self.assertTrue((directory / name).is_file())
        for stage in ("task_loader", "planner", "ux", "architect", "coordinator", "implementer", "verifier", "qa", "acceptance_gate"):
            entry = json.loads((directory / "ledger" / f"{stage}.json").read_text())
            self.assertEqual(entry["requirement_hash"], result["requirement_hash"])
            for path, digest in entry["artifact_hashes"].items():
                self.assertEqual(hashlib.sha256((directory / path).read_bytes()).hexdigest(), digest)
        with self.assertRaises(FileExistsError):
            record_stage(result, "qa", "changed")

    def test_qa_receives_verifier_evidence_and_source(self):
        result = self.invoke()
        qa = next(call["payload"] for call in self.client.calls if call["payload"].get("role") == "qa")
        self.assertEqual(qa["verification"], result["verification"])
        self.assertIn("<!DOCTYPE html>", qa["source_excerpt"])
        self.assertEqual(qa["task_spec"], result["task_spec"])
        self.assertEqual(qa["coordinated_specification"], result["coordinated_specification"])

    def test_qa_failure_does_not_reject_passing_artifact(self):
        for mode in ("malformed", "transport_failure"):
            with self.subTest(mode=mode):
                result = self.invoke(CollaborationMockClient(**{mode: "qa"}))
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["verification"]["status"], "passed")
                self.assertTrue(result["acceptance"]["accepted"])
                self.assertEqual(result["qa_status"], "failed")
                self.assertTrue((Path(result["run_dir"]) / "qa_error.json").exists())

    def test_qa_runs_after_failed_verification_and_cannot_override(self):
        def failed(path, task, output):
            report = stub_verifier(path, task, output)
            target = report["tests"][0]
            target.update(status="failed", requirement_id=task.tests[0].requirement_id, error="Missing selector")
            report["coverage"][task.tests[0].requirement_id]["status"] = "failed"
            report["status"] = "failed"
            return report
        result = self.invoke(verifier=failed)
        self.assertEqual(result["qa_status"], "completed")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["acceptance"]["accepted"])
        self.assertEqual(result["qa_review"]["findings"][0]["classification"], "hypothesis")

    def test_qa_rejects_fabricated_evidence_and_unknown_outcomes(self):
        finding = {"classification": "observed", "diagnostic_role": "independent", "category": "functional", "description": "Claim", "requirement_ids": ["add"],
            "test_ids": ["add-acceptance"], "evidence": [{"key": "tests.add-acceptance.status", "quote": "failed"}],
            "test_outcomes": {"add-acceptance": "failed"}, "corrective_action": "Review"}
        result = self.invoke(CollaborationMockClient(qa_override={"findings": [finding]}))
        self.assertEqual(result["qa_status"], "failed", result)
        self.assertIn("literal_error", result["qa_error"])
        self.assertTrue(result["acceptance"]["accepted"])
        finding["evidence"][0]["quote"] = "passed"
        result = self.invoke(CollaborationMockClient(qa_override={"findings": [finding]}))
        self.assertIn("change deterministic outcome", result["qa_error"])
        self.assertTrue(result["acceptance"]["accepted"])

    def test_qa_evidence_namespace_and_quotes_are_exact(self):
        registry = {"tests.add-acceptance.error": "TimeoutError: Locator.click: Timeout 5000ms exceeded."}
        validate_evidence(EvidenceReference(key="tests.add-acceptance.error", quote="Locator.click"), registry)
        with self.assertRaisesRegex(ValueError, "unknown evidence key"):
            validate_evidence(EvidenceReference(key="Locator.click", quote="Locator.click"), registry)
        with self.assertRaisesRegex(ValueError, "exact substring"):
            validate_evidence(EvidenceReference(key="tests.add-acceptance.error", quote="click timed out"), registry)
        with self.assertRaisesRegex(ValueError, "ellipsis"):
            validate_evidence(EvidenceReference(key="tests.add-acceptance.error", quote="TimeoutError: ..."), registry)

    def test_qa_schema_constrains_evidence_to_the_current_registry(self):
        registry = {"javascript_diagnostics.0.message": "Unexpected identifier 'aria'",
                    "tests.add-acceptance.error": "TimeoutError: Locator.click"}
        task = TaskSpec.model_validate_json((ROOT / "tasks/todo.json").read_text())
        schema = grounded_qa_review_schema(registry, task).model_json_schema()
        pairs = {key: evidence["quote"]["const"] for evidence in evidence_schema_variants(schema)
                 for key in literal_values(evidence["key"])}
        self.assertEqual(pairs, registry)

    def test_qa_identifier_schemas_follow_each_manifest(self):
        for name in ("eventhub", "todo", "catalog"):
            with self.subTest(task=name):
                task = TaskSpec.model_validate_json((ROOT / f"tasks/{name}.json").read_text())
                schema = grounded_qa_review_schema({"verification.status": "passed"}, task).model_json_schema()
                properties = schema["$defs"]["QAFinding"]["properties"]
                self.assertEqual(set(literal_values(properties["requirement_ids"]["items"])),
                                 {requirement.id for requirement in task.requirements})
                self.assertEqual(set(literal_values(properties["test_ids"]["items"])),
                                 {test.test_id for test in task.tests})
                self.assertEqual(properties["requirement_ids"]["minItems"], 1)

    def test_qa_rejects_swapped_or_unknown_ids_without_retry_or_repair(self):
        finding = {"classification": "observed", "diagnostic_role": "independent", "category": "functional",
            "description": "A tested outcome", "requirement_ids": ["add"], "test_ids": ["add-acceptance"],
            "evidence": [{"key": "verification.status", "quote": "passed"}], "corrective_action": "Review"}
        for field, invalid in (("requirement_ids", "add-acceptance"), ("test_ids", "add"),
                               ("requirement_ids", "unknown-requirement"), ("test_ids", "unknown-test")):
            with self.subTest(field=field, identifier=invalid):
                bad = {**finding, field: [invalid]}
                result = self.invoke(CollaborationMockClient(qa_override={"findings": [bad]}))
                self.assertEqual(result["qa_status"], "failed")
                self.assertTrue(result["acceptance"]["accepted"])
                root = Path(result["run_dir"])
                errors = json.loads((root / "qa_validation_error.json").read_text())["errors"]
                self.assertTrue(any(error["loc"] == ["findings", 0, field, 0] for error in errors))
                self.assertEqual(json.loads((root / "qa_raw.txt").read_text())["findings"][0][field], [invalid])
                self.assertTrue((root / "qa_metrics.json").is_file())
                self.assertEqual(self.client.roles.count("qa"), 1)
                self.assertFalse((root / "qa_review.json").exists())

    def test_qa_schema_rejects_quotes_paired_with_the_wrong_key(self):
        task = TaskSpec.model_validate_json((ROOT / "tasks/todo.json").read_text())
        registry = {"verification.status": "failed", "coverage.responsive.status": "passed"}
        schema = grounded_qa_review_schema(registry, task)
        finding = {"classification": "observed", "diagnostic_role": "independent", "category": "functional",
            "description": "A tested outcome", "requirement_ids": ["responsive"], "test_ids": [],
            "evidence": [{"key": "verification.status", "quote": "passed"}], "corrective_action": "Review"}
        with self.assertRaises(ValidationError):
            schema.model_validate({"summary": "Results", "findings": [finding]})
        finding["evidence"] = [{"key": "coverage.responsive.status", "quote": "passed"}]
        review = schema.model_validate({"summary": "Results", "findings": [finding]})
        validate_evidence(review.findings[0].evidence[0], registry)

    def test_qa_schema_supports_manifests_without_tests(self):
        task = TaskSpec.model_validate_json((ROOT / "tasks/todo.json").read_text())
        task = task.model_copy(update={"tests": []})
        schema = grounded_qa_review_schema({"verification.status": "passed"}, task)
        self.assertEqual(schema.model_json_schema()["$defs"]["QAFinding"]["properties"]["test_ids"]["maxItems"], 0)
        finding = {"classification": "hypothesis", "diagnostic_role": "independent", "category": "functional",
            "description": "Inspect an untested behavior", "requirement_ids": ["add"], "test_ids": [],
            "evidence": [{"key": "verification.status", "quote": "passed"}], "corrective_action": "Review"}
        schema.model_validate({"summary": "Results", "findings": [finding]})
        finding["test_ids"] = ["add-acceptance"]
        with self.assertRaises(ValidationError):
            schema.model_validate({"summary": "Results", "findings": [finding]})

    def test_archived_invalid_evidence_key_is_rejected_by_the_run_schema(self):
        def archived_failure(path, task, output):
            report = stub_verifier(path, task, output)
            report["status"] = "failed"
            report["checks"]["javascript_syntax"] = False
            report["javascript_diagnostics"] = [{"kind": "syntax_error", "message": "Unexpected identifier 'aria'",
                                                   "html_line": 283, "column": 22}]
            target = next(test for test in report["tests"] if test["test_id"] == "add-acceptance")
            target.update(status="failed", requirement_id="add", error="TimeoutError: Locator.click: Timeout 5000ms exceeded.",
                          runtime_errors=["Unexpected identifier 'aria'"])
            report["coverage"]["add"]["status"] = "failed"
            return report
        finding = {"classification": "observed", "diagnostic_role": "primary_defect", "category": "runtime_exception",
            "description": "The script cannot be evaluated", "requirement_ids": ["add"], "test_ids": ["add-acceptance"],
            "evidence": [{"key": "Unexpected identifier 'aria'", "quote": "invented explanation"}],
            "test_outcomes": {"add-acceptance": "failed"}, "corrective_action": "Correct the invalid statement"}
        result = self.invoke(CollaborationMockClient(qa_override={"findings": [finding]}), verifier=archived_failure)
        self.assertEqual(result["qa_status"], "failed", result)
        self.assertIn("ValidationError", result["qa_error"])
        request = json.loads((Path(result["run_dir"]) / "qa_request.json").read_text())
        variants = evidence_schema_variants(request["response_format"]["json_schema"]["schema"])
        keys = {key for variant in variants for key in literal_values(variant["key"])}
        self.assertIn("javascript_diagnostics.0.message", keys)
        self.assertNotIn("Unexpected identifier 'aria'", keys)
        self.assertTrue((Path(result["run_dir"]) / "qa_validation_error.json").is_file())
        self.assertFalse((Path(result["run_dir"]) / "qa_review.json").exists())

    def test_qa_contract_prioritizes_syntax_error_over_timeout_symptoms(self):
        report = {"status": "failed", "checks": {"javascript_syntax": False}, "errors": [], "coverage": {},
            "javascript_diagnostics": [{"kind": "syntax_error", "message": "SyntaxError: Unexpected identifier 'aria'",
                                         "html_line": 283, "column": 22}],
            "tests": [{"test_id": "add-acceptance", "status": "failed",
                       "error": "TimeoutError: Locator.click: Timeout 5000ms exceeded.",
                       "runtime_errors": ["Unexpected identifier 'aria'"]}]}
        registry = evidence_registry(report)
        task = TaskSpec.model_validate_json((ROOT / "tasks/todo.json").read_text())
        schema = grounded_qa_review_schema(registry, task)
        grounded_review = schema.model_validate({"summary": "Syntax prevents the interaction handler from running", "findings": [
            {"classification": "observed", "diagnostic_role": "primary_defect", "category": "runtime_exception",
             "description": "The standalone script has a syntax error", "requirement_ids": ["add"],
             "test_ids": ["add-acceptance"], "evidence": [{"key": "javascript_diagnostics.0.message",
             "quote": "SyntaxError: Unexpected identifier 'aria'"}], "test_outcomes": {"add-acceptance": "failed"},
             "likely_cause": "Invalid JavaScript prevents script evaluation", "corrective_action": "Correct the invalid statement"},
            {"classification": "observed", "diagnostic_role": "secondary_symptom", "category": "functional",
             "description": "The click timeout follows the script failure", "requirement_ids": ["add"],
             "test_ids": ["add-acceptance"], "evidence": [{"key": "tests.add-acceptance.error",
             "quote": report["tests"][0]["error"]}], "test_outcomes": {"add-acceptance": "failed"},
             "likely_cause": "The submit handler was never installed", "corrective_action": "Resolve the primary syntax defect"}]})
        review = QAReview.model_validate(grounded_review.model_dump(mode="json"))
        for finding in review.findings:
            for reference in finding.evidence:
                validate_evidence(reference, registry)
        validate_diagnostic_hierarchy(review, report, registry)
        bad = review.model_copy(deep=True)
        bad.findings[0].diagnostic_role = "independent"
        with self.assertRaisesRegex(ValueError, "primary defect"):
            validate_diagnostic_hierarchy(bad, report, registry)
        bad = review.model_copy(deep=True)
        bad.findings[1].diagnostic_role = "independent"
        with self.assertRaisesRegex(ValueError, "secondary symptoms"):
            validate_diagnostic_hierarchy(bad, report, registry)
        missing = review.model_copy(deep=True)
        missing.findings = missing.findings[:1]
        with self.assertRaisesRegex(ValueError, "grounded secondary symptoms"):
            validate_diagnostic_hierarchy(missing, report, registry)

    def test_qa_verdict_field_is_not_allowed(self):
        result = self.invoke(CollaborationMockClient(qa_override={"accepted": True}))
        self.assertEqual(result["qa_status"], "failed")
        self.assertTrue(result["acceptance"]["accepted"])

    def test_malformed_specialists_both_finish_and_join_rejects(self):
        for role in ("ux", "architect"):
            with self.subTest(role=role):
                result = self.invoke(CollaborationMockClient(malformed=role))
                self.assertEqual(result["status"], "failed")
                self.assertIn("ux", self.client.roles)
                self.assertIn("architect", self.client.roles)
                self.assertNotIn("coordinator", self.client.roles)
                self.assertNotIn("implementer", self.client.roles)
                self.assertIn(role + "_error", result)
                self.assertFalse(result["acceptance"]["accepted"])

    def test_bad_coordinator_preserves_authoritative_contract(self):
        for overrides in ({"mandatory_contract_hash": "altered"}, {"runtime_kind": "full_stack"},
                          {"technology_constraints": ["Invented infrastructure"]}, {"requirement_traceability": {}}):
            with self.subTest(overrides=overrides):
                result = self.invoke(CollaborationMockClient(coordinator_override=overrides))
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("implementer", self.client.roles)
                self.assertTrue((Path(result["run_dir"]) / "coordinator_raw.txt").exists())
                if 'requirement_traceability' not in overrides:
                    diagnostics = json.loads((Path(result['run_dir']) / 'coordinator_validation_error.json').read_text())
                    self.assertEqual(diagnostics['response_schema'], 'CoordinationProposal')
                    self.assertIn('extra_forbidden', [error['type'] for error in diagnostics['errors']])

    def test_framework_assembles_authoritative_fields_after_coordinator_proposal(self):
        from agentic_web.collaboration_schema import CoordinationProposal, CoordinatedSpecification
        from agentic_web.implementation_contract import derive_implementation_contract
        from agentic_web.provenance import fingerprint
        result = self.invoke()
        self.assertEqual(result['status'], 'passed')
        directory = Path(result['run_dir'])
        raw = json.loads((directory / 'coordinator_raw.txt').read_text())
        proposal = CoordinationProposal.model_validate(raw)
        for field in ('technology_constraints', 'runtime_kind', 'mandatory_contract_hash'):
            self.assertNotIn(field, raw)
            self.assertNotIn(field, CoordinationProposal.model_json_schema()['properties'])
        saved = json.loads((directory / 'coordination.json').read_text())
        spec = CoordinatedSpecification.model_validate(saved)
        task = TaskSpec.model_validate(result['task_spec'])
        self.assertEqual(spec.technology_constraints, task.technology_constraints)
        self.assertEqual(spec.runtime_kind, task.runtime.kind)
        self.assertEqual(spec.mandatory_contract_hash, fingerprint(derive_implementation_contract(task)))
        self.assertEqual({key: saved[key] for key in CoordinationProposal.model_fields}, proposal.model_dump(mode='json'))
        self.assertEqual(saved, result['coordinated_specification'])

    def test_unavailable_model_fails_without_substitution(self):
        result = self.invoke(CollaborationMockClient(unavailable="ux-model"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("ux-model", result["error"])
        self.assertIn("Unavailable LM Studio model IDs", result["error"])
        self.assertEqual(self.client.roles, [])

    def test_role_transport_failure_retained(self):
        result = self.invoke(CollaborationMockClient(transport_failure="architect"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("architecture-model", result["architect_error"])
        directory = Path(result["run_dir"])
        self.assertTrue((directory / "architect_metrics.json").exists())
        self.assertTrue((directory / "architect_error.json").exists())

    def test_unavailable_qa_is_incomplete_without_blocking_acceptance(self):
        result = self.invoke(CollaborationMockClient(unavailable="qa-model"))
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["acceptance"]["accepted"])
        self.assertEqual(result["qa_status"], "failed")
        self.assertIn("qa-model", result["qa_error"])
        self.assertNotIn("qa", self.client.roles)

    def test_unverified_requirements_remain_rejected_even_with_qa_success(self):
        def unverified(path, task, output):
            report = stub_verifier(path, task, output)
            report["status"] = "unverified"
            report["coverage"][task.features[0].id]["status"] = "unverified"
            return report
        result = self.invoke(verifier=unverified)
        self.assertEqual(result["qa_status"], "completed")
        self.assertFalse(result["acceptance"]["accepted"])
        self.assertEqual(result["status"], "unverified")

    def test_incomplete_implementation_skips_qa_and_rejects(self):
        class Incomplete(CollaborationMockClient):
            def complete(self, *args, planner=False, **kwargs):
                if not planner and "\"role\":" not in args[1]:
                    self.roles.append("implementer")
                    return "<html>", {"finish_reason": "length"}
                return super().complete(*args, planner=planner, **kwargs)
        result = self.invoke(Incomplete())
        self.assertEqual(result["qa_status"], "skipped")
        self.assertFalse(result["acceptance"]["accepted"])
        self.assertEqual(result["status"], "failed")
        self.assertTrue((Path(result["run_dir"]) / "implementer_raw.txt").exists())

    def test_gate_rejects_inconsistent_pass_report(self):
        def incomplete(path, task, output):
            return {"status": "passed", "checks": {"loads": True}, "tests": [], "coverage": {}, "errors": []}
        result = self.invoke(verifier=incomplete)
        self.assertFalse(result["acceptance"]["accepted"])
        self.assertEqual(result["status"], "failed")

    def test_collaborative_graph_schema_and_join(self):
        nodes = set(collaborative_app.get_graph().nodes)
        self.assertEqual(nodes, {"__start__", "requirements_prepare", "requirements_analyst", "requirements_review", "requirements_revision",
                                "requirements_finalize", "handoff_decomposer", "task_decomposition", "task_loader", "planner", "ux_agent", "architecture_agent", "coordinator",
                                 "implementer", "verifier", "qa_critic", "acceptance_gate", "__end__"})
        self.assertIn("task_path", collaborative_app.get_input_jsonschema()["properties"])
        self.assertIn("ux_review", collaborative_app.get_output_jsonschema()["properties"])

    def test_specialist_inference_preserves_parallelism(self):
        import threading
        import time
        class TrackingClient(CollaborationMockClient):
            active = 0
            maximum_active = 0
            counter_lock = threading.Lock()
            barrier = threading.Barrier(2)
            def complete(self, *args, **kwargs):
                role = json.loads(args[1]).get("role")
                if role not in {"ux", "architect"}:
                    return super().complete(*args, **kwargs)
                with self.counter_lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                try:
                    self.barrier.wait(timeout=5)
                    return super().complete(*args, **kwargs)
                finally:
                    with self.counter_lock:
                        self.active -= 1
        client = TrackingClient()
        result = self.invoke(client)
        self.assertTrue(result["acceptance"]["accepted"])
        self.assertEqual(client.maximum_active, 2)

    def test_malformed_coordinator_has_explicit_validation_failure(self):
        result = self.invoke(CollaborationMockClient(malformed="coordinator"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("ValidationError", result["error"])
        self.assertNotIn("implementer", self.client.roles)
        self.assertTrue((Path(result["run_dir"]) / "coordinator_error.json").exists())

    def test_cli_selects_collaborative_and_preserves_role_flags(self):
        from agentic_web.__main__ import main
        flags = ["agentic_web", "generate", "--task", str(ROOT / "tasks/todo.json"), "--workflow", "collaborative",
                 "--model", "coder", "--planner-model", "planner", "--ux-model", "ux", "--architect-model", "architect",
                 "--coordinator-model", "coordinator", "--qa-model", "qa"]
        with patch("sys.argv", flags), patch("agentic_web.__main__.collaborative_app") as graph, patch("builtins.print"):
            graph.invoke.return_value = {"status": "passed", "run_dir": self.temp.name}
            self.assertEqual(main(), 0)
            values = graph.invoke.call_args.args[0]["model_config"]
            self.assertEqual(values["planner_model"], "planner")
            self.assertEqual(values["qa_model"], "qa")

    def test_real_browser_collaborative_fixture_generations(self):
        graph = build_graph(CollaborationMockClient(), workflow="collaborative")
        for name in ("eventhub", "todo", "catalog"):
            with self.subTest(task=name):
                result = graph.invoke({"task_path": str(ROOT / f"tasks/{name}.json"), "model_config": self.config.model_dump(),
                                      "output_root": self.temp.name, "execution_mode": "mocked-model-real-browser"})
                self.assertEqual(result["status"], "passed", result)
                self.assertTrue(result["acceptance"]["accepted"])
                self.assertEqual(result["qa_status"], "completed")


if __name__ == "__main__":
    unittest.main()
