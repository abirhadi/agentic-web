"""A1 analysis contracts and compatibility; model responses are mocked."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from jsonschema_rs import Draft202012Validator

from agentic_web.client import ModelConfig, StructuredOutputRequestError, json_schema_response_format
from agentic_web.intake import build_intake_graph
from agentic_web.intake_schema import IntakeSource, RequirementsAnalysis, analysis_response_schema, assemble_analysis, quotation_choices
from agentic_web.provenance import fingerprint
from agentic_web.schema import load_task
from agentic_web.workflow import build_graph, contract_hash
from collaboration_mock_client import CollaborationMockClient
from mock_client import ROOT
from test_framework import stub_verifier


PROMPT = "Build a website that displays the heading 'Community Notice'."


def analysis_fixture(prompt=PROMPT, context=None):
    source = {"source": "request", "quote": prompt}
    return {"analysis_version": "1.0", "task_spec": {
        "task_id": "community-notice", "name": "Community Notice", "description": "Publish the requested notice",
        "original_requirements": prompt, "supporting_context": context,
        "runtime": {"kind": "single_html", "entrypoint": "index.html"}, "technology_constraints": [],
        "features": [
            {"id": "notice", "description": "Display the requested notice heading", "required": True,
             "acceptance_criteria": ["The requested heading is visible in page content"], "verification": "browser"},
            {"id": "accent", "description": "Consider a consistent optional accent palette", "required": False,
             "acceptance_criteria": ["A human can review the optional color treatment"], "verification": "manual"}],
        "nonfunctional_requirements": [{"id": "responsive", "description": "Keep content usable at desktop and mobile widths",
            "acceptance_criteria": ["Neither tested viewport has horizontal overflow"], "required": True, "verification": "browser"}],
        "design_freedom": ["Layout, typography, colors and component composition remain implementation choices"],
        "forbidden_assumptions": ["Do not invent factual content absent from the request or supporting context"],
        "tests": [{"test_id": "notice-content", "requirement_id": "notice", "required": True,
            "expected_results": [{"assertion": "contains_text", "selector": "body", "value": "Community Notice"}]},
            {"test_id": "responsive-viewports", "requirement_id": "responsive", "test_type": "universal",
             "universal_checks": ["desktop_no_horizontal_overflow", "mobile_no_horizontal_overflow"], "required": True},
            {"test_id": "accent-review", "requirement_id": "accent", "test_type": "manual", "required": False}]},
        "requirement_traceability": [
            {"requirement_id": "notice", "origin": "explicit", "rationale": "The requested heading is directly stated", "sources": [source]},
            {"requirement_id": "responsive", "origin": "inferred", "rationale": "A website is likely used at different viewport sizes", "sources": [source]},
            {"requirement_id": "accent", "origin": "recommendation", "rationale": "Visual polish is optional advice", "sources": [source]}],
        "constraint_traceability": [{"target": "runtime", "origin": "framework_default", "index": None,
            "rationale": "No runtime was specified; use the currently implemented single_html adapter", "sources": []}],
        "missing_information": [], "unresolved_ambiguities": []}


def wire_response(canonical):
    """Build mock wire responses; production never accepts the old flat form."""
    response = deepcopy(canonical)
    task = response["task_spec"]
    source_text = {"request": task["original_requirements"], "context": task.get("supporting_context")}
    def constrain_mock_references(value):
        if isinstance(value, dict):
            if "source" in value and "quote" in value:
                text = source_text.get(value["source"])
                if text and value["quote"] in text:
                    choices = quotation_choices(text)
                    if value["quote"] not in choices:
                        value["quote"] = next((choice for choice in choices if value["quote"] in choice), value["quote"])
            for child in value.values(): constrain_mock_references(child)
        elif isinstance(value, list):
            for child in value: constrain_mock_references(child)
    constrain_mock_references(response)
    for group in ("missing_information", "unresolved_ambiguities"):
        for issue in response[group]:
            issue.pop("issue_id", None)
    task.pop("supporting_context", None)
    constraints = response.pop("constraint_traceability")
    expected_targets = {("runtime", None)} | {("technology_constraint", i) for i in range(len(task.get("technology_constraints", [])))}
    targets = [(trace.get("target"), trace.get("index")) for trace in constraints]
    valid_constraints = len(set(targets)) == len(targets) and set(targets) == expected_targets and all(
        "index" in trace or trace.get("target") == "runtime" for trace in constraints)
    if valid_constraints:
        constraint_map = {(trace["target"], trace.get("index")): {key: value for key, value in trace.items() if key not in {"target", "index"}}
                          for trace in constraints}
        task["runtime"]["traceability"] = constraint_map[("runtime", None)]
        task["technology_constraints"] = [{"value": text, "traceability": constraint_map[("technology_constraint", index)]}
                                          for index, text in enumerate(task.get("technology_constraints", []))]
    else:
        response["invalid_constraint_traceability"] = constraints
        task["runtime"]["traceability"] = None
    tests = task.pop("tests")
    traces = response.pop("requirement_traceability")
    for requirement in task["features"] + task.get("nonfunctional_requirements", []):
        for issue_group in ("missing_information", "unresolved_ambiguities"):
            related = [issue for issue in response[issue_group] if issue["requirement_ids"] == [requirement["id"]]]
            requirement[issue_group] = [{key: value for key, value in issue.items() if key != "requirement_ids"} for issue in related]
            response[issue_group] = [issue for issue in response[issue_group] if issue not in related]
        matches = [trace for trace in traces if trace["requirement_id"] == requirement["id"]]
        requirement["traceability"] = {key: value for key, value in matches[0].items() if key != "requirement_id"} if matches else None
        requirement["tests"] = [{key: value for key, value in test.items() if key != "requirement_id"}
                                for test in tests if test["requirement_id"] == requirement["id"]]
        for test in requirement["tests"]:
            if test.get("required", True) == requirement.get("required", True):
                test.pop("required", None)
    ids = {requirement["id"] for requirement in task["features"] + task.get("nonfunctional_requirements", [])}
    if any(test["requirement_id"] not in ids for test in tests) or len(traces) != len(ids) or any(trace["requirement_id"] not in ids for trace in traces):
        response["invalid_orphan_references"] = True
    return response


class IntakeMockClient:
    def __init__(self, response=None, *, raw=None, finish_reason="stop", error=None):
        self.response = deepcopy(response if response is not None else analysis_fixture())
        self.raw, self.finish_reason, self.error = raw, finish_reason, error
        self.calls = []

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        self.calls.append({"system": system, "payload": json.loads(user), "config": config,
                           "planner": planner, "response_schema": response_schema})
        if self.error:
            raise self.error
        response = wire_response(self.response)
        if self.response["task_spec"].get("supporting_context") != json.loads(user)["supporting_context"]:
            response["task_spec"]["supporting_context"] = self.response["task_spec"].get("supporting_context")
        return self.raw if self.raw is not None else json.dumps(response, ensure_ascii=False), {
            "model": "mock-analyst", "elapsed_seconds": 2.5, "finish_reason": self.finish_reason,
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}, "mocked": True}


class CandidateClient(CollaborationMockClient):
    def complete(self, system, user, config, *, planner=False, response_schema=None):
        payload = json.loads(user)
        if not planner and "role" not in payload:
            self.roles.append("implementer")
            self.calls.append({"system": system, "payload": payload, "config": config.model_dump(),
                               "planner": False, "response_schema": response_schema})
            return '<!DOCTYPE html><html lang="en"><head><title>Community Notice</title></head><body><h1>Community Notice</h1></body></html>', {
                "model": "mock-candidate", "finish_reason": "stop", "elapsed_seconds": 0.1, "usage": None}
        return super().complete(system, user, config, planner=planner, response_schema=response_schema)


class IntegratedIntakeClient(CandidateClient):
    def __init__(self, response=None, **kwargs):
        super().__init__()
        self.intake_client = IntakeMockClient(response, **kwargs)

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        payload = json.loads(user)
        if payload.get("role") == "requirements_analyst":
            self.roles.append("requirements_analyst")
            return self.intake_client.complete(system, user, config, planner=planner, response_schema=response_schema)
        return super().complete(system, user, config, planner=planner, response_schema=response_schema)


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = ModelConfig(model="analyst-model", seed=42)

    def invoke(self, response=None, *, client=None, prompt=PROMPT, context=None, name="task.json"):
        self.client = client or IntakeMockClient(response)
        self.output = Path(self.temp.name) / name
        return build_intake_graph(self.client).invoke({"prompt": prompt, "supporting_context": context,
            "output_file": str(self.output), "model_config": self.config.model_dump()})

    def test_free_form_analysis_saves_valid_reusable_task_and_provenance(self):
        result = self.invoke()
        self.assertEqual(result["status"], "analyzed", result)
        task = load_task(self.output)
        self.assertEqual(task.original_requirements, PROMPT)
        record = json.loads(Path(result["analysis_file"]).read_text())
        self.assertEqual(result["analysis"], record)
        self.assertEqual(result["task_spec"], task.model_dump(mode="json"))
        self.assertEqual(record["analysis_version"], "1.0")
        self.assertEqual(record["source_hash"], fingerprint({"original_request": PROMPT, "supporting_context": None}))
        self.assertEqual(record["task_hash"], contract_hash(task))
        self.assertEqual(record["model_config"], {**self.config.model_dump(), 'inference_role': 'intake'})
        self.assertEqual(record["inference_metrics"]["usage"]["total_tokens"], 300)
        self.assertEqual(record["inference_metrics"]["finish_reason"], "stop")
        self.assertIsNone(record["inference_metrics"]["max_tokens"])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(set(json.loads(self.output.read_text())), set(task.model_dump()))
        root = Path(result["run_dir"])
        self.assertTrue((root / "ledger/requirements_analyst.json").is_file())
        self.assertFalse((root / "application").exists())

    def test_explicit_inferred_and_optional_recommendations_stay_distinct(self):
        result = self.invoke()
        record = json.loads(Path(result["analysis_file"]).read_text())
        self.assertEqual(record["requirement_groups"], {"explicit": ["notice"], "inferred": ["responsive"], "recommendation": ["accent"]})
        task = load_task(self.output)
        self.assertFalse(next(requirement for requirement in task.requirements if requirement.id == "accent").required)
        self.assertFalse(next(test for test in task.tests if test.requirement_id == "accent").required)
        self.assertTrue(task.design_freedom)

    def test_preserves_unicode_whitespace_and_supporting_context_exactly(self):
        prompt = "  " + PROMPT + "\r\nKeep the café’s wording.\n"
        context = "Approved text: café\r\nContact details are not supplied.\n"
        response = analysis_fixture(prompt, context)
        response["requirement_traceability"][0]["sources"].append({"source": "context", "quote": "café"})
        result = self.invoke(response, prompt=prompt, context=context)
        self.assertEqual(result["status"], "analyzed", result)
        task = load_task(self.output)
        self.assertEqual(task.original_requirements, prompt)
        self.assertEqual(task.supporting_context, context)
        self.assertEqual(Path(result["run_dir"], "original_request.txt").read_bytes(), prompt.encode("utf-8"))
        self.assertEqual(Path(result["run_dir"], "supporting_context.txt").read_bytes(), context.encode("utf-8"))

    def test_short_requests_remain_compatible_without_padding(self):
        prompt = "Make a site"
        response = analysis_fixture(prompt)
        response["task_spec"]["features"][0].update(description="Make the requested website available",
            acceptance_criteria=["Website renders its main content at both viewports"])
        response["task_spec"]["tests"][0] = {"test_id": "site-render", "requirement_id": "notice",
            "test_type": "universal", "universal_checks": ["desktop_renders", "mobile_renders"], "required": True}
        result = self.invoke(response, prompt=prompt)
        self.assertEqual(result["status"], "analyzed", result)
        self.assertEqual(load_task(self.output).original_requirements, prompt)

    def test_missing_source_material_is_blocking_without_invented_content(self):
        prompt = "Generate a professional personal website from my resume."
        response = analysis_fixture(prompt)
        response["task_spec"]["features"][0].update(description="Represent the user's supplied source material accurately",
            acceptance_criteria=["A human confirms website facts against the supplied source"], verification="manual")
        response["task_spec"]["tests"][0] = {"test_id": "source-review", "requirement_id": "notice", "test_type": "manual", "required": True}
        response["missing_information"] = [{"issue_id": "source-material", "description": "The requested source material was not supplied",
            "question": "What source text should the website represent?", "blocks_generation": True, "requirement_ids": ["notice"],
            "sources": [{"source": "request", "quote": "from my resume"}]}]
        result = self.invoke(response, prompt=prompt)
        self.assertEqual(result["status"], "needs_review", result)
        self.assertFalse(result["ready_for_generation"])
        record = json.loads(Path(result["analysis_file"]).read_text())
        self.assertEqual(record["blocking_issue_ids"], ["missing_001"])
        self.assertIsNone(record["task_spec"].get("supporting_context"))
        self.assertEqual(record["task_spec"]["features"][0]["verification"], "manual")

    def test_nonblocking_ambiguities_remain_visible(self):
        response = analysis_fixture()
        response["unresolved_ambiguities"] = [{"issue_id": "visual-preference", "description": "No visual treatment was specified",
            "question": "Is there a preferred visual style?", "blocks_generation": False, "requirement_ids": [], "sources": []}]
        result = self.invoke(response)
        self.assertTrue(result["ready_for_generation"])
        self.assertEqual(len(json.loads(Path(result["analysis_file"]).read_text())["unresolved_ambiguities"]), 1)

    def test_schema_constrains_original_sources_and_preserves_actual_root(self):
        result = self.invoke()
        call = self.client.calls[0]
        schema = call["response_schema"].model_json_schema()
        self.assertEqual(schema["$defs"]["IntakeTaskSpec"]["properties"]["original_requirements"]["const"], PROMPT)
        self.assertIn("task_spec", schema["properties"])
        self.assertNotIn("RequirementsAnalysis", schema["properties"])
        request = json.loads(Path(result["run_dir"], "intake_request.json").read_text())
        self.assertEqual(request["response_format"], json_schema_response_format(call["response_schema"]))
        self.assertTrue(call["planner"])
        self.assertNotIn("output_schema", call["payload"])
        self.assertEqual(call["payload"]["response_fields"], list(call["response_schema"].model_fields))
        self.assertIn("mobile_basic_accessibility", call["payload"]["universal_checks"])

    def test_large_evidence_occurs_once_in_prompt_and_never_in_schema_or_response(self):
        context = "UNIQUE_EVIDENCE_START\n" + ("Approved source text. " * 1000)
        result = self.invoke(analysis_fixture(PROMPT, context), context=context)
        self.assertEqual(result["status"], "analyzed", result)
        call = self.client.calls[0]
        schema = call["response_schema"].model_json_schema()
        self.assertNotIn("supporting_context", schema["$defs"]["IntakeTaskSpec"]["properties"])
        self.assertNotIn(context, json.dumps(schema))
        request = json.loads(Path(result["run_dir"], "intake_request.json").read_text())
        self.assertEqual(request["user"].count("UNIQUE_EVIDENCE_START"), 1)
        raw = json.loads(Path(result["run_dir"], "intake_raw.txt").read_text())
        self.assertNotIn("supporting_context", raw["task_spec"])
        self.assertEqual(load_task(self.output).supporting_context, context)
        self.assertEqual(Path(result["run_dir"], "supporting_context.txt").read_bytes(), context.encode())

    def test_long_evidence_is_not_embedded_as_a_full_document_constant(self):
        short = analysis_response_schema(IntakeSource(original_request=PROMPT, supporting_context="Evidence"))
        long = analysis_response_schema(IntakeSource(original_request=PROMPT, supporting_context="Evidence " * 10000))
        self.assertNotIn("supporting_context", long.model_json_schema()["$defs"]["IntakeTaskSpec"]["properties"])
        self.assertLess(len(json.dumps(long.model_json_schema())), len("Evidence " * 10000))

    def test_model_cannot_echo_or_replace_evidence_in_the_wire_response(self):
        context = "Original approved evidence"
        raw = wire_response(analysis_fixture(PROMPT, context))
        raw["task_spec"]["supporting_context"] = "Rewritten or echoed evidence"
        result = self.invoke(client=IntakeMockClient(raw=json.dumps(raw)), context=context)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())
        self.assertIn("Extra inputs are not permitted", result["error"])

    def test_quote_choices_are_exact_bounded_excerpts_with_unicode(self):
        text = "  Approved café wording.\r\n" + ("Document text " * 300)
        choices = quotation_choices(text)
        self.assertTrue(choices)
        self.assertEqual(len(choices), len(set(choices)))
        self.assertTrue(all(quote in text and quote.strip() and len(quote) <= 1000 for quote in choices))
        self.assertIn("Approved café wording.", choices)

    def test_json_schema_pairs_each_quote_with_its_actual_source(self):
        prompt = "Generate a personal website from my resume."
        context = "Alex Example\nApproved professional biography."
        response = wire_response(analysis_fixture(prompt, context))
        schema = analysis_response_schema(IntakeSource(original_request=prompt, supporting_context=context))
        validator = Draft202012Validator(schema.model_json_schema())
        reference = response["task_spec"]["nonfunctional_requirements"][0]["traceability"]["sources"][0]
        self.assertTrue(validator.is_valid(response))
        reference.update(source="context", quote=prompt)
        self.assertFalse(validator.is_valid(response))
        with self.assertRaises(ValueError): schema.model_validate(response)
        reference.update(source="request", quote="Alex Example")
        self.assertFalse(validator.is_valid(response))
        reference.update(source="context", quote="Alex Example")
        canonical = assemble_analysis(schema.model_validate(response), supporting_context=context)
        self.assertEqual(canonical.task_spec.supporting_context, context)

    def test_archived_request_as_context_failure_pattern_remains_rejected(self):
        prompt = "Generate a personal website from my resume."
        context = "Alex Example\nApproved professional biography."
        response = wire_response(analysis_fixture(prompt, context))
        for requirement in response["task_spec"]["nonfunctional_requirements"]:
            requirement["traceability"]["sources"] = [{"source": "context", "quote": prompt}]
        raw = json.dumps(response)
        result = self.invoke(client=IntakeMockClient(raw=raw), prompt=prompt, context=context)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())
        self.assertEqual(Path(result["run_dir"], "intake_raw.txt").read_text(), raw)
        self.assertEqual(len(self.client.calls), 1)

    def test_canonical_quote_validation_remains_strict_after_schema_parsing(self):
        context = "Alex Example"
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT, supporting_context=context))
        parsed = schema.model_validate(wire_response(analysis_fixture(PROMPT, context)))
        # Deliberately bypass the schema to verify the independent final guard.
        parsed.task_spec.features[0].traceability.sources[0].source = "context"
        with self.assertWarns(UserWarning):
            with self.assertRaisesRegex(ValueError, "exact substrings"):
                assemble_analysis(parsed, supporting_context=context)

    def test_requirement_issues_are_bound_by_containment_and_global_issues_stay_global(self):
        response = wire_response(analysis_fixture())
        response["task_spec"]["features"][0]["missing_information"] = [{"description": "Missing approved content",
            "question": "Provide content", "blocks_generation": True, "sources": [{"source": "request", "quote": PROMPT}]}]
        response["unresolved_ambiguities"] = [{"description": "Global decision", "question": "Clarify scope",
            "blocks_generation": False, "requirement_ids": [], "sources": []}]
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        canonical = assemble_analysis(schema.model_validate(response))
        self.assertEqual(canonical.missing_information[0].requirement_ids, ["notice"])
        self.assertEqual(canonical.unresolved_ambiguities[0].requirement_ids, [])
        response["task_spec"]["features"][0]["missing_information"][0]["requirement_ids"] = ["invented"]
        with self.assertRaises(ValueError): schema.model_validate(response)

    def test_top_level_issues_cannot_reference_independently_generated_requirement_ids(self):
        response = wire_response(analysis_fixture())
        response["missing_information"] = [{"description": "Missing information",
            "question": "Provide content", "blocks_generation": True, "requirement_ids": ["undeclared"], "sources": []}]
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        self.assertFalse(Draft202012Validator(schema.model_json_schema()).is_valid(response))
        with self.assertRaises(ValueError): schema.model_validate(response)

    def test_framework_assigns_unique_issue_ids_without_merging_or_changing_issues(self):
        response = wire_response(analysis_fixture())
        issue = {"description": "An unresolved content decision", "question": "What content is approved?",
                 "blocks_generation": True, "sources": [{"source": "request", "quote": PROMPT}]}
        response["missing_information"] = [{**deepcopy(issue), "requirement_ids": []}]
        response["unresolved_ambiguities"] = [{**deepcopy(issue), "requirement_ids": []}]
        response["task_spec"]["features"][0]["missing_information"] = [deepcopy(issue), deepcopy(issue)]
        response["task_spec"]["nonfunctional_requirements"][0]["unresolved_ambiguities"] = [deepcopy(issue)]
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        first = assemble_analysis(schema.model_validate(response))
        second = assemble_analysis(schema.model_validate(response))
        issues = first.missing_information + first.unresolved_ambiguities
        self.assertEqual(first, second)
        self.assertEqual([item.issue_id for item in issues], ["missing_001", "missing_002", "missing_003", "ambiguity_001", "ambiguity_002"])
        self.assertEqual([item.requirement_ids for item in issues], [[], ["notice"], ["notice"], [], ["responsive"]])
        self.assertTrue(all(item.blocks_generation and item.description == issue["description"] and item.question == issue["question"] for item in issues))

    def test_schema_prohibits_model_issue_ids_at_every_issue_location(self):
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        validator = Draft202012Validator(schema.model_json_schema())
        for group in ("missing_information", "unresolved_ambiguities"):
            for nested in (False, True):
                with self.subTest(group=group, nested=nested):
                    response = wire_response(analysis_fixture())
                    issue = {"description": "A decision", "question": "Please clarify", "blocks_generation": False,
                             "sources": [], "issue_id": "repeated_name"}
                    if nested:
                        response["task_spec"]["features"][0][group] = [issue]
                    else:
                        response[group] = [{**issue, "requirement_ids": []}]
                    self.assertFalse(validator.is_valid(response))
                    with self.assertRaises(ValueError): schema.model_validate(response)

    def test_canonical_duplicate_issue_ids_remain_invalid_for_archived_data(self):
        response = analysis_fixture()
        issue = {"issue_id": "repeated_name", "description": "A decision", "question": "Please clarify",
                 "blocks_generation": False, "sources": [], "requirement_ids": ["notice"]}
        response["missing_information"] = [deepcopy(issue)]
        response["unresolved_ambiguities"] = [deepcopy(issue)]
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            RequirementsAnalysis.model_validate(response)

    def test_json_schema_prohibits_mixed_browser_and_universal_tests(self):
        validator = Draft202012Validator(analysis_response_schema(IntakeSource(original_request=PROMPT)).model_json_schema())
        response = analysis_fixture()
        self.assertTrue(validator.is_valid(wire_response(response)))
        response["task_spec"]["tests"][1].update(test_type="browser",
            expected_results=[{"assertion": "visible", "selector": "body"}])
        self.assertFalse(validator.is_valid(wire_response(response)))
        response["task_spec"]["tests"][1]["test_type"] = "universal"
        self.assertFalse(validator.is_valid(wire_response(response)))
        response["task_spec"]["tests"][1]["expected_results"] = []
        self.assertTrue(validator.is_valid(wire_response(response)))

    def test_json_schema_distinguishes_runtime_and_technology_indexes(self):
        validator = Draft202012Validator(analysis_response_schema(IntakeSource(original_request=PROMPT)).model_json_schema())
        response = analysis_fixture()
        response["task_spec"]["technology_constraints"] = ["Inline styles within the default runtime"]
        response["constraint_traceability"].append({"target": "technology_constraint", "index": 0,
            "origin": "framework_default", "rationale": "An inline default-runtime constraint", "sources": []})
        self.assertTrue(validator.is_valid(wire_response(response)))
        del response["constraint_traceability"][1]["index"]
        self.assertFalse(validator.is_valid(wire_response(response)))
        for index in (None, -1, "index.html"):
            with self.subTest(index=index):
                response["constraint_traceability"][1]["index"] = index
                self.assertFalse(validator.is_valid(wire_response(response)))
        response["constraint_traceability"][1]["index"] = 0
        response["constraint_traceability"][0]["index"] = 0
        self.assertFalse(validator.is_valid(wire_response(response)))

    def test_json_schema_excludes_absent_context_and_framework_citations(self):
        response = analysis_fixture()
        validator = Draft202012Validator(analysis_response_schema(IntakeSource(original_request=PROMPT)).model_json_schema())
        response["requirement_traceability"][0]["sources"] = [{"source": "context", "quote": "Framework metadata"}]
        self.assertFalse(validator.is_valid(wire_response(response)))
        response["requirement_traceability"][0]["sources"] = [{"source": "request", "quote": PROMPT}]
        response["constraint_traceability"][0]["sources"] = [{"source": "request", "quote": PROMPT}]
        self.assertFalse(validator.is_valid(wire_response(response)))
        context = "Supplied source text"
        response = analysis_fixture(PROMPT, context)
        response["requirement_traceability"][0]["sources"] = [{"source": "context", "quote": context}]
        validator = Draft202012Validator(analysis_response_schema(IntakeSource(original_request=PROMPT, supporting_context=context)).model_json_schema())
        self.assertTrue(validator.is_valid(wire_response(response)))

    def test_archived_intake_failure_remains_invalid_and_is_not_repaired(self):
        raw = (ROOT / "tests/fixtures/intake_invalid_contract.json").read_text(encoding="utf-8")
        prompt = json.loads(raw)["task_spec"]["original_requirements"]
        validator = Draft202012Validator(analysis_response_schema(IntakeSource(original_request=prompt)).model_json_schema())
        self.assertFalse(validator.is_valid(json.loads(raw)))
        result = self.invoke(client=IntakeMockClient(raw=raw), prompt=prompt)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())
        self.assertEqual(Path(result["run_dir"], "intake_raw.txt").read_text(encoding="utf-8"), raw)
        self.assertTrue(Path(result["run_dir"], "intake_validation_error.json").exists())
        self.assertEqual(len(self.client.calls), 1)

    def test_archived_portfolio_undeclared_requirement_stays_rejected(self):
        raw = (ROOT / "tests/fixtures/intake_undeclared_requirement.json").read_text(encoding="utf-8")
        original = json.loads(raw)
        with self.assertRaisesRegex(ValueError, "Unknown requirement: NFR-responsiveness"):
            RequirementsAnalysis.model_validate(original)
        result = self.invoke(client=IntakeMockClient(raw=raw), prompt=original["task_spec"]["original_requirements"])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())
        self.assertEqual(Path(result["run_dir"], "intake_raw.txt").read_bytes(), raw.encode("utf-8"))
        self.assertEqual(len(self.client.calls), 1)

    def test_nested_contract_derives_exact_references_without_rewriting_ids(self):
        response = wire_response(analysis_fixture())
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        self.assertNotIn("tests", response["task_spec"])
        self.assertNotIn("requirement_traceability", response)
        self.assertTrue(Draft202012Validator(schema.model_json_schema()).is_valid(response))
        canonical = assemble_analysis(schema.model_validate(response))
        self.assertEqual({requirement.id for requirement in canonical.task_spec.requirements}, {"notice", "responsive", "accent"})
        for group in ("features", "nonfunctional_requirements"):
            for requirement in response["task_spec"][group]:
                self.assertNotIn("requirement_id", requirement["traceability"])
                for test in requirement["tests"]:
                    self.assertNotIn("requirement_id", test)
                    assembled = next(item for item in canonical.task_spec.tests if item.test_id == test["test_id"])
                    self.assertEqual(assembled.requirement_id, requirement["id"])
                    self.assertNotIn("required", test)
                    self.assertEqual(assembled.required, requirement["required"])
        self.assertEqual({trace.requirement_id for trace in canonical.requirement_traceability}, {"notice", "responsive", "accent"})

    def test_runtime_provenance_is_required_even_without_technology_constraints(self):
        response = wire_response(analysis_fixture())
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        validator = Draft202012Validator(schema.model_json_schema())
        self.assertTrue(validator.is_valid(response))
        canonical = assemble_analysis(schema.model_validate(response))
        self.assertEqual([(trace.target, trace.index) for trace in canonical.constraint_traceability], [("runtime", None)])
        for mode in ("missing", "null", "empty", "flat-empty-array"):
            with self.subTest(mode=mode):
                invalid = deepcopy(response)
                if mode == "missing": invalid["task_spec"]["runtime"].pop("traceability")
                elif mode == "null": invalid["task_spec"]["runtime"]["traceability"] = None
                elif mode == "empty": invalid["task_spec"]["runtime"]["traceability"] = {}
                else: invalid["constraint_traceability"] = []
                self.assertFalse(validator.is_valid(invalid))
                with self.assertRaises(ValueError): schema.model_validate(invalid)

    def test_multiple_technology_constraints_derive_exact_trace_indexes(self):
        response = wire_response(analysis_fixture())
        provenance = {"origin": "framework_default", "rationale": "Default offline runtime constraint", "sources": []}
        response["task_spec"]["technology_constraints"] = [
            {"value": "Offline document", "traceability": provenance},
            {"value": "Inline styles", "traceability": provenance}]
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        canonical = assemble_analysis(schema.model_validate(response))
        self.assertEqual(canonical.task_spec.technology_constraints, ["Offline document", "Inline styles"])
        self.assertEqual([(trace.target, trace.index) for trace in canonical.constraint_traceability],
                         [("runtime", None), ("technology_constraint", 0), ("technology_constraint", 1)])
        for mode in ("missing", "index", "target", "bare-string"):
            with self.subTest(mode=mode):
                invalid = deepcopy(response)
                if mode == "missing": invalid["task_spec"]["technology_constraints"][0].pop("traceability")
                elif mode == "bare-string": invalid["task_spec"]["technology_constraints"][0] = "Offline document"
                else: invalid["task_spec"]["technology_constraints"][0]["traceability"][mode] = 0 if mode == "index" else "runtime"
                self.assertFalse(Draft202012Validator(schema.model_json_schema()).is_valid(invalid))

    def test_schema_enforces_recommendation_optionality_and_verification_authority(self):
        response = wire_response(analysis_fixture())
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        validator = Draft202012Validator(schema.model_json_schema())
        for mode in ("mandatory-recommendation", "optional-blocker", "test-required", "disabled-verification", "disabled-accessibility"):
            with self.subTest(mode=mode):
                invalid = deepcopy(response)
                if mode == "mandatory-recommendation": invalid["task_spec"]["features"][1]["required"] = True
                elif mode == "optional-blocker":
                    invalid["task_spec"]["features"][1]["missing_information"] = [{"description": "Optional preference",
                        "question": "Which color", "blocks_generation": True, "sources": []}]
                elif mode == "test-required": invalid["task_spec"]["features"][1]["tests"][0]["required"] = True
                else: invalid["task_spec"]["verification"] = {"enabled" if mode == "disabled-verification" else "accessibility": False}
                self.assertFalse(validator.is_valid(invalid))
                with self.assertRaises(ValueError): schema.model_validate(invalid)

    def test_schema_encodes_browser_dsl_value_rules_and_nonblank_criteria(self):
        response = wire_response(analysis_fixture())
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        validator = Draft202012Validator(schema.model_json_schema())
        for mode in ("fill-no-value", "unsupported-key", "count-string", "count-negative", "text-null", "blank-name", "blank-criterion"):
            with self.subTest(mode=mode):
                invalid = deepcopy(response)
                requirement = invalid["task_spec"]["features"][0]
                test = requirement["tests"][0]
                if mode == "fill-no-value": test["actions"] = [{"action": "fill", "selector": "input"}]
                elif mode == "unsupported-key": test["actions"] = [{"action": "press", "selector": "input", "value": "Delete"}]
                elif mode == "blank-criterion": requirement["acceptance_criteria"] = ["   "]
                else:
                    assertion, value = {"count-string": ("count", "1"), "count-negative": ("count", -1),
                        "text-null": ("text", None), "blank-name": ("accessible_name", "   ")}[mode]
                    test["expected_results"] = [{"assertion": assertion, "selector": "body", "value": value}]
                self.assertFalse(validator.is_valid(invalid))
                with self.assertRaises(ValueError): schema.model_validate(invalid)

    def test_runtime_schema_encodes_conditional_entrypoint_and_url_requirements(self):
        response = wire_response(analysis_fixture())
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        validator = Draft202012Validator(schema.model_json_schema())
        for kind in ("single_html", "static_site", "web_app"):
            for value in (None, ""):
                with self.subTest(kind=kind, value=value):
                    invalid = deepcopy(response)
                    invalid["task_spec"]["runtime"].update(kind=kind, entrypoint=value)
                    self.assertFalse(validator.is_valid(invalid))
                    with self.assertRaises(ValueError): schema.model_validate(invalid)
        response["task_spec"]["runtime"].update(kind="web_app", entrypoint=None, base_url="http://localhost:8000")
        self.assertTrue(validator.is_valid(response))
        self.assertEqual(assemble_analysis(schema.model_validate(response)).task_spec.runtime.kind, "web_app")

    def test_wire_schema_forbids_independent_requirement_references(self):
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        validator = Draft202012Validator(schema.model_json_schema())
        for location in ("test", "trace", "task", "root"):
            with self.subTest(location=location):
                response = wire_response(analysis_fixture())
                requirement = response["task_spec"]["features"][0]
                if location == "test":
                    requirement["tests"][0]["requirement_id"] = "undeclared"
                elif location == "trace":
                    requirement["traceability"]["requirement_id"] = "undeclared"
                elif location == "task":
                    response["task_spec"]["tests"] = analysis_fixture()["task_spec"]["tests"]
                else:
                    response["requirement_traceability"] = analysis_fixture()["requirement_traceability"]
                self.assertFalse(validator.is_valid(response))
                with self.assertRaises(ValueError):
                    schema.model_validate(response)

    def test_wire_schema_enforces_manual_executable_alignment(self):
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        validator = Draft202012Validator(schema.model_json_schema())
        for mode in ("manual-test-browser-requirement", "browser-test-manual-requirement", "manual-actions", "missing-tests"):
            with self.subTest(mode=mode):
                response = wire_response(analysis_fixture())
                requirement = response["task_spec"]["features"][0]
                if mode == "manual-test-browser-requirement":
                    requirement["tests"] = [{"test_id": "review", "test_type": "manual"}]
                elif mode == "browser-test-manual-requirement":
                    requirement["verification"] = "manual"
                elif mode == "manual-actions":
                    requirement = response["task_spec"]["features"][1]
                    requirement["tests"][0]["actions"] = [{"action": "click", "selector": "button"}]
                else:
                    requirement["tests"] = []
                self.assertFalse(validator.is_valid(response))
                with self.assertRaises(ValueError):
                    schema.model_validate(response)

    def test_nested_contract_keeps_canonical_duplicate_id_validation(self):
        response = wire_response(analysis_fixture())
        response["task_spec"]["nonfunctional_requirements"][0]["id"] = "notice"
        schema = analysis_response_schema(IntakeSource(original_request=PROMPT))
        with self.assertRaisesRegex(ValueError, "requirement IDs must be unique"):
            assemble_analysis(schema.model_validate(response))

    def test_traceability_rejects_missing_duplicate_unknown_and_inexact_sources(self):
        for mode in ("missing", "duplicate", "unknown", "inexact", "absent-context"):
            with self.subTest(mode=mode):
                response = analysis_fixture()
                traces = response["requirement_traceability"]
                if mode == "missing":
                    traces.pop()
                elif mode == "duplicate":
                    traces.append(deepcopy(traces[0]))
                elif mode == "unknown":
                    traces[0]["requirement_id"] = "fabricated"
                else:
                    traces[0]["sources"] = [{"source": "context" if mode == "absent-context" else "request", "quote": "invented quotation"}]
                result = self.invoke(response, name=f"{mode}.json")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(self.output.exists())
                self.assertTrue(Path(result["run_dir"], "intake_validation_error.json").is_file())
                self.assertEqual(len(self.client.calls), 1)

    def test_optional_recommendations_cannot_be_promoted(self):
        response = analysis_fixture()
        response["task_spec"]["features"][1]["required"] = True
        response["task_spec"]["tests"][2]["required"] = True
        result = self.invoke(response)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Input should be", result["error"])

    def test_optional_tests_cannot_become_mandatory(self):
        response = analysis_fixture()
        response["task_spec"]["tests"][2]["required"] = True
        result = self.invoke(response)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Extra inputs are not permitted", result["error"])

    def test_optional_advice_cannot_become_a_generation_blocker(self):
        response = analysis_fixture()
        response["missing_information"] = [{"issue_id": "optional-color", "description": "No preferred accent color",
            "question": "Which optional accent do you prefer?", "blocks_generation": True,
            "requirement_ids": ["accent"], "sources": []}]
        result = self.invoke(response)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Input should be", result["error"])

    def test_each_requirement_needs_an_observable_test_or_manual_review(self):
        response = analysis_fixture()
        response["task_spec"]["tests"].pop()
        result = self.invoke(response)
        self.assertEqual(result["status"], "failed")
        self.assertIn("tests", result["error"])

    def test_invalid_constraints_issues_and_verification_settings_are_rejected(self):
        for mode in ("missing-constraint", "duplicate-constraint", "unknown-issue", "blank-criterion",
                     "disabled-verification", "disabled-accessibility", "manual-mismatch", "changed-context"):
            with self.subTest(mode=mode):
                response = analysis_fixture()
                if mode == "missing-constraint":
                    response["constraint_traceability"] = []
                elif mode == "duplicate-constraint":
                    response["constraint_traceability"] *= 2
                elif mode == "unknown-issue":
                    response["unresolved_ambiguities"] = [{"issue_id": "unknown", "description": "An unresolved choice",
                        "question": "What should happen?", "blocks_generation": False, "requirement_ids": ["invented"], "sources": []}]
                elif mode == "blank-criterion":
                    response["task_spec"]["features"][0]["acceptance_criteria"] = ["   "]
                elif mode.startswith("disabled-"):
                    response["task_spec"]["verification"] = {"enabled" if mode == "disabled-verification" else "accessibility": False}
                elif mode == "manual-mismatch":
                    response["task_spec"]["features"][0]["verification"] = "manual"
                else:
                    response["task_spec"]["supporting_context"] = "Invented source context"
                result = self.invoke(response, name=f"{mode}.json")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(self.output.exists())
                self.assertTrue(Path(result["run_dir"], "intake_validation_error.json").is_file())

    def test_structured_failures_are_retained_without_repair(self):
        valid = analysis_fixture()
        wrong_type = deepcopy(valid)
        wrong_type["task_spec"]["features"] = "not an array"
        changed_source = deepcopy(valid)
        changed_source["task_spec"]["original_requirements"] = "Rewritten original request that was not supplied"
        for index, raw in enumerate(("{malformed", json.dumps({"RequirementsAnalysis": valid}), json.dumps(wrong_type), json.dumps(changed_source))):
            with self.subTest(raw=raw[:30]):
                client = IntakeMockClient(raw=raw)
                result = self.invoke(client=client, name=f"invalid-{index}.json")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(self.output.exists())
                self.assertEqual(Path(result["run_dir"], "intake_raw.txt").read_text(), raw)
                self.assertTrue(Path(result["run_dir"], "intake_validation_error.json").exists())
                self.assertEqual(len(client.calls), 1)

    def test_length_terminated_analysis_is_never_accepted(self):
        result = self.invoke(client=IntakeMockClient(finish_reason="length"))
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())
        metrics = json.loads(Path(result["run_dir"], "intake_metrics.json").read_text())
        self.assertTrue(metrics["incomplete_generation"])
        self.assertEqual(metrics["finish_reason"], "length")
        self.assertTrue(Path(result["run_dir"], "intake_raw.txt").exists())

    def test_unsupported_schema_request_has_diagnostics_and_no_fallback(self):
        client = IntakeMockClient(error=StructuredOutputRequestError("Unsupported schema; no fallback",
            {"kind": "structured_output_request_rejected", "backend_response": {"error": "json_schema unsupported"}}))
        result = self.invoke(client=client)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(Path(result["run_dir"], "intake_structured_output_error.json").is_file())
        self.assertTrue(Path(result["run_dir"], "intake_metrics.json").is_file())
        self.assertFalse(self.output.exists())
        self.assertEqual(len(client.calls), 1)

    def test_constraints_are_traced_and_unsupported_runtime_is_not_substituted(self):
        prompt = PROMPT + " Use a Python backend."
        response = analysis_fixture(prompt)
        response["task_spec"]["runtime"] = {"kind": "full_stack", "needs_database": False}
        response["task_spec"]["technology_constraints"] = ["Use a Python backend"]
        source = {"source": "request", "quote": "Use a Python backend"}
        response["constraint_traceability"] = [
            {"target": "runtime", "index": None, "origin": "explicit", "rationale": "The user requested a backend", "sources": [source]},
            {"target": "technology_constraint", "index": 0, "origin": "explicit", "rationale": "The user named Python", "sources": [source]}]
        result = self.invoke(response, prompt=prompt)
        self.assertEqual(result["status"], "needs_review", result)
        self.assertFalse(result["ready_for_generation"])
        self.assertEqual(load_task(self.output).runtime.kind, "full_stack")
        record = json.loads(Path(result["analysis_file"]).read_text())
        self.assertFalse(record["runtime_supported"])
        self.assertIn("Unsupported runtime", record["runtime_error"])

    def test_manifest_reused_across_both_existing_workflows(self):
        intake = self.invoke(response=analysis_fixture(PROMPT, "Approved notice content"), context="Approved notice content")
        hashes = []
        for workflow in ("baseline", "collaborative"):
            with self.subTest(workflow=workflow):
                client = CandidateClient()
                result = build_graph(client, stub_verifier, workflow=workflow).invoke({
                    "task_path": str(self.output), "output_root": self.temp.name, "execution_mode": "mocked"})
                self.assertEqual(result["status"], "passed", result)
                hashes.append(result["requirement_hash"])
                self.assertEqual(result["requirements"], PROMPT)
                self.assertEqual(result["task_spec"]["supporting_context"], "Approved notice content")
        self.assertEqual(hashes, [intake["task_hash"], intake["task_hash"]])

    def test_generated_acceptance_tests_execute_in_the_existing_browser_verifier(self):
        self.invoke()
        result = build_graph(CandidateClient()).invoke({"task_path": str(self.output), "output_root": self.temp.name,
                                                       "execution_mode": "mocked-model-real-browser"})
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["verification"]["coverage"]["notice"]["status"], "passed")
        self.assertEqual(result["verification"]["coverage"]["accent"]["status"], "unverified")

    def test_existing_output_is_not_overwritten_or_reanalyzed(self):
        first = self.invoke()
        before = self.output.read_bytes()
        client = IntakeMockClient()
        result = self.invoke(client=client)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(client.calls, [])
        self.assertTrue(Path(first["analysis_file"]).is_file())

    def test_repeat_analysis_has_stable_source_and_task_hashes(self):
        first = self.invoke(name="first.json")
        second = self.invoke(name="second.json")
        self.assertEqual(first["source_hash"], second["source_hash"])
        self.assertEqual(first["task_hash"], second["task_hash"])
        self.assertNotEqual(first["run_dir"], second["run_dir"])

    def test_existing_manifest_hashes_are_unchanged_without_context(self):
        for name in ("eventhub", "todo", "catalog"):
            with self.subTest(task=name):
                task = load_task(ROOT / f"tasks/{name}.json")
                old_shape = task.model_dump(mode="json")
                self.assertNotIn("supporting_context", old_shape)
                old_hash = hashlib.sha256(json.dumps(old_shape, sort_keys=True).encode()).hexdigest()
                self.assertEqual(contract_hash(task), old_hash)

    def test_invalid_or_blank_input_never_invokes_a_model(self):
        for prompt in ("", " \r\n"):
            with self.subTest(prompt=prompt):
                result = self.invoke(prompt=prompt)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(self.client.calls, [])
                self.assertFalse(self.output.exists())

    def test_cli_intake_is_separate_from_generation_and_preserves_explicit_limit(self):
        from agentic_web.__main__ import main
        client = IntakeMockClient()
        output = Path(self.temp.name) / "cli-task.json"
        stdout = io.StringIO()
        with patch("sys.argv", ["agentic_web", "intake", "--prompt", PROMPT, "--output", str(output),
                "--model", "analyst-model", "--seed", "1025", "--max-tokens", "512"]), \
                patch("agentic_web.intake.intake_app", build_intake_graph(client)), \
                patch("agentic_web.__main__.app") as generation, redirect_stdout(stdout):
            self.assertEqual(main(), 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "analyzed")
        self.assertEqual(summary["output_file"], str(output))
        self.assertEqual(client.calls[0]["config"].planner_max_tokens, 512)
        self.assertEqual(client.calls[0]["config"].seed, 1025)
        generation.invoke.assert_not_called()

    def test_collaborative_prompt_flows_through_analysis_and_all_existing_agents(self):
        client = IntegratedIntakeClient()
        result = build_graph(client, workflow="collaborative").invoke({"prompt": PROMPT, "output_root": self.temp.name})
        self.assertEqual(result["status"], "passed", result)
        self.assertTrue(result["acceptance"]["accepted"])
        self.assertEqual(client.roles[0:2], ["requirements_analyst", "planner"])
        self.assertEqual(set(client.roles), {"requirements_analyst", "planner", "ux", "architect", "coordinator", "implementer", "qa"})
        self.assertEqual(client.roles.count("requirements_analyst"), 1)
        self.assertEqual(load_task(result["canonical_task_file"]).model_dump(mode='json'), result['task_spec'])
        self.assertFalse(Path(result['intake_run_dir'], 'task.json').exists())
        self.assertEqual(result["requirements"], PROMPT)
        self.assertEqual(result["requirement_hash"], result["analysis"]["task_hash"])
        self.assertEqual(json.loads(Path(result["run_dir"], "requirements_analysis.json").read_text()), result["analysis"])
        self.assertTrue(Path(result["intake_run_dir"], "ledger/requirements_analyst.json").is_file())

    def test_collaborative_original_requirements_input_also_uses_analysis(self):
        client = IntegratedIntakeClient()
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({"requirements": PROMPT, "output_root": self.temp.name})
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(client.roles[0], "requirements_analyst")
        self.assertEqual(result["task_spec"]["task_id"], "community-notice")

    def test_collaborative_analyze_only_stops_before_planning(self):
        client = IntegratedIntakeClient()
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({
            "prompt": PROMPT, "output_root": self.temp.name, "analyze_only": True})
        self.assertEqual(result["status"], "analyzed", result)
        self.assertTrue(result["ready_for_generation"])
        self.assertEqual(client.roles, ["requirements_analyst"])
        self.assertNotIn("plan", result)
        self.assertNotIn("acceptance", result)
        self.assertFalse(Path(result["run_dir"], "application").exists())

    def test_collaborative_blocking_information_stops_before_generation(self):
        response = analysis_fixture()
        response["missing_information"] = [{"issue_id": "source-missing", "description": "Required source content is absent",
            "question": "What content should appear?", "blocks_generation": True, "requirement_ids": ["notice"], "sources": []}]
        client = IntegratedIntakeClient(response)
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({"prompt": PROMPT, "output_root": self.temp.name})
        self.assertEqual(result["status"], "needs_review", result)
        self.assertFalse(result["ready_for_generation"])
        self.assertEqual(client.roles, ["requirements_analyst"])
        self.assertNotIn("plan", result)
        self.assertNotIn("acceptance", result)

    def test_collaborative_analysis_failure_never_invokes_candidate_agents(self):
        client = IntegratedIntakeClient(raw="{malformed")
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({"prompt": PROMPT, "output_root": self.temp.name})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(client.roles, ["requirements_analyst"])
        self.assertNotIn("task_spec", result)
        self.assertTrue(Path(result["intake_run_dir"], "intake_validation_error.json").exists())

    def test_collaborative_manifest_path_bypasses_analysis(self):
        client = CollaborationMockClient()
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({
            "task_path": str(ROOT / "tasks/todo.json"), "output_root": self.temp.name})
        self.assertEqual(result["status"], "passed", result)
        self.assertNotIn("requirements_analyst", client.roles)
        self.assertNotIn("analysis", result)

    def test_collaborative_prompt_conflicts_fail_before_inference(self):
        client = IntegratedIntakeClient()
        result = build_graph(client, stub_verifier, workflow="collaborative").invoke({
            "prompt": PROMPT, "requirements": "A different natural-language request", "output_root": self.temp.name})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(client.roles, [])

    def test_collaborative_analysis_uses_resource_admission_and_preserves_external_models(self):
        from agentic_web.resource_scheduler import ResourceScheduler, select_placements
        from agentic_web.resource_schema import ResourceSettings
        from resource_mock_backend import SimulatedBackend
        backend = SimulatedBackend()
        settings = ResourceSettings(context_length=4096, gpu_safety_bytes=0, ram_safety_bytes=0, estimate_padding=0,
            prefetch_all_if_gpu_fit=False, resident_devices={"manual-resident": "0"})
        placement = select_placements([backend.estimate("light", settings, 1)], backend.hardware(), backend.capabilities(), settings)[0]["light"]
        backend.load(placement, "manual-resident")
        scheduler = ResourceScheduler(backend, backend.hardware)
        class AdmittedClient(IntegratedIntakeClient):
            def complete(self, system, user, config, *, planner=False, response_schema=None):
                identifier = config.planner_model if planner else config.model
                if identifier not in backend.instances:
                    raise AssertionError("Inference bypassed placement admission")
                return super().complete(system, user, config, planner=planner, response_schema=response_schema)
        client = AdmittedClient()
        result = build_graph(client, stub_verifier, workflow="collaborative", resource_scheduler=scheduler).invoke({
            "prompt": PROMPT, "output_root": self.temp.name, "model_config": ModelConfig(model="tiny").model_dump(),
            "resource_config": settings.model_dump()})
        self.assertEqual(result["status"], "passed", result)
        metrics = json.loads(Path(result["intake_run_dir"], "intake_metrics.json").read_text())
        self.assertEqual(metrics["assigned_device"], "0")
        self.assertEqual(metrics["requested_model"], "tiny")
        self.assertEqual(set(backend.instances), {"manual-resident"})
        self.assertNotIn("manual-resident", backend.unloads)


if __name__ == "__main__":
    unittest.main()
