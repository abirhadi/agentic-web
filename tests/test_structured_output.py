"""Schema requests and fail-closed validation; no live inference or JSON repair."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import APIStatusError
from pydantic import RootModel
from inference_mock import sdk as stream_sdk
from agentic_web.reasoning import InferenceError
from agentic_web.client import LMStudioClient, ModelConfig, StructuredOutputRequestError, json_schema_response_format
from agentic_web.collaboration_schema import UXReview, ArchitectureReview, CoordinationProposal, QAReview
from agentic_web.schema import ImplementationPlan
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from mock_client import ROOT
from test_framework import stub_verifier


SCHEMAS = {"planner": ImplementationPlan, "ux": UXReview, "architect": ArchitectureReview,
           "coordinator": CoordinationProposal, "qa": QAReview}


class StructuredTransportTests(unittest.TestCase):
    def sdk(self, raw="{}"):
        return stream_sdk(raw, usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300})

    def test_existing_schemas_are_sent_verbatim_without_data_wrappers(self):
        for role, schema in SCHEMAS.items():
            with self.subTest(role=role):
                sdk = self.sdk()
                client = LMStudioClient()
                config, planner = ModelConfig().for_role(role)
                with patch.object(client, "_async_client", return_value=sdk):
                    with self.assertRaises(InferenceError) as caught:
                        client.complete("System", "User", config, planner=planner, response_schema=schema)
                    metrics = caught.exception.diagnostics
                    self.assertEqual(metrics['error_classification'], 'schema_validation_error')
                request = sdk.chat.completions.create.call_args.kwargs
                response_format = request["response_format"]
                self.assertEqual(response_format["type"], "json_schema")
                self.assertIs(response_format["json_schema"]["strict"], True)
                self.assertEqual(response_format["json_schema"]["schema"], schema.model_json_schema())
                self.assertNotIn(schema.__name__, response_format["json_schema"]["schema"]["properties"])
                self.assertGreater(request["max_tokens"], 0)
                self.assertEqual(metrics["response_schema"], schema.__name__)
                self.assertEqual(metrics["usage"]["total_tokens"], 300)
                self.assertEqual(metrics["finish_reason"], "stop")

    def test_actual_array_root_is_preserved(self):
        class Flags(RootModel[list[str]]):
            pass
        self.assertEqual(json_schema_response_format(Flags)["json_schema"]["schema"]["type"], "array")
        self.assertNotIn("properties", json_schema_response_format(Flags)["json_schema"]["schema"])

    def test_valid_json_response_is_not_rewritten(self):
        raw = json.dumps({"interaction_flows": [], "components_and_states": [], "accessibility_and_responsiveness": [],
                          "error_and_empty_states": [], "risks_and_ambiguities": [], "recommendations": []})
        sdk = self.sdk(raw)
        client = LMStudioClient()
        with patch.object(client, "_async_client", return_value=sdk):
            result, metrics = client.complete("System", "User", ModelConfig(max_tokens=777), response_schema=UXReview)
        self.assertEqual(result, raw)
        self.assertIsInstance(UXReview.model_validate_json(result), UXReview)
        self.assertEqual(sdk.chat.completions.create.call_args.kwargs["max_tokens"], 777)
        self.assertEqual(metrics["model"], "resolved-model")

    def test_html_implementation_does_not_request_json(self):
        sdk = self.sdk("<!DOCTYPE html><html></html>")
        client = LMStudioClient()
        with patch.object(client, "_async_client", return_value=sdk):
            raw, metrics = client.complete("System", "User", ModelConfig())
        self.assertNotIn("response_format", sdk.chat.completions.create.call_args.kwargs)
        self.assertIsNone(metrics["response_schema"])
        self.assertTrue(raw.startswith("<!DOCTYPE html>"))

    def test_unsupported_backend_format_fails_once_without_fallback(self):
        for status in (400, 422, 500, 501):
            with self.subTest(status=status):
                body = {"error": "response_format json_schema unsupported by this model/backend"}
                response = httpx.Response(status, request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions"), json=body)
                error = APIStatusError(body["error"], response=response, body=body)
                sdk = self.sdk()
                sdk.chat.completions.create.side_effect = error
                client = LMStudioClient()
                with patch.object(client, "_async_client", return_value=sdk), self.assertRaises(StructuredOutputRequestError) as caught:
                    client.complete("System", "User", ModelConfig(model="requested-model"), response_schema=UXReview)
                self.assertIn("No unconstrained fallback or retry", str(caught.exception))
                self.assertEqual(caught.exception.diagnostics["backend_response"], body)
                self.assertEqual(caught.exception.diagnostics["requested_model"], "requested-model")
                self.assertEqual(caught.exception.diagnostics["status_code"], status)
                self.assertEqual(sdk.chat.completions.create.call_count, 1)
                self.assertIn("response_format", sdk.chat.completions.create.call_args.kwargs)

    def test_unsupported_sdk_parameter_is_explicit(self):
        sdk = self.sdk()
        sdk.chat.completions.create.side_effect = TypeError("create() got an unexpected keyword argument 'response_format'")
        client = LMStudioClient()
        with patch.object(client, "_async_client", return_value=sdk), self.assertRaises(StructuredOutputRequestError):
            client.complete("System", "User", ModelConfig(), response_schema=UXReview)
        self.assertEqual(sdk.chat.completions.create.call_count, 1)

    def test_unrelated_backend_failure_retains_original_error(self):
        response = httpx.Response(400, request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions"))
        error = APIStatusError("Model not found", response=response, body={"error": "Model not found"})
        sdk = self.sdk()
        sdk.chat.completions.create.side_effect = error
        client = LMStudioClient()
        with patch.object(client, "_async_client", return_value=sdk), self.assertRaises(InferenceError) as caught:
            client.complete("System", "User", ModelConfig(), response_schema=UXReview)
        self.assertEqual(caught.exception.diagnostics["error_classification"], "model_unavailable")


class StructuredWorkflowTests(unittest.TestCase):
    def invoke(self, client, directory, workflow="collaborative"):
        return build_graph(client, stub_verifier, workflow=workflow).invoke({
            "task_path": str(ROOT / "tasks/todo.json"), "output_root": directory, "execution_mode": "mocked"})

    def test_all_structured_agents_receive_their_pydantic_schema(self):
        for workflow in ("baseline", "collaborative"):
            with self.subTest(workflow=workflow), tempfile.TemporaryDirectory() as directory:
                client = CollaborationMockClient()
                result = self.invoke(client, directory, workflow)
                self.assertEqual(result["status"], "passed")
                for call in client.calls:
                    role = call["payload"].get("role", "planner" if call["planner"] else "implementer")
                    if role in {"qa", "planner", "coordinator", "ux", "architect"}:
                        self.assertTrue(issubclass(call["response_schema"], SCHEMAS[role]))
                    else:
                        self.assertIs(call["response_schema"], SCHEMAS.get(role))
                    request = json.loads((Path(result["run_dir"]) / f"{role}_request.json").read_text())
                    if role != "implementer":
                        self.assertEqual(request["response_format"], json_schema_response_format(call["response_schema"]))
                    else:
                        self.assertNotIn("response_format", request)

    def test_malformed_bullets_and_schema_violations_fail_without_repair(self):
        for kind in ("bullets", "violation"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                class BadUX(CollaborationMockClient):
                    def complete(self, system, user, config, *, planner=False, response_schema=None):
                        raw, metrics = super().complete(system, user, config, planner=planner, response_schema=response_schema)
                        if json.loads(user).get("role") == "ux":
                            if kind == "bullets":
                                raw = '{"interaction_flows": [\n- unquoted Markdown bullet\n]}'
                            else:
                                value = json.loads(raw)
                                value["interaction_flows"] = "must be an array"
                                raw = json.dumps(value)
                            self.bad_raw = raw
                        return raw, metrics
                client = BadUX()
                result = self.invoke(client, directory)
                root = Path(result["run_dir"])
                self.assertEqual(result["status"], "failed")
                self.assertIn("ValidationError", result["ux_error"])
                self.assertEqual((root / "ux_raw.txt").read_text(), client.bad_raw)
                metrics = json.loads((root / "ux_metrics.json").read_text())
                self.assertEqual(metrics["finish_reason"], "stop")
                self.assertIsNone(metrics["max_tokens"])
                diagnostics = json.loads((root / "ux_validation_error.json").read_text())
                self.assertEqual(diagnostics["response_schema"], "UXReview")
                self.assertTrue(diagnostics["errors"])
                self.assertFalse((root / "ux_review.json").exists())
                self.assertEqual(client.roles.count("ux"), 1)
                self.assertNotIn("implementer", client.roles)

    def test_wrapped_roots_are_rejected_for_all_structured_roles(self):
        for target, schema in SCHEMAS.items():
            with self.subTest(role=target), tempfile.TemporaryDirectory() as directory:
                class Wrapped(CollaborationMockClient):
                    def complete(self, system, user, config, *, planner=False, response_schema=None):
                        raw, metrics = super().complete(system, user, config, planner=planner, response_schema=response_schema)
                        role = json.loads(user).get("role", "planner" if planner else "implementer")
                        if role == target:
                            raw = json.dumps({schema.__name__: json.loads(raw)})
                        return raw, metrics
                client = Wrapped()
                result = self.invoke(client, directory)
                root = Path(result["run_dir"])
                diagnostics = json.loads((root / f"{target}_validation_error.json").read_text())
                self.assertIn("extra_forbidden", [error["type"] for error in diagnostics["errors"]])
                self.assertEqual(client.roles.count(target), 1)
                if target == "qa":
                    self.assertEqual(result["qa_status"], "failed")
                    self.assertTrue(result["acceptance"]["accepted"])
                else:
                    self.assertEqual(result["status"], "failed")

    def test_unsupported_format_diagnostics_persist_for_planner_and_ux(self):
        for target in ("planner", "ux"):
            with self.subTest(role=target), tempfile.TemporaryDirectory() as directory:
                class Unsupported(CollaborationMockClient):
                    def complete(self, system, user, config, *, planner=False, response_schema=None):
                        role = json.loads(user).get("role", "planner" if planner else "implementer")
                        if role == target:
                            self.roles.append(role)
                            raise StructuredOutputRequestError("Backend does not support json_schema; no fallback",
                                {"kind": "structured_output_request_rejected", "response_schema": response_schema.__name__,
                                 "backend_response": {"error": "Unsupported response_format"}})
                        return super().complete(system, user, config, planner=planner, response_schema=response_schema)
                client = Unsupported()
                result = self.invoke(client, directory)
                self.assertEqual(result["status"], "failed")
                root = Path(result["run_dir"])
                self.assertTrue((root / f"{target}_structured_output_error.json").exists())
                self.assertTrue((root / f"{target}_metrics.json").exists())
                self.assertEqual(client.roles.count(target), 1)
                self.assertNotIn("implementer", client.roles)


if __name__ == "__main__":
    unittest.main()
