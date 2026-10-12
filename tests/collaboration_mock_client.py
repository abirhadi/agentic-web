"""Phase 2 model substitute with independent role responses and real fixture HTML."""
import json
from copy import deepcopy
from mock_client import MockClient


class CollaborationMockClient(MockClient):
    def __init__(self, *, malformed=None, unavailable=None, transport_failure=None, qa_override=None, coordinator_override=None):
        super().__init__()
        self.malformed = malformed
        self.unavailable = unavailable
        self.transport_failure = transport_failure
        self.qa_override = qa_override
        self.coordinator_override = coordinator_override
        self.roles = []

    def models(self, config):
        return [config.model_for(role) for role in ("planner", "ux", "architect", "coordinator", "implementer", "qa")
                if config.model_for(role) != self.unavailable]

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        payload = json.loads(user)
        role = payload.get("role", "planner" if planner else "implementer")
        self.roles.append(role)
        if role == self.transport_failure:
            raise ConnectionError("Model unavailable during inference")
        if role in {"planner", "implementer"}:
            return super().complete(system, user, config, planner=planner, response_schema=response_schema)
        self.calls.append({"system": system, "payload": deepcopy(payload), "planner": planner, "config": config.model_dump(), "response_schema": response_schema})
        ids = [r["id"] for r in payload["task_spec"]["features"] + payload["task_spec"]["nonfunctional_requirements"]]
        required = [r["id"] for r in payload["task_spec"]["features"] + payload["task_spec"]["nonfunctional_requirements"] if r["required"]]
        target = required[0]
        if role == "ux":
            output = {"interaction_flows": ["User completes the requested flow"], "components_and_states": ["Named controls and empty states"],
                "accessibility_and_responsiveness": ["Preserve accessible names after changes"], "error_and_empty_states": ["Show empty states"],
                "risks_and_ambiguities": [], "recommendations": [{"requirement_ids": [target], "recommendation": "Use an explicit submit interaction", "rationale": "Clear user intent"}]}
        elif role == "architect":
            output = {"implementation_structure": ["Offline document"], "components": ["Main content"],
                "state_and_data_flow": ["Keep local records and render changes"], "event_and_data_contracts": ["Bind tested interactions"],
                "runtime_feasibility": "Supported by single_html", "risks_dependencies_and_ambiguities": [],
                "recommendations": [{"requirement_ids": [target], "recommendation": "Prefer delegated event handling", "rationale": "Supports dynamic controls"}]}
        elif role == "coordinator":
            assert "ux_review" in payload and "architecture_review" in payload
            output = {"architecture_and_responsibilities": ["Use the approved offline document structure"],
                "state_data_and_interaction_contracts": ["Preserve tested DOM and data behavior"],
                "ux_accessibility_constraints": ["Label all dynamic controls"], "implementation_instructions": ["Implement the mandatory task without changing test hooks"],
                "agreements": ["Keep dynamic controls testable"], "conflict_resolutions": [{"requirement_ids": [target],
                    "ux_position": "Explicit submit action", "architecture_position": "Delegated handling", "resolution": "Delegate the explicit submit event",
                    "basis": "Both preserve the requested interaction and acceptance contract"}], "unresolved_issues": [],
                "requirement_traceability": {identifier: ["Preserve the manifest acceptance criteria"] for identifier in required}}
            if self.coordinator_override:
                output.update(self.coordinator_override)
        else:
            report = payload["verification"]
            failing = next((test for test in report.get("tests", []) if test["status"] == "failed"), None)
            output = {"summary": "Analysis of actual deterministic results", "findings": []}
            if failing:
                output["findings"] = [{"classification": "hypothesis", "diagnostic_role": "independent", "category": "interface_contract",
                    "description": "A failing selector/assertion suggests an implementation-contract defect",
                    "requirement_ids": [failing["requirement_id"]], "test_ids": [failing["test_id"]],
                    "evidence": [{"key": f"tests.{failing['test_id']}.status", "quote": "failed"}],
                    "test_outcomes": {failing["test_id"]: "failed"}, "likely_cause": "Inspect the tested DOM binding",
                    "corrective_action": "Review the reported assertion and preserve exact testability hooks"}]
            if self.qa_override:
                output.update(self.qa_override)
        raw = "{malformed" if role == self.malformed else json.dumps(output)
        return raw, {"model": "mock:" + config.model, "seed": config.seed + 1, "finish_reason": "stop", "elapsed_seconds": 0,
                     "usage": {"total_tokens": 100}, "mocked": True}
