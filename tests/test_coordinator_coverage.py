"""The decoder contract must enforce the same coverage as the workflow guard."""
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from jsonschema_rs import Draft202012Validator
from pydantic import ValidationError
from agentic_web.collaboration_schema import grounded_coordination_schema
from agentic_web.client import LMStudioClient, ModelConfig
from agentic_web.reasoning import InferenceError
from agentic_web.schema import load_task
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from inference_mock import sdk
from mock_client import ROOT
from test_framework import stub_verifier


def proposal(trace):
    return dict(architecture_and_responsibilities=['Approved structure'],
                state_data_and_interaction_contracts=[], ux_accessibility_constraints=[],
                implementation_instructions=['Implement all requirements'], agreements=[],
                conflict_resolutions=[], unresolved_issues=[], requirement_traceability=trace)


class CoordinatorCoverageTests(unittest.TestCase):
    def test_manifest_keys_and_nonblank_instructions_in_schema_and_validator(self):
        for name in ('todo', 'eventhub', 'catalog'):
            task = load_task(ROOT / f'tasks/{name}.json')
            schema = grounded_coordination_schema(task)
            grammar = Draft202012Validator(schema.model_json_schema())
            trace = {r.id: ['Implement observable outcome'] for r in task.requirements if r.required}
            valid = proposal(trace)
            self.assertTrue(grammar.is_valid(valid))
            self.assertEqual(schema.model_validate(valid).model_dump(), valid)
            first = next(iter(trace))
            variants = [dict(trace, invented=['Extra'])]
            missing = deepcopy(trace)
            del missing[first]
            variants.append(missing)
            variants.extend(dict(trace, **{first: value}) for value in ([], [''], ['  \n']))
            for invalid in variants:
                with self.subTest(task=name, invalid=invalid):
                    self.assertFalse(grammar.is_valid(proposal(invalid)))
                    with self.assertRaises(ValidationError):
                        schema.model_validate(proposal(invalid))

    def test_optional_keys_remain_optional_and_snapshot_is_independent(self):
        task = load_task(ROOT / 'tasks/todo.json')
        task.features[0].required = False
        schema = grounded_coordination_schema(task)
        optional = task.features[0].id
        trace = {r.id: ['Implement'] for r in task.requirements if r.required}
        task.features[0].required = True
        for value in (trace, dict(trace, **{optional: ['Optional guidance']})):
            self.assertTrue(Draft202012Validator(schema.model_json_schema()).is_valid(proposal(value)))
            schema.model_validate(proposal(value))

    def test_client_rejects_missing_key_even_when_backend_stops_normally(self):
        schema = grounded_coordination_schema(load_task(ROOT / 'tasks/todo.json'))
        provider = sdk(json.dumps(proposal({})))
        client = LMStudioClient()
        with patch.object(client, '_async_client', return_value=provider):
            with self.assertRaises(InferenceError) as caught:
                client.complete('System', 'User', ModelConfig(), response_schema=schema)
        self.assertEqual(caught.exception.diagnostics['error_classification'], 'schema_validation_error')
        self.assertEqual(caught.exception.diagnostics['finish_reason'], 'stop')
        sent = provider.chat.completions.create.call_args.kwargs['response_format']['json_schema']['schema']
        self.assertEqual(sent, schema.model_json_schema())
        self.assertEqual(provider.chat.completions.create.call_count, 1)

    def test_omitted_requirement_preserves_raw_and_specific_diagnostics(self):
        task = load_task(ROOT / 'tasks/todo.json')
        trace = {r.id: ['Implement'] for r in task.requirements if r.required}
        omitted = next(iter(trace))
        del trace[omitted]
        client = CollaborationMockClient(coordinator_override={'requirement_traceability': trace})
        with tempfile.TemporaryDirectory() as directory:
            result = build_graph(client, stub_verifier, workflow='collaborative').invoke({
                'task_spec': task.model_dump(mode='json'), 'output_root': directory})
            self.assertEqual(result['status'], 'failed')
            self.assertNotIn('implementer', client.roles)
            root = Path(result['run_dir'])
            raw = json.loads((root / 'coordinator_raw.txt').read_text())
            self.assertEqual(raw['requirement_traceability'], trace)
            diagnostic = json.loads((root / 'coordinator_validation_error.json').read_text())
            self.assertIn(omitted, diagnostic['errors'][0]['msg'])
            request = json.loads((root / 'coordinator_request.json').read_text())
            sent = request['response_format']['json_schema']['schema']
            self.assertIn(omitted, sent['properties']['requirement_traceability']['required'])
            self.assertEqual(sent, json.loads(request['user'])['output_schema'])


if __name__ == '__main__':
    unittest.main()
