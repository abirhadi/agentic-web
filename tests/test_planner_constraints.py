"""Regression for a successful inference that invented immutable constraints."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema_rs import Draft202012Validator
from pydantic import ValidationError
from agentic_web.client import LMStudioClient, ModelConfig
from agentic_web.reasoning import InferenceError
from agentic_web.schema import implementation_plan_schema, load_task
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from inference_mock import sdk
from mock_client import ROOT
from test_framework import stub_verifier


INVENTED = ['single_html', 'No backend required', 'No database required', 'No authentication required']


def plan(constraints):
    return {'architecture': 'Single document', 'components': ['Main content'], 'data_models': [],
            'interactions': [], 'technology_constraints': constraints, 'implementation_steps': ['Build'],
            'risks': [], 'assumptions': [], 'requirement_ids': []}


class PlannerConstraintTests(unittest.TestCase):
    def task(self, constraints):
        task = load_task(ROOT / 'tasks/todo.json')
        task.technology_constraints = list(constraints)
        return task

    def test_empty_contract_rejects_archived_invented_constraints(self):
        schema = implementation_plan_schema(self.task([]))
        wire = schema.model_json_schema()
        self.assertEqual(wire['properties']['technology_constraints']['const'], [])
        grammar = Draft202012Validator(wire)
        self.assertTrue(grammar.is_valid(plan([])))
        self.assertFalse(grammar.is_valid(plan(INVENTED)))
        with self.assertRaisesRegex(ValidationError, 'Planner changed technology constraints'):
            schema.model_validate(plan(INVENTED))

    def test_nonempty_contract_preserves_order_duplicates_and_exact_text(self):
        expected = ['Use local assets only', 'No server', 'No server']
        schema = implementation_plan_schema(self.task(expected))
        grammar = Draft202012Validator(schema.model_json_schema())
        self.assertEqual(schema.model_validate(plan(expected)).technology_constraints, expected)
        for changed in ([], expected[::-1], expected[:2], expected + ['extra'], ['use local assets only', 'No server', 'No server']):
            with self.subTest(changed=changed):
                self.assertFalse(grammar.is_valid(plan(changed)))
                with self.assertRaises(ValidationError):
                    schema.model_validate(plan(changed))

    def test_request_schema_captures_an_independent_manifest_snapshot(self):
        task = self.task(['Original'])
        first = implementation_plan_schema(task)
        task.technology_constraints.append('Later')
        second = implementation_plan_schema(task)
        self.assertEqual(first.model_validate(plan(['Original'])).technology_constraints, ['Original'])
        self.assertFalse(Draft202012Validator(second.model_json_schema()).is_valid(plan(['Original'])))

    def test_shared_client_sends_constant_and_rejects_backend_violation(self):
        schema = implementation_plan_schema(self.task([]))
        for constraints in ([], INVENTED):
            with self.subTest(constraints=constraints):
                provider = sdk(json.dumps(plan(constraints)))
                client = LMStudioClient()
                with patch.object(client, '_async_client', return_value=provider):
                    if constraints:
                        with self.assertRaises(InferenceError) as caught:
                            client.complete('System', 'User', ModelConfig(), planner=True, response_schema=schema)
                        self.assertEqual(caught.exception.diagnostics['error_classification'], 'schema_validation_error')
                        self.assertEqual(caught.exception.diagnostics['finish_reason'], 'stop')
                    else:
                        raw, metrics = client.complete('System', 'User', ModelConfig(), planner=True, response_schema=schema)
                        self.assertEqual(json.loads(raw)['technology_constraints'], [])
                        self.assertEqual(metrics['structured_validation'], 'passed')
                sent = provider.chat.completions.create.call_args.kwargs['response_format']['json_schema']['schema']
                self.assertEqual(sent, schema.model_json_schema())
                self.assertEqual(provider.chat.completions.create.call_count, 1)

    def test_both_workflows_preserve_contract_without_rewriting_output(self):
        for workflow in ('baseline', 'collaborative'):
            for invalid in (False, True):
                with self.subTest(workflow=workflow, invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                    class Client(CollaborationMockClient):
                        def complete(self, *args, planner=False, **kwargs):
                            raw, metrics = super().complete(*args, planner=planner, **kwargs)
                            if planner and invalid:
                                value = json.loads(raw)
                                value['technology_constraints'] = INVENTED
                                raw = json.dumps(value)
                            return raw, metrics
                    task = self.task([])
                    result = build_graph(Client(), stub_verifier, workflow=workflow).invoke({
                        'task_spec': task.model_dump(mode='json'), 'output_root': tmp})
                    root = Path(result['run_dir'])
                    request = json.loads((root / 'planner_request.json').read_text())
                    sent = request['response_format']['json_schema']['schema']
                    self.assertEqual(sent['properties']['technology_constraints']['const'], [])
                    self.assertEqual(json.loads(request['user'])['plan_schema'], sent)
                    self.assertEqual(json.loads((root / 'task.json').read_text())['technology_constraints'], [])
                    if invalid:
                        self.assertEqual(result['status'], 'failed')
                        self.assertFalse((root / 'plan.json').exists())
                        self.assertFalse((root / 'implementer_request.json').exists())
                        raw = json.loads((root / 'planner_raw.txt').read_text())
                        self.assertEqual(raw['technology_constraints'], INVENTED)
                        diagnostics = json.loads((root / 'planner_validation_error.json').read_text())
                        self.assertEqual(diagnostics['errors'][0]['loc'], ['technology_constraints'])
                    else:
                        self.assertEqual(result['status'], 'passed', result.get('error'))
                        self.assertEqual(result['plan']['technology_constraints'], [])


if __name__ == '__main__':
    unittest.main()
