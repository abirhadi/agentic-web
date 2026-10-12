from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agentic_web.intake import build_intake_graph
from agentic_web.schema import TaskSpec, load_task
from agentic_web.decomposition import load_decomposition
from test_intake import analysis_fixture, wire_response, PROMPT
from test_decomposition import tasks_for


def finding(kind='explicit_omission', capability=None):
    return {'artifact': 'task_spec', 'kind': kind, 'capability': capability,
            'problem': 'Requested content is missing', 'impact': 'Cannot cover the requested content',
            'evidence': [{'source': 'request', 'quote': PROMPT}], 'requirement_ids': []}


class HandoffClient:
    def __init__(self, outcomes=None, drafts=None):
        self.outcomes = outcomes or ['accepted']
        self.drafts = drafts or [analysis_fixture()]
        self.calls = []
        self.analyst_calls = 0
        self.decomposer_calls = 0

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        payload = json.loads(user)
        self.calls.append((response_schema.__name__, payload, config))
        if response_schema.__name__ == 'DecomposerHandoff':
            outcome = self.outcomes[min(self.decomposer_calls, len(self.outcomes) - 1)]
            self.decomposer_calls += 1
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, dict):
                value = outcome
            elif outcome == 'accepted':
                value = {'outcome': 'accepted', 'findings': [],
                         'decomposition': tasks_for(TaskSpec.model_validate(payload['task_spec']))}
            else:
                value = {'outcome': outcome, 'decomposition': None, 'findings': [finding()]}
        else:
            value = wire_response(self.drafts[min(self.analyst_calls, len(self.drafts) - 1)])
            self.analyst_calls += 1
        return json.dumps(value), {'finish_reason': 'stop', 'usage': {'total_tokens': 10}}


class HandoffTests(unittest.TestCase):
    def duplicate_client(self, repeat=False):
        class DuplicateAnalyst(HandoffClient):
            def complete(self, system, user, config, **kwargs):
                raw, metrics = super().complete(system, user, config, **kwargs)
                if kwargs['response_schema'].__name__ != 'DecomposerHandoff' and (repeat or self.analyst_calls == 1):
                    value = json.loads(raw)
                    duplicate = deepcopy(value['task_spec']['features'][0])
                    duplicate['id'] = 'duplicate_outcome'
                    for test in duplicate['tests']:
                        test['test_id'] = 'duplicate_' + test['test_id']
                    value['task_spec']['nonfunctional_requirements'].append(duplicate)
                    raw = json.dumps(value)
                return raw, metrics
        return DuplicateAnalyst()

    def test_invalid_analyst_draft_is_corrected_before_decomposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.duplicate_client()
            result = self.run_graph(Path(tmp), client)
            self.assertIsNone(result.get('error'), result)
            self.assertEqual(result['handoff_status'], 'accepted')
            self.assertEqual((client.analyst_calls, client.decomposer_calls), (2, 1))
            correction = client.calls[1][1]
            self.assertIn('Duplicate requirement outcomes', correction['validation_errors'][0]['msg'])
            self.assertIn('duplicate_outcome', correction['invalid_draft'])
            for key in ('original_request', 'supporting_context', 'response_fields', 'task_fields'):
                self.assertEqual(correction[key], client.calls[0][1][key])
            root = Path(result['run_dir']) / 'handoff'
            self.assertTrue((root / 'round_000/analysis/error.json').exists())
            self.assertTrue((root / 'round_000/analysis/raw.txt').exists())
            self.assertTrue((root / 'round_001/analysis/analysis.json').exists())
            self.assertFalse((root / 'round_000/decompose').exists())
            self.assertNotIn('validation_feedback', result['handoff_work'])
            load_decomposition(Path(tmp) / 'task.tasks.json', load_task(Path(tmp) / 'task.json'))

    def test_invalid_analyst_corrections_stop_on_no_progress_or_budget(self):
        for budget, expected, calls in [(0, 'exhausted', 1), (2, 'no_progress', 2)]:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as tmp:
                client = self.duplicate_client(repeat=True)
                result = self.run_graph(Path(tmp), client, max_revisions=budget)
                self.assertEqual(result['handoff_status'], expected)
                self.assertEqual((client.analyst_calls, client.decomposer_calls), (calls, 0))
                self.assertFalse(result['ready_for_generation'])
                self.assertFalse((Path(tmp) / 'task.json').exists())

    def test_analyst_transport_and_truncation_fail_without_correction(self):
        from agentic_web.reasoning import InferenceError
        for failure in (ConnectionError('offline'), InferenceError('Incomplete', {
                'error_classification': 'output_budget_exhausted', 'finish_reason': 'length'})):
            class BrokenAnalyst(HandoffClient):
                def complete(self, *args, **kwargs):
                    self.analyst_calls += 1
                    raise failure
            with self.subTest(failure=str(failure)), tempfile.TemporaryDirectory() as tmp:
                client = BrokenAnalyst()
                result = self.run_graph(Path(tmp), client)
                self.assertEqual(result['handoff_status'], 'upstream_failed')
                self.assertEqual(client.analyst_calls, 1)
                self.assertFalse((Path(tmp) / 'task.json').exists())

    def test_schema_rejection_is_corrected_through_existing_revision_node(self):
        from agentic_web.reasoning import InferenceError
        class InvalidSchemaAnalyst(HandoffClient):
            def complete(self, system, user, config, **kwargs):
                raw, metrics = super().complete(system, user, config, **kwargs)
                if self.analyst_calls == 1:
                    raise InferenceError('Final JSON violates the response schema', {
                        'error_classification': 'schema_validation_error', 'final_content': '{"task_spec": {}}',
                        'finish_reason': 'stop', 'usage': {'total_tokens': 10},
                        'validation_errors': [{'loc': ['task_spec', 'features'], 'msg': 'Field required'}]})
                return raw, metrics
        with tempfile.TemporaryDirectory() as tmp:
            client = InvalidSchemaAnalyst()
            graph = build_intake_graph(client)
            events = list(graph.stream({'prompt': PROMPT, 'output_file': str(Path(tmp) / 'task.json'),
                'handoff_policy': {'mode': 'collaborative'}}, stream_mode='updates'))
            self.assertEqual([next(iter(event)) for event in events], [
                'requirements_prepare', 'requirements_analyst', 'requirements_revision',
                'handoff_decomposer', 'requirements_finalize'])
            self.assertEqual(client.calls[1][1]['invalid_draft'], '{"task_spec": {}}')
            self.assertEqual((client.analyst_calls, client.decomposer_calls), (2, 1))
            self.assertTrue((Path(tmp) / 'task.json').exists())

    def test_generation_schema_enforces_outcome_contract_without_wrapper(self):
        from jsonschema_rs import Draft202012Validator
        from pydantic import ValidationError
        from agentic_web.handoff import handoff_schema
        from agentic_web.intake_schema import IntakeSource
        draft = analysis_fixture()
        task = TaskSpec.model_validate(draft['task_spec'])
        schema = handoff_schema(task, IntakeSource(original_request=PROMPT), json.dumps(draft))
        validator = Draft202012Validator(schema.model_json_schema())
        graph = tasks_for(task)
        valid = [dict(outcome='accepted', decomposition=graph, findings=[]),
                 dict(outcome='revision_requested', findings=[finding()]),
                 dict(outcome='unresolved', decomposition=graph, findings=[finding()])]
        for value in valid:
            self.assertTrue(validator.is_valid(value))
            self.assertEqual(schema.model_validate(value).model_dump()['outcome'], value['outcome'])
        # Archived failure: valid JSON and a graph, but unresolved without evidence.
        invalid = [dict(outcome='unresolved', decomposition=graph),
                   dict(outcome='revision_requested', findings=[]),
                   dict(outcome='accepted', decomposition=None, findings=[]),
                   dict(outcome='accepted', decomposition=graph, findings=[finding()]),
                   {'DecomposerHandoff': valid[0]}]
        for value in invalid:
            with self.subTest(value=value):
                self.assertFalse(validator.is_valid(value))
                with self.assertRaises(ValidationError):
                    schema.model_validate(value)

    def test_schema_failure_reports_diagnostics_and_preserves_upstream(self):
        from agentic_web.reasoning import InferenceError
        failure = InferenceError('Final JSON violates the response schema', {
            'validation_errors': [{'loc': ['findings'], 'msg': 'List should have at least 1 item'}]})
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_graph(Path(tmp), HandoffClient([failure]))
            self.assertEqual(result['handoff_status'], 'downstream_failed')
            self.assertIn('findings: List should have at least 1 item', result['error'])
            self.assertTrue(result['handoff_work']['analysis'])
            self.assertFalse((Path(tmp) / 'task.json').exists())

    def run_graph(self, root, client, **policy):
        return build_intake_graph(client).invoke({'prompt': client.drafts[0]['task_spec']['original_requirements'], 'output_file': str(root / 'task.json'),
            'handoff_policy': {'mode': 'collaborative', **policy}})

    def test_first_pass_two_calls_and_compatible_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = HandoffClient()
            result = self.run_graph(root, client)
            self.assertEqual(result.get('error'), None, result)
            self.assertEqual(result['handoff_status'], 'accepted')
            self.assertEqual(len(client.calls), 2)
            request = json.loads((Path(result['run_dir']) / 'handoff/round_000/decompose/request.json').read_text())
            self.assertEqual(client.calls[1][1]['output_schema'], request['response_format']['json_schema']['schema'])
            record = load_decomposition(root / 'task.tasks.json', load_task(root / 'task.json'))
            self.assertTrue(record['ready_for_planning'])
            self.assertEqual(record['handoff']['policy']['version'], '1.0')
            traces = result['analysis']['requirement_traceability']
            self.assertEqual(next(t['origin'] for t in traces if t['requirement_id'] == 'responsive'), 'inferred')
            summary = json.loads((Path(result['run_dir']) / 'handoff_summary.json').read_text())
            self.assertEqual(len(summary['calls']), 2)
            self.assertNotIn('analysis', summary)
            self.assertNotIn('decomposition', summary)
            self.assertNotIn('raw', summary)
            for call in summary['calls']:
                self.assertTrue((Path(call['directory']) / 'metrics.json').exists())
            self.assertFalse((Path(result['run_dir']) / 'task.json').exists())
            self.assertNotIn('analysis', result['analysis']['handoff'])

    def test_standalone_and_handoff_decomposition_share_the_frozen_contract(self):
        from agentic_web.decomposition import decompose
        from test_decomposition import DecompositionClient
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_graph(root, HandoffClient())
            standalone = decompose({'task_spec': result['task_spec'], 'analysis': result['analysis'],
                'output_file': str(root / 'independent' / 'decomposition.json')}, DecompositionClient())
            self.assertEqual(standalone.get('error'), None, standalone)
            for key in ('structure_hash', 'specification_hash', 'tasks', 'requirement_registry', 'dependency_order', 'ready_for_planning'):
                self.assertEqual(standalone['decomposition'][key], result['decomposition'][key])
            self.assertFalse((Path(standalone['decomposition_run_dir']) / 'decomposition.json').exists())

    def test_supported_omission_revised_without_reviewer_or_decision_arrays(self):
        prompt = PROMPT + " Also display the heading 'Opening Hours'."
        initial = analysis_fixture(prompt)
        final = deepcopy(initial)
        extra = deepcopy(final['task_spec']['features'][0])
        extra.update(id='extra', description='Display the Opening Hours heading', acceptance_criteria=['Opening Hours is visible'])
        final['task_spec']['features'].append(extra)
        test = deepcopy(final['task_spec']['tests'][0])
        test.update(test_id='extra-test', requirement_id='extra')
        test['expected_results'][0]['value'] = 'Opening Hours'
        final['task_spec']['tests'].append(test)
        trace = deepcopy(final['requirement_traceability'][0])
        trace['requirement_id'] = 'extra'
        final['requirement_traceability'].append(trace)
        with tempfile.TemporaryDirectory() as tmp:
            omission = finding()
            omission['evidence'] = [{'source': 'request', 'quote': prompt}]
            client = HandoffClient([{'outcome': 'revision_requested', 'findings': [omission]}, 'accepted'], [initial, final])
            result = self.run_graph(Path(tmp), client)
            self.assertEqual(result.get('error'), None, result)
            self.assertEqual((client.analyst_calls, client.decomposer_calls), (2, 2))
            self.assertIn('extra', [r['id'] for r in result['task_spec']['features']])
            self.assertNotIn('decisions', client.calls[2][1])

    def test_false_runtime_claim_retains_valid_upstream(self):
        outcome = {'outcome': 'revision_requested', 'findings': [finding('runtime_capability', 'multiple_sections')]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = HandoffClient([outcome])
            result = self.run_graph(root, client)
            self.assertEqual(result['handoff_status'], 'unresolved')
            self.assertEqual(client.analyst_calls, 1)
            self.assertTrue(result['handoff_work']['analysis'])
            self.assertTrue(result['handoff_work']['disagreements'])
            self.assertFalse((root / 'task.json').exists())

    def test_false_runtime_claim_cannot_veto_valid_decomposition(self):
        initial = analysis_fixture()
        outcome = {'outcome': 'revision_requested', 'findings': [finding('runtime_capability', 'multiple_sections')],
                   'decomposition': tasks_for(TaskSpec.model_validate(initial['task_spec']))}
        with tempfile.TemporaryDirectory() as tmp:
            client = HandoffClient([outcome])
            result = self.run_graph(Path(tmp), client)
            self.assertEqual(result['handoff_status'], 'accepted')
            self.assertTrue(result['handoff_work']['disagreements'])
            self.assertEqual(len(client.calls), 2)

    def test_analyst_can_challenge_feedback_by_retaining_valid_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = HandoffClient(['revision_requested', 'accepted'])
            result = self.run_graph(Path(tmp), client)
            self.assertEqual(result['handoff_status'], 'accepted')
            self.assertEqual(client.analyst_calls, 2)
            self.assertEqual(client.calls[2][1]['current_artifact'], result['handoff_work']['analysis'])

    def test_cli_collaborative_policy_and_output_forwarding(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from agentic_web.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / 'policy.json'
            policy.write_text(json.dumps({'mode': 'collaborative'}))
            with patch('sys.argv', ['agentic_web', 'intake', '--prompt', PROMPT, '--output', str(root / 'task.json'),
                       '--handoff-policy', str(policy), '--decomposition-output', str(root / 'custom.json')]), \
                 patch('agentic_web.intake.intake_app.invoke', return_value={'status': 'needs_review', 'handoff_status': 'unresolved'}) as invoke, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            self.assertEqual(invoke.call_args.args[0]['decomposition_output_file'], str(root / 'custom.json'))
            self.assertEqual(invoke.call_args.args[0]['handoff_policy']['max_revisions'], 2)

    def test_malformed_feedback_and_inference_failure_are_downstream(self):
        inexact = finding()
        inexact['evidence'][0]['quote'] = 'Invented quotation'
        for outcome in ({'outcome': 'revision_requested', 'findings': []},
                        {'outcome': 'revision_requested', 'findings': [inexact]}, RuntimeError('backend unavailable')):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                result = self.run_graph(Path(tmp), HandoffClient([outcome]))
                self.assertEqual(result['handoff_status'], 'downstream_failed')
                self.assertTrue(result['handoff_work']['analysis'])
                self.assertFalse((Path(tmp) / 'task.json').exists())

    def test_nonprogress_exhaustion_and_unresolved(self):
        for outcomes, budget, expected in [(['revision_requested'], 2, 'no_progress'),
                                          (['revision_requested'], 0, 'exhausted'),
                                          (['unresolved'], 2, 'unresolved')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                result = self.run_graph(Path(tmp), HandoffClient(outcomes), max_revisions=budget)
                self.assertEqual(result['handoff_status'], expected)
                self.assertFalse((Path(tmp) / 'task.json').exists())

    def test_interruption_records_terminal_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                self.run_graph(root, HandoffClient([KeyboardInterrupt()]))
            summaries = list(root.glob('task.intake/*/handoff_summary.json'))
            self.assertEqual(json.loads(summaries[0].read_text())['status'], 'interrupted')
            self.assertFalse((root / 'task.json').exists())

    def test_native_langgraph_nodes_and_checkpoint(self):
        from langgraph.checkpoint.memory import InMemorySaver
        with tempfile.TemporaryDirectory() as tmp:
            graph = build_intake_graph(HandoffClient(), checkpointer=InMemorySaver())
            settings = {'configurable': {'thread_id': 'handoff-test'}}
            events = list(graph.stream({'prompt': PROMPT, 'output_file': str(Path(tmp) / 'task.json'),
                'handoff_policy': {'mode': 'collaborative'}}, config=settings, stream_mode='updates'))
            self.assertEqual([next(iter(e)) for e in events],
                ['requirements_prepare', 'requirements_analyst', 'handoff_decomposer', 'requirements_finalize'])
            self.assertEqual(graph.get_state(settings).values['handoff_status'], 'accepted')

    def test_checkpoint_resume_does_not_rerun_analyst(self):
        from langgraph.checkpoint.memory import InMemorySaver
        with tempfile.TemporaryDirectory() as tmp:
            client = HandoffClient([KeyboardInterrupt(), 'accepted'])
            graph = build_intake_graph(client, checkpointer=InMemorySaver())
            config = {'configurable': {'thread_id': 'resume'}}
            with self.assertRaises(KeyboardInterrupt):
                graph.invoke({'prompt': PROMPT, 'output_file': str(Path(tmp) / 'task.json'),
                    'handoff_policy': {'mode': 'collaborative'}}, config)
            result = graph.invoke(None, config)
            self.assertEqual(result['handoff_status'], 'accepted')
            self.assertEqual(client.analyst_calls, 1)
            self.assertEqual(len(list(Path(result['run_dir']).glob('handoff/round_000/decompose/**/request.json'))), 2)

    def test_collaborative_graph_uses_same_handoff_nodes(self):
        from agentic_web.workflow import build_graph
        with tempfile.TemporaryDirectory() as tmp:
            client = HandoffClient()
            result = build_graph(client, workflow='collaborative').invoke({'prompt': PROMPT,
                'task_output_file': str(Path(tmp) / 'task.json'), 'analyze_only': True,
                'handoff_policy': {'mode': 'collaborative'}})
            self.assertEqual(result.get('error'), None, result)
            self.assertEqual(result['handoff_status'], 'accepted')
            self.assertEqual(len(client.calls), 2)

    def test_revision_cannot_remove_explicit_obligation(self):
        original = analysis_fixture()
        removed = deepcopy(original)
        removed['task_spec']['features'] = [r for r in removed['task_spec']['features'] if r['id'] != 'notice']
        removed['task_spec']['tests'] = [r for r in removed['task_spec']['tests'] if r['requirement_id'] != 'notice']
        removed['requirement_traceability'] = [r for r in removed['requirement_traceability'] if r['requirement_id'] != 'notice']
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_graph(Path(tmp), HandoffClient(['revision_requested'], [original, removed]))
            self.assertEqual(result['handoff_status'], 'upstream_failed')
            self.assertIn('notice', [r['id'] for r in result['handoff_work']['analysis']['task_spec']['features']])

    def test_optional_recommendation_cannot_be_promoted_by_feedback(self):
        original = analysis_fixture()
        revised = deepcopy(original)
        revised['task_spec']['features'][1]['required'] = True
        for test in revised['task_spec']['tests']:
            if test['requirement_id'] == 'accent':
                test['required'] = True
        for trace in revised['requirement_traceability']:
            if trace['requirement_id'] == 'accent':
                trace['origin'] = 'inferred'
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_graph(Path(tmp), HandoffClient(['revision_requested'], [original, revised]))
            self.assertEqual(result['handoff_status'], 'upstream_failed')
            self.assertIn('optional requirement', result['error'])

    def test_two_revision_limit_with_changing_feedback(self):
        outcomes = []
        for i in range(3):
            item = finding()
            item['problem'] = f'Distinct reported omission {i}'
            outcomes.append({'outcome': 'revision_requested', 'findings': [item]})
        with tempfile.TemporaryDirectory() as tmp:
            client = HandoffClient(outcomes)
            result = self.run_graph(Path(tmp), client)
            self.assertEqual(result['handoff_status'], 'exhausted')
            self.assertEqual((client.analyst_calls, client.decomposer_calls), (3, 3))
