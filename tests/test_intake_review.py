"""Feedback analysis with deterministic model substitutes; no network or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_web.intake import build_intake_graph
from agentic_web.intake_review import ReviewPolicy
from agentic_web.reasoning import InferenceError
from agentic_web.provenance import fingerprint
from agentic_web.schema import load_task
from test_intake import PROMPT, analysis_fixture, wire_response


def finding(observation='An explicit obligation is missing', severity='blocking', category='explicit_omission'):
    return {'category': category, 'severity': severity, 'observation': observation,
            'evidence': [{'source': 'request', 'quote': PROMPT}], 'recommendation': None}


class FeedbackClient:
    def __init__(self, drafts=None, reviews=None, action='resolve', assessments='resolved'):
        self.drafts = drafts or [wire_response(analysis_fixture())]
        self.reviews = reviews or [[]]
        self.calls = []
        self.counts = {'analyst': 0, 'reviewer': 0}
        self.action, self.assessments = action, assessments

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        payload = json.loads(user)
        role = 'reviewer' if response_schema.__name__ == 'SemanticReview' else 'analyst'
        index = self.counts[role]
        self.counts[role] += 1
        self.calls.append((role, payload, config, response_schema, system))
        if role == 'analyst':
            value = self.drafts[min(index, len(self.drafts) - 1)]
            value = value if isinstance(value, BaseException) else deepcopy(value)
            if index and isinstance(value, dict):
                value = {'analysis': value, 'decisions': [{'finding_id': f['finding_id'], 'action': self.action,
                    'rationale': 'Evaluated against the original request',
                    'evidence': [{'source': 'request', 'quote': PROMPT}]} for f in payload['findings']],
                    'requirement_lineage': {}}
        else:
            value = deepcopy(self.reviews[min(index, len(self.reviews) - 1)])
            if isinstance(value, list):
                value = {'findings': value, 'prior_finding_assessments': [
                    {'finding_id': f['finding_id'], 'status': self.assessments,
                     'rationale': 'Compared the original source and the revised analysis',
                     'evidence': [{'source': 'request', 'quote': PROMPT}]} for f in payload['prior_findings']]}
        if isinstance(value, BaseException):
            raise value
        raw = value if isinstance(value, str) else json.dumps(value)
        return raw, {'finish_reason': 'stop', 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}


class IntakeFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'task.json'

    def invoke(self, client, policy=None, **state):
        return build_intake_graph(client).invoke({'prompt': PROMPT, 'output_file': str(self.output),
            'review_policy': policy or {'mode': 'feedback'}, **state})

    def summary(self, result):
        return json.loads((Path(result['run_dir']) / 'review_summary.json').read_text())

    def test_omission_corrected_and_downstream_manifest_compatible(self):
        complete = wire_response(analysis_fixture())
        omitted = deepcopy(complete)
        omitted['task_spec']['features'] = omitted['task_spec']['features'][1:]
        client = FeedbackClient([omitted, complete], [[finding()], []])
        result = self.invoke(client, model_config={'seed': 42, 'temperature': .7, 'top_p': .8,
                                                 'model': 'analyst', 'qa_model': 'reviewer'})
        self.assertEqual(result['status'], 'analyzed', result.get('error'))
        task = load_task(self.output)
        self.assertIn('notice', {r.id for r in task.requirements})
        self.assertEqual(task.original_requirements, PROMPT)
        self.assertEqual(client.counts, {'analyst': 2, 'reviewer': 2})
        for role, payload, config, schema, system in client.calls:
            self.assertEqual((config.seed, config.temperature, config.top_p), (42, .7, .8))
            if role == 'reviewer':
                self.assertEqual(config.planner_model, 'reviewer')
                self.assertEqual(config.inference_role, 'qa')
            else:
                self.assertEqual(config.inference_role, 'intake')
        record = json.loads(Path(result['analysis_file']).read_text())
        self.assertEqual(record['review_policy_hash'], fingerprint(ReviewPolicy(mode='feedback').model_dump(mode='json')))
        self.assertEqual(record['requirement_groups']['inferred'], ['responsive'])
        self.assertEqual(record['requirement_groups']['recommendation'], ['accent'])
        self.assertEqual(len(record['review']['calls']), 4)
        self.assertEqual(record['review']['usage_totals']['total_tokens'], 480)
        self.assertTrue((Path(result['run_dir']) / 'review/round_001/lineage.json').exists())

    def test_misleading_review_rejected_with_evidence_and_independent_assessment(self):
        client = FeedbackClient(reviews=[[finding('Add a required shopping cart', category='unsupported_assumption')], []],
                                action='reject', assessments='rejected')
        result = self.invoke(client)
        self.assertEqual(result['status'], 'analyzed', result.get('error'))
        self.assertEqual(len(load_task(self.output).requirements), 3)
        decisions = json.loads((Path(result['run_dir']) / 'review/round_001/decisions.json').read_text())
        self.assertEqual(decisions[0]['action'], 'reject')

    def test_schema_and_canonical_validation_repaired_through_feedback(self):
        good = wire_response(analysis_fixture())
        duplicate = deepcopy(good)
        duplicate['task_spec']['features'].append(deepcopy(duplicate['task_spec']['features'][0]))
        invalid_trace = deepcopy(good)
        invalid_trace['task_spec']['features'][0]['traceability']['sources'][0]['quote'] = 'fabricated quotation'
        for draft in ('{bad JSON', duplicate, invalid_trace,
                      InferenceError('Schema rejected', {'error_classification': 'schema_validation_error',
                        'final_content': '{bad JSON', 'finish_reason': 'stop', 'usage': {'total_tokens': 200}})):
            with self.subTest(draft=str(draft)[:50]):
                self.output = Path(self.temp.name) / f'task-{len(list(Path(self.temp.name).glob("*.intake")))}.json'
                result = self.invoke(FeedbackClient([draft, good], [[], []]))
                self.assertEqual(result['status'], 'analyzed', result.get('error'))
                review = json.loads((Path(result['run_dir']) / 'review/round_000/deterministic_review.json').read_text())
                self.assertFalse(review['valid'])

    def test_early_convergence_and_advisory_findings(self):
        result = self.invoke(FeedbackClient(reviews=[[finding('Optional polish', 'advisory', 'implicit_requirement')]]))
        self.assertEqual(result['status'], 'analyzed')
        self.assertEqual(self.summary(result)['revisions'], 0)
        self.assertEqual(len(self.summary(result)['calls']), 2)

    def test_budget_exhaustion_and_no_progress_never_publish(self):
        for budget, expected in ((0, 'review_exhausted'), (1, 'review_exhausted'), (4, 'no_progress')):
            with self.subTest(budget=budget):
                client = FeedbackClient(reviews=[[finding()]], assessments='unresolved')
                result = self.invoke(client, {'mode': 'feedback', 'max_revisions': budget})
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['review_status'], expected)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.summary(result)['status'], expected)

    def test_reviewer_malformed_inexact_evidence_or_missing_assessment_fails(self):
        wrong_quote = finding()
        wrong_quote['evidence'][0]['quote'] = 'invented'
        for reviews in (['{broken'], [[wrong_quote]], [[finding()], {'findings': [], 'prior_finding_assessments': []}]):
            with self.subTest(reviews=reviews):
                result = self.invoke(FeedbackClient(reviews=reviews))
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(self.output.exists())

    def test_explicit_requirement_cannot_be_silently_demoted(self):
        good = wire_response(analysis_fixture())
        changed = deepcopy(good)
        changed['task_spec']['features'][0]['required'] = False
        client = FeedbackClient([good, changed], [[finding()], []])
        result = self.invoke(client, {'mode': 'feedback', 'max_revisions': 1})
        self.assertEqual(result['review_status'], 'validation_failed')
        self.assertFalse(self.output.exists())
        deterministic = (Path(result['run_dir']) / 'review/round_001/deterministic_review.json').read_text()
        self.assertIn('Explicit obligation lost or demoted', deterministic)

    def test_unsupported_explicit_claim_can_be_corrected_with_review_evidence(self):
        good = wire_response(analysis_fixture())
        bad = deepcopy(good)
        invented = deepcopy(bad['task_spec']['features'][0])
        invented.update(id='invented', description='An unsupported required shopping cart')
        invented['tests'][0]['test_id'] = 'invented-test'
        bad['task_spec']['features'].append(invented)
        criticism = finding('Shopping cart is not requested', category='unsupported_assumption')
        criticism['requirement_ids'] = ['invented']
        result = self.invoke(FeedbackClient([bad, good], [[criticism], []], action='accept'))
        self.assertEqual(result['status'], 'analyzed', result.get('error'))
        self.assertNotIn('invented', {r.id for r in load_task(self.output).requirements})

    def test_truncation_does_not_trigger_semantic_repair(self):
        error = InferenceError('Truncated', {'error_classification': 'output_budget_exhausted',
            'finish_reason': 'length', 'final_content': '{unfinished', 'usage': {'total_tokens': 300}})
        client = FeedbackClient([error])
        result = self.invoke(client)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(client.counts, {'analyst': 1, 'reviewer': 0})
        self.assertFalse(self.output.exists())
        self.assertEqual(self.summary(result)['usage_totals']['total_tokens'], 300)

    def test_withdrawn_criticism_cannot_authorize_dropping_explicit_obligation(self):
        good = wire_response(analysis_fixture())
        dropped = deepcopy(good)
        dropped['task_spec']['features'] = dropped['task_spec']['features'][1:]
        criticism = finding('Dispute the heading', category='unsupported_assumption')
        criticism['requirement_ids'] = ['notice']
        result = self.invoke(FeedbackClient([good, dropped], [[criticism], []], action='accept', assessments='rejected'),
                             {'mode': 'feedback', 'max_revisions': 1})
        self.assertEqual(result['review_status'], 'validation_failed')
        self.assertFalse(self.output.exists())

    def test_inference_failure_and_interrupt_are_auditable(self):
        result = self.invoke(FeedbackClient(reviews=[ConnectionError('offline')]))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(self.output.exists())
        self.assertEqual(self.summary(result)['status'], 'review_failed')
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(FeedbackClient(reviews=[KeyboardInterrupt()]))
        summaries = list(Path(self.temp.name).glob('*.intake/*/review_summary.json'))
        self.assertIn('interrupted', [json.loads(p.read_text())['status'] for p in summaries])
        self.assertFalse(self.output.exists())

    def test_blocking_uncertainty_not_a_validation_failure_and_not_frozen(self):
        wire = wire_response(analysis_fixture())
        wire['missing_information'] = [{'description': 'Need source content', 'question': 'Supply the source?',
            'blocks_generation': True, 'requirement_ids': [], 'sources': []}]
        result = self.invoke(FeedbackClient([wire]))
        self.assertEqual(result['review_status'], 'needs_review')
        self.assertFalse(self.output.exists())
        self.assertEqual(self.summary(result)['status'], 'needs_review')

    def test_off_baseline_and_stable_policy_hash(self):
        client = FeedbackClient()
        result = self.invoke(client, {'mode': 'off'})
        self.assertEqual(result['status'], 'analyzed')
        self.assertEqual(client.counts, {'analyst': 1, 'reviewer': 0})
        self.assertFalse((Path(result['run_dir']) / 'review').exists())
        self.assertEqual(fingerprint(ReviewPolicy().model_dump(mode='json')),
                         fingerprint(ReviewPolicy.model_validate({'version': '1.0', 'mode': 'off', 'max_revisions': 2}).model_dump(mode='json')))

    def test_restructuring_retains_explicit_provenance_with_lineage(self):
        from agentic_web.intake_review import _preservation
        from agentic_web.intake_schema import RequirementsAnalysis
        old = RequirementsAnalysis.model_validate(analysis_fixture())
        new = old.model_copy(deep=True)
        new.task_spec.features[0].id = 'new-notice'
        new.requirement_traceability[0].requirement_id = 'new-notice'
        new.task_spec.tests[0].requirement_id = 'new-notice'
        _preservation(old, new, {'notice': ['new-notice']})
        with self.assertRaisesRegex(ValueError, 'Explicit obligation'):
            _preservation(old, new, {})
        new.task_spec.forbidden_assumptions = []
        with self.assertRaisesRegex(ValueError, 'prohibitions'):
            _preservation(old, new, {'notice': ['new-notice']})

    def test_documents_and_existing_context_reach_both_roles_unchanged(self):
        document = Path(self.temp.name) / 'source.md'
        document.write_text('Source document text.\n', encoding='utf-8')
        client = FeedbackClient()
        result = self.invoke(client, supporting_context='Existing context.', documents=[str(document)], document_root=self.temp.name)
        self.assertEqual(result['status'], 'analyzed', result.get('error'))
        context = load_task(self.output).supporting_context
        self.assertIn('Existing context.', context)
        self.assertIn('Source document text.', context)
        for role, payload, *_ in client.calls:
            self.assertEqual(payload['supporting_context'], context)
        self.assertEqual((Path(result['run_dir']) / 'supporting_context.txt').read_bytes().decode('utf-8'), context)

    def test_collaborative_input_and_decomposition_provenance_compatibility(self):
        from agentic_web.workflow import build_graph
        from agentic_web.decomposition import analysis_for
        from test_framework import stub_verifier
        client = FeedbackClient()
        result = build_graph(client, stub_verifier, workflow='collaborative').invoke({
            'prompt': PROMPT, 'task_output_file': str(self.output), 'analyze_only': True,
            'review_policy': {'mode': 'feedback'}})
        self.assertEqual(result['status'], 'analyzed', result.get('error'))
        self.assertEqual(client.counts, {'analyst': 1, 'reviewer': 1})
        record = analysis_for(load_task(self.output), task_path=str(self.output))
        self.assertEqual(record['review']['status'], 'converged')

    def test_resource_admission_for_each_call_and_cleanup_on_reviewer_failure(self):
        from agentic_web.resource_scheduler import ResourceScheduler, select_placements
        from agentic_web.resource_schema import ResourceSettings
        from resource_mock_backend import SimulatedBackend
        backend = SimulatedBackend()
        settings = ResourceSettings(gpu_safety_bytes=0, ram_safety_bytes=0, estimate_padding=0,
            prefetch_all_if_gpu_fit=False, resident_devices={'manual-resident': '0'})
        placement = select_placements([backend.estimate('light', settings, 1)], backend.hardware(), backend.capabilities(), settings)[0]['light']
        backend.load(placement, 'manual-resident')
        scheduler = ResourceScheduler(backend, backend.hardware)
        class AdmittedClient(FeedbackClient):
            def complete(self, system, user, config, **kwargs):
                if config.planner_model not in backend.instances:
                    raise AssertionError('Inference bypassed admission')
                return super().complete(system, user, config, **kwargs)
        result = build_intake_graph(AdmittedClient(reviews=[ConnectionError('offline')]), resource_scheduler=scheduler).invoke({
            'prompt': PROMPT, 'output_file': str(self.output), 'model_config': {'model': 'tiny'},
            'review_policy': {'mode': 'feedback'}, 'resource_config': settings.model_dump()})
        self.assertEqual(result['status'], 'failed')
        self.assertIn('offline', result['error'])
        self.assertEqual(set(backend.instances), {'manual-resident'})
        self.assertNotIn('manual-resident', backend.unloads)

    def test_cli_review_policy_file(self):
        from agentic_web.__main__ import main
        from contextlib import redirect_stdout
        import io
        policy_path = Path(self.temp.name) / 'policy.json'
        policy_path.write_text('{"mode":"feedback"}', encoding='utf-8')
        client = FeedbackClient()
        with patch('sys.argv', ['agentic_web', 'intake', '--prompt', PROMPT, '--output', str(self.output),
                               '--review-policy', str(policy_path), '--qa-model', 'reviewer']), \
             patch('agentic_web.intake.intake_app', build_intake_graph(client)), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        self.assertEqual(client.calls[1][2].planner_model, 'reviewer')

    def test_collaborative_stream_exposes_real_review_and_revision_nodes(self):
        from agentic_web.workflow import build_graph
        from test_framework import stub_verifier
        client = FeedbackClient(reviews=[[finding()], []], action='reject', assessments='rejected')
        graph = build_graph(client, stub_verifier, workflow='collaborative')
        events = []
        for update in graph.stream({'prompt': PROMPT, 'task_output_file': str(self.output),
                'analyze_only': True, 'review_policy': {'mode': 'feedback'}}, stream_mode='updates'):
            name = next(iter(update))
            events.append(name)
            self.assertEqual(self.output.exists(), name == 'requirements_finalize')
            # Graph state contains serializable data, not a model, generator, or session.
            json.dumps(update)
        self.assertEqual(events, ['requirements_prepare', 'requirements_analyst', 'requirements_review',
                                  'requirements_revision', 'requirements_review', 'requirements_finalize'])
        self.assertEqual(client.counts, {'analyst': 2, 'reviewer': 2})

    def test_disabled_review_graph_bypasses_review_and_revision(self):
        graph = build_intake_graph(FeedbackClient())
        events = [next(iter(update)) for update in graph.stream({'prompt': PROMPT,
            'output_file': str(self.output)}, stream_mode='updates')]
        self.assertEqual(events, ['requirements_prepare', 'requirements_analyst', 'requirements_finalize'])

    def test_maximum_policy_budget_is_not_cut_short_by_graph_recursion_limit(self):
        client = FeedbackClient(reviews=[[finding(f'Blocking issue {i}')] for i in range(11)])
        result = self.invoke(client, {'mode': 'feedback', 'max_revisions': 10})
        self.assertEqual(result['review_status'], 'review_exhausted')
        self.assertEqual(client.counts, {'analyst': 11, 'reviewer': 11})
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
