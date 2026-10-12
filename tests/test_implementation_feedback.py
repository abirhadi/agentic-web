"""Bounded feedback through the existing graph; all model calls are mocked."""
from copy import deepcopy
import json
import shutil
from pathlib import Path
import tempfile
import unittest

from langgraph.checkpoint.memory import InMemorySaver
from agentic_web.schema import TaskSpec, load_task, verification_readiness
from agentic_web.verification import finish
from agentic_web.workflow import build_graph, contract_hash
from agentic_web.provenance import filesystem_path
from collaboration_mock_client import CollaborationMockClient
from mock_client import ROOT
from test_framework import stub_verifier


class RevisionClient(CollaborationMockClient):
    def __init__(self, changing=False):
        super().__init__()
        self.changing = changing
        self.implementations = 0

    def complete(self, system, user, config, **kwargs):
        raw, metrics = super().complete(system, user, config, **kwargs)
        if self.roles[-1] == 'implementer':
            self.implementations += 1
            if self.changing:
                raw = raw.replace('</body>', f'<!-- revision {self.implementations} --></body>')
            metrics['usage'] = {'total_tokens': 20}
        return raw, metrics


class ImplementationFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(shutil.rmtree, filesystem_path(self.tmp.name))
        self.task = load_task(ROOT / 'tasks/todo.json')
        self.client = RevisionClient()
        self.inputs = {'task_spec': self.task.model_dump(mode='json'), 'output_root': self.tmp.name,
                       'model_config': {'manual_verification': 'off', 'max_repair_attempts': 2}}

    def verify_sequence(self, failures):
        calls = []
        def verify(path, task, output):
            report = stub_verifier(path, task, output)
            for item, test in zip(report['tests'], task.tests):
                item['requirement_id'] = test.requirement_id
                if test.test_type == 'manual':
                    item['status'] = 'unverified'
            if len(calls) < failures:
                report['tests'][0].update(status='failed', error='Expected visible task; observed count 0', phase='expected_result[0]')
            calls.append(str(path))
            return finish(task, report)
        return verify, calls

    def run_graph(self, failures=0, **kwargs):
        verify, self.verifications = self.verify_sequence(failures)
        return build_graph(self.client, verify, workflow='collaborative', **kwargs).invoke(self.inputs)

    def test_first_pass_and_frozen_contract(self):
        result = self.run_graph()
        self.assertEqual(result['verification_outcome'], 'first_pass_verified')
        self.assertTrue(result['acceptance']['accepted'])
        self.assertEqual(result['implementation_work']['repair_count'], 0)
        self.assertEqual(result['requirement_hash'], contract_hash(self.task))

    def test_repair_preserves_upstream_and_evidence(self):
        result = self.run_graph(1)
        self.assertEqual(result['verification_outcome'], 'verified_after_repair')
        self.assertTrue(result['acceptance']['accepted'])
        for role in ('planner', 'ux', 'architect', 'coordinator', 'qa'):
            self.assertEqual(self.client.roles.count(role), 1, role)
        self.assertEqual(self.client.implementations, 2)
        work = result['implementation_work']
        self.assertEqual(work['repair_count'], 1)
        self.assertEqual(work['usage']['total_tokens'], 40)
        self.assertEqual(len(work['history']), 2)
        for entry in work['history']:
            self.assertTrue(Path(entry['artifact']).is_file())
            self.assertTrue(Path(entry['verification_file']).is_file())
        repair = [c for c in self.client.calls if 'verification_feedback' in c['payload']][0]
        finding = repair['payload']['verification_feedback'][0]
        self.assertEqual(finding['requirement_ids'], [self.task.tests[0].requirement_id])
        self.assertIn('observed count 0', finding['observed']['error'])
        self.assertIn('current_application', repair['payload'])
        self.assertEqual(result['task_spec'], self.inputs['task_spec'])

    def test_multiple_repairs_exhaustion_and_no_progress(self):
        for changing, failures, expected, count in ((True, 2, 'accepted', 2), (True, 9, 'exhausted', 2),
                                                    (False, 9, 'no_progress', 1)):
            with self.subTest(changing=changing, failures=failures):
                self.client = RevisionClient(changing)
                result = self.run_graph(failures)
                self.assertEqual(result['implementation_work']['termination'], expected)
                self.assertEqual(result['implementation_work']['repair_count'], count)
                self.assertEqual(result['acceptance']['accepted'], expected == 'accepted')

    def test_manual_on_off_and_uncovered_provenance(self):
        self.inputs['task_spec']['features'].append({'id': 'human_review', 'description': 'Human review of visual quality',
            'acceptance_criteria': ['A person judges visual quality'], 'verification': 'manual', 'required': True})
        self.inputs['task_spec']['tests'].append({'test_id': 'human-review', 'requirement_id': 'human_review',
                                                'test_type': 'manual', 'required': True})
        for mode, expected in (('off', True), ('on', False)):
            with self.subTest(mode=mode):
                self.inputs['model_config']['manual_verification'] = mode
                result = self.run_graph()
                self.assertEqual(result['acceptance']['accepted'], expected)
                self.assertEqual(result['verification']['tests'][-1]['status'], 'skipped' if mode == 'off' else 'unverified')
                self.assertIn('human_review', result['acceptance']['uncovered_requirements'])
                self.assertEqual(result['implementation_work']['repair_count'], 0)

    def test_manual_only_never_automatically_verified(self):
        spec = deepcopy(self.inputs['task_spec'])
        spec['features'] = spec['features'][:1]
        spec['features'][0]['verification'] = 'manual'
        spec['nonfunctional_requirements'] = []
        spec['tests'] = [{'test_id': 'human-review', 'requirement_id': spec['features'][0]['id'], 'test_type': 'manual'}]
        self.inputs['task_spec'] = spec
        result = self.run_graph()
        self.assertEqual(result['verification_outcome'], 'unverified')
        self.assertFalse(result['acceptance']['accepted'])
        self.assertFalse(result['verification_readiness']['has_automated_coverage'])
        self.assertEqual(result['implementation_work']['repair_count'], 0)

    def test_unsupported_and_infrastructure_do_not_repair(self):
        for kind in ('unsupported', 'infrastructure'):
            with self.subTest(kind=kind):
                client = RevisionClient()
                def verify(path, task, output):
                    report = stub_verifier(path, task, output)
                    report['tests'][0].update(status='unverified', failure_kind=kind)
                    if kind == 'infrastructure':
                        report['infrastructure_errors'] = ['Browser launch failed']
                        report['errors'] = ['Browser launch failed']
                    return finish(task, report)
                result = build_graph(client, verify, workflow='collaborative').invoke(self.inputs)
                self.assertEqual(client.implementations, 1)
                self.assertEqual(result['verification_outcome'], 'unverified' if kind == 'unsupported' else 'verification_infrastructure_error')

    def test_checkpoint_resume_only_reexecutes_local_stages(self):
        saver = InMemorySaver()
        verify, calls = self.verify_sequence(1)
        config = {'configurable': {'thread_id': 'same-candidate'}}
        graph = build_graph(self.client, verify, workflow='collaborative', checkpointer=saver)
        first = graph.invoke(self.inputs, config, interrupt_after=['verifier'])
        root = Path(first['run_dir'])
        original = {name: (root / name).read_bytes() for name in ('task.json', 'plan.json', 'ux_review.json', 'architecture_review.json')}
        resumed = build_graph(self.client, verify, workflow='collaborative', checkpointer=saver).invoke(None, config)
        self.assertEqual(resumed['run_dir'], first['run_dir'])
        self.assertEqual(resumed['verification_outcome'], 'verified_after_repair')
        self.assertEqual(self.client.roles.count('planner'), 1)
        self.assertEqual(len(calls), 2)
        for name, value in original.items():
            self.assertEqual((root / name).read_bytes(), value)

    def test_repair_resource_leases_and_audits_remain_candidate_owned(self):
        from agentic_web.resource_scheduler import ResourceScheduler
        from agentic_web.resource_schema import ResourceSettings
        from resource_mock_backend import SimulatedBackend
        backend = SimulatedBackend()
        scheduler = ResourceScheduler(backend, backend.hardware)
        self.inputs['model_config'].update(model='tiny', planner_model='light', ux_model='light',
                                          architect_model='tiny', coordinator_model='light', qa_model='tiny')
        self.inputs['resource_config'] = ResourceSettings(prefetch_all_if_gpu_fit=False).model_dump(mode='json')
        result = self.run_graph(1, resource_scheduler=scheduler)
        self.assertTrue(result['acceptance']['accepted'], result.get('error'))
        self.assertEqual(result['implementation_work']['repair_count'], 1)
        for call in result['implementation_work']['calls']:
            root = Path(call['directory'])
            self.assertTrue((root / 'resources.implementer.json').is_file())
            self.assertTrue((root / 'resources.inference.implementer.json').is_file())
            self.assertIn('assigned_device', call['metrics'])
        self.assertFalse(backend.instances)

    def test_baseline_uses_same_local_feedback_edge(self):
        verify, calls = self.verify_sequence(1)
        result = build_graph(self.client, verify).invoke(self.inputs)
        self.assertEqual(result['verification_outcome'], 'verified_after_repair')
        self.assertEqual(self.client.roles.count('planner'), 1)
        self.assertEqual(len(calls), 2)

    def test_exception_after_repair_replaces_stale_verifier_evidence(self):
        first_verify, calls = self.verify_sequence(1)
        def verify(path, task, output):
            if calls:
                raise RuntimeError('Verifier infrastructure unavailable')
            return first_verify(path, task, output)
        result = build_graph(self.client, verify, workflow='collaborative').invoke(self.inputs)
        self.assertEqual(result['verification_outcome'], 'verification_infrastructure_error')
        self.assertEqual(result['verification']['tests'], [])
        self.assertIn('Verifier infrastructure unavailable', result['verification']['errors'][0])
        self.assertEqual(self.client.implementations, 2)

    def test_empty_assertions_rejected_without_mutating_spec(self):
        from pydantic import ValidationError
        for changes in ({'expected_results': []}, {'expected_results': [{'assertion': 'contains_text', 'selector': 'body', 'value': ''}]}):
            spec = deepcopy(self.inputs['task_spec'])
            spec['tests'][0].update(changes)
            with self.assertRaises(ValidationError):
                TaskSpec.model_validate(spec)

    def test_candidate_identity_and_frozen_settings_survive_repair(self):
        from agentic_web.experiments import ExperimentConfig, build_plan, save_plan
        from agentic_web.batch import run_experiment
        from test_experiments import configuration
        root = Path(self.tmp.name)
        task_file = root / 'task.json'
        task_file.write_text(self.task.model_dump_json(), encoding='utf-8')
        before = task_file.read_bytes()
        value = configuration()
        value['factors'].update(model_configurations=['large'], temperatures=[.4], seeds=[1],
                                variance_conditions=[{'identifier': 'control'}])
        legacy = build_plan(ExperimentConfig.model_validate(value), root)
        self.assertNotIn('max_repair_attempts', legacy['candidates'][0]['model_configuration'])
        value['model_configurations']['large'].update(manual_verification='off', max_repair_attempts=2)
        plan = build_plan(ExperimentConfig.model_validate(value), root)
        candidate = plan['candidates'][0]
        self.assertNotEqual(candidate['candidate_id'], legacy['candidates'][0]['candidate_id'])
        path = save_plan(plan, root / 'config.json')
        frozen = path.read_bytes()
        verify, _ = self.verify_sequence(1)
        result = run_experiment(path, client=self.client, verifier=verify)
        record = json.loads(Path(result['outcomes'][0]['record']).read_text())
        self.assertEqual(record['candidate_id'], candidate['candidate_id'])
        self.assertEqual(record['verification_outcome'], 'verified_after_repair')
        self.assertEqual(record['implementation_feedback']['repair_count'], 1)
        self.assertEqual(record['requested_model_config']['max_repair_attempts'], 2)
        self.assertEqual(path.read_bytes(), frozen)
        self.assertEqual(task_file.read_bytes(), before)

    def test_readiness_is_observable_and_does_not_fabricate_checks(self):
        spec = self.task.model_dump(mode='json')
        spec['tests'] = []
        readiness = verification_readiness(TaskSpec.model_validate(spec))
        self.assertFalse(readiness['has_automated_coverage'])
        self.assertEqual(len(readiness['gaps']), len(self.task.requirements))
        self.assertFalse(readiness['mandatory_coverage_complete'])
        self.assertEqual(verification_readiness(self.task)['executable_tests'],
                         [t.model_dump(mode='json') for t in self.task.tests])
