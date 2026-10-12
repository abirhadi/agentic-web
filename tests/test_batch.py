"""B3 sequential execution through real graphs with mocked inference."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from agentic_web.batch import run_experiment, outcome
from agentic_web.experiments import ExperimentConfig, build_plan, save_plan
from agentic_web.provenance import fingerprint, filesystem_path
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from test_experiments import configuration
from test_framework import stub_verifier

ROOT = Path(__file__).resolve().parents[1]


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.value = configuration()
        self.value['factors'].update(temperatures=[0.4], seeds=[1], variance_conditions=[{'identifier': 'control'}])
        (self.root / 'task.json').write_bytes((ROOT / 'tasks/todo.json').read_bytes())

    def plan(self):
        config = ExperimentConfig.model_validate(self.value)
        plan = build_plan(config, self.root)
        return save_plan(plan, self.root / 'config.json'), plan

    def records(self, result):
        return [json.loads(Path(item['record']).read_text()) for item in result['outcomes']]

    def test_long_batch_paths_preserve_artifacts_and_resume_identity(self):
        self.addCleanup(shutil.rmtree, filesystem_path(self.root))
        self.value['experiment_id'] = 'long-' + 'x' * 95
        path, plan = self.plan()
        result = run_experiment(path, client=CollaborationMockClient(), verifier=stub_verifier)
        self.assertEqual(result['progress']['counts']['completed'], len(plan['candidates']))
        for record in self.records(result):
            directory = Path(record['run_dir'])
            diagnostic = directory / 'planner_structured_output_error.json'
            self.assertGreater(len(str(diagnostic)), 260)
            from agentic_web.workflow import write_json
            write_json(directory, diagnostic.name, {'error': 'diagnostic preserved'})
            self.assertEqual(json.loads(diagnostic.read_text())['error'], 'diagnostic preserved')
            self.assertTrue(Path(record['artifact']['path']).is_file())
        from agentic_web.checkpoint import progress
        # Ordinary and extended spellings must recognize the same terminal records.
        self.assertEqual(progress(json.loads(path.read_text()), path.parent / 'execution')['counts'],
                         result['progress']['counts'])
        resumed = run_experiment(path, resume=True, client=CollaborationMockClient(), verifier=stub_verifier)
        self.assertEqual(resumed['selected_count'], 0)

    def test_sequential_isolation_exact_assignments_and_artifacts(self):
        path, plan = self.plan()
        active, maximum = 0, 0
        client = CollaborationMockClient()
        def factory(*args, **kwargs):
            graph = build_graph(*args, **kwargs)
            class Tracked:
                def invoke(self, inputs):
                    nonlocal active, maximum
                    active += 1
                    maximum = max(maximum, active)
                    try:
                        return graph.invoke(inputs)
                    finally:
                        active -= 1
            return Tracked()
        with patch('agentic_web.batch.build_graph', side_effect=factory):
            result = run_experiment(path, client=client, verifier=stub_verifier)
        self.assertEqual(maximum, 1)
        self.assertEqual(result['status'], 'completed')
        records = self.records(result)
        self.assertEqual([r['candidate_id'] for r in records], [c['candidate_id'] for c in plan['candidates']])
        self.assertEqual(len({r['run_dir'] for r in records}), 2)
        for record, candidate in zip(records, plan['candidates']):
            self.assertEqual(record['execution_status'], 'generation_completed_verification_passed')
            self.assertTrue(record['artifact']['available'])
            self.assertTrue(Path(record['artifact']['path']).is_file())
            self.assertEqual(record['candidate'], candidate)
            self.assertEqual(record['role_sampling'], candidate['role_sampling'])
            self.assertTrue(record['metrics'])
            for role, identity in candidate['model_configuration']['roles'].items():
                metrics = record['metrics'][role + '_metrics']
                self.assertEqual(metrics['requested_model'], identity['model'])
                self.assertEqual(metrics['seed'], candidate['role_sampling'][role]['seed'])
            self.assertEqual(record['requested_model_config']['temperature'], candidate['temperature'])

    def test_variance_and_control_with_frozen_b1_definitions(self):
        profile = json.loads((ROOT / 'experiments/profiles/layout_exploration.json').read_text())
        (self.root / 'profile.json').write_text(json.dumps(profile))
        self.value['factors']['variance_conditions'].append(dict(identifier='layout_variance', profile_path='profile.json', profile_hash=fingerprint(profile)))
        path, plan = self.plan()
        result = run_experiment(path, client=CollaborationMockClient(), verifier=stub_verifier)
        for record in self.records(result):
            context = json.loads((Path(record['run_dir']) / 'variance_context.json').read_text())
            self.assertEqual(context['profile_hash'], record['candidate']['variance']['profile_hash'])
            prompt = json.loads((Path(record['run_dir']) / 'planner_request.json').read_text())
            self.assertEqual('EXPERIMENTAL VARIANCE' in prompt['system'], context['condition'] != 'control')

    def test_dry_run_and_prefix_have_no_inference_or_output(self):
        path, plan = self.plan()
        with patch('agentic_web.batch.build_graph', side_effect=AssertionError('inference')):
            result = run_experiment(path, max_candidates=1, dry_run=True)
        self.assertEqual(result['candidate_ids'], [plan['candidates'][0]['candidate_id']])
        self.assertFalse((path.parent / 'execution').exists())
        with self.assertRaises(ValueError):
            run_experiment(path, max_candidates=0)

    def test_changed_spec_and_tampered_candidate_rejected_before_inference(self):
        path, plan = self.plan()
        saved = path.read_bytes()
        raw = json.loads(saved)
        raw['candidates'][0]['seed'] += 99
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'Saved experiment plan'):
            run_experiment(path, client=CollaborationMockClient())
        path.write_bytes(saved)
        task = json.loads((self.root / 'task.json').read_text())
        task['description'] += ' changed'
        (self.root / 'task.json').write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, 'Specification hash'):
            run_experiment(path, client=CollaborationMockClient())
        self.assertFalse((path.parent / 'execution').exists())

    def test_unbound_variance_rejected_before_batch(self):
        self.value['factors']['variance_conditions'].append(dict(identifier='layout_variance'))
        path, _ = self.plan()
        with self.assertRaisesRegex(ValueError, 'requires a variance profile'):
            run_experiment(path)
        self.assertFalse((path.parent / 'execution').exists())

    def test_inference_failure_continues(self):
        path, _ = self.plan()
        class FailOnce(CollaborationMockClient):
            failed = False
            def complete(self, *args, **kwargs):
                if not self.failed:
                    self.failed = True
                    raise ConnectionError('mock transport failure')
                return super().complete(*args, **kwargs)
        result = run_experiment(path, client=FailOnce(), verifier=stub_verifier)
        records = self.records(result)
        self.assertEqual(records[0]['execution_status'], 'model_inference_failure')
        self.assertEqual(records[0]['failure_stage'], 'planner')
        self.assertEqual(records[1]['execution_status'], 'generation_completed_verification_passed')

    def test_generated_but_rejected_artifact_retained(self):
        path, _ = self.plan()
        def reject(path, task, output):
            report = stub_verifier(path, task, output)
            report['status'] = 'failed'
            report['errors'] = ['mock failure']
            return report
        result = run_experiment(path, client=CollaborationMockClient(), verifier=reject)
        for record in self.records(result):
            self.assertEqual(record['execution_status'], 'generation_completed_verification_failed')
            self.assertTrue(record['generation_completed'])
            self.assertTrue(Path(record['artifact']['path']).is_file())

    def test_mid_batch_invariant_change_aborts_and_retains_finished_record(self):
        path, _ = self.plan()
        def change(path, task, output):
            result = stub_verifier(path, task, output)
            frozen = json.loads((self.root / 'task.json').read_text())
            frozen['description'] += ' changed'
            (self.root / 'task.json').write_text(json.dumps(frozen))
            return result
        result = run_experiment(path, client=CollaborationMockClient(), verifier=change)
        self.assertEqual(result['status'], 'aborted')
        self.assertEqual(len(result['outcomes']), 1)
        self.assertTrue(self.records(result)[0]['artifact']['available'])

    def test_existing_execution_protected(self):
        path, _ = self.plan()
        run_experiment(path, max_candidates=1, client=CollaborationMockClient(), verifier=stub_verifier)
        before = {str(p): p.read_bytes() for p in (path.parent / 'execution').rglob('*') if p.is_file()}
        with self.assertRaisesRegex(FileExistsError, 'never resumes or overwrites'):
            run_experiment(path, client=CollaborationMockClient())
        self.assertEqual(before, {str(p): p.read_bytes() for p in (path.parent / 'execution').rglob('*') if p.is_file()})

    def test_unexpected_graph_exception_closes_owned_session_and_continues(self):
        path, _ = self.plan()
        closed = []
        class Session:
            def close(self):
                closed.append(True)
        def factory(*args, **kwargs):
            class Broken:
                def invoke(self, inputs):
                    kwargs['session_observer'](Session())
                    raise RuntimeError('Unexpected graph failure')
            return Broken()
        with patch('agentic_web.batch.build_graph', side_effect=factory):
            result = run_experiment(path)
        self.assertEqual(len(closed), 2)
        self.assertEqual([r['execution_status'] for r in self.records(result)], ['workflow_agent_failure'] * 2)

    def test_incomplete_and_resource_classification(self):
        directory = self.root / 'runs/run'
        directory.mkdir(parents=True)
        (directory / 'metadata.json').write_text('{}')
        (directory / 'implementer_metrics.json').write_text(json.dumps({'role': 'implementer', 'finish_reason': 'length'}))
        self.assertEqual(outcome({}, directory.parent)['execution_status'], 'generation_incomplete')
        self.assertEqual(outcome({}, directory.parent, 'ResourceAdmissionError: insufficient memory')['execution_status'], 'resource_failure')

    def test_cli_dry_run(self):
        from agentic_web.__main__ import main
        path, _ = self.plan()
        with patch('sys.argv', ['agentic_web', 'experiment', 'run', '--plan', str(path), '--max-candidates', '1', '--dry-run']), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.getvalue())['selected_count'], 1)

    def test_frozen_decomposition_reused_without_analyst_or_decomposer_calls(self):
        from agentic_web.schema import load_task
        from agentic_web.workflow import contract_hash
        from agentic_web.decomposition import conditions, validate_graph
        from test_decomposition import tasks_for
        task = load_task(self.root / 'task.json')
        registry = conditions(task, None)
        graph, order = validate_graph(tasks_for(task), registry)
        structure = dict(specification_hash=contract_hash(task), requirement_registry=registry,
            tasks=graph.model_dump(mode='json')['tasks'], dependency_order=order)
        record = dict(decomposition_version='1.0', **structure, structure_hash=fingerprint(structure), ready_for_planning=True)
        frozen = self.root / 'decomposition.json'
        frozen.write_text(json.dumps(record))
        before = frozen.read_bytes()
        self.value['tasks'][0].update(decomposition_path=frozen.name, decomposition_hash=record['structure_hash'])
        path, _ = self.plan()
        client = CollaborationMockClient()
        result = run_experiment(path, client=client, verifier=stub_verifier)
        self.assertEqual(frozen.read_bytes(), before)
        self.assertNotIn('requirements_analyst', client.roles)
        self.assertNotIn('task_decomposition', client.roles)
        for item in self.records(result):
            self.assertTrue(item['artifact']['available'])
            self.assertEqual(json.loads((Path(item['run_dir']) / 'task_decomposition.json').read_text()), record)


if __name__ == '__main__':
    unittest.main()
