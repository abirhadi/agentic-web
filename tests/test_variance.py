"""B2 prompt isolation and immutable contracts, using mocked inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_web.provenance import fingerprint
from agentic_web.variance import VarianceProfile, load_variance_context, condition_prompt
from agentic_web.workflow import build_graph, contract_hash
from agentic_web.schema import load_task
from agentic_web.decomposition import conditions, validate_graph
from collaboration_mock_client import CollaborationMockClient
from test_framework import stub_verifier
from test_decomposition import tasks_for

ROOT = Path(__file__).resolve().parents[1]


class VarianceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = load_task(ROOT / 'tasks/todo.json')
        self.profile = json.loads((ROOT / 'experiments/profiles/layout_exploration.json').read_text())
        self.path = self.root / 'profile.json'
        self.path.write_text(json.dumps(self.profile), encoding='utf-8')

    def invoke(self, workflow='baseline', **extra):
        client = CollaborationMockClient()
        result = build_graph(client, stub_verifier, workflow=workflow).invoke({
            'task_spec': self.task.model_dump(mode='json'), 'output_root': str(self.root / 'runs'),
            'execution_mode': 'mocked', **extra})
        return result, client

    def requests(self, result):
        return {file.stem: json.loads(file.read_text(encoding='utf-8'))
                for file in Path(result['run_dir']).glob('*_request.json')}

    def treatment(self):
        return dict(variance_condition='layout_variance', variance_profile=str(self.path),
                    variance_profile_hash=fingerprint(self.profile))

    def test_control_prompts_unchanged_in_both_workflows(self):
        for workflow in ('baseline', 'collaborative'):
            with self.subTest(workflow=workflow):
                default, _ = self.invoke(workflow)
                control, _ = self.invoke(workflow, variance_condition='control')
                self.assertEqual(self.requests(default), self.requests(control))
                # Compare against the pre-B2 construction with injection disabled.
                with patch('agentic_web.workflow.condition_prompt', side_effect=lambda state, role, system: system), \
                     patch('agentic_web.collaboration.condition_prompt', side_effect=lambda state, role, system: system):
                    original, _ = self.invoke(workflow)
                self.assertEqual(self.requests(default), self.requests(original))
                self.assertTrue(all('EXPERIMENTAL VARIANCE' not in r['system'] for r in self.requests(control).values()))

    def test_treatment_changes_only_system_suffix_and_preserves_verification(self):
        for workflow, roles in [('baseline', ['planner', 'implementer']),
                                ('collaborative', ['planner', 'ux', 'architect', 'coordinator', 'implementer'])]:
            with self.subTest(workflow=workflow):
                control, _ = self.invoke(workflow)
                result, _ = self.invoke(workflow, **self.treatment())
                self.assertEqual(result['status'], control['status'])
                context = result['variance_context']
                self.assertEqual(context['roles'], roles)
                old, new = self.requests(control), self.requests(result)
                for name in old:
                    role = name.removesuffix('_request')
                    expected = deepcopy(old[name])
                    if role in roles:
                        expected['system'] += context['effective_directives'][role]
                        self.assertIn(self.profile['directive'], expected['system'])
                    self.assertEqual(new[name], expected)
                self.assertEqual(result['verification'], control['verification'])
                self.assertEqual(result['model_config'], control['model_config'])
                self.assertEqual(result['task_spec'], self.task.model_dump(mode='json'))
                for file in ['task.json', 'requirements.txt', 'verification_plan.json', 'implementation_contract.json']:
                    self.assertEqual((Path(result['run_dir']) / file).read_bytes(), (Path(control['run_dir']) / file).read_bytes())

    def test_exact_directives_and_profile_preserved_in_ledger(self):
        result, _ = self.invoke('collaborative', **self.treatment())
        directory = Path(result['run_dir'])
        context = json.loads((directory / 'variance_context.json').read_text())
        self.assertEqual(context['profile'], self.profile)
        self.assertEqual(context['profile_hash'], fingerprint(self.profile))
        for role in context['roles']:
            request = json.loads((directory / (role + '_request.json')).read_text())
            self.assertTrue(request['system'].endswith(context['effective_directives'][role]))
        ledger = json.loads((directory / 'ledger/task_loader.json').read_text())
        self.assertEqual(ledger['artifact_hashes']['variance_context.json'], hashlib.sha256((directory / 'variance_context.json').read_bytes()).hexdigest())

    def test_b1_reference_and_hash_semantics(self):
        from agentic_web.experiments import VarianceCondition
        ref = VarianceCondition(identifier='layout_variance', profile_path=self.path.name, profile_hash=fingerprint(self.profile))
        context = load_variance_context(ref, 'collaborative', base=self.root)
        self.path.write_text(json.dumps(self.profile, sort_keys=True, indent=4), encoding='utf-8')
        self.assertEqual(context, load_variance_context(ref.model_dump(), 'collaborative', base=self.root))
        self.assertEqual(context['profile_hash'], ref.profile_hash)
        self.assertIn(self.profile['role_guidance']['ux'], context['effective_directives']['ux'])
        self.assertNotIn(self.profile['role_guidance']['ux'], context['effective_directives']['architect'])

    def test_invalid_treatments_rejected_before_models_or_resources(self):
        for extra in [dict(variance_condition='control', variance_profile=str(self.path)),
                      dict(variance_condition='layout_variance'),
                      dict(variance_profile_hash='0' * 64),
                      {**self.treatment(), 'variance_profile_hash': '0' * 64},
                      {**self.treatment(), 'variance_profile': str(self.root / 'missing.json')}]:
            with self.subTest(extra=extra), patch.object(CollaborationMockClient, 'models', side_effect=AssertionError('model access')):
                result, client = self.invoke('collaborative', **extra)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(client.calls)

    def test_profile_validation_and_extensible_dimensions(self):
        for edit in [dict(schema_version='2.0'), dict(directive=' '), dict(profile_id='../x'),
                     dict(dimensions=[]), dict(dimensions=['duplicate', 'duplicate']),
                     dict(role_guidance={'qa': 'Bias the critic'}), dict(runtime='other')]:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                VarianceProfile.model_validate({**self.profile, **edit})
        self.assertEqual(VarianceProfile.model_validate({**self.profile, 'dimensions': ['future design dimension']}).dimensions, ['future design dimension'])
        self.path.write_text('{invalid json', encoding='utf-8')
        result, client = self.invoke(**self.treatment())
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(client.calls)

    def test_qa_is_excluded_even_with_forged_context(self):
        self.assertEqual(condition_prompt({'variance_context': {'effective_directives': {'qa': 'do not verify'}}}, 'qa', 'original'), 'original')

    def test_unpinned_cli_style_profile_computes_same_hash(self):
        extra = self.treatment()
        extra.pop('variance_profile_hash')
        result, _ = self.invoke(**extra)
        self.assertEqual(result['variance_context']['profile_hash'], fingerprint(self.profile))

    def test_frozen_decomposition_and_file_unchanged(self):
        registry = conditions(self.task, None)
        graph, order = validate_graph(tasks_for(self.task), registry)
        structure = dict(specification_hash=contract_hash(self.task), requirement_registry=registry,
                         tasks=graph.model_dump(mode='json')['tasks'], dependency_order=order)
        record = dict(decomposition_version='1.0', **structure, structure_hash=fingerprint(structure), ready_for_planning=True)
        path = self.root / 'decomposition.json'
        path.write_text(json.dumps(record), encoding='utf-8')
        before = path.read_bytes()
        result, _ = self.invoke(decomposition_path=str(path), **self.treatment())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result['decomposition'], record)
        self.assertEqual(json.loads((Path(result['run_dir']) / 'task_decomposition.json').read_text()), record)

    def test_cli_forwards_flags_without_changing_config(self):
        from agentic_web.__main__ import main
        import io
        from contextlib import redirect_stdout
        args = ['agentic_web', 'generate', '--task', str(ROOT / 'tasks/todo.json'), '--workflow', 'collaborative',
                '--variance-condition', 'layout_variance', '--variance-profile', str(self.path),
                '--variance-profile-hash', fingerprint(self.profile), '--seed', '123', '--temperature', '0.7']
        with patch('sys.argv', args), patch('agentic_web.__main__.collaborative_app.invoke', return_value={'status': 'passed'}) as invoke, redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        inputs = invoke.call_args.args[0]
        self.assertEqual(inputs['variance_profile_hash'], fingerprint(self.profile))
        self.assertEqual(inputs['variance_condition'], 'layout_variance')
        self.assertEqual(inputs['model_config']['seed'], 123)
        self.assertEqual(inputs['model_config']['temperature'], 0.7)


if __name__ == '__main__':
    unittest.main()
