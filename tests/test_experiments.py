"""B1 is read-only planning until the exclusive plan write; no live inference."""
from contextlib import redirect_stdout, ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_web.experiments import (ExperimentConfig, build_plan, candidate_matrix,
    expected_counts, factor_summary, read_config, save_plan)
from agentic_web.provenance import fingerprint
from agentic_web.schema import load_task
from agentic_web.workflow import contract_hash
from agentic_web.decomposition import conditions, validate_graph
from test_decomposition import tasks_for

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    task = load_task(ROOT / 'tasks/todo.json')
    return {"experiment_id": "test-pilot", "experiment_version": "1.0", "description": "A reproducible matrix",
        "output_root": "output", "tasks": [{"name": "todo", "specification_path": "task.json",
            "specification_hash": contract_hash(task)}],
        "model_configurations": {
            "small": {"workflow": "baseline", "roles": {role: {"model": "model-small", "family": "family", "parameter_count": 4000000000}
                for role in ("planner", "implementer")}},
            "large": {"workflow": "collaborative", "roles": {role: {"model": "model-large", "family": "family", "parameter_count": 8000000000}
                for role in ("planner", "implementer", "ux", "architect", "coordinator", "qa")}}},
        "factors": {"model_configurations": ["small", "large"], "temperatures": [0.4, 0.8], "top_ps": [0.95],
            "seeds": [1, 2, 3], "variance_conditions": [{"identifier": x} for x in ("control", "layout", "architecture")]}}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = load_task(ROOT / 'tasks/todo.json')
        (self.root / 'task.json').write_text(self.task.model_dump_json(), encoding='utf-8')
        self.value = configuration()

    def config(self):
        return ExperimentConfig.model_validate(self.value)

    def test_factorial_count_paired_seeds_and_metadata(self):
        plan = build_plan(self.config(), self.root)
        self.assertEqual((plan['full_factorial_count'], plan['planned_candidate_count']), (36, 36))
        groups = {}
        for item in plan['candidates']:
            key = (item['specification_hash'], item['model_configuration_name'], item['temperature'], item['top_p'], item['seed'], item['replicate'])
            groups.setdefault(key, []).append(item['variance']['identifier'])
            model = item['model_configuration']['roles']['implementer']
            self.assertEqual(model['family'], 'family')
            self.assertEqual(model['parameter_count'], 4000000000 if item['model_configuration_name'] == 'small' else 8000000000)
            self.assertIsNone(model['quantization'])
            self.assertIsNone(model['revision'])
            self.assertEqual(item['role_sampling']['implementer']['seed'], item['seed'] + 1)
            self.assertEqual(item['role_sampling']['planner']['seed'], item['seed'])
            if item['model_configuration_name'] == 'large':
                self.assertEqual(item['role_sampling']['qa']['seed'], item['seed'] + 5)
            self.assertTrue(all(role['temperature'] == item['temperature'] and role['top_p'] == item['top_p'] for role in item['role_sampling'].values()))
        self.assertTrue(all(set(conditions) == {'control', 'layout', 'architecture'} for conditions in groups.values()))
        self.assertNotIn('supporting_context', json.dumps(plan))

    def test_stable_ids_key_and_factor_order_and_paths(self):
        original = candidate_matrix(self.config())
        self.value = json.loads(json.dumps(self.value, sort_keys=True))
        self.value['tasks'][0]['specification_path'] = '/another-machine/task.json'
        self.value['output_root'] = '/another-machine/output'
        for key in ('seeds', 'temperatures', 'model_configurations', 'variance_conditions'):
            self.value['factors'][key].reverse()
        changed = candidate_matrix(self.config())
        self.assertEqual([c['candidate_id'] for c in original], [c['candidate_id'] for c in changed])

    def test_replicates_and_multiple_tasks(self):
        self.value['factors']['replicates'] = 2
        second = load_task(ROOT / 'tasks/catalog.json') if (ROOT / 'tasks/catalog.json').exists() else load_task(ROOT / 'tasks/eventhub.json')
        (self.root / 'second.json').write_text(second.model_dump_json(), encoding='utf-8')
        self.value['tasks'].append(dict(name='second', specification_path='second.json', specification_hash=contract_hash(second)))
        plan = build_plan(self.config(), self.root)
        self.assertEqual(len(plan['candidates']), 144)
        self.assertEqual(len({c['candidate_id'] for c in plan['candidates']}), 144)
        self.assertEqual({c['replicate'] for c in plan['candidates']}, {1, 2})

    def test_duplicate_levels_aliases_and_tasks_rejected(self):
        mutations = [lambda v: v['factors']['seeds'].append(1),
            lambda v: v['factors']['temperatures'].append(0.4),
            lambda v: v['factors']['top_ps'].append(0.95),
            lambda v: v['factors']['model_configurations'].append('small'),
            lambda v: v['factors']['variance_conditions'].append({'identifier': 'control'}),
            lambda v: v['tasks'].append({**v['tasks'][0], 'name': 'alias'}),
            lambda v: v['model_configurations'].update(large=deepcopy(v['model_configurations']['small']))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = deepcopy(self.value)
                mutate(value)
                with self.assertRaises(ValueError):
                    ExperimentConfig.model_validate(value)

    def test_configuration_validation_and_workflow_roles(self):
        mutations = [lambda v: v.update(schema_version='2.0'), lambda v: v.update(experiment_id='../escape'),
            lambda v: v['factors'].update(model_configurations=['unknown']),
            lambda v: v['factors'].update(temperatures=[3]), lambda v: v['factors'].update(top_ps=[0]),
            lambda v: v['factors'].update(seeds=[True]), lambda v: v['factors'].update(replicates=0),
            lambda v: v.update(planning={'candidate_limit': 0}), lambda v: v.update(planning={'strategy': 'random'}),
            lambda v: v['model_configurations']['small']['roles'].update(qa={'model': 'extra'}),
            lambda v: v['model_configurations']['large']['roles'].pop('qa'),
            lambda v: v['model_configurations']['small'].update(workflow='other'),
            lambda v: v['model_configurations']['small'].update(review_max_tokens=10),
            lambda v: v['model_configurations']['small']['roles']['planner'].update(parameter_count=-1),
            lambda v: v['tasks'][0].update(decomposition_path='missing.json'),
            lambda v: v['factors']['variance_conditions'][0].update(profile_path='profile.json')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = deepcopy(self.value)
                mutate(value)
                with self.assertRaises(ValueError):
                    ExperimentConfig.model_validate(value)

    def test_no_metadata_inference_and_explicit_token_limits(self):
        for identity in self.value['model_configurations']['small']['roles'].values():
            identity.clear()
            identity['model'] = 'a-family-999b-q8'
        self.value['model_configurations']['small']['max_tokens'] = 456
        small = next(c for c in candidate_matrix(self.config()) if c['model_configuration_name'] == 'small')
        self.assertIsNone(small['model_configuration']['roles']['planner']['parameter_count'])
        self.assertIsNone(small['model_configuration']['roles']['planner']['family'])
        self.assertEqual(small['role_sampling']['implementer']['max_tokens'], 456)
        self.assertIsNone(small['role_sampling']['planner']['max_tokens'])

    def test_specification_hash_mismatch_and_missing_file(self):
        self.value['tasks'][0]['specification_hash'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Specification hash'):
            build_plan(self.config(), self.root)
        self.value['tasks'][0]['specification_path'] = 'missing.json'
        with self.assertRaises(FileNotFoundError):
            build_plan(self.config(), self.root)
        self.assertFalse((self.root / 'output').exists())

    def decomposition(self):
        registry = conditions(self.task, None)
        graph, order = validate_graph(tasks_for(self.task), registry)
        structure = dict(specification_hash=contract_hash(self.task), requirement_registry=registry,
                         tasks=graph.model_dump(mode='json')['tasks'], dependency_order=order)
        return dict(decomposition_version='1.0', **structure, structure_hash=fingerprint(structure), ready_for_planning=True)

    def test_decomposition_validation_and_tampering(self):
        record = self.decomposition()
        path = self.root / 'decomposition.json'
        path.write_text(json.dumps(record), encoding='utf-8')
        self.value['tasks'][0].update(decomposition_path=path.name, decomposition_hash=record['structure_hash'])
        self.assertEqual(build_plan(self.config(), self.root)['planned_candidate_count'], 36)
        self.value['tasks'][0]['decomposition_hash'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Decomposition hash'):
            build_plan(self.config(), self.root)
        self.value['tasks'][0]['decomposition_hash'] = record['structure_hash']
        for field, value in [('specification_hash', '0' * 64), ('structure_hash', '0' * 64), ('ready_for_planning', False)]:
            with self.subTest(field=field):
                path.write_text(json.dumps({**record, field: value}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    build_plan(self.config(), self.root)

    def test_profile_reference_and_hash_are_declarative(self):
        profile = {'future_directives': ['Reserved for B2']}
        path = self.root / 'profile.json'
        path.write_text(json.dumps(profile), encoding='utf-8')
        self.value['factors']['variance_conditions'][1].update(profile_path=path.name, profile_hash=fingerprint(profile))
        plan = build_plan(self.config(), self.root)
        self.assertNotIn('Reserved for B2', json.dumps(plan))
        path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'profile hash'):
            build_plan(self.config(), self.root)

    def test_subset_determinism_and_actual_coverage(self):
        self.value['planning'] = dict(candidate_limit=7, selection_seed=19)
        first = build_plan(self.config(), self.root)
        self.assertEqual(first, build_plan(self.config(), self.root))
        self.assertEqual(first['planned_candidate_count'], 7)
        self.assertEqual(first['full_factorial_count'], 36)
        for levels in first['factor_summary'].values():
            self.assertEqual(sum(v['candidate_count'] for v in levels), 7)
        self.value['planning']['selection_seed'] = 20
        self.assertNotEqual(first['candidates'], build_plan(self.config(), self.root)['candidates'])
        self.value['planning']['candidate_limit'] = 1
        summary = build_plan(self.config(), self.root)['factor_summary']
        self.assertTrue(any(level['candidate_count'] == 0 for level in summary['seed']))

    def test_id_changes_with_effective_factors(self):
        original = {c['candidate_id'] for c in candidate_matrix(self.config())}
        self.value['factors']['seeds'] = [101, 102, 103]
        self.assertFalse(original & {c['candidate_id'] for c in candidate_matrix(self.config())})

    def test_revision_and_quantization_are_identity_factors(self):
        original = {c['candidate_id'] for c in candidate_matrix(self.config()) if c['model_configuration_name'] == 'small'}
        self.value['model_configurations']['small']['roles']['implementer'].update(revision='revision-2', quantization='Q8_0')
        changed = {c['candidate_id'] for c in candidate_matrix(self.config()) if c['model_configuration_name'] == 'small'}
        self.assertFalse(original & changed)

    def test_configuration_hash_is_key_order_independent(self):
        first = build_plan(self.config(), self.root)
        self.value = json.loads(json.dumps(self.value, sort_keys=True))
        self.assertEqual(first['configuration_hash'], build_plan(self.config(), self.root)['configuration_hash'])

    def test_exclusive_output_and_identical_replanning(self):
        plan = build_plan(self.config(), self.root)
        path = save_plan(plan, self.root / 'config.json')
        original = path.read_bytes()
        self.assertEqual(save_plan(plan, self.root / 'config.json'), path)
        self.assertEqual(original, path.read_bytes())
        self.value['description'] = 'Changed configuration'
        with self.assertRaisesRegex(ValueError, 'Existing experiment plan differs'):
            save_plan(build_plan(self.config(), self.root), self.root / 'config.json')
        self.assertEqual(original, path.read_bytes())

    def test_cli_and_1000_candidates_have_no_execution_side_effects(self):
        from agentic_web.__main__ import main
        self.value['factors']['seeds'] = list(range(125))
        self.value['factors']['variance_conditions'] = self.value['factors']['variance_conditions'][:2]
        path = self.root / 'config.json'
        path.write_text(json.dumps(self.value), encoding='utf-8')
        task_bytes = (self.root / 'task.json').read_bytes()
        with ExitStack() as stack:
            for target in ('agentic_web.client.LMStudioClient.complete', 'agentic_web.client.LMStudioClient.models',
                    'agentic_web.hardware.discover_hardware', 'agentic_web.resource_scheduler.local_scheduler',
                    'agentic_web.lmstudio_placement.LMStudioPlacementBackend.load',
                    'langgraph.graph.state.CompiledStateGraph.invoke', 'playwright.sync_api.sync_playwright'):
                stack.enter_context(patch(target, side_effect=AssertionError('Planning executed ' + target)))
            stack.enter_context(patch('sys.argv', ['agentic_web', 'experiment', 'plan', '--config', str(path)]))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(main(), 0)
        saved = json.loads((self.root / 'output/test-pilot/experiment_plan.json').read_text())
        self.assertEqual(saved['planned_candidate_count'], 1000)
        self.assertEqual(len(saved['candidates']), 1000)
        self.assertEqual(task_bytes, (self.root / 'task.json').read_bytes())

    def test_checked_in_schema_matches_model(self):
        self.assertEqual(json.loads((ROOT / 'experiments/experiment.schema.json').read_text()), ExperimentConfig.model_json_schema())


if __name__ == '__main__':
    unittest.main()
