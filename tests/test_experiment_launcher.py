"""Generic PowerShell launch: mocked model calls, real planning, no live batch."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from mock_client import ROOT
from test_intake import PROMPT


@unittest.skipUnless(os.name == 'nt' and shutil.which('powershell'), 'Windows PowerShell required')
class ExperimentLauncherTests(unittest.TestCase):
    def test_counts_optional_document_model_settings_and_resume(self):
        with tempfile.TemporaryDirectory(prefix='generic launcher ') as tmp:
            root = Path(tmp)
            script = root / 'run_experiment.ps1'
            shutil.copyfile(ROOT / script.name, script)
            shutil.copyfile(ROOT / 'handoff.example.json', root / 'handoff.example.json')
            shutil.copytree(ROOT / 'experiments/profiles', root / 'experiments/profiles')
            model = root / 'models.json'
            model.write_text(json.dumps({'model': 'test-builder', 'planner_model': 'test-planner',
                'qa_model': 'test-reviewer', 'seed': 77, 'temperature': .2, 'top_p': .8,
                'base_url': 'http://127.0.0.1:19999/v1'}))
            off_policy = root / 'off.json'
            off_policy.write_text(json.dumps({'mode': 'off'}))
            driver = root / 'mock_driver.py'
            driver.write_text('''import os, sys
from pathlib import Path
sys.path[:0] = [os.environ['TEST_REPO'], str(Path(os.environ['TEST_REPO']) / 'tests')]
args = sys.argv[1:]
if args[0] == '-c':
    import agentic_web.intake as intake
    from test_intake_review import FeedbackClient
    from test_decomposition import DecompositionClient
    import json
    class Combined(FeedbackClient):
        def complete(self, system, user, config, **kwargs):
            if json.loads(user).get('role') == 'task_decomposition':
                raw, metrics = DecompositionClient().complete(system, user, config, **kwargs)
                if kwargs['response_schema'].__name__ == 'DecomposerHandoff':
                    raw = json.dumps({'outcome': 'accepted', 'decomposition': json.loads(raw), 'findings': []})
                return raw, metrics
            return super().complete(system, user, config, **kwargs)
    original = intake.build_intake_graph
    class MockGraph:
        def invoke(self, inputs):
            if os.environ.get('TEST_FAIL_INTAKE'):
                return {'error': 'Simulated intake failure', 'status': 'failed'}
            assert inputs['model_config']['seed'] == 77
            assert inputs['model_config']['qa_model'] == 'test-reviewer'
            inputs.pop('resource_config', None)
            return original(Combined()).invoke(inputs)
    intake.build_intake_graph = lambda: MockGraph()
    code = args[1]
    sys.argv = ['-c', *args[2:]]
    exec(compile(code, '<launcher>', 'exec'))
else:
    assert args[:3] == ['-m', 'agentic_web', 'experiment'], args
    if args[3] in ('run', 'resume'):
        assert args[args.index('--base-url') + 1] == 'http://127.0.0.1:19999/v1'
        if '--dry-run' not in args: args.append('--dry-run')
    from agentic_web.__main__ import main
    sys.argv = ['agentic_web', *args[2:]]
    sys.exit(main())
''', encoding='utf-8')
            wrapper = root / 'python_mock.ps1'
            wrapper.write_text('& $env:TEST_PYTHON $env:TEST_DRIVER @args\nexit $LASTEXITCODE\n', encoding='utf-8')
            env = {**os.environ, 'TEST_REPO': str(ROOT), 'TEST_PYTHON': sys.executable, 'TEST_DRIVER': str(driver)}
            base = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), '-Python', str(wrapper)]
            command = base + ['-Prompt', PROMPT, '-ModelConfig', str(model),
                '-ResourceConfig', str(ROOT / 'resources.example.json'),
                '-ReasoningPolicy', str(ROOT / 'reasoning.example.json')]
            for count, document, mode in ((1, False, 'collaborative'), (3, True, 'collaborative'), (1, False, 'off'), (1, False, 'review')):
                with self.subTest(count=count, document=document, mode=mode):
                    identifier = f'test-{count}' if mode == 'collaborative' else f'test-{mode}'
                    current = command + ['-Candidates', str(count), '-ExperimentId', identifier]
                    if mode == 'off':
                        current += ['-HandoffPolicy', str(off_policy)]
                    elif mode == 'review':
                        current += ['-ReviewPolicy', str(ROOT / 'review.example.json'), '-ManualVerification', 'on', '-MaxRepairAttempts', '0']
                    if document:
                        current += ['-Document', str(ROOT / 'tests/fixtures/documents/resume.pdf')]
                    result = subprocess.run(current, env=env, capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    directory = root / f'runs/experiments/{identifier}'
                    plan = json.loads((directory / 'experiment_plan.json').read_text())
                    self.assertEqual(plan['planned_candidate_count'], 3 * count)
                    self.assertEqual([v['candidate_count'] for v in plan['factor_summary']['variance_condition']], [count] * 3)
                    self.assertEqual(plan['configuration']['factors']['seeds'], list(range(78, 78 + count)))
                    self.assertEqual(plan['configuration']['factors']['temperatures'], [.2])
                    agents = plan['configuration']['model_configurations']['selected']
                    self.assertEqual(agents['manual_verification'], 'on' if mode == 'review' else 'off')
                    self.assertEqual(agents['max_repair_attempts'], 0 if mode == 'review' else 2)
                    self.assertIn('verification_readiness', plan)
                    task = json.loads((directory / 'task.json').read_text())
                    self.assertEqual(task['original_requirements'], PROMPT)
                    analyses = list(directory.glob('task.intake/*/analysis.json'))
                    self.assertEqual(len(analyses), 1)
                    analysis = json.loads(analyses[0].read_text())
                    if mode == 'collaborative':
                        self.assertEqual(analysis['handoff']['status'], 'accepted')
                        self.assertEqual(len(analysis['handoff']['calls']), 2)
                        self.assertNotIn('analysis', analysis['handoff'])
                    self.assertFalse(list(directory.glob('decomposition.decompose')))
                    self.assertFalse(list(directory.glob('task.intake/*/task.json')))
                    self.assertEqual(bool(task.get('supporting_context')), document)
                    self.assertFalse((directory / 'execution').exists())
            # Exercise the actual default identifier, not only explicit lowercase IDs.
            result = subprocess.run(command + ['-Candidates', '1'], env=env,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            default_plans = list((root / 'runs/experiments').glob('website-experiment-*/experiment_plan.json'))
            self.assertEqual(len(default_plans), 1)
            default_plan = json.loads(default_plans[0].read_text())
            self.assertRegex(default_plan['configuration']['experiment_id'], r'^[a-z][a-z0-9_-]*$')
            self.assertEqual(default_plan['planned_candidate_count'], 3)
            for identifier in ('Uppercase', '1starts-with-digit', 'has.dot'):
                result = subprocess.run(command + ['-ExperimentId', identifier], env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ExperimentId must start with a lowercase letter', result.stderr)
                self.assertFalse((root / 'runs/experiments' / identifier).exists())
            result = subprocess.run(base + ['-ResumePlan', str(root / 'runs/experiments/test-3/experiment_plan.json')],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for flags in (['-Candidates', '0'], ['-ExperimentId', 'bad-intake']):
                result = subprocess.run(command + flags, env={**env, 'TEST_FAIL_INTAKE': '1'},
                                        capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'runs/experiments/bad-intake/experiment_plan.json').exists())


if __name__ == '__main__':
    unittest.main()
