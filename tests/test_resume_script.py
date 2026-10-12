"""Exercise the PowerShell launcher with mocked intake and dry-run batches only."""
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
class ResumeExperimentScriptTests(unittest.TestCase):
    def test_launcher_and_resume_without_live_inference(self):
        with tempfile.TemporaryDirectory(prefix='resume script ') as tmp:
            root = Path(tmp)
            script = root / 'run_resume_experiment.ps1'
            shutil.copyfile(ROOT / script.name, script)
            driver = root / 'mock_driver.py'
            driver.write_text('''import os, sys
from pathlib import Path
sys.path[:0] = [os.environ['TEST_REPO'], str(Path(os.environ['TEST_REPO']) / 'tests')]
args = sys.argv[1:]
if args[0] == '-c':
    code = args[1]
    sys.argv = ['-c', *args[2:]]
    exec(compile(code, '<launcher>', 'exec'))
elif args[:3] == ['-m', 'agentic_web', 'intake']:
    if os.environ.get('TEST_FAIL_INTAKE'):
        print('Simulated intake failure', file=sys.stderr)
        sys.exit(1)
    from agentic_web.intake import build_intake_graph
    from agentic_web.decomposition import decompose
    from test_intake_review import FeedbackClient
    from test_decomposition import DecompositionClient
    def flag(name): return args[args.index(name) + 1]
    result = build_intake_graph(FeedbackClient()).invoke({'prompt': flag('--prompt'),
        'output_file': flag('--output'), 'review_policy': {'mode': 'feedback'}})
    assert result['ready_for_generation'], result
    result = decompose({'task_path': flag('--output'), 'output_file': flag('--decomposition-output')}, DecompositionClient())
    assert result['ready_for_planning'], result
else:
    assert args[:3] == ['-m', 'agentic_web', 'experiment'], args
    if args[3] in ('run', 'resume') and '--dry-run' not in args:
        args.append('--dry-run')
    from agentic_web.__main__ import main
    sys.argv = ['agentic_web', *args[2:]]
    sys.exit(main())
''', encoding='utf-8')
            wrapper = root / 'python_mock.ps1'
            wrapper.write_text('& $env:TEST_PYTHON $env:TEST_DRIVER @args\nexit $LASTEXITCODE\n', encoding='utf-8')
            env = {**os.environ, 'TEST_REPO': str(ROOT), 'TEST_PYTHON': sys.executable, 'TEST_DRIVER': str(driver)}
            command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                '-Python', str(wrapper), '-ExperimentId', 'test-six', '-Prompt', PROMPT,
                '-Document', str(ROOT / 'tests/fixtures/documents/resume.pdf'),
                '-TemplateConfig', str(ROOT / 'experiments/resume-nemotron-qwen-6-v1.json'),
                '-ResourceConfig', str(ROOT / 'resources.example.json'),
                '-ReasoningPolicy', str(ROOT / 'reasoning.example.json'),
                '-ReviewPolicy', str(ROOT / 'review.example.json')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            directory = root / 'runs/experiments/test-six'
            plan = json.loads((directory / 'experiment_plan.json').read_text())
            self.assertEqual(plan['planned_candidate_count'], 6)
            self.assertEqual([v['candidate_count'] for v in plan['factor_summary']['variance_condition']], [2, 2, 2])
            self.assertFalse((directory / 'execution').exists())
            result = subprocess.run(command[:8] + ['-ResumePlan', str(directory / 'experiment_plan.json')],
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            failed = command.copy()
            failed[failed.index('-ExperimentId') + 1] = 'test-failed'
            result = subprocess.run(failed, env={**env, 'TEST_FAIL_INTAKE': '1'}, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'runs/experiments/test-failed/experiment_plan.json').exists())


if __name__ == '__main__':
    unittest.main()
