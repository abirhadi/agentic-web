"""B4 recovery from record boundaries, with real local locks and mocked inference."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from agentic_web.batch import run_experiment
from agentic_web.checkpoint import execution_lock, progress, atomic_json
from collaboration_mock_client import CollaborationMockClient
import test_batch
from test_framework import stub_verifier


class CheckpointTests(unittest.TestCase):
    setUp = test_batch.BatchTests.setUp

    def plan(self):
        path, _ = test_batch.BatchTests.plan(self)
        return path, json.loads(path.read_text())

    def run_plan(self, path, **kwargs):
        return run_experiment(path, client=CollaborationMockClient(), verifier=stub_verifier, **kwargs)

    def interrupt(self, path):
        closed = []
        class Session:
            def close(self):
                closed.append(True)
        def factory(*args, **kwargs):
            class Interrupted:
                def invoke(self, inputs):
                    kwargs['session_observer'](Session())
                    partial = Path(inputs['output_root']) / 'partial'
                    partial.mkdir(parents=True)
                    (partial / 'index.html').write_text('<html>partial</html>')
                    raise KeyboardInterrupt()
            return Interrupted()
        with patch('agentic_web.batch.build_graph', side_effect=factory), self.assertRaises(KeyboardInterrupt):
            self.run_plan(path)
        self.assertEqual(closed, [True])

    def test_prefix_resume_and_repeated_noop(self):
        path, plan = self.plan()
        initial = self.run_plan(path, max_candidates=1)
        original = Path(initial['outcomes'][0]['record']).read_bytes()
        final = self.run_plan(path, resume=True, max_new_candidates=1)
        self.assertEqual(final['progress']['counts'], dict(not_started=0, interrupted=0, completed=2, failed=0))
        self.assertEqual(final['progress']['execution_attempts'], 2)
        self.assertEqual(Path(initial['outcomes'][0]['record']).read_bytes(), original)
        with patch('agentic_web.batch.build_graph', side_effect=AssertionError('regeneration')):
            repeat = self.run_plan(path, resume=True)
        self.assertEqual(repeat['selected_count'], 0)
        self.assertEqual(repeat['progress']['execution_attempts'], 2)

    def test_interrupted_not_retried_without_option(self):
        path, plan = self.plan()
        self.interrupt(path)
        state = progress(plan, path.parent / 'execution')
        self.assertEqual(state['counts']['interrupted'], 1)
        self.assertEqual(state['counts']['completed'], 0)
        result = self.run_plan(path, resume=True)
        self.assertEqual(result['selected_count'], 1)
        self.assertEqual(result['progress']['counts']['interrupted'], 1)
        self.assertEqual(result['progress']['counts']['completed'], 1)

    def test_explicit_retry_preserves_partial_attempt(self):
        path, plan = self.plan()
        self.interrupt(path)
        root = path.parent / 'execution' / plan['candidates'][0]['candidate_id']
        before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        result = self.run_plan(path, resume=True, retry_interrupted=True, max_new_candidates=1)
        record = json.loads(Path(result['outcomes'][0]['record']).read_text())
        self.assertEqual(record['attempt_number'], 2)
        self.assertTrue(record['operational_retry'])
        self.assertEqual(record['candidate_id'], plan['candidates'][0]['candidate_id'])
        self.assertIn('attempt-0002', record['run_dir'])
        for file, content in before.items():
            self.assertEqual(Path(file).read_bytes(), content)
        self.assertEqual(result['progress']['total_planned_candidates'], 2)
        self.assertEqual(result['progress']['execution_attempts'], 2)
        self.assertEqual(result['progress']['counts']['completed'], 1)

    def test_failed_verification_and_inference_are_terminal(self):
        path, plan = self.plan()
        def reject(path, task, output):
            report = stub_verifier(path, task, output)
            report['status'] = 'failed'
            return report
        run_experiment(path, max_candidates=1, client=CollaborationMockClient(), verifier=reject)
        class Failed(CollaborationMockClient):
            def complete(self, *args, **kwargs):
                raise ConnectionError('failed inference')
        result = run_experiment(path, resume=True, client=Failed(), verifier=stub_verifier)
        self.assertEqual(result['progress']['counts']['completed'], 1)
        self.assertEqual(result['progress']['counts']['failed'], 1)
        with patch('agentic_web.batch.build_graph', side_effect=AssertionError('retry')):
            final = self.run_plan(path, resume=True, retry_interrupted=True)
        self.assertEqual(final['selected_count'], 0)

    def test_invalid_terminal_record_is_interrupted_not_html_success(self):
        path, plan = self.plan()
        self.interrupt(path)
        result_file = path.parent / 'execution' / plan['candidates'][0]['candidate_id'] / 'result.json'
        for content in ['{"execution_status":', '{"execution_status":"generation_completed_verification_passed"}']:
            result_file.write_text(content)
            self.assertEqual(progress(plan, path.parent / 'execution')['counts']['interrupted'], 1)
        result = self.run_plan(path, resume=True, retry_interrupted=True)
        self.assertEqual(result_file.read_text(), content)
        self.assertEqual(result['progress']['execution_attempts'], 3)

    def test_corrupt_summary_ignored_and_legacy_b3_records_supported(self):
        path, plan = self.plan()
        initial = self.run_plan(path, max_candidates=1)
        record_file = Path(initial['outcomes'][0]['record'])
        record = json.loads(record_file.read_text())
        # Simulate B3 records before B4 attempt metadata existed.
        for filename in ('started.json', 'result.json'):
            p = record_file.parent / filename
            data = json.loads(p.read_text())
            data.pop('attempt_number')
            data.pop('operational_retry')
            p.write_text(json.dumps(data))
        (path.parent / 'execution/progress.json').write_text('corrupt')
        result = self.run_plan(path, resume=True)
        self.assertEqual(result['progress']['counts']['completed'], 2)

    def test_changed_frozen_input_resume_rejected(self):
        path, _ = self.plan()
        self.run_plan(path, max_candidates=1)
        task = json.loads((self.root / 'task.json').read_text())
        task['description'] += ' changed'
        (self.root / 'task.json').write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, 'Specification hash'):
            self.run_plan(path, resume=True)

    def test_concurrent_writer_rejected_and_abandoned_os_lock_released(self):
        path, _ = self.plan()
        script = ('import sys,time; from agentic_web.checkpoint import execution_lock; '
                  'lock=execution_lock(sys.argv[1]); lock.__enter__(); print("locked",flush=True); time.sleep(60)')
        child = subprocess.Popen([sys.executable, '-c', script, str(path.parent)], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            with self.assertRaisesRegex(RuntimeError, 'Another writer'):
                self.run_plan(path)
        finally:
            child.kill()
            child.wait(timeout=10)
            child.stdout.close()
        result = self.run_plan(path, resume=True, max_new_candidates=1)
        self.assertEqual(result['progress']['counts']['completed'], 1)

    def test_atomic_summary_failure_preserves_previous_json(self):
        file = self.root / 'progress.json'
        atomic_json(file, {'old': True})
        with patch('agentic_web.checkpoint.os.replace', side_effect=OSError('interrupted replace')), self.assertRaises(OSError):
            atomic_json(file, {'new': True})
        self.assertEqual(json.loads(file.read_text()), {'old': True})

    def test_36_candidate_incremental_resume_and_1000_progress(self):
        self.value['factors'].update(seeds=list(range(18)))
        path, plan = self.plan()
        self.run_plan(path, max_candidates=1)
        result = self.run_plan(path, resume=True, max_new_candidates=2)
        self.assertEqual(result['progress']['total_planned_candidates'], 36)
        self.assertEqual(result['progress']['counts']['completed'], 3)
        self.assertEqual(result['progress']['counts']['not_started'], 33)
        # Read-only progress uses the same mechanism for 1,000 validated candidates.
        from agentic_web.experiments import ExperimentConfig, build_plan
        self.value['factors']['seeds'] = list(range(500))
        big = build_plan(ExperimentConfig.model_validate(self.value), self.root)
        self.assertEqual(progress(big, self.root / 'unused')['counts']['not_started'], 1000)

    def test_resume_cli_and_limit(self):
        from agentic_web.__main__ import main
        path, _ = self.plan()
        self.run_plan(path, max_candidates=1)
        with patch('sys.argv', ['agentic_web', 'experiment', 'resume', '--plan', str(path), '--max-new-candidates', '1', '--dry-run']), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.getvalue())['selected_count'], 1)


if __name__ == '__main__':
    unittest.main()
