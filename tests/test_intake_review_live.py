"""Explicit opt-in integration probe; never runs in the default regression suite."""
import os
from pathlib import Path
import tempfile
import unittest

from agentic_web.intake import build_intake_graph


@unittest.skipUnless(os.getenv('AGENTIC_WEB_LIVE_INTAKE_REVIEW') == '1', 'explicit live intake review opt-in required')
class LiveIntakeReviewTests(unittest.TestCase):
    def test_reviewed_manifest(self):
        # Retain artifacts even on failure for inspection.
        directory = Path(tempfile.mkdtemp(prefix='agentic-web-live-intake-'))
        result = build_intake_graph().invoke({
            'prompt': 'Create a responsive website displaying Hello world. Do not include a contact form.',
            'output_file': str(directory / 'task.json'), 'review_policy': {'mode': 'feedback'}})
        self.assertEqual(result['status'], 'analyzed', result)
        self.assertTrue((directory / 'task.json').exists())


if __name__ == '__main__':
    unittest.main()
