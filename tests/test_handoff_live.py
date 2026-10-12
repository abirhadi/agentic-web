"""Optional LM Studio smoke probe; disabled for the automated suite."""
import os
from pathlib import Path
import tempfile
import unittest

from agentic_web.intake import build_intake_graph


@unittest.skipUnless(os.getenv('AGENTIC_WEB_LIVE_HANDOFF') == '1', 'explicit live handoff opt-in required')
class LiveHandoffTests(unittest.TestCase):
    def test_handoff(self):
        root = Path(tempfile.mkdtemp(prefix='agentic-web-handoff-'))
        result = build_intake_graph().invoke({'prompt': 'Create a responsive website displaying Hello world.',
            'output_file': str(root / 'task.json'), 'handoff_policy': {'mode': 'collaborative'}})
        self.assertEqual(result.get('handoff_status'), 'accepted', result)
