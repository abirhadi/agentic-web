"""API calls share a 50-minute default; browser budgets remain independent."""
import io
import subprocess
import unittest
from unittest.mock import patch

from agentic_web.client import LMStudioClient, ModelConfig
from agentic_web.lmstudio_placement import LMStudioPlacementBackend
from agentic_web.schema import VerificationSettings


class APITimeoutTests(unittest.TestCase):
    def test_every_role_uses_3000_second_sdk_timeout(self):
        # Intake and decomposition also use this planner configuration/client.
        for role in ('planner', 'ux', 'architect', 'coordinator', 'implementer', 'qa'):
            with self.subTest(role=role):
                config, _ = ModelConfig().for_role(role)
                with patch('openai.OpenAI') as sdk:
                    LMStudioClient()._client(config)
                self.assertEqual(sdk.call_args.kwargs['timeout'], 3000)
                self.assertEqual(sdk.call_args.kwargs['max_retries'], 0)

    def test_explicit_timeout_override_and_browser_budgets_preserved(self):
        config, _ = ModelConfig(timeout_seconds=4000).for_role('qa')
        with patch('openai.OpenAI') as sdk:
            LMStudioClient()._client(config)
        self.assertEqual(sdk.call_args.kwargs['timeout'], 4000)
        self.assertEqual(VerificationSettings().timeout_ms, 5000)
        self.assertEqual(VerificationSettings().total_timeout_seconds, 120)

    def test_native_management_and_cli_use_3000_seconds(self):
        backend = LMStudioPlacementBackend('http://127.0.0.1:1234/v1')
        with patch('agentic_web.lmstudio_placement.urlopen', return_value=io.BytesIO(b'{"models":[]}')) as api:
            self.assertEqual(backend.inventory(), [])
        self.assertEqual(api.call_args.kwargs['timeout'], 3000)
        with patch('agentic_web.lmstudio_placement.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '[]', '')) as cli:
            self.assertEqual(backend.live_instances(), [])
        self.assertEqual(cli.call_args.kwargs['timeout'], 3000)
