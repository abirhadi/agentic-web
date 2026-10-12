"""Context size is a load setting, independent of optional output token caps."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_web.client import ModelConfig
from agentic_web.resource_schema import ResourceSettings


class ContextDefaultsTests(unittest.TestCase):
    def test_resource_context_defaults_to_32k_without_output_cap(self):
        self.assertEqual(ResourceSettings().context_length, 32768)
        config = ModelConfig()
        for role in ('planner', 'ux', 'architect', 'coordinator', 'implementer', 'qa'):
            role_config, planner = config.for_role(role)
            self.assertIsNone(role_config.planner_max_tokens if planner else role_config.max_tokens)
        self.assertEqual(ResourceSettings(context_length=65536).context_length, 65536)

    def test_intake_cli_forwards_resource_config_through_existing_graph(self):
        from agentic_web.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            resource_file = Path(directory) / 'resources.json'
            resource_file.write_text(json.dumps({'parallel_slots': 1}))
            argv = ['agentic_web', 'intake', '--prompt', 'Create a website.', '--output', str(Path(directory) / 'task.json'),
                    '--resource-config', str(resource_file), '--decompose']
            with patch('sys.argv', argv), patch('agentic_web.intake.intake_app.invoke', return_value={'status': 'analyzed'}) as invoke, redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            inputs = invoke.call_args.args[0]
            self.assertEqual(ResourceSettings.model_validate(inputs['resource_config']).context_length, 32768)
            self.assertTrue(inputs['decompose'])
            self.assertIsNone(inputs['model_config']['planner_max_tokens'])


if __name__ == '__main__':
    unittest.main()
