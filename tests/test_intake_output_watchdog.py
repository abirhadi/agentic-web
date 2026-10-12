import json
import unittest
from unittest.mock import patch

from agentic_web.client import LMStudioClient, ModelConfig
from agentic_web.intake_watchdog import IntakeOutputWatchdog
from agentic_web.reasoning import InferenceError
from agentic_web.schema import StrictModel
from inference_mock import sdk, chunk


def record(index, description='Show technical report (2024).'):
    return {'id': f'feat_{index:03}', 'description': description, 'required': False,
            'acceptance_criteria': ['All entries are visible.'],
            'tests': [{'test_id': f't_feat_{index:03}_01', 'expected_results': []}]}


class Response(StrictModel):
    task_spec: dict


class IntakeOutputTests(unittest.TestCase):
    def test_completed_duplicate_requirements_are_rejected_without_repair(self):
        from copy import deepcopy
        from pydantic import ValidationError
        from agentic_web.intake_schema import RequirementsAnalysis
        from test_intake import analysis_fixture
        value = analysis_fixture()
        duplicate = deepcopy(value['task_spec']['features'][0])
        duplicate['id'] = 'duplicate'
        value['task_spec']['features'].append(duplicate)
        test = deepcopy(value['task_spec']['tests'][0])
        test.update(test_id='duplicate-test', requirement_id='duplicate')
        value['task_spec']['tests'].append(test)
        with self.assertRaisesRegex(ValidationError, 'Duplicate requirement outcomes'):
            RequirementsAnalysis.model_validate(value)

    def test_archived_cycle_with_changing_ids_and_chunk_boundaries(self):
        descriptions = ['Show extracurricular leadership (UPE President).',
                        'Show awards (Young Talent Award).', 'Show technical report (2024).',
                        'Show conference presentations (2024).', 'Show mentorship (mentors).']
        raw = json.dumps({'analysis': {'task_spec': {'features': [
            record(i, descriptions[i % 5]) for i in range(30)]}}})
        for size in (1, 7, len(raw)):
            watchdog = IntakeOutputWatchdog()
            evidence = None
            for end in range(size, len(raw) + size, size):
                evidence = watchdog.observe(raw[:end])
                if evidence:
                    break
            self.assertEqual(evidence['occurrences'], 4)
            self.assertEqual(evidence['kind'], 'structured_output_repetition')

    def test_distinct_requirements_shared_boilerplate_and_escaped_text(self):
        records = [record(i, f'Outcome {i}: quotes " and braces {{}} and slash \\') for i in range(100)]
        records[0]['tests'] = [record(1)] * 10
        raw = json.dumps({'task_spec': {'features': records}, 'sources': [record(1)] * 20})
        watchdog = IntakeOutputWatchdog()
        for end in range(1, len(raw) + 1):
            self.assertIsNone(watchdog.observe(raw[:end]))

    def test_same_description_with_distinct_criteria_is_not_a_loop(self):
        records = [record(i) for i in range(10)]
        for i, item in enumerate(records):
            item['acceptance_criteria'] = [f'Distinct observable outcome {i}']
        self.assertIsNone(IntakeOutputWatchdog().observe(json.dumps({'task_spec': {'features': records}})))

    def test_stream_cancelled_partial_output_preserved_without_retry(self):
        records = [record(i) for i in range(10)]
        parts = ['{"task_spec":{"features":['] + [json.dumps(item) + ',' for item in records]
        provider = sdk(chunks=[chunk(part) for part in parts])
        client = LMStudioClient()
        with patch.object(client, '_async_client', return_value=provider), self.assertRaises(InferenceError) as caught:
            client.complete('System', 'User', ModelConfig(inference_role='intake'), response_schema=Response)
        metrics = caught.exception.diagnostics
        self.assertEqual(metrics['error_classification'], 'structured_output_repetition')
        self.assertEqual(metrics['final_content'], ''.join(parts[:5]))
        self.assertTrue(metrics['incomplete_generation'])
        self.assertTrue(metrics['watchdog_triggered'])
        self.assertEqual(len(metrics['attempts']), 1)
        self.assertTrue(provider.stream.closed)
        provider.chat.completions.create.assert_awaited_once()

    def test_valid_response_still_passes_schema_validation(self):
        raw = json.dumps({'task_spec': {'nonfunctional_requirements': [record(1)]}})
        provider = sdk(raw)
        client = LMStudioClient()
        with patch.object(client, '_async_client', return_value=provider):
            actual, metrics = client.complete('System', 'User', ModelConfig(inference_role='intake'), response_schema=Response)
        self.assertEqual(actual, raw)
        self.assertEqual(metrics['structured_validation'], 'passed')

    def test_malformed_json_keeps_json_error_classification(self):
        provider = sdk('{"task_spec":{"name":"invalid\\q"}}')
        client = LMStudioClient()
        with patch.object(client, '_async_client', return_value=provider), self.assertRaises(InferenceError) as caught:
            client.complete('System', 'User', ModelConfig(inference_role='intake'), response_schema=Response)
        self.assertEqual(caught.exception.diagnostics['error_classification'], 'json_parsing_error')
