import unittest
from copy import deepcopy

from pydantic import ValidationError
from jsonschema_rs import Draft202012Validator

from agentic_web.intake_schema import IntakeSource
from agentic_web.intake_review import grounded_review_schema


class GroundedReviewTests(unittest.TestCase):
    def test_revision_evidence_uses_available_exact_sources(self):
        from agentic_web.intake_review import revision_response_schema
        from agentic_web.schema import StrictModel
        previous = '{"description": "Display the supplied content"}'
        schema = revision_response_schema(StrictModel, [{'finding_id': 'known'}], self.source, previous)
        value = {'analysis': {}, 'requirement_lineage': {}, 'decisions': [{
            'finding_id': 'known', 'action': 'reject', 'rationale': 'Compared against supplied request',
            'evidence': [{'source': 'request', 'quote': self.source.original_request}]}]}
        schema.model_validate(value)
        validator = Draft202012Validator(schema.model_json_schema())
        for source, quote in [('draft', '"Display"'), ('previous_draft', '"Display"'),
                              ('request', 'The user requires one HTML page'), ('context', 'Absent')]:
            with self.subTest(source=source):
                invalid = deepcopy(value)
                invalid['decisions'][0]['evidence'] = [{'source': source, 'quote': quote}]
                with self.assertRaises(ValidationError):
                    schema.model_validate(invalid)
                self.assertFalse(validator.is_valid(invalid))
        value['decisions'][0]['evidence'] = [{'source': 'previous_draft', 'quote': previous}]
        schema.model_validate(value)
        self.assertTrue(validator.is_valid(value))

    def test_revision_schema_requires_supplied_decisions(self):
        from agentic_web.intake_review import revision_response_schema
        from agentic_web.schema import StrictModel
        schema = revision_response_schema(StrictModel, [{'finding_id': 'first'}, {'finding_id': 'second'}])
        decision = {'finding_id': 'first', 'action': 'reject', 'rationale': 'Evidence does not support this concern',
                    'evidence': [{'source': 'request', 'quote': 'Create a website'}]}
        valid = {'analysis': {}, 'decisions': [decision, {**decision, 'finding_id': 'second'}],
                 'requirement_lineage': {}}
        schema.model_validate(valid)
        validator = Draft202012Validator(schema.model_json_schema())
        self.assertTrue(validator.is_valid(valid))
        for decisions in ([], [decision], [decision] * 3, [decision, {**decision, 'finding_id': 'invented'}]):
            with self.subTest(decisions=decisions):
                invalid = {**valid, 'decisions': decisions}
                with self.assertRaises(ValidationError):
                    schema.model_validate(invalid)
                self.assertFalse(validator.is_valid(invalid))

    def test_reviewer_default_and_launcher_caps_are_16k(self):
        from pathlib import Path
        from agentic_web.client import ModelConfig
        from agentic_web.reasoning import resolve_policy, load_policy
        for policy in (None, load_policy(Path(__file__).resolve().parents[1] / 'reasoning.example.json')):
            config = ModelConfig(inference_role='qa', reasoning_policy=policy)
            self.assertEqual(resolve_policy(config, planner=True)['max_output_tokens'], 16384)

    def test_reviewer_repetition_cancels_stream_and_preserves_evidence(self):
        import json
        from unittest.mock import patch
        from agentic_web.client import LMStudioClient, ModelConfig
        from agentic_web.reasoning import InferenceError
        from inference_mock import sdk, chunk
        finding = self.valid['findings'][0]
        pieces = ['{"findings":['] + [json.dumps(finding) + ','] * 8
        provider = sdk(chunks=[chunk(piece) for piece in pieces])
        client = LMStudioClient()
        with patch.object(client, '_async_client', return_value=provider), self.assertRaises(InferenceError) as caught:
            client.complete('System', 'User', ModelConfig(inference_role='qa'), response_schema=self.schema)
        metrics = caught.exception.diagnostics
        self.assertEqual(metrics['error_classification'], 'structured_output_repetition')
        self.assertEqual(metrics['watchdog_evidence']['array'], 'findings')
        self.assertEqual(metrics['final_content'], ''.join(pieces[:5]))
        self.assertEqual(len(metrics['attempts']), 1)
        self.assertTrue(provider.stream.closed)

    def test_distinct_review_findings_can_share_evidence(self):
        import json
        from agentic_web.intake_watchdog import IntakeOutputWatchdog
        findings = []
        for i in range(10):
            finding = deepcopy(self.valid['findings'][0])
            finding['observation'] = f'Distinct defect {i}'
            findings.append(finding)
        self.assertIsNone(IntakeOutputWatchdog(review=True).observe(json.dumps({'findings': findings})))

    def setUp(self):
        self.source = IntakeSource(original_request='Create a website from my resume.')
        self.schema = grounded_review_schema(self.source, '{"description": "Display source content"}', '', [])
        self.valid = {'findings': [{'category': 'ambiguity', 'severity': 'advisory',
            'observation': 'Clarify the intended audience.',
            'evidence': [{'source': 'request', 'quote': self.source.original_request}]}],
            'prior_finding_assessments': []}

    def assert_rejected(self, value):
        with self.assertRaises(ValidationError):
            self.schema.model_validate(value)
        self.assertFalse(Draft202012Validator(self.schema.model_json_schema()).is_valid(value))

    def test_valid_root_and_empty_first_round_assessments(self):
        self.schema.model_validate(self.valid)
        self.assertTrue(Draft202012Validator(self.schema.model_json_schema()).is_valid(self.valid))
        self.assertEqual(self.schema.__name__, 'SemanticReview')
        self.assert_rejected({'SemanticReview': self.valid})

    def test_archived_empty_previous_draft_and_inexact_quotes(self):
        for source, quote in [('previous_draft', ''), ('previous_draft', 'Invented'),
                              ('context', 'Absent'), ('request', '"' + self.source.original_request + '"'),
                              ('draft', '(The generated HTML page contains all sections ... as in the PDF.)')]:
            with self.subTest(source=source, quote=quote):
                value = deepcopy(self.valid)
                value['findings'][0]['evidence'] = [{'source': source, 'quote': quote}]
                self.assert_rejected(value)

    def test_no_placeholder_assessments_in_first_round(self):
        value = deepcopy(self.valid)
        value['prior_finding_assessments'] = [{'finding_id': '', 'status': 'unresolved',
            'rationale': 'Invented prior finding', 'evidence': value['findings'][0]['evidence']}]
        self.assert_rejected(value)

    def test_subsequent_round_allows_only_supplied_prior_ids_and_sources(self):
        self.schema = grounded_review_schema(self.source, 'New draft', 'Old draft', [{'finding_id': 'known'}])
        value = deepcopy(self.valid)
        assessment = {'finding_id': 'known', 'status': 'resolved', 'rationale': 'Compared drafts',
                      'evidence': [{'source': 'previous_draft', 'quote': 'Old draft'}]}
        value['prior_finding_assessments'] = [assessment]
        self.schema.model_validate(value)
        self.assertTrue(Draft202012Validator(self.schema.model_json_schema()).is_valid(value))
        assessment['finding_id'] = 'invented'
        self.assert_rejected(value)
        value['prior_finding_assessments'] = []
        self.assert_rejected(value)
