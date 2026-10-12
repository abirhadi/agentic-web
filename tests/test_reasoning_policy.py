"""Deterministic policy, adapter, cancellation, and reproducibility regressions."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from agentic_web.client import LMStudioClient, ModelConfig
from agentic_web.schema import StrictModel
from agentic_web.reasoning import (ReasoningPolicy, ReasoningWatchdog, InferenceError,
    UnsupportedReasoningError, UnsupportedTransportError, resolve_policy, persist_failure)
from agentic_web.inference import native_url
from agentic_web.experiments import ExperimentConfig, build_plan, save_plan
from agentic_web.batch import run_experiment, validated_plan
from agentic_web.provenance import fingerprint, filesystem_path
from inference_mock import sdk, Stream, chunk
from test_experiments import configuration, ROOT
from collaboration_mock_client import CollaborationMockClient
from test_framework import stub_verifier


class Answer(StrictModel):
    answer: str


def capability(model='model', modes=('off', 'on'), default='on'):
    return [{'key': model, 'loaded_instances': [{'id': 'owned-instance', 'config': {'context_length': 32768}}],
             'capabilities': {'reasoning': {'allowed_options': list(modes), 'default': default}}}]


class PolicyTests(unittest.TestCase):
    def test_role_aliases_and_restrictive_budget_precedence(self):
        policy = ReasoningPolicy(roles={'requirements_analyst': {'mode': 'off', 'max_output_tokens': 5000},
                                       'architect': {'mode': 'high'}}, max_output_tokens=2000, timeout_seconds=200)
        config = ModelConfig(reasoning_policy=policy, planner_max_tokens=1000, timeout_seconds=100,
                             inference_role='intake')
        resolved = resolve_policy(config, True)
        self.assertEqual((resolved['role'], resolved['requested_mode'], resolved['max_output_tokens'], resolved['timeout_seconds']),
                         ('intake', 'off', 1000, 100))
        self.assertEqual(resolve_policy(config.model_copy(update={'inference_role': 'architect'}))['requested_mode'], 'high')
        for role in ('planner', 'ux', 'architect', 'coordinator', 'implementer', 'qa'):
            role_config, planner = ModelConfig().for_role(role)
            self.assertGreater(resolve_policy(role_config, planner)['max_output_tokens'], 0)
            self.assertEqual(resolve_policy(role_config, planner)['timeout_seconds'], 3000)

    def test_invalid_configuration_and_duplicate_aliases(self):
        for value in ({'roles': {'unknown': {}}}, {'roles': {'architect': {}, 'architecture': {}}},
                      {'max_output_tokens': 0}, {'roles': {'qa': {'mode': 'none'}}},
                      {'watchdog': {'segment_chars': 1000, 'window_chars': 1024}},
                      {'recovery': {'max_attempts': 4}}, {'recovery': {'retry_on': ['schema_validation_error']}}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ReasoningPolicy.model_validate(value)

    def test_backward_config_and_hash_stability(self):
        old = ModelConfig().model_dump()
        self.assertNotIn('reasoning_policy', old)
        self.assertEqual(ModelConfig.model_validate(old).model_dump(), old)
        p = ReasoningPolicy(roles={'architect': {'mode': 'auto'}})
        self.assertEqual(p.model_dump(), ReasoningPolicy.model_validate_json(p.model_dump_json()).model_dump())
        self.assertEqual(resolve_policy(ModelConfig(reasoning_policy=p))['policy_hash'], fingerprint(p.model_dump(mode='json')))


class InferenceTests(unittest.TestCase):
    def run_client(self, provider, config=None, schema=None):
        client = LMStudioClient()
        with patch.object(client, '_async_client', return_value=provider):
            return client.complete('System', 'User', config or ModelConfig(model='model'), response_schema=schema)

    def test_auto_without_capabilities_keeps_seed_schema_and_sampling(self):
        provider = sdk('{"answer":"hello"}')
        raw, result = self.run_client(provider, ModelConfig(model='model', seed=12, top_p=.7, temperature=.8), Answer)
        request = provider.chat.completions.create.call_args.kwargs
        self.assertEqual((request['seed'], request['temperature'], request['top_p']), (13, .8, .7))
        self.assertEqual(request['response_format']['json_schema']['schema'], Answer.model_json_schema())
        self.assertTrue(request['stream'])
        self.assertGreater(request['max_tokens'], 0)
        self.assertEqual(raw, '{"answer":"hello"}')
        self.assertIsNone(result['effective_reasoning_mode'])
        self.assertFalse(result['reasoning_mode_confirmed'])
        self.assertIsNone(result['reasoning_tokens'])
        self.assertTrue(provider.stream.closed)

    def test_native_on_off_only_when_no_seed_or_schema_is_dropped(self):
        for mode in ('on', 'off', 'low', 'medium', 'high'):
            with self.subTest(mode=mode):
                provider = sdk(capability=capability(modes=('on', 'off', 'low', 'medium', 'high')))
                stream = Stream([{'type': 'reasoning.delta', 'content': 'Separate analysis'},
                    {'type': 'message.delta', 'content': 'Final answer'}, {'type': 'chat.end', 'result': {
                        'model_instance_id': 'owned-instance', 'stats': {
                            'input_tokens': 5, 'total_output_tokens': 10, 'reasoning_output_tokens': 6}}}])
                provider.post = AsyncMock(return_value=stream)
                config = ModelConfig(model='owned-instance', seed=None, reasoning_policy=ReasoningPolicy(roles={'implementer': {'mode': mode}}))
                raw, metrics = self.run_client(provider, config)
                request = provider.post.call_args.kwargs['body']
                self.assertEqual(request['reasoning'], mode)
                self.assertEqual(request['model'], 'owned-instance')
                self.assertEqual(request['integrations'], [])
                self.assertFalse(request['store'])
                self.assertNotIn('seed', request)
                self.assertEqual(raw, 'Final answer')
                self.assertEqual(metrics['reasoning_content'], 'Separate analysis')
                self.assertEqual(metrics['final_answer_tokens'], 4)
                self.assertTrue(metrics['reasoning_mode_confirmed'])
                self.assertTrue(stream.closed)
                provider.chat.completions.create.assert_not_called()

    def test_explicit_controls_cannot_drop_seed_or_schema(self):
        for mode in ('on', 'off'):
            for seed, schema in ((12, None), (None, Answer), (12, Answer)):
                with self.subTest(mode=mode, seed=seed, schema=schema):
                    provider = sdk(capability=capability())
                    config = ModelConfig(model='model', seed=seed, reasoning_policy=ReasoningPolicy(roles={'planner': {'mode': mode}}), inference_role='planner')
                    with self.assertRaises(UnsupportedTransportError) as caught:
                        self.run_client(provider, config, schema)
                    self.assertEqual(caught.exception.diagnostics['error_classification'], 'unsupported_transport')
                    self.assertEqual(len(caught.exception.diagnostics['attempts']), 1)
                    provider.post.assert_not_called()
                    provider.chat.completions.create.assert_not_called()

    def test_unsupported_mode_and_discovery_failure(self):
        for discovery_failure in (False, True):
            provider = sdk(capability=capability())
            if discovery_failure:
                provider.get.side_effect = RuntimeError('Unavailable')
            config = ModelConfig(model='model', reasoning_policy=ReasoningPolicy(roles={'qa': {'mode': 'high'}}), inference_role='qa')
            with self.assertRaises(UnsupportedReasoningError):
                self.run_client(provider, config)
            provider.chat.completions.create.assert_not_called()
        provider = sdk()
        provider.get.side_effect = RuntimeError('Old server lacks native discovery')
        _, metrics = self.run_client(provider)
        self.assertEqual(metrics['capability']['status'], 'discovery_failed')
        self.assertFalse(metrics['reasoning_mode_confirmed'])

    def test_observed_reasoning_only_context_exhaustion_is_not_final_json(self):
        usage = {'prompt_tokens': 4299, 'completion_tokens': 28469, 'total_tokens': 32768,
                 'completion_tokens_details': {'reasoning_tokens': 28466}}
        provider = sdk(chunks=[chunk(reasoning='{"answer":"plausible but not final"}'), chunk(finish='length', usage=usage)])
        with self.assertRaises(InferenceError) as caught:
            self.run_client(provider, schema=Answer)
        metrics = caught.exception.diagnostics
        self.assertEqual(metrics['error_classification'], 'output_budget_exhausted')
        self.assertEqual(metrics['final_content'], '')
        self.assertEqual(metrics['reasoning_tokens'], 28466)
        self.assertEqual(metrics['generated_tokens'], 28469)
        self.assertEqual(metrics['finish_reason'], 'length')
        self.assertTrue(metrics['budget_exhausted'])
        self.assertTrue(provider.stream.closed)
        with tempfile.TemporaryDirectory() as tmp:
            persist_failure(tmp, 'decomposition', caught.exception)
            self.assertEqual((Path(tmp) / 'decomposition_raw.txt').read_text(), '')
            self.assertEqual(len(json.loads((Path(tmp) / 'decomposition_inference_attempts.json').read_text())), 1)

    def test_missing_truncated_malformed_and_schema_invalid_final(self):
        for raw, finish, expected in (('', 'stop', 'missing_final_content'), ('{"answer":"ok"}', 'length', 'output_budget_exhausted'),
                ('{"answer":', 'stop', 'json_parsing_error'), ('{"wrong": "key"}', 'stop', 'schema_validation_error'),
                ('{"Answer": {"answer":"wrong wrapper"}}', 'stop', 'schema_validation_error'),
                ('{"answer":"ok"}', None, 'incomplete_generation')):
            with self.subTest(expected=expected):
                provider = sdk(raw, finish)
                with self.assertRaises(InferenceError) as caught:
                    self.run_client(provider, schema=Answer)
                self.assertEqual(caught.exception.diagnostics['error_classification'], expected)
                self.assertEqual(caught.exception.diagnostics['final_content'], raw)
                self.assertTrue(provider.stream.closed)

    def test_repetition_cancels_without_accepting_later_final(self):
        sentence = 'We must inspect all requirements carefully before returning the complete result and verify each dependency against its authoritative source. '
        provider = sdk(chunks=[chunk(reasoning=sentence) for _ in range(20)] + [chunk('{"answer":"late"}', finish='stop')])
        with self.assertRaises(InferenceError) as caught:
            self.run_client(provider, schema=Answer)
        result = caught.exception.diagnostics
        self.assertEqual(result['error_classification'], 'repetition_watchdog')
        self.assertTrue(result['watchdog_triggered'])
        self.assertGreaterEqual(result['watchdog_evidence']['occurrences'], 4)
        self.assertLessEqual(len(result['reasoning_content']), 4096)
        self.assertEqual(result['final_content'], '')
        self.assertTrue(provider.stream.closed)

    def test_watchdog_false_positive_resistance_and_excessive_reasoning(self):
        settings = ReasoningPolicy().watchdog.model_dump()
        for text in ('{}[](); ' * 4000, 'Check. ' * 4000,
                     ' '.join(f'item_{i} describes a distinct condition number {i}' for i in range(500))):
            self.assertIsNone(ReasoningWatchdog(settings).observe(text, False))
        watchdog = ReasoningWatchdog({**settings, 'max_reasoning_chars_without_final': 1024})
        self.assertEqual(watchdog.observe('x' * 1024, False)['kind'], 'excessive_reasoning')

    def test_timeout_and_stall_cancel_pending_read_and_close_client(self):
        class HangingStream(Stream):
            async def __anext__(self):
                try:
                    await asyncio.sleep(60)
                finally:
                    self.cancelled = True
        for timeout, stall, expected in ((.05, 1, 'timeout'), (1, .05, 'stalled_generation')):
            provider = sdk()
            stream = HangingStream([])
            provider.chat.completions.create.return_value = stream
            config = ModelConfig(reasoning_policy=ReasoningPolicy(timeout_seconds=timeout, watchdog={'stall_seconds': stall}))
            with self.assertRaises(InferenceError) as caught:
                self.run_client(provider, config)
            self.assertEqual(caught.exception.diagnostics['error_classification'], expected)
            self.assertTrue(stream.cancelled)
            self.assertTrue(stream.closed)
            provider.__aexit__.assert_awaited_once()

    def test_bounded_same_settings_retry_and_all_failed_attempts(self):
        provider = sdk('{"answer":"ok"}')
        provider.chat.completions.create.side_effect = [ConnectionError('network dropped'), provider.stream]
        config = ModelConfig(reasoning_policy=ReasoningPolicy(recovery={'max_attempts': 2, 'retry_on': ['transport_failure']}))
        _, metrics = self.run_client(provider, config, Answer)
        self.assertEqual([a['status'] for a in metrics['attempts']], ['failed', 'succeeded'])
        calls = provider.chat.completions.create.call_args_list
        self.assertEqual(calls[0], calls[1])
        provider = sdk()
        provider.chat.completions.create.side_effect = ConnectionError('still unavailable')
        with self.assertRaises(InferenceError) as caught:
            self.run_client(provider, config, Answer)
        self.assertEqual(len(caught.exception.diagnostics['attempts']), 2)
        self.assertEqual(provider.chat.completions.create.call_count, 2)

    def test_retry_records_are_exclusive_and_durable_before_next_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = sdk('{"answer":"ok"}')
            seen = []
            async def create(**request):
                seen.append(request)
                if len(seen) == 1:
                    raise ConnectionError('transient')
                first = Path(tmp) / 'implementer_inference_attempt_0001.json'
                self.assertEqual(json.loads(first.read_text())['status'], 'failed')
                return provider.stream
            provider.chat.completions.create.side_effect = create
            config = ModelConfig(inference_artifact_dir=tmp, reasoning_policy=ReasoningPolicy(
                recovery={'max_attempts': 2, 'retry_on': ['transport_failure']}))
            self.run_client(provider, config, Answer)
            self.assertEqual(seen[0], seen[1])
            second = Path(tmp) / 'implementer_inference_attempt_0002.json'
            self.assertEqual(json.loads(second.read_text())['status'], 'succeeded')
            self.assertNotIn('inference_artifact_dir', config.model_dump())

    def test_long_inference_artifact_path_preserves_valid_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = filesystem_path(tmp) / ('a' * 90) / ('b' * 90) / ('c' * 60)
            try:
                directory.mkdir(parents=True)
                raw, metrics = self.run_client(sdk('{"answer":"ok"}'),
                    ModelConfig(inference_artifact_dir=str(directory)), Answer)
                self.assertEqual(json.loads(raw), {'answer': 'ok'})
                path = directory / 'implementer_inference_attempt_0001.json'
                self.assertGreater(len(str(path)), 260)
                self.assertEqual(json.loads(path.read_text())['status'], 'succeeded')
                self.assertEqual(metrics['structured_validation'], 'passed')
            finally:
                shutil.rmtree(filesystem_path(tmp) / ('a' * 90), ignore_errors=False)

    def test_timeout_before_stream_creation_closes_transport(self):
        provider = sdk()
        cancelled = []
        async def create(**kwargs):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(True)
        provider.chat.completions.create.side_effect = create
        with self.assertRaises(InferenceError) as caught:
            self.run_client(provider, ModelConfig(reasoning_policy=ReasoningPolicy(timeout_seconds=.05)))
        self.assertEqual(caught.exception.diagnostics['error_classification'], 'timeout')
        self.assertTrue(cancelled)
        provider.__aexit__.assert_awaited_once()

    def test_url_preserves_reverse_proxy_prefix(self):
        self.assertEqual(native_url('http://localhost:1234/proxy/v1/', 'models'), 'http://localhost:1234/proxy/api/v1/models')

    def test_installed_sdk_consumes_real_http_sse_for_both_transports(self):
        import httpx
        from openai import AsyncOpenAI
        for native in (False, True):
            requests = []
            async def handle(request):
                requests.append(request)
                if request.method == 'GET':
                    return httpx.Response(200, json={'models': capability()})
                if native:
                    body = 'event: reasoning.delta\ndata: {"type":"reasoning.delta","content":"analysis"}\n\n'
                    body += 'event: message.delta\ndata: {"type":"message.delta","content":"hello"}\n\n'
                    body += 'event: chat.end\ndata: {"type":"chat.end","result":{"model_instance_id":"model","stats":{"input_tokens":4,"total_output_tokens":8,"reasoning_output_tokens":5}}}\n\n'
                else:
                    event = {'id': 'mock', 'created': 0, 'object': 'chat.completion.chunk', 'model': 'model',
                        'choices': [{'index': 0, 'delta': {'content': '{"answer":"hello"}', 'reasoning_content': 'analysis'}, 'finish_reason': 'stop'}]}
                    body = 'data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n'
                return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=body.encode())
            config = ModelConfig(model='model', seed=None if native else 42,
                reasoning_policy=ReasoningPolicy(roles={'implementer': {'mode': 'on' if native else 'auto'}}))
            http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            provider = AsyncOpenAI(base_url=config.base_url, api_key='test', http_client=http, max_retries=0)
            raw, metrics = self.run_client(provider, config, None if native else Answer)
            self.assertEqual(raw, 'hello' if native else '{"answer":"hello"}')
            self.assertEqual(metrics['reasoning_content'], 'analysis')
            self.assertEqual(requests[0].url.path, '/api/v1/models')
            self.assertEqual(requests[1].url.path, '/api/v1/chat' if native else '/v1/chat/completions')
            self.assertTrue(http.is_closed)


class OrchestrationPolicyTests(unittest.TestCase):
    def test_all_eight_roles_retain_policy_through_scheduled_intake_and_graph(self):
        from test_decomposition import DecompositionClient
        from test_intake import PROMPT
        from agentic_web.workflow import build_graph
        from agentic_web.resource_scheduler import ResourceScheduler
        from agentic_web.resource_schema import ResourceSettings
        from resource_mock_backend import SimulatedBackend
        from agentic_web.reasoning import canonical_role, ROLES
        backend = SimulatedBackend()
        scheduler = ResourceScheduler(backend, backend.hardware)
        policy = ReasoningPolicy(roles={role: {'max_output_tokens': 1000 + i} for i, role in enumerate(ROLES)})
        observed = {}
        class Client(DecompositionClient):
            def complete(self, system, user, config, *, planner=False, response_schema=None):
                role = canonical_role(json.loads(user).get('role', 'planner' if planner else 'implementer'))
                resolved = resolve_policy(config, planner)
                self_test.assertEqual(resolved['role'], role)
                self_test.assertEqual(resolved['max_output_tokens'], policy.roles[role].max_output_tokens)
                observed[role] = resolved
                return super().complete(system, user, config, planner=planner, response_schema=response_schema)
        self_test = self
        config = ModelConfig(model='heavy', planner_model='heavy', ux_model='light', architect_model='heavy',
                             coordinator_model='heavy', qa_model='light', reasoning_policy=policy)
        settings = ResourceSettings(gpu_safety_bytes=0, ram_safety_bytes=0, estimate_padding=0)
        with tempfile.TemporaryDirectory() as tmp:
            result = build_graph(Client(), stub_verifier, workflow='collaborative', resource_scheduler=scheduler).invoke({
                'prompt': PROMPT, 'decompose': True, 'output_root': tmp, 'model_config': config.model_dump(),
                'resource_config': settings.model_dump()})
        self.assertEqual(result['status'], 'passed', result.get('error'))
        self.assertEqual(set(observed), set(ROLES))
        self.assertEqual(backend.instances, {})

    def test_terminal_inference_failure_releases_owned_scheduler_instances(self):
        from agentic_web.workflow import build_graph
        from agentic_web.resource_scheduler import ResourceScheduler
        from agentic_web.resource_schema import ResourceSettings
        from resource_mock_backend import SimulatedBackend
        backend = SimulatedBackend()
        class Client(CollaborationMockClient):
            def complete(self, *args, **kwargs):
                raise InferenceError('Reasoning budget exhausted', {'error_classification': 'output_budget_exhausted',
                    'finish_reason': 'length', 'final_content': '', 'attempts': [{'status': 'failed'}]})
        with tempfile.TemporaryDirectory() as tmp:
            result = build_graph(Client(), stub_verifier, resource_scheduler=ResourceScheduler(backend, backend.hardware)).invoke({
                'task_path': str(ROOT / 'tasks/todo.json'), 'output_root': tmp, 'model_config': {'model': 'heavy'},
                'resource_config': ResourceSettings(gpu_safety_bytes=0, ram_safety_bytes=0).model_dump()})
            self.assertEqual(result['status'], 'failed')
            self.assertTrue((Path(result['run_dir']) / 'planner_inference_attempts.json').is_file())
        self.assertEqual(backend.instances, {})

    def test_cli_passes_shared_policy_to_intake_and_decompose(self):
        from agentic_web.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            policy_file = Path(tmp) / 'policy.json'
            policy_file.write_text(ReasoningPolicy(max_output_tokens=1000).model_dump_json())
            for command, args, target in (
                ('intake', ['--prompt', 'Create a website'], 'agentic_web.intake.intake_app.invoke'),
                ('decompose', ['--task', 'task.json'], 'agentic_web.decomposition.decompose')):
                with patch(target, return_value={'status': 'analyzed'}) as invoke, patch('sys.argv', ['agentic_web', command,
                    *args, '--output', str(Path(tmp) / 'new.json'), '--reasoning-policy', str(policy_file)]), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
                self.assertEqual(invoke.call_args.args[0]['model_config']['reasoning_policy']['max_output_tokens'], 1000)


class FrozenPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'task.json').write_bytes((ROOT / 'tasks/todo.json').read_bytes())
        value = configuration()
        value['factors'].update(seeds=[1], temperatures=[.4], variance_conditions=[{'identifier': 'control'}])
        self.config = ExperimentConfig.model_validate(value)

    def test_plan_freezes_defaults_and_detects_resume_policy_mismatch(self):
        plan = build_plan(self.config, self.root)
        path = save_plan(plan, self.root / 'config.json')
        self.assertEqual(validated_plan(path)['reasoning_policy_hash'], plan['reasoning_policy_hash'])
        result = run_experiment(path, max_candidates=1, client=CollaborationMockClient(), verifier=stub_verifier)
        first = Path(result['outcomes'][0]['record']).read_bytes()
        changed = ReasoningPolicy(roles={'planner': {'mode': 'off'}})
        with self.assertRaisesRegex(ValueError, 'differs from frozen'):
            run_experiment(path, resume=True, reasoning_policy=changed)
        self.assertEqual(Path(result['outcomes'][0]['record']).read_bytes(), first)
        resumed = run_experiment(path, resume=True, client=CollaborationMockClient(), verifier=stub_verifier,
                                 reasoning_policy=ReasoningPolicy())
        self.assertEqual(resumed['selected_count'], 1)
        record = json.loads(Path(resumed['outcomes'][0]['record']).read_text())
        self.assertEqual(record['reasoning_policy_hash'], plan['reasoning_policy_hash'])
        dry = run_experiment(path, resume=True, dry_run=True)
        self.assertIn('reasoning_policies', dry)

    def test_changed_policy_changes_candidate_identity_but_not_task_hash(self):
        first = build_plan(self.config, self.root)
        second = build_plan(self.config.model_copy(update={'reasoning_policy': ReasoningPolicy(max_output_tokens=1000)}), self.root)
        self.assertNotEqual(first['configuration_hash'], second['configuration_hash'])
        self.assertNotEqual(first['candidates'][0]['candidate_id'], second['candidates'][0]['candidate_id'])
        self.assertEqual(first['candidates'][0]['specification_hash'], second['candidates'][0]['specification_hash'])

    def test_legacy_plans_readable_but_not_silently_reinterpreted(self):
        legacy = build_plan(self.config, self.root, freeze_policy=False)
        path = save_plan(legacy, self.root / 'old.json')
        self.assertEqual(validated_plan(path)['candidates'], legacy['candidates'])
        self.assertEqual(run_experiment(path, dry_run=True)['status'], 'dry_run')
        with self.assertRaisesRegex(ValueError, 'Legacy plan'):
            run_experiment(path)

    def test_cli_plan_freezes_policy_and_resume_rejects_changed_file(self):
        from agentic_web.__main__ import main
        config_path, policy_path = self.root / 'config.json', self.root / 'policy.json'
        config_path.write_text(self.config.model_dump_json())
        policy_path.write_text(ReasoningPolicy(max_output_tokens=1000).model_dump_json())
        with patch('sys.argv', ['agentic_web', 'experiment', 'plan', '--config', str(config_path),
                               '--reasoning-policy', str(policy_path)]), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        plan_path = self.root / 'output' / self.config.experiment_id / 'experiment_plan.json'
        plan = validated_plan(plan_path)
        self.assertEqual(plan['configuration']['reasoning_policy']['max_output_tokens'], 1000)
        argv = ['agentic_web', 'experiment', 'resume', '--plan', str(plan_path), '--dry-run', '--reasoning-policy', str(policy_path)]
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(), 0)
        dry = json.loads(stdout.getvalue())
        self.assertTrue(dry['reasoning_policies'])
        policy_path.write_text(ReasoningPolicy(max_output_tokens=2000).model_dump_json())
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(), 2)
        self.assertIn('differs from frozen plan', json.loads(stdout.getvalue())['error'])


if __name__ == '__main__':
    unittest.main()
