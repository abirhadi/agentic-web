"""Streaming LM Studio adapters and bounded recovery, shared by every role."""
from __future__ import annotations

import asyncio
import json
import time
from urllib.parse import urlsplit, urlunsplit
from typing import TypedDict
from pydantic import ValidationError
from .reasoning import (InferenceError, UnsupportedReasoningError, UnsupportedTransportError,
                        ReasoningWatchdog, ReasoningCapability, resolve_policy)


class InferenceResult(TypedDict, total=False):
    """Normalized metric contract shared by both transports and stage artifacts."""
    role: str
    requested_model: str
    model: str | None
    requested_reasoning_mode: str
    effective_reasoning_mode: str | None
    reasoning_mode_confirmed: bool
    resolved_policy: dict
    policy_hash: str
    endpoint: str
    transport: str | None
    final_content: str
    reasoning_content: str
    reasoning_content_truncated: bool
    prompt_tokens: int | None
    generated_tokens: int | None
    reasoning_tokens: int | None
    final_answer_tokens: int | None
    finish_reason: str | None
    provider_status: str | None
    elapsed_seconds: float
    incomplete_generation: bool
    budget_exhausted: bool
    watchdog_triggered: bool
    structured_validation: str
    error_classification: str | None
    attempts: list[dict]
    seed: int | None
    temperature: float
    top_p: float
    max_tokens: int
    requested_max_tokens: int | None
    response_schema: str | None
    usage: dict | None
    attempt_number: int
    capability: dict
    watchdog_evidence: dict
    error: str
    status: str


def native_url(base_url, endpoint):
    url = urlsplit(base_url)
    # Keep reverse-proxy prefixes; replace only the final OpenAI /v1 component.
    path = url.path.rstrip('/')
    if path.endswith('/v1'):
        path = path[:-3]
    return urlunsplit((url.scheme, url.netloc, path + '/api/v1/' + endpoint, '', ''))


async def discover(client, config, model):
    result = await client.get(native_url(config.base_url, 'models'), cast_to=object)
    if not isinstance(result, dict) or not isinstance(result.get('models'), list):
        raise ValueError('Invalid native model capability registry')
    matches = [item for item in result['models'] if item.get('key') == model or
               any(instance.get('id') == model for instance in item.get('loaded_instances', []))]
    if len(matches) != 1:
        return {'status': 'model_not_advertised', 'reasoning': None}
    item = matches[0]
    reasoning = item.get('capabilities', {}).get('reasoning')
    if reasoning is not None:
        reasoning = ReasoningCapability.model_validate(reasoning).model_dump(mode='json')
    return {'status': 'advertised', 'model_key': item.get('key'),
            'reasoning': reasoning,
            'max_context_length': item.get('max_context_length')}


def route(policy, capability, seed, schema, temperature):
    """Do not send undocumented reasoning fields to chat/completions."""
    requested = policy['requested_mode']
    advertised = capability.get('reasoning') or {}
    if requested == 'auto':
        return 'openai_chat', advertised.get('default'), False
    if requested not in advertised.get('allowed_options', []):
        raise UnsupportedReasoningError(f'Reasoning mode {requested!r} is not advertised for this model',
                                        {'error_classification': 'unsupported_reasoning', 'capability': capability})
    if seed is not None or schema is not None or temperature > 1:
        raise UnsupportedTransportError(
            'Native LM Studio reasoning control cannot preserve this request: /api/v1/chat has no documented '
            'seed or JSON Schema parameter and supports temperature only up to 1. The OpenAI-compatible '
            'transport does not advertise enforceable on/off controls. Use auto explicitly or a backend '
            'with verified joint support; no settings were dropped and no inference was sent.',
            {'error_classification': 'unsupported_transport', 'capability': capability,
             'required_seed': seed, 'required_schema': schema.__name__ if schema else None})
    return 'lmstudio_native', requested, True


async def openai_events(client, request):
    stream = await client.chat.completions.create(**request)
    try:
        async for chunk in stream:
            usage = chunk.usage.model_dump() if chunk.usage else None
            if not chunk.choices:
                yield {'usage': usage, 'model': chunk.model}
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            yield {'final': delta.content or '',
                   'reasoning': getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None) or '',
                   'finish_reason': choice.finish_reason, 'usage': usage, 'model': chunk.model}
    finally:
        await stream.close()


async def native_events(client, config, request):
    # Native SSE is deliberately a separate adapter; never assume API parity.
    from openai import AsyncStream
    stream = await client.post(native_url(config.base_url, 'chat'), body=request,
                               cast_to=dict, stream=True, stream_cls=AsyncStream[dict])
    try:
        async for item in stream:
            kind = item.get('type')
            if kind in {'reasoning.delta', 'message.delta'}:
                yield {'reasoning' if kind == 'reasoning.delta' else 'final': item.get('content', '')}
            elif kind == 'chat.end':
                result = item.get('result', {})
                stats = result.get('stats', {})
                yield {'provider_status': 'chat.end', 'model': result.get('model_instance_id'),
                       'usage': {'prompt_tokens': stats.get('input_tokens'),
                                 'completion_tokens': stats.get('total_output_tokens'),
                                 'completion_tokens_details': {'reasoning_tokens': stats.get('reasoning_output_tokens')}}}
            elif kind == 'error':
                failure = classify(RuntimeError(json.dumps(item.get('error', item))), None)
                failure.diagnostics['backend_response'] = item
                raise failure
            else:
                # Model-load and prompt-processing progress are activity too.
                yield {}
    finally:
        await stream.close()


def classify(exc, schema):
    from .client import StructuredOutputRequestError
    if isinstance(exc, InferenceError):
        return exc
    message = str(exc).lower()
    if 'reasoning' in message and any(x in message for x in ('unsupported', 'not support', 'invalid', 'unrecognized')):
        return UnsupportedReasoningError('Backend rejected request-level reasoning control',
            {'error_classification': 'unsupported_reasoning', 'backend_response': getattr(exc, 'body', None)})
    if schema and (getattr(exc, 'status_code', None) == 501 or (any(x in message for x in ('response_format', 'json_schema', 'json schema', 'grammar', 'schema')) and (
            getattr(exc, 'status_code', None) in {400, 422, 501} or
            any(x in message for x in ('unsupported', 'not support', 'unexpected keyword'))))):
        return StructuredOutputRequestError('Schema-constrained generation rejected. No unconstrained fallback or retry.',
            {'error_classification': 'unsupported_schema', 'kind': 'structured_output_request_rejected',
             'status_code': getattr(exc, 'status_code', None), 'backend_response': getattr(exc, 'body', None)})
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or 'timeout' in type(exc).__name__.lower():
        kind = 'timeout'
    elif 'out of memory' in message or 'oom' in message:
        kind = 'out_of_memory'
    elif 'model' in message and any(x in message for x in ('not found', 'unavailable', 'not loaded')):
        kind = 'model_unavailable'
    else:
        kind = 'transport_failure'
    return InferenceError(f'{kind}: {type(exc).__name__}: {exc}', {'error_classification': kind})


async def complete_async(owner, system, user, config, planner, schema):
    from .client import json_schema_response_format
    policy = resolve_policy(config, planner)
    model = (config.planner_model or config.model) if planner else config.model
    seed = config.seed if planner or config.seed is None else config.seed + 1
    attempts = []
    started = time.monotonic()
    deadline = started + policy['timeout_seconds']  # Includes discovery, all retries and streaming.
    last_error = None
    for number in range(1, policy['recovery']['max_attempts'] + 1):
        if number > 1 and time.monotonic() >= deadline:
            break
        metrics: InferenceResult = {'role': policy['role'], 'requested_model': model, 'model': None, 'seed': seed,
            'temperature': config.temperature, 'top_p': config.top_p, 'max_tokens': policy['max_output_tokens'],
            'requested_max_tokens': config.planner_max_tokens if planner else config.max_tokens,
            'response_schema': schema.__name__ if schema else None, 'resolved_policy': policy,
            'policy_hash': policy['policy_hash'], 'requested_reasoning_mode': policy['requested_mode'],
            'effective_reasoning_mode': None, 'reasoning_mode_confirmed': False,
            'transport': None, 'endpoint': config.base_url, 'final_content': '', 'reasoning_content': '',
            'reasoning_content_truncated': False, 'usage': None, 'finish_reason': None,
            'provider_status': None, 'prompt_tokens': None, 'generated_tokens': None,
            'reasoning_tokens': None, 'final_answer_tokens': None, 'incomplete_generation': False,
            'budget_exhausted': False, 'watchdog_triggered': False, 'structured_validation': 'not_run',
            'error_classification': None, 'attempt_number': number}
        attempt_start = time.monotonic()
        async def attempt():
            async with owner._async_client(config) as client:
                try:
                    capability = await asyncio.wait_for(discover(client, config, model),
                        timeout=min(10, max(.001, deadline - time.monotonic())))
                except Exception as exc:
                    capability = {'status': 'discovery_failed', 'reasoning': None, 'error': type(exc).__name__}
                metrics['capability'] = capability
                transport, effective, confirmed = route(policy, capability, seed, schema, config.temperature)
                metrics.update(transport=transport, effective_reasoning_mode=effective, reasoning_mode_confirmed=confirmed)
                if transport == 'openai_chat':
                    request = {'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                        'temperature': config.temperature, 'top_p': config.top_p, 'stream': True,
                        'stream_options': {'include_usage': True}, 'max_tokens': policy['max_output_tokens']}
                    if seed is not None:
                        request['seed'] = seed
                    if schema:
                        request['response_format'] = json_schema_response_format(schema)
                    events = openai_events(client, request)
                else:
                    request = {'model': model, 'input': user, 'system_prompt': system, 'stream': True, 'store': False,
                        'integrations': [], 'reasoning': effective, 'temperature': config.temperature,
                        'top_p': config.top_p, 'max_output_tokens': policy['max_output_tokens']}
                    metrics['endpoint'] = native_url(config.base_url, 'chat')
                    events = native_events(client, config, request)
                watchdog = ReasoningWatchdog(policy['watchdog'])
                from .intake_watchdog import IntakeOutputWatchdog
                is_intake_review = schema is not None and schema.__name__ == 'SemanticReview'
                output_watchdog = (IntakeOutputWatchdog(policy['watchdog']['repetition_count'], review=is_intake_review)
                                  if schema and (policy['role'] == 'intake' or is_intake_review) else None)
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(anext(events), policy['watchdog']['stall_seconds'])
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            raise InferenceError('No stream event before stall deadline',
                                                 {'error_classification': 'stalled_generation'}) from exc
                        if event.get('usage') is not None:
                            metrics['usage'] = event['usage']
                        for key in ('finish_reason', 'model', 'provider_status'):
                            if event.get(key) is not None:
                                metrics[key] = event[key]
                        metrics['final_content'] += event.get('final', '')
                        if output_watchdog and event.get('final'):
                            evidence = output_watchdog.observe(metrics['final_content'])
                            if evidence:
                                metrics.update(watchdog_triggered=True, watchdog_evidence=evidence)
                                raise InferenceError('Repeated structured records in final JSON; generation stopped without repair',
                                                     {'error_classification': evidence['kind']})
                        reasoning = event.get('reasoning', '')
                        if reasoning:
                            diagnostic_limit = policy['watchdog']['diagnostic_chars']
                            tail = metrics['reasoning_content'] + reasoning
                            metrics['reasoning_content'] = tail[-diagnostic_limit:] if diagnostic_limit else ''
                            metrics['reasoning_content_truncated'] |= len(tail) > diagnostic_limit
                            evidence = watchdog.observe(reasoning, bool(metrics['final_content']))
                            if evidence:
                                metrics.update(watchdog_triggered=True, watchdog_evidence=evidence)
                                raise InferenceError('Reasoning watchdog terminated generation',
                                                     {'error_classification': evidence['kind']})
                        # Memory bound as a last resort against a provider ignoring its token cap.
                        if len(metrics['final_content']) > policy['max_output_tokens'] * 256:
                            raise InferenceError('Output exceeds diagnostic memory bound',
                                                 {'error_classification': 'output_budget_exhausted'})
                finally:
                    await events.aclose()
                usage = metrics['usage'] or {}
                generated = usage.get('completion_tokens')
                reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                metrics.update(prompt_tokens=usage.get('prompt_tokens'), generated_tokens=generated,
                    reasoning_tokens=reasoning_tokens,
                    final_answer_tokens=generated - reasoning_tokens if generated is not None and reasoning_tokens is not None and generated >= reasoning_tokens else None)
                if metrics['finish_reason'] == 'length' or (generated is not None and generated >= policy['max_output_tokens']):
                    raise InferenceError('Generation output budget exhausted; response is incomplete',
                                         {'error_classification': 'output_budget_exhausted'})
                if metrics['finish_reason'] not in {'stop', None} or (metrics['finish_reason'] is None and metrics['provider_status'] != 'chat.end'):
                    raise InferenceError('Stream ended without a successful provider completion status',
                                         {'error_classification': 'incomplete_generation'})
                if not metrics['final_content'].strip():
                    raise InferenceError('Model returned no final content', {'error_classification': 'missing_final_content'})
                if schema:
                    try:
                        json.loads(metrics['final_content'])
                    except (ValueError, TypeError) as exc:
                        raise InferenceError('Final content is not JSON', {'error_classification': 'json_parsing_error'}) from exc
                    try:
                        schema.model_validate_json(metrics['final_content'])
                    except ValidationError as exc:
                        raise InferenceError('Final JSON violates the response schema',
                            {'error_classification': 'schema_validation_error',
                             'validation_errors': exc.errors(include_input=False, include_context=False)}) from exc
                    metrics['structured_validation'] = 'passed'
                else:
                    metrics['structured_validation'] = 'not_applicable'
        try:
            await asyncio.wait_for(attempt(), timeout=max(.001, deadline - time.monotonic()))
        except Exception as exc:
            last_error = classify(exc, schema)
            metrics.update(last_error.diagnostics)
            if not isinstance(exc, InferenceError):
                metrics['backend_error'] = str(exc)[:4096]
                metrics['status_code'] = getattr(exc, 'status_code', None)
                metrics['backend_response'] = getattr(exc, 'body', None)
            metrics['incomplete_generation'] = True
            metrics['budget_exhausted'] = metrics['error_classification'] == 'output_budget_exhausted'
            if metrics['error_classification'] in {'json_parsing_error', 'schema_validation_error'}:
                metrics['structured_validation'] = 'failed'
            metrics['error'] = str(last_error)
        metrics['elapsed_seconds'] = time.monotonic() - attempt_start
        usage = metrics['usage'] or {}
        generated = usage.get('completion_tokens')
        reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        metrics.update(prompt_tokens=usage.get('prompt_tokens'), generated_tokens=generated,
            reasoning_tokens=reasoning_tokens,
            final_answer_tokens=generated - reasoning_tokens if generated is not None and reasoning_tokens is not None and generated >= reasoning_tokens else None)
        metrics['status'] = 'failed' if metrics['error_classification'] else 'succeeded'
        attempts.append(dict(metrics))
        if config.inference_artifact_dir:
            from .provenance import filesystem_path
            # Durable, exclusive per-attempt records survive a later retry interruption.
            path = filesystem_path(config.inference_artifact_dir) / f'{policy["role"]}_inference_attempt_{number:04d}.json'
            try:
                with path.open('x', encoding='utf-8') as file:
                    json.dump(metrics, file, indent=2, ensure_ascii=False, allow_nan=False)
                    file.flush()
                    import os
                    os.fsync(file.fileno())
            except OSError as exc:
                raise InferenceError('Cannot preserve inference attempt; stopping without retry',
                    {**metrics, 'status': 'failed', 'incomplete_generation': True,
                     'attempts': attempts, 'error_classification': 'artifact_write_failure',
                     'artifact_error': type(exc).__name__, 'artifact_error_message': str(exc),
                     'artifact_path': str(path)}) from exc
        if not metrics['error_classification']:
            return metrics['final_content'], {**metrics, 'attempts': attempts, 'total_elapsed_seconds': time.monotonic() - started}
        if metrics['error_classification'] not in policy['recovery']['retry_on']:
            break
    assert last_error is not None
    last_error.diagnostics = {**attempts[-1], 'attempts': attempts, 'total_elapsed_seconds': time.monotonic() - started}
    raise last_error


def complete(owner, system, user, config, planner, schema):
    # LangGraph's synchronous nodes normally have no event loop. Support direct
    # calls from notebook/event-loop threads without nesting asyncio.run.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(complete_async(owner, system, user, config, planner, schema))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(complete_async(owner, system, user, config, planner, schema))).result()
