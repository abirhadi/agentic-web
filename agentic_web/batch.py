"""B3 sequential execution of frozen B1 candidates through existing graphs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .experiments import ExperimentConfig, AgentConfiguration, build_plan, resolve_reference
from .schema import load_task
from .provenance import fingerprint, filesystem_path
from .variance import load_variance_context
from .workflow import build_graph, contract_hash
from .progress import log_event
from .checkpoint import atomic_json, execution_lock, progress


def now():
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def validated_plan(path):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    config = ExperimentConfig.model_validate(plan['configuration'])
    base = Path(plan['reference_base'])
    if not base.is_absolute():
        raise ValueError('Saved plan reference_base must be absolute')
    expected = build_plan(config, base, freeze_policy='reasoning_policy' in plan['configuration'])
    if plan != {**expected, 'reference_base': str(base)}:
        raise ValueError('Saved experiment plan differs from its validated B1 configuration')
    return plan


def candidate_inputs(candidate, base, output, *, base_url=None, resource_config=None):
    agents = AgentConfiguration.model_validate(candidate['model_configuration'])
    config = agents.inference_config(candidate['temperature'], candidate['top_p'], candidate['seed'])
    if base_url is not None:
        config = config.model_copy(update={'base_url': base_url})
    frozen = candidate['task']
    task = load_task(resolve_reference(base, frozen['specification_path']))
    if contract_hash(task) != frozen['specification_hash']:
        raise ValueError('Frozen specification hash changed before candidate execution')
    if frozen['decomposition_path']:
        from .decomposition import load_decomposition
        decomposition = load_decomposition(resolve_reference(base, frozen['decomposition_path']), task)
        if decomposition['structure_hash'] != frozen['decomposition_hash']:
            raise ValueError('Frozen decomposition hash changed before candidate execution')
    context = load_variance_context(candidate['variance_condition'], agents.workflow, base=base)
    inputs = {'task_spec': task.model_dump(mode='json'), 'model_config': config.model_dump(mode='json'),
              'output_root': str(output), 'variance_condition': context['condition']}
    if frozen['decomposition_path']:
        # Existing frozen-artifact branch validates/loads A3; it makes no inference call.
        inputs['decomposition_path'] = str(resolve_reference(base, frozen['decomposition_path']))
    if candidate['variance_condition']['profile_path']:
        inputs.update(variance_profile=str(resolve_reference(base, candidate['variance_condition']['profile_path'])),
                      variance_profile_hash=context['profile_hash'])
    if resource_config is not None:
        inputs['resource_config'] = resource_config
    return inputs


def outcome(result, output, exception=None, cleanup_errors=None):
    directories = sorted(Path(output).glob('*/metadata.json'))
    run = Path(result['run_dir']) if result.get('run_dir') else directories[0].parent if directories else None
    def read(name, default=None):
        file = run / name if run else None
        return json.loads(file.read_text(encoding='utf-8')) if file and file.is_file() else default
    metrics = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in run.glob('*_metrics.json')} if run else {}
    resources = {p.name: json.loads(p.read_text(encoding='utf-8')) for p in run.glob('resources*.json')} if run else {}
    if run:
        metrics.update({str(p.relative_to(run).with_suffix('')): json.loads(p.read_text(encoding='utf-8'))
                        for p in run.glob('implementation/**/*_metrics.json')})
        resources.update({str(p.relative_to(run)): json.loads(p.read_text(encoding='utf-8'))
                          for p in run.glob('implementation/**/resources*.json')})
    failure = read('failure.json', {})
    error = exception or result.get('error') or failure.get('error')
    stage = failure.get('phase') or ('workflow' if exception else None)
    artifact = result.get('output_file') or result.get('html_path')
    if not artifact and run:
        # Recover reference after an exception following successful implementation.
        task = read('task.json', {})
        entry = task.get('runtime', {}).get('entrypoint')
        if entry and (run / 'application' / entry).is_file():
            artifact = str(run / 'application' / entry)
    available = bool(artifact and Path(artifact).is_file())
    verification = result.get('verification') or read('verification_results.json', {})
    acceptance = result.get('acceptance') or read('acceptance.json')
    cleanup = read('resources.cleanup.json', {})
    errors = [error or '', *(cleanup_errors or []), *[str(m.get('error', '')) for m in metrics.values()]]
    diagnostic = ' '.join(errors)
    if cleanup_errors or cleanup.get('status') == 'failed' or any(x in diagnostic for x in ('ResourceAdmissionError', 'ResourcePlacement', 'resource admission')):
        status = 'resource_failure'
        stage = 'resource_cleanup' if cleanup_errors or cleanup.get('status') == 'failed' else stage
    elif any(m.get('finish_reason') in {'length', 'content_filter'} and m.get('role') != 'qa' for m in metrics.values()):
        status = 'generation_incomplete'
        stage = stage or next((m.get('role') for m in metrics.values() if m.get('finish_reason') in {'length', 'content_filter'}), None)
    elif any(x in diagnostic for x in ('ConnectionError', 'APIConnectionError', 'APIStatusError', 'BadRequestError', 'TimeoutError', 'RateLimitError', 'StructuredOutputRequestError', 'InferenceError', 'UnsupportedReasoningError', 'UnsupportedTransportError', 'Unavailable LM Studio model')) and not available:
        status = 'model_inference_failure'
    elif error:
        status = 'workflow_agent_failure'
    elif available:
        status = 'generation_completed_verification_passed' if verification.get('status') == 'passed' and (acceptance is None or acceptance['accepted']) else 'generation_completed_verification_failed'
    else:
        status = 'generation_incomplete'
    return {'execution_status': status, 'failure_stage': stage, 'error': error,
        'verification_outcome': result.get('verification_outcome') or read('implementation_summary.json', {}).get('outcome'),
        'implementation_feedback': read('implementation_summary.json'),
        'generation_completed': available, 'run_dir': str(run) if run else result.get('decomposition_run_dir'),
        'artifact': {'path': artifact, 'available': available},
        'verification_status': verification.get('status', 'missing'), 'acceptance': acceptance,
        'qa_status': result.get('qa_status'), 'qa_error': result.get('qa_error'),
        'metrics': metrics, 'resource_metrics': resources, 'cleanup_errors': cleanup_errors or []}


def run_experiment(plan_path, *, max_candidates=None, dry_run=False, base_url=None,
                   resource_config=None, client=None, verifier=None, resource_scheduler=None,
                   resume=False, max_new_candidates=None, retry_interrupted=False, reasoning_policy=None):
    for name, limit in [('max_candidates', max_candidates), ('max_new_candidates', max_new_candidates)]:
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f'{name} must be a positive integer')
    if (resume and max_candidates is not None) or (not resume and (max_new_candidates is not None or retry_interrupted)):
        raise ValueError('Resume controls and original run limits cannot be mixed')
    path = filesystem_path(plan_path).resolve()
    # Validation is read-only; revalidated again while holding ownership.
    plan = validated_plan(path)
    if reasoning_policy is not None:
        from .reasoning import ReasoningPolicy
        requested_hash = fingerprint(ReasoningPolicy.model_validate(reasoning_policy).model_dump(mode='json'))
        if any(c.get('reasoning_policy_hash') != requested_hash for c in plan['candidates']):
            raise ValueError('Reasoning policy differs from frozen plan; create a new experiment plan')
    if not dry_run and 'reasoning_policy_hash' not in plan:
        raise ValueError('Legacy plan has no frozen reasoning safeguards. Preserve it and create a new experiment ID/plan before execution')
    with execution_lock(path.parent):
        return _run_locked(path, max_candidates=max_candidates, dry_run=dry_run, base_url=base_url,
            resource_config=resource_config, client=client, verifier=verifier,
            resource_scheduler=resource_scheduler, resume=resume,
            max_new_candidates=max_new_candidates, retry_interrupted=retry_interrupted)


def _run_locked(path, *, max_candidates, dry_run, base_url, resource_config, client, verifier,
                resource_scheduler, resume, max_new_candidates, retry_interrupted):
    plan = validated_plan(path)
    base = Path(plan['reference_base'])
    if resource_config is not None:
        from .resource_schema import ResourceSettings
        resource_config = ResourceSettings.model_validate(resource_config).model_dump(mode='json')
    destination = path.parent / 'execution'
    if destination.exists() and not resume:
        raise FileExistsError(f'Execution destination already exists: {destination}; B3 never resumes or overwrites; use experiment resume')
    snapshot = progress(plan, destination)
    states = {c['candidate_id']: c for c in snapshot['candidates']}
    if resume:
        candidates = [c for c in plan['candidates'] if states[c['candidate_id']]['state'] == 'not_started' or
            (retry_interrupted and states[c['candidate_id']]['state'] == 'interrupted')][:max_new_candidates]
    else:
        candidates = plan['candidates'][:max_candidates]
    for candidate in candidates:
        candidate_inputs(candidate, base, destination / candidate['candidate_id'] / 'runs',
                         base_url=base_url, resource_config=resource_config)
    summary = {'experiment_id': plan['configuration']['experiment_id'], 'configuration_hash': plan['configuration_hash'],
        'plan_hash': fingerprint(plan), 'candidate_ids': [c['candidate_id'] for c in candidates],
        'selected_count': len(candidates), 'execution_directory': str(destination)}
    summary['reasoning_policies'] = {c['candidate_id']: {'policy_hash': c.get('reasoning_policy_hash'),
        'roles': {role: sampling.get('resolved_policy') for role, sampling in c['role_sampling'].items()}}
        for c in candidates}
    if dry_run:
        return {**summary, 'status': 'dry_run', 'progress': snapshot}
    if not destination.exists():
        destination.mkdir(exist_ok=False)
        exclusive_json(destination / 'batch_started.json', {**summary, 'started_at': now(),
            'base_url': base_url, 'resource_config': resource_config})
    records = []
    atomic_json(destination / 'progress.json', snapshot)
    try:
        for candidate in candidates:
            if validated_plan(path) != plan:
                raise ValueError('Experiment plan changed during execution')
            root = destination / candidate['candidate_id']
            prior = states[candidate['candidate_id']]['attempts']
            directory = root if not prior else root / 'attempts' / f'attempt-{len(prior) + 1:04d}'
            inputs = candidate_inputs(candidate, base, directory / 'runs', base_url=base_url, resource_config=resource_config)
            if prior:
                (root / 'attempts').mkdir(exist_ok=True)
            directory.mkdir(exist_ok=False)
            started = time.perf_counter()
            request = {'candidate_id': candidate['candidate_id'], 'candidate': candidate,
                'reasoning_policy_hash': candidate.get('reasoning_policy_hash'),
                'attempt_number': len(prior) + 1, 'operational_retry': bool(prior),
                'requested_model_config': inputs['model_config'], 'role_sampling': candidate['role_sampling'],
                'started_at': now(), 'resource_config': resource_config}
            exclusive_json(directory / 'started.json', request)
            atomic_json(destination / 'progress.json', progress(plan, destination))
            log_event('batch_candidate', 'started', candidate_id=candidate['candidate_id'])
            sessions, result, exception, cleanup_errors = [], {}, None, []
            try:
                graph = build_graph(client, verifier, workflow=candidate['model_configuration']['workflow'],
                                    resource_scheduler=resource_scheduler, session_observer=sessions.append)
                result = graph.invoke(inputs)
            except Exception as exc:
                exception = f'{type(exc).__name__}: {exc}'
            finally:
                for session in sessions:
                    try:
                        session.close()
                    except Exception as exc:
                        cleanup_errors.append(f'{type(exc).__name__}: {exc}')
            record = {**request, 'ended_at': now(), 'elapsed_seconds': time.perf_counter() - started,
                      **outcome(result, directory / 'runs', exception, cleanup_errors)}
            exclusive_json(directory / 'result.json', record)
            records.append({'candidate_id': candidate['candidate_id'], 'execution_status': record['execution_status'],
                            'record': str(directory / 'result.json')})
            atomic_json(destination / 'progress.json', progress(plan, destination))
            log_event('batch_candidate', 'finished', candidate_id=candidate['candidate_id'], status=record['execution_status'])
            if validated_plan(path) != plan:
                raise ValueError('Experiment plan changed during execution')
        summary.update(status='completed', outcomes=records, ended_at=now())
    except Exception as exc:
        summary.update(status='aborted', outcomes=records, ended_at=now(), error=f'{type(exc).__name__}: {exc}')
    finally:
        # Also checkpoints KeyboardInterrupt/SystemExit while propagating them.
        atomic_json(destination / 'progress.json', progress(plan, destination))
    summary['progress'] = progress(plan, destination)
    # B3 batch_result stays immutable. Resume publishes a separate mutable summary.
    if resume:
        atomic_json(destination / 'resume_result.json', summary)
    else:
        exclusive_json(destination / 'batch_result.json', summary)
    return summary
