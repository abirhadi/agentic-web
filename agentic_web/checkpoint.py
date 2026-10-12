"""Local durable summaries and OS-owned execution locks; records remain authoritative."""
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
from uuid import uuid4

from .provenance import fingerprint, filesystem_path

TERMINAL = {'generation_completed_verification_passed', 'generation_completed_verification_failed',
            'generation_incomplete', 'model_inference_failure', 'workflow_agent_failure', 'resource_failure'}


def atomic_json(path, value):
    path = filesystem_path(path)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def execution_lock(directory):
    # The kernel releases the lock on process exit. Never delete the lock file:
    # replacing its inode would permit two simultaneous owners on POSIX.
    path = Path(directory) / '.execution.lock'
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another writer owns this experiment execution directory') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_record(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
        return None


def validate_identity(record, candidate):
    if record is None:
        return
    for key, value in [('candidate_id', candidate['candidate_id']), ('candidate', candidate),
                       ('role_sampling', candidate['role_sampling'])]:
        if key in record and record[key] != value:
            raise ValueError(f'Execution record disagrees with frozen candidate: {key}')
    if candidate.get('reasoning_policy_hash') is not None and 'reasoning_policy_hash' in record and record['reasoning_policy_hash'] != candidate['reasoning_policy_hash']:
        raise ValueError('Execution record changed frozen reasoning policy hash')
    if 'requested_model_config' in record:
        from .experiments import AgentConfiguration
        from .client import ModelConfig
        try:
            actual = ModelConfig.model_validate(record['requested_model_config'])
        except ValueError:
            return  # Malformed partial record; valid_terminal rejects it below.
        agents = AgentConfiguration.model_validate(candidate['model_configuration'])
        expected = agents.inference_config(candidate['temperature'], candidate['top_p'], candidate['seed'])
        for key in ('model', 'planner_model', 'ux_model', 'architect_model', 'coordinator_model', 'qa_model',
                    'temperature', 'top_p', 'seed', 'max_tokens', 'planner_max_tokens', 'review_max_tokens', 'reasoning_policy',
                    'manual_verification', 'max_repair_attempts'):
            if getattr(actual, key) != getattr(expected, key):
                raise ValueError(f'Execution record changed candidate inference setting: {key}')


def valid_terminal(record, location):
    from datetime import datetime
    required = {'candidate_id', 'candidate', 'requested_model_config', 'role_sampling', 'started_at', 'ended_at',
                'execution_status', 'generation_completed', 'artifact', 'verification_status', 'run_dir',
                'metrics', 'resource_metrics', 'failure_stage', 'error', 'elapsed_seconds'}
    if not record or not required <= record.keys() or not isinstance(record['execution_status'], str) or record['execution_status'] not in TERMINAL:
        return False
    if not isinstance(record['candidate'], dict):
        return False
    if record['candidate'].get('reasoning_policy_hash') is not None and record.get('reasoning_policy_hash') != record['candidate']['reasoning_policy_hash']:
        return False
    try:
        from .client import ModelConfig
        ModelConfig.model_validate(record['requested_model_config'])
        start, end = datetime.fromisoformat(record['started_at']), datetime.fromisoformat(record['ended_at'])
        if end < start or not isinstance(record['elapsed_seconds'], (int, float)) or record['elapsed_seconds'] < 0:
            return False
        artifact = record['artifact']
        if type(record['generation_completed']) is not bool or type(artifact['available']) is not bool:
            return False
        if record['generation_completed'] != artifact['available']:
            return False
        if artifact['available'] and (not isinstance(artifact['path'], str) or not artifact['path']):
            return False
        if not isinstance(record['metrics'], dict) or not isinstance(record['resource_metrics'], dict):
            return False
        if record['run_dir'] is not None:
            run = filesystem_path(record['run_dir']).resolve()
            if not run.is_relative_to(filesystem_path(location / 'runs').resolve()):
                return False
        if artifact['available'] and (not record['run_dir'] or not filesystem_path(artifact['path']).resolve().is_relative_to(run)):
            return False
        if record['execution_status'].startswith('generation_completed_') and not record['generation_completed']:
            return False
        if record['execution_status'] == 'generation_completed_verification_passed' and (
                record['verification_status'] != 'passed' or
                (record.get('acceptance') is not None and record['acceptance'].get('accepted') is not True)):
            return False
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
    return True


def progress(plan, directory):
    directory = filesystem_path(directory)
    initial = read_record(directory / 'batch_started.json')
    if initial and (initial.get('plan_hash') != fingerprint(plan) or initial.get('configuration_hash') != plan['configuration_hash']):
        raise ValueError('Execution directory belongs to a different experiment plan')
    expected = {c['candidate_id'] for c in plan['candidates']}
    if directory.exists() and any(p.is_dir() and p.name.startswith('candidate-') and p.name not in expected for p in directory.iterdir()):
        raise ValueError('Execution directory contains an unknown candidate')
    candidates = []
    for candidate in plan['candidates']:
        root = directory / candidate['candidate_id']
        attempts = []
        # B3's root record is attempt 1. Retry records never replace it.
        locations = [root] if root.exists() else []
        if (root / 'attempts').exists():
            locations += sorted((root / 'attempts').iterdir())
        terminal = None
        for location in locations:
            if not location.is_dir():
                raise ValueError('Invalid candidate attempt directory')
            started, result = read_record(location / 'started.json'), read_record(location / 'result.json')
            validate_identity(started, candidate)
            validate_identity(result, candidate)
            if terminal is not None:
                raise ValueError('Unexpected attempt after a terminal candidate result')
            valid = valid_terminal(result, location)
            if valid:
                terminal = result
            attempts.append({'directory': str(location), 'started_record': str(location / 'started.json'),
                'result_record': str(location / 'result.json'), 'terminal': valid,
                'run_dir': result.get('run_dir') if result else None,
                'run_directories': [str(p) for p in sorted((location / 'runs').glob('*')) if p.is_dir()],
                'artifact': result.get('artifact') if result else None,
                'diagnostic': None if valid else 'Missing or invalid terminal result; interrupted'})
        state = ('completed' if terminal['generation_completed'] else 'failed') if terminal else 'interrupted' if attempts else 'not_started'
        candidates.append({'candidate_id': candidate['candidate_id'], 'state': state, 'attempts': attempts,
            'execution_status': terminal['execution_status'] if terminal else None,
            'verification_status': terminal['verification_status'] if terminal else None,
            'record': attempts[-1]['result_record'] if terminal else None})
    counts = Counter(c['state'] for c in candidates)
    return {'total_planned_candidates': len(candidates),
        'counts': {name: counts[name] for name in ('not_started', 'interrupted', 'completed', 'failed')},
        'execution_attempts': sum(len(c['attempts']) for c in candidates),
        'generation_outcomes': dict(Counter(c['execution_status'] for c in candidates if c['execution_status'])),
        'verification_outcomes': dict(Counter(c['verification_status'] for c in candidates if c['verification_status'])),
        'all_candidates_terminal': counts['not_started'] + counts['interrupted'] == 0,
        'candidates': candidates}
