"""Versioned inference policy. No model names, prompts, or resource placement decisions."""
from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from .schema import StrictModel
from .provenance import fingerprint

Mode = Literal['auto', 'off', 'on', 'low', 'medium', 'high']
ExplicitMode = Literal['off', 'on', 'low', 'medium', 'high']
ROLES = ('intake', 'decomposition', 'planner', 'ux', 'architecture', 'coordinator', 'implementer', 'qa')
ALIASES = {'requirements_analyst': 'intake', 'task_decomposition': 'decomposition', 'architect': 'architecture'}


def canonical_role(role):
    role = ALIASES.get(role, role)
    if role not in ROLES:
        raise ValueError(f'Unknown inference role: {role}')
    return role


class RolePolicy(StrictModel):
    mode: Mode = 'auto'
    max_output_tokens: int = Field(default=4096, gt=0, strict=True)
    timeout_seconds: float = Field(default=3000, gt=0, allow_inf_nan=False)


class ReasoningCapability(StrictModel):
    allowed_options: list[ExplicitMode] = Field(min_length=1)
    default: ExplicitMode | None = None

    @model_validator(mode='after')
    def consistent_default(self):
        if self.default is not None and self.default not in self.allowed_options:
            raise ValueError('Advertised reasoning default is not in allowed options')
        return self


class WatchdogPolicy(StrictModel):
    stall_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    max_reasoning_chars_without_final: int = Field(default=65536, ge=1024, strict=True)
    window_chars: int = Field(default=16384, ge=1024, le=131072, strict=True)
    segment_chars: int = Field(default=200, ge=100, strict=True)
    repetition_count: int = Field(default=4, ge=3, strict=True)
    diagnostic_chars: int = Field(default=4096, ge=0, le=16384, strict=True)

    @model_validator(mode='after')
    def fits_window(self):
        if self.segment_chars * self.repetition_count > self.window_chars:
            raise ValueError('Repetition evidence must fit in the rolling window')
        return self


class RecoveryPolicy(StrictModel):
    max_attempts: int = Field(default=1, ge=1, le=3, strict=True)
    retry_on: list[Literal['transport_failure', 'timeout', 'stalled_generation']] = Field(default_factory=list)


def default_roles():
    return {role: RolePolicy(max_output_tokens=16384 if role in {'intake', 'qa'} else
                            16000 if role == 'implementer' else
                            6144 if role in {'planner', 'coordinator'} else 4096) for role in ROLES}


class ReasoningPolicy(StrictModel):
    policy_version: Literal['1.0'] = '1.0'
    strict_capabilities: bool = True
    max_output_tokens: int = Field(default=32768, gt=0, strict=True)
    timeout_seconds: float = Field(default=3000, gt=0, allow_inf_nan=False)
    roles: dict[str, RolePolicy] = Field(default_factory=default_roles)
    watchdog: WatchdogPolicy = Field(default_factory=WatchdogPolicy)
    recovery: RecoveryPolicy = Field(default_factory=RecoveryPolicy)

    @model_validator(mode='before')
    @classmethod
    def normalize_roles(cls, value):
        if not isinstance(value, dict) or 'roles' not in value:
            return value
        roles = default_roles()
        seen = set()
        for key, item in value['roles'].items():
            role = canonical_role(key)
            if role in seen:
                raise ValueError(f'Duplicate policy role/alias: {role}')
            seen.add(role)
            roles[role] = {**roles[role].model_dump(), **item} if isinstance(item, dict) else item
        return {**value, 'roles': roles}


def load_policy(path):
    return ReasoningPolicy.model_validate_json(Path(path).read_text(encoding='utf-8'))


def resolve_policy(config, planner=False):
    role = canonical_role(config.inference_role or ('planner' if planner else 'implementer'))
    policy = config.reasoning_policy or ReasoningPolicy()
    settings = policy.roles[role]
    declared = config.planner_max_tokens if planner else config.max_tokens
    # All ceilings apply; a less restrictive override never bypasses a safety cap.
    limit = min(policy.max_output_tokens, settings.max_output_tokens, declared or policy.max_output_tokens)
    return {'role': role, 'requested_mode': settings.mode, 'max_output_tokens': limit,
            'timeout_seconds': min(policy.timeout_seconds, settings.timeout_seconds, config.timeout_seconds),
            'watchdog': policy.watchdog.model_dump(), 'recovery': policy.recovery.model_dump(),
            'strict_capabilities': policy.strict_capabilities, 'policy_hash': fingerprint(policy.model_dump(mode='json')),
            'policy_version': policy.policy_version}


class InferenceError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class UnsupportedReasoningError(InferenceError):
    pass


class UnsupportedTransportError(UnsupportedReasoningError):
    pass


class ReasoningWatchdog:
    def __init__(self, policy):
        self.policy = WatchdogPolicy.model_validate(policy)
        self.window = ''
        self.reasoning_chars = 0

    def observe(self, text, has_final):
        self.reasoning_chars += len(text)
        self.window = (self.window + text)[-self.policy.window_chars:]
        if not has_final and self.reasoning_chars >= self.policy.max_reasoning_chars_without_final:
            return {'kind': 'excessive_reasoning', 'reasoning_chars': self.reasoning_chars}
        # Long, word-rich segments only: punctuation, braces and short code motifs
        # are insufficient evidence. Count non-overlapping occurrences.
        size = self.policy.segment_chars
        if len(self.window) >= size * self.policy.repetition_count:
            for offset in range(0, len(self.window) - size + 1, max(1, size // 2)):
                segment = self.window[offset:offset + size]
                words = segment.split()
                if len(set(words)) < 12:
                    continue
                count = self.window.count(segment)
                if count >= self.policy.repetition_count:
                    return {'kind': 'repetition_watchdog', 'segment_chars': size, 'occurrences': count,
                            'window_chars': len(self.window), 'segment_hash': fingerprint(segment)}
        return None


def persist_failure(directory, role, exc):
    """Existing stage error handlers preserve failed attempts before fallback metrics."""
    diagnostics = getattr(exc, 'diagnostics', None)
    if not diagnostics or 'attempts' not in diagnostics:
        return
    import json
    root = Path(directory)
    for name, value in [(f'{role}_metrics.json', diagnostics), (f'{role}_inference_attempts.json', diagnostics['attempts'])]:
        (root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    (root / f'{role}_raw.txt').write_text(diagnostics.get('final_content', ''), encoding='utf-8')
