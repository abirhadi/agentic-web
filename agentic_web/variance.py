"""B2 advisory treatment context; no inference, mutation of tasks, or execution."""
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator
from .schema import StrictModel
from .provenance import fingerprint

Role = Literal['planner', 'ux', 'architect', 'coordinator', 'implementer']
Text = Annotated[str, Field(min_length=1, max_length=3000, pattern=r'\S')]


class VarianceProfile(StrictModel):
    schema_version: Literal['1.0']
    profile_id: str = Field(pattern=r'^[a-z][a-z0-9_-]*$', max_length=100)
    dimensions: list[Annotated[str, Field(min_length=1, max_length=100, pattern=r'\S')]] = Field(min_length=1, max_length=20)
    directive: Text
    role_guidance: dict[Role, Text] = Field(default_factory=dict)

    @model_validator(mode='after')
    def unique_dimensions(self):
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError('Variance dimensions must be unique')
        return self


AUTHORITY = (
    '\n\nEXPERIMENTAL VARIANCE GUIDANCE (advisory, subordinate to the task)\n'
    'Priority: original user requirements and constraints > frozen task specification > experimental variance guidance. '
    'Explore only choices left open by the authoritative task. Disregard conflicting profile guidance. '
    'Never change requirements, frozen decomposition, acceptance criteria, mandatory selectors, accessibility, '
    'runtime/technology constraints, or sampling parameters. Do not promote optional features to requirements. '
    'Explore alternative solutions without prescribing a fixed final design.\n'
)


def load_variance_context(condition, workflow, *, base='.'):
    """Accept B1's VarianceCondition or its JSON dictionary, with identical raw JSON hashing."""
    from .experiments import VarianceCondition, resolve_reference
    condition = VarianceCondition.model_validate(condition)
    if workflow not in {'baseline', 'collaborative'}:
        raise ValueError('Unknown variance workflow')
    if condition.identifier == 'control':
        if condition.profile_path is not None:
            raise ValueError('Control must not have a variance profile')
        return {'condition': 'control', 'profile_id': None, 'profile_hash': None,
                'dimensions': [], 'roles': [], 'effective_directives': {}}
    if condition.profile_path is None:
        raise ValueError('Non-control generation requires a variance profile')
    raw = json.loads(resolve_reference(base, condition.profile_path).read_text(encoding='utf-8'))
    profile = VarianceProfile.model_validate(raw)
    digest = fingerprint(raw)
    if digest != condition.profile_hash:
        raise ValueError('Variance profile hash mismatch')
    roles = ['planner', 'implementer'] if workflow == 'baseline' else ['planner', 'ux', 'architect', 'coordinator', 'implementer']
    directives = {role: AUTHORITY + json.dumps({'condition': condition.identifier,
        'profile_id': profile.profile_id, 'dimensions': profile.dimensions,
        'directive': profile.directive, 'role_guidance': profile.role_guidance.get(role)},
        sort_keys=True, ensure_ascii=False) for role in roles}
    return {'condition': condition.identifier, 'profile_id': profile.profile_id, 'profile_hash': digest,
            'dimensions': profile.dimensions, 'roles': roles, 'effective_directives': directives,
            'profile': raw}


def generation_variance(state, workflow):
    path = state.get('variance_profile')
    expected = state.get('variance_profile_hash')
    if expected and not path:
        raise ValueError('Variance profile hash requires a profile')
    # B1 requires paired references/hashes; the single-run CLI can calculate an
    # unpinned hash once, then snapshot the validated profile before inference.
    if path and expected is None:
        from .experiments import resolve_reference
        expected = fingerprint(json.loads(resolve_reference('.', path).read_text(encoding='utf-8')))
    return load_variance_context({'identifier': state.get('variance_condition', 'control'),
        'profile_path': path, 'profile_hash': expected}, workflow)


def condition_prompt(state, role, system):
    # QA and any future non-generation role are excluded, even if called directly.
    if role not in {'planner', 'ux', 'architect', 'coordinator', 'implementer'}:
        return system
    return system + state.get('variance_context', {}).get('effective_directives', {}).get(role, '')
