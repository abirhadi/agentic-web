"""Validated requirements and an executable, deliberately bounded browser DSL."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from .task_base import Feature, RuntimeSpec, TaskSpec as BaseTaskSpec


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Assertion(StrictModel):
    assertion: Literal["visible", "hidden", "count", "text", "contains_text", "checked", "not_checked", "value", "accessible_name"]
    selector: str = Field(min_length=1, max_length=500)
    value: str | int | None = None

    @model_validator(mode="after")
    def valid_value(self):
        if not self.selector.strip():
            raise ValueError('Assertion selector cannot be blank')
        if self.assertion == 'contains_text' and (not isinstance(self.value, str) or not self.value.strip()):
            raise ValueError('contains_text requires nonempty observable text')
        if self.assertion == "count" and (type(self.value) is not int or self.value < 0):
            raise ValueError("count requires a nonnegative integer")
        if self.assertion in {"text", "contains_text", "value"} and not isinstance(self.value, str):
            raise ValueError("Text/value assertions require a string")
        if self.assertion == "accessible_name" and self.value is not None and (not isinstance(self.value, str) or not self.value.strip()):
            raise ValueError("accessible_name requires a nonempty string or no value (any nonempty name)")
        return self


class Action(StrictModel):
    action: Literal["click", "fill", "select", "check", "uncheck", "press"]
    selector: str = Field(min_length=1, max_length=500)
    value: str | None = Field(default=None, max_length=2000)
    assert_after: list[Assertion] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def valid_value(self):
        if self.action in {"fill", "select", "press"} and self.value is None:
            raise ValueError(f"{self.action} requires a value")
        if self.action == "press" and self.value not in {"Enter", "Tab", "Escape", "Space"}:
            raise ValueError("Unsupported key")
        return self


class TestCase(StrictModel):
    test_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    requirement_id: str
    test_type: Literal["browser", "manual", "universal"] = "browser"
    universal_checks: list[Literal["html_structure", "desktop_loads", "mobile_loads", "desktop_renders", "mobile_renders",
                                  "desktop_no_horizontal_overflow", "mobile_no_horizontal_overflow",
                                  "desktop_basic_accessibility", "mobile_basic_accessibility",
                                  "desktop_no_javascript_errors", "mobile_no_javascript_errors"]] = Field(default_factory=list)
    preconditions: list[Assertion] = Field(default_factory=list, max_length=30)
    actions: list[Action] = Field(default_factory=list, max_length=50)
    expected_results: list[Assertion] = Field(default_factory=list, max_length=30)
    required: bool = True

    @model_validator(mode="after")
    def observable(self):
        if self.test_type == "browser" and not self.expected_results:
            raise ValueError("Executable tests must contain observable assertions")
        if self.test_type == "universal" and (not self.universal_checks or self.actions or self.preconditions or self.expected_results):
            raise ValueError("Universal tests require only allowlisted universal_checks")
        if self.test_type != "universal" and self.universal_checks:
            raise ValueError("universal_checks requires universal test type")
        return self


class VerificationSettings(StrictModel):
    enabled: bool = True
    timeout_ms: int = Field(default=5000, ge=100, le=30000)
    total_timeout_seconds: int = Field(default=120, ge=5, le=600)
    screenshots: bool = True
    accessibility: bool = True
    required_content: list[str] = Field(default_factory=list, max_length=30)


class TaskSpec(BaseTaskSpec):
    # Short natural-language requests remain verbatim; do not pad them to fit
    # the legacy free-text length threshold.
    original_requirements: str = Field(min_length=1)
    # Omit absent context so existing manifests retain their canonical hashes.
    supporting_context: str | None = Field(default=None, exclude_if=lambda value: value is None)
    description: str = Field(min_length=1, max_length=10000)
    nonfunctional_requirements: list[Feature] = Field(default_factory=list)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    tests: list[TestCase] = Field(default_factory=list, max_length=100)

    @property
    def requirements(self) -> list[Feature]:
        return self.features + self.nonfunctional_requirements

    @model_validator(mode="after")
    def valid_contract(self):
        if not self.original_requirements.strip():
            raise ValueError("Original requirements cannot be blank")
        ids = [r.id for r in self.requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("All requirement IDs must be unique")
        test_ids = [t.test_id for t in self.tests]
        if len(set(test_ids)) != len(test_ids):
            raise ValueError("Test IDs must be unique")
        requirements = {r.id: r for r in self.requirements}
        if any(not criterion.strip() for r in self.requirements for criterion in r.acceptance_criteria):
            raise ValueError('Acceptance criteria cannot be empty')
        for test in self.tests:
            if test.requirement_id not in requirements:
                raise ValueError(f"Unknown requirement: {test.requirement_id}")
            if requirements[test.requirement_id].required and not test.required:
                raise ValueError("Tests for mandatory requirements must be required")
        entry = self.runtime.entrypoint
        if entry:
            path = PurePosixPath(entry)
            if path.is_absolute() or ".." in path.parts or "\\" in entry or ":" in entry:
                raise ValueError("Entrypoint must be a relative path within the run directory")
        return self


def verification_readiness(task):
    """Compile the existing observable DSL; never infer selectors from prose."""
    compiled, gaps = [], []
    for test in task.tests:
        reason = None
        if test.test_type == 'manual':
            reason = 'manual_only'
        elif not task.verification.enabled:
            reason = 'browser_verification_disabled'
        elif test.test_type == 'universal' and not task.verification.accessibility and any(
                'accessibility' in check for check in test.universal_checks):
            reason = 'required_check_disabled'
        if reason:
            gaps.append({'requirement_id': test.requirement_id, 'test_id': test.test_id, 'reason': reason})
        else:
            compiled.append(test.model_dump(mode='json'))
    for requirement in task.requirements:
        if not any(t.requirement_id == requirement.id for t in task.tests):
            gaps.append({'requirement_id': requirement.id, 'test_id': None,
                         'reason': 'manual_only' if requirement.verification == 'manual' else 'no_executable_test'})
    return {'executable_tests': compiled, 'gaps': gaps,
            'has_automated_coverage': bool(compiled),
            'mandatory_coverage_complete': all(any(t['requirement_id'] == r.id and t['required'] for t in compiled)
                                              for r in task.requirements if r.required),
            'basis': 'Manifest observable assertions and allowlisted universal checks; schema validity alone is insufficient'}


class ImplementationPlan(StrictModel):
    architecture: str = Field(min_length=1)
    components: list[str] = Field(min_length=1)
    data_models: list[str]
    interactions: list[str]
    technology_constraints: list[str]
    implementation_steps: list[str] = Field(min_length=1)
    risks: list[str]
    assumptions: list[str]
    requirement_ids: list[str]


def implementation_plan_schema(task: TaskSpec) -> type[ImplementationPlan]:
    """Lock authoritative constraints in both the generation grammar and validation.

    Runtime configuration is supplied separately; it is not permission to add
    technology constraints. Capture a snapshot so later mutations cannot change
    this request's contract.
    """
    expected = tuple(task.technology_constraints)

    def preserve_constraints(value):
        if tuple(value) != expected:
            raise ValueError("Planner changed technology constraints")
        return value

    return create_model("ImplementationPlan", __base__=ImplementationPlan,
        technology_constraints=(list[str], Field(json_schema_extra={"const": list(expected)},
            description="Exact authoritative technology_constraints, including an empty list. Do not add runtime settings or inferred constraints.")),
        __validators__={"preserve_constraints": field_validator("technology_constraints")(preserve_constraints)})


def load_task(path) -> TaskSpec:
    from pathlib import Path
    return TaskSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
