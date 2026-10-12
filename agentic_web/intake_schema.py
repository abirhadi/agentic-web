"""A1 analysis provenance around the existing reusable task manifest."""
from copy import deepcopy
from typing import Annotated, Literal, Union, get_args
from pydantic import Field, create_model, model_validator
from .schema import Action, Assertion, Feature, RuntimeSpec, StrictModel, TaskSpec, TestCase, VerificationSettings


ANALYSIS_VERSION = "1.0"


def quotation_choices(text):
    """Bounded exact excerpts for constrained decoding, never rewritten text."""
    choices = [text] if len(text) <= 1000 and text.strip() else []
    for line in text.splitlines():
        if not line.strip():
            continue
        for offset in range(0, len(line), 1000):
            excerpt = line[offset:offset + 1000].strip()
            if excerpt:
                choices.append(excerpt)
        choices.extend(line.split())
    return list(dict.fromkeys(choices))


class IntakeSource(StrictModel):
    original_request: str = Field(min_length=1)
    supporting_context: str | None = None

    @model_validator(mode="after")
    def nonblank(self):
        if not self.original_request.strip():
            raise ValueError("Provide a nonblank natural-language request")
        return self


class SourceReference(StrictModel):
    source: Literal["request", "context"]
    quote: str = Field(min_length=1)


class RequirementTrace(StrictModel):
    requirement_id: str
    origin: Literal["explicit", "inferred", "recommendation"]
    rationale: str = Field(min_length=1)
    sources: list[SourceReference] = Field(min_length=1)


class ConstraintTrace(StrictModel):
    target: Literal["runtime", "technology_constraint"]
    index: int | None = Field(default=None, ge=0)
    origin: Literal["explicit", "inferred", "framework_default"]
    rationale: str = Field(min_length=1)
    sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_target(self):
        if (self.target == "runtime") != (self.index is None):
            raise ValueError("Only technology_constraint traces have an index")
        if self.origin != "framework_default" and not self.sources:
            raise ValueError("User-derived constraints require source references")
        if self.origin == "framework_default" and self.sources:
            raise ValueError("Framework defaults must not cite user/context sources")
        return self


class IntakeIssue(StrictModel):
    issue_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(min_length=1)
    question: str = Field(min_length=1)
    blocks_generation: bool
    requirement_ids: list[str]
    sources: list[SourceReference] = Field(default_factory=list)


class RequirementsAnalysis(StrictModel):
    analysis_version: Literal["1.0"] = ANALYSIS_VERSION
    task_spec: TaskSpec
    requirement_traceability: list[RequirementTrace]
    constraint_traceability: list[ConstraintTrace]
    missing_information: list[IntakeIssue]
    unresolved_ambiguities: list[IntakeIssue]

    @model_validator(mode="after")
    def grounded_contract(self):
        task = self.task_spec
        requirements = {requirement.id: requirement for requirement in task.requirements}
        from .intake_watchdog import requirement_signature
        from .provenance import fingerprint
        seen_outcomes = {}
        for requirement in task.requirements:
            value = requirement.model_dump(mode='json')
            value['tests'] = [test.model_dump(mode='json') for test in task.tests
                              if test.requirement_id == requirement.id]
            signature = fingerprint(requirement_signature(value))
            if signature in seen_outcomes:
                raise ValueError(f'Duplicate requirement outcomes: {seen_outcomes[signature]}, {requirement.id}')
            seen_outcomes[signature] = requirement.id
        trace_ids = [trace.requirement_id for trace in self.requirement_traceability]
        if len(trace_ids) != len(set(trace_ids)) or set(trace_ids) != set(requirements):
            raise ValueError("Traceability must cover every requirement exactly once, with no unknown IDs")
        for trace in self.requirement_traceability:
            if trace.origin == "recommendation" and requirements[trace.requirement_id].required:
                raise ValueError("Design recommendations must remain optional")
        for requirement in requirements.values():
            if any(not criterion.strip() for criterion in requirement.acceptance_criteria):
                raise ValueError("Observable success criteria cannot be blank")
            tests = [test for test in task.tests if test.requirement_id == requirement.id]
            if not tests:
                raise ValueError(f"Every requirement needs an acceptance test or manual review: {requirement.id}")
            for test in tests:
                if test.required != requirement.required:
                    raise ValueError("Tests must preserve their requirement's mandatory/optional status")
                if requirement.verification == "manual" and test.test_type != "manual":
                    raise ValueError("Manual criteria cannot be represented as executable verification")
                if requirement.verification == "browser" and test.test_type == "manual":
                    raise ValueError("Manual tests must label their requirement verification as manual")
        if not task.verification.enabled or not task.verification.accessibility:
            raise ValueError("Intake cannot disable the existing verification or accessibility checks")
        targets = [(trace.target, trace.index) for trace in self.constraint_traceability]
        expected = {("runtime", None)} | {("technology_constraint", index) for index in range(len(task.technology_constraints))}
        if len(targets) != len(set(targets)) or set(targets) != expected:
            raise ValueError("Trace every runtime/technology constraint exactly once")
        issues = self.missing_information + self.unresolved_ambiguities
        if len({issue.issue_id for issue in issues}) != len(issues):
            raise ValueError("Missing-information and ambiguity IDs must be unique")
        for issue in issues:
            if set(issue.requirement_ids) - set(requirements):
                raise ValueError("Issues reference unknown requirement IDs")
            if issue.blocks_generation and issue.requirement_ids and not any(
                    requirements[identifier].required for identifier in issue.requirement_ids):
                raise ValueError("Optional requirements cannot become generation blockers")
        source_values = {"request": task.original_requirements, "context": task.supporting_context}
        traces = [*self.requirement_traceability, *self.constraint_traceability, *issues]
        for trace in traces:
            for reference in trace.sources:
                source = source_values[reference.source]
                if source is None or reference.quote not in source:
                    raise ValueError("Source quotations must be exact substrings of the original request or supplied context")
        return self


def _fields(model, exclude=()):
    return {name: (field.annotation, deepcopy(field)) for name, field in model.model_fields.items()
            if name not in exclude}


def assemble_analysis(response, *, supporting_context: str | None = None) -> RequirementsAnalysis:
    """Project declared, nested references into the unchanged canonical contract.

    No missing requirements are invented and no model-provided IDs are rewritten.
    Tests and provenance belong to their containing requirement by construction.
    """
    payload = response.model_dump(mode="json")
    task = payload["task_spec"]
    # Evidence is authoritative input, not generated data. Do not ask the model
    # to echo it or embed a potentially large document in its response schema.
    task["supporting_context"] = supporting_context
    runtime_trace = task["runtime"].pop("traceability")
    payload["constraint_traceability"] = [{"target": "runtime", "index": None, **runtime_trace}]
    technology = task["technology_constraints"]
    task["technology_constraints"] = [entry["value"] for entry in technology]
    payload["constraint_traceability"].extend({"target": "technology_constraint", "index": index, **entry["traceability"]}
        for index, entry in enumerate(technology))
    tests, traces = [], []
    for group in ("features", "nonfunctional_requirements"):
        for requirement in task[group]:
            for issue_group in ("missing_information", "unresolved_ambiguities"):
                payload[issue_group].extend({"requirement_ids": [requirement["id"]], **issue}
                    for issue in requirement.pop(issue_group))
            traces.append({"requirement_id": requirement["id"], **requirement.pop("traceability")})
            tests.extend({"requirement_id": requirement["id"], "required": requirement["required"], **test}
                         for test in requirement.pop("tests"))
    task["tests"] = tests
    payload["requirement_traceability"] = traces
    # IDs are bookkeeping, not model-authored semantics. Keep every issue and
    # its scope intact; assign unique IDs once when freezing the canonical data.
    for group, prefix in (("missing_information", "missing"), ("unresolved_ambiguities", "ambiguity")):
        for index, issue in enumerate(payload[group], 1):
            issue["issue_id"] = f"{prefix}_{index:03d}"
    return RequirementsAnalysis.model_validate(payload)


def analysis_response_schema(source: IntakeSource) -> type[StrictModel]:
    """Keep requirement definitions, tests and provenance in one schema entry."""
    reference_models = []
    for name, text in (("request", source.original_request), ("context", source.supporting_context)):
        if text and text.strip():
            reference_models.append(create_model(name.title() + "SourceReference", __base__=SourceReference,
                source=(Literal.__getitem__((name,)), ...),
                quote=(Literal.__getitem__(tuple(quotation_choices(text))), ...)))
    reference_model = Union.__getitem__(tuple(reference_models)) if len(reference_models) > 1 else reference_models[0]
    requirement_trace = create_model("IntakeRequirementTrace", __base__=StrictModel,
        **_fields(RequirementTrace, ("requirement_id", "sources")),
        sources=(list[reference_model], Field(min_length=1)))
    issue_model = create_model("IntakeIssue", __base__=StrictModel,
        **_fields(IntakeIssue, ("issue_id", "requirement_ids", "sources")),
        requirement_ids=(list[str], Field(default_factory=list, max_length=0)),
        sources=(list[reference_model], Field(default_factory=list)))
    nested_issue_model = create_model("RequirementIssue", __base__=StrictModel,
        **_fields(IntakeIssue, ("issue_id", "requirement_ids", "sources")),
        sources=(list[reference_model], Field(default_factory=list)))

    constraint_fields = _fields(ConstraintTrace, ("target", "index", "origin", "sources"))
    user_constraint = create_model("UserConstraintProvenance", __base__=StrictModel, **constraint_fields,
        origin=(Literal["explicit", "inferred"], ...), sources=(list[reference_model], Field(min_length=1)))
    default_constraint = create_model("DefaultConstraintProvenance", __base__=StrictModel, **constraint_fields,
        origin=(Literal["framework_default"], ...), sources=(list[reference_model], Field(default_factory=list, max_length=0)))
    constraint_type = Union[user_constraint, default_constraint]
    runtime_model = create_model("IntakeRuntime", __base__=RuntimeSpec, traceability=(constraint_type, ...))
    static_runtime = create_model("StaticIntakeRuntime", __base__=runtime_model,
        kind=(Literal["single_html", "static_site"], ...), entrypoint=(str, Field(min_length=1)))
    app_entry_runtime = create_model("AppEntryIntakeRuntime", __base__=runtime_model,
        kind=(Literal["web_app"], ...), entrypoint=(str, Field(min_length=1)))
    app_url_runtime = create_model("AppURLIntakeRuntime", __base__=runtime_model,
        kind=(Literal["web_app"], ...), base_url=(get_args(RuntimeSpec.model_fields["base_url"].annotation)[0], ...))
    service_runtime = create_model("ServiceIntakeRuntime", __base__=runtime_model,
        kind=(Literal["frontend", "full_stack", "api"], ...))
    runtime_type = Union[static_runtime, app_entry_runtime, app_url_runtime, service_runtime]
    technology_model = create_model("IntakeTechnologyConstraint", __base__=StrictModel,
        value=(str, Field(min_length=1, pattern=r"\S")), traceability=(constraint_type, ...))
    verification_model = create_model("IntakeVerification", __base__=VerificationSettings,
        enabled=(Literal[True], True), accessibility=(Literal[True], True))

    assertion_type = Union[
        create_model("CountAssertion", __base__=Assertion, assertion=(Literal["count"], ...), value=(int, Field(ge=0, strict=True))),
        create_model("TextAssertion", __base__=Assertion, assertion=(Literal["text", "contains_text", "value"], ...), value=(str, ...)),
        create_model("NameAssertion", __base__=Assertion, assertion=(Literal["accessible_name"], ...),
                     value=(Annotated[str, Field(min_length=1, pattern=r"\S")] | None, None)),
        create_model("StateAssertion", __base__=Assertion, assertion=(Literal["visible", "hidden", "checked", "not_checked"], ...))]
    action_type = Union[
        create_model("ValueAction", __base__=Action, action=(Literal["fill", "select"], ...),
                     value=(str, Field(max_length=2000)), assert_after=(list[assertion_type], Field(default_factory=list, max_length=30))),
        create_model("KeyAction", __base__=Action, action=(Literal["press"], ...),
                     value=(Literal["Enter", "Tab", "Escape", "Space"], ...), assert_after=(list[assertion_type], Field(default_factory=list, max_length=30))),
        create_model("ControlAction", __base__=Action, action=(Literal["click", "check", "uncheck"], ...),
                     assert_after=(list[assertion_type], Field(default_factory=list, max_length=30)))]

    checks_type = TestCase.model_fields["universal_checks"].annotation
    test_fields = _fields(TestCase, ("requirement_id", "required", "test_type", "universal_checks", "expected_results", "preconditions", "actions"))
    browser_test = create_model("BrowserAcceptanceTest", __base__=StrictModel, **test_fields,
        test_type=(Literal["browser"], "browser"),
        preconditions=(list[assertion_type], Field(default_factory=list, max_length=30)),
        actions=(list[action_type], Field(default_factory=list, max_length=50)),
        universal_checks=(checks_type, Field(default_factory=list, max_length=0)),
        expected_results=(list[assertion_type], Field(min_length=1, max_length=30)))
    universal_test = create_model("UniversalAcceptanceTest", __base__=StrictModel, **test_fields,
        test_type=(Literal["universal"], ...), universal_checks=(checks_type, Field(min_length=1)),
        preconditions=(list[Assertion], Field(default_factory=list, max_length=0)),
        actions=(list[Action], Field(default_factory=list, max_length=0)),
        expected_results=(list[Assertion], Field(default_factory=list, max_length=0)))
    manual_test = create_model("ManualAcceptanceTest", __base__=StrictModel, **test_fields,
        test_type=(Literal["manual"], ...), universal_checks=(checks_type, Field(default_factory=list, max_length=0)),
        preconditions=(list[Assertion], Field(default_factory=list, max_length=0)),
        actions=(list[Action], Field(default_factory=list, max_length=0)),
        expected_results=(list[Assertion], Field(default_factory=list, max_length=0)))
    requirement_fields = _fields(Feature, ("verification", "acceptance_criteria"))
    requirement_fields["acceptance_criteria"] = (list[Annotated[str, Field(min_length=1, pattern=r"\S")]], Field(min_length=1))
    browser_requirement = create_model("BrowserRequirement", __base__=StrictModel, **requirement_fields,
        missing_information=(list[nested_issue_model], Field(default_factory=list)),
        unresolved_ambiguities=(list[nested_issue_model], Field(default_factory=list)),
        verification=(Literal["browser", "api", "static"], "browser"),
        traceability=(requirement_trace, ...),
        tests=(list[Union[browser_test, universal_test]], Field(min_length=1, max_length=100)))
    manual_requirement = create_model("ManualRequirement", __base__=StrictModel, **requirement_fields,
        missing_information=(list[nested_issue_model], Field(default_factory=list)),
        unresolved_ambiguities=(list[nested_issue_model], Field(default_factory=list)),
        verification=(Literal["manual"], ...), traceability=(requirement_trace, ...),
        tests=(list[manual_test], Field(min_length=1, max_length=100)))
    required_trace = create_model("RequiredRequirementTrace", __base__=requirement_trace,
        origin=(Literal["explicit", "inferred"], ...))
    optional_issue = create_model("OptionalRequirementIssue", __base__=nested_issue_model,
        blocks_generation=(Literal[False], False))
    requirement_models = []
    for base in (browser_requirement, manual_requirement):
        requirement_models.append(create_model("Required" + base.__name__, __base__=base,
            required=(Literal[True], True), traceability=(required_trace, ...)))
        requirement_models.append(create_model("Optional" + base.__name__, __base__=base,
            required=(Literal[False], ...),
            missing_information=(list[optional_issue], Field(default_factory=list)),
            unresolved_ambiguities=(list[optional_issue], Field(default_factory=list))))
    requirement_type = Union.__getitem__(tuple(requirement_models))
    task_model = create_model("IntakeTaskSpec", __base__=StrictModel,
        **_fields(TaskSpec, ("original_requirements", "supporting_context", "features", "nonfunctional_requirements", "tests", "runtime", "technology_constraints", "verification")),
        runtime=(runtime_type, ...),
        technology_constraints=(list[technology_model], Field(default_factory=list)),
        verification=(verification_model, Field(default_factory=verification_model)),
        original_requirements=(Literal.__getitem__((source.original_request,)), ...),
        features=(list[requirement_type], Field(min_length=1)),
        nonfunctional_requirements=(list[requirement_type], Field(default_factory=list)))
    return create_model("RequirementsAnalysisResponse", __base__=StrictModel,
        analysis_version=(Literal["1.0"], ANALYSIS_VERSION), task_spec=(task_model, ...),
        missing_information=(list[issue_model], ...), unresolved_ambiguities=(list[issue_model], ...))
