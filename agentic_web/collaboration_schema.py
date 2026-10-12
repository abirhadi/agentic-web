"""Small validated communication contracts; all recommendations remain advisory."""
from typing import Annotated, Literal, Union
from pydantic import Field, create_model, field_validator
from .schema import StrictModel, TaskSpec, ImplementationPlan


class PlanningContext(StrictModel):
    original_requirements: str
    task_spec: TaskSpec
    plan: ImplementationPlan
    runtime_constraints: dict


class Recommendation(StrictModel):
    requirement_ids: list[str] = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class UXReview(StrictModel):
    interaction_flows: list[str]
    components_and_states: list[str]
    accessibility_and_responsiveness: list[str]
    error_and_empty_states: list[str]
    risks_and_ambiguities: list[str]
    recommendations: list[Recommendation]


class ArchitectureReview(StrictModel):
    implementation_structure: list[str]
    components: list[str]
    state_and_data_flow: list[str]
    event_and_data_contracts: list[str]
    runtime_feasibility: str = Field(min_length=1)
    risks_dependencies_and_ambiguities: list[str]
    recommendations: list[Recommendation]


def grounded_specialist_schema(task: TaskSpec, review_schema):
    """Bind advisory recommendations to this manifest before generation."""
    identifiers = tuple(requirement.id for requirement in task.requirements)
    recommendation = create_model("Recommendation", __base__=Recommendation,
        requirement_ids=(list[Literal.__getitem__(identifiers)], Field(min_length=1,
            description="Exact requirement IDs from task_spec; never test IDs, field names, or variance identifiers.")))
    return create_model(review_schema.__name__, __base__=review_schema,
        recommendations=(list[recommendation], ...))


class ConflictResolution(StrictModel):
    requirement_ids: list[str] = Field(min_length=1)
    ux_position: str
    architecture_position: str
    resolution: str = Field(min_length=1)
    basis: str = Field(min_length=1)

class CoordinationProposal(StrictModel):
    """LLM-generated coordination decisions; no authoritative contract fields."""

    architecture_and_responsibilities: list[str] = Field(min_length=1)
    state_data_and_interaction_contracts: list[str]
    ux_accessibility_constraints: list[str]
    implementation_instructions: list[str] = Field(min_length=1)
    agreements: list[str]
    conflict_resolutions: list[ConflictResolution]
    unresolved_issues: list[str]
    requirement_traceability: dict[str, list[str]]


def grounded_coordination_schema(task: TaskSpec) -> type[CoordinationProposal]:
    """Expose immutable requirement coverage to the decoder and validate it locally."""
    identifiers = tuple(requirement.id for requirement in task.requirements)
    required = tuple(requirement.id for requirement in task.requirements if requirement.required)
    instructions = Annotated[list[Annotated[str, Field(min_length=1, pattern=r"\S")]], Field(min_length=1)]
    item_schema = {"type": "array", "minItems": 1,
                   "items": {"type": "string", "minLength": 1, "pattern": r"\S"}}

    def validate_coverage(value):
        missing = sorted(set(required) - value.keys())
        unknown = sorted(value.keys() - set(identifiers))
        if missing or unknown:
            raise ValueError(f"Coordinator requirement traceability: missing mandatory IDs={missing}; unknown IDs={unknown}")
        return value

    return create_model("CoordinationProposal", __base__=CoordinationProposal,
        requirement_traceability=(dict[str, instructions], Field(json_schema_extra={
            "properties": {identifier: item_schema for identifier in identifiers},
            "required": list(required), "additionalProperties": False},
            description="Include every mandatory requirement ID with nonblank implementation instructions. Optional IDs may be omitted.")),
        __validators__={"validate_coverage": field_validator("requirement_traceability")(validate_coverage)})


class CoordinatedSpecification(CoordinationProposal):
    """Validated proposal combined with framework-authoritative fields."""

    technology_constraints: list[str]
    runtime_kind: str
    mandatory_contract_hash: str


class EvidenceReference(StrictModel):
    key: str
    quote: str = Field(min_length=1)


class QAFinding(StrictModel):
    classification: Literal["observed", "hypothesis"]
    diagnostic_role: Literal["primary_defect", "secondary_symptom", "independent"]
    category: Literal["functional", "interface_contract", "accessibility", "runtime_exception", "incomplete_output", "other"]
    description: str = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    test_ids: list[str]
    evidence: list[EvidenceReference] = Field(min_length=1)
    test_outcomes: dict[str, Literal["passed", "failed", "unverified", "skipped"]] = Field(default_factory=dict)
    likely_cause: str | None = None
    corrective_action: str = Field(min_length=1)


class QAReview(StrictModel):
    summary: str = Field(min_length=1)
    findings: list[QAFinding]


def grounded_qa_review_schema(evidence_registry: dict[str, str], task: TaskSpec) -> type[QAReview]:
    """Bind QA identifiers and paired evidence to this run's authoritative inputs.

    Full registry values are valid exact-substring quotations. Constraining both
    fields together prevents a schema-aware backend from inventing identifiers,
    prose quotations or mismatched key/value pairs. Normal reference, evidence
    and diagnostic validators remain independent checks on the returned review.
    """
    if not evidence_registry:
        raise ValueError("QA evidence registry cannot be empty")
    keys_by_quote: dict[str, list[str]] = {}
    for key, quote in evidence_registry.items():
        if quote:
            keys_by_quote.setdefault(quote, []).append(key)
    if not keys_by_quote:
        raise ValueError("QA evidence registry has no quotable values")
    evidence_models = [create_model(f"EvidenceReference{index}", __base__=EvidenceReference,
        key=(Literal.__getitem__(tuple(keys)), ...),
        quote=(Literal.__getitem__((quote,)), Field(min_length=1)))
        for index, (quote, keys) in enumerate(keys_by_quote.items())]
    evidence_model = Union.__getitem__(tuple(evidence_models))
    requirement_literal = Literal.__getitem__(tuple(requirement.id for requirement in task.requirements))
    test_ids = tuple(test.test_id for test in task.tests)
    test_literal = Literal.__getitem__(test_ids) if test_ids else str
    finding_model = create_model("QAFinding", __base__=QAFinding,
        requirement_ids=(list[requirement_literal], Field(min_length=1,
            description="Existing requirement IDs from task_spec; never acceptance-test IDs.")),
        test_ids=(list[test_literal], Field(description="Existing test_id values from task_spec.tests.",
            **({"max_length": 0} if not test_ids else {}))),
        evidence=(list[evidence_model], Field(min_length=1)))
    return create_model("QAReview", __base__=QAReview, findings=(list[finding_model], ...))


class CoordinationContext(PlanningContext):
    ux_review: UXReview
    architecture_review: ArchitectureReview
    mandatory_implementation_contract: dict
    mandatory_contract_hash: str


class QAContext(StrictModel):
    original_requirements: str
    task_spec: TaskSpec
    coordinated_specification: CoordinatedSpecification | None = None
    verification: dict
    source_excerpt: str
    source_truncated: bool
    evidence_registry: dict[str, str]
