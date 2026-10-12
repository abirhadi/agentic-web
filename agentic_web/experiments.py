"""B1: validated experimental factors and deterministic planning, never execution."""
from collections import Counter
from itertools import product
import json
from math import prod
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .client import ModelConfig
from .reasoning import ReasoningPolicy, resolve_policy
from .decomposition import load_decomposition
from .provenance import fingerprint
from .schema import StrictModel, load_task
from .workflow import contract_hash


Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=100)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Role = Literal["planner", "implementer", "ux", "architect", "coordinator", "qa"]
BASELINE_ROLES = {"planner", "implementer"}
COLLABORATIVE_ROLES = BASELINE_ROLES | {"ux", "architect", "coordinator", "qa"}


class FrozenTask(StrictModel):
    name: Identifier
    specification_path: Text
    specification_hash: Digest
    decomposition_path: Text | None = None
    decomposition_hash: Digest | None = None

    @model_validator(mode="after")
    def paired_decomposition(self):
        if (self.decomposition_path is None) != (self.decomposition_hash is None):
            raise ValueError("Decomposition path and structure hash must be supplied together")
        return self


class ModelIdentity(StrictModel):
    model: Text
    family: Text | None = None
    parameter_count: int | None = Field(default=None, gt=0, strict=True)
    quantization: Text | None = None
    revision: Text | None = None


class AgentConfiguration(StrictModel):
    workflow: Literal["baseline", "collaborative"]
    roles: dict[Role, ModelIdentity]
    planner_max_tokens: int | None = Field(default=None, gt=0, strict=True)
    review_max_tokens: int | None = Field(default=None, gt=0, strict=True)
    max_tokens: int | None = Field(default=None, gt=0, strict=True)
    reasoning_policy: ReasoningPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    manual_verification: Literal['on', 'off'] | None = Field(default=None, exclude_if=lambda v: v is None)
    max_repair_attempts: int | None = Field(default=None, ge=0, le=10, strict=True, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def compatible_roles(self):
        expected = BASELINE_ROLES if self.workflow == "baseline" else COLLABORATIVE_ROLES
        if set(self.roles) != expected:
            raise ValueError(f"{self.workflow} requires exactly these explicit roles: {sorted(expected)}")
        if self.workflow == "baseline" and self.review_max_tokens is not None:
            raise ValueError("Baseline has no specialist review token limit")
        return self

    def inference_config(self, temperature, top_p, seed):
        # No environment defaults influence experimental assignments or sampling.
        return ModelConfig(model=self.roles["implementer"].model,
            **{role + "_model": identity.model for role, identity in self.roles.items() if role != "implementer"},
            temperature=temperature, top_p=top_p, seed=seed,
            planner_max_tokens=self.planner_max_tokens, review_max_tokens=self.review_max_tokens,
            max_tokens=self.max_tokens, reasoning_policy=self.reasoning_policy,
            manual_verification=self.manual_verification, max_repair_attempts=self.max_repair_attempts)

    def effective_identity(self):
        value = self.model_dump(mode="json")
        # Metadata and aliases do not turn identical inference assignments into replicates.
        value["roles"] = {role: {"model": identity.model, "quantization": identity.quantization,
                               "revision": identity.revision} for role, identity in self.roles.items()}
        return value


class VarianceCondition(StrictModel):
    identifier: Identifier
    profile_path: Text | None = None
    profile_hash: Digest | None = None

    @model_validator(mode="after")
    def paired_profile(self):
        if (self.profile_path is None) != (self.profile_hash is None):
            raise ValueError("Variance profile path and canonical JSON hash must be supplied together")
        return self


class Factors(StrictModel):
    model_configurations: list[Identifier] = Field(min_length=1)
    temperatures: list[Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)]] = Field(min_length=1)
    top_ps: list[Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]] = Field(min_length=1)
    seeds: list[Annotated[int, Field(strict=True)]] = Field(min_length=1)
    variance_conditions: list[VarianceCondition] = Field(min_length=1)
    replicates: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def unique_levels(self):
        for name in ("model_configurations", "temperatures", "top_ps", "seeds"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate factor levels: {name}; use replicates for intentional repetition")
        ids = [item.identifier for item in self.variance_conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("Variance identifiers must be unique")
        # Normalize IEEE negative zero, which is the same sampling factor as zero.
        self.temperatures = [0.0 if value == 0 else value for value in self.temperatures]
        return self


class PlanningControls(StrictModel):
    strategy: Literal["full_factorial"] = "full_factorial"
    candidate_limit: int | None = Field(default=None, ge=1, strict=True)
    selection_seed: int = Field(default=0, strict=True)


class ExperimentConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    experiment_id: Identifier
    experiment_version: Text
    description: Text
    output_root: Text
    tasks: list[FrozenTask] = Field(min_length=1)
    model_configurations: dict[Identifier, AgentConfiguration]
    factors: Factors
    planning: PlanningControls = Field(default_factory=PlanningControls)
    reasoning_policy: ReasoningPolicy | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def unique_inputs(self):
        names = [task.name for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("Task names must be unique")
        frozen = [(task.specification_hash, task.decomposition_hash) for task in self.tasks]
        if len(frozen) != len(set(frozen)):
            raise ValueError("Duplicate frozen task inputs; use replicates for intentional repetition")
        if set(self.factors.model_configurations) - self.model_configurations.keys():
            raise ValueError("Unknown named model configuration")
        identities = [fingerprint(self.model_configurations[name].effective_identity())
                      for name in self.factors.model_configurations]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate effective model configurations")
        return self


def read_config(path):
    return ExperimentConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def resolve_reference(base, reference):
    if "://" in reference or reference.startswith(("\\\\", "//")):
        raise ValueError("Experiment references must be local files")
    return (Path(base) / reference).resolve()


def validate_inputs(config, base):
    """Read-only validation, using existing canonical task and A3 validators."""
    for frozen in config.tasks:
        task = load_task(resolve_reference(base, frozen.specification_path))
        if contract_hash(task) != frozen.specification_hash:
            raise ValueError(f"Specification hash mismatch: {frozen.name}")
        if frozen.decomposition_path:
            record = load_decomposition(resolve_reference(base, frozen.decomposition_path), task)
            if record["structure_hash"] != frozen.decomposition_hash:
                raise ValueError(f"Decomposition hash mismatch: {frozen.name}")
    for condition in config.factors.variance_conditions:
        if condition.profile_path:
            value = json.loads(resolve_reference(base, condition.profile_path).read_text(encoding="utf-8"))
            if fingerprint(value) != condition.profile_hash:
                raise ValueError(f"Variance profile hash mismatch: {condition.identifier}")


def expected_counts(config):
    factors = config.factors
    total = prod([len(config.tasks), len(factors.model_configurations), len(factors.temperatures),
                  len(factors.top_ps), len(factors.seeds), len(factors.variance_conditions), factors.replicates])
    return total, min(total, config.planning.candidate_limit or total)


def candidate_matrix(config):
    """Pure deterministic function of validated config; no I/O or model calls."""
    factors = config.factors
    candidates = []
    for task, name, temperature, top_p, seed, variance, replicate in product(
            sorted(config.tasks, key=lambda item: item.name), sorted(factors.model_configurations),
            sorted(factors.temperatures), sorted(factors.top_ps), sorted(factors.seeds),
            sorted(factors.variance_conditions, key=lambda item: item.identifier), range(1, factors.replicates + 1)):
        agents = config.model_configurations[name]
        identity = {"candidate_schema_version": "1.0", "experiment_id": config.experiment_id,
            "experiment_version": config.experiment_version,
            "specification_hash": task.specification_hash, "decomposition_hash": task.decomposition_hash,
            "agents": agents.effective_identity(), "temperature": temperature, "top_p": top_p, "seed": seed,
            "variance": {"identifier": variance.identifier, "profile_hash": variance.profile_hash}, "replicate": replicate}
        inference = agents.inference_config(temperature, top_p, seed)
        role_sampling = {}
        for role in sorted(agents.roles):
            role_config, planner = inference.for_role(role)
            role_sampling[role] = {"seed": role_config.seed if planner else role_config.seed + 1,
                "temperature": role_config.temperature, "top_p": role_config.top_p,
                "max_tokens": role_config.planner_max_tokens if planner else role_config.max_tokens}
            if agents.reasoning_policy is not None:
                role_sampling[role]['resolved_policy'] = resolve_policy(role_config, planner)
        candidates.append({"candidate_id": "candidate-" + fingerprint(identity), **identity,
            "task": task.model_dump(mode="json"), "model_configuration_name": name,
            "model_configuration": agents.model_dump(mode="json"),
            "variance_condition": variance.model_dump(mode="json"), "role_sampling": role_sampling})
    ids = [candidate["candidate_id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate identities")
    limit = config.planning.candidate_limit
    if limit is not None and limit < len(candidates):
        # Hash ranking is reproducible across Python versions and independent of list order.
        candidates.sort(key=lambda candidate: (fingerprint({"seed": config.planning.selection_seed,
            "candidate_id": candidate["candidate_id"]}), candidate["candidate_id"]))
        candidates = candidates[:limit]
    return sorted(candidates, key=lambda candidate: candidate["candidate_id"])


def factor_summary(config, candidates):
    levels = {"task": [task.name for task in config.tasks],
        "model_configuration": config.factors.model_configurations,
        "temperature": config.factors.temperatures, "top_p": config.factors.top_ps,
        "seed": config.factors.seeds, "variance_condition": [v.identifier for v in config.factors.variance_conditions],
        "replicate": list(range(1, config.factors.replicates + 1))}
    extract = {"task": lambda c: c["task"]["name"], "model_configuration": lambda c: c["model_configuration_name"],
        "variance_condition": lambda c: c["variance_condition"]["identifier"]}
    result = {}
    for factor, values in levels.items():
        counts = Counter(extract[factor](c) if factor in extract else c[factor] for c in candidates)
        result[factor] = [{"value": value, "candidate_count": counts[value]} for value in sorted(values)]
    return result


def build_plan(config, base, *, freeze_policy=True):
    if freeze_policy:
        policy = config.reasoning_policy or ReasoningPolicy()
        config = config.model_copy(update={'reasoning_policy': policy, 'model_configurations': {
            key: agents.model_copy(update={'reasoning_policy': agents.reasoning_policy or policy})
            for key, agents in config.model_configurations.items()}})
    validate_inputs(config, base)
    total, selected = expected_counts(config)
    candidates = candidate_matrix(config)
    assert len(candidates) == selected
    result = {"plan_schema_version": "1.0", "configuration_hash": fingerprint(config.model_dump(mode="json")),
        "configuration": config.model_dump(mode="json"), "full_factorial_count": total,
        "planned_candidate_count": selected, "selection_method": "sha256_rank" if selected < total else "full_factorial",
        "factor_summary": factor_summary(config, candidates), "candidates": candidates}
    if config.reasoning_policy is not None:
        result['reasoning_policy_hash'] = fingerprint(config.reasoning_policy.model_dump(mode='json'))
        for candidate in candidates:
            candidate['reasoning_policy_hash'] = fingerprint(candidate['model_configuration']['reasoning_policy'])
    if any(a.manual_verification is not None or a.max_repair_attempts is not None for a in config.model_configurations.values()):
        from .schema import verification_readiness
        result['verification_readiness'] = {t.name: verification_readiness(load_task(resolve_reference(base, t.specification_path)))
                                            for t in config.tasks}
    return result


def save_plan(plan, config_path):
    """Exclusive write; an identical existing plan is reusable, any change is rejected."""
    config = ExperimentConfig.model_validate(plan["configuration"])
    base = Path(config_path).resolve().parent
    directory = resolve_reference(base, config.output_root) / config.experiment_id
    destination = directory / "experiment_plan.json"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump({**plan, "reference_base": str(base)}, stream, indent=2, ensure_ascii=False, allow_nan=False)
    except FileExistsError:
        old = json.loads(destination.read_text(encoding="utf-8"))
        if old != {**plan, "reference_base": str(base)}:
            raise ValueError("Existing experiment plan differs; choose a new experiment ID/output root") from None
    return destination
