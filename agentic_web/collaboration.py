"""Independent specialists, structured coordination, advisory QA and deterministic gate."""
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from .client import ModelConfig, json_schema_response_format
from pydantic import ValidationError
from .collaboration_schema import (ArchitectureReview, CoordinatedSpecification, CoordinationContext,
                                   PlanningContext, QAContext, QAReview, UXReview, grounded_qa_review_schema, grounded_coordination_schema,
                                   grounded_specialist_schema)
from .implementation_contract import derive_implementation_contract
from .provenance import fingerprint, record_stage
from .variance import condition_prompt


def _helpers():
    # Shared workflow persistence and requirements validation; no second runner.
    from .workflow import authoritative_task, failure, metadata, write_json
    return authoritative_task, failure, metadata, write_json


def validate_references(task, ids):
    unknown = set(ids) - {requirement.id for requirement in task.requirements}
    if unknown:
        raise ValueError(f"Unknown requirement IDs: {sorted(unknown)}")


def evidence_registry(report):
    registry = {"verification.status": str(report.get("status", "unverified"))}
    for name, value in report.get("checks", {}).items():
        registry[f"checks.{name}"] = json.dumps(value)
    for test in report.get("tests", []):
        for name in ("status", "error", "phase", "runtime_errors"):
            if name in test:
                value = test[name]
                registry[f"tests.{test['test_id']}.{name}"] = value if isinstance(value, str) else json.dumps(value)
    for identifier, coverage in report.get("coverage", {}).items():
        registry[f"coverage.{identifier}.status"] = str(coverage.get("status", "unverified"))
    for index, error in enumerate(report.get("errors", [])):
        registry[f"errors.{index}"] = str(error)
    for index, diagnostic in enumerate(report.get("javascript_diagnostics", [])):
        for name, value in diagnostic.items():
            registry[f"javascript_diagnostics.{index}.{name}"] = str(value)
    return registry


def validate_evidence(reference, registry):
    if reference.key not in registry:
        raise ValueError(f"QA references unknown evidence key: {reference.key}")
    if "..." in reference.quote or "…" in reference.quote:
        raise ValueError(f"QA evidence quote contains an ellipsis: {reference.key}")
    if reference.quote not in registry[reference.key]:
        raise ValueError(f"QA evidence quote is not an exact substring: {reference.key}")


def validate_diagnostic_hierarchy(review, report, registry):
    syntax_indexes = {str(index) for index, diagnostic in enumerate(report.get("javascript_diagnostics", []))
                      if diagnostic.get("kind") == "syntax_error"}
    timeout_tests = {test["test_id"] for test in report.get("tests", [])
                     if "TimeoutError" in str(test.get("error", ""))}
    if not syntax_indexes or not timeout_tests:
        return
    syntax_keys = {key for key in registry if any(key.startswith(f"javascript_diagnostics.{index}.")
                   for index in syntax_indexes)}
    primary = [finding for finding in review.findings
               if finding.classification == "observed" and finding.diagnostic_role == "primary_defect"
               and finding.category == "runtime_exception"
               and any(reference.key in syntax_keys for reference in finding.evidence)]
    if not primary:
        raise ValueError("QA must identify the observed JavaScript syntax error as the primary defect")
    secondary = [finding for finding in review.findings
                 if finding.diagnostic_role == "secondary_symptom" and set(finding.test_ids) & timeout_tests
                 and any(reference.key.startswith("tests.") and reference.key.endswith(".error")
                         and "TimeoutError" in registry.get(reference.key, "") for reference in finding.evidence)]
    if not secondary:
        raise ValueError("QA must identify downstream interaction timeouts as grounded secondary symptoms")
    for finding in review.findings:
        if set(finding.test_ids) & timeout_tests and finding.category != "runtime_exception" \
                and finding.diagnostic_role != "secondary_symptom":
            raise ValueError("QA must classify downstream interaction timeouts as secondary symptoms")


def make_collaboration_nodes(client, invoke_model, close_resources):
    def context(state):
        authoritative_task, _, _, _ = _helpers()
        task = authoritative_task(state)
        approved = PlanningContext(original_requirements=task.original_requirements, task_spec=task,
                                   plan=state["plan"], runtime_constraints=task.runtime.model_dump(mode="json"))
        return task, approved

    def invoke(state, role, schema, payload, instructions):
        _, _, _, write_json = _helpers()
        config, planner = ModelConfig.model_validate(state["model_config"]).for_role(role)
        system = (f"You are the {role} agent. Return ONLY JSON matching output_schema. "
                  "Authority order: immutable original requirements/task, mandatory deterministic acceptance contracts, "
                  "planner/specialist advice, then coordinator guidance. Never add mandatory features, remove requirements, "
                  "change acceptance criteria, or promote optional requirements. " + instructions)
        user = json.dumps({"role": role, **payload, "output_schema": schema.model_json_schema()}, ensure_ascii=False)
        system = condition_prompt(state, role, system)
        write_json(state["run_dir"], f"{role}_request.json", {"system": system, "user": user,
            "response_format": json_schema_response_format(schema),
            "inference_config": {"requested_model": config.model, "seed": (config.seed + 1 if config.seed is not None else None), "max_tokens": config.max_tokens,
                                 "temperature": config.temperature, "top_p": config.top_p, "base_url": config.base_url}})
        started = time.perf_counter()
        try:
            raw, metrics = invoke_model(state, role, system, user, config, planner=planner, response_schema=schema)
        except Exception as exc:
            write_json(state["run_dir"], f"{role}_metrics.json", {"role": role, "requested_model": config.model,
                "seed": (config.seed + 1 if config.seed is not None else None), "elapsed_seconds": time.perf_counter() - started, "error": str(exc)})
            from .reasoning import persist_failure
            persist_failure(state["run_dir"], role, exc)
            if getattr(exc, "diagnostics", None):
                write_json(state["run_dir"], f"{role}_structured_output_error.json", exc.diagnostics)
            raise RuntimeError(f"{role} model '{config.model}' failed: {type(exc).__name__}: {exc}") from exc
        metrics = {**metrics, "role": role, "requested_model": config.model,
                   "seed": (config.seed + 1 if config.seed is not None else None), "max_tokens": metrics.get("max_tokens", config.max_tokens),
                   "incomplete_generation": metrics.get("finish_reason") == "length",
                   "temperature": config.temperature, "top_p": config.top_p}
        (Path(state["run_dir"]) / f"{role}_raw.txt").write_text(raw, encoding="utf-8")
        write_json(state["run_dir"], f"{role}_metrics.json", metrics)
        if metrics.get("finish_reason") in {"length", "content_filter"}:
            raise ValueError(f"Incomplete {role} output: finish_reason={metrics['finish_reason']}")
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1])
        try:
            return schema.model_validate_json(cleaned)
        except ValidationError as exc:
            write_json(state["run_dir"], f"{role}_validation_error.json", {"response_schema": schema.__name__,
                       "errors": exc.errors(include_input=False, include_context=False)})
            raise

    def specialist(role, schema, field, filename, instructions):
        def node(state):
            _, _, _, write_json = _helpers()
            if state.get("error"):
                record_stage(state, role, "skipped", details={"error": state["error"]})
                return {role + "_error": "Upstream stage failed"}
            try:
                task, approved = context(state)
                # Only approved planning context is forwarded. The sibling's
                # output is never part of either specialist's request.
                response_schema = grounded_specialist_schema(task, schema)
                review = invoke(state, role, response_schema, approved.model_dump(mode="json"), instructions)
                for recommendation in review.recommendations:
                    validate_references(task, recommendation.requirement_ids)
                value = review.model_dump(mode="json")
                write_json(state["run_dir"], filename, value)
                record_stage(state, role, "completed")
                return {field: value}
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                write_json(state["run_dir"], f"{role}_error.json", {"role": role, "error": message})
                record_stage(state, role, "failed", details={"error": message})
                return {role + "_error": message}
        return node

    ux = specialist("ux", UXReview, "ux_review", "ux_review.json",
        "Independently analyze interaction flows, component states, accessibility, responsiveness, errors/empty states, risks and ambiguities. "
        "Associate recommendations with existing requirement IDs; do not assume another specialist's opinions.")
    architect = specialist("architect", ArchitectureReview, "architecture_review", "architecture_review.json",
        "Independently analyze implementation structure, components, state/data flow, events, data contracts, dependencies and feasibility "
        "within the declared runtime. Associate recommendations with existing requirement IDs; no invented backend infrastructure.")

    def coordinator(state):
        _, failure, metadata, write_json = _helpers()
        if state.get("error"):
            record_stage(state, "coordinator", "skipped")
            return {}
        try:
            errors = {role: state[role + "_error"] for role in ("ux", "architect") if state.get(role + "_error")}
            if errors:
                raise ValueError(f"Specialist reviews unavailable: {errors}")
            task, approved = context(state)
            contract = derive_implementation_contract(task)
            digest = fingerprint(contract)
            joined = CoordinationContext(**approved.model_dump(mode="json"), ux_review=state["ux_review"],
                architecture_review=state["architecture_review"], mandatory_implementation_contract=contract, mandatory_contract_hash=digest)

            # The Coordinator generates decisions, not authoritative contract fields.
            proposal = invoke(
                                state,
                                "coordinator",
                                grounded_coordination_schema(task),
                                joined.model_dump(mode="json"),
                                (
                                    "Reconcile both independent reviews. "
                                    "Explicitly record agreements, conflicts, resolutions "
                                    "grounded in the task, and unresolved issues. "
                                    "Provide one coherent implementation proposal and "
                                    "trace every mandatory requirement to implementation instructions. "
                                    "requirement_traceability must include every required key in output_schema, including nonfunctional and exclusion requirements. "
                                    "Each included key needs at least one nonblank instruction; mentioning an ID elsewhere does not supply its traceability entry. "
                                    "Preserve mandatory selectors and observable behaviors. "
                                    "Do not introduce new mandatory technology constraints. "
                                    "Runtime, technology constraints, and the mandatory contract hash "
                                    "are maintained by the framework, not generated by you. "
                                    "Guidance is advisory."
                                ),
                            )
            validate_references(task, proposal.requirement_traceability)
            required = {
                requirement.id
                for requirement in task.requirements
                if requirement.required
            }

            if (
                not required <= proposal.requirement_traceability.keys()
                or any(
                    not notes
                    for notes in proposal.requirement_traceability.values()
                )
            ):
                raise ValueError(
                    "Coordinator must trace every mandatory requirement "
                    "to nonempty instructions"
                )

            for resolution in proposal.conflict_resolutions:
                validate_references(task, resolution.requirement_ids)

            # Assemble the final specification using authoritative framework data.
            spec = CoordinatedSpecification.model_validate({
                **proposal.model_dump(mode="json"),
                "technology_constraints": list(task.technology_constraints),
                "runtime_kind": task.runtime.kind,
                "mandatory_contract_hash": digest,
            })

            # Defensive contract-integrity validation.
            if (
                spec.mandatory_contract_hash != digest
                or spec.technology_constraints != task.technology_constraints
                or spec.runtime_kind != task.runtime.kind
            ):
                raise ValueError(
                    "Coordinator changed the mandatory acceptance contract "
                    "or runtime/technology constraints"
                )
            value = spec.model_dump(mode="json")
            write_json(state["run_dir"], "coordination.json", value)
            record_stage(state, "coordinator", "completed")
            metadata(state, status="coordinated")
            return {"coordinated_specification": value, "status": "coordinated"}
        except Exception as exc:
            return failure(state, "coordinator", exc)

    def qa(state):
        authoritative_task, _, _, write_json = _helpers()
        if not state.get("verification") or not state.get("html_path"):
            message = "Insufficient verifier/artifact evidence; QA not executed"
            write_json(state["run_dir"], "qa_error.json", {"error": message, "status": "skipped"})
            record_stage(state, "qa", "skipped", details={"error": message})
            return {"qa_status": "skipped", "qa_error": message}
        try:
            task = authoritative_task(state)
            if state.get("qa_model_unavailable"):
                raise RuntimeError(f"QA model unavailable: '{state['qa_model_unavailable']}'; no substitution")
            source = Path(state["html_path"]).read_text(encoding="utf-8")
            excerpt = source if len(source) <= 80000 else source[:60000] + "\n[EXCERPT TRUNCATED]\n" + source[-20000:]
            registry = evidence_registry(state["verification"])
            approved = QAContext(original_requirements=task.original_requirements, task_spec=task,
                coordinated_specification=state.get("coordinated_specification"), verification=state["verification"],
                source_excerpt=excerpt, source_truncated=len(source) > 80000,
                evidence_registry=registry)
            qa_schema = grounded_qa_review_schema(registry, task)
            grounded_review = invoke(state, "qa", qa_schema, approved.model_dump(mode="json"),
                "Analyze actual verifier diagnostics and source evidence only. Clearly distinguish observed facts from hypotheses. "
                "MANDATORY EVIDENCE RULES: evidence[].key must exactly equal a key supplied in evidence_registry; source_excerpt is analysis context, not an evidence key. "
                "The output schema restricts evidence keys and quotations to this run's registry. Select the full registry value paired with the chosen key for every evidence[].quote; never invent or paraphrase evidence. "
                "IDENTIFIER RULES: requirement_ids contains only task_spec requirement IDs; test_ids contains only task_spec.tests test_id values. "
                "Use each test's requirement_id to link its result to a requirement. Never copy a test_id into requirement_ids. "
                "DIAGNOSTIC PRIORITY: inspect javascript_diagnostics, top-level errors, and per-test runtime_errors before interaction failures. When a syntax/runtime error can prevent event handlers or DOM updates, "
                "report it as an observed primary_defect and treat subsequent click/assertion timeouts as secondary_symptom findings rather than independent root causes. Do not recommend longer timeouts for such symptoms. "
                "Use diagnostic_role to distinguish primary defects, secondary symptoms, and independent findings. Classify defects, suggest specific future corrections, and associate requirement/test IDs. "
                "Put a JavaScript syntax defect and its downstream timeout symptoms in separate findings: the defect is primary_defect, the timeout is secondary_symptom. "
                "A passed check is evidence of that check passing, not evidence of a defect. Optional unverified behavior is not an observed failure; omit unsupported findings. "
                "Any test_outcomes must exactly match deterministic results. Never claim untested behavior passed, "
                "fabricate observations, override results, change acceptance criteria, modify source or execute repair. QA is advisory.")
            review = QAReview.model_validate(grounded_review.model_dump(mode="json"))
            tests = {test.test_id for test in task.tests}
            outcomes = {test["test_id"]: test["status"] for test in state["verification"].get("tests", [])}
            for finding in review.findings:
                validate_references(task, finding.requirement_ids)
                if set(finding.test_ids) - tests:
                    raise ValueError("QA references unknown test IDs")
                for reference in finding.evidence:
                    validate_evidence(reference, registry)
                for identifier, outcome in finding.test_outcomes.items():
                    if identifier not in tests or outcome != outcomes.get(identifier, "unverified"):
                        raise ValueError(f"QA attempted to change deterministic outcome: {identifier}")
            validate_diagnostic_hierarchy(review, state["verification"], registry)
            value = review.model_dump(mode="json")
            write_json(state["run_dir"], "qa_review.json", value)
            record_stage(state, "qa", "completed")
            return {"qa_review": value, "qa_status": "completed"}
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            write_json(state["run_dir"], "qa_error.json", {"error": message, "status": "failed"})
            record_stage(state, "qa", "failed", details={"error": message})
            # Never set shared status/error or change the verifier's report.
            return {"qa_status": "failed", "qa_error": message}

    def acceptance_gate(state):
        authoritative_task, _, metadata, write_json = _helpers()
        report = state.get("verification", {})
        reasons = []
        try:
            task = authoritative_task(state)
            if not state.get("html_path") or not Path(state["html_path"]).is_file():
                reasons.append("Generated artifact missing")
            if report.get("status") != "passed":
                reasons.append(f"Deterministic verification status: {report.get('status', 'missing')}")
            if state.get('verification_outcome') == 'verification_infrastructure_error':
                reasons.append('Verification infrastructure failed')
            elif state.get('verification_outcome') == 'unverified':
                reasons.append('Executable automated acceptance remains unverified')
            if not report.get("checks") or not all(value is True for value in report["checks"].values()) or report.get("errors"):
                reasons.append("Universal verification incomplete or failed")
            coverage = report.get("coverage", {})
            for requirement in task.requirements:
                if report.get('manual_verification') == 'off' and coverage.get(requirement.id, {}).get('status') == 'skipped':
                    continue
                if requirement.required and coverage.get(requirement.id, {}).get("status") != "passed":
                    reasons.append(f"Mandatory requirement not passed: {requirement.id}")
            outcomes = {test["test_id"]: test["status"] for test in report.get("tests", [])}
            for test in task.tests:
                if report.get('manual_verification') == 'off' and test.test_type == 'manual':
                    continue
                if test.required and outcomes.get(test.test_id) != "passed":
                    reasons.append(f"Mandatory test not passed: {test.test_id}")
        except Exception as exc:
            reasons.append(f"Requirements unavailable or changed: {exc}")
        accepted = not reasons
        decision = {"accepted": accepted, "decision": "accepted" if accepted else "rejected",
                    "basis": "deterministic verification only", "reasons": reasons,
                    "qa_status": state.get("qa_status", "skipped"), "qa_error": state.get("qa_error"),
                    "verification_hash": fingerprint(report)}
        decision.update(verification_outcome=state.get('verification_outcome'),
                        repair_count=state.get('implementation_work', {}).get('repair_count', 0),
                        uncovered_requirements=[key for key, value in report.get('coverage', {}).items()
                                                if value['status'] in {'unverified', 'skipped'}])
        write_json(state["run_dir"], "acceptance.json", decision)
        record_stage(state, "acceptance_gate", decision["decision"])
        status = "passed" if accepted else "unverified" if report.get("status") == "unverified" and not state.get("error") else "failed"
        metadata(state, status=status, acceptance=decision["decision"], qa_status=decision["qa_status"],
                 completed_at=datetime.now(timezone.utc).isoformat())
        close_resources(state)
        return {"acceptance": decision, "status": status}

    return {"ux_agent": ux, "architecture_agent": architect, "coordinator": coordinator,
            "qa_critic": qa, "acceptance_gate": acceptance_gate}
