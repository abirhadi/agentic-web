# Implemented architecture

The existing pipeline now uses one application-independent LangGraph:

START -> Task Loader -> Planner -> Implementer -> Verifier -> END

- Task Loader validates a JSON or inline task, preserves original requirements,
  records an authoritative contract fingerprint, and allocates an exclusive run.
- Planner uses the reusable model client and produces Pydantic-validated JSON.
- Implementer receives the original requirements, task, plan, and runtime constraints.
- Runtime adapters own output format instructions, artifact extraction and verification.
- Verifier runs universal checks plus allowlisted manifest-defined acceptance tests.
  Missing tests remain unverified. Failures include actionable reports and screenshots.

Only single_html is implemented and verified. Other runtime kinds are explicitly
unsupported; no backend service, database, API adapter or repair loop is implemented.

See [README.md](../README.md) for schemas, commands, project structure, validation
results, browser isolation details, and Phase 2 interfaces. The previous research
roadmap is preserved at legacy/architecture_original.md.

## Phase 2 extension

The baseline remains the default. The selectable collaborative graph adds independent UX and Architecture branches after planning, joins them at a Coordinator, feeds its validated guidance to the same Implementer, then adds advisory QA and a deterministic acceptance gate after the unchanged verifier. All stages record exclusive ledger events. See [PHASE2.md](PHASE2.md) for the complete node/edge structure and tested behavior. No prompt optimizers, repair loops or model scheduling are implemented.
