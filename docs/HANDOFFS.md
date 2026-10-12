# Analyst–Decomposer handoffs

`handoff.example.json` enables collaborative intake with at most two upstream
revisions. The default, with no handoff policy, remains the existing workflow.
Historical `--review-policy` remains an explicit replay option in the same graph;
enabling both feedback policies is rejected. Collaborative handoffs do not call
the semantic reviewer. Both policies share intake, resource-managed inference,
decomposition validation and finalization; neither has a separate entry point.

```powershell
conda run -n agentic-web --no-capture-output python -m agentic_web intake `
  --prompt "Generate a professional personal website from my resume." `
  --document ".\resume.pdf" `
  --model nvidia/nemotron-3-nano-4b `
  --handoff-policy handoff.example.json `
  --reasoning-policy reasoning.example.json `
  --resource-config resources.local.json `
  --output tasks/generated/personal_site.json
```

Choose a new output name on every fresh invocation. The document is optional.
Collaborative mode implies decomposition; no separate `--decompose` call is
needed. `--decomposition-output` selects its path (default: `<task>.tasks.json`).
Use the frozen pair with existing experiment planning and generation commands.
`run_experiment.ps1` calls this same graph and defaults to `handoff.example.json`.
Use `-HandoffPolicy` to select an existing policy with `mode=off` or `collaborative`.
`-ReviewPolicy` explicitly selects historical review behavior when reproducing an
older procedure. Resume consumes frozen inputs and never reruns either policy.

The shared LangGraph nodes run preparation → Analyst → validation → Decomposer.
Acceptance publishes the canonical task and compatible validated decomposition.
Feedback routes to Analyst revision and then back to actual decomposition.
There are no decision arrays and no added reviewer or reflection calls. The
Analyst may retain a valid artifact when challenging unsupported criticism;
the next Decomposer pass judges the artifact. An unchanged artifact plus the
same feedback stops as `no_progress`.

In collaborative mode, an Analyst draft rejected by JSON/schema or canonical
validation returns to the existing `requirements_revision` node with its raw
draft and exact validation errors. The Analyst must produce a complete corrected
response, which passes the same validators before decomposition. These corrections
share `max_revisions` with Decomposer-requested revisions; an identical invalid
draft stops as `no_progress`. Each attempt retains its own round artifacts and
metrics. Transport failures and truncated responses remain terminal. No invalid
draft is published as a canonical task, and the framework does not edit model output.

Programmatic callers pass `handoff_policy={"mode": "collaborative"}` to either
`build_intake_graph()` or the existing collaborative generation graph. Both
builders accept the normal LangGraph `checkpointer=` argument and thread config.
An interrupted node can resume with `graph.invoke(None, config)` without
re-running completed nodes. The application does not introduce a new checkpoint
backend or enable durable checkpoints implicitly. Resumed inference gets a new
artifact subdirectory; transport retry rules remain unchanged.

Artifacts under `<task>.intake/<run>/handoff/round_NNN/` contain original requests,
raw outputs, validated analysis versions, structured handoffs, metrics, resource
records and stage ledgers. `handoff_summary.json` indexes calls and artifact hashes,
usage totals, policy identity and terminal outcome without copying full artifacts.
There is one canonical task output and one decomposition output. One-shot
decomposition invoked from intake keeps diagnostics in the same intake run,
without creating a second independent run tree. The final analysis and
decomposition include policy provenance; existing experiment plans are never
modified. Failed revisions retain the last validated analysis. Malformed or
failed downstream responses are `downstream_failed`, not upstream defects.

Runtime capability claims are compared to the actual adapter registry. A false
claim that `single_html` cannot support `multiple_sections` is recorded and dismissed.
If a usable decomposition was supplied and no other findings remain, it proceeds
after normal graph validation. Otherwise it stops `unresolved`, preserving the
valid analysis without an unnecessary upstream revision. Blocked or unsupported requirements
cannot become ready through a model assertion.

Limitations: exact evidence validates provenance, not the truth of an inference.
Semantic omissions, paraphrased non-progress and mislabeled capability complaints
still depend on model judgment. Unsupported criticism is never automatically
turned into mandatory requirements. Conservative preservation rejects removal,
demotion or ID changes of explicit obligations; such disputes require escalation.
The CLI starts fresh runs; checkpoint resume requires a caller-supplied checkpointer.

Tests (mocked, no GPU): `conda run -n agentic-web --no-capture-output python -m unittest discover -s tests -p test_handoff.py -v`.
Optional live probe, using existing default LM Studio configuration:

```powershell
$env:AGENTIC_WEB_LIVE_HANDOFF = '1'
conda run -n agentic-web --no-capture-output python -m unittest discover -s tests -p test_handoff_live.py -v
Remove-Item Env:AGENTIC_WEB_LIVE_HANDOFF
```

Live probe artifacts are retained in the printed result's run directory on failure.

## Implementation verification feedback

The same baseline/collaborative LangGraph has a conditional Verifier → Implementer
edge. It shares round-directory allocation and bounded/non-progress termination
with Analyst–Decomposer handoffs. Planning and specialist results stay in state;
local repairs do not rerun those stages or change the TaskSpec or candidate ID.

With `max_repair_attempts` configured, each round preserves the application,
request, raw response, inference metrics, resource audits and verification evidence
under `implementation/round_NNN/`. `implementation_summary.json` records history,
repair count, token usage, elapsed time and termination. Resuming a checkpoint
inside a stage uses an additional attempt directory rather than overwriting files.
The final `output_file` points to the selected round's application.

Only executable application failures request repair. Evidence includes affected
requirements, expected assertions, observations and artifact references. Browser
infrastructure failures, unsupported checks and skipped manual tests do not.
Every repair reruns the full deterministic suite to detect regressions. QA remains
advisory and runs only after local feedback terminates.

Reports distinguish `first_pass_verified`, `verified_after_repair`,
`verification_failed`, `unverified`, and `verification_infrastructure_error`.
Verification readiness compiles the existing manifest DSL and reports missing,
disabled and manual coverage before generation. It does not invent selectors or
translate subjective prose into a passing test. Manual-only coverage cannot
establish automated acceptance. Browser universal checks remain enforced.
