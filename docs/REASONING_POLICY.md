# Reasoning policy and inference safeguards

Every `LMStudioClient.complete` request resolves a versioned policy, including direct
programmatic calls. The graphs, model assignments, scheduler and deterministic
acceptance authority remain unchanged. Thinking is not globally disabled.

## Configuration

Use `reasoning.example.json`, or provide `reasoning_policy` in `ModelConfig` or an
experiment configuration. JSON is the shared configuration format. These commands
accept `--reasoning-policy`: `intake`, `decompose`, `generate`, `experiment plan`,
`experiment run`, and `experiment resume`. The latter two compare it with the frozen
plan; they cannot override it.

```json
{
  "strict_capabilities": true,
  "max_output_tokens": 32768,
  "timeout_seconds": 3000,
  "roles": {
    "intake": {"mode": "auto", "max_output_tokens": 4096},
    "decomposition": {"mode": "auto", "max_output_tokens": 4096},
    "planner": {"mode": "auto", "max_output_tokens": 6144},
    "coordinator": {"mode": "auto", "max_output_tokens": 6144},
    "implementer": {"mode": "auto", "max_output_tokens": 16000}
  },
  "watchdog": {
    "stall_seconds": 300,
    "max_reasoning_chars_without_final": 65536,
    "window_chars": 16384,
    "segment_chars": 200,
    "repetition_count": 4,
    "diagnostic_chars": 4096
  },
  "recovery": {"max_attempts": 1, "retry_on": []}
}
```

Omitted roles use the version 1 defaults: 4,096 generated tokens for intake,
decomposition, UX, architecture and QA; 6,144 for Planner and Coordinator; 16,000 for
Implementer. All roles retain the existing 3,000-second timeout default. Modes default
to `auto` to preserve provider behavior without claiming unverified controls. These
are preliminary safety budgets, not optimized experimental parameters.

The aliases `requirements_analyst`, `task_decomposition`, and `architect` normalize
to `intake`, `decomposition`, and `architecture`. Supplying a role and its alias twice
is rejected. Unknown roles/modes and invalid budgets are rejected.

Budget ceilings apply in this order: global policy, experiment/CLI declarations,
role policy, then provider enforcement. The effective ceiling is the minimum of
the applicable limits. Legacy optional `max_tokens`, `planner_max_tokens`, and
`review_max_tokens` remain optional overrides; `None` now uses a finite policy
budget. Set the role and global ceilings explicitly when an experiment needs a
larger output budget. A 32k **context** is not a 32k **output** allowance; prompt
tokens also consume context. No undocumented reasoning-token parameter is sent.
Timeout covers capability discovery, all configured attempts, and streaming together.

## Transport support and the current limitation

The adapter reads `GET /api/v1/models`, matching the exact model key or scheduler-owned
loaded instance ID. It records advertised reasoning options and defaults. Discovery
failure or absent controls with `auto` leaves the provider default unconfirmed.
Discovery is read-only and limited to ten seconds within the overall deadline.

| Request | Transport and behavior |
| --- | --- |
| `auto`, seeded, with or without a Pydantic schema | Streaming `/v1/chat/completions`; preserves seed, temperature, top-p and exact `response_format=json_schema` |
| Advertised `on`, `off`, `low`, `medium`, `high`, explicitly unseeded plain text | Streaming native `/api/v1/chat`, request-level `reasoning`; separate `reasoning.delta` and `message.delta` |
| Explicit mode with seed or JSON Schema contract | `UnsupportedTransportError` before generation; no settings are dropped |
| Mode not advertised for the selected model | `UnsupportedReasoningError`; no inference or fallback |

This is a **backend compatibility limitation**, not a completed on/off implementation
for structured agents. LM Studio's documented native chat parameters do not include
seed or a JSON Schema response format. Its documented Chat Completions parameters
do not advertise request-level reasoning control. Consequently, Planner/Coordinator
cannot currently be forced on and intake cannot be forced off while preserving all
existing contracts. Native requests never substitute prompt instructions for schema
enforcement. No SDK private configuration, template changes, global settings or
two-request reasoning/final workaround is used. A verified transport supporting
these controls together is required to remove this restriction.

`strict_capabilities=false` does not permit an explicit unsupported mode or dropping
a seed/schema. `auto` does not mean off: an advertised default such as `on` is recorded
as the provider default, with `reasoning_mode_confirmed=false` because no request-level
override was sent. Plain native requests require explicit programmatic `seed=None`
and temperature at most 1. Existing seeded configurations are not changed.

References: [native chat parameters](https://lmstudio.ai/docs/developer/rest/chat),
[model capabilities](https://lmstudio.ai/docs/developer/rest/list),
[native streaming events](https://lmstudio.ai/docs/developer/rest/streaming-events),
[Chat Completions parameters](https://lmstudio.ai/docs/developer/openai-compat/chat-completions),
[structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output).

## Streaming safeguards and diagnostics

Both adapters close streams and client connections on completion, cancellation or
failure. An asynchronous deadline cancels even a pending read. Stall limits apply
between events. Native requests set `store=false` and enable no integrations.

The watchdog keeps a bounded rolling reasoning window. It needs at least four
non-overlapping repetitions of a 200-character, word-rich segment. Short phrases,
braces and simple code motifs are insufficient evidence. A separate character ceiling
detects excessive reasoning without final content. Character thresholds are not token
estimates. Models that do not expose reasoning deltas still have total generation,
wall-time, stall and final-response checks; early repetition detection is unavailable
for hidden reasoning.

Final content never comes from reasoning. Empty content, `length`, other incomplete
provider statuses, exhausted budgets, invalid JSON, and Pydantic failures are errors.
No schema repair or automatic promotion of reasoning text occurs. Native chat does
not supply a documented finish reason; metrics preserve `finish_reason=null` and
`provider_status=chat.end`, and conservatively reject reported output at its cap.
Unknown token counts stay null; final token counts are derived only when both total
generated and reasoning counts are available.

Each stage saves `<stage>_reasoning_policy.json`. Existing `<stage>_metrics.json`
contains resolved settings/hash, capabilities, transport, final content, a bounded
reasoning tail, completion status, tokens, validation status, and all attempts. Failed
inference also saves `<stage>_inference_attempts.json`, raw final text, and the existing
diagnostic artifacts. Console logs show role, mode, cap and hash, not reasoning text.
Each completed attempt is also written exclusively to
`<canonical_role>_inference_attempt_0001.json` before a retry starts, so an interrupted
retry retains earlier attempts. Failure to persist an attempt stops inference.
Diagnostic artifacts may contain source-related reasoning; treat them as private run
data just like the existing raw response artifacts.

Recovery defaults to one attempt. Explicit `max_attempts` can be 1–3 with `retry_on`
drawn from `transport_failure`, `timeout`, and `stalled_generation`. Retries use exactly
the same settings and share the original wall-time deadline. Unsupported settings,
repetition, exhaustion and invalid outputs are terminal. SDK retries remain disabled.
No alternative modes, models, budgets or sampling settings are selected automatically.
Other candidates can continue after a terminal inference failure. Scheduler leases
and owned-model cleanup continue through existing `finally` paths.

## Experiments and resume

New plans materialize the full policy in configuration and each model configuration,
and record policy hashes and resolved per-role budgets in every candidate. Changing
policy changes candidate identity; frozen task/decomposition hashes are unchanged.
Plan validation re-derives all frozen values. Run/resume rejects policy overrides that
differ, and checkpoint validation rejects altered candidate policy records. Dry runs
display per-candidate policy hashes and role settings without prompts/documents.

Old task/configuration inputs still load. Historical plans remain readable and support
dry-run inspection with their original IDs. **Execution/resume of plans without a
frozen reasoning policy is refused**, because applying these new caps would silently
change the experiment. Create a new experiment ID/output root and plan; never edit
the historical plan or overwrite results.

```powershell
python -m agentic_web experiment plan --config experiments/my_new_experiment.json --reasoning-policy reasoning.example.json
python -m agentic_web experiment run --plan runs/experiments/my-new-experiment/experiment_plan.json --reasoning-policy reasoning.example.json --dry-run
python -m agentic_web experiment resume --plan runs/experiments/my-new-experiment/experiment_plan.json --reasoning-policy reasoning.example.json --max-new-candidates 1
```

## Retry decomposition and test the adapters

The archived failure under `personal_site_tasks_qat_v1.decompose` used seed 1024 and
the canonical specification matching `personal_site_qat_v1.json`. Preserve that run
and choose a new output filename. In the existing Conda environment, this uses bounded
provider-default reasoning with the same model and sampling defaults:

```powershell
conda activate agentic-web
python -m agentic_web decompose --task tasks/generated/personal_site_qat_v1.json --output tasks/generated/personal_site_tasks_qat_policy_v1.json --model google/gemma-4-12b-qat --seed 1024 --temperature 0.4 --top-p 0.95 --reasoning-policy reasoning.example.json
```

If the output already exists, choose a new name. Inspect the canonical input's ready/
blocked status as usual. When diagnosing a failure, check `error_classification`,
finish reason, reasoning/generated counts, watchdog evidence and raw **final** text.
An empty final answer at `length` is unsuccessful even when reasoning contains plausible
JSON. Increase role budgets only as an explicit new experimental condition; they do
not fix repetition or missing source information.

Run deterministic tests with `python -m pytest -q`. An optional live comparison is
available only through this explicit command; it is never run by the test suite:

```powershell
python -m agentic_web.reasoning_probe --model google/gemma-4-12b-qat --output runs/reasoning_probes
```

The probe requests the same short, seeded, Pydantic-constrained task with `on` and
`off`, saving separate settings/results/metrics in a new directory. On the currently
documented backend it records unsupported-transport failures without generating.
It does not silently relax the probe to unconstrained text, change global settings,
or load/unload models. Live on/off comparison of structured results remains blocked
until the backend supports the required parameter combination.
