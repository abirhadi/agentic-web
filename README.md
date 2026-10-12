# Agentic Web

Application-independent website generation using LangGraph and local LM Studio models.
The supported runtime produces one offline HTML document with inline CSS and JavaScript.

Start an experiment with a prompt, model configuration, and count **per category**:

```powershell
.\run_experiment.ps1 -Prompt "Create a professional personal website." -ModelConfig .\model_configs\nemotron-qwen.json -Candidates 2
```

Add `-Document .\resume.pdf` for supporting evidence. Two candidates per category
means six total across control, layout variance, and architecture variance.
See [the experiment launcher guide](docs/EXPERIMENT_LAUNCHER.md) for configuration and resume.

For Analyst–Decomposer collaboration without a mandatory reviewer call, use
`intake --handoff-policy handoff.example.json`. It freezes a compatible task and
decomposition after bounded feedback. See [handoff usage](docs/HANDOFFS.md).

## Setup

Use the existing `agentic-web` Conda environment, or create it from the recorded environment:

```powershell
conda env create -f environment.yml
conda activate agentic-web
python -m playwright install chromium
```

Copy `.env.example` to `.env` when needed. For resource-aware execution, copy
`resources.example.json` to `resources.local.json` and adjust for your machine.
Existing local configuration files are preserved and ignored by Git.

Managed model loads default to a **32,768-token context window**, independently
of output-token limits. Use the same resource file from intake onward:

```powershell
python -m agentic_web intake --prompt "Create a website from my resume." --document resume.pdf --output tasks/generated/resume_site.json --resource-config resources.local.json
```

`decompose`, `generate`, `experiment run` and `experiment resume` also accept
`--resource-config`. Programmatic/LangGraph callers can pass `resource_config: {}`
for the 32k resource default. Unmanaged calls use LM Studio's loaded-model context;
its saved default must also be 32,768, and already loaded 8k instances need reloading.

## Generate a website

```powershell
python -m agentic_web validate --task tasks/todo.json
python -m agentic_web generate --task tasks/todo.json --workflow collaborative --output runs
```

Baseline uses Planner ? Implementer ? deterministic verification. Collaborative adds
parallel UX/Architecture reviews, Coordinator and advisory QA. Deterministic verification
remains the acceptance authority. Every inference now has a finite role-specific output
budget and streaming safeguards. Optional CLI token limits further restrict that budget.
Model API calls and LM Studio management operations default to a 3,000-second
(50-minute) timeout. Browser verification keeps its separate time budgets.

## Prepare requirements and experiments

```powershell
python -m agentic_web intake --prompt "Create a responsive personal website." --output tasks/generated/new_site.json
python -m agentic_web decompose --task tasks/generated/new_site.json --output tasks/generated/new_site_tasks.json
python -m agentic_web experiment plan --config experiments/executable_pilot.json
python -m agentic_web experiment run --plan runs/experiments/resume-executable-b3/experiment_plan.json --max-candidates 3 --dry-run
python -m agentic_web experiment resume --plan runs/experiments/resume-executable-b3/experiment_plan.json --max-new-candidates 3 --dry-run
```

Example experiment configurations use the bundled task manifests. To use your own task, generate a new manifest and decomposition and update the configuration. New intake/decomposition filenames must not already exist.

Intake supports optional bounded evidence-driven review with
`--review-policy review.example.json`. One-shot analysis remains the default.
See [requirements review](docs/INTAKE_REVIEW.md) for configuration, artifacts,
model selection, failure statuses and an end-to-end example.

## Documentation

- [Collaborative workflows and model configuration](docs/PHASE2.md)
- [Resource placement and cleanup](docs/RESOURCE_PLACEMENT.md)
- [Reasoning policy, safeguards and transport limitations](docs/REASONING_POLICY.md)
- [A1: requirements analysis](docs/A1.md)
- [A2: document ingestion](docs/A2.md)
- [A3: requirement-driven decomposition](docs/A3.md)
- [B1: reproducible experiment planning](docs/B1.md)
- [B2: variance conditioning](docs/B2.md)
- [B3: sequential batch execution](docs/B3.md)
- [B4: checkpointing and resume](docs/B4.md)
- [Architecture](docs/architecture.md)
- [Repository branches and preservation](docs/REPOSITORY.md)

## Repository layout

| Path | Purpose |
| --- | --- |
| `agentic_web/` | Framework, CLI, schemas, workflows and verification |
| `tests/` | Automated tests and portable fixtures |
| `tasks/*.json` | Validated example task manifests |
| `experiments/` | Experiment configurations, JSON schema and variance profiles |
| `docs/` | Technical guides and milestone reports |
| `legacy/` | Original implementation snapshots |
| Root Python scripts | Preserved compatibility and historical experiment entry points |
| `runs/`, `tasks/generated/`, `eventhub_experiments/` | Local outputs, ignored on main |

## Tests

```powershell
python -m pytest -q
```

Tests use mocked model inference; browser verification tests use local Chromium.
No live experiment is required to run the suite.

## Published contents

`main` is a clean source snapshot. Historical runs, personal source documents, credentials, local settings and private development history are not published. See [repository contents](docs/REPOSITORY.md).
