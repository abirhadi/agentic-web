#Requires -Version 5.1
<#
Run from the agentic-web environment. Starts a NEW prompt-driven experiment:
feedback intake -> decomposition -> frozen plan -> sequential batch execution.
Use -ResumePlan to continue an existing plan without repeating intake/decomposition.
#>
[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$Document,
    [string]$ModelConfig = 'model_configs/nemotron-qwen.json',
    [ValidateRange(1, 2147483647)][int]$Candidates = 2,
    [ValidateSet('on', 'off')][string]$ManualVerification = 'off',
    [ValidateRange(0, 10)][int]$MaxRepairAttempts = 2,
    [string]$ResourceConfig = 'resources.local.json',
    [string]$ReasoningPolicy = 'reasoning.example.json',
    [string]$HandoffPolicy = 'handoff.example.json',
    [string]$ReviewPolicy,
    [string]$ExperimentId = ('website-experiment-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss').ToLowerInvariant() + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)),
    [string]$Prompt,
    [string]$ResumePlan
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-PythonStage {
    param([string]$Stage, [string[]]$Arguments)
    Write-Host "`n=== $Stage ===" -ForegroundColor Cyan
    # Windows PowerShell wraps native stderr (normal INFO logs) in ErrorRecord.
    # Check the process exit code, not whether the application logged to stderr.
    $priorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            Add-Content -LiteralPath $script:ConsoleLog -Value $line -Encoding UTF8
        }
        $stageExitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $priorPreference }
    if ($stageExitCode -ne 0) {
        throw "$Stage exited with code $stageExitCode. Artifacts and logs are preserved at $script:RunRoot"
    }
}

Push-Location $PSScriptRoot
try {
    # Resolve the requested executable before creating any run artifacts.
    $null = Get-Command $Python -ErrorAction Stop
    if ($ResumePlan) {
        if ($PSBoundParameters.ContainsKey('ManualVerification') -or $PSBoundParameters.ContainsKey('MaxRepairAttempts')) {
            throw 'Resume uses the frozen manual-verification and repair settings. Create a new experiment to change them.'
        }
        $planPath = (Resolve-Path -LiteralPath $ResumePlan).Path
        $script:RunRoot = Split-Path -Parent $planPath
        $resourceSnapshot = Join-Path $script:RunRoot 'resources.json'
        $reasoningSnapshot = Join-Path $script:RunRoot 'reasoning_policy.json'
        foreach ($file in @($resourceSnapshot, $reasoningSnapshot)) {
            if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing frozen configuration: $file" }
        }
        $script:ConsoleLog = Join-Path $script:RunRoot ('resume-' + [guid]::NewGuid().ToString('N') + '.log')
        $frozenModel = Get-Content -LiteralPath (Join-Path $script:RunRoot 'model_config.json') -Raw | ConvertFrom-Json
        Invoke-PythonStage 'Resume remaining candidates' @('-m', 'agentic_web', 'experiment', 'resume',
            '--plan', $planPath, '--base-url', $frozenModel.base_url, '--resource-config', $resourceSnapshot, '--reasoning-policy', $reasoningSnapshot)
        return
    }

    if ($ExperimentId -cnotmatch '^[a-z][a-z0-9_-]*$') { throw 'ExperimentId must start with a lowercase letter and contain only lowercase letters, numbers, underscores and hyphens.' }
    if ([string]::IsNullOrWhiteSpace($Prompt)) { throw 'Provide -Prompt with the website request.' }
    $documentPath = if ($Document) { (Resolve-Path -LiteralPath $Document).Path } else { $null }
    $modelPath = (Resolve-Path -LiteralPath $ModelConfig).Path
    $resourcePath = (Resolve-Path -LiteralPath $ResourceConfig).Path
    $reasoningPath = (Resolve-Path -LiteralPath $ReasoningPolicy).Path
    if ($ReviewPolicy -and $PSBoundParameters.ContainsKey('HandoffPolicy')) { throw 'Choose HandoffPolicy or an explicit historical ReviewPolicy, not both.' }
    $policyPath = if ($ReviewPolicy) { (Resolve-Path -LiteralPath $ReviewPolicy).Path } else { (Resolve-Path -LiteralPath $HandoffPolicy).Path }
    $policyKind = if ($ReviewPolicy) { 'review' } else { 'handoff' }
    $script:RunRoot = Join-Path $PSScriptRoot "runs/experiments/$ExperimentId"
    if (Test-Path -LiteralPath $script:RunRoot) { throw "Experiment directory already exists: $script:RunRoot. Use a new ID or -ResumePlan." }
    $null = New-Item -ItemType Directory -Path $script:RunRoot
    $script:ConsoleLog = Join-Path $script:RunRoot 'console.log'
    $taskPath = Join-Path $script:RunRoot 'task.json'
    $decompositionPath = Join-Path $script:RunRoot 'decomposition.json'
    $configPath = Join-Path $script:RunRoot 'experiment.json'
    $planPath = Join-Path $script:RunRoot 'experiment_plan.json'
    $resourceSnapshot = Join-Path $script:RunRoot 'resources.json'
    $reasoningSnapshot = Join-Path $script:RunRoot 'reasoning_policy.json'

    # Preserve quotes and newlines without native-command argument rewriting.
    $launchInput = @{ prompt = $Prompt; document = $documentPath; candidates_per_category = $Candidates; manual_verification = $ManualVerification; max_repair_attempts = $MaxRepairAttempts }
    [System.IO.File]::WriteAllText((Join-Path $script:RunRoot 'launch_input.json'),
        (ConvertTo-Json -InputObject $launchInput -Depth 10), [System.Text.UTF8Encoding]::new($false))
    $prepare = @'
import json, sys
from pathlib import Path
from agentic_web.client import ModelConfig
from agentic_web.experiments import AgentConfiguration
from agentic_web.intake_review import ReviewPolicy
from agentic_web.handoff import HandoffPolicy
from agentic_web.reasoning import load_policy
from agentic_web.resource_schema import ResourceSettings
from agentic_web.provenance import fingerprint
from agentic_web.variance import VarianceProfile
root, model_file, resources, reasoning, policy_file, policy_kind, repo = sys.argv[1:]
root, repo = Path(root), Path(repo)
selected = ModelConfig.model_validate_json(Path(model_file).read_text(encoding='utf-8'))
if selected.seed is None:
    raise ValueError('A seeded experiment requires an integer model-config seed')
ResourceSettings.model_validate_json(Path(resources).read_text(encoding='utf-8'))
reasoning_policy = selected.reasoning_policy or load_policy(reasoning)
launch = json.loads((root / 'launch_input.json').read_text(encoding='utf-8'))
selected = selected.model_copy(update={'reasoning_policy': reasoning_policy,
    'manual_verification': launch['manual_verification'], 'max_repair_attempts': launch['max_repair_attempts']})
(HandoffPolicy if policy_kind == 'handoff' else ReviewPolicy).model_validate_json(Path(policy_file).read_text(encoding='utf-8'))
for source, name in [(resources, 'resources.json'), (policy_file, policy_kind + '_policy.json')]:
    with (root / name).open('xb') as stream:
        stream.write(Path(source).read_bytes())
for name, value in [('model_config.json', selected.model_dump(mode='json')), ('reasoning_policy.json', reasoning_policy.model_dump(mode='json'))]:
    with (root / name).open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2)
count = json.loads((root / 'launch_input.json').read_text(encoding='utf-8'))['candidates_per_category']
roles = ('planner', 'implementer', 'ux', 'architect', 'coordinator', 'qa')
agents = AgentConfiguration(workflow='collaborative', roles={r: {'model': selected.model_for(r)} for r in roles},
    planner_max_tokens=selected.planner_max_tokens, review_max_tokens=selected.review_max_tokens,
    max_tokens=selected.max_tokens, reasoning_policy=reasoning_policy,
    manual_verification=selected.manual_verification, max_repair_attempts=selected.max_repair_attempts)
conditions = [{'identifier': 'control', 'profile_path': None, 'profile_hash': None}]
for category, filename in [('layout_variance', 'layout_exploration.json'), ('architecture_variance', 'architecture_exploration.json')]:
    source = repo / 'experiments/profiles' / filename
    raw = json.loads(source.read_text(encoding='utf-8'))
    VarianceProfile.model_validate(raw)
    target = root / filename
    with target.open('xb') as stream: stream.write(source.read_bytes())
    conditions.append({'identifier': category, 'profile_path': str(target), 'profile_hash': fingerprint(raw)})
config = {'schema_version': '1.0', 'experiment_id': root.name, 'experiment_version': '1.0',
    'description': 'Reviewed request and optional document; matched seeds across three variance categories.',
    'output_root': str(root.parent), 'model_configurations': {'selected': agents.model_dump(mode='json')},
    'factors': {'model_configurations': ['selected'], 'seeds': list(range(selected.seed + 1, selected.seed + count + 1)),
        'temperatures': [selected.temperature], 'top_ps': [selected.top_p], 'replicates': 1, 'variance_conditions': conditions},
    'planning': {'strategy': 'full_factorial', 'candidate_limit': None, 'selection_seed': selected.seed}}
with (root / 'experiment_template.json').open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
print('Frozen models:', {role: value.model for role, value in agents.roles.items()})
print(f'Matrix: {count} per category x 3 categories = {count * 3} candidates')
'@
    Invoke-PythonStage 'Freeze experiment settings' @('-c', $prepare, $script:RunRoot, $modelPath,
        $resourcePath, $reasoningPath, $policyPath, $policyKind, $PSScriptRoot)
    $frozenModel = Get-Content -LiteralPath (Join-Path $script:RunRoot 'model_config.json') -Raw | ConvertFrom-Json

    $intake = @'
import json, sys
from pathlib import Path
from agentic_web.intake import build_intake_graph
from agentic_web.progress import configure_console_logging
configure_console_logging('INFO')
root = Path(sys.argv[1])
launch = json.loads((root / 'launch_input.json').read_text(encoding='utf-8'))
inputs = {'prompt': launch['prompt'], 'output_file': str(root / 'task.json'),
    'model_config': json.loads((root / 'model_config.json').read_text(encoding='utf-8')),
    'resource_config': json.loads((root / 'resources.json').read_text(encoding='utf-8')),
    'decompose': True, 'decomposition_output_file': str(root / 'decomposition.json')}
policy_name = 'handoff_policy' if (root / 'handoff_policy.json').exists() else 'review_policy'
inputs[policy_name] = json.loads((root / (policy_name + '.json')).read_text(encoding='utf-8'))
if launch['document']:
    inputs.update(documents=[launch['document']], document_root=str(Path(launch['document']).parent))
result = build_intake_graph().invoke(inputs)
print(json.dumps({key: result[key] for key in ('status', 'error', 'review_status', 'run_dir', 'ready_for_generation', 'ready_for_planning') if key in result}, indent=2))
if result.get('error') or not result.get('ready_for_generation') or not result.get('ready_for_planning'):
    sys.exit(1)
'@
    Invoke-PythonStage 'Intake and decomposition' @('-c', $intake, $script:RunRoot)

    $freeze = @'
import json, sys
from pathlib import Path
from agentic_web.schema import load_task
from agentic_web.workflow import contract_hash
from agentic_web.decomposition import load_decomposition
root = Path(sys.argv[1])
task = load_task(root / 'task.json')
decomposition = load_decomposition(root / 'decomposition.json', task)
if not decomposition['ready_for_planning']:
    raise ValueError('Decomposition is blocked; no candidates will run')
analyses = list((root / 'task.intake').glob('*/analysis.json'))
if len(analyses) != 1:
    raise ValueError('Expected one fresh intake analysis')
analysis = json.loads(analyses[0].read_text(encoding='utf-8'))
if not analysis['ready_for_generation']:
    raise ValueError('Requirements are unresolved; no candidates will run')
if analysis.get('handoff') and analysis['handoff']['status'] != 'accepted':
    raise ValueError('Handoff is unresolved; no candidates will run')
if analysis.get('review') and analysis['review']['status'] != 'converged':
    raise ValueError('Historical review is unresolved; no candidates will run')
config = json.loads((root / 'experiment_template.json').read_text(encoding='utf-8'))
config['tasks'] = [{'name': 'website', 'specification_path': str(root / 'task.json'),
    'specification_hash': contract_hash(task), 'decomposition_path': str(root / 'decomposition.json'),
    'decomposition_hash': decomposition['structure_hash']}]
with (root / 'experiment.json').open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
print('Validated and frozen task/decomposition hashes')
'@
    Invoke-PythonStage 'Validate and freeze canonical inputs' @('-c', $freeze, $script:RunRoot)
    Invoke-PythonStage 'Plan candidates' @('-m', 'agentic_web', 'experiment', 'plan',
        '--config', $configPath, '--reasoning-policy', $reasoningSnapshot)

    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    if ($plan.planned_candidate_count -ne (3 * $Candidates)) { throw 'Candidate plan does not match the requested count.' }
    foreach ($condition in $plan.factor_summary.variance_condition) {
        if ($condition.candidate_count -ne $Candidates) { throw "Incorrect candidate count for $($condition.value)." }
    }
    Write-Host "`nFrozen plan: $planPath" -ForegroundColor Green
    Write-Host "Resume later: .\run_experiment.ps1 -ResumePlan '$planPath'"
    Invoke-PythonStage 'Check batch inputs without inference' @('-m', 'agentic_web', 'experiment', 'run',
        '--plan', $planPath, '--base-url', $frozenModel.base_url, '--resource-config', $resourceSnapshot, '--reasoning-policy', $reasoningSnapshot, '--dry-run')
    Invoke-PythonStage 'Run all candidates' @('-m', 'agentic_web', 'experiment', 'run',
        '--plan', $planPath, '--base-url', $frozenModel.base_url, '--resource-config', $resourceSnapshot, '--reasoning-policy', $reasoningSnapshot)
    Write-Host "`nExperiment finished. Results: $script:RunRoot" -ForegroundColor Green
}
finally { Pop-Location }
