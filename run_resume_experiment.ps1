#Requires -Version 5.1
<#
Run from the agentic-web environment. Starts a NEW six-candidate experiment:
feedback intake -> decomposition -> frozen plan -> sequential batch execution.
Use -ResumePlan to continue an existing plan without repeating intake/decomposition.
#>
[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$Document = 'resume.pdf',
    [string]$TemplateConfig = 'experiments/resume-nemotron-qwen-6-v1.json',
    [string]$ResourceConfig = 'resources.local.json',
    [string]$ReasoningPolicy = 'reasoning.example.json',
    [string]$ReviewPolicy = 'review.example.json',
    [string]$ExperimentId = ('resume-reviewed-nemotron-qwen-6-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss').ToLowerInvariant() + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)),
    [string]$Prompt = 'Generate a professional academic personal website from my resume. Include a professional overview, education, research interests, research experience, teaching experience, industry experience, and publications. Preserve factual accuracy without inventing credentials or achievements. Use responsive and accessible design. Do not display residential street addresses or personal phone numbers.',
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
        $planPath = (Resolve-Path -LiteralPath $ResumePlan).Path
        $script:RunRoot = Split-Path -Parent $planPath
        $resourceSnapshot = Join-Path $script:RunRoot 'resources.json'
        $reasoningSnapshot = Join-Path $script:RunRoot 'reasoning_policy.json'
        foreach ($file in @($resourceSnapshot, $reasoningSnapshot)) {
            if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing frozen configuration: $file" }
        }
        $script:ConsoleLog = Join-Path $script:RunRoot ('resume-' + [guid]::NewGuid().ToString('N') + '.log')
        Invoke-PythonStage 'Resume remaining candidates' @('-m', 'agentic_web', 'experiment', 'resume',
            '--plan', $planPath, '--resource-config', $resourceSnapshot, '--reasoning-policy', $reasoningSnapshot)
        return
    }

    if ($ExperimentId -cnotmatch '^[a-z][a-z0-9_-]*$') { throw 'ExperimentId must start with a lowercase letter and contain only lowercase letters, numbers, underscores and hyphens.' }
    $documentPath = (Resolve-Path -LiteralPath $Document).Path
    $templatePath = (Resolve-Path -LiteralPath $TemplateConfig).Path
    $resourcePath = (Resolve-Path -LiteralPath $ResourceConfig).Path
    $reasoningPath = (Resolve-Path -LiteralPath $ReasoningPolicy).Path
    $reviewPath = (Resolve-Path -LiteralPath $ReviewPolicy).Path
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
    $reviewSnapshot = Join-Path $script:RunRoot 'review_policy.json'

    $prepare = @'
import json, sys
from pathlib import Path
from agentic_web.experiments import ExperimentConfig
from agentic_web.intake_review import ReviewPolicy
from agentic_web.reasoning import load_policy
from agentic_web.resource_schema import ResourceSettings
root, template, resources, reasoning, review, identifier = map(str, sys.argv[1:])
root, template = Path(root), Path(template)
config = json.loads(template.read_text(encoding='utf-8'))
ExperimentConfig.model_validate(config)
ResourceSettings.model_validate_json(Path(resources).read_text(encoding='utf-8'))
load_policy(reasoning)
policy = ReviewPolicy.model_validate_json(Path(review).read_text(encoding='utf-8'))
if policy.mode != 'feedback':
    raise ValueError('This experiment requires feedback-enabled intake')
for source, name in [(resources, 'resources.json'), (reasoning, 'reasoning_policy.json'), (review, 'review_policy.json')]:
    with (root / name).open('xb') as stream:
        stream.write(Path(source).read_bytes())
name = 'heterogeneous-collaborative'
agents = config['model_configurations'][name]
if agents['workflow'] != 'collaborative':
    raise ValueError('Expected the existing collaborative configuration')
config.update(experiment_id=identifier, output_root=str(root.parent),
              description='Fresh feedback-reviewed resume specification; two matched seeds per variance condition.')
config['model_configurations'] = {name: agents}
config['factors'].update(model_configurations=[name], seeds=[1025, 1026], temperatures=[0.4], top_ps=[0.95], replicates=1)
conditions = config['factors']['variance_conditions']
if {v['identifier'] for v in conditions} != {'control', 'layout_variance', 'architecture_variance'} or len(conditions) != 3:
    raise ValueError('Expected control, layout_variance and architecture_variance')
for variance in conditions:
    if variance.get('profile_path'):
        source = (template.parent / variance['profile_path']).resolve()
        target = root / (variance['identifier'] + '.json')
        with target.open('xb') as stream: stream.write(source.read_bytes())
        variance['profile_path'] = str(target)
config['planning'].update(strategy='full_factorial', candidate_limit=None)
with (root / 'experiment_template.json').open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
print('Frozen models:', {role: value['model'] for role, value in agents['roles'].items()})
print('Matrix: control/layout_variance/architecture_variance x seeds 1025/1026 = 6 candidates')
'@
    Invoke-PythonStage 'Freeze experiment settings' @('-c', $prepare, $script:RunRoot, $templatePath,
        $resourcePath, $reasoningPath, $reviewPath, $ExperimentId)
    $template = Get-Content -LiteralPath (Join-Path $script:RunRoot 'experiment_template.json') -Raw | ConvertFrom-Json
    $analystModel = $template.model_configurations.'heterogeneous-collaborative'.roles.planner.model

    Invoke-PythonStage 'Reviewed intake and decomposition' @('-m', 'agentic_web', 'intake',
        '--prompt', $Prompt, '--document', $documentPath, '--document-root', (Split-Path -Parent $documentPath),
        '--output', $taskPath, '--model', $analystModel, '--seed', '1024', '--temperature', '0.4', '--top-p', '0.95',
        '--review-policy', $reviewSnapshot, '--reasoning-policy', $reasoningSnapshot,
        '--resource-config', $resourceSnapshot, '--decompose', '--decomposition-output', $decompositionPath)

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
if not analysis['ready_for_generation'] or analysis.get('review', {}).get('status') != 'converged':
    raise ValueError('Requirements review is unresolved; no candidates will run')
config = json.loads((root / 'experiment_template.json').read_text(encoding='utf-8'))
config['tasks'] = [{'name': 'resume-site', 'specification_path': str(root / 'task.json'),
    'specification_hash': contract_hash(task), 'decomposition_path': str(root / 'decomposition.json'),
    'decomposition_hash': decomposition['structure_hash']}]
with (root / 'experiment.json').open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
print('Validated and frozen task/decomposition hashes')
'@
    Invoke-PythonStage 'Validate and freeze canonical inputs' @('-c', $freeze, $script:RunRoot)
    Invoke-PythonStage 'Plan six candidates' @('-m', 'agentic_web', 'experiment', 'plan',
        '--config', $configPath, '--reasoning-policy', $reasoningSnapshot)

    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    if ($plan.planned_candidate_count -ne 6) { throw 'Expected exactly six planned candidates.' }
    foreach ($condition in $plan.factor_summary.variance_condition) {
        if ($condition.candidate_count -ne 2) { throw "Expected two candidates for $($condition.value)." }
    }
    Write-Host "`nFrozen plan: $planPath" -ForegroundColor Green
    Write-Host "Resume later: .\run_resume_experiment.ps1 -ResumePlan '$planPath'"
    Invoke-PythonStage 'Check batch inputs without inference' @('-m', 'agentic_web', 'experiment', 'run',
        '--plan', $planPath, '--resource-config', $resourceSnapshot, '--reasoning-policy', $reasoningSnapshot, '--dry-run')
    Invoke-PythonStage 'Run all six candidates' @('-m', 'agentic_web', 'experiment', 'run',
        '--plan', $planPath, '--resource-config', $resourceSnapshot, '--reasoning-policy', $reasoningSnapshot)
    Write-Host "`nExperiment finished. Results: $script:RunRoot" -ForegroundColor Green
}
finally { Pop-Location }
