"""Minimal CLI for the existing Conda environment."""
import argparse
import json
from .client import LMStudioClient, ModelConfig
from .schema import load_task
from .runtime import get_adapter
from .workflow import app, collaborative_app
from .progress import configure_console_logging, log_event
from .reasoning import load_policy


def policy_values(args):
    return {'reasoning_policy': load_policy(args.reasoning_policy)} if args.reasoning_policy else {}


def main():
    parser = argparse.ArgumentParser(description="Application-independent LangGraph -> LM Studio pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    experiment = sub.add_parser("experiment", help="Plan reproducible experiments without execution")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_plan = experiment_sub.add_parser("plan")
    experiment_plan.add_argument("--config", required=True)
    experiment_run = experiment_sub.add_parser('run', help='Execute a frozen candidate plan sequentially')
    experiment_run.add_argument('--plan', required=True)
    experiment_run.add_argument('--max-candidates', type=int)
    experiment_run.add_argument('--dry-run', action='store_true')
    experiment_run.add_argument('--resource-config')
    experiment_run.add_argument('--base-url')
    experiment_run.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    experiment_resume = experiment_sub.add_parser('resume', help='Resume unstarted candidates; preserve terminal records')
    experiment_resume.add_argument('--plan', required=True)
    experiment_resume.add_argument('--max-new-candidates', type=int)
    experiment_resume.add_argument('--retry-interrupted', action='store_true')
    experiment_resume.add_argument('--dry-run', action='store_true')
    experiment_resume.add_argument('--resource-config')
    experiment_resume.add_argument('--base-url')
    experiment_resume.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    models = sub.add_parser("models")
    models.add_argument("--base-url")
    validate = sub.add_parser("validate")
    validate.add_argument("--task", required=True)
    intake = sub.add_parser("intake", help="Analyze a natural-language request and save a reusable task manifest")
    intake.add_argument("--prompt", required=True)
    intake.add_argument("--context", help="Optional supporting text, combined with document evidence")
    intake.add_argument("--document", action="append", help="Local PDF, DOCX, Markdown or TXT; repeat for multiple sources")
    intake.add_argument("--document-root", help="Allowed local document directory (default: current directory)")
    intake.add_argument("--decompose", action="store_true", help="Save A3 tasks after A1; no website generation")
    intake.add_argument("--decomposition-output", help="New A3 artifact path (default: <task>.tasks.json)")
    intake.add_argument("--output", required=True, help="New canonical task .json file; existing files are preserved")
    intake.add_argument("--model")
    intake.add_argument("--qa-model", help="Semantic intake reviewer model (default: analyst model)")
    intake.add_argument("--review-policy", help="Versioned intake feedback policy JSON; default: off")
    intake.add_argument('--handoff-policy', help='Versioned Analyst/Decomposer handoff policy; default: off')
    intake.add_argument("--base-url")
    intake.add_argument("--resource-config", help="Resource settings for intake and optional decomposition (default context: 32768)")
    intake.add_argument("--seed", type=int)
    intake.add_argument("--temperature", type=float)
    intake.add_argument("--top-p", type=float)
    intake.add_argument("--max-tokens", type=int, help="Optional analyst output cap (existing planner inference setting)")
    intake.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    decomposition = sub.add_parser("decompose", help="Prepare reusable requirement-level tasks without generating a website")
    decomposition.add_argument("--task", required=True)
    decomposition.add_argument("--output", required=True)
    decomposition.add_argument("--analysis", help="A1 analysis record; matching adjacent records are discovered automatically")
    for flag in ("model", "base-url", "resource-config"):
        decomposition.add_argument("--" + flag)
    decomposition.add_argument("--seed", type=int)
    decomposition.add_argument("--max-tokens", type=int)
    decomposition.add_argument("--temperature", type=float)
    decomposition.add_argument("--top-p", type=float)
    decomposition.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    generate = sub.add_parser("generate")
    generate.add_argument("--task", required=True)
    generate.add_argument("--output", default="runs")
    generate.add_argument("--model")
    generate.add_argument("--planner-model")
    generate.add_argument("--workflow", choices=["baseline", "collaborative"], default="baseline")
    generate.add_argument("--variance-condition", default="control")
    generate.add_argument("--variance-profile", help="B2 advisory JSON profile for a non-control condition")
    generate.add_argument("--variance-profile-hash", help="Optional expected B1 profile fingerprint")
    for role in ("ux", "architect", "coordinator", "qa"):
        generate.add_argument(f"--{role}-model")
    generate.add_argument("--review-max-tokens", type=int)
    generate.add_argument("--base-url")
    generate.add_argument("--temperature", type=float)
    generate.add_argument("--top-p", type=float)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--max-tokens", type=int)
    generate.add_argument("--planner-max-tokens", type=int)
    generate.add_argument("--timeout-ms", type=int)
    generate.add_argument("--no-screenshots", action="store_true")
    generate.add_argument("--resource-config", help="JSON resource settings; enables hardware-aware placement")
    generate.add_argument('--manual-verification', choices=['on', 'off'])
    generate.add_argument('--max-repair-attempts', type=int)
    decomposition_choice = generate.add_mutually_exclusive_group()
    decomposition_choice.add_argument("--decompose", action="store_true", help="Prepare A3 tasks before candidate planning")
    decomposition_choice.add_argument("--decomposition", help="Reuse a frozen A3 task artifact without decomposition inference")
    generate.add_argument("--analysis", help="Matching A1 record for A3 blocker/origin validation")
    generate.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                          help="Console progress level (stderr); final JSON remains on stdout")
    resources = sub.add_parser("resources", help="Read-only hardware/backend discovery and placement feasibility")
    resources.add_argument("--resource-config", required=True)
    resources.add_argument("--models", nargs="+", required=True)
    resources.add_argument("--base-url")
    resources.add_argument("--output", default="runs/resource_checks")
    for command in (intake, decomposition, generate, experiment_plan, experiment_run, experiment_resume):
        command.add_argument('--reasoning-policy', help='Shared JSON reasoning policy; execution must match the frozen experiment plan')
    args = parser.parse_args()
    configure_console_logging(getattr(args, "log_level", "INFO"))
    try:
        if args.command == "experiment":
            from pathlib import Path
            if args.experiment_command in {'run', 'resume'}:
                from .batch import run_experiment
                resources = json.loads(Path(args.resource_config).read_text(encoding='utf-8')) if args.resource_config else None
                controls = ({'resume': True, 'max_new_candidates': args.max_new_candidates,
                             'retry_interrupted': args.retry_interrupted} if args.experiment_command == 'resume'
                            else {'max_candidates': args.max_candidates})
                result = run_experiment(args.plan, dry_run=args.dry_run,
                                        base_url=args.base_url, resource_config=resources, **controls, **policy_values(args))
                print(json.dumps(result, indent=2))
                return 0 if result['status'] == 'dry_run' or (result['status'] == 'completed' and
                    result['progress']['counts']['interrupted'] == 0 and
                    all(c['execution_status'] in {None, 'generation_completed_verification_passed'}
                        for c in result['progress']['candidates'])) else 1
            from .experiments import read_config, build_plan, save_plan
            config = read_config(args.config)
            if args.reasoning_policy:
                policy = load_policy(args.reasoning_policy)
                config = config.model_copy(update={'reasoning_policy': policy, 'model_configurations': {
                    key: agents.model_copy(update={'reasoning_policy': policy}) for key, agents in config.model_configurations.items()}})
            plan = build_plan(config, Path(args.config).resolve().parent)
            if 'verification_readiness' in plan:
                print(json.dumps({'verification_readiness': plan['verification_readiness']}, indent=2), flush=True)
            print(json.dumps({"experiment_id": config.experiment_id, "experiment_version": config.experiment_version,
                "full_factorial_count": plan["full_factorial_count"],
                "planned_candidate_count": plan["planned_candidate_count"],
                "factor_summary": plan["factor_summary"], 'reasoning_policy_hash': plan['reasoning_policy_hash'],
                'reasoning_policy': plan['configuration']['reasoning_policy']}, indent=2), flush=True)
            destination = save_plan(plan, args.config)
            print(json.dumps({"status": "planned", "plan_file": str(destination)}, indent=2))
            return 0
        if args.command == "models":
            config = ModelConfig(**({"base_url": args.base_url} if args.base_url else {}))
            print(json.dumps(LMStudioClient().models(config), indent=2))
            return 0
        if args.command == "intake":
            from .intake import intake_app
            values = {key: getattr(args, key) for key in ("model", "qa_model", "base_url", "seed", "temperature", "top_p")
                      if getattr(args, key) is not None}
            if args.max_tokens is not None:
                values["planner_max_tokens"] = args.max_tokens
            config = ModelConfig(**values, **policy_values(args))
            inputs = {"prompt": args.prompt, "supporting_context": args.context,
                      "output_file": args.output, "model_config": config.model_dump()}
            if args.review_policy:
                from pathlib import Path
                from .intake_review import ReviewPolicy
                inputs['review_policy'] = ReviewPolicy.model_validate_json(Path(args.review_policy).read_text(encoding='utf-8')).model_dump(mode='json')
            if args.handoff_policy:
                from pathlib import Path
                from .handoff import HandoffPolicy
                inputs['handoff_policy'] = HandoffPolicy.model_validate_json(Path(args.handoff_policy).read_text(encoding='utf-8')).model_dump(mode='json')
            if args.resource_config:
                from pathlib import Path
                inputs['resource_config'] = json.loads(Path(args.resource_config).read_text(encoding='utf-8'))
            if args.document is not None:
                inputs.update(documents=args.document, document_root=args.document_root)
            if args.decomposition_output:
                inputs['decomposition_output_file'] = args.decomposition_output
            if args.decompose:
                inputs["decompose"] = True
                if args.decomposition_output:
                    inputs["decomposition_output_file"] = args.decomposition_output
            result = intake_app.invoke(inputs)
            print(json.dumps({key: result[key] for key in ("status", "run_dir", "output_file", "analysis_file",
                             "source_hash", "task_hash", "ready_for_generation", "decomposition_file", "ready_for_planning", "review_status", "handoff_status", "error") if key in result}, indent=2))
            return 1 if result["status"] == "failed" or result.get('handoff_status') not in {None, 'accepted'} else 0
        if args.command == "decompose":
            from .decomposition import decompose
            from pathlib import Path
            values = {key: getattr(args, key) for key in ("model", "base_url", "seed", "temperature", "top_p") if getattr(args, key) is not None}
            if args.max_tokens is not None:
                values["planner_max_tokens"] = args.max_tokens
            inputs = {"task_path": args.task, "output_file": args.output, "model_config": ModelConfig(**values, **policy_values(args)).model_dump()}
            if args.analysis:
                inputs["analysis_file"] = args.analysis
            if args.resource_config:
                inputs["resource_config"] = json.loads(Path(args.resource_config).read_text(encoding="utf-8"))
            result = decompose(inputs)
            print(json.dumps({key: result[key] for key in ("status", "decomposition_file", "decomposition_run_dir", "ready_for_planning", "error") if key in result}, indent=2))
            return 1 if result["status"] == "failed" else 0
        if args.command == "resources":
            from .resource_schema import ResourceSettings
            from .resource_scheduler import select_placements
            from .hardware import discover_hardware
            from .lmstudio_placement import LMStudioPlacementBackend
            from pathlib import Path
            from uuid import uuid4
            settings = ResourceSettings.model_validate_json(Path(args.resource_config).read_text(encoding="utf-8"))
            config = ModelConfig(**({"base_url": args.base_url} if args.base_url else {}))
            backend = LMStudioPlacementBackend(config.base_url, settings.lms_executable)
            hardware, capabilities = discover_hardware(), backend.capabilities()
            estimates = {model: backend.estimate(model, settings, settings.parallel_slots) for model in args.models}
            report = {"hardware": hardware.model_dump(), "capabilities": capabilities.model_dump(), "loaded_models": backend.inventory(),
                      "settings": settings.model_dump(), "estimates": {key: value.model_dump() for key, value in estimates.items()},
                      "read_only": True}
            try:
                if len(estimates) > 1 and not capabilities.concurrent_models:
                    raise ValueError("Backend cannot preserve concurrent agent inference")
                placements, _ = select_placements(estimates.values(), hardware, capabilities, settings)
                report.update(feasible=True, placements={key: value.model_dump() for key, value in placements.items()})
            except Exception as exc:
                report.update(feasible=False, error=f"{type(exc).__name__}: {exc}")
            directory = Path(args.output) / uuid4().hex
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"feasible": report["feasible"], "inspection": str(directory.resolve() / "inspection.json"),
                              "error": report.get("error"), "placements": {key: value["device"] for key, value in report.get("placements", {}).items()}}, indent=2))
            return 0 if report["feasible"] else 1
        task = load_task(args.task)
        if args.command == "validate":
            try:
                get_adapter(task)
                supported, reason = True, None
            except ValueError as exc:
                supported, reason = False, str(exc)
            print(json.dumps({"valid": True, "task_id": task.task_id, "runtime_supported": supported,
                              "runtime_error": reason, "tests": len(task.tests)}, indent=2))
            return 0 if supported else 2
        values = {key: getattr(args, key) for key in ("model", "planner_model", "ux_model", "architect_model", "coordinator_model", "qa_model", "review_max_tokens", "base_url", "temperature", "top_p", "seed", "max_tokens", "planner_max_tokens") if getattr(args, key) is not None}
        values.update({key: getattr(args, key) for key in ('manual_verification', 'max_repair_attempts')
                       if getattr(args, key) is not None})
        config = ModelConfig(**values, **policy_values(args))
        spec = task.model_dump(mode="json")
        if args.timeout_ms is not None:
            spec["verification"]["timeout_ms"] = args.timeout_ms
        if args.no_screenshots:
            spec["verification"]["screenshots"] = False
        graph = collaborative_app if args.workflow == "collaborative" else app
        inputs = {"task_spec": spec, "model_config": config.model_dump(), "output_root": args.output}
        inputs.update(variance_condition=args.variance_condition)
        if args.variance_profile:
            inputs['variance_profile'] = args.variance_profile
        if args.variance_profile_hash:
            inputs['variance_profile_hash'] = args.variance_profile_hash
        if args.decompose:
            inputs.update(decompose=True, task_path=args.task)
        if args.decomposition:
            inputs.update(decomposition_path=args.decomposition, task_path=args.task)
        if args.analysis:
            inputs["analysis_file"] = args.analysis
        if args.resource_config:
            from pathlib import Path
            inputs["resource_config"] = json.loads(Path(args.resource_config).read_text(encoding="utf-8"))
        log_event("run", "started", workflow=args.workflow, task_id=task.task_id, output=args.output,
                  resource_placement=bool(args.resource_config))
        result = graph.invoke(inputs)
        log_event("run", "finished", state=result, status=result["status"], output_file=result.get("output_file"))
        print(json.dumps({"status": result["status"], "run_dir": result.get("run_dir", result.get("decomposition_run_dir")),
                          "output_file": result.get("output_file"), "error": result.get("error"),
                          'verification_outcome': result.get('verification_outcome'),
                          'repair_count': result.get('implementation_work', {}).get('repair_count', 0),
                          **({"decomposition_file": result["decomposition_file"], "ready_for_planning": result["ready_for_planning"]}
                             if result.get("decomposition_file") else {}),
                          **({"acceptance": result.get("acceptance"), "qa_status": result.get("qa_status"),
                              "qa_error": result.get("qa_error")} if args.workflow == "collaborative" else {})}, indent=2))
        return 0 if result["status"] == "passed" else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
