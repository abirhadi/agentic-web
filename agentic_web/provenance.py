"""Exclusive stage records form an append-only experiment ledger."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def filesystem_path(value):
    """Use Windows extended paths for artifact I/O without a machine-wide setting."""
    path = Path(value)
    if os.name != 'nt':
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith('\\\\?\\'):
        return Path(absolute)
    if absolute.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + absolute[2:])
    return Path('\\\\?\\' + absolute)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def record_stage(state, role, status, *, details=None):
    directory = Path(state["run_dir"])
    ledger = directory / "ledger"
    ledger.mkdir(exist_ok=True)
    prefixes = {"task_loader": ("input", "task", "requirements", "model_config", "models_available", "model_availability", "verification_plan", "resource_config"),
                "planner": ("planner", "plan.json", "resources.planner", "resources.specialists"), "ux": ("ux",), "architect": ("architect", "architecture"),
                "coordinator": ("coordinator", "coordination"), "implementer": ("implementer", "implementation_contract"),
                "verifier": ("verification_results",), "qa": ("qa",), "acceptance_gate": ("acceptance",),
                "resource_cleanup": ("resources.cleanup",),
                "task_decomposition": ("decomposition", "task.json", "model_config", "resource"),
                "requirements_analyst": ("intake", "analysis", "input", "task.json", "model_config",
                                         "original_request", "supporting_context", "resource")}[role]
    paths = [path for path in directory.iterdir() if path.is_file() and path.name.startswith(prefixes)]
    if role == 'task_loader' and (directory / 'variance_context.json').exists():
        paths.append(directory / 'variance_context.json')
    paths.extend(path for path in directory.iterdir() if path.is_file() and
                 (path.name == f"resources.inference.{role}.json" or path.name == f"resources.{role}.json") and path not in paths)
    if role == "implementer" and state.get("html_path"):
        paths.append(Path(state["html_path"]))
    record = {"stage": role, "status": status, "recorded_at": datetime.now(timezone.utc).isoformat(),
              "requirement_hash": state.get("requirement_hash"), "details": details or {},
              "input_hashes": {name: fingerprint(state[name]) for name in ("plan", "ux_review", "architecture_review", "coordinated_specification", "verification") if name in state},
              "artifact_hashes": {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    # No stage rewrites an earlier ledger event, including at branch joins.
    with (ledger / f"{role}.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
