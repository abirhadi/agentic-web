"""Console observability without changing graph state or experiment artifacts."""
from functools import wraps
import json
import logging
from pathlib import Path
import sys
import time


logger = logging.getLogger("agentic_web")


def configure_console_logging(level="INFO"):
    """Configure only our namespace; leave application/root handlers alone."""
    for handler in logger.handlers[:]:
        if getattr(handler, "_agentic_web_console", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(sys.stderr)
    handler._agentic_web_console = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def log_event(stage, event, *, state=None, level=logging.INFO, **values):
    if not logger.isEnabledFor(level):
        return
    run_dir = (state or {}).get("run_dir")
    fields = {"stage": stage, "event": event}
    if run_dir:
        fields["run"] = Path(run_dir).name
    fields.update(values)
    # JSON quoting keeps multiline diagnostics and concurrent records legible.
    logger.log(level, " ".join(f"{key}={value:.3f}" if isinstance(value, float)
               else f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in fields.items()))


def observed_node(stage, node):
    """Instrument the existing node; retain its return value and exceptions."""
    @wraps(node)
    def run(state):
        started = time.perf_counter()
        verification = state.get("task_spec", {}).get("verification", {}) if stage == "verifier" else {}
        log_event(stage, "started", state=state, **({"timeout_ms": verification.get("timeout_ms"),
                  "total_timeout_seconds": verification.get("total_timeout_seconds")} if verification else {}))
        try:
            result = node(state)
        except Exception as exc:
            log_event(stage, "failed", state=state, level=logging.ERROR,
                      elapsed_seconds=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
            raise
        context = {**state, **result}
        error = result.get("error") or result.get(f"{stage}_error")
        status = "failed" if error else "completed"
        values = {"elapsed_seconds": time.perf_counter() - started}
        if state.get("error") and stage in {"planner", "implementer", "verifier", "ux", "architect", "coordinator"}:
            status = "skipped"
        if stage == "task_loader":
            task = result.get("task_spec", {})
            values.update(task_id=task.get("task_id"), run_dir=result.get("run_dir"), tests=len(task.get("tests", [])))
        elif stage == "implementer" and result.get("output_file"):
            values["output_file"] = result["output_file"]
        elif stage == "verifier" and result.get("verification"):
            report = result["verification"]
            status = report.get("status", "unverified")
            tests = report.get("tests", [])
            checks = report.get("checks", {})
            values.update(checks_passed=sum(value is True for value in checks.values()), checks_total=len(checks),
                          verification_outcome=result.get('verification_outcome'),
                          repair_count=result.get('implementation_work', {}).get('repair_count', 0),
                          feedback_route=result.get('implementation_work', {}).get('route'),
                          tests_skipped=sum(test['status'] == 'skipped' for test in tests),
                          tests_passed=sum(test["status"] == "passed" for test in tests),
                          tests_failed=sum(test["status"] == "failed" for test in tests),
                          tests_unverified=sum(test["status"] == "unverified" for test in tests))
            for diagnostic in report.get("javascript_diagnostics", []):
                log_event(stage, "javascript_diagnostic", state=context, level=logging.WARNING,
                          kind=diagnostic.get("kind"), message=diagnostic.get("message"),
                          html_line=diagnostic.get("html_line"), column=diagnostic.get("column"))
            for test in tests:
                log_event(stage, "test_result", state=context,
                          level=logging.WARNING if test["status"] != "passed" else logging.INFO,
                          test_id=test["test_id"], status=test["status"], phase=test.get("phase"), error=test.get("error"))
        elif stage == "qa":
            status = result.get("qa_status", status)
            if result.get("qa_review"):
                values["findings"] = len(result["qa_review"]["findings"])
        elif stage == "acceptance_gate":
            acceptance = result.get("acceptance", {})
            status = acceptance.get("decision", status)
            values.update(accepted=acceptance.get("accepted"), reasons=acceptance.get("reasons", []),
                          qa_status=acceptance.get("qa_status"))
        if error:
            values["error"] = error
        level = logging.ERROR if status == "failed" else logging.WARNING if status in {"rejected", "skipped", "unverified"} else logging.INFO
        log_event(stage, status, state=context, level=level, **values)
        return result
    return run
