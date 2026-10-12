"""Universal browser checks and task-provided acceptance tests. No generated code DSL."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from html.parser import HTMLParser

from .schema import Assertion, TaskSpec


# Framework-owned JavaScript only. Placeholder text deliberately never counts
# as a label; this same check runs at load and after interaction checkpoints.
ACCESSIBILITY_VIOLATIONS = r"""() => {
    const issues = [];
    const describe = e => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') +
        (e.hasAttribute('data-testid') ? '[data-testid="' + e.getAttribute('data-testid') + '"]' : '');
    const named = e => (e.labels && [...e.labels].some(l => l.textContent.trim())) ||
        e.getAttribute('aria-label')?.trim() ||
        (e.getAttribute('aria-labelledby') || '').split(/\s+/).some(id => document.getElementById(id)?.textContent.trim());
    if (!document.documentElement.lang.trim()) issues.push('Document needs a language');
    if (!document.title.trim()) issues.push('Document needs a title');
    for (const e of document.querySelectorAll('input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=reset]), select, textarea')) {
        if (!named(e)) issues.push(describe(e) + ': missing accessible label (placeholder is not a label)');
    }
    for (const e of document.querySelectorAll('button,input[type=button],input[type=submit],input[type=reset]')) {
        const nativeName = e.tagName === 'BUTTON' ? e.textContent.trim() : e.value.trim();
        const defaultName = e.tagName === 'INPUT' && !e.hasAttribute('value') && ['submit', 'reset'].includes(e.type);
        if (!(nativeName || defaultName || named(e))) issues.push(describe(e) + ': missing accessible name');
    }
    for (const e of document.querySelectorAll('img')) {
        if (!e.hasAttribute('alt')) issues.push(describe(e) + ': missing alt attribute');
    }
    return issues;
}"""


def assert_accessibility(page, timeout):
    try:
        page.wait_for_function(f"() => ({ACCESSIBILITY_VIOLATIONS})().length === 0", timeout=timeout)
    except Exception as exc:
        issues = page.evaluate(ACCESSIBILITY_VIOLATIONS)
        raise AssertionError("Accessibility violations: " + "; ".join(issues)) from exc


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = set()
        self.closed = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)

    def handle_endtag(self, tag):
        self.closed.add(tag)


def coverage(task, results, manual_verification='on'):
    statuses = {result["test_id"]: result["status"] for result in results}
    mapped = {}
    for requirement in task.requirements:
        tests = [test for test in task.tests if test.requirement_id == requirement.id]
        outcomes = [statuses.get(test.test_id, "unverified") for test in tests
                    if manual_verification != 'off' or test.test_type != 'manual']
        status = ("failed" if "failed" in outcomes else
                  "skipped" if not outcomes and manual_verification == 'off' and
                      (tests and all(t.test_type == 'manual' for t in tests) or not tests and requirement.verification == 'manual') else
                  "unverified" if not outcomes or any(s != 'passed' for s in outcomes) else "passed")
        mapped[requirement.id] = {"required": requirement.required, "status": status,
                                  "test_ids": [test.test_id for test in tests]}
    return mapped


def finish(task, report):
    manual = report.get('manual_verification', 'on')
    if manual == 'off':
        manual_ids = {t.test_id for t in task.tests if t.test_type == 'manual'}
        for result in report['tests']:
            if result['test_id'] in manual_ids:
                result.update(status='skipped', error=None, reason='Manual verification disabled')
    report["coverage"] = coverage(task, report["tests"], manual)
    required = [item["status"] for item in report["coverage"].values() if item["required"]]
    test_failure = any(t["required"] and t["status"] == "failed" for t in report["tests"])
    if report["errors"] or report.get('infrastructure_errors') or not report["checks"] or not all(report["checks"].values()) or "failed" in required or test_failure:
        report["status"] = "failed"
    elif "unverified" in required or not any(
            t.test_type != 'manual' and any(r['test_id'] == t.test_id and r['status'] == 'passed'
                                           for r in report['tests']) for t in task.tests):
        report["status"] = "unverified"
    else:
        report["status"] = "passed"
    report["test_counts"] = {status: sum(test["status"] == status for test in report["tests"])
                             for status in (("passed", "failed", "unverified", "skipped") if manual == 'off' else ("passed", "failed", "unverified"))}
    return report


def implementation_feedback(task, report, artifact):
    """Only deterministic, executable application failures can request repair."""
    from .schema import verification_readiness
    executable = {t['test_id']: t for t in verification_readiness(task)['executable_tests']}
    if report.get('infrastructure_errors') or report.get('checks', {}).get('worker_completed') is False or any(
            str(e).startswith('Browser unavailable or failed:') for e in report.get('errors', [])):
        return 'verification_infrastructure_error', []
    if not report.get('checks'):
        return 'verification_infrastructure_error', []
    feedback = []
    for result in report.get('tests', []):
        test = executable.get(result['test_id'])
        if test and result['status'] == 'failed' and result.get('failure_kind') not in {'unsupported', 'infrastructure'}:
            feedback.append({'requirement_ids': [test['requirement_id']], 'test_id': result['test_id'],
                'expected': test, 'observed': result, 'artifact': str(artifact),
                'evidence': {'key': 'tests.' + result['test_id'], 'value': result}})
    # These are framework-owned, executable checks, not model-authored diagnostics.
    for name, passed in report.get('checks', {}).items():
        if passed is False and name != 'browser_verification_enabled' and (name in {'html_structure', 'javascript_syntax'} or
                name.startswith(('desktop_', 'mobile_'))):
            feedback.append({'requirement_ids': [t.requirement_id for t in task.tests if name in t.universal_checks],
                'test_id': None, 'expected': {name: True}, 'observed': {name: passed}, 'artifact': str(artifact),
                'evidence': {'key': 'checks.' + name, 'value': passed,
                             'errors': report.get('errors', []), 'javascript_diagnostics': report.get('javascript_diagnostics', []),
                             'accessibility_violations': report.get('accessibility_violations', {})}})
    if report.get('status') == 'passed' and feedback:
        return 'verification_failed', feedback
    if report.get('status') == 'passed' and executable:
        outcomes = {r['test_id']: r['status'] for r in report.get('tests', [])}
        missing = [t['test_id'] for t in executable.values() if t['required'] and outcomes.get(t['test_id']) != 'passed']
        if missing or report.get('errors'):
            report['infrastructure_errors'] = [f'Inconsistent passed report; missing/nonpassing executable evidence: {missing}']
            return 'verification_infrastructure_error', []
    if report.get('status') == 'passed' and executable and any(
            r['test_id'] in executable and r['status'] == 'passed' for r in report.get('tests', [])):
        return 'verified', []
    return ('verification_failed' if feedback else 'unverified'), feedback


def failure_kind(exc):
    message = str(exc).lower()
    if type(exc).__name__ == 'TargetClosedError' or any(s in message for s in (
            'browser has been closed', 'browser closed', 'browser disconnected', 'browser crashed', 'page crashed')):
        return 'infrastructure'
    if any(s in message for s in ('while parsing css selector', 'unknown engine', 'invalid selector')):
        return 'unsupported'
    return 'application'


def assert_observable(page, assertion, timeout):
    from playwright.sync_api import expect
    locator = page.locator(assertion.selector)
    name, value = assertion.assertion, assertion.value
    if name == "not_checked":
        expect(locator).to_have_count(1, timeout=timeout)
        expect(locator).not_to_be_checked(timeout=timeout)
        return
    if name == "accessible_name":
        # Require explicit labeling for form controls before checking their
        # computed name; some browsers use placeholders as a name fallback.
        expect(locator).to_have_count(1, timeout=timeout)
        label_source = locator.evaluate("""e => !e.matches('input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=reset]),select,textarea') ||
            (e.labels && [...e.labels].some(l => l.textContent.trim())) || !!e.getAttribute('aria-label')?.trim() ||
            (e.getAttribute('aria-labelledby') || '').split(/\\s+/).some(id => document.getElementById(id)?.textContent.trim())""")
        if not label_source:
            raise AssertionError(f"{assertion.selector}: missing accessible label; placeholder is not a label")
        expect(locator).to_have_accessible_name(value if value is not None else re.compile(r"\S"), timeout=timeout)
        return
    methods = {"visible": "to_be_visible", "hidden": "to_be_hidden", "checked": "to_be_checked",
               "count": "to_have_count", "text": "to_have_text", "contains_text": "to_contain_text", "value": "to_have_value"}
    method = getattr(expect(locator), methods[name])
    if name in {"visible", "hidden", "checked"}:
        method(timeout=timeout)
    else:
        method(value, timeout=timeout)


def act(page, action):
    locator = page.locator(action.selector)
    if action.action == "select":
        locator.select_option(value=action.value)
    elif action.action in {"fill", "press"}:
        getattr(locator, action.action)(action.value)
    else:
        getattr(locator, action.action)()


SCRIPT_PATTERN = re.compile(r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script\s*>", re.I | re.S)


def inline_classic_scripts(source):
    """Return inline classic scripts with their HTML source positions."""
    scripts = []
    for match in SCRIPT_PATTERN.finditer(source):
        attrs = match.group("attrs")
        if re.search(r"\bsrc\s*=", attrs, re.I):
            continue
        type_match = re.search(r"\btype\s*=\s*(['\"])(.*?)\1", attrs, re.I | re.S)
        script_type = type_match.group(2).strip().lower() if type_match else ""
        if script_type and script_type not in {"text/javascript", "application/javascript"}:
            continue
        start_line = source.count("\n", 0, match.start("body")) + 1
        scripts.append((len(scripts), match.group("body"), start_line))
    return scripts


def javascript_syntax_preflight(browser, source):
    """Compile inline classic scripts without executing them."""
    diagnostics = []
    context = browser.new_context()
    try:
        page = context.new_page()
        session = context.new_cdp_session(page)
        session.send("Runtime.enable")
        for index, script, start_line in inline_classic_scripts(source):
            result = session.send("Runtime.compileScript", {"expression": script,
                "sourceURL": f"inline-script-{index + 1}.js", "persistScript": False})
            details = result.get("exceptionDetails")
            if not details:
                continue
            exception = details.get("exception", {})
            local_line = int(details.get("lineNumber", 0))
            diagnostics.append({"kind": "syntax_error", "script_index": index,
                "message": exception.get("description") or details.get("text", "JavaScript syntax error"),
                "html_line": start_line + local_line, "column": int(details.get("columnNumber", 0)) + 1})
    finally:
        context.close()
    return diagnostics


def page_error_diagnostic(error):
    message = str(error)
    stack = getattr(error, "stack", None) or ""
    diagnostic = {"kind": "runtime_error", "message": message}
    location = re.search(r"https://artifact\.invalid/index\.html:(\d+):(\d+)", stack)
    if location:
        diagnostic.update(html_line=int(location.group(1)), column=int(location.group(2)))
    if stack:
        diagnostic["stack"] = stack
    return diagnostic


def _verify(path: Path, task: TaskSpec, output: Path):
    from playwright.sync_api import sync_playwright
    settings = task.verification
    def log(message):
        with (output / "worker.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
    report = {"status": "failed", "checks": {}, "tests": [], "errors": [], "artifacts": {},
              "javascript_diagnostics": [],
              "scope": "single_html universal checks and manifest acceptance tests"}
    checks = report["checks"]
    source = path.read_text(encoding="utf-8")
    parser = Structure()
    parser.feed(source)
    checks["html_structure"] = (bool(re.search(r"<!doctype\s+html\b", source, re.I))
                                and {"html", "head", "body"} <= parser.tags
                                and {"html", "head", "body"} <= parser.closed)
    if not settings.enabled:
        report["checks"]["browser_verification_enabled"] = False
        report["errors"].append("Browser verification disabled; application acceptance is unverified")
        return finish(task, report)

    url = "https://artifact.invalid/index.html"
    # Reuse the existing working Chromium launch settings on Windows. The only
    # served resource is the in-memory artifact; never navigate to file:// or
    # mount the run directory in a server. Browser contexts are disposable.
    policy = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
              "img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; "
              "worker-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'")

    def new_page(browser, width=1366, height=900):
        log("Creating isolated browser context")
        context = browser.new_context(viewport={"width": width, "height": height},
                                      accept_downloads=False, service_workers="block", permissions=[])
        # WebRTC can use UDP outside HTTP routing/CSP. This runtime has no
        # peer-network capability; remove its constructors before app scripts.
        context.add_init_script("""for (const name of ['RTCPeerConnection', 'webkitRTCPeerConnection', 'RTCIceGatherer']) {
            Object.defineProperty(globalThis, name, {value: undefined, writable: false, configurable: false});
        }""")
        def route(request_route):
            request = request_route.request
            if request.url == url and request.is_navigation_request() and request.resource_type == "document":
                request_route.fulfill(status=200, content_type="text/html", body=source,
                                      headers={"Content-Security-Policy": policy})
            else:
                request_route.abort()
        context.route("**/*", route)
        # The response CSP connect-src 'none' blocks WebSockets as well as fetch.
        # A socket interception callback can stall navigation on this Windows
        # Playwright build; enforce the browser policy before any script runs.
        page = context.new_page()
        log("Page created")
        page.on("popup", lambda popup: popup.close())
        page.on("dialog", lambda dialog: dialog.dismiss())
        page.set_default_timeout(settings.timeout_ms)
        page.set_default_navigation_timeout(settings.timeout_ms)
        errors, diagnostics = [], []
        def page_error(error):
            errors.append(str(error))
            diagnostics.append(page_error_diagnostic(error))
        page.on("pageerror", page_error)
        return context, page, errors, diagnostics

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True,
                executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_PATH"))
            log("Chromium launched")
            report["browser_version"] = browser.version
            try:
                syntax_diagnostics = javascript_syntax_preflight(browser, source)
                report["javascript_diagnostics"].extend(syntax_diagnostics)
                checks["javascript_syntax"] = not syntax_diagnostics
                for diagnostic in syntax_diagnostics:
                    report["errors"].append(f"{diagnostic['message']} at HTML line {diagnostic['html_line']}, column {diagnostic['column']}")
                for label, width, height in [("desktop", 1366, 900), ("mobile", 390, 844)]:
                    context, page, errors, diagnostics = new_page(browser, width, height)
                    try:
                        page.goto(url, wait_until="load")
                        log(f"{label}: loaded")
                        checks[f"{label}_loads"] = True
                        checks[f"{label}_renders"] = bool(page.locator("body").inner_text().strip())
                        for index, text in enumerate(settings.required_content):
                            assert_observable(page, Assertion(
                                assertion="contains_text", selector="body", value=text), settings.timeout_ms)
                            checks[f"{label}_content_{index}"] = True
                        checks[f"{label}_no_horizontal_overflow"] = page.evaluate(
                            "document.documentElement.scrollWidth <= innerWidth + 2")
                        if settings.accessibility:
                            issues = page.evaluate(ACCESSIBILITY_VIOLATIONS)
                            checks[f"{label}_basic_accessibility"] = not issues
                            report.setdefault("accessibility_violations", {})[label] = issues
                        if settings.screenshots:
                            screenshot = output / f"{label}.png"
                            page.screenshot(path=str(screenshot), full_page=False)
                            report["artifacts"][f"{label}_screenshot"] = str(screenshot)
                        checks[f"{label}_no_javascript_errors"] = not errors
                        report["errors"].extend(errors)
                        for diagnostic in diagnostics:
                            if diagnostic not in report["javascript_diagnostics"]:
                                report["javascript_diagnostics"].append(diagnostic)
                        log(f"{label}: checks complete")
                    except Exception as exc:
                        checks[f"{label}_completed"] = False
                        report["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
                        if failure_kind(exc) == 'infrastructure':
                            report.setdefault('infrastructure_errors', []).append(str(exc))
                    finally:
                        log(f"{label}: closing context")
                        context.close()
                for test in task.tests:
                    result = {"test_id": test.test_id, "requirement_id": test.requirement_id,
                              "required": test.required, "status": "unverified", "error": None}
                    report["tests"].append(result)
                    if test.test_type == "universal":
                        missing = [name for name in test.universal_checks if name not in checks]
                        failed = [name for name in test.universal_checks if name in checks and not checks[name]]
                        result["status"] = "failed" if failed else "unverified" if missing else "passed"
                        result["error"] = f"Failed: {failed}; unavailable: {missing}" if failed or missing else None
                        continue
                    if test.test_type != "browser":
                        result["error"] = "Manual verification has no executable adapter"
                        continue
                    context, page, errors, diagnostics = new_page(browser)
                    log(f"{test.test_id}: starting")
                    phase = "load"
                    try:
                        page.goto(url, wait_until="load")
                        log(f"{test.test_id}: loaded")
                        for index, precondition in enumerate(test.preconditions):
                            phase = f"precondition[{index}]"
                            assert_observable(page, precondition, settings.timeout_ms)
                        for index, action in enumerate(test.actions):
                            phase = f"action[{index}]"
                            act(page, action)
                            for assertion_index, assertion in enumerate(action.assert_after):
                                phase = f"action[{index}].assert_after[{assertion_index}]"
                                assert_observable(page, assertion, settings.timeout_ms)
                            if settings.accessibility:
                                phase = f"accessibility_after_action[{index}]"
                                assert_accessibility(page, settings.timeout_ms)
                        for index, assertion in enumerate(test.expected_results):
                            phase = f"expected_result[{index}]"
                            assert_observable(page, assertion, settings.timeout_ms)
                        if settings.accessibility:
                            phase = "accessibility_final_state"
                            assert_accessibility(page, settings.timeout_ms)
                        if errors:
                            raise AssertionError("Uncaught JavaScript exceptions: " + "; ".join(errors))
                        result["status"] = "passed"
                    except Exception as exc:
                        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", phase=phase,
                                      runtime_errors=errors, runtime_diagnostics=diagnostics, failure_kind=failure_kind(exc))
                        if result['failure_kind'] == 'infrastructure':
                            report.setdefault('infrastructure_errors', []).append(str(exc))
                        elif result['failure_kind'] == 'unsupported':
                            result['status'] = 'unverified'
                        for diagnostic in diagnostics:
                            if diagnostic not in report["javascript_diagnostics"]:
                                report["javascript_diagnostics"].append(diagnostic)
                        if settings.screenshots:
                            screenshot = output / f"failure-{test.test_id}.png"
                            try:
                                page.screenshot(path=str(screenshot), full_page=False)
                                result["screenshot"] = str(screenshot)
                            except Exception:
                                pass
                    finally:
                        log(f"{test.test_id}: closing context")
                        context.close()
            finally:
                browser.close()
    except Exception as exc:
        report["errors"].append(f"Browser unavailable or failed: {type(exc).__name__}: {exc}")
        report['infrastructure_errors'] = [report['errors'][-1]]
    return finish(task, report)


def verify_application(path, task, output):
    """Execute browser code in a worker with a hard wall-clock budget."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "verification-task.json"
    task_path.write_text(task.model_dump_json(indent=2), encoding="utf-8")
    report_path = output / "verification.json"
    try:
        # File handles avoid a descendant retaining a pipe after worker timeout.
        log_path = output / "worker-output.log"
        worker = subprocess.Popen([sys.executable, "-m", "agentic_web.verification", str(Path(path).resolve()),
                                   str(task_path), str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=(os.name != "nt"), cwd=Path(__file__).resolve().parent.parent)
        try:
            worker.wait(timeout=task.verification.total_timeout_seconds)
        except subprocess.TimeoutExpired:
            # Stop only this worker and its owned Chromium/driver descendants.
            if os.name == "nt":
                cleanup = subprocess.run(["taskkill", "/PID", str(worker.pid), "/T", "/F"], capture_output=True, timeout=5)
                (output / "cleanup.log").write_text(cleanup.stdout.decode(errors="replace") + cleanup.stderr.decode(errors="replace"), encoding="utf-8")
            else:
                import signal
                os.killpg(worker.pid, signal.SIGKILL)
            worker.kill()
            worker.wait(timeout=5)
            raise TimeoutError("Browser verification exceeded its total time budget")
        if worker.returncode or not report_path.exists():
            details = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else "No worker report"
            raise RuntimeError(f"Verification worker exited {worker.returncode}: {details}")
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report = finish(task, {"checks": {"worker_completed": False}, "tests": [],
                              'infrastructure_errors': [f'{type(exc).__name__}: {exc}'],
                              "errors": [f"{type(exc).__name__}: {exc}"], "artifacts": {}})
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


if __name__ == "__main__":
    from .schema import load_task
    artifact, manifest, destination = sys.argv[1:]
    try:
        report = _verify(Path(artifact), load_task(manifest), Path(destination))
        (Path(destination) / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        import traceback
        (Path(destination) / "worker-output.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
