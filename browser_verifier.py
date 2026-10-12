"""Deterministic browser smoke verification for standalone generated HTML.

Install inside Conda: python -m pip install playwright
                    python -m playwright install chromium
Usage: python browser_verifier.py path/to/index.html
Graph integration: from browser_verifier import verify_site
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def verify_site(html_path: str | Path, *, timeout_ms: int = 15000) -> dict[str, Any]:
    """Return an explicit pass/fail report; always save it beside the HTML.

    This is deliberately a smoke test, NOT proof that all EventHub interactions work.
    """
    from playwright.sync_api import sync_playwright

    path = Path(html_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Website not found: {path}")

    report: dict[str, Any] = {
        "html_path": str(path),
        "status": "failed",
        "checks": {},
        "runtime_errors": [],
        "artifacts": {},
        "scope": "Static HTML and browser smoke tests only; interactions not verified",
    }
    checks: dict[str, bool] = report["checks"]
    source = path.read_text(encoding="utf-8")
    checks["doctype"] = bool(re.search(r"<!doctype\s+html\b", source, re.I))
    checks["closing_html"] = bool(re.search(r"</html\s*>", source, re.I))

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_PATH"),
            )
            try:
                # Separate contexts emulate different viewport sizes.
                for label, width, height in [
                    ("desktop", 1366, 900),
                    ("mobile", 390, 844),
                ]:
                    context = browser.new_context(
                        viewport={"width": width, "height": height},
                        device_scale_factor=1,
                    )
                    try:
                        # Prevent remote assets/requests from reaching the network.
                        context.route('**/*', lambda route: route.abort())
                        page = context.new_page()
                        page.set_default_timeout(timeout_ms)
                        page.on("pageerror", lambda error: report["runtime_errors"].append(str(error)))
                        # Single-file website: render without external network access.
                        page.set_content(source, wait_until="load", timeout=timeout_ms)
                        page.wait_for_timeout(500)
                        rendered = page.locator("body").inner_text(timeout=timeout_ms).strip()
                        checks[f"{label}_renders_text"] = len(rendered) >= 30
                        checks[f"{label}_no_horizontal_overflow"] = bool(
                            page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
                        )
                        screenshot = path.parent / f"{label}.png"
                        page.screenshot(path=str(screenshot), full_page=True, timeout=timeout_ms)
                        report["artifacts"][f"{label}_screenshot"] = str(screenshot)
                    finally:
                        context.close()
            finally:
                browser.close()
    except Exception as exc:
        report["runtime_errors"].append(f"Browser verification failed: {type(exc).__name__}: {exc}")

    checks["no_uncaught_javascript_exceptions"] = not report["runtime_errors"]
    report["status"] = "passed" if all(checks.values()) else "failed"
    report_path = path.parent / "verification.json"
    report["artifacts"]["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser smoke test for generated standalone HTML")
    parser.add_argument("html_path", help="Path to generated index.html")
    args = parser.parse_args()
    report = verify_site(args.html_path)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
