#!/usr/bin/env python3
"""Batch-test Gemma 4 E4B generating a single-file EventHub website in LM Studio.

Python 3.10+; standard library only for generation and static checks.
Optional browser checks: pip install playwright && python -m playwright install chromium

Usage:
    python gemma_eventhub_experiment.py --suite temperature --runs 3
    python gemma_eventhub_experiment.py --suite all --runs 2 --browser
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "google/gemma-4-e4b"
API_URL = "http://localhost:1234/v1/chat/completions"

SYSTEM_BASE = (
    "You are an expert frontend developer. Produce a complete, runnable single-file HTML5 "
    "website. Output ONLY the source HTML, starting with <!DOCTYPE html>, with CSS in "
    "<style> and JavaScript in <script>. No Markdown fences or explanations. "
    "Do not use CDNs, external assets, frameworks, or network requests."
)

EVENTHUB_PROMPT = """Build a modern, responsive single-page community events website called EventHub.

Functional requirements:
- Header with logo and Home, Events, About links.
- Hero with headline, short introduction, and an Explore Events button linking to the events section.
- Six fictional upcoming events in different categories, each displaying name, date, time, location,
  category, description, and a Register button. Register should display an on-page confirmation.
- Working live search by event name, and a working category dropdown filter.
- Working Add Event form that inserts a new event card without reloading the page.
- Footer with contact details.

Implementation contract (important for automatic functional tests):
- Place cards inside #events-grid; each card must have class .event-card.
- Live search input id="event-search" and category select id="category-filter".
- Add Event form id="event-form", with fields having ids:
  event-name, event-date (type=date), event-time (type=time), event-location,
  event-category (select), event-description (textarea).
- Make all form fields clearly labeled and the form submit button type="submit".
- Render events from a JavaScript array; the six initial events must appear after load.
- Give new events the same markup as the initial six; update filtering after adding.

Technical requirements: HTML5, embedded CSS3, vanilla JavaScript only. One complete index.html file.
Responsive layout for desktop and mobile, CSS Grid/Flexbox, accessible labels, semantic tags,
viewport meta tag, and no external dependencies. Use attractive but readable styling.

Output the full working HTML document only, without explanations or code fences."""

PERSONAS = {
    "minimal": "Use a restrained editorial aesthetic, generous whitespace, elegant typography, and muted colors.",
    "bold": "Use a vibrant festival-inspired aesthetic, strong color contrast, playful shapes, and energetic typography.",
}

COLUMNS = [
    "case_id", "suite", "variant", "temperature", "top_p", "top_k", "seed",
    "frequency_penalty", "presence_penalty", "duration_seconds", "prompt_tokens",
    "completion_tokens", "finish_reason", "html_extracted", "static_passed",
    "static_total", "browser_status", "event_cards", "search_ok", "filter_ok",
    "add_event_ok", "mobile_overflow", "js_errors", "html_sha256", "output_dir", "error",
]


class TagInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: set[str] = set()
        self.ids: set[str] = set()
        self.classes: set[str] = set()
        self.viewport = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)
        data = dict(attrs)
        if data.get("id"):
            self.ids.add(str(data["id"]))
        if data.get("class"):
            self.classes.update(str(data["class"]).split())
        if tag == "meta" and (data.get("name") or "").lower() == "viewport":
            self.viewport = True


def extract_html(response_text: str) -> str | None:
    """Prefer fenced HTML if present, otherwise extract a whole HTML document."""
    blocks = re.findall(r"```(?:html)?\s*([\s\S]*?)```", response_text, re.I)
    candidates = blocks + [response_text]
    for value in sorted(candidates, key=len, reverse=True):
        start = re.search(r"<!doctype\s+html\b|<html\b", value, re.I)
        end = list(re.finditer(r"</html\s*>", value, re.I))
        if start and end and end[-1].end() > start.start():
            return value[start.start():end[-1].end()].strip() + "\n"
    return None


def static_checks(html: str) -> dict[str, bool]:
    parser = TagInspector()
    parser.feed(html)
    needed_ids = (
        "events-grid", "event-search", "category-filter", "event-form",
        "event-name", "event-date", "event-time", "event-location",
        "event-category", "event-description",
    )
    result = {
        "html_document": "html" in parser.tags and "body" in parser.tags,
        "page_structure": all(t in parser.tags for t in ("head", "main", "nav", "footer", "h1")),
        "embedded_css": "style" in parser.tags,
        "embedded_javascript": "script" in parser.tags,
        "responsive_viewport": parser.viewport,
        "all_test_ids": all(x in parser.ids for x in needed_ids),
        "event_card_class": "event-card" in parser.classes or "event-card" in html,
        "register_button": "register" in html.lower(),
        "branding": "eventhub" in html.lower(),
    }
    return result


def get_json(url: str, timeout: int, api_key: str | None) -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout) as r:
        return json.load(r)


def request_completion(url: str, payload: dict, timeout: int, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=json.dumps(payload).encode("utf-8"),
                  headers=headers, method="POST")
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"HTTP {exc.code}: {message}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Could not reach LM Studio: {exc}") from exc
        time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")


def browser_checks(html_path: Path, folder: Path) -> dict:
    """Run an actual Chromium smoke test. Opt-in because this executes model-produced JS."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        return {"browser_status": "skipped: install playwright", "js_errors": ""}

    results: dict = {"browser_status": "failed", "js_errors": ""}
    errors: list[str] = []
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as exc:
                # On systems with a system Chromium but no Playwright browser download.
                system_chromium = (shutil.which("chromium") or
                                   shutil.which("chromium-browser") or
                                   shutil.which("google-chrome"))
                if "Executable doesn't exist" not in str(exc) or not system_chromium:
                    raise
                browser = playwright.chromium.launch(
                    headless=True, executable_path=system_chromium
                )
            page = browser.new_page(viewport={"width": 1365, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.set_content(html_path.read_text(encoding="utf-8"), wait_until="load")
            page.wait_for_timeout(450)
            page.screenshot(path=str(folder / "desktop.png"), full_page=True)
            cards = page.locator(".event-card")
            initial_count = cards.count()
            results["event_cards"] = initial_count
            results["six_events_ok"] = initial_count >= 6

            # Input-driven functionality checks; failures are recorded individually.
            try:
                search = page.locator("#event-search")
                search.fill("zz___impossible_event_name___zz")
                results["search_ok"] = cards.count() == 0
                search.fill("")
            except Exception as exc:
                results["search_ok"] = False
                results["search_error"] = str(exc)[:220]

            try:
                category = page.locator("#category-filter")
                options = category.locator("option").evaluate_all(
                    "opts => opts.map(o => ({value:o.value, text:o.textContent.trim()}))"
                )
                choice = next((o for o in options if o["value"] and
                              o["value"].lower() not in ("all", "any", "all categories")), None)
                if choice:
                    category.select_option(value=choice["value"])
                    filtered = cards.count()
                    results["filter_ok"] = 0 <= filtered <= initial_count and filtered != initial_count
                else:
                    results["filter_ok"] = False
                category.select_option(index=0)
            except Exception as exc:
                results["filter_ok"] = False
                results["filter_error"] = str(exc)[:220]

            try:
                page.locator("#event-name").fill("Automated Test Event 123")
                page.locator("#event-date").fill("2027-05-20")
                page.locator("#event-time").fill("17:00")
                page.locator("#event-location").fill("Local Test Center")
                cat = page.locator("#event-category")
                cat_options = cat.locator("option").evaluate_all(
                    "opts => opts.map(o => ({value:o.value, disabled:o.disabled}))"
                )
                choose = next((o["value"] for o in cat_options if o["value"] and not o["disabled"]), None)
                if choose:
                    cat.select_option(value=choose)
                page.locator("#event-description").fill("Generated by an automated browser check.")
                page.locator('#event-form button[type="submit"]').click()
                page.wait_for_timeout(200)
                results["add_event_ok"] = (
                    cards.count() == initial_count + 1
                    and page.get_by_text("Automated Test Event 123", exact=False).count() >= 1
                )
            except Exception as exc:
                results["add_event_ok"] = False
                results["add_error"] = str(exc)[:220]

            mobile = browser.new_page(viewport={"width": 390, "height": 844},
                                      device_scale_factor=1)
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            mobile.set_content(html_path.read_text(encoding="utf-8"), wait_until="load")
            mobile.wait_for_timeout(350)
            mobile.screenshot(path=str(folder / "mobile.png"), full_page=True)
            results["mobile_overflow"] = bool(mobile.evaluate(
                "document.documentElement.scrollWidth > window.innerWidth + 1"
            ))
            results["browser_status"] = "completed"
            browser.close()
    except Exception as exc:
        results["browser_status"] = f"failed: {str(exc)[:200]}"
    results["js_errors"] = " | ".join(errors)[:500]
    return results


def generate_cases(suite: str, runs: int, temperatures: list[float], base_seed: int) -> list[dict]:
    cases = []

    def add(group: str, variant: str, temp: float, freq: float, presence: float,
            persona: str, number: int):
        cases.append({
            "case_id": f"{len(cases)+1:03d}_{group}_{variant}_s{number+1}",
            "suite": group, "variant": variant, "temperature": temp, "top_p": 0.95,
            "top_k": 64, "seed": base_seed + number * 7919,
            "frequency_penalty": freq, "presence_penalty": presence,
            "system": SYSTEM_BASE + (" " + PERSONAS[persona] if persona else ""),
        })

    if suite in ("temperature", "all"):
        for temp in temperatures:
            for i in range(runs):
                add("temperature", f"t{temp:g}", temp, 0, 0, "", i)
    if suite in ("penalty", "all"):
        for penalty in (0.0, 0.5, 1.0):
            for i in range(runs):
                add("penalty", f"p{penalty:g}", 0.8, penalty, penalty, "", i)
    if suite in ("persona", "all"):
        for persona in PERSONAS:
            for i in range(runs):
                add("persona", persona, 0.7, 0, 0, persona, i)
    return cases


def save_csv(rows: list[dict], filename: Path) -> None:
    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=API_URL, help="LM Studio OpenAI-compatible completions endpoint")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--suite", choices=("temperature", "penalty", "persona", "all"),
                        default="temperature")
    parser.add_argument("--runs", type=int, default=2, help="Seed replicates per condition (default: 2)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    parser.add_argument("--base-seed", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=6500)
    parser.add_argument("--timeout", type=int, default=3000, help="Seconds per request")
    parser.add_argument("--output", default="eventhub_experiments")
    parser.add_argument("--browser", action="store_true", help="Run Chromium checks and screenshots (requires Playwright)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs <= 0 or args.max_tokens <= 0 or args.timeout <= 0:
        print("Runs, max-tokens, and timeout must be positive.", file=sys.stderr)
        return 2
    if not args.url.rstrip("/").endswith("/v1/chat/completions"):
        print("--url must point to /v1/chat/completions", file=sys.stderr)
        return 2
    key = os.getenv("LM_STUDIO_API_KEY")
    try:
        models = get_json(args.url.rsplit("/chat/completions", 1)[0] + "/models", args.timeout, key)
        ids = [entry.get("id") for entry in models.get("data", [])]
        if args.model not in ids:
            print(f"Warning: {args.model!r} not in /v1/models: {ids}", file=sys.stderr)
    except Exception as exc:
        print(f"Cannot query LM Studio. Start its local server first. Detail: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(EVENTHUB_PROMPT, encoding="utf-8")
    cases = generate_cases(args.suite, args.runs, args.temperatures, args.base_seed)
    rows: list[dict] = []
    import hashlib
    print(f"Model: {args.model} | Suite: {args.suite} | Requests: {len(cases)}")
    for index, case in enumerate(cases, start=1):
        folder = output / case["case_id"]
        folder.mkdir(exist_ok=True)
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": case["system"]},
                {"role": "user", "content": EVENTHUB_PROMPT},
            ],
            "temperature": case["temperature"], "top_p": case["top_p"],
            "top_k": case["top_k"], "seed": case["seed"],
            "frequency_penalty": case["frequency_penalty"],
            "presence_penalty": case["presence_penalty"],
            "max_tokens": args.max_tokens, "stream": False,
        }
        (folder / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        row = {k: case.get(k, "") for k in COLUMNS}
        row["output_dir"] = str(folder)
        started = time.perf_counter()
        print(f"[{index}/{len(cases)}] {case['case_id']} ...", flush=True)
        try:
            response = request_completion(args.url, payload, args.timeout, key)
            row["duration_seconds"] = round(time.perf_counter() - started, 2)
            (folder / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
            choice = response["choices"][0]
            content = choice.get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(part.get("text", "")) if isinstance(part, dict)
                                    else str(part) for part in content)
            content = content or ""
            (folder / "raw_output.txt").write_text(content, encoding="utf-8")
            row["finish_reason"] = choice.get("finish_reason", "")
            usage = response.get("usage") or {}
            row["prompt_tokens"] = usage.get("prompt_tokens", "")
            row["completion_tokens"] = usage.get("completion_tokens", "")
            html = extract_html(content)
            row["html_extracted"] = bool(html)
            if not html:
                row["error"] = "No complete HTML document detected; inspect raw_output.txt"
            else:
                row["html_sha256"] = hashlib.sha256(html.encode("utf-8")).hexdigest()
                html_file = folder / "index.html"
                html_file.write_text(html, encoding="utf-8")
                checks = static_checks(html)
                (folder / "static_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
                row["static_passed"] = sum(checks.values())
                row["static_total"] = len(checks)
                if args.browser:
                    browser = browser_checks(html_file, folder)
                    (folder / "browser_checks.json").write_text(json.dumps(browser, indent=2), encoding="utf-8")
                    row.update({key: browser[key] for key in
                                ("browser_status", "event_cards", "search_ok", "filter_ok",
                                 "add_event_ok", "mobile_overflow", "js_errors") if key in browser})
                else:
                    row["browser_status"] = "not requested"
            if row["finish_reason"] == "length":
                row["error"] = (row["error"] + "; " if row["error"] else "") + "Token limit reached"
        except (RuntimeError, KeyError, TypeError, ValueError) as exc:
            row["duration_seconds"] = round(time.perf_counter() - started, 2)
            row["error"] = str(exc)[:1000]
        rows.append(row)
        save_csv(rows, output / "results.csv")
        print(f"  HTML={row.get('html_extracted', False)} static={row.get('static_passed', '-')}/{row.get('static_total', '-')}"
              f" elapsed={row.get('duration_seconds')}s error={row.get('error') or '-'}", flush=True)

    summary = {
        "model": args.model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "total_requests": len(rows),
        "html_generated": sum(bool(r.get("html_extracted")) for r in rows),
        "error_count": sum(bool(r.get("error")) for r in rows),
        "unique_html_hashes": len({r["html_sha256"] for r in rows if r.get("html_sha256")}),
        "report": str(output / "results.csv"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinished. Report: {output / 'results.csv'}")
    print(json.dumps(summary, indent=2))
    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
