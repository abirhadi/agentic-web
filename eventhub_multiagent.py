#!/usr/bin/env python3
"""Multi-model EventHub refinement and evaluation with LM Studio.

Works on websites created by gemma_eventhub_experiment.py.
Python 3.10+, standard library. Optional: playwright for browser testing.

Examples:
    python eventhub_multiagent.py --list-models
    python eventhub_multiagent.py --source eventhub_experiments --max-cases 1
    python eventhub_multiagent.py --source eventhub_experiments --browser --rounds 2
    python eventhub_multiagent.py --source eventhub_experiments --require-distinct --browser

Place this file, gemma_eventhub_experiment.py and multiagent_models.json together.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Reuse the exact acceptance tests and HTML extraction from the user's existing script.
from gemma_eventhub_experiment import extract_html, static_checks, browser_checks, EVENTHUB_PROMPT

ROLES = ("ux", "qa", "coordinator", "implementer")
DEFAULT_API = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-e4b"
DEFAULT_CONFIG = {
    "ux": {"model": DEFAULT_MODEL, "url": DEFAULT_API, "temperature": 0.55},
    "qa": {"model": DEFAULT_MODEL, "url": DEFAULT_API, "temperature": 0.35},
    "coordinator": {"model": DEFAULT_MODEL, "url": DEFAULT_API, "temperature": 0.35},
    "implementer": {"model": DEFAULT_MODEL, "url": DEFAULT_API, "temperature": 0.45},
}

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "evidence": {"type": "string"},
                "recommendation": {"type": "string"},
            },
            "required": ["priority", "evidence", "recommendation"],
        }},
    },
    "required": ["summary", "findings"],
}
PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "goal": {"type": "string"},
        "changes": {"type": "array", "items": {"type": "string"}},
        "must_preserve": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["goal", "changes", "must_preserve"],
}

UX_SYSTEM = """You are an independent frontend UX and accessibility reviewer.
Identify evidence-based design, navigation, responsive-layout, and usability issues.
Focus on issues demonstrable from supplied HTML, not invented browser outcomes.
Preserve functionality. Do not propose cosmetic changes merely for novelty.
Return only the JSON requested by the schema, no source code. Limit to five findings."""
QA_SYSTEM = """You are an independent JavaScript and frontend correctness reviewer.
Inspect functional requirements, event rendering, search/filter integration, DOM safety,
form behavior, data validation, and runtime risk. Cite concrete elements in the HTML.
Do not invent findings that cannot be justified by code. Return only JSON.
Limit to five high-value findings."""
COORD_SYSTEM = """You are a technical lead coordinating two independent model reviewers.
They may disagree. Prioritize reproducible correctness defects over aesthetics.
Reject unsupported reviewer claims; create at most 6 concrete implementation changes.
Preserve existing working functionality and distinctive visual design.
Ensure no external frameworks, libraries, CDNs, or network calls are introduced.
Return only JSON according to the provided schema."""
BUILD_SYSTEM = """You are an expert frontend implementer. Return a FULL, working HTML5 document
starting <!DOCTYPE html> and ending </html>; no explanations or Markdown fences.
Apply the technical lead's plan to the CURRENT WEBSITE below. Do not start from scratch
unless necessary. Retain EventHub behavior and all tested IDs. Keep HTML, CSS and
vanilla JS in one offline-compatible file; no external dependencies or network calls.
Do not add browser-executed string evaluation (eval / new Function).
Preserve six initial events, #events-grid, .event-card, #event-search,
#category-filter, #event-form, #event-name, #event-date, #event-time,
#event-location, #event-category, and #event-description."""


def write_json(file: Path, obj: Any) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def text_message(obj: dict) -> str:
    content = obj.get("content") or ""
    if isinstance(content, list):
        return "\n".join(item.get("text", "") if isinstance(item, dict) else str(item)
                         for item in content)
    return str(content)


def llm_call(role: str, settings: dict, system: str, prompt: str,
             *, seed: int, max_tokens: int, timeout: int, schema: dict | None,
             trace_dir: Path, trace: list[dict], label: str) -> str:
    url = settings.get("url", DEFAULT_API)
    if not url.endswith("/v1/chat/completions"):
        raise ValueError(f"Agent {role}: URL must end with /v1/chat/completions")
    if not settings.get("model"):
        raise ValueError(f"Agent {role}: missing model ID")
    payload: dict[str, Any] = {
        "model": settings["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": settings.get("temperature", 0.5), "top_p": 0.95,
        "seed": seed, "max_tokens": max_tokens, "stream": False,
    }
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": f"{role}_output", "strict": True, "schema": schema,
        }}
    stem = f"{len(trace)+1:02d}_{label}_{role}"
    write_json(trace_dir / f"{stem}_request.json", payload)
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("LM_STUDIO_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    response: dict | None = None
    last_error: str | None = None
    for attempt in range(3):
        try:
            request = Request(url, data=json.dumps(payload).encode("utf-8"),
                              headers=headers, method="POST")
            with urlopen(request, timeout=timeout) as connection:
                response = json.load(connection)
            break
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {message}"
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        if attempt < 2:
            time.sleep(2 ** attempt)
    elapsed = round(time.perf_counter() - started, 2)
    if response is None:
        trace.append({"role": role, "label": label, "model": settings["model"],
                      "seconds": elapsed, "error": last_error})
        raise RuntimeError(f"Agent {role} ({settings['model']}) failed: {last_error}")
    write_json(trace_dir / f"{stem}_response.json", response)
    choice = response["choices"][0]
    message = text_message(choice.get("message", {}))
    usage = response.get("usage") or {}
    record = {"role": role, "label": label, "model": settings["model"],
              "seconds": elapsed, "finish_reason": choice.get("finish_reason"),
              "prompt_tokens": usage.get("prompt_tokens"),
              "completion_tokens": usage.get("completion_tokens"), "error": ""}
    trace.append(record)
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"Agent {role} reached the {max_tokens} token limit")
    return message


def as_json(text: str, required: tuple[str, ...]) -> dict:
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        # Limited fallback for models that surround JSON with markdown fences.
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, flags=re.I)
        if not match:
            raise ValueError(f"Model returned invalid JSON: {exc}") from exc
        result = json.loads(match.group(1))
    if not isinstance(result, dict) or any(k not in result for k in required):
        raise ValueError(f"Incomplete structured agent response; required: {required}")
    return result


def evaluate(html: str, folder: Path, browser: bool) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    html_path = folder / "index.html"
    html_path.write_text(html, encoding="utf-8")
    checks = static_checks(html)
    result = {"sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
              "static": checks, "static_passed": sum(checks.values()),
              "static_total": len(checks)}
    if browser:
        # set_content from the original runner uses an about:blank document,
        # rather than opening the model-generated HTML as a local file.
        result["browser"] = browser_checks(html_path, folder)
    write_json(folder / "evaluation.json", result)
    return result


def browser_flags(evaluation: dict) -> dict[str, bool]:
    b = evaluation.get("browser") or {}
    if b.get("browser_status") != "completed":
        return {}
    return {
        "six_events": bool(b.get("six_events_ok")),
        "search": bool(b.get("search_ok")),
        "filter": bool(b.get("filter_ok")),
        "add_event": bool(b.get("add_event_ok")),
        "no_overflow": b.get("mobile_overflow") is False,
        "no_js_errors": not bool(b.get("js_errors")),
    }


def compare(previous: dict, candidate: dict, browser: bool) -> tuple[bool, list[str]]:
    """Non-regression check; does NOT claim an aesthetic improvement."""
    issues = []
    for key, passed in previous["static"].items():
        if passed and not candidate["static"].get(key):
            issues.append(f"Static regression: {key}")
    # Required static contract must be present even if the original omitted it.
    for key in ("html_document", "embedded_css", "embedded_javascript",
                "all_test_ids", "event_card_class"):
        if not candidate["static"].get(key):
            issues.append(f"Required contract missing: {key}")
    if browser:
        orig_b = previous.get("browser") or {}
        new_b = candidate.get("browser") or {}
        if new_b.get("browser_status") != "completed":
            issues.append("Candidate browser verification did not complete")
        else:
            prev_flags, new_flags = browser_flags(previous), browser_flags(candidate)
            for key, passed in prev_flags.items():
                if passed and not new_flags.get(key):
                    issues.append(f"Browser regression: {key}")
            if not new_flags.get("no_js_errors"):
                issues.append("Candidate has JavaScript exceptions")
            if new_b.get("mobile_overflow") is True:
                issues.append("Candidate has mobile overflow")
    if candidate["sha256"] == previous["sha256"]:
        issues.append("Identical HTML: no revision")
    return not issues, issues


def discover(source: Path) -> list[Path]:
    if source.is_file():
        if source.name.lower().endswith(".html"):
            return [source]
        raise ValueError("--source file must be an HTML document")
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    files = sorted(p for p in source.glob("*/index.html") if p.is_file())
    if (source / "index.html").is_file():
        files.insert(0, source / "index.html")
    if not files:
        raise ValueError(f"No index.html found in immediate child folders of {source}")
    return files


def list_models(config: dict, timeout: int) -> None:
    seen = set()
    for role, settings in config.items():
        url = settings["url"].replace("/v1/chat/completions", "/v1/models")
        if url in seen:
            continue
        seen.add(url)
        print(f"Models visible at {url}:")
        headers = {"Accept": "application/json"}
        if os.getenv("LM_STUDIO_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['LM_STUDIO_API_KEY']}"
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                info = json.load(response)
            for model in info.get("data", []):
                print("  ", model.get("id"))
        except Exception as exc:
            print(f"  Unavailable: {exc}")


def run_case(path: Path, config: dict, output: Path, args: argparse.Namespace) -> dict:
    case_id = path.parent.name if path.name == "index.html" else path.stem
    case_dir = output / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    trace: list[dict] = []
    started = time.perf_counter()
    row = {"case": case_id, "source": str(path), "accepted_rounds": 0,
           "total_rounds": args.rounds, "status": "error", "error": ""}
    try:
        current_html = path.read_text(encoding="utf-8")
        baseline = evaluate(current_html, case_dir / "baseline", args.browser)
        if args.browser and baseline.get("browser", {}).get("browser_status") != "completed":
            raise RuntimeError(f"Cannot evaluate baseline in browser: {baseline.get('browser', {}).get('browser_status')}")
        prev = baseline
        rejection_feedback: list[dict] = []
        write_json(case_dir / "models.json", config)
        for round_index in range(1, args.rounds + 1):
            print(f"  Round {round_index}: independent UX and QA reviews", flush=True)
            cycle = case_dir / f"round_{round_index:02d}"
            cycle.mkdir(exist_ok=True)
            observed = {"static": prev["static"], "browser": prev.get("browser"),
                        "previous_rejection_feedback": rejection_feedback[-2:]}
            # Fresh isolated prompts; reviewers cannot see one another's findings.
            html_context = f"EVENTHUB REQUIREMENTS:\n{EVENTHUB_PROMPT}\n\nCURRENT HTML:\n{current_html}"
            ux_text = llm_call("ux", config["ux"], UX_SYSTEM, html_context,
                               seed=args.seed+round_index*101, max_tokens=args.review_tokens,
                               timeout=args.timeout, schema=REVIEW_SCHEMA,
                               trace_dir=cycle, trace=trace, label="review")
            qa_text = llm_call("qa", config["qa"], QA_SYSTEM,
                               html_context + "\n\nAUTOMATED CHECKS:\n" + json.dumps(observed),
                               seed=args.seed+round_index*103, max_tokens=args.review_tokens,
                               timeout=args.timeout, schema=REVIEW_SCHEMA,
                               trace_dir=cycle, trace=trace, label="review")
            ux_review = as_json(ux_text, ("summary", "findings"))
            qa_review = as_json(qa_text, ("summary", "findings"))
            write_json(cycle / "ux_review.json", ux_review)
            write_json(cycle / "qa_review.json", qa_review)
            print("  Round {}: coordinator resolves reviews".format(round_index), flush=True)
            plan_text = llm_call(
                "coordinator", config["coordinator"], COORD_SYSTEM,
                "EVENTHUB REQUIREMENTS:\n" + EVENTHUB_PROMPT +
                "\nCURRENT AUTOMATED FINDINGS:\n" + json.dumps(observed) +
                "\nUX REVIEW:\n" + json.dumps(ux_review) +
                "\nQA REVIEW:\n" + json.dumps(qa_review),
                seed=args.seed+round_index*107, max_tokens=args.review_tokens,
                timeout=args.timeout, schema=PLAN_SCHEMA,
                trace_dir=cycle, trace=trace, label="plan")
            plan = as_json(plan_text, ("goal", "changes", "must_preserve"))
            write_json(cycle / "plan.json", plan)
            if not isinstance(plan["changes"], list):
                raise ValueError("Coordinator returned non-list changes")
            print("  Round {}: implementer creates full HTML".format(round_index), flush=True)
            prompt = ("EVENTHUB REQUIREMENTS:\n" + EVENTHUB_PROMPT +
                      "\n\nAPPROVED PLAN:\n" + json.dumps(plan) +
                      "\n\nCURRENT HTML:\n" + current_html)
            build_text = llm_call(
                "implementer", config["implementer"], BUILD_SYSTEM, prompt,
                seed=args.seed+round_index*109, max_tokens=args.build_tokens,
                timeout=args.timeout, schema=None,
                trace_dir=cycle, trace=trace, label="build")
            (cycle / "implementer_raw.txt").write_text(build_text, encoding="utf-8")
            candidate = extract_html(build_text)
            if not candidate:
                rejection = ["Implementer did not produce a complete HTML document"]
                write_json(cycle / "decision.json", {"accepted": False, "reasons": rejection})
                rejection_feedback.append({"round": round_index, "reasons": rejection})
                print(f"  Rejected: {rejection[0]}", flush=True)
                continue
            candidate_eval = evaluate(candidate, cycle / "candidate", args.browser)
            accepted, rejection = compare(prev, candidate_eval, args.browser)
            write_json(cycle / "decision.json", {"accepted": accepted, "reasons": rejection,
                                                     "note": "No automated aesthetic-quality claims"})
            print(f"  {'Accepted' if accepted else 'Rejected'}: {', '.join(rejection) if rejection else 'no measured regressions'}", flush=True)
            if accepted:
                current_html, prev = candidate, candidate_eval
                row["accepted_rounds"] += 1
                rejection_feedback.clear()
            else:
                rejection_feedback.append({"round": round_index, "reasons": rejection,
                    "candidate_static": candidate_eval["static"],
                    "candidate_browser": candidate_eval.get("browser")})
            # If rejected, later rounds review the original accepted version; the
            # full decision trace and rejected candidate remain available.
        (case_dir / "final.html").write_text(current_html, encoding="utf-8")
        row.update({"status": "completed", "baseline_static": baseline["static_passed"],
                    "final_static": prev["static_passed"],
                    "baseline_browser": browser_flags(baseline),
                    "final_browser": browser_flags(prev),
                    "baseline_sha256": baseline["sha256"], "final_sha256": prev["sha256"]})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  ERROR: {row['error']}", file=sys.stderr, flush=True)
    finally:
        row["duration_seconds"] = round(time.perf_counter() - started, 2)
        row["calls"] = trace
        row["total_prompt_tokens"] = sum(t.get("prompt_tokens") or 0 for t in trace)
        row["total_completion_tokens"] = sum(t.get("completion_tokens") or 0 for t in trace)
        write_json(case_dir / "trace.json", {"case": case_id, "runs": trace, "result": row})
    return row


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).with_name("multiagent_models.json")))
    parser.add_argument("--source", default="eventhub_experiments")
    parser.add_argument("--output", default="eventhub_multiagent_results")
    parser.add_argument("--max-cases", type=int, default=1, help="Number of existing websites to refine; 0 = all")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--browser", action="store_true", help="Playwright validation and screenshots")
    parser.add_argument("--require-distinct", action="store_true", help="Fail unless at least two distinct models are assigned")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--review-tokens", type=int, default=1300)
    parser.add_argument("--build-tokens", type=int, default=8500)
    parser.add_argument("--timeout", type=int, default=3000)
    return parser.parse_args()


def main() -> int:
    args = cli()
    for name in ("rounds", "review_tokens", "build_tokens", "timeout"):
        if getattr(args, name) < 1:
            print(f"--{name.replace('_', '-')} must be positive", file=sys.stderr)
            return 2
    if args.max_cases < 0:
        print("--max-cases must be >= 0", file=sys.stderr)
        return 2
    path = Path(args.config)
    try:
        config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else DEFAULT_CONFIG
        for role in ROLES:
            if role not in config or not config[role].get("model"):
                raise ValueError(f"Missing model configuration for role {role}")
            config[role].setdefault("url", DEFAULT_API)
        if args.list_models:
            list_models(config, args.timeout)
            return 0
        distinct = {(settings["url"], settings["model"]) for settings in config.values()}
        if args.require_distinct and len(distinct) < 2:
            raise ValueError("This is one model playing four roles. Configure at least two distinct model IDs in multiagent_models.json")
        if len(distinct) < 2:
            print("WARNING: Only one unique model. This is a MULTI-AGENT workflow, not MULTI-MODEL collaboration.", flush=True)
        else:
            print(f"Using {len(distinct)} distinct (server, model) configurations.", flush=True)
        source_files = discover(Path(args.source))
        if args.max_cases:
            source_files = source_files[:args.max_cases]
        output = Path(args.output)
        output.mkdir(exist_ok=True, parents=True)
        records = []
        print(f"Refining {len(source_files)} website(s), {args.rounds} round(s) each.", flush=True)
        for index, source in enumerate(source_files, 1):
            print(f"[{index}/{len(source_files)}] {source}", flush=True)
            record = run_case(source, config, output, args)
            records.append(record)
            write_json(output / "report.json", {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "distinct_model_assignments": len(distinct),
                "browser_enabled": args.browser,
                "cases": records,
            })
            with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case", "source", "status", "error",
                    "accepted_rounds", "total_rounds", "baseline_static", "final_static",
                    "duration_seconds", "total_prompt_tokens", "total_completion_tokens"])
                writer.writeheader()
                for rec in records:
                    writer.writerow({key: rec.get(key, "") for key in writer.fieldnames})
        print(f"Finished. Results: {output.resolve()}")
        return 0 if all(r["status"] == "completed" for r in records) else 1
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"Configuration/error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
