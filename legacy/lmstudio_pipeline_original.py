"""LangGraph -> LM Studio, using a planner and implementer.

Run from an activated Conda environment:
    python lmstudio_pipeline.py

Requires: pip install langgraph openai
LM Studio: http://127.0.0.1:1234 with google/gemma-4-e4b available.
"""

import json
import os
import time
from uuid import uuid4
from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

MODEL = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-e4b")
BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
OUTPUT_ROOT = Path("runs") / "lmstudio_pipeline"

client = OpenAI(base_url=BASE_URL, api_key="lm-studio", timeout=300.0, max_retries=0)


class WorkflowState(TypedDict):
    requirements: str
    run_dir: NotRequired[str]
    plan: NotRequired[str]
    output_file: NotRequired[str]
    planner_metrics: NotRequired[dict]
    implementer_metrics: NotRequired[dict]


def ask_model(system: str, user: str, *, temperature: float, seed: int,
              max_tokens: int) -> tuple[str, dict]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        max_tokens=max_tokens,
        stream=False,
    )
    elapsed = round(time.perf_counter() - started, 2)
    if not response.choices:
        raise RuntimeError("LM Studio returned no completion choices")
    choice = response.choices[0]
    content = choice.message.content or ""
    metrics = {
        "model": response.model,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "elapsed_seconds": elapsed,
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if response.usage else None,
    }
    return content, metrics


def planner(state: WorkflowState) -> dict:
    print("[Planner] Requesting a plan from", MODEL, flush=True)
    plan, metrics = ask_model(
        "You are a web application planner. Write a concise and concrete implementation plan. "
        "Do not write code. List the essential UI regions, data model, and JavaScript interactions.",
        state["requirements"],
        temperature=0.4,
        seed=1024,
        max_tokens=1200,
    )
    if not plan.strip():
        raise RuntimeError(f"Planner returned no visible text. Metadata: {metrics}")
    run_dir = Path(state.get("run_dir") or (OUTPUT_ROOT / uuid4().hex[:12]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.txt").write_text(plan, encoding="utf-8")
    return {"plan": plan, "planner_metrics": metrics, "run_dir": str(run_dir)}


def extract_html(text: str) -> str:
    lower = text.lower()
    starts = [pos for pos in (lower.find("<!doctype html"), lower.find("<html")) if pos >= 0]
    if not starts:
        raise ValueError("No HTML document start found in implementer response")
    beginning = min(starts)
    ending = lower.rfind("</html>")
    if ending < beginning:
        raise ValueError("Missing </html>; the output might be truncated")
    return text[beginning:ending + len("</html>")].strip() + "\n"


def implementer(state: WorkflowState) -> dict:
    print("[Implementer] Generating website from plan", flush=True)
    raw, metrics = ask_model(
        "You are an expert HTML/CSS/vanilla-JavaScript developer. "
        "Return only a COMPLETE standalone HTML document with inline CSS and JS. "
        "No Markdown code fences, no explanations, no external libraries, no remote assets. "
        "Implement functional interactions, not placeholder buttons.",
        "ORIGINAL REQUIREMENTS (must be preserved):\n"
        + state["requirements"]
        + "\n\nPLANNER'S PROPOSED IMPLEMENTATION:\n"
        + state["plan"],
        temperature=0.7,
        seed=1025,
        max_tokens=8000,
    )
    run_dir = Path(state["run_dir"])
    (run_dir / "raw_website_response.txt").write_text(raw, encoding="utf-8")
    try:
        html = extract_html(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid HTML response: {exc}. Generation metadata: {metrics}") from exc
    output_file = run_dir / "index.html"
    output_file.write_text(html, encoding="utf-8")
    print(f"[Implementer] Saved {output_file}", flush=True)
    return {"output_file": str(output_file), "implementer_metrics": metrics}


# Module-level compiled graph for LangSmith Studio imports.
_builder = StateGraph(WorkflowState)
_builder.add_node("planner", planner)
_builder.add_node("implementer", implementer)
_builder.add_edge(START, "planner")
_builder.add_edge("planner", "implementer")
_builder.add_edge("implementer", END)
app = _builder.compile()


def run() -> None:
    print(f"LM Studio: {BASE_URL} | model: {MODEL}")
    # Connectivity check. This can list downloaded models, not only loaded ones.
    models = client.models.list()
    available = [item.id for item in models.data]
    print("Models exposed:", ", ".join(available) or "(none)")
    if MODEL not in available:
        print("Note: model identifier was not listed; verify model key with 'lms ps' or /v1/models")

    requirements = (
        "Build EventHub, a responsive, single-page community event discovery site. "
        "Include a navigation bar, headline, at least six fictional events with "
        "date, category, and location, searchable event listings, a working category "
        "filter, a form that dynamically adds an event, registration confirmation "
        "buttons, and a footer. Use only one index.html with embedded CSS and "
        "vanilla JavaScript. Use accessible labels. All functionality must work "
        "offline when the HTML file is opened in a browser."
    )
    result = app.invoke({"requirements": requirements})
    summary = {
        "requirements": result["requirements"],
        "plan": result["plan"],
        "output_file": result["output_file"],
        "planner_metrics": result["planner_metrics"],
        "implementer_metrics": result["implementer_metrics"],
    }
    (Path(result["run_dir"]) / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nSuccess: graph completed. Open", Path(result["output_file"]).resolve())
    print("Planner finish reason:", result["planner_metrics"]["finish_reason"])
    print("Implementer finish reason:", result["implementer_metrics"]["finish_reason"])
    print("NOTE: HTML document extraction passed; browser functionality has not been validated.")


if __name__ == "__main__":
    run()
