
from typing import TypedDict, NotRequired, Any
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from pathlib import Path
from uuid import uuid4

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    timeout=240.0,
    max_retries=1
)


# Shared workflow state

class WorkflowState(TypedDict):
    requirements: str
    plan: NotRequired[str]
    result: NotRequired[str]
    html_path: NotRequired[str]
    verification: NotRequired[dict[str, Any]]



# Agent 1: Planning

def planner(state: WorkflowState):
    print("[Planner] Calling Gemma 4 E4B...")

    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior web application architect. "
                    "Your responsibility is to transform user "
                    "requirements into a clear, actionable "
                    "website implementation plan. "
                    "Identify the page structure, core features, "
                    "data structures, and JavaScript interactions. "
                    "Do not generate HTML or JavaScript code."
                )
            },
            {
                "role": "user",
                "content": state["requirements"]
            }
        ],
        temperature=0.5,
        top_p=0.95,
        max_tokens=3072,
        stream=False
    )

    plan = response.choices[0].message.content

    if not plan or not plan.strip():
        raise RuntimeError(
            "Gemma returned an empty plan. "
            f"Finish reason: {response.choices[0].finish_reason}"
        )

    print("[Planner] Plan generated successfully.")

    return {"plan": plan}



# Agent 2: Implementation

def implementer(state: WorkflowState):
    print("[Implementer] Generating website with Gemma...")

    system_prompt = """
You are an expert frontend software engineer.

Create a complete, functional, responsive single-page
website using HTML5, CSS3, and vanilla JavaScript.

MANDATORY REQUIREMENTS:
1. Deliver one complete index.html file.
2. Embed all CSS and JavaScript inside that file.
3. Display at least six fictional events.
4. Implement functional event search.
5. Implement functional category filtering.
6. Implement a form for adding new events.
7. Make registration buttons produce visible feedback.
8. Use responsive layouts and accessible form labels.
9. Do not use external libraries or backend services.
10. Use textContent for displaying user-submitted text.

IMPORTANT:
- Original user requirements take priority.
- The planning document is advisory.
- Do not introduce unnecessary routing or authentication.
- Do not implement simulated backend APIs.
- Do not generate placeholder functions.
- All interactive features must actually work.

OUTPUT FORMAT:
Return ONLY the complete HTML source code.
Start with <!DOCTYPE html> and end with </html>.
Do not wrap the output in Markdown code fences.
"""

    user_prompt = f"""
ORIGINAL USER REQUIREMENTS:
{state['requirements']}

PLANNER'S IMPLEMENTATION PLAN:
{state['plan']}

Implement the complete website.
"""

    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        top_p=0.95,
        max_tokens=12000,
        stream=False
    )

    content = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason

    # Keep each run independent.
    output_dir = Path("runs") / "step4" / uuid4().hex[:12]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw response for debugging.
    (output_dir / "raw_response.txt").write_text(
        content, encoding="utf-8"
    )

    # Extract a complete HTML document.
    lower = content.lower()
    start = lower.find("<!doctype html")
    end = lower.rfind("</html>")

    if start == -1 or end <= start:
        raise RuntimeError(
            "Model did not produce a complete HTML document. "
            f"Finish reason: {finish_reason}. "
            f"Raw response: {output_dir / 'raw_response.txt'}"
        )

    html = content[start:end + len("</html>")]

    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"[Implementer] Saved: {html_path}")

    return {
        "result": (
            f"Website generated: {html_path}\n"
            f"Finish reason: {finish_reason}"
        ),
        "html_path": str(html_path.resolve())
    }


from browser_verifier import verify_site


def verifier(state: WorkflowState):
    print("[Verifier] Starting Playwright checks...")

    report = verify_site(state["html_path"])

    print(
        "[Verifier] Verification status:",
        report["status"]
    )

    for name, passed in report["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    return {"verification": report}





# Build the execution graph
builder = StateGraph(WorkflowState)

builder.add_node("planner", planner)
builder.add_node("implementer", implementer)
builder.add_node("verifier", verifier)

builder.add_edge(START, "planner")
builder.add_edge("planner", "implementer")
builder.add_edge("implementer", "verifier")
builder.add_edge("verifier", END)

app = builder.compile()

# Run directly from terminal only
if __name__ == "__main__":
    result = app.invoke({
        "requirements": "Build an EventHub website"
    })

    print("\nFinal state:")
    print(result)
