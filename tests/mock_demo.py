"""Persist reproducible mocked-model demonstrations with real browser checks."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mock_client import MockClient
from agentic_web.workflow import build_graph


if __name__ == "__main__":
    graph = build_graph(MockClient())
    summaries = []
    for name in ("eventhub", "todo", "catalog"):
        result = graph.invoke({"task_path": f"tasks/{name}.json", "output_root": "runs/mocked", "execution_mode": "mocked-model-real-browser"})
        summaries.append({"task_id": name, "status": result["status"], "run_dir": result["run_dir"],
                          "error": result.get("error"), "coverage": result.get("verification", {}).get("coverage")})
    print(json.dumps(summaries, indent=2))
    raise SystemExit(0 if all(r["status"] == "passed" for r in summaries) else 1)
