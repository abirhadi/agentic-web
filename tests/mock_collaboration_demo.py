"""Persist Phase 2 graph demonstrations: mocked models, real Chromium, no LM calls."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_web.client import ModelConfig
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from mock_client import ROOT


if __name__ == "__main__":
    config = ModelConfig.model_validate_json((ROOT / "phase2_models.json").read_text())
    graph = build_graph(CollaborationMockClient(), workflow="collaborative")
    summaries = []
    for task in ("eventhub", "todo", "catalog"):
        result = graph.invoke({"task_path": str(ROOT / f"tasks/{task}.json"), "model_config": config.model_dump(),
                              "output_root": str(ROOT / "runs/collaborative_mocked"), "execution_mode": "mocked-model-real-browser"})
        summaries.append({"task_id": task, "status": result["status"], "run_dir": result["run_dir"],
                          "qa_status": result.get("qa_status"), "acceptance": result.get("acceptance")})
    print(json.dumps(summaries, indent=2))
    raise SystemExit(0 if all(summary["status"] == "passed" for summary in summaries) else 1)
