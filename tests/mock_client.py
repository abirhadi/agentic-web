"""Deterministic model substitute. Browser verification is still real Chromium."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class MockClient:
    def __init__(self):
        self.calls = []

    def complete(self, system, user, config, *, planner=False, response_schema=None):
        payload = json.loads(user)
        self.calls.append({"system": system, "payload": payload, "planner": planner, "config": config.model_dump(), "response_schema": response_schema})
        task = payload["task_spec"]
        if planner:
            content = json.dumps({"architecture": "Single offline document", "components": ["Main content", "Controls"],
                "data_models": ["Task-defined records"], "interactions": [r["description"] for r in task["features"]],
                "technology_constraints": task["technology_constraints"], "implementation_steps": ["Build markup", "Implement interactions"],
                "risks": [], "assumptions": [],
                "requirement_ids": [r["id"] for r in task["features"] + task["nonfunctional_requirements"]]})
        else:
            content = (ROOT / "tests" / "fixtures" / (task["task_id"] + ".html")).read_text(encoding="utf-8")
        return content, {"model": "mock-model", "requested_model": config.planner_model or config.model if planner else config.model,
                         "seed": config.seed + (not planner), "finish_reason": "stop", "elapsed_seconds": 0,
                         "usage": None, "mocked": True}
