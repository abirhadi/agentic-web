"""Application-independent task specification for agentic web generation.

Use inside the Conda environment:
    python web_task_spec.py validate tasks/eventhub.json
    python web_task_spec.py validate tasks/todo.json
    python web_task_spec.py validate tasks/inventory_webapp.json

This validates *requirements*, not generated application behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Feature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=8)
    acceptance_criteria: list[str] = Field(min_length=1)
    required: bool = True
    verification: Literal["browser", "api", "static", "manual"] = "browser"


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["single_html", "static_site", "web_app"]
    entrypoint: str | None = None
    base_url: HttpUrl | None = None
    # Startup commands are deliberately excluded. Any service startup must
    # use a locally reviewed runner, not a command written by an LLM.
    needs_database: bool = False
    needs_authentication: bool = False

    @model_validator(mode="after")
    def validate_runtime(self):
        if self.kind in ("single_html", "static_site") and not self.entrypoint:
            raise ValueError("Static and single-HTML tasks require an entrypoint")
        if self.kind == "web_app" and self.entrypoint is None and self.base_url is None:
            raise ValueError("Web apps require an entrypoint or a base_url")
        if self.base_url and self.base_url.host not in {"localhost", "127.0.0.1"}:
            raise ValueError("Only localhost web app URLs are allowed for local evaluation")
        return self


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    task_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=3)
    original_requirements: str = Field(min_length=20)
    runtime: RuntimeSpec
    technology_constraints: list[str] = Field(default_factory=list)
    features: list[Feature] = Field(min_length=1)
    design_freedom: list[str] = Field(default_factory=list)
    forbidden_assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_features(self):
        ids = [feature.id for feature in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError("Feature IDs must be unique")
        return self


def load_task(path: str | Path) -> TaskSpec:
    return TaskSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generic web task specs")
    sub = parser.add_subparsers(dest="action", required=True)
    validator = sub.add_parser("validate", help="Validate a task specification")
    validator.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "validate":
        task = load_task(args.path)
        print(json.dumps({
            "valid": True,
            "task_id": task.task_id,
            "runtime_kind": task.runtime.kind,
            "features": len(task.features),
            "required_features": sum(f.required for f in task.features),
        }, indent=2))


if __name__ == "__main__":
    main()
