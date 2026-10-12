"""Backwards-compatible task schema import and validation command."""
from agentic_web.schema import Feature, RuntimeSpec, TaskSpec, load_task

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate"])
    parser.add_argument("path")
    args = parser.parse_args()
    task = load_task(args.path)
    print(json.dumps({"valid": True, "task_id": task.task_id, "features": len(task.features)}, indent=2))
