"""Compatibility CLI. Original preserved in legacy/lmstudio_pipeline_original.py."""
from agentic_web.workflow import WorkflowState, app
from agentic_web.runtime import extract_html

def run():
    import os
    import sys
    from agentic_web.__main__ import main
    if len(sys.argv) == 1:
        sys.argv.extend(["generate", "--task", os.getenv("AGENTIC_WEB_TASK", "tasks/eventhub.json")])
    return main()

if __name__ == "__main__":
    raise SystemExit(run())
