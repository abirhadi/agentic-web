"""Compatibility entry point for existing Studio sessions.

The original experiment is preserved in legacy/test_langgraph_original.py.
"""
from agentic_web.workflow import WorkflowState, app

if __name__ == "__main__":
    from lmstudio_pipeline import run
    raise SystemExit(run())
