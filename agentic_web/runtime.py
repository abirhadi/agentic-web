"""Runtime adapter registry. Only single_html is implemented and tested."""
from pathlib import Path
from typing import Protocol
from .schema import TaskSpec


class UnsupportedRuntime(ValueError):
    pass


class RuntimeAdapter(Protocol):
    implementation_instructions: str
    def validate(self, task: TaskSpec) -> None: ...
    def save(self, raw: str, task: TaskSpec, run_dir: Path) -> Path: ...
    def verify(self, path: Path, task: TaskSpec, output: Path) -> dict: ...


def extract_html(text: str) -> str:
    lower = text.lower()
    starts = [pos for pos in (lower.find("<!doctype html"), lower.find("<html")) if pos >= 0]
    if not starts or lower.rfind("</html>") < min(starts):
        raise ValueError("Model did not produce a complete HTML document")
    return text[min(starts):lower.rfind("</html>") + 7].strip() + "\n"


class SingleHTMLAdapter:
    capabilities = ('multiple_sections', 'navigation', 'responsive_layout', 'inline_css', 'inline_javascript')
    implementation_instructions = (
        "The runtime is single_html: return one complete HTML document with inline CSS and vanilla JavaScript. "
        "No Markdown, external dependencies, remote assets, or placeholder interactions."
    )

    def verify(self, path, task, output):
        from .verification import verify_application
        return verify_application(path, task, output)

    def validate(self, task):
        runtime = task.runtime
        if runtime.base_url or runtime.needs_database or runtime.needs_authentication:
            raise UnsupportedRuntime("single_html does not support service URLs, databases, or authentication")
        if not runtime.entrypoint or Path(runtime.entrypoint).suffix.lower() != ".html":
            raise UnsupportedRuntime("single_html requires an .html entrypoint")

    def save(self, raw, task, run_dir):
        self.validate(task)
        output = run_dir / "application" / task.runtime.entrypoint
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(extract_html(raw), encoding="utf-8")
        return output


ADAPTERS: dict[str, RuntimeAdapter] = {"single_html": SingleHTMLAdapter()}


def get_adapter(task):
    adapter = ADAPTERS.get(task.runtime.kind)
    if adapter is None:
        raise UnsupportedRuntime(f"Unsupported runtime: {task.runtime.kind}; implemented runtimes: single_html")
    adapter.validate(task)
    return adapter
