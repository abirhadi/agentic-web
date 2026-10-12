"""Local A2 document evidence; extraction belongs exclusively to MarkItDown."""
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from zipfile import ZipFile


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_CONTEXT_BYTES = 1024 * 1024
MAX_DOCX_EXPANDED_BYTES = 50 * 1024 * 1024
MAX_DOCUMENTS = 20
SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt"}


class DocumentIngestionError(ValueError):
    pass


class _LocalOnlySession:
    def request(self, *args, **kwargs):
        raise DocumentIngestionError("Network access is disabled during document conversion")

    get = post = request


def _convert(data, extension):
    from markitdown import MarkItDown
    from markitdown.converters import DocxConverter, PdfConverter
    converter = MarkItDown(enable_builtins=False, enable_plugins=False,
                          requests_session=_LocalOnlySession())
    converter.register_converter(PdfConverter() if extension == ".pdf" else DocxConverter())
    return converter.convert_stream(BytesIO(data), file_extension=extension).text_content


def _decode(data, *, plain_text=False):
    encoding = "utf-8-sig"
    if plain_text and data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    text = data.decode(encoding, errors="strict")
    if "\x00" in text:
        raise DocumentIngestionError("NUL bytes are not supported in document text")
    return text.replace("\r\n", "\n").replace("\r", "\n") if plain_text else text


def _read_local(filename, root):
    # Do not let MarkItDown's permissive URL/path interface interpret this input.
    if not isinstance(filename, str) or not filename or "://" in filename or filename.startswith(("\\\\", "//")):
        raise DocumentIngestionError("Documents must be local file paths, not URLs or network shares")
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise DocumentIngestionError(f"Document is outside the allowed document root: {root}")
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise DocumentIngestionError("Only regular local files are supported")
        if info.st_size > MAX_FILE_BYTES:
            raise DocumentIngestionError(f"Document exceeds {MAX_FILE_BYTES} bytes; no content was read")
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise DocumentIngestionError(f"Document exceeds {MAX_FILE_BYTES} bytes; no conversion was attempted")
    return path, data


def ingest_documents(documents, artifact_dir, *, supporting_context=None, document_root=None):
    """Save exact intermediate Markdown and return the bounded assembled context.

    Relative paths resolve under document_root (default: process cwd). A caller
    may explicitly authorize another local root. Nothing in content is executed.
    Each invocation uses exclusive artifact creation in a new intake directory.
    """
    if not isinstance(documents, list) or len(documents) > MAX_DOCUMENTS:
        raise DocumentIngestionError(f"documents must be a list of at most {MAX_DOCUMENTS} local paths")
    root = Path(document_root or Path.cwd()).resolve(strict=True)
    if str(root).startswith(("\\\\", "//")) or not root.is_dir():
        raise DocumentIngestionError("document_root must be a local directory")
    directory = Path(artifact_dir) / "documents"
    directory.mkdir(exist_ok=False)
    records, sections, failures = [], [], []
    for index, filename in enumerate(documents, 1):
        identifier = f"document_{index:03d}"
        record = {"source_id": identifier, "filename": Path(filename).name if isinstance(filename, str) else None,
                  "type": Path(filename).suffix.lower() if isinstance(filename, str) else None,
                  "original_sha256": None, "markdown_sha256": None, "status": "failed", "diagnostics": []}
        records.append(record)
        try:
            path, data = _read_local(filename, root)
            record.update(path=str(path), type=path.suffix.lower(),
                          original_sha256=hashlib.sha256(data).hexdigest(), original_bytes=len(data))
            extension = path.suffix.lower()
            if extension not in SUPPORTED:
                raise DocumentIngestionError(f"Unsupported document format: {extension or '(none)'}")
            if extension == ".docx":
                with ZipFile(BytesIO(data)) as archive:
                    if sum(item.file_size for item in archive.infolist()) > MAX_DOCX_EXPANDED_BYTES:
                        raise DocumentIngestionError("DOCX expanded content exceeds the safety limit")
            if extension in {".pdf", ".docx"}:
                record["converter"] = {"name": "markitdown", "version": version("markitdown")}
                markdown = _convert(data, extension)
            else:
                record["converter"] = {"name": "direct_text", "encoding_policy": "UTF-8; BOM-marked UTF-16 for TXT"}
                markdown = _decode(data, plain_text=extension == ".txt")
            encoded = markdown.encode("utf-8")
            artifact = directory / f"{identifier}.md"
            with artifact.open("xb") as stream:
                stream.write(encoded)
            record.update(markdown_file=f"documents/{artifact.name}", markdown_sha256=hashlib.sha256(encoded).hexdigest(),
                          markdown_bytes=len(encoded))
            if not markdown.strip():
                message = "PDF has no extractable text; it may require OCR. OCR is not supported in A2." if extension == ".pdf" else "Document conversion produced empty content"
                raise DocumentIngestionError(message)
            if len(encoded) > MAX_CONTEXT_BYTES:
                raise DocumentIngestionError(f"Converted content exceeds {MAX_CONTEXT_BYTES} bytes; nothing was truncated")
            if extension == ".pdf":
                record["diagnostics"].append("Text extraction only; image/scanned content is not OCR-processed, including mixed PDFs.")
            record["status"] = "converted" if extension in {".pdf", ".docx"} else "ingested"
            # JSON metadata prevents a filename containing newlines from masquerading as a boundary.
            label = json.dumps({"source_id": identifier, "filename": record["filename"], "type": extension}, ensure_ascii=False)
            sections.append(f"<!-- BEGIN DOCUMENT {identifier} -->\nSource: {label}\n\n{markdown}\n<!-- END DOCUMENT {identifier} -->")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            record["diagnostics"].append(message)
            failures.append(f"{identifier}: {message}")
    context = "\n\n".join(([supporting_context] if supporting_context is not None else []) + sections)
    report = {"version": "1.0", "document_root": str(root), "sources": records,
              "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
              "status": "failed" if failures else "completed"}
    if len(context.encode("utf-8")) > MAX_CONTEXT_BYTES:
        failures.append(f"Combined supporting context exceeds {MAX_CONTEXT_BYTES} bytes; nothing was truncated")
        report["status"] = "failed"
    report["diagnostics"] = failures
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    if failures:
        raise DocumentIngestionError("; ".join(failures))
    with (directory / "assembled_context.md").open("xb") as stream:
        stream.write(context.encode("utf-8"))
    return context, report
