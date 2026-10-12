"""LM Studio native inventory/unload plus verified CLI placement controls."""
import json
import math
import re
import subprocess
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .resource_schema import BackendCapabilities, ModelEstimate, OffloadProfile, ResourceAdmissionError
from .client import API_TIMEOUT_SECONDS


def parse_memory_estimate(text):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    values = {}
    for field, number, unit in re.findall(r"Estimated (GPU|Total) Memory:\s*([\d.,]+)\s*(KiB|MiB|GiB|TiB|KB|MB|GB|TB|B)\b", clean):
        multiplier = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": 1024 ** 4,
                      "KB": 1000, "MB": 1000 ** 2, "GB": 1000 ** 3, "TB": 1000 ** 4}[unit]
        values[field] = math.ceil(float(number.replace(",", "")) * multiplier)
    if set(values) != {"GPU", "Total"}:
        raise ResourceAdmissionError("LM Studio memory-estimate output was not recognized; no guessed allocation")
    return values


class LMStudioPlacementBackend:
    def __init__(self, base_url, executable="lms"):
        parsed = urlparse(base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ResourceAdmissionError("Local LM Studio placement requires a localhost backend; remote/HPC backends need their own adapter")
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.executable = executable
        self._estimates = {}

    def _api(self, path, data=None):
        request = Request(self.root + path, data=None if data is None else json.dumps(data).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            return json.load(response)

    def _cli(self, args):
        result = subprocess.run([self.executable, *args], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=API_TIMEOUT_SECONDS)
        if result.returncode:
            raise ResourceAdmissionError(f"LM Studio CLI failed: {result.stderr.strip() or result.stdout.strip()}")
        # Successful estimate diagnostics are written to stderr by lms.
        # Keep JSON inventory stdout clean for its parser.
        return result.stdout + result.stderr if "--estimate-only" in args else result.stdout

    def inventory(self):
        return self._api("/api/v1/models")["models"]

    def capabilities(self):
        help_text = self._cli(["load", "--help"])
        if not all(flag in help_text for flag in ("--gpu", "--context-length", "--estimate-only", "--identifier")):
            raise ResourceAdmissionError("Installed lms CLI lacks required offload/context/estimate/instance controls")
        slots = "--parallel" in help_text
        return BackendCapabilities(concurrent_models=slots, cpu_gpu_concurrency=slots, explicit_gpu_device=False,
            parallel_slots_configurable=slots, source="installed lms load --help plus native instance parallel metadata")

    def model_info(self, model):
        for item in self.inventory():
            if item["key"] == model or any(instance["id"] == model for instance in item.get("loaded_instances", [])):
                if item.get("format") != "gguf" or item.get("type") != "llm":
                    raise ResourceAdmissionError("This adapter estimates and places local GGUF LLMs only")
                return item
        raise ResourceAdmissionError(f"Model unavailable in LM Studio native inventory: {model}")

    def canonical_model(self, model):
        return self.model_info(model)["key"]

    def estimate(self, model, settings, slots):
        info = self.model_info(model)
        if info.get("max_context_length") and settings.context_length > info["max_context_length"]:
            raise ResourceAdmissionError(f"Configured context exceeds the model's context capacity: {model}")
        key = (info["key"], settings.context_length, slots, settings.estimate_padding)
        if key in self._estimates:
            return self._estimates[key]
        if info["key"].startswith("-"):
            raise ResourceAdmissionError("Invalid model key for CLI placement")
        outputs = {}
        values = {}
        for mode in ("max", "0.75", "0.5", "0.25", "off"):
            outputs[mode] = self._cli(["load", info["key"], "--estimate-only", "--gpu", mode,
                "--context-length", str(settings.context_length), "--parallel", str(slots)])
            values[mode] = parse_memory_estimate(outputs[mode])
        params = info.get("params_string") or ""
        matches = re.findall(r"(?:A)?([\d.]+)\s*([BMK])", params.upper())
        # MoE active parameter count, when supplied, is usually the last count.
        count, suffix = matches[-1] if matches else (str(info["size_bytes"] * 8 / ((info.get("quantization") or {}).get("bits_per_weight") or 4)), "")
        parameters = float(count) * {"B": 1e9, "M": 1e6, "K": 1e3, "": 1}[suffix]
        pad = lambda value: math.ceil(value * (1 + settings.estimate_padding))
        profiles = [OffloadProfile(gpu_offload=mode, fraction=float(mode),
            gpu_bytes=pad(values[mode]["GPU"]),
            ram_bytes=pad(max(info["size_bytes"], values[mode]["Total"] - values[mode]["GPU"])))
            for mode in ("0.75", "0.5", "0.25")]
        estimate = ModelEstimate(model=model, context_length=settings.context_length, parallel_slots=slots,
            weight_bytes=info["size_bytes"], gpu_bytes=pad(values["max"]["GPU"]),
            # Conservative host allowance covers mapped weights/staging even
            # when CLI total memory is dominated by GPU allocation.
            ram_on_gpu_bytes=pad(max(info["size_bytes"], values["max"]["Total"] - values["max"]["GPU"])),
            ram_on_cpu_bytes=pad(values["off"]["Total"]), gpu_on_cpu_bytes=pad(values["off"]["GPU"]),
            kv_and_overhead_bytes=max(0, max(values["max"]["Total"], values["off"]["Total"]) - info["size_bytes"]),
            cpu_cost=parameters * settings.context_length * slots, source="lms --estimate-only at configured context/parallel slots",
            diagnostics={"raw_estimates": outputs, "parameter_count_used": parameters, "padding_fraction": settings.estimate_padding},
            offload_profiles=profiles)
        self._estimates[key] = estimate
        return estimate

    def live_instances(self):
        return json.loads(self._cli(["ps", "--json"]))

    def is_active(self, identifier):
        for item in self.live_instances():
            if item.get("identifier") == identifier:
                return item.get("status") != "idle" or item.get("queued", 0) > 0
        return False

    def load(self, placement, identifier):
        info = self.model_info(placement.model)
        started = time.perf_counter()
        self._cli(["load", info["key"], "--gpu", placement.gpu_offload,
            "--context-length", str(placement.estimate.context_length), "--parallel", str(placement.estimate.parallel_slots),
            "--identifier", identifier, "--yes"])
        for item in self.inventory():
            for instance in item.get("loaded_instances", []):
                if instance["id"] == identifier:
                    config = instance["config"]
                    if config.get("context_length") != placement.estimate.context_length or config.get("parallel", 1) < placement.estimate.parallel_slots:
                        raise ResourceAdmissionError(f"Backend did not apply requested context/parallel slots for {identifier}")
                    return identifier, time.perf_counter() - started
        raise ResourceAdmissionError("CLI-loaded instance not exposed by the configured native endpoint; daemon/endpoint mismatch")

    def unload(self, identifier):
        if self.is_active(identifier):
            raise ResourceAdmissionError(f"Refusing to unload an active or queued instance: {identifier}")
        started = time.perf_counter()
        self._api("/api/v1/models/unload", {"instance_id": identifier})
        return time.perf_counter() - started
