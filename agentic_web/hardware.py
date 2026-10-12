"""Read-only discovery without new dependencies: NVIDIA CLI and OS RAM APIs."""
import csv
import ctypes
import io
import os
from pathlib import Path
import subprocess
from .resource_schema import GPU, Hardware


def discover_hardware() -> Hardware:
    notes = []
    gpus = []
    try:
        command = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
        for row in csv.reader(io.StringIO(command.stdout)):
            identifier, name, total, free = [item.strip() for item in row]
            gpus.append(GPU(device_id=identifier, name=name, total_bytes=int(float(total) * 1024 ** 2),
                            free_bytes=int(float(free) * 1024 ** 2)))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        notes.append(f"NVIDIA discovery unavailable: {type(exc).__name__}: {exc}; no GPU capacity assumed")
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "total_page", "available_page",
                                                       "total_virtual", "available_virtual", "extended")]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("GlobalMemoryStatusEx failed; RAM admission cannot be performed")
        total, free = memory.total, memory.available
    elif Path("/proc/meminfo").exists():
        values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()}
        total, free = values["MemTotal"], values["MemAvailable"]
    else:
        raise OSError("RAM discovery unsupported on this OS; supply a hardware provider for this backend")
    return Hardware(gpus=gpus, ram_total_bytes=total, ram_free_bytes=free,
                    cpu_threads=os.cpu_count() or 1, discovery_notes=notes)
