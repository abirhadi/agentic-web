"""Hardware, memory estimates and placement contracts independent of backend."""
from pydantic import Field, field_validator
from .schema import StrictModel


GiB = 1024 ** 3


class GPU(StrictModel):
    device_id: str
    name: str
    total_bytes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)


class Hardware(StrictModel):
    gpus: list[GPU]
    ram_total_bytes: int = Field(gt=0)
    ram_free_bytes: int = Field(ge=0)
    cpu_threads: int = Field(ge=1)
    discovery_notes: list[str] = Field(default_factory=list)


class BackendCapabilities(StrictModel):
    concurrent_models: bool
    cpu_gpu_concurrency: bool
    explicit_gpu_device: bool
    parallel_slots_configurable: bool
    source: str


class ResourceSettings(StrictModel):
    context_length: int = Field(default=32768, ge=256)
    parallel_slots: int = Field(default=4, ge=1)
    gpu_safety_bytes: int = Field(default=GiB // 2, ge=0)
    ram_safety_bytes: int = Field(default=2 * GiB, ge=0)
    estimate_padding: float = Field(default=0.1, ge=0, le=1)
    prefetch_all_if_gpu_fit: bool = True
    lms_executable: str = "lms"
    # An explicitly supplied residence declaration can bind a manually loaded
    # instance when the backend does not expose its weight offload location.
    resident_devices: dict[str, str] = Field(default_factory=dict)


class OffloadProfile(StrictModel):
    gpu_offload: str
    fraction: float = Field(gt=0, le=1)
    gpu_bytes: int = Field(gt=0)
    ram_bytes: int = Field(ge=0)

    @field_validator("gpu_offload")
    @classmethod
    def valid_offload(cls, value):
        if value == "max":
            return value
        try:
            ratio = float(value)
        except ValueError as exc:
            raise ValueError("gpu_offload must be 'max' or a ratio in (0, 1]") from exc
        if not 0 < ratio <= 1:
            raise ValueError("gpu_offload ratio must be in (0, 1]")
        return value


class ModelEstimate(StrictModel):
    model: str
    context_length: int = Field(ge=256)
    parallel_slots: int = Field(ge=1)
    weight_bytes: int = Field(gt=0)
    gpu_bytes: int = Field(ge=0)
    ram_on_gpu_bytes: int = Field(ge=0)
    ram_on_cpu_bytes: int = Field(gt=0)
    gpu_on_cpu_bytes: int = Field(default=0, ge=0)
    kv_and_overhead_bytes: int = Field(ge=0)
    cpu_cost: float = Field(gt=0)
    source: str
    diagnostics: dict = Field(default_factory=dict)
    offload_profiles: list[OffloadProfile] = Field(default_factory=list)


class Placement(StrictModel):
    model: str
    device: str  # 'cpu' or an inventory GPU device_id
    gpu_offload: str
    gpu_allocations: dict[str, int]
    ram_bytes: int = Field(ge=0)
    estimate: ModelEstimate

    @field_validator("gpu_offload")
    @classmethod
    def valid_offload(cls, value):
        if value in {"off", "max"}:
            return value
        try:
            ratio = float(value)
        except ValueError as exc:
            raise ValueError("gpu_offload must be 'off', 'max', or a ratio in (0, 1]") from exc
        if not 0 < ratio <= 1:
            raise ValueError("gpu_offload ratio must be in (0, 1]")
        return value


class Resident(StrictModel):
    instance_id: str
    placement: Placement
    managed: bool = False
    active: int = 0
    pins: int = 0


class ResourceAdmissionError(RuntimeError):
    pass
