"""Simulated hardware and placement backend, including actual allocation accounting."""
import threading
from agentic_web.resource_schema import BackendCapabilities, GPU, Hardware, ModelEstimate, OffloadProfile, GiB, ResourceAdmissionError


class SimulatedBackend:
    def __init__(self, gpu_sizes=(16 * GiB,), ram=64 * GiB, weights=None, concurrent=True, explicit_devices=True,
                 base_gpu_used=0, actual_gpu_factor=1.0):
        self.gpu_sizes, self.ram = gpu_sizes, ram
        self.weights = weights or {"heavy": 8 * GiB, "light": 4 * GiB, "tiny": GiB}
        self.instances = {}
        self.active = set()
        self.loads, self.unloads = [], []
        self.lock = threading.RLock()
        self.concurrent, self.explicit_devices = concurrent, explicit_devices
        self.base_gpu_used, self.actual_gpu_factor = base_gpu_used, actual_gpu_factor

    def capabilities(self):
        return BackendCapabilities(concurrent_models=self.concurrent, cpu_gpu_concurrency=self.concurrent,
            explicit_gpu_device=self.explicit_devices, parallel_slots_configurable=True, source="simulated backend")

    def canonical_model(self, model):
        if model not in self.weights:
            raise ResourceAdmissionError(f"Unknown simulated model: {model}")
        return model

    def hardware(self):
        with self.lock:
            used = [self.base_gpu_used + int(sum(instance.gpu_allocations.get(str(index), 0)
                for instance in self.instances.values()) * self.actual_gpu_factor) for index in range(len(self.gpu_sizes))]
            return Hardware(gpus=[GPU(device_id=str(index), name="Simulated GPU", total_bytes=size,
                free_bytes=size - used[index]) for index, size in enumerate(self.gpu_sizes)],
                ram_total_bytes=self.ram, ram_free_bytes=self.ram - sum(instance.ram_bytes for instance in self.instances.values()), cpu_threads=64)

    def estimate(self, model, settings, slots):
        weight = self.weights[self.canonical_model(model)]
        kv = settings.context_length * slots * 1024
        overhead = 64 * 1024 ** 2
        size = int((weight + kv + overhead) * (1 + settings.estimate_padding))
        profiles = [OffloadProfile(gpu_offload=str(fraction), fraction=fraction,
            gpu_bytes=max(1, int(size * fraction)), ram_bytes=size) for fraction in (0.75, 0.5, 0.25)]
        return ModelEstimate(model=model, context_length=settings.context_length, parallel_slots=slots, weight_bytes=weight,
            gpu_bytes=size, ram_on_gpu_bytes=size, ram_on_cpu_bytes=size, kv_and_overhead_bytes=kv + overhead,
            cpu_cost=weight * settings.context_length * slots, source="simulated weights + KV + overhead",
            offload_profiles=profiles)

    def inventory(self):
        with self.lock:
            return [{"key": model, "loaded_instances": [{"id": identifier, "config": {
                "context_length": placement.estimate.context_length, "parallel": placement.estimate.parallel_slots}}
                for identifier, placement in self.instances.items() if placement.model == model]} for model in self.weights]

    def is_active(self, identifier):
        return identifier in self.active

    def load(self, placement, identifier):
        with self.lock:
            self.instances[identifier] = placement
            self.loads.append((identifier, placement))
        return identifier, 0.05

    def unload(self, identifier):
        with self.lock:
            if identifier in self.active:
                raise ResourceAdmissionError("Active simulated inference")
            del self.instances[identifier]
            self.unloads.append(identifier)
        return 0.02
