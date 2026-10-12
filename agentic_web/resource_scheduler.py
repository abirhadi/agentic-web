"""Deterministic admission and residency leases; no lock spans inference."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from threading import RLock
import time
from uuid import uuid4
import json
import logging
from typing import Protocol

from .hardware import discover_hardware
from .progress import log_event
from .resource_schema import (BackendCapabilities, Hardware, ModelEstimate, Placement, Resident,
                              ResourceAdmissionError, ResourceSettings)


class PlacementBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def canonical_model(self, model: str) -> str: ...
    def estimate(self, model: str, settings: ResourceSettings, slots: int) -> ModelEstimate: ...
    def inventory(self) -> list[dict]: ...
    def is_active(self, identifier: str) -> bool: ...
    def load(self, placement: Placement, identifier: str) -> tuple[str, float]: ...
    def unload(self, identifier: str) -> float: ...


def select_placements(estimates, hardware, capabilities, settings, residents=(), credited=()):
    """Prefer costly work on GPU, while admitting the entire concurrent stage."""
    gpu_free = {gpu.device_id: max(0, gpu.free_bytes - settings.gpu_safety_bytes) for gpu in hardware.gpus}
    ram_free = max(0, hardware.ram_free_bytes - settings.ram_safety_bytes)
    devices = list(gpu_free) if capabilities.explicit_gpu_device or len(gpu_free) == 1 else []
    ordered = sorted(estimates, key=lambda item: (-item.cpu_cost, item.model))

    def search(index, ram_free, gpu_free):
        if index == len(ordered):
            return {}, {}
        estimate = ordered[index]
        compatible = [resident for resident in residents if resident.placement.model == estimate.model
                      and resident.placement.estimate.context_length >= estimate.context_length
                      and resident.placement.estimate.parallel_slots >= estimate.parallel_slots]
        options = []
        # Reuse a compatible GPU instance first. Full GPU remains preferable to
        # CPU, but a scoped single-GPU CLI cannot promise multi-device selection.
        for resident in compatible:
            if resident.placement.device in devices and resident.instance_id not in credited:
                options.append((resident.placement, resident))
        profiles = [("max", 1.0, estimate.gpu_bytes, estimate.ram_on_gpu_bytes)]
        profiles.extend((profile.gpu_offload, profile.fraction, profile.gpu_bytes, profile.ram_bytes)
                        for profile in estimate.offload_profiles)
        for mode, _, gpu_bytes, ram_bytes in sorted(profiles, key=lambda item: -item[1]):
            for device in sorted(devices, key=lambda key: (gpu_free[key], key)):
                options.append((Placement(model=estimate.model, device=device, gpu_offload=mode,
                    gpu_allocations={device: gpu_bytes}, ram_bytes=ram_bytes, estimate=estimate), None))
        # During a reclaim evaluation, try a stronger offload before falling
        # back to the credited resident. This permits upgrading an idle partial
        # or smaller ready-model placement when capacity becomes available.
        for resident in compatible:
            if resident.placement.device in devices and resident.instance_id in credited:
                options.append((resident.placement, resident))
        for resident in compatible:
            if resident.placement.device == "cpu":
                options.append((resident.placement, resident))
        if estimate.gpu_on_cpu_bytes == 0 or gpu_free:
            options.append((Placement(model=estimate.model, device="cpu", gpu_offload="off",
                # CPU offload can retain GPU overhead. Its device is not
                # advertised, so reserve that overhead on every visible GPU.
                gpu_allocations={} if not estimate.gpu_on_cpu_bytes else {device: estimate.gpu_on_cpu_bytes for device in gpu_free},
                ram_bytes=estimate.ram_on_cpu_bytes, estimate=estimate), None))
        for placement, resident in options:
            incremental = resident is None or resident.instance_id in credited
            ram = placement.ram_bytes if incremental else 0
            gpu = placement.gpu_allocations if incremental else {}
            if ram <= ram_free and all(amount <= gpu_free.get(device, 0) for device, amount in gpu.items()):
                remaining_gpu = gpu_free.copy()
                for device, amount in gpu.items():
                    remaining_gpu[device] -= amount
                # A locally fitting allocation can strand the next model,
                # even with CPU offload (which may retain GPU overhead).
                # Try the next offload/device only if the rest cannot fit.
                result = search(index + 1, ram_free - ram, remaining_gpu)
                if result is not None:
                    placements, reused = result
                    return ({estimate.model: placement, **placements},
                            ({estimate.model: resident.instance_id, **reused} if resident else reused))
        return None

    result = search(0, ram_free, gpu_free)
    if result is None:
        raise ResourceAdmissionError("Infeasible concurrent placement for " +
            ", ".join(estimate.model for estimate in ordered) +
            ": insufficient GPU/RAM after safety margins or unsupported device control")
    return result


def _exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


class ResourceScheduler:
    def __init__(self, backend: PlacementBackend, hardware=discover_hardware):
        self.backend, self.hardware = backend, hardware
        self.residents = {}
        self.instance_jobs = {}
        self.accounting_lock = RLock()
        self.change_lock = RLock()  # Only estimates, admission and load/unload.
        self.capabilities = backend.capabilities()

    def unload(self, identifier):
        with self.change_lock:
            with self.accounting_lock:
                resident = self.residents[identifier]
                if resident.active or resident.pins or not resident.managed:
                    raise ResourceAdmissionError(f"Cannot unload active, reserved or externally managed instance: {identifier}")
                del self.residents[identifier]
                jobs = self.instance_jobs.pop(identifier, set())
            try:
                log_event("resources", "unload_started", instance=identifier, model=resident.placement.model)
                elapsed = self.backend.unload(identifier)
                log_event("resources", "unloaded", instance=identifier, model=resident.placement.model, elapsed_seconds=elapsed)
                return elapsed
            except Exception as exc:
                log_event("resources", "unload_failed", level=logging.ERROR, instance=identifier,
                          error=f"{type(exc).__name__}: {exc}")
                with self.accounting_lock:
                    self.residents[identifier] = resident
                    self.instance_jobs[identifier] = jobs
                raise

    def _adopt_declared_instances(self, settings):
        if not settings.resident_devices:
            return
        hardware = self.hardware()
        known_devices = {gpu.device_id for gpu in hardware.gpus} | {"cpu"}
        for item in self.backend.inventory():
            for instance in item.get("loaded_instances", []):
                identifier = instance["id"]
                if identifier not in settings.resident_devices or identifier in self.residents:
                    continue
                device = settings.resident_devices[identifier]
                if device not in known_devices:
                    raise ResourceAdmissionError(f"Declared residency uses unknown device: {device}")
                config = instance["config"]
                actual_settings = settings.model_copy(update={"context_length": config["context_length"]})
                estimate = self.backend.estimate(item["key"], actual_settings, config.get("parallel", 1))
                gpu = {} if device == "cpu" else {device: estimate.gpu_bytes}
                if device == "cpu" and estimate.gpu_on_cpu_bytes:
                    if not hardware.gpus:
                        raise ResourceAdmissionError("CPU instance has GPU overhead but GPU inventory is unavailable")
                    gpu = {gpu.device_id: estimate.gpu_on_cpu_bytes for gpu in hardware.gpus}
                placement = Placement(model=item["key"], device=device, gpu_offload="off" if device == "cpu" else "max",
                    gpu_allocations=gpu, ram_bytes=estimate.ram_on_cpu_bytes if device == "cpu" else estimate.ram_on_gpu_bytes, estimate=estimate)
                with self.accounting_lock:
                    self.residents[identifier] = Resident(instance_id=identifier, placement=placement)

    def prepare(self, role_models, all_models, settings, run_dir, stage, *, artifact_dir=None):
        started = time.perf_counter()
        context = {"run_dir": str(run_dir)}
        log_event(f"resources.{stage}", "admission_started", state=context, requested_models=role_models)
        audit = {"stage": stage, "requested_models": role_models, "settings": settings.model_dump(),
                 "capabilities": self.capabilities.model_dump(), "loading_seconds": 0.0, "unloading_seconds": 0.0}
        try:
            with self.change_lock:
                self._adopt_declared_instances(settings)
                loaded_ids = {instance["id"] for item in self.backend.inventory() for instance in item.get("loaded_instances", [])}
                with self.accounting_lock:
                    for identifier, resident in list(self.residents.items()):
                        if identifier not in loaded_ids:
                            if resident.active or resident.pins:
                                raise ResourceAdmissionError(f"Backend removed an active/reserved instance: {identifier}")
                            del self.residents[identifier]
                canonical = {role: self.backend.canonical_model(model) for role, model in role_models.items()}
                counts = Counter(canonical.values())
                if len(role_models) > 1 and not self.capabilities.concurrent_models:
                    raise ResourceAdmissionError("Backend concurrency unavailable; refusing to serialize independent agents")
                slots = max(settings.parallel_slots, max(counts.values(), default=1))
                if slots > 1 and not self.capabilities.parallel_slots_configurable:
                    raise ResourceAdmissionError("Backend cannot guarantee the requested concurrent prediction slots")
                estimates = {model: self.backend.estimate(model, settings, slots) for model in set(canonical.values())}
                ready_estimates = dict(estimates)
                observed = self.hardware()
                audit["hardware_before"] = observed.model_dump()
                log_event(f"resources.{stage}", "hardware", state=context,
                          ram_free_gib=observed.ram_free_bytes / (1024 ** 3),
                          gpu_free_gib={gpu.device_id: round(gpu.free_bytes / (1024 ** 3), 3) for gpu in observed.gpus})
                with self.accounting_lock:
                    snapshot = deepcopy(list(self.residents.values()))
                # Eligible owned idle models may be reclaimed. Credits are
                # provisional only; fresh physical budgets are rechecked after
                # eviction and before each load, never trusted for admission.
                # Reclaim only idle instances owned exclusively by this job.
                # A shared instance remains protected even between another
                # job's stages when it has no active lease or pin.
                eligible = [resident for resident in snapshot if not resident.active and not resident.pins and resident.managed
                            and not (self.instance_jobs.get(resident.instance_id, set()) - {str(run_dir)})]
                candidates = [resident for resident in snapshot
                              if resident.placement.estimate.parallel_slots - resident.pins >= counts.get(resident.placement.model, 0)]
                # External busy predictions have no observable active count;
                # never count their remaining slots as available.
                candidates = [r for r in candidates if r.managed or not self.backend.is_active(r.instance_id)]
                all_estimates = dict(estimates)
                if settings.prefetch_all_if_gpu_fit:
                    for model in all_models:
                        key = self.backend.canonical_model(model)
                        all_estimates.setdefault(key, self.backend.estimate(key, settings, slots))
                audit.update(estimates={key: value.model_dump() for key, value in all_estimates.items()},
                             residency_before=[resident.model_dump() for resident in snapshot])
                def evaluate(inventory, credits, available_candidates=candidates):
                    if settings.prefetch_all_if_gpu_fit:
                        try:
                            full, reuse = select_placements(all_estimates.values(), inventory, self.capabilities, settings, available_candidates, credits)
                            if all(p.gpu_offload == "max" for p in full.values()):
                                return full, reuse, True
                        except ResourceAdmissionError:
                            pass
                    plan, reuse = select_placements(ready_estimates.values(), inventory, self.capabilities, settings, available_candidates, credits)
                    return plan, reuse, False
                ready_models = set(canonical.values())
                def offload_fraction(placement):
                    if placement.device == "cpu":
                        return 0.0
                    return 1.0 if placement.gpu_offload == "max" else float(placement.gpu_offload)
                score = lambda result: sum(all_estimates[key].cpu_cost * offload_fraction(placement)
                    for key, placement in result[0].items() if key in ready_models)
                plans = []
                for count in range(len(eligible) + 1):
                    for subset in combinations(eligible, count):
                        inventory = observed.model_copy(deep=True)
                        inventory.ram_free_bytes = min(inventory.ram_total_bytes,
                            inventory.ram_free_bytes + sum(r.placement.ram_bytes for r in subset))
                        for gpu in inventory.gpus:
                            gpu.free_bytes = min(gpu.total_bytes, gpu.free_bytes + sum(
                                r.placement.gpu_allocations.get(gpu.device_id, 0) for r in subset))
                        try:
                            result = evaluate(inventory, {r.instance_id for r in subset})
                            plans.append((result, subset, score(result)))
                        except ResourceAdmissionError:
                            continue
                if not plans:
                    raise ResourceAdmissionError("No feasible placement after evaluating protected idle residency")
                # Ready-stage acceleration is primary. Then retain useful full
                # prefetch, minimize evictions, and minimize potential reload
                # bytes. Stable instance IDs make the final tie deterministic.
                chosen = max(plans, key=lambda item: (item[2], int(item[0][2]), -len(item[1]),
                    -sum(r.placement.estimate.weight_bytes for r in item[1]),
                    tuple(sorted(r.instance_id for r in item[1]))))
                (placements, reused, prefetch), reclaimed_residents, reclaimed_score = chosen
                ordinary_plan = next((item for item in plans if not item[1]), None)
                ordinary_score = None if ordinary_plan is None else ordinary_plan[2]
                reclaim = bool(reclaimed_residents)
                planned_placements = placements
                # Reclaim only owned unpinned instances that this plan does not
                # reuse. Unmanaged/manual instances are never auto-unloaded.
                unloaded = []
                for resident in reclaimed_residents:
                    duplicate = resident.placement.model in placements and resident.placement.model not in reused
                    if resident.instance_id not in reused.values() and (reclaim or duplicate):
                        audit["unloading_seconds"] += self.unload(resident.instance_id)
                        unloaded.append(resident.instance_id)
                # Estimated allocations are admission bounds, not measurements
                # of memory the backend actually committed. Re-discover after
                # reclamation and plan again from physical free memory.
                if unloaded:
                    post_reclaim = self.hardware()
                    with self.accounting_lock:
                        live_snapshot = deepcopy(list(self.residents.values()))
                    post_candidates = [resident for resident in live_snapshot
                        if resident.placement.estimate.parallel_slots - resident.pins >= counts.get(resident.placement.model, 0)]
                    post_candidates = [resident for resident in post_candidates
                        if resident.managed or not self.backend.is_active(resident.instance_id)]
                    try:
                        placements, reused, prefetch = evaluate(post_reclaim, (), post_candidates)
                    except ResourceAdmissionError as exc:
                        raise ResourceAdmissionError(
                            "No feasible placement using physical resources measured after idle-instance reclamation") from exc
                    audit["hardware_after_reclaim"] = post_reclaim.model_dump()
                estimates = all_estimates if prefetch else ready_estimates
                final_score = score((placements, reused, prefetch))
                if len(role_models) > 1 and not self.capabilities.cpu_gpu_concurrency and len({placements[key].device == "cpu" for key in canonical.values()}) > 1:
                    raise ResourceAdmissionError("Backend cannot concurrently execute CPU/GPU predictions")
                audit.update(prefetched_all_on_gpu=prefetch, estimates={key: value.model_dump() for key, value in estimates.items()},
                             planned_placements={key: value.model_dump() for key, value in planned_placements.items()},
                             placements={key: value.model_dump() for key, value in placements.items()}, reused_instances=reused,
                             residency_before=[resident.model_dump() for resident in snapshot],
                             placement_scores={"without_reclaim": ordinary_score, "hypothetical_after_reclaim": reclaimed_score,
                                               "physical_after_reclaim": final_score},
                             post_reclaim_adjusted=placements != planned_placements,
                             reclaimed_idle_for_ready_acceleration=reclaim and
                                 (ordinary_score is None or final_score > ordinary_score),
                             reclaimed_instances=unloaded)
                resolved = {}
                for model, placement in placements.items():
                    if model in reused:
                        resolved[model] = reused[model]
                        log_event(f"resources.{stage}", "reused", state=context, model=model, instance=reused[model],
                                  device=placement.device, gpu_offload=placement.gpu_offload)
                        continue
                    fresh = self.hardware()
                    if placement.ram_bytes > max(0, fresh.ram_free_bytes - settings.ram_safety_bytes) or any(
                        amount > max(0, next((gpu.free_bytes for gpu in fresh.gpus if gpu.device_id == device), 0) - settings.gpu_safety_bytes)
                        for device, amount in placement.gpu_allocations.items()):
                        raise ResourceAdmissionError("Physical resources changed or reclamation was smaller than estimated; placement rejected before load")
                    identifier = "agentic-web-" + uuid4().hex
                    log_event(f"resources.{stage}", "load_started", state=context, model=model, instance=identifier,
                              device=placement.device, gpu_offload=placement.gpu_offload,
                              estimated_ram_gib=placement.ram_bytes / (1024 ** 3),
                              estimated_gpu_gib=sum(placement.gpu_allocations.values()) / (1024 ** 3))
                    identifier, elapsed = self.backend.load(placement, identifier)
                    log_event(f"resources.{stage}", "loaded", state=context, model=model, instance=identifier,
                              device=placement.device, elapsed_seconds=elapsed)
                    audit["loading_seconds"] += elapsed
                    with self.accounting_lock:
                        self.residents[identifier] = Resident(instance_id=identifier, placement=placement, managed=True)
                        self.instance_jobs.setdefault(identifier, set()).add(str(run_dir))
                    resolved[model] = identifier
                with self.accounting_lock:
                    for identifier in resolved.values():
                        self.instance_jobs.setdefault(identifier, set()).add(str(run_dir))
                loaded_ids = {instance["id"] for item in self.backend.inventory() for instance in item.get("loaded_instances", [])}
                if set(resolved.values()) - loaded_ids:
                    raise ResourceAdmissionError("Backend evicted a required instance while preparing residency; disable automatic eviction for parallel runs")
                audit["hardware_after"] = self.hardware().model_dump()
                bindings = {}
                with self.accounting_lock:
                    for role, model in canonical.items():
                        resident = self.residents[resolved[model]]
                        resident.pins += 1
                        bindings[role] = {"instance_id": resident.instance_id, "requested_model": role_models[role],
                            "placement": resident.placement.model_dump(), "reused": model in reused,
                            "stage_loading_seconds": audit["loading_seconds"], "stage_unloading_seconds": audit["unloading_seconds"],
                            "stage_admission_seconds": time.perf_counter() - started}
                audit["status"] = "prepared"
                audit["bindings"] = bindings
                log_event(f"resources.{stage}", "prepared", state=context, elapsed_seconds=time.perf_counter() - started,
                          loading_seconds=audit["loading_seconds"], unloading_seconds=audit["unloading_seconds"],
                          reclaimed_instances=unloaded,
                          devices={role: binding["placement"]["device"] for role, binding in bindings.items()})
                return bindings
        except Exception as exc:
            audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            log_event(f"resources.{stage}", "failed", state=context, level=logging.ERROR,
                      elapsed_seconds=time.perf_counter() - started, error=audit["error"])
            raise
        finally:
            audit["admission_seconds"] = time.perf_counter() - started
            _exclusive_json(Path(artifact_dir or run_dir) / f"resources.{stage}.json", audit)


class ResourceSession:
    def __init__(self, scheduler, settings, config, run_dir, roles=("planner", "ux", "architect", "coordinator", "implementer", "qa")):
        self.scheduler, self.settings, self.config, self.run_dir = scheduler, settings, config, run_dir
        self.bindings = {}
        self.finished = set()
        self.active_roles = set()
        self.all_models = sorted({config.model_for(role) for role in roles})

    def prepare(self, stage, roles, *, artifact_dir=None):
        bindings = self.scheduler.prepare({role: self.config.model_for(role) for role in roles}, self.all_models,
                                         self.settings, self.run_dir, stage,
                                         **({'artifact_dir': artifact_dir} if artifact_dir else {}))
        self.bindings.update(bindings)

    @contextmanager
    def inference(self, role, *, artifact_dir=None):
        if artifact_dir:
            # A feedback round acquires a fresh lease using the existing scheduler.
            # Ownership remains this candidate; only its audit directory changes.
            if role in self.active_roles:
                raise ResourceAdmissionError(f'Inference lease still active: {role}')
            self.prepare(role, [role], artifact_dir=artifact_dir)
            self.finished.discard(role)
        if role not in self.bindings:
            self.prepare(role, [role])
        started = time.perf_counter()
        binding = self.bindings[role]
        with self.scheduler.accounting_lock:
            if role in self.finished:
                raise ResourceAdmissionError(f"Inference lease already consumed: {role}")
            resident = self.scheduler.residents[binding["instance_id"]]
            resident.active += 1
            self.active_roles.add(role)
        queueing = time.perf_counter() - started
        metrics = {"requested_model": binding["requested_model"], "assigned_instance": binding["instance_id"],
                   "assigned_device": binding["placement"]["device"], "gpu_offload": binding["placement"]["gpu_offload"],
                   "residency_reused": binding["reused"], "queueing_seconds": queueing,
                   "stage_admission_seconds": binding["stage_admission_seconds"],
                   "stage_loading_seconds": binding["stage_loading_seconds"], "stage_unloading_seconds": binding["stage_unloading_seconds"],
                   "peak_memory_bytes": None}
        event = {"role": role, **metrics}
        try:
            before = self.scheduler.hardware()
            yield binding["instance_id"], metrics
            event["status"] = "completed"
        except Exception as exc:
            event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            event.update(metrics)
            event["lease_wall_seconds"] = time.perf_counter() - started - queueing
            try:
                after = self.scheduler.hardware()
                event["hardware_after_inference"] = after.model_dump()
                if "before" in locals():
                    event["hardware_before_inference"] = before.model_dump()
                    event["sampled_device_used_peak_bytes"] = {gpu.device_id: max(gpu.total_bytes - gpu.free_bytes,
                        next((item.total_bytes - item.free_bytes for item in after.gpus if item.device_id == gpu.device_id), 0)) for gpu in before.gpus}
            except Exception as exc:
                event["telemetry_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                with self.scheduler.accounting_lock:
                    resident.active -= 1
                    resident.pins -= 1
                    self.active_roles.remove(role)
                    self.finished.add(role)
                _exclusive_json(Path(artifact_dir or self.run_dir) / f"resources.inference.{role}.json", event)

    def close(self):
        if getattr(self, "closed", False):
            return
        audit = {"unloaded_instances": [], "preserved_instances": [], "errors": [], "unloading_seconds": 0.0}
        with self.scheduler.change_lock:
            with self.scheduler.accounting_lock:
                pending = [(role, self.scheduler.residents[binding["instance_id"]])
                           for role, binding in self.bindings.items() if role not in self.finished]
                if self.active_roles:
                    raise ResourceAdmissionError("Cannot close a resource session with active inference")
                for role, resident in pending:
                    resident.pins -= 1
                    self.finished.add(role)
                for jobs in self.scheduler.instance_jobs.values():
                    jobs.discard(str(self.run_dir))
                candidates = list(self.scheduler.residents.values())
            for resident in candidates:
                identifier = resident.instance_id
                if not resident.managed or resident.active or resident.pins or self.scheduler.instance_jobs.get(identifier):
                    audit["preserved_instances"].append(identifier)
                    continue
                try:
                    audit["unloading_seconds"] += self.scheduler.unload(identifier)
                    self.scheduler.instance_jobs.pop(identifier, None)
                    audit["unloaded_instances"].append(identifier)
                except Exception as exc:
                    audit["errors"].append({"instance_id": identifier, "error": f"{type(exc).__name__}: {exc}"})
            self.closed = True
        audit["status"] = "failed" if audit["errors"] else "completed"
        _exclusive_json(Path(self.run_dir) / "resources.cleanup.json", audit)


_registry_lock = RLock()
_registry = {}


def local_scheduler(base_url, executable):
    from .lmstudio_placement import LMStudioPlacementBackend
    key = (base_url.rstrip("/"), executable)
    with _registry_lock:
        if key not in _registry:
            _registry[key] = ResourceScheduler(LMStudioPlacementBackend(base_url, executable))
        return _registry[key]
