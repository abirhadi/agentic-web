from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from agentic_web.client import ModelConfig
from agentic_web.resource_schema import GiB, ModelEstimate, Placement, Resident, ResourceSettings, ResourceAdmissionError
from agentic_web.resource_scheduler import ResourceScheduler, ResourceSession, select_placements
from agentic_web.lmstudio_placement import LMStudioPlacementBackend, parse_memory_estimate
from agentic_web.workflow import build_graph
from collaboration_mock_client import CollaborationMockClient
from test_framework import stub_verifier
from mock_client import ROOT
from resource_mock_backend import SimulatedBackend


def archived_specialist_estimates():
    # Memory bounds from the failed 20261012t013017 run. In particular, CPU
    # offload still uses GPU memory, unlike the original simulated backend.
    values = {
        "ux": (9366249932, 5718801405, 1128657847, 262144000000000,
               [(7429219681, 5718801405), (5492189431, 5718801405), (3307124819, 6059125113)]),
        "architect": (5881957713, 2741087296, 848650306, 98304000000000,
                      [(4783519827, 2741087296), (3519725700, 2741087296), (2102386492, 3779571221)]),
        "planner": (6378026436, 4657156422, 1405528049, 131072000000000, []),
    }
    return {name: ModelEstimate(model=name, context_length=32768, parallel_slots=1,
        weight_bytes=ram, gpu_bytes=gpu, ram_on_gpu_bytes=ram, ram_on_cpu_bytes=gpu,
        gpu_on_cpu_bytes=cpu_gpu, kv_and_overhead_bytes=0, cpu_cost=cost, source="archived admission bounds",
        offload_profiles=[dict(gpu_offload=str(fraction), fraction=fraction, gpu_bytes=g, ram_bytes=r)
                          for fraction, (g, r) in zip((0.75, 0.5, 0.25), profiles)])
        for name, (gpu, ram, cpu_gpu, cost, profiles) in values.items()}


class ResourcePolicyTests(unittest.TestCase):
    def settings(self):
        return ResourceSettings(context_length=4096, gpu_safety_bytes=0, ram_safety_bytes=0, estimate_padding=0,
                                prefetch_all_if_gpu_fit=False)

    def plan(self, backend, models=("heavy", "light")):
        settings = self.settings()
        estimates = [backend.estimate(model, settings, 2) for model in models]
        return select_placements(estimates, backend.hardware(), backend.capabilities(), settings)[0]

    def test_single_gpu_all_models_fit(self):
        placements = self.plan(SimulatedBackend(gpu_sizes=(16 * GiB,)))
        self.assertEqual({placement.device for placement in placements.values()}, {"0"})

    def test_single_gpu_heavier_workload_gets_gpu_and_light_gets_cpu(self):
        placements = self.plan(SimulatedBackend(gpu_sizes=(9 * GiB,)))
        self.assertEqual(placements["heavy"].device, "0")
        self.assertEqual(placements["light"].device, "cpu")

    def test_multiple_gpus_distribute_models(self):
        placements = self.plan(SimulatedBackend(gpu_sizes=(9 * GiB, 5 * GiB)))
        self.assertEqual(placements["heavy"].device, "0")
        self.assertEqual(placements["light"].device, "1")

    def test_backend_without_device_selection_does_not_invent_gpu_controls(self):
        placements = self.plan(SimulatedBackend(gpu_sizes=(9 * GiB, 5 * GiB), explicit_devices=False))
        self.assertEqual({placement.device for placement in placements.values()}, {"cpu"})

    def test_unknown_cpu_gpu_overhead_is_reserved_on_every_gpu(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB, 5 * GiB), explicit_devices=False)
        settings = self.settings()
        estimate = backend.estimate("light", settings, 2).model_copy(update={"gpu_on_cpu_bytes": GiB})
        placements, _ = select_placements([estimate], backend.hardware(), backend.capabilities(), settings)
        self.assertEqual(placements["light"].gpu_allocations, {"0": GiB, "1": GiB})
        hardware = backend.hardware()
        hardware.gpus[1].free_bytes = GiB // 2
        with self.assertRaises(ResourceAdmissionError):
            select_placements([estimate], hardware, backend.capabilities(), settings)

    def test_insufficient_gpu_and_ram_rejected(self):
        with self.assertRaises(ResourceAdmissionError):
            self.plan(SimulatedBackend(gpu_sizes=(GiB,), ram=2 * GiB))

    def test_archived_specialists_backtrack_to_fit_entire_stage(self):
        estimates = archived_specialist_estimates()
        backend = SimulatedBackend(gpu_sizes=(12820938752,), explicit_devices=False)
        settings = ResourceSettings(gpu_safety_bytes=1610612736, ram_safety_bytes=2147483648)
        hardware = backend.hardware()
        for free, expected in [(5197791232, ("off", "0.25")),
                               (5197791232 + 6378026436, ("0.75", "0.25"))]:
            with self.subTest(free=free):
                hardware.gpus[0].free_bytes = free
                placements, reused = select_placements(
                    [estimates["architect"], estimates["ux"]], hardware, backend.capabilities(), settings)
                self.assertEqual((placements["ux"].gpu_offload, placements["architect"].gpu_offload), expected)
                self.assertEqual(reused, {})
                self.assertLessEqual(sum(p.gpu_allocations.get("0", 0) for p in placements.values()),
                                     free - settings.gpu_safety_bytes)
        hardware.gpus[0].free_bytes = settings.gpu_safety_bytes + 1128657847 + 848650306 - 1
        with self.assertRaises(ResourceAdmissionError):
            select_placements([estimates["ux"], estimates["architect"]], hardware, backend.capabilities(), settings)

    def test_safety_margins_enforced(self):
        backend = SimulatedBackend(gpu_sizes=(16 * GiB,), ram=64 * GiB)
        settings = self.settings().model_copy(update={"gpu_safety_bytes": 8 * GiB})
        placements, _ = select_placements([backend.estimate("heavy", settings, 2)], backend.hardware(), backend.capabilities(), settings)
        self.assertEqual(placements["heavy"].device, "0")
        self.assertNotEqual(placements["heavy"].gpu_offload, "max")
        self.assertLessEqual(placements["heavy"].gpu_allocations["0"], 8 * GiB)

    def test_context_and_parallel_slots_change_estimated_memory(self):
        backend = SimulatedBackend()
        short = backend.estimate("heavy", self.settings(), 1)
        long = backend.estimate("heavy", self.settings().model_copy(update={"context_length": 16384}), 2)
        self.assertGreater(long.gpu_bytes, short.gpu_bytes)
        self.assertGreater(long.kv_and_overhead_bytes, short.kv_and_overhead_bytes)


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = ResourceSettings(context_length=4096, gpu_safety_bytes=0, ram_safety_bytes=0,
            estimate_padding=0, prefetch_all_if_gpu_fit=False)
        self.config = ModelConfig(model="light", planner_model="tiny", ux_model="light", architect_model="heavy",
                                  coordinator_model="tiny", qa_model="tiny")

    def session(self, backend, prefetch=False):
        scheduler = ResourceScheduler(backend, backend.hardware)
        settings = self.settings.model_copy(update={"prefetch_all_if_gpu_fit": prefetch})
        return ResourceSession(scheduler, settings, self.config, self.temp.name)

    def test_archived_admission_reclaims_planner_and_runs_both_specialists(self):
        estimates = archived_specialist_estimates()
        backend = SimulatedBackend(gpu_sizes=(12820938752,), ram=68477820928,
            base_gpu_used=12820938752 - 5197791232 - 6378026436,
            explicit_devices=False, weights={key: value.weight_bytes for key, value in estimates.items()})
        settings = ResourceSettings(parallel_slots=1, gpu_safety_bytes=1610612736,
                                    ram_safety_bytes=2147483648, prefetch_all_if_gpu_fit=True)
        config = self.config.model_copy(update={"planner_model": "planner", "ux_model": "ux",
                                               "architect_model": "architect"})
        session = ResourceSession(ResourceScheduler(backend, backend.hardware), settings, config,
                                  self.temp.name, roles=("planner", "ux", "architect"))
        with patch.object(backend, "estimate", side_effect=lambda model, settings, slots: estimates[model]):
            with session.inference("planner") as (planner_id, _):
                pass
            session.prepare("specialists", ["ux", "architect"])
            self.assertIn(planner_id, backend.unloads)
            with session.inference("ux") as (_, ux), session.inference("architect") as (_, architect):
                self.assertEqual(ux["gpu_offload"], "0.75")
                self.assertEqual(architect["gpu_offload"], "0.25")
                self.assertEqual(sum(r.active for r in session.scheduler.residents.values()), 2)
            session.close()
        self.assertEqual(backend.instances, {})

    def test_gpu_cpu_inferences_overlap_and_cannot_unload_active_models(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB,))
        session = self.session(backend)
        session.prepare("specialists", ["ux", "architect"])
        barrier = threading.Barrier(2)
        def worker(role):
            with session.inference(role) as (identifier, metrics):
                backend.active.add(identifier)
                try:
                    barrier.wait(timeout=3)
                    self.assertEqual(sum(r.active for r in session.scheduler.residents.values()), 2)
                    with self.assertRaises(ResourceAdmissionError):
                        session.scheduler.unload(identifier)
                    barrier.wait(timeout=3)
                    return metrics["assigned_device"]
                finally:
                    backend.active.remove(identifier)
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(worker, ("ux", "architect")))
        self.assertEqual(set(results), {"cpu", "0"})
        session.close()

    def test_residency_reused_across_roles_and_stages(self):
        backend = SimulatedBackend()
        session = self.session(backend)
        with session.inference("planner"):
            pass
        with session.inference("coordinator") as (_, metrics):
            self.assertTrue(metrics["residency_reused"])
        self.assertEqual(len(backend.loads), 1)
        self.assertEqual(backend.unloads, [])
        session.close()

    def test_ready_heavy_model_reclaims_idle_gpu_and_uses_largest_safe_offload(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB,))
        settings = self.settings.model_copy(update={"gpu_safety_bytes": GiB})
        config = self.config.model_copy(update={"planner_model": "light", "model": "heavy"})
        session = ResourceSession(ResourceScheduler(backend, backend.hardware), settings, config, self.temp.name,
                                  roles=("planner", "implementer"))
        with session.inference("planner") as (_, metrics):
            self.assertEqual(metrics["assigned_device"], "0")
            self.assertEqual(metrics["gpu_offload"], "max")
        old_identifier = backend.loads[0][0]
        with session.inference("implementer") as (_, metrics):
            self.assertEqual(metrics["assigned_device"], "0")
            self.assertEqual(metrics["gpu_offload"], "0.75")
            self.assertIn(old_identifier, backend.unloads)
        decision = json.loads((Path(self.temp.name) / "resources.implementer.json").read_text())
        self.assertTrue(decision["reclaimed_idle_for_ready_acceleration"])
        self.assertGreater(decision["placement_scores"]["physical_after_reclaim"],
                           decision["placement_scores"]["without_reclaim"])
        session.close()

    def test_post_reclaim_replans_from_physical_memory_instead_of_estimated_credit(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB,), base_gpu_used=2 * GiB, actual_gpu_factor=0.5)
        settings = self.settings.model_copy(update={"gpu_safety_bytes": GiB})
        config = self.config.model_copy(update={"planner_model": "light", "model": "heavy"})
        session = ResourceSession(ResourceScheduler(backend, backend.hardware), settings, config, self.temp.name,
                                  roles=("planner", "implementer"))
        with session.inference("planner"):
            pass
        with session.inference("implementer") as (_, metrics):
            # Estimated reclamation suggested 75%; the physical post-unload
            # snapshot safely supports 50%, so the stage continues at 50%.
            self.assertEqual(metrics["gpu_offload"], "0.5")
        decision = json.loads((Path(self.temp.name) / "resources.implementer.json").read_text())
        self.assertTrue(decision["post_reclaim_adjusted"])
        self.assertIn("hardware_after_reclaim", decision)
        self.assertEqual(decision["planned_placements"]["heavy"]["gpu_offload"], "0.75")
        self.assertEqual(decision["placements"]["heavy"]["gpu_offload"], "0.5")
        session.close()

    def test_reclaim_evicts_only_instances_needed_for_better_ready_offload(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB,))
        settings = self.settings.model_copy(update={"gpu_safety_bytes": GiB})
        config = self.config.model_copy(update={"planner_model": "light", "model": "heavy"})
        scheduler = ResourceScheduler(backend, backend.hardware)
        session = ResourceSession(scheduler, settings, config, self.temp.name, roles=("planner", "implementer"))
        with session.inference("planner"):
            pass
        estimate = backend.estimate("tiny", settings, 1)
        cpu = Placement(model="tiny", device="cpu", gpu_offload="off", gpu_allocations={},
                        ram_bytes=estimate.ram_on_cpu_bytes, estimate=estimate)
        backend.load(cpu, "idle-cpu")
        scheduler.residents["idle-cpu"] = Resident(instance_id="idle-cpu", placement=cpu, managed=True)
        scheduler.instance_jobs["idle-cpu"] = {str(self.temp.name)}
        gpu_idle = session.bindings["planner"]["instance_id"]
        with session.inference("implementer"):
            self.assertIn(gpu_idle, backend.unloads)
            self.assertNotIn("idle-cpu", backend.unloads)
            self.assertIn("idle-cpu", backend.instances)
        decision = json.loads((Path(self.temp.name) / "resources.implementer.json").read_text())
        self.assertEqual(decision["reclaimed_instances"], [gpu_idle])
        session.close()

    def test_idle_instance_shared_with_other_job_is_not_reclaimed(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB,))
        scheduler = ResourceScheduler(backend, backend.hardware)
        first_dir, second_dir = (Path(self.temp.name) / "owner", Path(self.temp.name) / "shared")
        first_dir.mkdir()
        second_dir.mkdir()
        config = self.config.model_copy(update={"planner_model": "light", "model": "heavy"})
        first = ResourceSession(scheduler, self.settings, config, first_dir, roles=("planner", "implementer"))
        second = ResourceSession(scheduler, self.settings, config, second_dir, roles=("planner",))
        with first.inference("planner"):
            pass
        second.prepare("planner", ["planner"])
        shared = second.bindings["planner"]["instance_id"]
        with first.inference("implementer"):
            pass
        self.assertNotIn(shared, backend.unloads)
        second.close()
        first.close()

    def test_completion_unloads_owned_prefetch_and_preserves_external_instance(self):
        backend = SimulatedBackend(gpu_sizes=(80 * GiB,), ram=256 * GiB)
        placement = self.plan_external(backend)
        backend.load(placement, "manual-instance")
        session = self.session(backend, prefetch=True)
        with session.inference("planner"):
            pass
        owned = set(backend.instances) - {"manual-instance"}
        session.close()
        session.close()  # Cleanup is idempotent.
        self.assertEqual(set(backend.instances), {"manual-instance"})
        report = json.loads((Path(self.temp.name) / "resources.cleanup.json").read_text())
        self.assertEqual(set(report["unloaded_instances"]), owned)
        self.assertEqual(report["status"], "completed")

    def plan_external(self, backend):
        return select_placements([backend.estimate("tiny", self.settings, 2)], backend.hardware(),
                                 backend.capabilities(), self.settings)[0]["tiny"]

    def test_shared_instance_survives_until_last_job_finishes(self):
        backend = SimulatedBackend()
        scheduler = ResourceScheduler(backend, backend.hardware)
        directories = [Path(self.temp.name) / name for name in ("first", "second")]
        for directory in directories:
            directory.mkdir()
        first, second = [ResourceSession(scheduler, self.settings, self.config, directory) for directory in directories]
        with first.inference("planner"):
            pass
        second.prepare("planner", ["planner"])
        identifier = second.bindings["planner"]["instance_id"]
        first.close()
        self.assertIn(identifier, backend.instances)
        with second.inference("planner"):
            pass
        second.close()
        self.assertEqual(backend.instances, {})
        self.assertEqual(backend.unloads, [identifier])

    def test_failed_job_cleanup_records_unload_error_and_keeps_instance_tracked(self):
        backend = SimulatedBackend()
        session = self.session(backend)
        with self.assertRaisesRegex(ValueError, "inference failed"):
            with session.inference("planner"):
                raise ValueError("inference failed")
        with patch.object(backend, "unload", side_effect=RuntimeError("backend unavailable")):
            session.close()
        report = json.loads((Path(self.temp.name) / "resources.cleanup.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("backend unavailable", report["errors"][0]["error"])
        self.assertEqual(len(session.scheduler.residents), 1)

    def test_closing_unused_reservation_does_not_interrupt_another_job(self):
        backend = SimulatedBackend()
        scheduler = ResourceScheduler(backend, backend.hardware)
        directories = [Path(self.temp.name) / name for name in ("active", "cancelled")]
        for directory in directories:
            directory.mkdir()
        first, second = [ResourceSession(scheduler, self.settings, self.config, directory) for directory in directories]
        with first.inference("planner") as (identifier, _):
            second.prepare("planner", ["planner"])
            second.close()
            self.assertIn(identifier, backend.instances)
            self.assertEqual(scheduler.residents[identifier].active, 1)
        first.close()
        self.assertEqual(backend.instances, {})

    def test_hpc_prefetch_keeps_every_required_model_gpu_resident(self):
        backend = SimulatedBackend(gpu_sizes=(80 * GiB, 80 * GiB), ram=256 * GiB)
        session = self.session(backend, prefetch=True)
        with session.inference("planner"):
            pass
        self.assertEqual(len(backend.loads), 3)
        session.prepare("specialists", ["ux", "architect"])
        for role in ("ux", "architect", "coordinator", "implementer", "qa"):
            with session.inference(role) as (_, metrics):
                self.assertNotEqual(metrics["assigned_device"], "cpu")
                self.assertTrue(metrics["residency_reused"])
        self.assertEqual(len(backend.loads), 3)
        self.assertEqual(backend.unloads, [])
        session.close()

    def test_inactive_cache_can_be_evicted_but_reserved_models_cannot(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB,))
        session = self.session(backend)
        with session.inference("planner"):
            pass
        session.prepare("specialists", ["ux", "architect"])
        self.assertEqual(len(backend.unloads), 1)
        for binding in session.bindings.values():
            if binding["instance_id"] in session.scheduler.residents and binding["instance_id"] not in backend.unloads:
                with self.assertRaises(ResourceAdmissionError):
                    session.scheduler.unload(binding["instance_id"])
        session.close()

    def test_shared_model_has_two_slots_and_is_loaded_once(self):
        backend = SimulatedBackend()
        config = self.config.model_copy(update={"ux_model": "heavy"})
        session = ResourceSession(ResourceScheduler(backend, backend.hardware), self.settings, config, self.temp.name)
        session.prepare("specialists", ["ux", "architect"])
        self.assertEqual(len(backend.loads), 1)
        self.assertEqual(session.bindings["ux"]["instance_id"], session.bindings["architect"]["instance_id"])
        self.assertGreaterEqual(backend.loads[0][1].estimate.parallel_slots, 2)
        session.close()

    def test_concurrency_unsupported_is_rejected_not_serialized(self):
        backend = SimulatedBackend(concurrent=False)
        session = self.session(backend)
        with self.assertRaisesRegex(ResourceAdmissionError, "refusing to serialize"):
            session.prepare("specialists", ["ux", "architect"])
        self.assertEqual(backend.loads, [])

    def test_parallel_scheduled_graph_preserves_contracts_join_and_ledger(self):
        backend = SimulatedBackend(gpu_sizes=(9 * GiB,))
        scheduler = ResourceScheduler(backend, backend.hardware)
        barrier = threading.Barrier(2)
        class ConcurrentClient(CollaborationMockClient):
            def complete(self, system, user, config, *, planner=False, response_schema=None):
                role = json.loads(user).get("role", "planner" if planner else "implementer")
                identifier = config.planner_model if planner else config.model
                if role in {"ux", "architect"}:
                    backend.active.add(identifier)
                    try:
                        barrier.wait(timeout=3)
                        return super().complete(system, user, config, planner=planner, response_schema=response_schema)
                    finally:
                        backend.active.remove(identifier)
                return super().complete(system, user, config, planner=planner, response_schema=response_schema)
        client = ConcurrentClient()
        result = build_graph(client, stub_verifier, workflow="collaborative", resource_scheduler=scheduler).invoke({
            "task_path": str(ROOT / "tasks/todo.json"), "model_config": self.config.model_dump(),
            "resource_config": self.settings.model_dump(), "output_root": self.temp.name, "execution_mode": "mocked"})
        self.assertEqual(result["status"], "passed", result)
        self.assertTrue(result["acceptance"]["accepted"])
        root = Path(result["run_dir"])
        ux = json.loads((root / "ux_metrics.json").read_text())
        architect = json.loads((root / "architect_metrics.json").read_text())
        self.assertEqual(ux["assigned_device"], "cpu")
        self.assertEqual(architect["assigned_device"], "0")
        self.assertEqual(ux["requested_model"], "light")
        self.assertEqual(architect["requested_model"], "heavy")
        self.assertIn("inference_wall_seconds", ux)
        ledger = json.loads((root / "ledger/ux.json").read_text())
        self.assertIn("resources.inference.ux.json", ledger["artifact_hashes"])
        self.assertTrue(all(resident.active == resident.pins == 0 for resident in scheduler.residents.values()))
        self.assertEqual(backend.instances, {})

    def test_baseline_scheduled_graph_remains_functional(self):
        backend = SimulatedBackend()
        result = build_graph(CollaborationMockClient(), stub_verifier, resource_scheduler=ResourceScheduler(backend, backend.hardware)).invoke({
            "task_path": str(ROOT / "tasks/todo.json"), "model_config": self.config.model_dump(),
            "resource_config": self.settings.model_dump(), "output_root": self.temp.name})
        self.assertEqual(result["status"], "passed")
        self.assertNotIn("ux_review", result)


class LMStudioAdapterTests(unittest.TestCase):
    def test_estimator_uses_native_partial_offload_profiles(self):
        backend = LMStudioPlacementBackend("http://127.0.0.1:1234/v1")
        settings = ResourceSettings(context_length=4096, estimate_padding=0)
        info = {"key": "model", "format": "gguf", "type": "llm", "size_bytes": GiB,
                "params_string": "1B", "max_context_length": 8192, "quantization": {"bits_per_weight": 4}}
        calls = []
        def estimate(args):
            calls.append(args)
            mode = args[args.index("--gpu") + 1]
            gpu = {"max": 1.2, "0.75": 0.9, "0.5": 0.6, "0.25": 0.3, "off": 0.1}[mode]
            return f"Estimated GPU Memory: {gpu} GiB\nEstimated Total Memory: 1.5 GiB"
        with patch.object(backend, "model_info", return_value=info), patch.object(backend, "_cli", side_effect=estimate):
            result = backend.estimate("model", settings, 1)
        self.assertEqual([profile.gpu_offload for profile in result.offload_profiles], ["0.75", "0.5", "0.25"])
        self.assertEqual([call[call.index("--gpu") + 1] for call in calls], ["max", "0.75", "0.5", "0.25", "off"])

    def test_successful_estimate_stderr_is_parsed_without_polluting_inventory(self):
        from subprocess import CompletedProcess
        backend = LMStudioPlacementBackend("http://127.0.0.1:1234/v1")
        estimate = "Estimated GPU Memory: 116.46 MiB\nEstimated Total Memory: 4.71 GiB"
        with patch("agentic_web.lmstudio_placement.subprocess.run", return_value=CompletedProcess([], 0, "\n", estimate)):
            self.assertGreater(parse_memory_estimate(backend._cli(["load", "model", "--estimate-only"]))["GPU"], 0)
        with patch("agentic_web.lmstudio_placement.subprocess.run", return_value=CompletedProcess([], 0, "[]", "diagnostic")):
            self.assertEqual(backend.live_instances(), [])

    def test_memory_estimate_parser_keeps_cpu_mode_gpu_overhead(self):
        values = parse_memory_estimate('Estimated GPU Memory: 116.46 MiB\nEstimated Total Memory: 4.71 GiB')
        self.assertGreater(values["GPU"], 0)
        self.assertGreater(values["Total"], values["GPU"])
        with self.assertRaises(ResourceAdmissionError):
            parse_memory_estimate("Unrecognized output")

    def test_cli_load_and_native_unload_use_supported_arguments(self):
        backend = LMStudioPlacementBackend("http://127.0.0.1:1234/v1")
        simulation = SimulatedBackend()
        settings = ResourceSettings(context_length=4096)
        placement = select_placements([simulation.estimate("light", settings, 2)], simulation.hardware(), simulation.capabilities(), settings)[0]["light"]
        inventory = [{"key": "light", "loaded_instances": [{"id": "instance", "config": {"context_length": 4096, "parallel": 2}}]}]
        with patch.object(backend, "model_info", return_value={"key": "light"}), patch.object(backend, "inventory", return_value=inventory), patch.object(backend, "_cli") as cli:
            identifier, _ = backend.load(placement, "instance")
            self.assertEqual(identifier, "instance")
            arguments = cli.call_args.args[0]
            self.assertIn("--gpu", arguments)
            self.assertIn("--parallel", arguments)
            self.assertNotIn("--device", arguments)
        with patch.object(backend, "is_active", return_value=False), patch.object(backend, "_api") as api:
            backend.unload("instance")
            api.assert_called_once_with("/api/v1/models/unload", {"instance_id": "instance"})

    def test_adapter_refuses_to_unload_backend_busy_instance(self):
        backend = LMStudioPlacementBackend("http://127.0.0.1:1234/v1")
        with patch.object(backend, "is_active", return_value=True), patch.object(backend, "_api") as api:
            with self.assertRaises(ResourceAdmissionError):
                backend.unload("instance")
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
