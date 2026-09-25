import importlib
import sys
import textwrap

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
def test_lazy_sensor_discovery(show_viewer, tmp_path):
    from genesis.engine.sensors.camera import RasterizerCameraSensor
    from genesis.engine.sensors.contact_force import ContactSensor
    from genesis.engine.sensors.depth_camera import DepthCameraSensor
    from genesis.engine.sensors.imu import IMUSensor
    from genesis.engine.sensors.sensor_manager import SensorManager

    # Verify built-in registrations resolve to the exact sensor classes
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.Contact] is ContactSensor
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.IMU] is IMUSensor
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.RasterizerCameraOptions] is RasterizerCameraSensor
    # DepthCamera inherits from Raycaster without re-parameterizing, registered only by sensor side
    assert SensorManager.SENSOR_TYPES_MAP[gs.sensors.DepthCamera] is DepthCameraSensor

    # Create a fake plugin package in a temp directory
    pkg_dir = tmp_path / "fake_sensor_plugin"
    pkg_dir.mkdir()

    (pkg_dir / "__init__.py").write_text("")

    (pkg_dir / "options.py").write_text(
        textwrap.dedent(
            """\
        from genesis.options.sensors.options import SensorOptions

        class FakeSensorOptions(SensorOptions["FakeSensor"]):
            pass
        """
        )
    )

    (pkg_dir / "sensor.py").write_text(
        textwrap.dedent(
            """\
        from dataclasses import dataclass

        import genesis as gs
        import torch
        from genesis.engine.sensors.base_sensor import Sensor, SharedSensorMetadata

        from .options import FakeSensorOptions


        @dataclass
        class FakeSensorMetadata(SharedSensorMetadata):
            pass


        class FakeSensor(Sensor[FakeSensorOptions, None, FakeSensorMetadata]):
            def _get_return_format(self):
                return (1,)

            @classmethod
            def _get_cache_dtype(cls):
                return gs.tc_float

            @classmethod
            def _update_shared_cache(
                cls, context, metadata, gt_cache, ground_truth_data_timeline, measured_data_timeline, intermediate_cache,
            ):
                pass

            @classmethod
            def reset(cls, metadata, shared_ground_truth_cache, envs_idx):
                pass

            def build(self):
                pass
        """
        )
    )

    sys.path.insert(0, str(tmp_path))
    try:
        # Import ONLY the options module (not the sensor module)
        options_mod = importlib.import_module("fake_sensor_plugin.options")
        FakeSensorOptions = options_mod.FakeSensorOptions

        # Verify it's not yet registered
        assert FakeSensorOptions not in SensorManager.SENSOR_TYPES_MAP

        # Trigger lazy discovery via resolve
        sensor_cls = SensorManager._resolve_sensor_cls(FakeSensorOptions)
        assert sensor_cls.__name__ == "FakeSensor"

        # Now it should be registered
        assert SensorManager.SENSOR_TYPES_MAP[FakeSensorOptions] is sensor_cls

        # Verify it works end-to-end with a scene
        scene = gs.Scene(show_viewer=show_viewer)
        scene.add_entity(gs.morphs.Plane())
        sensor = scene.add_sensor(FakeSensorOptions())
        scene.build()
        scene.step()
        data = sensor.read()
        assert data.shape[-1] == 1
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name.startswith("fake_sensor_plugin"):
                del sys.modules[mod_name]
        SensorManager.SENSOR_TYPES_MAP.pop(FakeSensorOptions, None)


@pytest.mark.required
def test_post_process_requires_intermediate_override():
    # Strict-override rule: a subclass overriding `_post_process` without also overriding `_get_intermediate_format`
    # or `_get_intermediate_dtype` must raise TypeError at class-definition time. The intermediate buffer is
    # structurally distinct from the return buffer (timeline ring is in intermediate space); the explicit override
    # forces the author to declare it - even a no-op override is acceptable when shape/dtype coincide with return.
    # Local import: importing `genesis.engine.sensors.base_sensor` at module top triggers the sensors package
    # `__init__.py`, which transitively loads `genesis.utils.sdf` and dereferences `gs.qd_float`. That attribute is
    # only set by `gs.init(...)`, which runs in the autouse conftest fixture after pytest collection. Defer here.
    from genesis.engine.sensors.base_sensor import Sensor

    with pytest.raises(TypeError, match="_get_intermediate"):

        class BadSensor(Sensor):
            def _get_return_format(self):
                return (1,)

            @classmethod
            def _get_cache_dtype(cls):
                return gs.tc_float

            @classmethod
            def _post_process(cls, shared_metadata, tensor):
                return tensor * 2


@pytest.mark.required
def test_pipeline_contract(tol):
    # Two synthetic sensor families share a single scene/build:
    #   * `FakePipelineSensor` is a vector sensor whose components each take a different
    #     (physics_imp, measured_only_imp, transform_alpha, hardware_imp) path, so GT-cleanliness, physics
    #     propagation through transform recurrence, the `is_measured` gate on `_apply_transform`, and HW non-
    #     compounding are all verified in one batched pass.
    #   * `FakeSimpleSensor` instances cover the return-space ring allocation paths: no-ring (delay=0,
    #     history=0), history-only ring, delay+history ring, sub-step delay, and jitter declared at build or
    #     enabled afterwards. Every instance shares the same per-class step counter, so they all see the same
    #     raw value at each step and the expected outputs are simple shifts / windows of that sequence.
    from dataclasses import dataclass

    from genesis.engine.sensors.base_sensor import SimpleSensor, SimpleSensorMetadata
    from genesis.options.sensors.options import SimpleSensorOptions

    @dataclass
    class FakeMetadata(SimpleSensorMetadata):
        # Per-component knob vectors, shape `(1, vec_size)` so they broadcast over the batch dim of slot 0.
        step_counter: int = 0
        physics_imp: torch.Tensor = None
        measured_only_imp: torch.Tensor = None
        transform_alpha: torch.Tensor = None
        hardware_imp: torch.Tensor = None

    class FakeOptions(SimpleSensorOptions["FakePipelineSensor"]):
        physics_imp: tuple[float, ...] = (0.0,)
        measured_only_imp: tuple[float, ...] = (0.0,)
        transform_alpha: tuple[float, ...] = (0.0,)
        hardware_imp: tuple[float, ...] = (0.0,)

    class FakePipelineSensor(SimpleSensor[FakeOptions, None, FakeMetadata]):
        def _get_return_format(self):
            return (len(self._options.physics_imp),)

        @classmethod
        def _get_cache_dtype(cls):
            return gs.tc_float

        def build(self):
            super().build()
            self._shared_metadata.physics_imp = torch.tensor(
                [self._options.physics_imp], device=gs.device, dtype=gs.tc_float
            )
            self._shared_metadata.measured_only_imp = torch.tensor(
                [self._options.measured_only_imp], device=gs.device, dtype=gs.tc_float
            )
            self._shared_metadata.transform_alpha = torch.tensor(
                [self._options.transform_alpha], device=gs.device, dtype=gs.tc_float
            )
            self._shared_metadata.hardware_imp = torch.tensor(
                [self._options.hardware_imp], device=gs.device, dtype=gs.tc_float
            )

        @classmethod
        def reset(cls, shared_metadata, ground_truth_cache, envs_idx):
            super().reset(shared_metadata, ground_truth_cache, envs_idx)
            shared_metadata.step_counter = 0

        @classmethod
        def _update_raw_data(cls, context, metadata, raw_data_T):
            # Same scalar raw value across all components and envs; per-component divergence is introduced by the
            # downstream hook vectors. 1-indexed step.
            metadata.step_counter += 1
            raw_data_T.fill_(float(metadata.step_counter))

        @classmethod
        def _apply_physics_imperfections(cls, metadata, slot_0, timeline):
            slot_0.add_(metadata.physics_imp)

        @classmethod
        def _apply_transform(cls, metadata, data, timeline, *, is_measured):
            # Measured-only pre-acquisition contribution: exercises the `is_measured` gate.
            if is_measured:
                data.add_(metadata.measured_only_imp)
            # Stateful linear recurrence per component, branch-symmetric. `timeline.at(1)` is the previous step's
            # post-transform value on this branch (clean of hardware noise - the load-bearing invariant under test).
            data.add_(timeline.at(1) * metadata.transform_alpha)

        @classmethod
        def _apply_hardware_imperfections(cls, metadata, working_buf):
            working_buf.add_(metadata.hardware_imp)

    # Each row is one (physics_imp, measured_only_imp, transform_alpha, hardware_imp) tuple. Components are
    # independent.
    paths = [
        (0.0, 0.0, 0.0, 0.0),  # identity pipeline
        (0.0, 0.0, 0.0, 100.0),  # hardware only: GT must stay clean, measured = raw + H
        (0.0, 0.0, 1.0, 0.0),  # stateful transform on both branches
        (0.0, 0.0, 1.0, 100.0),  # stateful transform + large H: HW must NOT compound through recurrence
        (5.0, 0.0, 0.0, 0.0),  # physics imperfection measured-only, no transform
        (0.0, 5.0, 0.0, 0.0),  # measured-only pre-acquisition (transform with is_measured)
        (5.0, 0.0, 1.0, 0.0),  # physics imperfection compounds through transform recurrence
        (5.0, 5.0, 1.0, 100.0),  # all four together
    ]
    P = np.array([row[0] for row in paths], dtype=np.float32)
    M = np.array([row[1] for row in paths], dtype=np.float32)
    A = np.array([row[2] for row in paths], dtype=np.float32)
    H = np.array([row[3] for row in paths], dtype=np.float32)

    # Companion simple sensor for the ring-allocation paths. No knobs, no overrides beyond raw write - the read
    # just echoes the shared per-class step counter.
    @dataclass
    class FakeSimpleMetadata(SimpleSensorMetadata):
        step_counter: int = 0

    class FakeSimpleOptions(SimpleSensorOptions["FakeSimpleSensor"]):
        pass

    class FakeSimpleSensor(SimpleSensor[FakeSimpleOptions, None, FakeSimpleMetadata]):
        def _get_return_format(self):
            return (1,)

        @classmethod
        def _get_cache_dtype(cls):
            return gs.tc_float

        @classmethod
        def reset(cls, shared_metadata, ground_truth_cache, envs_idx):
            super().reset(shared_metadata, ground_truth_cache, envs_idx)
            shared_metadata.step_counter = 0

        @classmethod
        def _update_raw_data(cls, context, metadata, raw_data_T):
            metadata.step_counter += 1
            raw_data_T.fill_(float(metadata.step_counter))

    DT = 1e-2
    DELAY_STEPS = 2
    DEEP_DELAY_STEPS = 3
    HISTORY_LEN = 3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())  # minimum scene; the sensors do not depend on any physics.
    sensor = scene.add_sensor(
        FakeOptions(
            physics_imp=tuple(P.tolist()),
            measured_only_imp=tuple(M.tolist()),
            transform_alpha=tuple(A.tolist()),
            hardware_imp=tuple(H.tolist()),
        )
    )
    s_baseline = scene.add_sensor(FakeSimpleOptions())
    s_history = scene.add_sensor(FakeSimpleOptions(history_length=HISTORY_LEN))
    s_delay = scene.add_sensor(FakeSimpleOptions(delay=DELAY_STEPS * DT))
    s_both = scene.add_sensor(FakeSimpleOptions(history_length=HISTORY_LEN, delay=DELAY_STEPS * DT))
    s_jitter = scene.add_sensor(FakeSimpleOptions(delay=DELAY_STEPS * DT, jitter=DT))
    s_late_jitter = scene.add_sensor(FakeSimpleOptions(delay=DEEP_DELAY_STEPS * DT))
    s_substep_delay = scene.add_sensor(FakeSimpleOptions(delay=DT / 4))
    scene.build()
    scene.reset()  # zero the build-warmup counter increment so step 1 sees raw = 1.
    # `set_jitter` enables jitter on a sensor built without it, so the ring reservation keys off the delay.
    # `s_late_jitter` holds the class's deepest delay, so its sampling is what wraps if the extra slot is missing.
    s_late_jitter.set_jitter(DT)
    with pytest.raises(Exception, match="read delay"):
        s_baseline.set_jitter(DT)

    n_steps = 8
    gt_observed = np.zeros((n_steps, len(paths)), dtype=np.float32)
    measured_observed = np.zeros((n_steps, len(paths)), dtype=np.float32)
    baseline_observed = np.zeros(n_steps, dtype=np.float32)
    history_observed = np.zeros((n_steps, HISTORY_LEN), dtype=np.float32)
    delay_observed = np.zeros(n_steps, dtype=np.float32)
    both_observed = np.zeros((n_steps, HISTORY_LEN), dtype=np.float32)
    jitter_observed = np.zeros(n_steps, dtype=np.float32)
    late_jitter_observed = np.zeros(n_steps, dtype=np.float32)
    substep_delay_observed = np.zeros(n_steps, dtype=np.float32)
    for i in range(n_steps):
        scene.step()
        gt_observed[i] = tensor_to_array(sensor.read_ground_truth()).reshape(-1)
        measured_observed[i] = tensor_to_array(sensor.read()).reshape(-1)
        baseline_observed[i] = tensor_to_array(s_baseline.read()).item()
        history_observed[i] = tensor_to_array(s_history.read()).reshape(-1)
        delay_observed[i] = tensor_to_array(s_delay.read()).item()
        both_observed[i] = tensor_to_array(s_both.read()).reshape(-1)
        jitter_observed[i] = tensor_to_array(s_jitter.read()).item()
        late_jitter_observed[i] = tensor_to_array(s_late_jitter.read()).item()
        substep_delay_observed[i] = tensor_to_array(s_substep_delay.read()).item()

    # Analytical expectation for the vector sensor, per component. Let raw[k] = k, and (P, M, A, H) be the per-
    # component vectors.
    # GT ring:    gt[k]  = k + A * gt[k-1]                       (raw -> transform with is_measured=False)
    # Meas ring:  m[k]   = (k + P + M) + A * m[k-1]              (raw -> physics_imp -> transform is_measured=True)
    # Measured:   meas[k] = m[k] + H                             (working buffer adds H; no compounding into m)
    gt_expected = np.zeros_like(gt_observed)
    measured_expected = np.zeros_like(measured_observed)
    gt_prev = np.zeros(len(paths), dtype=np.float32)
    m_prev = np.zeros(len(paths), dtype=np.float32)
    for k in range(1, n_steps + 1):
        gt_k = k + A * gt_prev
        m_k = (k + P + M) + A * m_prev
        gt_expected[k - 1] = gt_k
        measured_expected[k - 1] = m_k + H
        gt_prev, m_prev = gt_k, m_k

    assert_allclose(gt_observed, gt_expected, tol=tol)
    assert_allclose(measured_observed, measured_expected, tol=tol)

    # Ring-allocation paths. raw[k] = k for every FakeSimpleSensor instance (shared step counter); delayed reads
    # before slot D has been filled return zero (ring initialized to zero on reset). History reads source slots
    # `at(0..H-1)` of the return-space ring directly - i.e. the last H post-`_post_process` snapshots - without
    # additional delay shift. A sensor that configures both `delay > 0` and `history_length > 0` therefore sees
    # undelayed history alongside a delayed non-history read; this matches the implementation and is what the
    # combined test asserts.
    raw = np.arange(1, n_steps + 1, dtype=np.float32)
    expected_baseline = raw
    expected_delay = np.where(raw - DELAY_STEPS >= 1, raw - DELAY_STEPS, 0.0)
    expected_history = np.zeros((n_steps, HISTORY_LEN), dtype=np.float32)
    for k in range(1, n_steps + 1):
        for h in range(HISTORY_LEN):
            past_step = k - h
            expected_history[k - 1, h] = past_step if past_step >= 1 else 0.0

    assert_allclose(baseline_observed, expected_baseline, tol=tol)
    # A delay under half a step rounds to zero slots, so the read is undelayed.
    assert_allclose(substep_delay_observed, expected_baseline, tol=tol)
    assert_allclose(delay_observed, expected_delay, tol=tol)
    assert_allclose(history_observed, expected_history, tol=tol)
    # The combined delay + history sensor returns the same history as the history-only sensor (delay is bypassed
    # by the ring-gather history path); verify they match.
    assert_allclose(both_observed, expected_history, tol=tol)
    # `jitter == dt` shifts a read by zero or one extra slot, so every sample is bracketed by the delayed sequence and
    # the raw value one step older.
    for observed, delay_steps in ((jitter_observed, DELAY_STEPS), (late_jitter_observed, DEEP_DELAY_STEPS)):
        hi = np.where(raw - delay_steps >= 1, raw - delay_steps, 0.0)
        lo = np.where(raw - delay_steps - 1 >= 1, raw - delay_steps - 1, 0.0)
        assert ((observed >= lo) & (observed <= hi)).all()


@pytest.mark.required
def test_pipeline_contract_uint8_delay(tol):
    # ZOH delay sampling must work on non-float return dtypes. A sensor whose `_post_process` casts a float
    # intermediate to a `uint8` return stores `uint8` snapshots in the per-class return-space ring; delay
    # sampling reads those slots verbatim (the dtype-safe ZOH default). Verifies the slot is correctly typed
    # and the delayed values match the cast of `raw[k - delay]`.
    from dataclasses import dataclass

    from genesis.engine.sensors.base_sensor import SimpleSensor, SimpleSensorMetadata
    from genesis.options.sensors.options import SimpleSensorOptions

    @dataclass
    class FakeQuantizedMetadata(SimpleSensorMetadata):
        step_counter: int = 0

    class FakeQuantizedOptions(SimpleSensorOptions["FakeQuantizedSensor"]):
        pass

    class FakeQuantizedSensor(SimpleSensor[FakeQuantizedOptions, None, FakeQuantizedMetadata]):
        def _get_return_format(self):
            return (1,)

        @classmethod
        def _get_cache_dtype(cls):
            return torch.uint8

        @classmethod
        def _get_intermediate_dtype(cls):
            return gs.tc_float

        @classmethod
        def reset(cls, shared_metadata, ground_truth_cache, envs_idx):
            super().reset(shared_metadata, ground_truth_cache, envs_idx)
            shared_metadata.step_counter = 0

        @classmethod
        def _update_raw_data(cls, context, metadata, raw_data_T):
            metadata.step_counter += 1
            raw_data_T.fill_(float(metadata.step_counter))

        @classmethod
        def _post_process(cls, shared_metadata, tensor, timeline, *, is_measured):
            return tensor.clamp(0, 255).to(torch.uint8)

    DT = 1e-2
    DELAY_STEPS = 2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    sensor = scene.add_sensor(FakeQuantizedOptions(delay=DELAY_STEPS * DT))
    scene.build()
    scene.reset()

    n_steps = 8
    observed = np.zeros(n_steps, dtype=np.uint8)
    for i in range(n_steps):
        scene.step()
        observed[i] = tensor_to_array(sensor.read()).item()

    raw = np.arange(1, n_steps + 1, dtype=np.float32)
    expected = np.where(raw - DELAY_STEPS >= 1, raw - DELAY_STEPS, 0.0).astype(np.uint8)
    assert observed.dtype == np.uint8
    assert_equal(observed, expected)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_add_and_read_all_registered_sensors():
    from genesis.engine.sensors.sensor_manager import SensorManager

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        )
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.2, 0.0, 0.1),
        )
    )

    sensors = []

    for option_cls in SensorManager.SENSOR_TYPES_MAP.keys():
        sensor_kwargs = {}
        if issubclass(option_cls, gs.sensors.BaseCameraOptions):
            continue  # skip camera options
        if issubclass(option_cls, gs.sensors.RigidSensorOptionsMixin):
            sensor_kwargs.update(
                entity_idx=box.idx,
            )
        if issubclass(option_cls, gs.sensors.Raycaster):
            sensor_kwargs.update(
                pattern=gs.sensors.raycaster.DepthCameraPattern(),
            )
        if issubclass(
            option_cls,
            (gs.sensors.SurfaceDistanceProbe, gs.sensors.ProximityTaxel, gs.sensors.ElastomerTaxel),
        ):
            sensor_kwargs.update(
                track_link_idx=(sphere.base_link_idx,),
            )
        if issubclass(option_cls, gs.sensors.TemperatureGrid):
            sensor_kwargs.update(
                properties_dict={
                    -1: gs.sensors.TemperatureProperties(),
                },
            )

        sensor = scene.add_sensor(option_cls(**sensor_kwargs))
        sensors.append(sensor)

    scene.build(n_envs=2)

    scene.step()
    for sensor in sensors:
        sensor.read()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_sensor_history_length_contact_and_imu(show_viewer, tol, n_envs):
    # history_length stacks recent frames from the ring snapshot buffers.
    GRAVITY = -10.0
    DT = 1e-2
    HISTORY_LEN = 4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )

    contact_h = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box.idx,
            history_length=HISTORY_LEN,
        )
    )
    imu_h = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
            history_length=HISTORY_LEN,
        )
    )
    imu_ref = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
        )
    )

    scene.build(n_envs=n_envs)

    def _expected_shape_with_history(shape: tuple[int, ...]):
        return (HISTORY_LEN, *shape) if n_envs == 0 else (n_envs, HISTORY_LEN, *shape)

    prev_c = None
    prev_i = None
    for _ in range(HISTORY_LEN * 2):
        scene.step()
        cg = contact_h.read_ground_truth()
        assert cg.shape == _expected_shape_with_history((1,))
        ig = imu_h.read_ground_truth()
        assert ig.lin_acc.shape == _expected_shape_with_history((3,))
        assert ig.ang_vel.shape == _expected_shape_with_history((3,))
        assert ig.mag.shape == _expected_shape_with_history((3,))

        assert_equal(contact_h.read(), cg)

        batch_shape = () if n_envs == 0 else (slice(None),)
        cur_slice = (*batch_shape, 0)
        prev_slice = (*batch_shape, 1)
        assert_allclose(ig.lin_acc[cur_slice], imu_ref.read_ground_truth().lin_acc, tol=tol)
        assert_allclose(ig.ang_vel[cur_slice], imu_ref.read_ground_truth().ang_vel, tol=tol)
        assert_allclose(ig.mag[cur_slice], imu_ref.read_ground_truth().mag, tol=tol)

        if prev_c is not None:
            assert_equal(cg[prev_slice], prev_c[cur_slice])
        if prev_i is not None:
            assert_allclose(ig.lin_acc[prev_slice], prev_i.lin_acc[cur_slice], tol=gs.EPS)
            assert_allclose(ig.ang_vel[prev_slice], prev_i.ang_vel[cur_slice], tol=gs.EPS)
            assert_allclose(ig.mag[prev_slice], prev_i.mag[cur_slice], tol=gs.EPS)
        prev_c = cg
        prev_i = ig


@pytest.mark.required
def test_shared_context(show_viewer):
    # Raycaster and DepthCamera are distinct sensor types that both cast against the scene geometry; they must share
    # one RaycastContext (a single BVH set) instead of each building its own. A sensor type declaring no context (IMU)
    # must resolve to None.
    from genesis.engine.sensors.raycaster import RaycastContext

    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.4, 0.4, 0.4),
            pos=(0.0, 0.0, 1.0),
        )
    )

    raycaster = scene.add_sensor(
        gs.sensors.Raycaster(
            pattern=gs.sensors.raycaster.GridPattern(resolution=0.2, size=(0.4, 0.4), direction=(0.0, 0.0, -1.0)),
            pos_offset=(0.0, 0.0, 2.0),
        )
    )
    depth_camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.raycaster.DepthCameraPattern(res=(4, 4)),
            pos_offset=(0.0, 0.0, 2.0),
        )
    )
    imu = scene.add_sensor(gs.sensors.IMU(entity_idx=box.idx))
    scene.build()

    contexts = list(raycaster._manager._shared_contexts.values())
    # Exactly one shared context instance, of type RaycastContext.
    assert len(contexts) == 1
    assert isinstance(contexts[0], RaycastContext)
    # Both raycast-casting sensor types resolve to that single instance, so they cast against the very same BVH list
    # instead of one built per sensor type.
    assert raycaster._shared_context is contexts[0]
    assert depth_camera._shared_context is contexts[0]
    assert raycaster._shared_context.bvh_contexts is depth_camera._shared_context.bvh_contexts
    # The plane+box scene mixes static and dynamic collision faces, so the shared collision mesh is split into a
    # static and a dynamic BVH - two entries, both shared by the two sensor types.
    assert len(raycaster._shared_context.bvh_contexts) == 2
    # A sensor type that declares no context resolves to None.
    assert imu._shared_context is None

    # Functional smoke: both casters return finite hit distances after a step.
    scene.step()
    assert torch.isfinite(raycaster.read().distances).all()
    assert torch.isfinite(depth_camera.read_image()).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 3])
def test_read_sensors_bulk_api(show_viewer, n_envs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box_a = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )
    box_b = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0.0, 0.2),
        ),
    )

    # Diverse sensor set covering multiple dtypes (float for IMU/ContactForce, bool for Contact, uint8 for the static
    # camera) and heterogeneous per-sensor cache sizes within the float dtype (9 cells for IMU vs 3 for ContactForce).
    # Two IMUs on box_a, one IMU on box_b. ContactForce and Contact sensors on both boxes. A static camera not attached
    # to any entity (entity_idx defaults to -1).
    imu_a1 = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box_a.idx,
        ),
    )
    imu_a2 = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box_a.idx,
        ),
    )
    imu_b = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box_b.idx,
        ),
    )
    force_a = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_a.idx,
        ),
    )
    force_b = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=box_b.idx,
        ),
    )
    contact_a = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_a.idx,
        ),
    )
    contact_b = scene.add_sensor(
        gs.sensors.Contact(
            entity_idx=box_b.idx,
        ),
    )
    static_cam = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(32, 32),
        ),
    )

    scene.build(n_envs=n_envs)
    for _ in range(5):
        scene.step()

    # Scene-wide read returns every sensor class. Per-entity reads restrict to classes present on that entity, so the
    # static camera class is excluded from both box_a and box_b reads. Each call allocates a fresh tensor per class.
    scene_data = scene.read_sensors()
    a_data = box_a.read_sensors()
    b_data = box_b.read_sensors()
    assert set(scene_data.keys()) == {
        gs.sensors.types.IMU,
        gs.sensors.types.ContactForce,
        gs.sensors.types.Contact,
        gs.sensors.types.RasterizerCameraOptions,
    }
    assert set(a_data.keys()) == {gs.sensors.types.IMU, gs.sensors.types.ContactForce, gs.sensors.types.Contact}
    assert set(b_data.keys()) == {gs.sensors.types.IMU, gs.sensors.types.ContactForce, gs.sensors.types.Contact}

    # Sensors within a class are sorted by entity_idx, so per-entity reads must match contiguous slices of the
    # scene-wide read.
    for type_tag, a_slice, b_slice in (
        (gs.sensors.types.IMU, slice(0, 18), slice(18, 27)),
        (gs.sensors.types.ContactForce, slice(0, 3), slice(3, 6)),
        (gs.sensors.types.Contact, slice(0, 1), slice(1, 2)),
    ):
        assert_equal(a_data[type_tag], scene_data[type_tag][..., a_slice])
        assert_equal(b_data[type_tag], scene_data[type_tag][..., b_slice])

    # Individual sensor reads must agree with bulk reads at both scene and entity levels.
    # IMU cache layout per sensor is 3 acc + 3 gyro + 3 mag in that order.
    for local_idx, imu in enumerate((imu_a1, imu_a2, imu_b)):
        base = local_idx * 9
        imu_data = imu.read()
        assert_equal(scene_data[gs.sensors.types.IMU][..., base : base + 3], imu_data.lin_acc)
        assert_equal(scene_data[gs.sensors.types.IMU][..., base + 3 : base + 6], imu_data.ang_vel)
        assert_equal(scene_data[gs.sensors.types.IMU][..., base + 6 : base + 9], imu_data.mag)
    for entity_local_idx, imu in enumerate((imu_a1, imu_a2)):
        base = entity_local_idx * 9
        imu_data = imu.read()
        assert_equal(a_data[gs.sensors.types.IMU][..., base : base + 3], imu_data.lin_acc)
        assert_equal(a_data[gs.sensors.types.IMU][..., base + 3 : base + 6], imu_data.ang_vel)
        assert_equal(a_data[gs.sensors.types.IMU][..., base + 6 : base + 9], imu_data.mag)
    # ContactForce returns a 3-vector per sensor.
    for local_idx, force in enumerate((force_a, force_b)):
        base = local_idx * 3
        assert_equal(scene_data[gs.sensors.types.ContactForce][..., base : base + 3], force.read())
    assert_equal(a_data[gs.sensors.types.ContactForce], force_a.read())
    assert_equal(b_data[gs.sensors.types.ContactForce], force_b.read())
    # Contact returns a bool per sensor.
    assert_equal(scene_data[gs.sensors.types.Contact][..., 0:1], contact_a.read())
    assert_equal(scene_data[gs.sensors.types.Contact][..., 1:2], contact_b.read())
    assert_equal(a_data[gs.sensors.types.Contact], contact_a.read())
    assert_equal(b_data[gs.sensors.types.Contact], contact_b.read())

    # `read_sensors` always returns a fresh tensor independent of internal sensor storage. Two successive calls (at
    # the scene level or the entity level) must back onto distinct storage. Verified on both IMU (identity
    # `_post_process`, intermediate-space ring) and Contact (overridden `_post_process`, per-class return-space ring).
    for type_tag in (gs.sensors.types.IMU, gs.sensors.types.Contact):
        scene_a = scene.read_sensors()[type_tag]
        scene_b = scene.read_sensors()[type_tag]
        entity_a = box_a.read_sensors()[type_tag]
        assert scene_a.untyped_storage().data_ptr() != scene_b.untyped_storage().data_ptr()
        assert scene_a.untyped_storage().data_ptr() != entity_a.untyped_storage().data_ptr()
        assert_equal(scene_a, scene_b)

    # Batching must be exercised end-to-end. For n_envs > 0, every per-env row of the bulk view must equal that env's
    # individual sensor read.
    if n_envs > 0:
        for env_idx in range(n_envs):
            assert_equal(scene.read_sensors()[gs.sensors.types.IMU][env_idx, 0:3], imu_a1.read().lin_acc[env_idx])
            assert_equal(scene.read_sensors()[gs.sensors.types.Contact][env_idx, 0:1], contact_a.read()[env_idx])
