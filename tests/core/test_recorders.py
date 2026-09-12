import csv
import time

import numpy as np
import pytest

import av

import genesis as gs
from genesis.utils.image_exporter import as_grayscale_image

from ..utils.assertions import assert_allclose, rgb_array_to_png_bytes


@pytest.fixture
def mpl_agg_backend():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Force using Agg backend for repeatability
    try:
        mpl_backend = mpl.get_backend()
    except AttributeError:
        mpl_backend = "Agg"
    plt.switch_backend("Agg")

    yield

    # Restore original backend
    plt.switch_backend(mpl_backend)


@pytest.mark.required
def test_vector_field_plotter_subplots(mpl_agg_backend, png_snapshot):
    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        )
    )

    n_probes = 9
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3)), -1).reshape(-1, 2)
    positions = np.c_[grid, np.zeros(n_probes)]
    titles = ("a", "b", "c", "d")

    def grid_data():
        return np.stack([(positions - 0.5) * (i_title + 1) * 0.1 for i_title in range(len(titles))])

    grid_plotter = scene.add_recorder(
        data_func=grid_data,
        rec_options=gs.recorders.MPLVectorFieldPlot(
            title="grid",
            show_window=False,
            positions=positions,
            normal=(0.0, 0.0, 1.0),
            subplot_titles=titles,
        ),
    )
    single_plotter = scene.add_recorder(
        data_func=lambda: (positions - 0.5) * 0.1,
        rec_options=gs.recorders.MPLVectorFieldPlot(
            title="single",
            show_window=False,
            positions=positions,
            normal=(0.0, 0.0, 1.0),
        ),
    )

    twist_signs = np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    expected_twist = np.linspace(-0.5, 0.5, n_probes)[None, :] * twist_signs

    def twist_data():
        vectors = grid_data()
        twist = np.zeros_like(vectors)
        # normal is +z, so the twist about the view axis is the z component of twist_vectors.
        twist[..., 2] = expected_twist
        return vectors, twist

    twist_plotter = scene.add_recorder(
        data_func=twist_data,
        rec_options=gs.recorders.MPLVectorFieldPlot(
            title="twist",
            show_window=False,
            positions=positions,
            normal=(0.0, 0.0, 1.0),
            subplot_titles=titles,
            twist_scale_factor=1.0,
            twist_max_magnitude=0.5,
        ),
    )

    scene.build()
    # The plotter overwrites the quiver data in place each call, so a single step already exercises process().
    scene.step()
    for plotter in (grid_plotter, single_plotter, twist_plotter):
        if plotter.run_in_thread:
            plotter.sync()

    titled = [ax.get_title() for ax in grid_plotter.fig.get_axes() if ax.get_title() in titles]
    assert tuple(titled) == titles  # one subplot per title, in order
    assert len(grid_plotter._quivers) == len(titles)
    assert len(single_plotter._quivers) == 1  # no subplot_titles -> a single quiver
    for plotter in (grid_plotter, single_plotter):
        assert rgb_array_to_png_bytes(plotter.get_image_array()) == png_snapshot

    # Twist overlay: one curved-arrow collection per subplot, each colored by the signed twist fed to it.
    assert len(twist_plotter._twist_arcs) == len(titles)
    assert len(twist_plotter._twist_heads) == len(titles)
    for i_ax, arcs in enumerate(twist_plotter._twist_arcs):
        assert len(arcs.get_segments()) == n_probes
        assert_allclose(arcs.get_array(), expected_twist[i_ax], tol=gs.EPS)


@pytest.mark.required
def test_plotter(tmp_path, monkeypatch, mpl_agg_backend, png_snapshot):
    DT = 0.01
    STEPS = 10
    HISTORY_LENGTH = 5

    # FIXME: Hijack video writter to keep track of all the frames that are being recorded
    buffers = []

    def process(self, data, cur_time):
        nonlocal buffers
        buffers.append((data, cur_time))

    monkeypatch.setattr("genesis.recorders.file_writers.VideoFileWriter.process", process)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
        ),
    )

    call_count = 0

    def dummy_data_func():
        nonlocal call_count
        call_count += 1
        return {
            "a": [call_count * 0.1, call_count * 0.2, call_count * 0.3],
            "b": [call_count * 0.01, call_count * 0.02],
        }

    plotter = scene.add_recorder(
        data_func=dummy_data_func,
        rec_options=gs.recorders.MPLLinePlot(
            labels={"a": ("x", "y", "z"), "b": ("u", "v")},
            history_length=HISTORY_LENGTH,
            hz=1.0 / DT / 2,  # half of the simulation frequency, so every other step
            title="Test MPLPlotter",
            window_size=(400, 300),
            save_to_filename=tmp_path / "video.mp4",
            show_window=False,
        ),
    )

    scene.build()

    for _ in range(STEPS):
        scene.step()

    if plotter.run_in_thread:
        plotter.sync()

    assert call_count == STEPS // 2 + 1  # one additional call during plot setup
    assert len(plotter.line_plot.x_data) == HISTORY_LENGTH
    # A recorder reads the state a step starts from, so the last sample is the one before the second to last step
    assert np.isclose(plotter.line_plot.x_data[-1], (STEPS - 2) * DT, atol=gs.EPS)
    assert rgb_array_to_png_bytes(plotter.get_image_array()) == png_snapshot

    assert len(buffers) == 5
    assert_allclose([cur_time for _, cur_time in buffers], np.arange(STEPS)[::2] * DT, tol=gs.EPS)
    for rgb_diff in np.diff([data for data, _ in buffers], axis=0):
        assert rgb_diff.max() > 10.0

    # Intentionally do not stop the recording to test the destructor
    # scene.stop_recording()


@pytest.mark.required
def test_file_writers(tmp_path):
    STEPS = 10

    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.06),
        ),
    )

    contact_sensor = scene.add_sensor(gs.sensors.Contact(entity_idx=box.idx))

    csv_file = tmp_path / "contact_data.csv"
    csv_writer = gs.recorders.CSVFile(filename=csv_file, header=("in_contact",))
    contact_sensor.start_recording(csv_writer)

    csv_array_file = tmp_path / "array_data.csv"
    scene.add_recorder(
        data_func=lambda: {"batch": np.arange(6).reshape(2, 3)},
        rec_options=gs.recorders.CSVFile(filename=csv_array_file),
    )

    npz_file = tmp_path / "scene_data.npz"
    scene.add_recorder(
        data_func=lambda: {"box_pos": box.get_pos(), "dummy": 1},
        rec_options=gs.recorders.NPZFile(filename=npz_file),
    )

    scene.build()

    for _ in range(STEPS):
        scene.step()

    scene.stop_recording()

    assert csv_file.exists()
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

        assert len(rows) == STEPS + 2  # header, the state before each step and the state the run stopped at
        assert rows[1][1] in ("False", "0")  # not in contact initially
        assert rows[-1][1] in ("True", "1")  # in contact after falling

    assert csv_array_file.exists()
    with open(csv_array_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

        assert rows[0] == ["timestamp", "batch_0", "batch_1", "batch_2", "batch_3", "batch_4", "batch_5"]
        assert rows[1][1:] == ["0", "1", "2", "3", "4", "5"]

    assert npz_file.exists()
    data = np.load(npz_file)
    assert "timestamp" in data
    assert "box_pos" in data
    assert "dummy" in data
    assert len(data["timestamp"]) == STEPS + 1


@pytest.mark.required
def test_error_handling(tmp_path):
    scene = gs.Scene(
        show_viewer=False,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 1.0),
        ),
    )
    # Recorder options are checked as they are built
    with pytest.raises(gs.GenesisException, match="not supported"):
        gs.recorders.VideoFile(filename=tmp_path / "video.mp4", codec="no_such_codec")
    # A recorder that raises records no more, a reset included. Registered first, the rejection of a frame aborts the
    # step before the other recorders sample it. The first video rejects the frame the stop records, so it samples every
    # step, the reset and the stop, and the second one rejects its third frame.
    stop_video_frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 7 + [np.full((64, 64, 3), 0.5)]
    scene.add_recorder(
        data_func=lambda: stop_video_frames.pop(0),
        rec_options=gs.recorders.VideoFile(
            filename=tmp_path / "stop.mp4",
        ),
    )
    step_video_path = tmp_path / "step.mp4"
    step_video_frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 2 + [np.full((64, 64, 3), 0.5)]
    scene.add_recorder(
        data_func=lambda: step_video_frames.pop(0),
        rec_options=gs.recorders.VideoFile(
            filename=step_video_path,
        ),
    )
    pos_path = tmp_path / "pos.csv"
    scene.add_recorder(
        data_func=box.get_pos,
        rec_options=gs.recorders.CSVFile(
            filename=pos_path,
        ),
    )
    # Both reject the value of their second sample. Registered last, the failure of the second one is raised once the
    # other recorders have sampled the step.
    synced_values, stepped_values = [1.0, None], [1.0, None]
    synced_path = tmp_path / "synced.csv"
    synced_recorder = scene.add_recorder(
        data_func=lambda: synced_values.pop(0),
        rec_options=gs.recorders.CSVFile(
            filename=synced_path,
        ),
    )
    stepped_recorder = scene.add_recorder(
        data_func=lambda: stepped_values.pop(0),
        rec_options=gs.recorders.CSVFile(
            filename=tmp_path / "stepped.csv",
        ),
    )
    scene.build()
    # A recorder sizes its buffers from the build, so none is registered after it
    with pytest.raises(gs.GenesisException):
        scene.add_recorder(data_func=box.get_pos, rec_options=gs.recorders.CSVFile(filename=tmp_path / "late.csv"))

    scene.step()
    scene.step()
    # A failure on the recording thread is raised on the stepping thread, once, by the sync that waits for it or at the
    # next step the recorder is reached once it failed
    with pytest.raises(gs.GenesisException, match="Unsupported data type"):
        synced_recorder.sync()
    for _ in range(1000):
        if not stepped_recorder.is_alive:
            break
        time.sleep(0.01)
    assert not stepped_recorder.is_alive
    # A normalized floating-point frame cannot be represented and is reported, rather than truncated to a black one. The
    # step recording it is aborted.
    with pytest.raises(gs.GenesisException, match="integer type"):
        scene.step()
    with pytest.raises(gs.GenesisException, match="Unsupported data type"):
        scene.step()
    # A reset restarts the recorders that did not fail
    scene.reset()
    scene.step()
    scene.step()

    # The state the run stopped at reaches every recorder, then the recorders are stopped and the failure is raised,
    # and the destruction of the scene completes behind it
    with pytest.raises(gs.GenesisException, match="integer type"):
        scene.destroy()
    assert scene.sim is None and scene.visualizer is None
    # The header, then the first, second and fourth steps and the reset, then the two steps and the stop
    with open(pos_path, "r") as f:
        assert len(list(csv.reader(f))) == 1 + 4 + 3
    # The header, then the first step
    with open(synced_path, "r") as f:
        assert len(list(csv.reader(f))) == 1 + 1
    with av.open(step_video_path) as container:
        assert sum(1 for _ in container.decode(video=0)) == 2


@pytest.mark.required
def test_video_writer(tmp_path):
    STEPS = 10

    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 1.0),
        ),
    )
    camera = scene.add_camera(
        res=(300, 200),  # Using weird resolution to trigger padding
        pos=(2.0, 2.0, 2.0),
        lookat=(0.0, 0.0, 0.2),
        GUI=False,
    )
    video_rgb_path = tmp_path / "test_rgb.mp4"
    scene.add_recorder(
        data_func=lambda: camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0],
        # No explicit codec — exercises automatic codec selection
        rec_options=gs.recorders.VideoFile(
            filename=video_rgb_path,
        ),
    )
    video_depth_path = tmp_path / "test_depth.mp4"
    scene.add_recorder(
        data_func=lambda: as_grayscale_image(camera.render(rgb=False, depth=True, segmentation=False, normal=False)[1]),
        rec_options=gs.recorders.VideoFile(
            filename=video_depth_path,
        ),
    )
    scene.build()

    for _ in range(STEPS):
        scene.step()

    scene.stop_recording()

    # Every sampled step must survive into the file: frames buffered by the encoder but never flushed, or dropped
    # because it fell behind, would leave a shorter video that a size check cannot distinguish from a complete one. Each
    # file holds the state before each step, then the state the run stopped at.
    for video_path in (video_rgb_path, video_depth_path):
        assert video_path.exists(), "Recorded video file should exist"
        with av.open(video_path) as container:
            assert sum(1 for _ in container.decode(video=0)) == STEPS + 1
