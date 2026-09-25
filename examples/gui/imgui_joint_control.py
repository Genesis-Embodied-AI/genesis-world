"""Interactive joint control example using ImGui overlay.

Demonstrates:
- Simulation controls (play/pause/step/reset)
- Entity browser with joint sliders
- Visualization toggles
- Camera controls
- Custom user panels via register_panel()
- Scene rebuild (add entities, change scale)
"""

import os

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

gs.init(backend=gs.cpu)

# enable_gui attaches the ImGui overlay and lets it manage scene editing internally, so no manual
# InteractiveScene is needed: a plain Scene is the whole setup, and Rebuild Scene is handled inside step().
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, 2.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        enable_gui=True,
    ),
    show_viewer=True,
)
scene.add_entity(
    morph=gs.morphs.Plane(),
    name="Plane",
)
scene.add_entity(
    morph=gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda.xml",
    ),
    name="Panda",
)
scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0, 0, 1.0),
        size=(0.2, 0.2, 0.2),
    ),
    name="Box",
)
scene.build()

# Grab the auto-attached overlay to register a custom panel.
plugin = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))


def custom_panel(imgui):
    imgui.text("Custom Demo Panel")
    imgui.text("This panel was registered via register_panel()")


plugin.register_panel(custom_panel)

is_test = "PYTEST_VERSION" in os.environ
horizon = 5 if is_test else None

# step() honors the GUI: it advances only while playing, and applies a pending Rebuild Scene first. The viewer
# paces the loop to real time (ViewerOptions.realtime_factor), so no manual sleep is needed here.
frame = 0
while scene.viewer.is_alive():
    scene.step()
    frame += 1
    if horizon is not None and frame >= horizon:
        break
