"""A chain defined by a user generator morph swings from a fixed anchor into a stack of boxes.

'Chain' subclasses gs.morphs.Generator: it holds the options of the chain alone, and the function registered with
gs.register_generator builds its geometry as the entity is added, here an inline MJCF model of capsule links joined by
ball joints, the first one hanging from the anchor. The pose of the generator morph places and tilts the whole chain.
With the viewer, a panel edits the options of the chain and rebuilds the scene, which generates the chain again.
"""

import argparse
import os
import xml.etree.ElementTree as ET

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin


class Chain(gs.morphs.Generator, specs=(gs.entities.Rigid,)):
    """A chain of capsule links hanging along -z from its origin, each joined to the previous one by a ball joint."""

    n_links: int = 8
    link_length: float = 0.12
    link_radius: float = 0.02
    weight_radius: float = 0.08


@gs.register_generator(Chain)
def generate_chain(morph: Chain, context: gs.engine.generators.GeneratorContext) -> gs.morphs.MJCF:
    mjcf = ET.Element("mujoco", model="chain")
    parent = ET.SubElement(mjcf, "worldbody")
    for i_link in range(morph.n_links):
        body = ET.SubElement(parent, "body", pos=f"0 0 {-morph.link_length if i_link else 0.0}")
        ET.SubElement(body, "joint", type="ball", damping="0.01")
        ET.SubElement(
            body,
            "geom",
            type="capsule",
            fromto=f"0 0 0 0 0 {-morph.link_length}",
            size=f"{morph.link_radius}",
            rgba="0.4 0.4 0.45 1",
        )
        parent = body
    # A heavy weight at the free end, so that the swing knocks the boxes over
    ET.SubElement(
        parent,
        "geom",
        type="sphere",
        pos=f"0 0 {-morph.link_length - morph.weight_radius}",
        size=f"{morph.weight_radius}",
        density="7800",
        rgba="0.3 0.3 0.32 1",
    )
    return gs.morphs.MJCF(
        file=ET.tostring(mjcf, encoding="unicode"),
        pos=morph.pos,
        quat=morph.quat,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument(
        "-v", "--vis", action="store_true", help="Open the interactive viewer, with a panel generating the chain"
    )
    parser.add_argument("-r", "--record", action="store_true", help="Record the swing to 'out/generator_chain.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    camera_pos = (2.0, -2.2, 1.2)
    camera_lookat = (0.3, 0.0, 0.5)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            enable_gui=args.vis,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    # The chain hangs from 1.3 m and starts tilted 70 degrees from the vertical, swinging towards +x
    chain = Chain(
        pos=(0.0, 0.0, 1.3),
        euler=(0.0, -70.0, 0.0),
    )
    scene.add_entity(
        spec=gs.entities.Rigid(
            morph=chain,
            name="chain",
        ),
    )
    box_size = 0.1
    for i_z in range(4):
        scene.add_entity(
            gs.morphs.Box(
                pos=(0.25, 0.0, (i_z + 0.5) * box_size),
                size=(box_size, box_size, box_size),
            ),
            surface=gs.surfaces.Rough(
                color=(0.62, 0.48, 0.42),
            ),
        )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )
    if args.vis:
        scene.viewer.add_plugin(gs.vis.viewer_plugins.MouseInteractionPlugin())

    scene.build()

    if args.vis:
        overlay = next(plugin for plugin in scene.viewer.plugins if isinstance(plugin, ImGuiOverlayPlugin))

        def chain_panel(imgui):
            nonlocal chain
            imgui.text("Chain generator")
            # The sliders edit the options of the chain to generate next, the chain in the scene staying as built
            _, n_links = imgui.slider_int("Links", chain.n_links, 1, 20)
            _, link_length = imgui.slider_float("Link length", chain.link_length, 0.02, 0.3)
            _, link_radius = imgui.slider_float("Link radius", chain.link_radius, 0.005, 0.05)
            _, weight_radius = imgui.slider_float("Weight radius", chain.weight_radius, 0.01, 0.15)
            chain = chain.model_copy(
                update=dict(
                    n_links=n_links,
                    link_length=link_length,
                    link_radius=link_radius,
                    weight_radius=weight_radius,
                )
            )
            if imgui.button("Generate"):
                # Every other entity keeps the spec it was built from, so that the rebuild reuses its description
                entities_kwargs = {entity.name: {"spec": entity.spec} for entity in scene.entities}
                entities_kwargs["chain"] = {"spec": gs.entities.Rigid(morph=chain, name="chain")}
                overlay.interactive_scene.rebuild(entities_kwargs=entities_kwargs)

        overlay.register_panel(chain_panel)
        while scene.viewer.is_alive():
            scene.step()
        return

    if camera is not None:
        camera.start_recording(save_to_filename="out/generator_chain.mp4", fps=30)
    horizon = 5 if "PYTEST_VERSION" in os.environ else args.steps
    for _ in range(horizon):
        scene.step()
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
