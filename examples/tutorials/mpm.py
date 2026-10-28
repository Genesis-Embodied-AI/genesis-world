import os
import genesis as gs

gs.init(backend=gs.cpu)


scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.5, -1.0, 0.0),
        upper_bound=(0.5, 1.0, 1),
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(960, 640),
        camera_fov=30,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

obj_elastic = scene.add_entity(
    options=gs.options.MPMEntityOptions(
        morph=gs.morphs.Box(
            pos=(0.0, -0.5, 0.25),
            size=(0.2, 0.2, 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode="visual",
        ),
        material=gs.materials.Elastic(),
    ),
)

obj_sand = scene.add_entity(
    options=gs.options.MPMEntityOptions(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.3, 1.0),
            vis_mode="particle",
        ),
        material=gs.materials.Liquid(),
    ),
)

obj_plastic = scene.add_entity(
    options=gs.options.MPMEntityOptions(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.5, 0.35),
            radius=0.1,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode="particle",
        ),
        material=gs.materials.ElastoPlastic(),
    ),
)


scene.build()

horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
for i in range(horizon):
    scene.step()
