import os

import genesis as gs

gs.init(backend=gs.cpu)


scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 720),
        camera_fov=30,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

cloth_1 = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=2.0,
        pos=(0, 0, 0.5),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0),
        vis_mode="visual",
    ),
)

cloth_2 = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=2.0,
        pos=(0, 0, 1.0),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.8, 0.4, 0.2, 1.0),
        vis_mode="particle",
    ),
)

scene.build()

cloth_1.fix_particles(cloth_1.find_closest_particle((-1, -1, 1.0)))
cloth_1.fix_particles(cloth_1.find_closest_particle((1, 1, 1.0)))
cloth_1.fix_particles(cloth_1.find_closest_particle((-1, 1, 1.0)))
cloth_1.fix_particles(cloth_1.find_closest_particle((1, -1, 1.0)))
cloth_2.fix_particles(cloth_2.find_closest_particle((-1, -1, 1.0)))

horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
for i in range(horizon):
    scene.step()
