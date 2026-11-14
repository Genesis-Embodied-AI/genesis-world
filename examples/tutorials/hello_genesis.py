import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene()

plane = scene.add_entity(
    options=gs.entities.Rigid(
        morph=gs.morphs.Plane(),
    ),
)
franka = scene.add_entity(
    options=gs.entities.Rigid(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    ),
)

scene.build()
for i in range(1000):
    scene.step()
