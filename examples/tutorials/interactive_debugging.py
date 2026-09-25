import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=False)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

cam_0 = scene.add_camera()
scene.build()

# enter IPython's interactive mode
import IPython

IPython.embed()
