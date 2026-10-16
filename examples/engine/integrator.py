"""Integrate the DOFs of a rigid box and a cloth through one Integrator manager, inside one graph-captured step.

The Engine adds Cloth by hand, adds Rigid because create_rigid (rigid_system.py) returns it for a scene containing a
rigid entity, and adds the Integrator because the Engine and both participants require it. The user only builds the
scene.
"""

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from cloth_system import Cloth
from integrator_engine import IntegratorEngine

# Importing the module registers create_rigid for IntegratorEngine
import rigid_system


def main():
    gs.init(backend=gs.cpu)

    # The free box makes create_rigid return Rigid, which the ranks place before the cloth in the Integrator
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.5), show_viewer=False)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    engine = IntegratorEngine(scene)
    cloth = engine.add_system(Cloth(scene))
    engine.build()
    print("systems:", [system_cls.__name__ for system_cls in engine.systems])
    print("cloth.rigid:", type(cloth.rigid).__name__)

    engine.init_pipeline.run()
    print("segments n_dofs:", qd_to_numpy(engine.integrator.info.segments_n_dofs))
    print("segments dof_start:", qd_to_numpy(engine.integrator.info.segments_dof_start))
    for i_step in range(2):
        engine.step_pipeline.run()
        print(f"step {i_step}: dofs_pos={qd_to_numpy(engine.integrator.state.dofs_pos)}")
        print(f"  cloth={qd_to_numpy(cloth.state.verts_pos)}")


if __name__ == "__main__":
    main()
