import math

import numpy as np

import genesis as gs


def test_jets(show_viewer):
    import quadrants as qd

    res = 384
    orbit_tau = 0.2
    orbit_radius = 0.3
    orbit_radius_vel = 0.0

    jet_radius = 0.02

    sub_orbit_radius = 0.03
    sub_orbit_tau = 3.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
        ),
        sf_options=gs.options.SFOptions(
            res=res,
            solver_iters=200,
            decay=0.025,
        ),
        show_viewer=show_viewer,
    )

    @qd.data_oriented
    class Jet:
        def __init__(
            self,
            world_center,
            jet_radius,
            orbit_radius,
            orbit_radius_vel,
            orbit_init_degree,
            orbit_tau,
            sub_orbit_radius,
            sub_orbit_tau,
        ):
            self.world_center = qd.Vector(world_center)
            self.orbit_radius = orbit_radius
            self.orbit_radius_vel = orbit_radius_vel
            self.orbit_init_radian = math.radians(orbit_init_degree)
            self.orbit_tau = orbit_tau

            self.jet_radius = jet_radius

            self.num_sub_jets = 3
            self.sub_orbit_radian_delta = 2.0 * math.pi / self.num_sub_jets
            self.sub_orbit_radius = sub_orbit_radius
            self.sub_orbit_tau = sub_orbit_tau

        @qd.func
        def get_pos(self, t: float):
            rel_pos = qd.Vector([self.orbit_radius + t * self.orbit_radius_vel, 0.0, 0.0])
            rot_mat = qd.math.rot_by_axis(qd.Vector([0.0, 1.0, 0.0]), self.orbit_init_radian + t * self.orbit_tau)[
                :3, :3
            ]
            rel_pos = rot_mat @ rel_pos
            return rel_pos

        @qd.func
        def get_factor(self, i: int, j: int, k: int, dx: float, t: float):
            rel_pos = self.get_pos(t)
            tan_dir = self.get_tan_dir(t)
            ijk = qd.Vector([i, j, k], dt=gs.qd_float) * dx
            dist = 2 * self.jet_radius
            for q in qd.static(range(self.num_sub_jets)):
                jet_pos = qd.Vector([0.0, self.sub_orbit_radius, 0.0])
                rot_mat = qd.math.rot_by_axis(tan_dir, self.sub_orbit_radian_delta * q + self.sub_orbit_tau * t)[:3, :3]
                jet_pos = (rot_mat @ jet_pos) + self.world_center + rel_pos
                dist_q = (ijk - jet_pos).norm(gs.EPS)
                if dist_q < dist:
                    dist = dist_q
            factor = 0.0
            if dist < self.jet_radius:
                factor = 1.0
            return factor

        @qd.func
        def get_inward_dir(self, t: float):
            neg_pos = -self.get_pos(t)
            return neg_pos.normalized(gs.EPS)

        @qd.func
        def get_tan_dir(self, t: float):
            inward_dir = self.get_inward_dir(t)
            tan_rot_mat = qd.math.rot_by_axis(qd.Vector([0.0, 1.0, 0.0]), 0.0)[:3, :3]
            return tan_rot_mat @ inward_dir

    jet = [
        Jet(
            world_center=[0.5, 0.5, 0.5],
            orbit_radius=orbit_radius,
            orbit_radius_vel=orbit_radius_vel,
            orbit_init_degree=orbit_init_degree,
            orbit_tau=orbit_tau,
            sub_orbit_radius=sub_orbit_radius,
            jet_radius=jet_radius,
            sub_orbit_tau=sub_orbit_tau,
        )
        for orbit_init_degree in np.linspace(0, 360, 3, endpoint=False)
    ]
    scene.sim.sf_solver.set_jets(jet)
    scene.build()
    scene.step()
