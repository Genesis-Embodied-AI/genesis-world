import quadrants as qd

import genesis as gs

from .base_solver import Solver, TimeBasedMixin


@qd.data_oriented
class SFSolver(TimeBasedMixin, Solver):
    """
    Stable Fluid solver for eulerian-based gaseous simulation.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        if options is None:
            return

        self.n_grid = options.res
        self.dx = 1 / self.n_grid
        self.res = (self.n_grid, self.n_grid, self.n_grid)
        self.solver_iters = options.solver_iters
        self.decay = options.decay

        self.t = 0.0
        self.inlet_s = options.inlet_s

        self.jets = ()

    def setup_fields(self):
        assert self.jets

        cell_state = qd.types.struct(
            v=gs.qd_vec3,
            v_tmp=gs.qd_vec3,
            div=gs.qd_float,
            p=gs.qd_float,
            q=qd.types.vector(len(self.jets), gs.qd_float),
        )

        self.grid = cell_state.field(shape=self.res, layout=qd.Layout.SOA)

        # swap area for pressure projection solver
        self.p_swap = TexPair(
            cur=qd.field(dtype=gs.qd_float, shape=self.res),
            nxt=qd.field(dtype=gs.qd_float, shape=self.res),
        )

    @qd.kernel
    def init_fields(self):
        for I in qd.grouped(qd.ndrange(*self.res)):
            for q in qd.static(range(self.grid.q.n)):
                self.grid.q[I][q] = 0.0

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        if self.is_active:
            self.t = 0.0
            self.setup_fields()
            self.init_fields()

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return bool(self.jets)

    def set_jets(self, jets):
        assert isinstance(jets, (list, tuple))
        self.jets = tuple(jets)

    def reset_swap(self):
        self.p_swap.cur.fill(0)
        self.p_swap.nxt.fill(0)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def pressure_jacobi(self, pf: qd.template(), new_pf: qd.template()):
        for u, v, w in qd.ndrange(*self.res):
            pl = pf[self.compute_location(u, v, w, -1, 0, 0)]
            pr = pf[self.compute_location(u, v, w, 1, 0, 0)]
            pb = pf[self.compute_location(u, v, w, 0, -1, 0)]
            pt = pf[self.compute_location(u, v, w, 0, 1, 0)]
            pp = pf[self.compute_location(u, v, w, 0, 0, -1)]
            pq = pf[self.compute_location(u, v, w, 0, 0, 1)]

            new_pf[u, v, w] = (pl + pr + pb + pt + pp + pq - self.grid[u, v, w].div) / 6.0

    @qd.kernel
    def advect_and_impulse(self, f: qd.i32, t: qd.f32):
        for i, j, k in qd.ndrange(*self.res):
            p = qd.Vector([i, j, k], dt=gs.qd_float) + 0.5
            p = self.backtrace(self.grid.v, p, self.substep_dt)
            v_tmp = self.trilerp(self.grid.v, p)

            for q in qd.static(range(self.grid.q.n)):
                q_f = self.trilerp_scalar(self.grid.q, p, q)

                imp_dir = self.jets[q].get_tan_dir(t)
                factor = self.jets[q].get_factor(i, j, k, self.dx, t)
                momentum = (imp_dir * self.inlet_s * factor) * self.substep_dt

                v_tmp += momentum

                self.grid.q[i, j, k][q] = (1 - factor) * q_f + factor
                # self.grid.q[i, j, k][q] *= self.decay
                self.grid.q[i, j, k][q] -= self.decay * self.substep_dt
                self.grid.q[i, j, k][q] = max(0.0, self.grid.q[i, j, k][q])

            self.grid.v_tmp[i, j, k] = v_tmp

    @qd.kernel
    def divergence(self):
        for u, v, w in qd.ndrange(*self.res):
            vl = self.grid.v_tmp[self.compute_location(u, v, w, -1, 0, 0)]
            vr = self.grid.v_tmp[self.compute_location(u, v, w, 1, 0, 0)]
            vb = self.grid.v_tmp[self.compute_location(u, v, w, 0, -1, 0)]
            vt = self.grid.v_tmp[self.compute_location(u, v, w, 0, 1, 0)]
            vp = self.grid.v_tmp[self.compute_location(u, v, w, 0, 0, -1)]
            vq = self.grid.v_tmp[self.compute_location(u, v, w, 0, 0, 1)]
            vc = self.grid.v_tmp[self.compute_location(u, v, w, 0, 0, 0)]

            if not self.is_free(u, v, w, -1, 0, 0):
                vl.x = -vc.x
            if not self.is_free(u, v, w, 1, 0, 0):
                vr.x = -vc.x
            if not self.is_free(u, v, w, 0, -1, 0):
                vb.y = -vc.y
            if not self.is_free(u, v, w, 0, 1, 0):
                vt.y = -vc.y
            if not self.is_free(u, v, w, 0, 0, -1):
                vp.z = -vc.z
            if not self.is_free(u, v, w, 0, 0, 1):
                vq.z = -vc.z

            self.grid.div[u, v, w] = 0.5 * (vr.x - vl.x + vt.y - vb.y + vq.z - vp.z)

    @qd.kernel
    def pressure_to_swap(self):
        for I in qd.grouped(qd.ndrange(*self.res)):
            self.p_swap.cur[I] = self.grid.p[I]

    @qd.kernel
    def pressure_from_swap(self):
        for I in qd.grouped(qd.ndrange(*self.res)):
            self.grid.p[I] = self.p_swap.cur[I]

    @qd.kernel
    def subtract_gradient(self):
        for I in qd.grouped(qd.ndrange(*self.res)):
            u, v, w = I
            pl = self.grid.p[self.compute_location(u, v, w, -1, 0, 0)]
            pr = self.grid.p[self.compute_location(u, v, w, 1, 0, 0)]
            pb = self.grid.p[self.compute_location(u, v, w, 0, -1, 0)]
            pt = self.grid.p[self.compute_location(u, v, w, 0, 1, 0)]
            pp = self.grid.p[self.compute_location(u, v, w, 0, 0, -1)]
            pq = self.grid.p[self.compute_location(u, v, w, 0, 0, 1)]

            self.grid.v[I] = self.grid.v_tmp[I] - 0.5 * qd.Vector([pr - pl, pt - pb, pq - pp], dt=gs.qd_float)

    @qd.func
    def compute_location(self, u, v, w, du, dv, dw):
        I = qd.Vector([u + du, v + dv, w + dw], dt=gs.qd_int)
        return qd.math.clamp(I, 0, self.n_grid - 1)

    @qd.func
    def is_free(self, u, v, w, du, dv, dw):
        I = qd.Vector([u + du, v + dv, w + dw], dt=gs.qd_int)
        return gs.qd_bool((0 <= I).all() and (I < self.n_grid).all())

    @qd.func
    def trilerp_scalar(self, qf, p, qf_idx):
        """
        p: position, within (0, 1).
        qf: field for interpolation
        """
        # convert position to grid index
        base_I = qd.floor(p - 0.5, gs.qd_int)
        p_I = p - 0.5

        q = 0.0
        w_total = 0.0
        for offset in qd.static(qd.grouped(qd.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - qd.abs(p_I - grid_I)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[grid_I_][qf_idx]
            w_total += w
        # w_total is less then one when at boundary
        q /= w_total
        return q

    @qd.func
    def trilerp(self, qf, p):
        """
        p: position, within (0, 1).
        qf: field for interpolation
        """
        # convert position to grid index
        base_I = qd.floor(p - 0.5, gs.qd_int)
        p_I = p - 0.5

        q = qd.Vector([0.0, 0.0, 0.0], dt=gs.qd_float)
        w_total = 0.0
        for offset in qd.static(qd.grouped(qd.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - qd.abs(p_I - grid_I)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[grid_I_]
            w_total += w
        # w_total is less then one when at boundary
        q /= w_total
        return q

    # RK3
    @qd.func
    def backtrace(self, vf, p, dt):
        """
        vf: velocity field
        """
        v1 = self.trilerp(vf, p)
        p1 = p - 0.5 * dt * v1
        v2 = self.trilerp(vf, p1)
        p2 = p - 0.75 * dt * v2
        v3 = self.trilerp(vf, p2)
        p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
        return p

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward):
        return None

    def substep_pre_coupling(self, f):
        self.advect_and_impulse(f, self.t)
        self.divergence()

        # projection
        self.reset_swap()
        self.pressure_to_swap()
        for _ in range(self.solver_iters):
            self.pressure_jacobi(self.p_swap.cur, self.p_swap.nxt)
            self.p_swap.swap()
        self.pressure_from_swap()
        self.reset_swap()

        self.subtract_gradient()
        self.t += self.substep_dt

    def substep_post_coupling(self, f):
        return

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        pass

    def set_state(self, f, state, envs_idx=None):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur
