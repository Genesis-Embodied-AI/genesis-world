# Integrator protocol guide: `Integrator.on_integrate`

Companion guide for the `on_integrate` protocol in [integrator_engine.py](integrator_engine.py). The protocol docstring
points here, and this file is the contract a participant System implements.

## Model

The Integrator owns two global vectors of general degrees of freedom (DOFs), the positions $q$
(`IntegratorState.dofs_pos`) and the velocities $\dot q$ (`IntegratorState.dofs_vel`), of length

$$N = \sum_{p} n_p$$

where $n_p$ is the number of DOFs participant $p$ reports. Participant $p$ owns the contiguous range
$[o_p,\ o_p + n_p)$ with

$$o_p = \sum_{p' \prec p} n_{p'}$$

where $\prec$ is the frozen registration order (explicit `rank` first, then add order). `IntegratorInfo` stores $n_p$ in
`segments_n_dofs` and $o_p$ in `segments_dof_start`. Each step advances every DOF with the explicit Euler method [1]:

$$q_{i_d} \leftarrow q_{i_d} + \Delta t \, \dot q_{i_d}, \qquad i_d \in [0, N)$$

A participant reaches the global vectors only through a `segment` (`IntegratorSegment`), a view of its own range. Its
DOF index `i_d_` is relative to the start of that range, so the global index is $i_d = o_p + i_d'$. How a participant
lays out its own state in its range is its own choice: Rigid maps rigid DOF $i_d'$ to DOF $i_d'$, and Cloth maps
component $j$ of vertex $i_v$ to DOF $3 i_v + j$.

| Segment method | Meaning |
| --- | --- |
| `segment.set_n_dofs(n_dofs)` | report $n_p$ = `n_dofs` |
| `segment.set_dof_position(i_d_, position)` | write $q_{o_p+i_d'}$, for $i_d' \in [0, n_p)$ |
| `segment.set_dof_velocity(i_d_, velocity)` | write $\dot q_{o_p+i_d'}$, for $i_d' \in [0, n_p)$ |
| `segment.get_dof_position(i_d_)` | read $q_{o_p+i_d'}$, for $i_d' \in [0, n_p)$ |

## Registering

Call once from your `build()`, with all four Actions:

```python
self.integrator.on_integrate(init=self.init, count=self.count, gather=self.gather, scatter=self.scatter, rank=0)
```

One call registers one complete record, so the four Actions share one slot. `rank: int | None = None` is the position
of this participant in the Integrator. An `int >= 0` places it before every unranked participant, sorted by rank, and
each rank is held by one participant at most. `None` keeps the add order. The rank is per manager.

## Order of invocation

```text
init pipeline   init (all) -> count (all) -> offsets o_p derived, dofs_pos / dofs_vel allocated
every step      gather (all) -> dofs_pos += dt * dofs_vel -> scatter (all)
```

## Actions

### `init: HostAction`

| | |
| --- | --- |
| Signature | `function(*bound_args)`, e.g. `return init_rigid, self.state, self._scene` |
| Invoked | once, from the host function of the init pipeline, before counting |
| Must | allocate and fill the Data of this participant, so `count` can read its size |
| Must not | touch the Data of the Integrator, which is allocated afterwards |

### `count: InlineAction`

| | |
| --- | --- |
| Signature | `function(*bound_args, segment)`, e.g. `return func_count_rigid_dofs, self.state` |
| Invoked | once, inside one serial device loop over all participants, after every `init` |
| Must | call `segment.set_n_dofs(n_dofs)` exactly once, with `n_dofs` = $n_p$ |
| Must not | loop over its DOFs or launch work, since it is inlined into the loop of the Integrator |

### `gather: StageAction`

- **Signature:** `function(*bound_args, segment)`, e.g. `return func_gather_rigid_dofs, self.state`.
- **Invoked:** every step, at the top level of the step kernel of the Engine, before integration.
- **Must:** for $i_d' \in [0, n_p)$, call `segment.set_dof_position(i_d_, position)` and
  `segment.set_dof_velocity(i_d_, velocity)`.
- **Must not:** write outside $[0, n_p)$ or call `segment.get_dof_position`, since positions are overwritten during
  this stage.

### `scatter: StageAction`

- **Signature:** `function(*bound_args, segment)`, e.g. `return func_scatter_rigid_dofs, self.state`.
- **Invoked:** every step, at the top level of the step kernel of the Engine, after integration.
- **Must:** for $i_d' \in [0, n_p)$, read `segment.get_dof_position(i_d_)` back into the Data of this participant.
- **Must not:** call `segment.set_dof_position` or `segment.set_dof_velocity`, since the integrated state is read-only
  here.

## Rejected at registration

- An Action of the wrong kind for its parameter, for example an `InlineAction` passed as `gather`.
- A function whose parameters after the bound arguments differ from `(segment,)` for `count`, `gather` and `scatter`,
  or from `()` for `init`.
- A rank another participant of the Integrator already holds.

## References

1. Euler method. <https://en.wikipedia.org/wiki/Euler_method>
