import pytest
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32", "64"])
def test_get_contacts_applies_the_contact_permutation(precision, show_viewer):
    """A resting box on a plane queries its contacts through the entity-level API and the collider's four data
    accessors. With the zero-copy contact views active the query applies the contact permutation with
    torch.gather, whose index must be int64 although contact_sort_idx is a 32-bit field.
    """
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.2)))
    scene.build()

    for _ in range(30):
        scene.step()

    solver = scene.sim.rigid_solver
    n_contacts = int(qd_to_numpy(solver.collider.collider_state.n_contacts).reshape(-1)[0])
    assert n_contacts > 0

    # The collider returns the same contacts through every (as_tensor, to_torch) accessor: the int fields carry
    # no -1 sentinel in the live range, and the positions sit on the plane whatever index the gather applied.
    for as_tensor in (False, True):
        for to_torch in (False, True):
            contacts = solver.collider.get_contacts(as_tensor, to_torch)
            geom_a, position, normal = contacts["geom_a"], contacts["position"], contacts["normal"]
            assert len(geom_a) == n_contacts
            assert (geom_a != -1).all() and (contacts["geom_b"] != -1).all()
            if not to_torch:
                position, normal = torch.as_tensor(position), torch.as_tensor(normal)
            assert torch.isfinite(position).all() and torch.isfinite(normal).all()
            assert_allclose(position[:, 2], torch.zeros_like(position[:, 2]), atol=1e-3)
            assert_allclose(torch.abs(normal[:, 2]), torch.ones_like(normal[:, 2]), atol=1e-3)

    # The entity-level API returns exactly the pairs that involve the box, one geom of each pair in it.
    contacts = box.get_contacts()
    assert len(contacts["geom_a"]) == n_contacts
    in_box = {
        key: (contacts[key] >= box.geom_start) & (contacts[key] < box.geom_end)
        for key in ("geom_a", "geom_b")
    }
    assert (in_box["geom_a"] ^ in_box["geom_b"]).all()
    assert_allclose(contacts["position"][:, 2], torch.zeros(n_contacts), atol=1e-3)

    # The padded accessor returns the fixed capacity with 'valid_mask' marking the live contacts.
    padded = box.get_contacts(is_padded=True)
    assert padded["geom_a"].shape[0] >= n_contacts
    assert padded["valid_mask"].sum() == n_contacts
