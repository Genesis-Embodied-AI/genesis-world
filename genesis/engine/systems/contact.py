from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

CONTACT_CONFIG_DEFAULTS = MappingProxyType(
    {
        "contact/enable": 1,
        "contact/d_hat": 0.01,
        "contact/max_step_in_d_hat": -1.0,
        "contact/ccd_bound": "directional",
        "contact/adaptive_kappa_mode": "per-body",
        "contact/adaptive_kappa_tick": "newton",
        "contact/init_collision_pair_capacity": 1_000,
        "contact/ccd_partition": 1,
        "contact/ccd_partition_sv_max_iter": 64,
        "contact/intersection_check": 0,
        "contact/intersection_check_capacity": 1_024,
        "contact/constitution": "auto",
        "friction/eps_v": 1e-2,
        "linear_system/tol_rate": 1e-4,
        "extras/capacity_grow_factor": 1.2,
        "extras/capacity_shrink_threshold": 0.8,
        "extras/pipeline/genesis_serial": 0,
        "extras/ls_forensics/test_energy_bias": 0.0,
        "topo/grow_factor": 1.5,
        "bvh/type": "info_lbvh_batched_dop14",
        "bvh/pt_query": "warp",
        "bvh/ee_query": "dual",
        "bvh/dual/frontier_levels": 0,
        "bvh/dual/target_waves": 24.0,
        "bvh/dual/max_levels": 18,
        "rigid_proxy/globalization": "merit",
        "rigid_proxy/restoration": 1,
        "rigid_proxy/test_merit_energy_bias": 0.0,
        "rigid_forest/fused": 1,
        "extras/rigid_forest/genesis_legacy": 0,
        "extras/sort_reduce/genesis_legacy": 0,
        "extras/bvh/genesis_legacy_fp64_bounds": 0,
        "extras/bvh/genesis_legacy_refit": 0,
    }
)


@dataclass(frozen=True)
class ContactModel:
    element_a: int
    element_b: int
    friction_rate: float
    resistance: float
    enable: bool
    enable_ee: bool


class ContactElement:
    __slots__ = ("_id", "_name")

    def __init__(self, id: int, name: str) -> None:
        self._id = id
        self._name = name

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name


class ContactTabular:
    """CGQ contact-element registry and ordinary pair model table."""

    def __init__(self) -> None:
        self._elements = [ContactElement(0, "default")]
        self._models = {
            (0, 0): ContactModel(
                element_a=0,
                element_b=0,
                friction_rate=0.05,
                resistance=1e4,
                enable=True,
                enable_ee=True,
            )
        }

    def create(self, name: str = "") -> ContactElement:
        element = ContactElement(len(self._elements), name)
        self._elements.append(element)
        return element

    def insert(
        self,
        a: ContactElement,
        b: ContactElement,
        *,
        friction_rate: float,
        resistance: float,
        enable: bool = True,
        enable_ee: bool | None = None,
    ) -> None:
        key = (min(a.id, b.id), max(a.id, b.id))
        self._models[key] = ContactModel(
            element_a=key[0],
            element_b=key[1],
            friction_rate=float(friction_rate),
            resistance=float(resistance),
            enable=bool(enable),
            enable_ee=bool(enable if enable_ee is None else enable_ee),
        )

    def default_model(
        self,
        friction_rate: float,
        resistance: float,
        enable: bool = True,
        *,
        enable_ee: bool | None = None,
    ) -> None:
        default = self._elements[0]
        self.insert(
            default,
            default,
            friction_rate=friction_rate,
            resistance=resistance,
            enable=enable,
            enable_ee=enable_ee,
        )

    def at(self, i: int, j: int) -> ContactModel:
        return self._models.get((min(i, j), max(i, j)), self._models[(0, 0)])

    def default_element(self) -> ContactElement:
        return self._elements[0]
