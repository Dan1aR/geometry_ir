"""CAD adapters for GeoScript IR."""

from .slvs_adapter import (
    AdapterFail,
    AdapterOK,
    AdapterResult,
    CadFailure,
    SlvsAdapter,
    SlvsAdapterOptions,
    CAD_MAPPING_TABLE,
    solve_equalities_safe,
)

__all__ = [
    "AdapterFail",
    "AdapterOK",
    "AdapterResult",
    "CadFailure",
    "SlvsAdapter",
    "SlvsAdapterOptions",
    "CAD_MAPPING_TABLE",
    "solve_equalities_safe",
]
