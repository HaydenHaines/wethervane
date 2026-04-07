"""Senate forecast router package.

Re-exports a merged ``router`` so that ``from api.routers import senate``
followed by ``senate.router`` works identically to the old single-file module.

All public symbols from the old monolithic senate.py are re-exported here
so existing test imports continue to work unchanged.
"""
from fastapi import APIRouter

from api.routers.senate._helpers import (  # noqa: F401 — re-exports for tests
    _CLASS_II_INCUMBENT,
    _DEFAULT_SAFE_MARGIN,
    _DEM_CLASS_II_COUNT,
    _DEM_HOLDOVER_SEATS,
    _GOP_CLASS_II_COUNT,
    _GOP_HOLDOVER_SEATS,
    _TOSSUP_MAX,
    DEM_SAFE_SEATS,
    GOP_SAFE_SEATS,
    SENATE_2026_STATES,
    SENATE_DELEGATION,
    _build_headline,
    _compute_baseline_label,
    _margin_to_rating,
    _rating_sort_key,
    _rating_to_zone,
)
from api.routers.senate.overview import router as _overview_router
from api.routers.senate.scrolly import router as _scrolly_router
from api.routers.senate.simulation import (  # noqa: F401 — re-export for tests
    _simulate_chamber_probability,
)
from api.routers.senate.simulation import router as _simulation_router

router = APIRouter()
router.include_router(_overview_router)
router.include_router(_simulation_router)
router.include_router(_scrolly_router)
