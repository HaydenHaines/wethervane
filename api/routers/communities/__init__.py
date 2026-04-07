"""Communities and types router package."""
from fastapi import APIRouter

from api.routers.communities.legacy import router as _legacy_router
from api.routers.communities.types import router as _types_router

router = APIRouter()
router.include_router(_legacy_router)
router.include_router(_types_router)

# Re-export for backward compatibility
from api.routers.communities.legacy import (  # noqa: E402, F401
    get_community,
    list_communities,
)
from api.routers.communities.types import (  # noqa: E402, F401
    get_correlated_types,
    get_type,
    get_type_scatter_data,
    list_super_types,
    list_types,
)
