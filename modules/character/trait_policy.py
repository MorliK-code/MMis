from __future__ import annotations

from modules.character.trait_limits import (
    DEFAULT_TRAIT_SOFT_CAPS,
    TRAIT_SOFT_CAPS,
    _apply_trait_soft_cap,
    apply_trait_delta,
    clamp_trait_map,
    clamp_trait_scalar,
    normalize_trait_name,
    normalize_trait_record,
    normalize_trait_value,
    resolve_trait_bounds,
    trait_soft_cap,
)

__all__ = [
    "DEFAULT_TRAIT_SOFT_CAPS",
    "TRAIT_SOFT_CAPS",
    "_apply_trait_soft_cap",
    "apply_trait_delta",
    "clamp_trait_map",
    "clamp_trait_scalar",
    "normalize_trait_name",
    "normalize_trait_record",
    "normalize_trait_value",
    "resolve_trait_bounds",
    "trait_soft_cap",
]
