from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from .types import (
    DerivationPlan,
    SeedHints,
)
from ..ast import Program


def build_seed_hints(program: Program, plan: Optional[DerivationPlan]) -> SeedHints:
    """Construct per-point and global hints for the solver seed."""

    return SeedHints(...)


__all__ = [
    "build_seed_hints",
]
