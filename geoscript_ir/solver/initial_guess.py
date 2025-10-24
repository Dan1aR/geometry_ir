"""Initial guess generation for the CAD solver."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from .model import DerivationPlan, Model, PointName
from .utils import initial_point_positions

logger = logging.getLogger(__name__)


def _base_positions(model: Model) -> Dict[PointName, Tuple[float, float]]:
    if model.initial_positions:
        logger.info(
            "Using %d stored initial positions for guess", len(model.initial_positions)
        )
        return dict(model.initial_positions)
    logger.info("Falling back to default initial positions for %d points", len(model.point_order))
    return initial_point_positions(model.point_order)


def initial_guess(
    model: Model,
    rng: np.random.Generator,
    attempt: int,
    *,
    plan: Optional[DerivationPlan] = None,
) -> np.ndarray:
    """Produce a simple initial guess for the CAD solver."""

    positions = _base_positions(model)
    guess = np.zeros(2 * len(model.point_order), dtype=float)
    jitter_scale = 0.0 if attempt <= 0 else 0.05 * attempt

    logger.info(
        "Constructing initial guess attempt=%d jitter_scale=%.3f for %d points",
        attempt,
        jitter_scale,
        len(model.point_order),
    )

    for idx, name in enumerate(model.point_order):
        x, y = positions.get(name, (0.0, 0.0))
        if jitter_scale > 0.0:
            jitter = rng.normal(loc=0.0, scale=jitter_scale, size=2)
            x += float(jitter[0])
            y += float(jitter[1])
        guess[2 * idx] = x
        guess[2 * idx + 1] = y

    return guess


def apply_initial_guess(model: Model, guess: np.ndarray) -> None:
    """Write ``guess`` into the underlying solver system."""

    if guess.shape[0] != 2 * len(model.point_order):  # pragma: no cover - defensive guard
        raise ValueError("Initial guess length does not match number of points")

    logger.info("Applying initial guess to solver system for %d points", len(model.point_order))
    for idx, name in enumerate(model.point_order):
        x = float(guess[2 * idx])
        y = float(guess[2 * idx + 1])
        entity = model.point_entity(name)
        model.system.set_params(entity.params, [x, y])
