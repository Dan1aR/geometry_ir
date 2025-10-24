"""Initial guess generation for the CAD solver."""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

from .model import CadConstraint, DerivationPlan, Model, PointName
from .utils import edge_key

logger = logging.getLogger(__name__)


def _parse_edge_token(token: str) -> Optional[Tuple[str, str]]:
    if "-" not in token:
        return None
    parts = token.split("-", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _collect_length_hints(model: Model) -> Dict[Tuple[str, str], float]:
    """Extract numeric length hints from the CAD model."""

    hints: Dict[Tuple[str, str], float] = {}
    for constraint in model.constraints:
        if constraint.value is None:
            continue
        if constraint.kind in {"segment_length", "distance"} and constraint.entities:
            edge = _parse_edge_token(constraint.entities[0])
            if edge:
                hints[edge_key(*edge)] = float(constraint.value)
        elif constraint.kind == "point_on_circle" and len(constraint.entities) >= 2:
            center, point = constraint.entities[:2]
            hints[edge_key(center, point)] = float(constraint.value)
    for spec in model.circles.values():
        if spec.radius_value is not None and spec.radius_point is not None:
            hints[edge_key(spec.center, spec.radius_point)] = float(spec.radius_value)
    return hints


def _typical_length(hints: Dict[Tuple[str, str], float]) -> float:
    if not hints:
        return 1.0
    values = list(hints.values())
    return sum(values) / len(values)


def _position_third_point(
    anchor_length: Optional[float],
    baseline_length: Optional[float],
    baseline_span: float,
    default_scale: float,
) -> Tuple[float, float]:
    if anchor_length is not None and baseline_length is not None and baseline_span > 0:
        d = baseline_span
        r1 = anchor_length
        r2 = baseline_length
        x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        term = r1 * r1 - x * x
        if term >= 0:
            y = math.sqrt(term)
            return x, y if y > 1e-6 else default_scale * 0.5
    return default_scale * 0.35, default_scale * 0.8


def _fallback_position(idx: int, total: int, scale: float) -> Tuple[float, float]:
    angle = 2.0 * math.pi * (idx - 2) / max(4, total)
    radius = scale * (1.2 + 0.15 * (idx - 2))
    return radius * math.cos(angle), radius * math.sin(angle)


def _base_positions(model: Model) -> Dict[PointName, Tuple[float, float]]:
    order = model.point_order
    positions: Dict[PointName, Tuple[float, float]] = {}
    if not order:
        return positions

    hints = _collect_length_hints(model)
    scale = _typical_length(hints)
    anchor = order[0]
    positions[anchor] = (0.0, 0.0)

    baseline_length = scale
    if len(order) >= 2:
        baseline = order[1]
        baseline_length = hints.get(edge_key(anchor, baseline), scale)
        positions[baseline] = (baseline_length, 0.0)

    if len(order) >= 3:
        third = order[2]
        anchor_length = hints.get(edge_key(anchor, third))
        baseline_length_hint = None
        if len(order) >= 2:
            baseline_length_hint = hints.get(edge_key(order[1], third))
        x, y = _position_third_point(
            anchor_length,
            baseline_length_hint,
            baseline_length,
            scale,
        )
        positions[third] = (x, y)

    for idx, name in enumerate(order[3:], start=3):
        positions[name] = _fallback_position(idx, len(order), scale)

    logger.info(
        "Constructed base initial positions for %d points using %d length hints",
        len(order),
        len(hints),
    )
    return positions


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


def _ensure_gauge_constraints(model: Model) -> None:
    if getattr(model, "_gauge_applied", False):
        return

    gauge_specs = model.gauge_points
    if not gauge_specs:
        gauge_specs = []
        if model.point_order:
            gauge_specs.append((model.point_order[0], "fixed origin"))
        if len(model.point_order) >= 2:
            gauge_specs.append((model.point_order[1], "fixed baseline"))

    for point, note in gauge_specs:
        entity = model.point_entity(point)
        model.system.dragged(entity, model.workplane)
        cad_id = model.reserve_constraint_id()
        constraint = CadConstraint(
            cad_id=cad_id,
            kind="dragged",
            entities=(point,),
            value=None,
            source=None,
            note=note,
        )
        model.constraints.append(constraint)
        logger.info(
            "Registered gauge constraint #%d for point=%s note=%s",
            cad_id,
            point,
            note,
        )

    setattr(model, "_gauge_applied", True)


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

    _ensure_gauge_constraints(model)
