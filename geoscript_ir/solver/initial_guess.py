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
    seed_hints = getattr(model, "seed_hints", None)
    if isinstance(seed_hints, dict):
        for hint in seed_hints.get("global_hints", []):
            if hint.get("kind") != "length":
                continue
            payload = hint.get("payload", {})
            edge = payload.get("edge")
            value = payload.get("value")
            if (
                isinstance(edge, (list, tuple))
                and len(edge) == 2
                and isinstance(value, (int, float))
            ):
                a, b = map(str, edge)
                hints[edge_key(a, b)] = float(value)
        for point, entries in seed_hints.get("by_point", {}).items():
            if not isinstance(entries, (list, tuple)):
                continue
            for hint in entries:
                if hint.get("kind") != "length":
                    continue
                payload = hint.get("payload", {})
                other = payload.get("other")
                value = payload.get("value")
                if isinstance(other, str) and isinstance(value, (int, float)):
                    hints[edge_key(str(point), other)] = float(value)
    return hints


def _typical_length(hints: Dict[Tuple[str, str], float]) -> float:
    if not hints:
        return 1.0
    values = list(hints.values())
    return sum(values) / len(values)


def _default_gauge_points(
    model: Model,
) -> Tuple[Optional[PointName], Optional[PointName], Optional[float]]:
    origin: Optional[PointName] = None
    baseline: Optional[PointName] = None
    length_hint: Optional[float] = None

    gauge_meta = model.metadata.get("default_gauge") if isinstance(model.metadata, dict) else None
    if isinstance(gauge_meta, dict):
        origin = gauge_meta.get("origin")
        baseline = gauge_meta.get("baseline")
        length_value = gauge_meta.get("length")
        if isinstance(length_value, (int, float)):
            length_hint = float(length_value)

    if origin is None and model.gauge_points:
        origin = model.gauge_points[0][0]
    if baseline is None and model.gauge_points:
        for point, _ in model.gauge_points[1:]:
            if point != origin:
                baseline = point
                break

    if origin is None and model.point_order:
        origin = model.point_order[0]
    if baseline is None and model.point_order:
        for name in model.point_order:
            if name != origin:
                baseline = name
                break

    return origin, baseline, length_hint


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
    origin, baseline, gauge_length = _default_gauge_points(model)
    if origin is None:
        origin = order[0]
    positions[origin] = (0.0, 0.0)

    baseline_length = scale
    if baseline and baseline != origin:
        length_hint = hints.get(edge_key(origin, baseline))
        if length_hint is None and gauge_length is not None:
            length_hint = gauge_length
        if length_hint is not None:
            baseline_length = float(length_hint)
        positions[baseline] = (baseline_length, 0.0)
    else:
        baseline = None

    third: Optional[PointName] = None
    gauge_meta = model.metadata.get("default_gauge") if isinstance(model.metadata, dict) else None
    if isinstance(gauge_meta, dict):
        candidate = gauge_meta.get("third")
        if isinstance(candidate, str):
            third = candidate
    if third is None:
        for name in order:
            if name not in positions:
                third = name
                break

    if third is not None and third not in positions:
        anchor_length = hints.get(edge_key(origin, third)) if origin else None
        baseline_length_hint = None
        if baseline:
            baseline_length_hint = hints.get(edge_key(baseline, third))
        baseline_span = baseline_length if baseline else scale
        x, y = _position_third_point(
            anchor_length,
            baseline_length_hint,
            baseline_span,
            scale,
        )
        positions[third] = (x, y)

    total = len(order)
    fallback_idx = 3
    for name in order:
        if name in positions:
            continue
        positions[name] = _fallback_position(fallback_idx, total, scale)
        fallback_idx += 1

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

    origin, baseline, gauge_length = _default_gauge_points(model)
    hints = _collect_length_hints(model)
    notes = {point: note for point, note in model.gauge_points if note}

    def _note(point: PointName, fallback: str) -> str:
        return notes.get(point, fallback)

    def _drag_point(
        point: PointName,
        target: Tuple[float, float],
        *,
        note: str,
        value: Optional[float] = None,
    ) -> None:
        entity = model.point_entity(point)
        model.system.set_params(entity.params, [target[0], target[1]])
        model.system.dragged(entity, model.workplane)
        cad_id = model.reserve_constraint_id()
        constraint = CadConstraint(
            cad_id=cad_id,
            kind="dragged",
            entities=(point,),
            value=value,
            source=None,
            note=note,
        )
        model.constraints.append(constraint)
        logger.info(
            "Registered gauge constraint #%d for point=%s note=%s value=%s",
            cad_id,
            point,
            note,
            "{:.6f}".format(value) if value is not None else None,
        )

    handled: set[PointName] = set()

    if origin is not None:
        _drag_point(origin, (0.0, 0.0), note=_note(origin, "fixed origin"))
        handled.add(origin)

    if baseline and baseline not in handled:
        length_value: Optional[float] = None
        if origin is not None:
            length_value = hints.get(edge_key(origin, baseline))
        if length_value is None and gauge_length is not None:
            length_value = gauge_length

        if length_value is not None:
            target_x = float(abs(length_value))
            baseline_note = f"{_note(baseline, 'fixed baseline')} (length={length_value:g})"
            _drag_point(
                baseline,
                (target_x, 0.0),
                note=baseline_note,
                value=float(length_value),
            )
        else:
            baseline_note = f"{_note(baseline, 'fixed baseline')} (unit length)"
            _drag_point(baseline, (1.0, 0.0), note=baseline_note)
        handled.add(baseline)

    for point, note in model.gauge_points:
        if point in handled:
            continue
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
