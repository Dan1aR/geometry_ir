"""TikZ renderer adhering to the GeoScript style contract."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .utils import latex_escape_keep_math
from ..ast import Program
from ..numbers import SymbolicNumber


standalone_tpl = r"""\documentclass[border=2pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc,angles,quotes,intersections,decorations.markings,arrows.meta,positioning}
\tikzset{
  %% global sizes (scale-aware; override per scene if needed)
  gs/dot radius/.store in=\gsDotR,       gs/dot radius=1.4pt,
  gs/line width/.store in=\gsLW,         gs/line width=0.8pt,
  gs/aux width/.store  in=\gsLWaux,      gs/aux width=0.6pt,
  gs/angle radius/.store in=\gsAngR,     gs/angle radius=8pt,
  gs/angle sep/.store   in=\gsAngSep,    gs/angle sep=2pt,
  gs/tick len/.store   in=\gsTick,       gs/tick len=4pt,
  point/.style={circle,fill=black,inner sep=0pt,minimum size=0pt},
  ptlabel/.style={font=\footnotesize, inner sep=1pt},
  carrier/.style={line width=\gsLW},
  circle/.style={line width=\gsLW},
  aux/.style={line width=\gsLWaux, dash pattern=on 3pt off 2pt},
  tick1/.style={postaction=decorate, decoration={markings,
      mark=at position 0.5 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);}}},
  tick2/.style={postaction=decorate, decoration={markings,
      mark=at position 0.4 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);},
      mark=at position 0.6 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);}}},
  tick3/.style={postaction=decorate, decoration={markings,
      mark=at position 0.35 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);},
      mark=at position 0.5  with {\draw (-\gsTick/2,0)--(\gsTick/2,0);},
      mark=at position 0.65 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);}}},
}
%% optional layers
\pgfdeclarelayer{bg}\pgfdeclarelayer{fg}\pgfsetlayers{bg,main,fg}
\begin{document}
%s
\end{document}
"""


@dataclass
class LabelSpec:
    """Label for either a point or a segment."""

    kind: str  # "point" or "side"
    target: Union[str, Tuple[str, str]]
    text: str
    position: Optional[str] = None
    slope: bool = False
    explicit: bool = False


@dataclass
class AuxPath:
    kind: str
    data: Dict[str, object] = field(default_factory=dict)


@dataclass
class RenderPlan:
    points: Dict[str, Tuple[float, float]]
    carriers: List[Tuple[str, str, Dict[str, object]]]
    aux_lines: List[Tuple[AuxPath, Dict[str, object]]]
    circles: List[Tuple[str, str, Dict[str, object]]]
    ticks: List[Tuple[str, str, int]]
    equal_angle_groups: List[List[Tuple[str, str, str]]]
    right_angles: List[Tuple[str, str, str]]
    angle_arcs: List[Tuple[str, str, str, Optional[float], Optional[str]]]
    labels: List[LabelSpec]
    tick_overflow_edges: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    helper_tick_edges: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)
    carrier_lookup: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)


def generate_tikz_document(
    program: Program,
    point_coords: Mapping[str, Tuple[float, float]],
    *,
    problem_text: Optional[str] = None,
    normalize: bool = False,
) -> str:
    """Render a standalone document using the minimal TikZ preamble."""

    header = ""
    if problem_text:
        header = (
            "\\noindent\\textbf{Problem:} "
            + latex_escape_keep_math(problem_text.strip())
            + "\\par\\vspace{4pt}\n"
        )
    tikz_code = generate_tikz_code(program, point_coords, normalize=normalize)
    return standalone_tpl % (header + tikz_code)


def generate_tikz_code(
    program: Program,
    point_coords: Mapping[str, Tuple[float, float]],
    *,
    normalize: bool = False,
) -> str:
    """Generate TikZ code that respects the rendering contract."""

    coords = _prepare_coordinates(point_coords, normalize=normalize)
    if not isinstance(program, Program):
        raise TypeError("program must be an instance of Program")

    layout_scale = _extract_layout_scale(program)
    rules = _extract_rules(program)
    plan = _build_render_plan(program, coords, rules)
    return _emit_tikz_picture(plan, layout_scale, rules)


# ---------------------------------------------------------------------------
# Render plan construction
# ---------------------------------------------------------------------------

def _prepare_coordinates(
    point_coords: Mapping[str, Tuple[float, float]], *, normalize: bool
) -> Dict[str, Tuple[float, float]]:
    coords: Dict[str, Tuple[float, float]] = {
        key: (float(value[0]), float(value[1])) for key, value in point_coords.items()
    }
    if not coords:
        return {}
    if not normalize:
        return coords
    xs = [pt[0] for pt in coords.values()]
    ys = [pt[1] for pt in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1e-9)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    scale = 8.0 / span
    return {
        key: ((pt[0] - cx) * scale, (pt[1] - cy) * scale) for key, pt in coords.items()
    }


def _extract_layout_scale(program: Program) -> float:
    for stmt in program.stmts:
        if stmt.kind == "layout":
            value = stmt.data.get("scale")
            if isinstance(value, (int, float)):
                return float(value)
    return 1.0


def _extract_rules(program: Program) -> Dict[str, bool]:
    rules: Dict[str, bool] = {
        "mark_right_angles_as_square": False,
        "no_equations_on_sides": False,
        "no_unicode_degree": False,
        "allow_auxiliary": True,
    }
    for stmt in program.stmts:
        if stmt.kind != "rules" or not stmt.opts:
            continue
        for key, value in stmt.opts.items():
            if isinstance(value, bool):
                rules[key] = value
    return rules

def _build_render_plan(
    program: Program,
    coords: Mapping[str, Tuple[float, float]],
    rules: Mapping[str, bool],
) -> RenderPlan:
    carriers: Dict[Tuple[str, str], Tuple[str, str, Dict[str, object]]] = {}
    carrier_lookup: Dict[Tuple[str, str], Tuple[str, str]] = {}
    circles: Dict[Tuple[str, str], Dict[str, object]] = {}
    aux_lines: List[Tuple[AuxPath, Dict[str, object]]] = []
    ticks: List[Tuple[str, str, int]] = []
    tick_overflow_edges: Dict[Tuple[str, str], bool] = {}
    helper_tick_edges: Dict[Tuple[str, str], Tuple[str, str]] = {}
    equal_angle_groups: List[List[Tuple[str, str, str]]] = []
    right_angles: List[Tuple[str, str, str]] = []
    angle_arcs: List[Tuple[str, str, str, Optional[float], Optional[str]]] = []
    point_labels: Dict[str, LabelSpec] = {}
    side_labels: List[LabelSpec] = []
    explicit_side_edges: Set[Tuple[str, str]] = set()
    auto_length_labels: Dict[Tuple[str, str], str] = {}

    tick_group = 0

    for stmt in program.stmts:
        kind = stmt.kind
        data = stmt.data
        opts = stmt.opts or {}

        if kind == "segment":
            edge = _edge_from_data(data.get("edge"))
            if not edge:
                continue
            key = _normalize_edge(edge)
            carriers.setdefault(key, (edge[0], edge[1], {}))
            carrier_lookup[key] = (edge[0], edge[1])
            if not rules.get("no_equations_on_sides", False):
                length_text = _extract_segment_length_text(opts)
                if length_text:
                    auto_length_labels[key] = length_text
        elif kind == "diameter":
            edge = _edge_from_data(data.get("edge"))
            if not edge:
                continue
            key = _normalize_edge(edge)
            carriers.setdefault(key, (edge[0], edge[1], {}))
            carrier_lookup[key] = (edge[0], edge[1])
        elif kind in {
            "triangle",
            "quadrilateral",
            "trapezoid",
            "parallelogram",
            "rectangle",
            "square",
            "rhombus",
            "polygon",
        }:
            ids = data.get("ids")
            if not isinstance(ids, (list, tuple)):
                continue
            for edge in _cycle_edges(tuple(ids)):
                key = _normalize_edge(edge)
                carriers.setdefault(key, (edge[0], edge[1], {}))
                carrier_lookup[key] = (edge[0], edge[1])
        elif kind == "ray":
            edge = _edge_from_data(data.get("ray"))
            if edge:
                aux_lines.append((AuxPath("ray", {"points": edge}), {}))
        elif kind == "line":
            edge = _edge_from_data(data.get("edge"))
            if edge:
                aux_lines.append((AuxPath("line", {"points": edge}), {}))
        elif kind == "perpendicular_at":
            if not rules.get("allow_auxiliary", True):
                continue
            to_edge = _edge_from_data(data.get("to"))
            at = data.get("at")
            if to_edge and isinstance(at, str):
                aux_lines.append(
                    (
                        AuxPath(
                            "perpendicular",
                            {"at": at, "to": to_edge, "foot": data.get("foot")},
                        ),
                        {},
                    )
                )
        elif kind == "parallel_through":
            if not rules.get("allow_auxiliary", True):
                continue
            to_edge = _edge_from_data(data.get("to"))
            through = data.get("through")
            if to_edge and isinstance(through, str):
                aux_lines.append(
                    (
                        AuxPath("parallel", {"through": through, "to": to_edge}),
                        {},
                    )
                )
        elif kind == "median_from_to":
            if not rules.get("allow_auxiliary", True):
                continue
            frm = data.get("frm")
            midpoint = data.get("midpoint")
            edge = _edge_from_data(data.get("to"))
            if edge and isinstance(frm, str) and isinstance(midpoint, str):
                aux_lines.append(
                    (
                        AuxPath(
                            "median",
                            {"frm": frm, "midpoint": midpoint, "base": edge},
                        ),
                        {},
                    )
                )
        elif kind == "circle_center_radius_through":
            center = data.get("center")
            through = data.get("through")
            if isinstance(center, str) and isinstance(through, str):
                circles.setdefault((center, through), {})
        elif kind == "label_point":
            point = data.get("point")
            if not isinstance(point, str):
                continue
            text = opts.get("label") if isinstance(opts.get("label"), str) else point
            pos = opts.get("pos") if isinstance(opts.get("pos"), str) else None
            point_labels[point] = LabelSpec(
                kind="point",
                target=point,
                text=text,
                position=pos,
                explicit=True,
            )
        elif kind == "sidelabel":
            edge = _edge_from_data(data.get("edge"))
            text = data.get("text")
            if not edge or not isinstance(text, str):
                continue
            pos = opts.get("pos") if isinstance(opts.get("pos"), str) else None
            explicit_side_edges.add(_normalize_edge(edge))
            side_labels.append(
                LabelSpec(
                    kind="side",
                    target=edge,
                    text=text,
                    position=pos,
                    slope=True,
                    explicit=True,
                )
            )
        elif kind == "equal_segments":
            groups = []
            lhs = data.get("lhs")
            rhs = data.get("rhs")
            if isinstance(lhs, (list, tuple)):
                groups.append([edge for edge in lhs if _edge_from_data(edge)])
            if isinstance(rhs, (list, tuple)):
                groups.append([edge for edge in rhs if _edge_from_data(edge)])
            for raw_group in groups:
                group_edges = [(_edge_from_data(edge)) for edge in raw_group]
                group_edges = [edge for edge in group_edges if edge]
                if not group_edges:
                    continue
                tick_group += 1
                style_index = ((tick_group - 1) % 3) + 1
                overflow = tick_group > 3
                for edge in group_edges:
                    a, b = edge  # type: ignore[misc]
                    ticks.append((a, b, style_index))
                    key = _normalize_edge((a, b))
                    if overflow:
                        tick_overflow_edges[key] = True
                    if key not in carriers:
                        helper_tick_edges.setdefault(key, (a, b))
        elif kind == "equal_angles":
            lhs = data.get("lhs")
            rhs = data.get("rhs")
            for group in (lhs, rhs):
                if not isinstance(group, (list, tuple)):
                    continue
                angles: List[Tuple[str, str, str]] = []
                for angle in group:
                    triple = _angle_triple(angle)
                    if triple:
                        angles.append(triple)
                if angles:
                    equal_angle_groups.append(angles)
        elif kind == "angle_at":
            triple = _angle_triple(data.get("points"))
            if not triple:
                continue
            label, numeric = _angle_label_from_opts(opts, rules)
            angle_arcs.append((triple[0], triple[1], triple[2], numeric, label))
        elif kind == "right_angle_at":
            triple = _angle_triple(data.get("points"))
            if triple:
                right_angles.append(triple)
        elif kind in {
            "target_angle",
            "target_length",
            "target_point",
            "target_circle",
            "target_arc",
        }:
            # Targets are currently ignored by the TikZ renderer.
            continue

    # Auto length labels (respect explicit sidelabel suppression)
    for key, text in auto_length_labels.items():
        if key in explicit_side_edges:
            continue
        if key not in carrier_lookup:
            continue
        a, b = carrier_lookup[key]
        side_labels.append(
            LabelSpec(
                kind="side",
                target=(a, b),
                text=text,
                position="above",
                slope=False,
                explicit=False,
            )
        )

    labels = list(point_labels.values()) + side_labels

    plan = RenderPlan(
        points=dict(coords),
        carriers=list(carriers.values()),
        aux_lines=aux_lines,
        circles=[(center, through, style) for (center, through), style in circles.items()],
        ticks=ticks,
        equal_angle_groups=equal_angle_groups,
        right_angles=right_angles,
        angle_arcs=angle_arcs,
        labels=labels,
    )
    plan.tick_overflow_edges = tick_overflow_edges
    plan.helper_tick_edges = helper_tick_edges
    plan.carrier_lookup = carrier_lookup
    return plan


def _edge_from_data(value: object) -> Optional[Tuple[str, str]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = value[0], value[1]
    if isinstance(a, str) and isinstance(b, str):
        return (a, b)
    return None


def _normalize_edge(edge: Tuple[str, str]) -> Tuple[str, str]:
    a, b = edge
    return (a, b) if a <= b else (b, a)


def _cycle_edges(ids: Tuple[str, ...]) -> Iterable[Tuple[str, str]]:
    if len(ids) < 2:
        return []
    for idx, current in enumerate(ids):
        nxt = ids[(idx + 1) % len(ids)]
        if isinstance(current, str) and isinstance(nxt, str):
            if current != nxt:
                yield (current, nxt)


def _angle_triple(value: object) -> Optional[Tuple[str, str, str]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    a, b, c = value[0], value[1], value[2]
    if all(isinstance(x, str) for x in (a, b, c)):
        return (a, b, c)
    return None


def _angle_label_from_opts(
    opts: Mapping[str, object], rules: Mapping[str, bool]
) -> Tuple[Optional[str], Optional[float]]:
    if not opts:
        return None, None
    if "degrees" in opts:
        text = _format_measurement_value(opts.get("degrees"))
        numeric = _coerce_float(opts.get("degrees"))
        if text:
            text = text.strip()
            degree_token = "^\\circ" if rules.get("no_unicode_degree", False) else "°"
            if text.startswith("$") and text.endswith("$"):
                inner = text[1:-1]
                if "\\circ" not in inner and "°" not in inner:
                    text = f"${inner}{degree_token}$"
                elif rules.get("no_unicode_degree", False):
                    text = text.replace("°", "^\\circ")
            else:
                if "\\circ" not in text and "°" not in text:
                    text = f"${text}{degree_token}$"
                else:
                    if rules.get("no_unicode_degree", False):
                        text = text.replace("°", "^\\circ")
                    if not text.startswith("$"):
                        text = f"${text}$"
        return text, numeric
    label = opts.get("label")
    if isinstance(label, str):
        return _format_label_text(label), None
    return None, None

def _emit_tikz_picture(plan: RenderPlan, layout_scale: float, rules: Mapping[str, bool]) -> str:
    lines: List[str] = []
    lines.append(f"\\begin{{tikzpicture}}[scale={_format_float(layout_scale)}]")

    if plan.points:
        for name in sorted(plan.points.keys()):
            x, y = plan.points[name]
            lines.append(
                f"  \\coordinate ({name}) at ({_format_float(x)}, {_format_float(y)});"
            )
        lines.append("")

    tick_map = _build_tick_map(plan.ticks)

    lines.append("  \\begin{pgfonlayer}{main}")
    for start, end, style in plan.carriers:
        if start not in plan.points or end not in plan.points:
            continue
        key = _normalize_edge((start, end))
        style_tokens = ["carrier"]
        for idx in tick_map.get(key, []):
            style_tokens.append(f"tick{idx}")
        if plan.tick_overflow_edges.get(key):
            style_tokens.append("densely dashed")
        lines.append(
            "    \\draw[{styles}] ({a}) -- ({b});".format(
                styles=", ".join(style_tokens), a=start, b=end
            )
        )
    for center, through, style in plan.circles:
        if center not in plan.points or through not in plan.points:
            continue
        radius = _distance(plan.points[center], plan.points[through])
        if radius <= 0:
            continue
        tokens = ["circle"]
        extra = style.get("extra") if isinstance(style, dict) else None
        if isinstance(extra, str) and extra:
            tokens.append(extra)
        lines.append(
            "    \\draw[{styles}] ({center}) circle ({radius});".format(
                styles=", ".join(tokens),
                center=center,
                radius=_format_float(radius),
            )
        )
    lines.append("  \\end{pgfonlayer}")
    lines.append("")

    span = _scene_span(plan.points.values())
    bbox = _coords_bbox(plan.points.values())
    point_label_map = {
        label.target: label for label in plan.labels if label.kind == "point"
    }
    side_label_specs = [label for label in plan.labels if label.kind == "side"]

    lines.append("  \\begin{pgfonlayer}{fg}")
    for path, style in plan.aux_lines:
        aux_lines = _emit_aux_path(path, plan.points, span, style)
        for entry in aux_lines:
            lines.append("    " + entry)

    for key, oriented in plan.helper_tick_edges.items():
        if oriented[0] not in plan.points or oriented[1] not in plan.points:
            continue
        tokens = ["aux"]
        for idx in tick_map.get(key, []):
            tokens.append(f"tick{idx}")
        if plan.tick_overflow_edges.get(key):
            tokens.append("densely dashed")
        lines.append(
            "    \\draw[{styles}] ({a}) -- ({b});".format(
                styles=", ".join(tokens), a=oriented[0], b=oriented[1]
            )
        )

    for group_index, group in enumerate(plan.equal_angle_groups, start=1):
        for A, B, C in group:
            if not _points_present((A, B, C), plan.points):
                continue
            for arc_idx in range(group_index):
                radius_expr = f"\\gsAngR+{arc_idx}*\\gsAngSep"
                lines.append(
                    "    \\path pic[draw, angle radius={radius}] {{{body}}};".format(
                        radius=radius_expr, body=f"angle={A}--{B}--{C}"
                    )
                )

    for A, B, C in plan.right_angles:
        if not _points_present((A, B, C), plan.points):
            continue
        lines.append(
            f"    \\path pic[draw, angle radius=\\gsAngR] {{right angle={A}--{B}--{C}}};"
        )

    for A, B, C, degrees, label in plan.angle_arcs:
        if not _points_present((A, B, C), plan.points):
            continue
        options = ["draw", "angle radius=\\gsAngR"]
        if label:
            options.append(f"\"{label}\"{{scale=0.9}}")
        start, end = A, C
        orientation = _oriented_angle_degrees(plan.points, A, B, C)
        swapped_orientation = _oriented_angle_degrees(plan.points, C, B, A)
        if orientation is not None and swapped_orientation is not None:
            if degrees is not None:
                target = _normalise_angle_degrees(degrees)
                candidates = [
                    (A, C, _normalise_angle_degrees(orientation)),
                    (C, A, _normalise_angle_degrees(swapped_orientation)),
                ]
                best = min(
                    candidates,
                    key=lambda item: _angular_difference_degrees(item[2], target),
                )
                start, end = best[0], best[1]
            else:
                tol = 1e-4
                candidates = [
                    (A, C, orientation),
                    (C, A, swapped_orientation),
                ]
                less_than = [cand for cand in candidates if cand[2] < 180.0 - tol]
                if less_than:
                    start, end = less_than[0][0], less_than[0][1]
                else:
                    not_reflex = [cand for cand in candidates if cand[2] <= 180.0 + tol]
                    if not_reflex:
                        best = min(
                            not_reflex,
                            key=lambda item: abs(item[2] - 180.0),
                        )
                        start, end = best[0], best[1]
                    else:
                        best = min(candidates, key=lambda item: item[2])
                        start, end = best[0], best[1]
        lines.append(
            "    \\path pic[{opts}] {{{body}}};".format(
                opts=", ".join(options), body=f"angle={start}--{B}--{end}"
            )
        )

    for name in sorted(plan.points.keys()):
        lines.append(f"    \\fill ({name}) circle (\\gsDotR);")
        label_spec = point_label_map.get(name)
        anchor = label_spec.position if label_spec and label_spec.position else _default_point_anchor(plan.points[name], bbox)
        text = label_spec.text if label_spec else name
        formatted = _format_label_text(text)
        anchor_token = anchor if anchor else "above"
        lines.append(
            f"    \\node[ptlabel,{anchor_token}] at ({name}) {{{formatted}}};"
        )

    for label in side_label_specs:
        if not isinstance(label.target, tuple):
            continue
        a, b = label.target
        if a not in plan.points or b not in plan.points:
            continue
        opts: List[str] = ["ptlabel", "midway"]
        if label.slope:
            opts.append("sloped")
        if label.position:
            opts.append(label.position)
        formatted = _format_label_text(label.text)
        lines.append(
            "    \\node[{opts}] at ($({a})!0.5!({b})$) {{{text}}};".format(
                opts=", ".join(opts), a=a, b=b, text=formatted
            )
        )

    lines.append("  \\end{pgfonlayer}")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def _build_tick_map(ticks: Sequence[Tuple[str, str, int]]) -> Dict[Tuple[str, str], List[int]]:
    mapping: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for a, b, idx in ticks:
        key = _normalize_edge((a, b))
        if idx not in mapping[key]:
            mapping[key].append(idx)
    return mapping


def _emit_aux_path(
    path: AuxPath,
    coords: Mapping[str, Tuple[float, float]],
    span: float,
    style: Mapping[str, object],
) -> List[str]:
    kind = path.kind
    data = path.data
    tokens = ["aux"]
    extra_style = style.get("style") if isinstance(style, dict) else None
    if isinstance(extra_style, str) and extra_style:
        tokens.append(extra_style)

    if kind == "line":
        edge = _edge_from_data(data.get("points"))
        if not edge or not _points_present(edge, coords):
            return []
        p1, p2 = _extend_line(coords[edge[0]], coords[edge[1]], span)
        return [
            "\\draw[{styles}] ({x1}, {y1}) -- ({x2}, {y2});".format(
                styles=", ".join(tokens),
                x1=_format_float(p1[0]),
                y1=_format_float(p1[1]),
                x2=_format_float(p2[0]),
                y2=_format_float(p2[1]),
            )
        ]
    if kind == "ray":
        edge = _edge_from_data(data.get("points"))
        if not edge or not _points_present(edge, coords):
            return []
        origin = coords[edge[0]]
        direction = _normalise_vector(_vector(origin, coords[edge[1]]))
        if direction is None:
            return []
        length = max(span * 0.8, 1.0)
        tip = (origin[0] + direction[0] * length, origin[1] + direction[1] * length)
        return [
            "\\draw[{styles},-{{Latex[length=2mm]}}] ({start}) -- ({x}, {y});".format(
                styles=", ".join(tokens),
                start=edge[0],
                x=_format_float(tip[0]),
                y=_format_float(tip[1]),
            )
        ]
    if kind == "perpendicular":
        at = data.get("at")
        to_edge = _edge_from_data(data.get("to"))
        if not isinstance(at, str) or not to_edge:
            return []
        if not _points_present((at, to_edge[0], to_edge[1]), coords):
            return []
        base_vec = _vector(coords[to_edge[0]], coords[to_edge[1]])
        perp = _perp_vector(base_vec)
        if perp is None:
            return []
        length = max(span * 0.5, 1.0)
        start = coords[at]
        p1 = (start[0] - perp[0] * length, start[1] - perp[1] * length)
        p2 = (start[0] + perp[0] * length, start[1] + perp[1] * length)
        return [
            "\\draw[{styles}] ({x1}, {y1}) -- ({x2}, {y2});".format(
                styles=", ".join(tokens),
                x1=_format_float(p1[0]),
                y1=_format_float(p1[1]),
                x2=_format_float(p2[0]),
                y2=_format_float(p2[1]),
            )
        ]
    if kind == "parallel":
        through = data.get("through")
        to_edge = _edge_from_data(data.get("to"))
        if not isinstance(through, str) or not to_edge:
            return []
        if not _points_present((through, to_edge[0], to_edge[1]), coords):
            return []
        direction = _normalise_vector(_vector(coords[to_edge[0]], coords[to_edge[1]]))
        if direction is None:
            return []
        length = max(span * 0.6, 1.0)
        origin = coords[through]
        p1 = (origin[0] - direction[0] * length, origin[1] - direction[1] * length)
        p2 = (origin[0] + direction[0] * length, origin[1] + direction[1] * length)
        return [
            "\\draw[{styles}] ({x1}, {y1}) -- ({x2}, {y2});".format(
                styles=", ".join(tokens),
                x1=_format_float(p1[0]),
                y1=_format_float(p1[1]),
                x2=_format_float(p2[0]),
                y2=_format_float(p2[1]),
            )
        ]
    if kind == "median":
        frm = data.get("frm")
        midpoint = data.get("midpoint")
        if isinstance(frm, str) and isinstance(midpoint, str):
            if frm in coords and midpoint in coords:
                return [
                    "\\draw[{styles}] ({frm}) -- ({mid});".format(
                        styles=", ".join(tokens), frm=frm, mid=midpoint
                    )
                ]
        return []
    return []


def _points_present(names: Sequence[str], coords: Mapping[str, Tuple[float, float]]) -> bool:
    return all(isinstance(name, str) and name in coords for name in names)


def _coords_bbox(points: Iterable[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), max(xs), min(ys), max(ys))


def _default_point_anchor(
    point: Tuple[float, float], bbox: Tuple[float, float, float, float]
) -> str:
    min_x, max_x, min_y, max_y = bbox
    mid_y = 0.5 * (min_y + max_y)
    mid_x = 0.5 * (min_x + max_x)
    x, y = point
    if y < mid_y - 1e-6:
        return "above"
    if y > mid_y + 1e-6:
        return "below"
    return "left" if x > mid_x else "right"


def _scene_span(points: Iterable[Tuple[float, float]]) -> float:
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    if not xs or not ys:
        return 1.0
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return max(span, 1.0)


def _vector(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def _normalise_vector(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    length = _vector_length(vec)
    if length <= 1e-9:
        return None
    return (vec[0] / length, vec[1] / length)


def _perp_vector(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    base = _normalise_vector(vec)
    if base is None:
        return None
    return (-base[1], base[0])


def _vector_length(vec: Tuple[float, float]) -> float:
    return math.hypot(vec[0], vec[1])


def _oriented_angle_degrees(
    coords: Mapping[str, Tuple[float, float]],
    a: str,
    b: str,
    c: str,
) -> Optional[float]:
    if not all(name in coords for name in (a, b, c)):
        return None
    origin = coords[b]
    vec1 = _vector(origin, coords[a])
    vec2 = _vector(origin, coords[c])
    len1 = _vector_length(vec1)
    len2 = _vector_length(vec2)
    if len1 <= 1e-9 or len2 <= 1e-9:
        return None
    cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    angle = math.degrees(math.atan2(cross, dot))
    if angle < 0:
        angle += 360.0
    return angle


def _normalise_angle_degrees(value: float) -> float:
    angle = value % 360.0
    if angle < 0:
        angle += 360.0
    if math.isclose(angle, 0.0, abs_tol=1e-9) and value > 0:
        return 360.0
    return angle


def _angular_difference_degrees(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _extend_line(
    a: Tuple[float, float], b: Tuple[float, float], span: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    direction = _normalise_vector(_vector(a, b))
    if direction is None:
        return (a, b)
    length = max(span * 0.6, 1.0)
    return (
        (a[0] - direction[0] * length, a[1] - direction[1] * length),
        (b[0] + direction[0] * length, b[1] + direction[1] * length),
    )


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


_NUMERIC_TIMES_SQRT_RE = re.compile(r"(?<![\\w)])(-?\d+(?:\.\d+)?)\s*\*\s*(?=\\sqrt)")


def _simplify_numeric_times_sqrt(text: str) -> str:
    return _NUMERIC_TIMES_SQRT_RE.sub(r"\1", text)


def _convert_sqrt_expressions(text: str) -> Tuple[str, bool]:
    result: List[str] = []
    i = 0
    changed = False
    while i < len(text):
        if text.startswith("sqrt", i):
            i += 4
            if i >= len(text) or text[i] != "(":
                return text, False
            depth = 1
            inner_start = i + 1
            i += 1
            while i < len(text) and depth > 0:
                ch = text[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                i += 1
            if depth != 0:
                return text, False
            inner_raw = text[inner_start : i - 1]
            inner_converted, ok = _convert_sqrt_expressions(inner_raw)
            if not ok:
                return text, False
            result.append(r"\sqrt{" + inner_converted + "}")
            changed = True
            continue
        result.append(text[i])
        i += 1
    return ("".join(result), True if changed or result else True)


def _latexify_math_text(text: str) -> str:
    converted, ok = _convert_sqrt_expressions(text)
    if not ok:
        return text
    return _simplify_numeric_times_sqrt(converted)


def _format_label_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped
    return f"${_latexify_math_text(stripped)}$"


def _format_measurement_value(value: object) -> Optional[str]:
    if isinstance(value, SymbolicNumber):
        return _latexify_math_text(value.text)
    if isinstance(value, (int, float)):
        return _format_float(float(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return _latexify_math_text(stripped)
    return None


def _extract_segment_length_text(
    opts: Optional[Mapping[str, object]]
) -> Optional[str]:
    if not opts:
        return None
    for key in ("length", "distance", "value"):
        if key in opts:
            text = _format_measurement_value(opts[key])
            if text:
                return text
    return None


def _coerce_float(value: object) -> Optional[float]:
    if isinstance(value, SymbolicNumber):
        return float(value.value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().strip("$")
        cleaned = cleaned.replace("\\,", "")
        cleaned = cleaned.replace("\\circ", "")
        cleaned = cleaned.replace("\\deg", "")
        cleaned = cleaned.replace("°", "")
        cleaned = cleaned.replace("^", "")
        cleaned = re.sub(r"[{}]", "", cleaned)
        try:
            return float(cleaned)
        except ValueError:
            match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return None
        return None
    return None


def _format_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Cannot format non-finite float for TikZ output")
    formatted = f"{value:.4f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"
