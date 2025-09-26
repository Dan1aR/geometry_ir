"""Utilities to translate GeoScript programs into TikZ code."""

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .utils import latex_escape_keep_math
from ..ast import Program
from ..numbers import SymbolicNumber


standalone_tpl = r"""\documentclass[border=2pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian,english]{babel}
\usepackage{tikz}
\usetikzlibrary{calc,intersections,angles,quotes,through,positioning,decorations.markings,arrows.meta}
\usepackage{amsmath,amssymb}
\usepackage{varwidth}
\usepackage{adjustbox}
\pagestyle{empty}
\begin{document}
\begin{varwidth}{\linewidth}
%s
\begingroup\shorthandoff{"}
\begin{adjustbox}{max width=\linewidth, max totalheight=\textheight, keepaspectratio}
%s
\end{adjustbox}
\endgroup
\end{varwidth}
\end{document}
"""


_BASE_STYLE_LINES = [
    "  \\tikzset{",
    "    point/.style={circle,fill=black,inner sep=1.5pt},",
    "    labelr/.style={right}, labell/.style={left},",
    "    labela/.style={above}, labelb/.style={below},",
    "    tick/.style={postaction=decorate, decoration={markings,",
    "      mark=at position 0.5 with {\\draw (-2pt,0)--(2pt,0);} }},",
    "    tick2/.style={postaction=decorate, decoration={markings,",
    "      mark=at position 0.4 with {\\draw (-2pt,0)--(2pt,0);},",
    "      mark=at position 0.6 with {\\draw (-2pt,0)--(2pt,0);} }}",
    "  }",
]

_LABEL_POS_TO_STYLE = {
    "right": "labelr",
    "left": "labell",
    "above": "labela",
    "below": "labelb",
}

_SIDELABEL_POS_TO_STYLE = {
    "right": "labelr",
    "left": "labell",
    "above": "labela",
    "below": "labelb",
}


@dataclass
class _SegmentLengthSpec:
    edge: Tuple[str, str]
    text: str


@dataclass
class _AngleSpec:
    vertex: str
    start: str
    end: str
    kind: str  # 'angle' or 'right'
    label: Optional[str] = None
    degrees: Optional[float] = None


@dataclass
class _TargetLengthSpec:
    edge: Tuple[str, str]
    label: str
    style: Optional[str]


def generate_tikz_document(
    program: Program,
    point_coords: Mapping[str, Tuple[float, float]],
    *,
    problem_text: Optional[str] = None,
    normalize: bool = False,
) -> str:
    """Render a complete standalone LaTeX document for ``program``.

    Args:
        program: GeoScript program describing the geometry scene.
        point_coords: Mapping of point identifiers to coordinates (typically
            obtained from ``Solution.point_coords``).
        problem_text: Optional textual header placed above the diagram.
        normalize: When ``True`` the supplied coordinates are normalised to a
            centred unit square before rendering.  This is helpful when the
            solver returns raw coordinates with large spans.
    """

    header = ""
    if problem_text:
        header = (
            "\\noindent\\textbf{Problem:} "
            + latex_escape_keep_math(problem_text.strip())
            + "\\par\\vspace{4pt}\n"
        )
    tikz_code = generate_tikz_code(program, point_coords, normalize=normalize)
    return standalone_tpl % (header, tikz_code)


def generate_tikz_code(
    program: Program,
    point_coords: Mapping[str, Tuple[float, float]],
    *,
    normalize: bool = False,
) -> str:
    """Generate TikZ code that draws the scene described by ``program``."""

    coords = _prepare_coordinates(point_coords, normalize=normalize)
    if not isinstance(program, Program):
        raise TypeError("program must be an instance of Program")

    layout_scale = _extract_layout_scale(program)
    segments, segment_lengths = _extract_segments(program)
    rays = _extract_rays(program)
    lines = _extract_lines(program)
    circles = _extract_circles(program)
    labels = _extract_point_labels(program)
    sidelabels = _extract_sidelabels(program)
    target_lengths = _extract_target_lengths(program)
    angles = _extract_angle_markings(program)

    tikz: List[str] = []
    tikz.append(f"\\begin{{tikzpicture}}[scale={_format_float(layout_scale)}]")
    tikz.extend(_BASE_STYLE_LINES)
    tikz.append("")

    if coords:
        for name in sorted(coords.keys()):
            x, y = coords[name]
            tikz.append(f"  \\coordinate ({name}) at ({_format_float(x)}, {_format_float(y)});")
        tikz.append("")

    existing_segment_keys: Set[Tuple[str, str]] = {
        tuple(sorted(edge)) for edge in segments
    }
    for target_spec in target_lengths:
        key = tuple(sorted(target_spec.edge))
        if key not in existing_segment_keys:
            segments.append(target_spec.edge)
            existing_segment_keys.add(key)

    for edge in segments:
        a, b = edge
        if a in coords and b in coords:
            tikz.append(f"  \\draw ({a}) -- ({b});")
    for start, end in rays:
        if start in coords and end in coords:
            tikz.append(
                "  \\draw ({start}) -- ($({start})!2!({end})$);".format(start=start, end=end)
            )
    for start, end in lines:
        if start in coords and end in coords:
            tikz.append(
                "  \\draw ($({start})!-1!({end})$) -- ($({start})!2!({end})$);".format(
                    start=start, end=end
                )
            )
    for center, through in circles:
        if center in coords and through in coords:
            radius = _distance(coords[center], coords[through])
            if radius > 0:
                tikz.append(
                    f"  \\draw ({center}) circle ({_format_float(radius)});")
    if segments or rays or lines or circles:
        tikz.append("")

    angle_lines = _render_angle_markings(angles, coords)
    if angle_lines:
        tikz.extend(angle_lines)
        tikz.append("")

    tikz.extend(_render_point_markers(coords, labels))
    if coords:
        tikz.append("")

    sidelabel_edges = {tuple(sorted(edge)) for edge, _, _ in sidelabels}
    for target_spec in target_lengths:
        key = tuple(sorted(target_spec.edge))
        if key in sidelabel_edges:
            continue
        label = target_spec.label.strip()
        if not label:
            continue
        sidelabels.append((target_spec.edge, target_spec.label, target_spec.style))
        sidelabel_edges.add(key)
    tikz.extend(
        _render_segment_lengths(segment_lengths.values(), sidelabel_edges, coords)
    )
    tikz.extend(_render_sidelabels(sidelabels, coords))

    tikz.append("\\end{tikzpicture}")
    return "\n".join(tikz)


def _prepare_coordinates(
    point_coords: Mapping[str, Tuple[float, float]], *, normalize: bool
) -> Dict[str, Tuple[float, float]]:
    coords: Dict[str, Tuple[float, float]] = {
        key: (float(value[0]), float(value[1]))
        for key, value in point_coords.items()
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
        key: ((pt[0] - cx) * scale, (pt[1] - cy) * scale)
        for key, pt in coords.items()
    }


def _extract_layout_scale(program: Program) -> float:
    for stmt in program.stmts:
        if stmt.kind == "layout":
            value = stmt.data.get("scale")
            if isinstance(value, (int, float)):
                return float(value)
    return 1.0


def _extract_segments(
    program: Program,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], _SegmentLengthSpec]]:
    seen: Dict[Tuple[str, str], Tuple[str, str]] = {}
    order: List[Tuple[str, str]] = []
    lengths: Dict[Tuple[str, str], _SegmentLengthSpec] = {}
    for stmt in program.stmts:
        if stmt.kind == "segment":
            edge = tuple(stmt.data.get("edge", ()))
            if len(edge) != 2:
                continue
            key = tuple(sorted(edge))
            if key not in seen:
                seen[key] = edge
                order.append(edge)
            length_text = _extract_segment_length_text(stmt.opts)
            if length_text:
                orientation = seen.get(key, edge)
                lengths[key] = _SegmentLengthSpec(edge=orientation, text=length_text)
        elif stmt.kind == "diameter":
            edge = tuple(stmt.data.get("edge", ()))
            if len(edge) != 2:
                continue
            key = tuple(sorted(edge))
            if key not in seen:
                seen[key] = edge
                order.append(edge)
    return order, lengths


def _extract_rays(program: Program) -> List[Tuple[str, str]]:
    rays: List[Tuple[str, str]] = []
    for stmt in program.stmts:
        if stmt.kind == "ray":
            ray = tuple(stmt.data.get("ray", ()))
            if len(ray) == 2:
                rays.append(ray)
    return rays


def _extract_lines(program: Program) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    for stmt in program.stmts:
        if stmt.kind == "line":
            edge = tuple(stmt.data.get("edge", ()))
            if len(edge) == 2:
                lines.append(edge)
    return lines


def _extract_circles(program: Program) -> List[Tuple[str, str]]:
    circles: List[Tuple[str, str]] = []
    seen: set = set()
    for stmt in program.stmts:
        if stmt.kind == "circle_center_radius_through":
            center = stmt.data.get("center")
            through = stmt.data.get("through")
            if isinstance(center, str) and isinstance(through, str):
                key = (center, through)
                if key not in seen:
                    seen.add(key)
                    circles.append((center, through))
    return circles


@dataclass
class _LabelSpec:
    text: str
    style: Optional[str]


def _extract_point_labels(program: Program) -> Dict[str, _LabelSpec]:
    labels: Dict[str, _LabelSpec] = {}
    for stmt in program.stmts:
        if stmt.kind == "label_point":
            point = stmt.data.get("point")
            if not isinstance(point, str):
                continue
            label_text = stmt.opts.get("label") if stmt.opts else None
            pos = stmt.opts.get("pos") if stmt.opts else None
            style = None
            if isinstance(pos, str):
                style = _LABEL_POS_TO_STYLE.get(pos.lower())
            text = label_text if isinstance(label_text, str) else point
            labels[point] = _LabelSpec(text=text, style=style)
    return labels


def _extract_sidelabels(program: Program) -> List[Tuple[Tuple[str, str], str, Optional[str]]]:
    sidelabels: List[Tuple[Tuple[str, str], str, Optional[str]]] = []
    for stmt in program.stmts:
        if stmt.kind == "sidelabel":
            edge = tuple(stmt.data.get("edge", ()))
            text = stmt.data.get("text")
            if len(edge) != 2 or not isinstance(text, str):
                continue
            pos = stmt.opts.get("pos") if stmt.opts else None
            pos_style = None
            if isinstance(pos, str):
                pos_style = _SIDELABEL_POS_TO_STYLE.get(pos.lower())
            sidelabels.append((edge, text, pos_style))
    return sidelabels


def _extract_target_lengths(program: Program) -> List[_TargetLengthSpec]:
    targets: List[_TargetLengthSpec] = []
    for stmt in program.stmts:
        if stmt.kind != "target_length":
            continue
        edge = tuple(stmt.data.get("edge", ()))
        if len(edge) != 2:
            continue
        label_value: Optional[str] = None
        if stmt.opts and "label" in stmt.opts:
            raw_label = stmt.opts.get("label")
            if isinstance(raw_label, str):
                label_value = raw_label
        label = label_value if label_value is not None else "?"
        pos_style: Optional[str] = None
        if stmt.opts and "pos" in stmt.opts:
            raw_pos = stmt.opts.get("pos")
            if isinstance(raw_pos, str):
                pos_style = _SIDELABEL_POS_TO_STYLE.get(raw_pos.lower())
        targets.append(_TargetLengthSpec(edge=edge, label=label, style=pos_style))
    return targets


def _extract_angle_markings(program: Program) -> List[_AngleSpec]:
    angles: List[_AngleSpec] = []
    for stmt in program.stmts:
        if stmt.kind == "angle_at":
            at = stmt.data.get("at")
            rays = stmt.data.get("rays", ())
            if not isinstance(at, str) or not isinstance(rays, tuple) or len(rays) != 2:
                continue
            start = _ray_endpoint(rays[0], at)
            end = _ray_endpoint(rays[1], at)
            if not start or not end:
                continue
            degrees_value = stmt.opts.get("degrees") if stmt.opts else None
            label, numeric = (
                _format_angle_label_and_value(degrees_value)
                if degrees_value is not None
                else (None, None)
            )
            if degrees_value is not None and _is_ninety_degrees(degrees_value):
                angles.append(
                    _AngleSpec(
                        vertex=at,
                        start=start,
                        end=end,
                        kind="right",
                        label=label,
                        degrees=numeric,
                    )
                )
            elif label:
                angles.append(
                    _AngleSpec(
                        vertex=at,
                        start=start,
                        end=end,
                        kind="angle",
                        label=label,
                        degrees=numeric,
                    )
                )
        elif stmt.kind == "right_angle_at":
            at = stmt.data.get("at")
            rays = stmt.data.get("rays", ())
            if not isinstance(at, str) or not isinstance(rays, tuple) or len(rays) != 2:
                continue
            start = _ray_endpoint(rays[0], at)
            end = _ray_endpoint(rays[1], at)
            if not start or not end:
                continue
            label: Optional[str] = None
            degrees_numeric: Optional[float] = None
            if stmt.opts and "degrees" in stmt.opts:
                label, degrees_numeric = _format_angle_label_and_value(
                    stmt.opts.get("degrees")
                )
            angles.append(
                _AngleSpec(
                    vertex=at,
                    start=start,
                    end=end,
                    kind="right",
                    label=label,
                    degrees=degrees_numeric,
                )
            )
        elif stmt.kind == "target_angle":
            at = stmt.data.get("at")
            rays = stmt.data.get("rays", ())
            if not isinstance(at, str) or not isinstance(rays, tuple) or len(rays) != 2:
                continue
            start = _ray_endpoint(rays[0], at)
            end = _ray_endpoint(rays[1], at)
            if not start or not end:
                continue
            label: Optional[str] = "?"
            if stmt.opts and "label" in stmt.opts:
                raw_label = stmt.opts.get("label")
                if isinstance(raw_label, str):
                    label = raw_label
            angles.append(
                _AngleSpec(
                    vertex=at,
                    start=start,
                    end=end,
                    kind="angle",
                    label=label,
                    degrees=None,
                )
            )
    return angles


def _render_point_markers(
    coords: Mapping[str, Tuple[float, float]],
    labels: Mapping[str, _LabelSpec],
) -> List[str]:
    if not coords:
        return []
    lines: List[str] = []
    centre = _coords_centre(coords.values())
    for name in sorted(coords.keys()):
        lines.append(f"  \\fill ({name}) circle (1.5pt);")
        label_spec = labels.get(name)
        style = label_spec.style if label_spec else None
        if style is None:
            style = _infer_label_style(coords[name], centre)
        text = label_spec.text if label_spec else name
        formatted_text = _format_label_text(text)
        if formatted_text:
            lines.append(f"  \\node[{style}] at ({name}) {{{formatted_text}}};")
    return lines


def _render_sidelabels(
    sidelabels: Sequence[Tuple[Tuple[str, str], str, Optional[str]]],
    coords: Mapping[str, Tuple[float, float]],
) -> List[str]:
    lines: List[str] = []
    for (a, b), text, style in sidelabels:
        if a not in coords or b not in coords:
            continue
        anchor = style or "labela"
        formatted = _format_label_text(text)
        if not formatted:
            continue
        lines.append(
            f"  \\node[{anchor}] at ($({a})!0.5!({b})$) {{{formatted}}};"
        )
    return lines


def _render_segment_lengths(
    lengths: Iterable[_SegmentLengthSpec],
    sidelabel_edges: Set[Tuple[str, str]],
    coords: Mapping[str, Tuple[float, float]],
) -> List[str]:
    lines: List[str] = []
    for spec in lengths:
        a, b = spec.edge
        if a not in coords or b not in coords:
            continue
        key = tuple(sorted((a, b)))
        if key in sidelabel_edges:
            continue
        formatted = _format_label_text(spec.text)
        if not formatted:
            continue
        lines.append(
            f"  \\node[labela] at ($({a})!0.5!({b})$) {{{formatted}}};"
        )
    return lines


def _render_angle_markings(
    angles: Sequence[_AngleSpec],
    coords: Mapping[str, Tuple[float, float]],
) -> List[str]:
    lines: List[str] = []
    for spec in angles:
        vertex = spec.vertex
        start_name = spec.start
        end_name = spec.end
        if (
            vertex not in coords
            or start_name not in coords
            or end_name not in coords
        ):
            continue
        vertex_pt = coords[vertex]
        start_pt = coords[start_name]
        end_pt = coords[end_name]
        if spec.degrees is not None:
            if _should_swap_angle_rays(vertex_pt, start_pt, end_pt, spec.degrees):
                start_name, end_name = end_name, start_name
                start_pt, end_pt = end_pt, start_pt
        radius = _compute_angle_radius(vertex_pt, start_pt, end_pt)
        label_text = _format_label_text(spec.label) if spec.label else ""
        if spec.kind == "right":
            lines.append(
                "  \\pic [draw, angle radius={radius}, angle eccentricity=1.12] {{right angle = {start}--{vertex}--{end}}};".format(
                    radius=_format_float(radius), start=start_name, vertex=vertex, end=end_name
                )
            )
            if label_text:
                label_pos = _angle_label_position(vertex_pt, start_pt, end_pt, radius)
                if label_pos:
                    px, py = label_pos
                    lines.append(
                        "  \\node at ({px}, {py}) {{{label}}};".format(
                            px=_format_float(px), py=_format_float(py), label=label_text
                        )
                    )
            continue
        arc_cmd = _angle_arc_command(vertex_pt, start_pt, end_pt, radius)
        if arc_cmd:
            lines.append(arc_cmd)
        if label_text:
            label_pos = _angle_label_position(vertex_pt, start_pt, end_pt, radius)
            if label_pos:
                px, py = label_pos
                lines.append(
                    "  \\node at ({px}, {py}) {{{label}}};".format(
                        px=_format_float(px), py=_format_float(py), label=label_text
                    )
                )
    return lines


def _coords_centre(coords: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    if not xs or not ys:
        return (0.0, 0.0)
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))


def _infer_label_style(point: Tuple[float, float], centre: Tuple[float, float]) -> str:
    dx = point[0] - centre[0]
    dy = point[1] - centre[1]
    if abs(dx) >= abs(dy):
        return "labelr" if dx >= 0 else "labell"
    return "labela" if dy >= 0 else "labelb"


def _latexify_math_text(text: str) -> str:
    converted, ok = _convert_sqrt_expressions(text)
    if not ok:
        return text
    return _simplify_numeric_times_sqrt(converted)


def _convert_sqrt_expressions(text: str) -> Tuple[str, bool]:
    result: List[str] = []
    i = 0
    changed = False
    while i < len(text):
        if text.startswith("sqrt(", i):
            i += 5
            depth = 1
            inner_start = i
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


_NUMERIC_TIMES_SQRT_RE = re.compile(r"(?<![\\w)])(-?\d+(?:\.\d+)?)\s*\*\s*(?=\\sqrt)")


def _simplify_numeric_times_sqrt(text: str) -> str:
    return _NUMERIC_TIMES_SQRT_RE.sub(r"\1", text)


def _format_label_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped
    return f"${_latexify_math_text(stripped)}$"


def _format_measurement_value(value: object) -> Optional[str]:
    if isinstance(value, SymbolicNumber):
        return value.text
    if isinstance(value, (int, float)):
        return _format_float(float(value))
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
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


def _format_angle_label_and_value(value: object) -> Tuple[Optional[str], Optional[float]]:
    numeric = _coerce_float(value)
    text = _format_measurement_value(value)
    if not text:
        return None, numeric
    if "\\circ" not in text and "°" not in text:
        text = f"{text}^\\circ"
    return text, numeric


def _is_ninety_degrees(value: object) -> bool:
    numeric = _coerce_float(value)
    if numeric is None:
        return False
    return math.isfinite(numeric) and abs(numeric - 90.0) <= 1e-6


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


def _compute_angle_radius(
    vertex: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    dist1 = _distance(vertex, start)
    dist2 = _distance(vertex, end)
    min_dist = min(dist1, dist2)
    if min_dist <= 1e-6:
        return 0.4
    radius = max(min_dist * 0.35, 0.35)
    return min(radius, min_dist * 0.9)


def _angle_label_position(
    vertex: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
    radius: float,
) -> Optional[Tuple[float, float]]:
    dir1 = _normalise_vector((start[0] - vertex[0], start[1] - vertex[1]))
    dir2 = _normalise_vector((end[0] - vertex[0], end[1] - vertex[1]))
    if dir1 is None or dir2 is None:
        return None
    bisector = (dir1[0] + dir2[0], dir1[1] + dir2[1])
    norm = _vector_length(bisector)
    if norm <= 1e-9:
        return None
    bisector = (bisector[0] / norm, bisector[1] / norm)
    offset = radius + max(0.1, radius * 0.15)
    return (
        vertex[0] + bisector[0] * offset,
        vertex[1] + bisector[1] * offset,
    )


def _should_swap_angle_rays(
    vertex: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
    target_degrees: float,
) -> bool:
    ccw = _counter_clockwise_angle(vertex, start, end)
    if ccw is None:
        return False
    if not math.isfinite(target_degrees):
        return False
    target = target_degrees % 360.0
    if target < 0:
        target += 360.0
    alt = (360.0 - ccw) % 360.0
    current_diff = _angle_difference(ccw, target)
    alternate_diff = _angle_difference(alt, target)
    if alternate_diff + 1e-6 < current_diff:
        return True
    return False


def _counter_clockwise_angle(
    vertex: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Optional[float]:
    start_vec = (start[0] - vertex[0], start[1] - vertex[1])
    end_vec = (end[0] - vertex[0], end[1] - vertex[1])
    if _vector_length(start_vec) <= 1e-9 or _vector_length(end_vec) <= 1e-9:
        return None
    cross = start_vec[0] * end_vec[1] - start_vec[1] * end_vec[0]
    dot = start_vec[0] * end_vec[0] + start_vec[1] * end_vec[1]
    angle = math.degrees(math.atan2(cross, dot))
    if angle < 0:
        angle += 360.0
    return angle


def _angle_difference(a: float, b: float) -> float:
    diff = ((a - b + 180.0) % 360.0) - 180.0
    return abs(diff)


def _angle_arc_command(
    vertex: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
    radius: float,
) -> Optional[str]:
    start_vec = (start[0] - vertex[0], start[1] - vertex[1])
    end_vec = (end[0] - vertex[0], end[1] - vertex[1])
    start_dir = _normalise_vector(start_vec)
    end_dir = _normalise_vector(end_vec)
    if start_dir is None or end_dir is None:
        return None
    start_angle = math.degrees(math.atan2(start_dir[1], start_dir[0]))
    end_angle = math.degrees(math.atan2(end_dir[1], end_dir[0]))
    delta = (end_angle - start_angle) % 360.0
    if delta <= 1e-3:
        return None
    end_angle = start_angle + delta
    sx = _format_float(start_dir[0] * radius)
    sy = _format_float(start_dir[1] * radius)
    vx = _format_float(vertex[0])
    vy = _format_float(vertex[1])
    radius_str = _format_float(radius)
    options = ", ".join(
        [
            f"shift={{({vx},{vy})}}",
            "line cap=round",
            "line width=0.6pt",
        ]
    )
    return (
        "  \\draw[{options}] ({sx}, {sy}) arc[start angle={sa}, end angle={ea}, radius={radius}];".format(
            options=options,
            sx=sx,
            sy=sy,
            sa=_format_float(start_angle),
            ea=_format_float(end_angle),
            radius=radius_str,
        )
    )


def _normalise_vector(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    length = _vector_length(vec)
    if length <= 1e-9:
        return None
    return (vec[0] / length, vec[1] / length)


def _vector_length(vec: Tuple[float, float]) -> float:
    return math.hypot(vec[0], vec[1])


def _ray_endpoint(ray: object, vertex: str) -> Optional[str]:
    if not isinstance(ray, Sequence) or len(ray) != 2:
        return None
    start, end = ray
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    if start != vertex:
        return None
    return end


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _format_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Cannot format non-finite float for TikZ output")
    formatted = f"{value:.4f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"
