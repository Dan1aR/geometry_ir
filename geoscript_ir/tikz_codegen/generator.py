"""TikZ renderer adhering to the GeoScript style contract."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils import latex_escape_keep_math
from ..ast import Program
from ..numbers import SymbolicNumber
from ..orientation import apply_orientation


PT_PER_CM = 28.3464567
FOOTNOTE_EM_PT = 8.0
GS_DOT_RADIUS_PT = 1.4
GS_ANGLE_SEP_PT = 2.0
LABEL_WIDTH_EM = 0.52
LABEL_HEIGHT_EM = 0.9
ANGLE_LABEL_OFFSET_EM = 0.6
EPS_LEN_FACTOR = 1e-9
COLLINEAR_EPS_FACTOR = 1e-8

ANCHOR_SEQUENCE = [
    "above",
    "above right",
    "right",
    "below right",
    "below",
    "below left",
    "left",
    "above left",
]

ANCHOR_DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "above": (0.0, 1.0),
    "below": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "above right": (math.sqrt(0.5), math.sqrt(0.5)),
    "below right": (math.sqrt(0.5), -math.sqrt(0.5)),
    "below left": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "above left": (-math.sqrt(0.5), math.sqrt(0.5)),
}


def _direction_from_anchor(anchor: Optional[str]) -> Tuple[float, float]:
    if not anchor:
        return (0.0, 1.0)
    direction = ANCHOR_DIRECTIONS.get(anchor.strip().lower())
    if direction is None:
        return (0.0, 1.0)
    return direction


def _side_label_shift_tokens(anchor: Optional[str], side_offset_pt: float) -> List[str]:
    if side_offset_pt <= 0:
        return []
    dx, dy = _direction_from_anchor(anchor)
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-9:
        return []
    scale = side_offset_pt / magnitude
    dx *= scale
    dy *= scale
    tokens: List[str] = ["anchor=center"]
    if abs(dx) > 1e-9:
        tokens.append(f"xshift={_format_float(dx)}pt")
    if abs(dy) > 1e-9:
        tokens.append(f"yshift={_format_float(dy)}pt")
    return tokens

standalone_tpl = r"""\documentclass[border=2pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian,english]{babel}
\usepackage{adjustbox}
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
      mark=at position 0.5 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);}}},
  tick2/.style={postaction=decorate, decoration={markings,
      mark=at position 0.47 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);},
      mark=at position 0.53 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);}}},
  tick3/.style={postaction=decorate, decoration={markings,
      mark=at position 0.44 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);},
      mark=at position 0.5  with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);},
      mark=at position 0.56 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);}}},
}
%% optional layers
\pgfdeclarelayer{bg}\pgfdeclarelayer{fg}\pgfsetlayers{bg,main,fg}
\begin{document}
\begin{minipage}[t]{\linewidth} 
%s

\begin{adjustbox}{max width=\linewidth, max totalheight=\textheight, keepaspectratio}
%s
\end{adjustbox}
\end{minipage}
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
    leader: Optional[Tuple[float, float]] = None


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
    angles: List[Dict[str, object]]
    labels: List[LabelSpec]
    tick_overflow_edges: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    helper_tick_edges: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)
    tick_overlay_edges: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)
    carrier_lookup: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)
    angle_groups: List[List[Tuple[str, str, str]]] = field(default_factory=list)
    angle_group_arc_counts: List[int] = field(default_factory=list)
    right_angles: List[Tuple[str, str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    polygon_vertices: set = field(default_factory=set)
    circle_centers: set = field(default_factory=set)
    special_points: set = field(default_factory=set)


class _UnionFind:
    """Simple disjoint-set structure for deterministic grouping."""

    def __init__(self) -> None:
        self._parent: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._rank: Dict[Tuple[str, str], int] = {}

    def add(self, item: Tuple[str, str]) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._rank[item] = 0

    def find(self, item: Tuple[str, str]) -> Tuple[str, str]:
        parent = self._parent.get(item)
        if parent is None:
            self.add(item)
            return item
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, a: Tuple[str, str], b: Tuple[str, str]) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self._rank[root_a]
        rank_b = self._rank[root_b]
        if rank_a < rank_b:
            self._parent[root_a] = root_b
        elif rank_a > rank_b:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] += 1

    def components(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for item in self._parent:
            root = self.find(item)
            groups.setdefault(root, []).append(item)
        return groups


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
    return standalone_tpl % (header, tikz_code)


def generate_tikz_code(
    program: Program,
    point_coords: Mapping[str, Tuple[float, float]],
    *,
    normalize: bool = False,
) -> str:
    """Generate TikZ code that respects the rendering contract."""

    oriented_coords, _ = apply_orientation(program, point_coords)
    coords = _prepare_coordinates(oriented_coords, normalize=normalize)
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
        "no_equations_on_sides": False,
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
    tick_overlay_edges: Dict[Tuple[str, str], Tuple[str, str]] = {}
    angle_entries: List[Dict[str, object]] = []
    point_labels: Dict[str, LabelSpec] = {}
    side_labels: List[LabelSpec] = []
    polygon_vertices: set = set()
    circle_centers: set = set()
    special_points: set = set()

    segment_union = _UnionFind()
    edge_orientations: Dict[Tuple[str, str], Tuple[str, str]] = {}
    explicit_edge_occurrence: Dict[Tuple[str, str], int] = {}
    implicit_segment_pairs: List[Tuple[Tuple[str, str], Tuple[str, str], int]] = []
    implicit_edge_occurrence: Dict[Tuple[str, str], int] = {}
    segment_conflicts_logged: set = set()

    angle_union = _UnionFind()
    wedge_orientations: Dict[Tuple[str, Tuple[str, str]], List[Tuple[str, str, str]]] = {}
    explicit_wedge_occurrence: Dict[Tuple[str, Tuple[str, str]], int] = {}
    bisector_occurrence: Dict[Tuple[str, Tuple[str, str]], int] = {}
    bisector_orientations: Dict[Tuple[str, Tuple[str, str]], Tuple[str, str, str]] = {}
    wedge_conflicts_logged: set = set()

    right_angle_marks: List[Tuple[str, str, str]] = []
    right_angle_seen: set = set()
    diagnostics: List[str] = []

    def record_edge_orientation(norm: Tuple[str, str], oriented: Tuple[str, str]) -> None:
        edge_orientations.setdefault(norm, oriented)

    def record_wedge_orientation(
        norm: Tuple[str, Tuple[str, str]], triple: Tuple[str, str, str]
    ) -> None:
        entries = wedge_orientations.setdefault(norm, [])
        if triple not in entries:
            entries.append(triple)

    def add_right_angle(a: str, b: str, c: str) -> None:
        key = (a, b, c)
        if key not in right_angle_seen:
            right_angle_seen.add(key)
            right_angle_marks.append((a, b, c))

    scene_diag = _scene_bbox_diag(_coords_bbox(coords.values()))
    collinear_epsilon = max(EPS_LEN_FACTOR * scene_diag, COLLINEAR_EPS_FACTOR)

    for idx, stmt in enumerate(program.stmts):
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
            record_edge_orientation(key, (edge[0], edge[1]))
        elif kind == "diameter":
            edge = _edge_from_data(data.get("edge"))
            if not edge:
                continue
            key = _normalize_edge(edge)
            carriers.setdefault(key, (edge[0], edge[1], {}))
            carrier_lookup[key] = (edge[0], edge[1])
            record_edge_orientation(key, (edge[0], edge[1]))
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
                carriers.setdefault(key, (edge[0], edge[1], {"source": kind}))
                carrier_lookup[key] = (edge[0], edge[1])
                record_edge_orientation(key, (edge[0], edge[1]))
            for ident in ids:
                if isinstance(ident, str):
                    polygon_vertices.add(ident)
            if kind == "triangle":
                apex = opts.get("isosceles") if isinstance(opts.get("isosceles"), str) else None
                if isinstance(apex, str) and len(ids) == 3:
                    apex = apex.strip()
                    apex_key = apex[-1] if apex.startswith("at") and len(apex) == 3 else None
                    if apex_key in {"A", "B", "C"}:
                        vertex_map = {"A": ids[0], "B": ids[1], "C": ids[2]}
                        apex_vertex = vertex_map.get(apex_key)
                        if all(isinstance(v, str) for v in ids) and apex_vertex in ids:
                            pairs = []
                            if apex_key == "A":
                                pairs = [(ids[0], ids[1]), (ids[0], ids[2])]
                            elif apex_key == "B":
                                pairs = [(ids[1], ids[0]), (ids[1], ids[2])]
                            elif apex_key == "C":
                                pairs = [(ids[2], ids[0]), (ids[2], ids[1])]
                            normals: List[Tuple[str, str]] = []
                            for oriented in pairs:
                                if oriented[0] == oriented[1]:
                                    continue
                                norm = _normalize_edge(oriented)
                                record_edge_orientation(norm, oriented)
                                implicit_edge_occurrence.setdefault(norm, idx)
                                normals.append(norm)
                            if len(normals) == 2:
                                implicit_segment_pairs.append((normals[0], normals[1], idx))
        elif kind == "ray":
            edge = _edge_from_data(data.get("ray"))
            if edge:
                aux_lines.append((AuxPath("ray", {"points": edge}), {}))
        elif kind == "line":
            edge = _edge_from_data(data.get("edge"))
            if edge:
                aux_lines.append((AuxPath("line", {"points": edge}), {}))
        elif kind == "perpendicular_at":
            to_edge = _edge_from_data(data.get("to"))
            at = data.get("at")
            foot = data.get("foot")
            if to_edge and isinstance(at, str):
                if rules.get("allow_auxiliary", True):
                    aux_lines.append(
                        (
                            AuxPath(
                                "perpendicular",
                                {"at": at, "to": to_edge, "foot": foot},
                            ),
                            {},
                        )
                    )
                special_points.add(at)
                if isinstance(foot, str):
                    special_points.add(foot)
                    add_right_angle(to_edge[0], foot, at)
        elif kind == "foot":
            foot = data.get("foot")
            frm = data.get("from")
            edge = _edge_from_data(data.get("edge"))
            if isinstance(foot, str) and isinstance(frm, str) and edge:
                special_points.add(foot)
                if rules.get("allow_auxiliary", True):
                    aux_lines.append(
                        (
                            AuxPath("segment", {"points": (frm, foot)}),
                            {},
                        )
                    )
                add_right_angle(edge[0], foot, frm)
                if all(
                    isinstance(name, str) and name in coords
                    for name in (foot, edge[0], edge[1])
                ):
                    base_vec = _vector(coords[edge[0]], coords[edge[1]])
                    off_vec = _vector(coords[edge[0]], coords[foot])
                    base_len = _vector_length(base_vec)
                    if base_len > 1e-9:
                        distance = abs(base_vec[0] * off_vec[1] - base_vec[1] * off_vec[0]) / base_len
                        if distance > collinear_epsilon:
                            key = _normalize_edge((edge[0], foot))
                            carriers.setdefault(key, (edge[0], foot, {"source": "foot"}))
                            carrier_lookup[key] = (edge[0], foot)
                            record_edge_orientation(key, (edge[0], foot))
        elif kind == "parallel_through":
            to_edge = _edge_from_data(data.get("to"))
            through = data.get("through")
            if to_edge and isinstance(through, str) and rules.get("allow_auxiliary", True):
                aux_lines.append(
                    (
                        AuxPath("parallel", {"through": through, "to": to_edge}),
                        {},
                    )
                )
        elif kind == "median_from_to":
            frm = data.get("frm")
            midpoint = data.get("midpoint")
            base = _edge_from_data(data.get("to"))
            if base and isinstance(frm, str) and isinstance(midpoint, str):
                special_points.add(midpoint)
                edges = [(base[0], midpoint), (midpoint, base[1])]
                normals: List[Tuple[str, str]] = []
                for oriented in edges:
                    if oriented[0] == oriented[1]:
                        continue
                    norm = _normalize_edge(oriented)
                    record_edge_orientation(norm, oriented)
                    implicit_edge_occurrence.setdefault(norm, idx)
                    normals.append(norm)
                if len(normals) == 2:
                    implicit_segment_pairs.append((normals[0], normals[1], idx))
                if rules.get("allow_auxiliary", True):
                    aux_lines.append(
                        (
                            AuxPath(
                                "median",
                                {"frm": frm, "midpoint": midpoint, "base": base},
                            ),
                            {},
                        )
                    )
        elif kind == "midpoint":
            midpoint = data.get("midpoint")
            base = _edge_from_data(data.get("edge"))
            if base and isinstance(midpoint, str):
                special_points.add(midpoint)
                edges = [(base[0], midpoint), (midpoint, base[1])]
                normals: List[Tuple[str, str]] = []
                for oriented in edges:
                    if oriented[0] == oriented[1]:
                        continue
                    norm = _normalize_edge(oriented)
                    record_edge_orientation(norm, oriented)
                    implicit_edge_occurrence.setdefault(norm, idx)
                    normals.append(norm)
                if len(normals) == 2:
                    implicit_segment_pairs.append((normals[0], normals[1], idx))
        elif kind == "point_on":
            point = data.get("point")
            path = data.get("path")
            if not (
                isinstance(point, str)
                and isinstance(path, tuple)
                and len(path) == 2
            ):
                continue
            path_kind = path[0]
            if path_kind == "segment":
                base = _edge_from_data(path[1])
                mark = opts.get("mark") if isinstance(opts.get("mark"), str) else None
                if base and mark == "midpoint":
                    special_points.add(point)
                    edges = [(base[0], point), (point, base[1])]
                    normals: List[Tuple[str, str]] = []
                    for oriented in edges:
                        if oriented[0] == oriented[1]:
                            continue
                        norm = _normalize_edge(oriented)
                        record_edge_orientation(norm, oriented)
                        implicit_edge_occurrence.setdefault(norm, idx)
                        normals.append(norm)
                    if len(normals) == 2:
                        implicit_segment_pairs.append((normals[0], normals[1], idx))
            elif path_kind == "circle":
                center = path[1]
                if isinstance(center, str):
                    special_points.add(point)
                    circle_centers.add(center)
                    existing = next(
                        (through for (ctr, through) in circles if ctr == center),
                        None,
                    )
                    if existing is None:
                        circles.setdefault((center, point), {})
        elif kind == "circle_center_radius_through":
            center = data.get("center")
            through = data.get("through")
            if isinstance(center, str) and isinstance(through, str):
                circles.setdefault((center, through), {})
                circle_centers.add(center)
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
            edges_in_stmt: List[Tuple[str, str]] = []
            for group in (data.get("lhs"), data.get("rhs")):
                if not isinstance(group, (list, tuple)):
                    continue
                for raw_edge in group:
                    oriented = _edge_from_data(raw_edge)
                    if not oriented or oriented[0] == oriented[1]:
                        continue
                    norm = _normalize_edge(oriented)
                    record_edge_orientation(norm, oriented)
                    segment_union.add(norm)
                    edges_in_stmt.append(norm)
                    prev = explicit_edge_occurrence.get(norm)
                    if prev is None:
                        explicit_edge_occurrence[norm] = idx
                    elif prev != idx and norm not in segment_conflicts_logged:
                        diagnostics.append(
                            f"edge {norm[0]}-{norm[1]} declared in multiple equal-segments statements"
                        )
                        segment_conflicts_logged.add(norm)
            if edges_in_stmt:
                anchor = edges_in_stmt[0]
                for other in edges_in_stmt[1:]:
                    segment_union.union(anchor, other)
        elif kind == "equal_angles":
            wedges_in_stmt: List[Tuple[str, Tuple[str, str]]] = []
            for group in (data.get("lhs"), data.get("rhs")):
                if not isinstance(group, (list, tuple)):
                    continue
                for raw_angle in group:
                    triple = _angle_triple(raw_angle)
                    if not triple:
                        continue
                    norm = _normalize_wedge(triple)
                    record_wedge_orientation(norm, triple)
                    angle_union.add(norm)
                    wedges_in_stmt.append(norm)
                    prev = explicit_wedge_occurrence.get(norm)
                    if prev is None:
                        explicit_wedge_occurrence[norm] = idx
                    elif prev != idx and norm not in wedge_conflicts_logged:
                        diagnostics.append(
                            f"angle at {norm[0]} declared in multiple equal-angles statements"
                        )
                        wedge_conflicts_logged.add(norm)
            if wedges_in_stmt:
                anchor = wedges_in_stmt[0]
                for other in wedges_in_stmt[1:]:
                    angle_union.union(anchor, other)
        elif kind == "angle_bisector_at":
            triple = _angle_triple(data.get("points"))
            if triple:
                norm = _normalize_wedge(triple)
                record_wedge_orientation(norm, triple)
                current = bisector_occurrence.get(norm)
                if current is None or idx < current:
                    bisector_occurrence[norm] = idx
                    bisector_orientations[norm] = triple
        elif kind == "angle_at":
            triple = _angle_triple(data.get("points"))
            if not triple:
                continue
            label, numeric = _angle_label_from_opts(opts)
            angle_entries.append(
                {
                    "kind": "numeric",
                    "A": triple[0],
                    "B": triple[1],
                    "C": triple[2],
                    "degrees": numeric,
                    "label": label,
                }
            )
        elif kind == "right_angle_at":
            triple = _angle_triple(data.get("points"))
            if triple:
                add_right_angle(triple[0], triple[1], triple[2])
        elif kind in {
            "target_angle",
            "target_length",
            "target_point",
            "target_circle",
            "target_arc",
        }:
            continue

        for triple in _extract_angle_bisectors(data):
            norm = _normalize_wedge(triple)
            record_wedge_orientation(norm, triple)
            current = bisector_occurrence.get(norm)
            if current is None or idx < current:
                bisector_occurrence[norm] = idx
                bisector_orientations[norm] = triple

    explicit_edges = set(explicit_edge_occurrence)
    for edge_a, edge_b, order in implicit_segment_pairs:
        if edge_a in explicit_edges or edge_b in explicit_edges:
            continue
        segment_union.add(edge_a)
        segment_union.add(edge_b)
        segment_union.union(edge_a, edge_b)
        implicit_edge_occurrence.setdefault(edge_a, order)
        implicit_edge_occurrence.setdefault(edge_b, order)

    segment_components = segment_union.components()
    ordered_segment_groups: List[List[Tuple[str, str]]] = []
    explicit_groups: List[Tuple[int, List[Tuple[str, str]]]] = []
    implicit_groups: List[Tuple[int, List[Tuple[str, str]]]] = []
    for component in segment_components.values():
        sorted_edges = sorted(component)
        if any(edge in explicit_edges for edge in sorted_edges):
            order = min(explicit_edge_occurrence.get(edge, math.inf) for edge in sorted_edges)
            explicit_groups.append((order, sorted_edges))
        else:
            order = min(implicit_edge_occurrence.get(edge, math.inf) for edge in sorted_edges)
            implicit_groups.append((order, sorted_edges))
    explicit_groups.sort(key=lambda item: (item[0], tuple(item[1])))
    implicit_groups.sort(key=lambda item: (item[0], tuple(item[1])))
    ordered_segment_groups.extend(group for _, group in explicit_groups)
    ordered_segment_groups.extend(group for _, group in implicit_groups)

    for group_index, edges in enumerate(ordered_segment_groups, start=1):
        style_index = ((group_index - 1) % 3) + 1
        overflow = group_index > 3
        for norm in edges:
            oriented = carrier_lookup.get(norm) or edge_orientations.get(norm) or norm
            ticks.append((oriented[0], oriented[1], style_index))
            if overflow:
                tick_overflow_edges[norm] = True
                if norm in carrier_lookup:
                    tick_overlay_edges.setdefault(norm, oriented)
            if norm not in carrier_lookup:
                helper_tick_edges.setdefault(norm, oriented)

    angle_groups: List[List[Tuple[str, str, str]]] = []
    angle_group_arc_counts: List[int] = []
    angle_components = angle_union.components()
    explicit_angle_groups: List[Tuple[int, List[Tuple[str, Tuple[str, str]]]]] = []
    for component in angle_components.values():
        sorted_wedges = sorted(component)
        order = min(explicit_wedge_occurrence.get(wedge, math.inf) for wedge in sorted_wedges)
        explicit_angle_groups.append((order, sorted_wedges))
    explicit_angle_groups.sort(key=lambda item: (item[0], tuple(item[1])))

    for idx, (_, wedges) in enumerate(explicit_angle_groups, start=1):
        members: List[Tuple[str, str, str]] = []
        seen_triples: set = set()
        for wedge in wedges:
            orientations = wedge_orientations.get(wedge)
            if not orientations:
                a, c = wedge[1]
                orientations = [(a, wedge[0], c)]
            for triple in orientations:
                if triple not in seen_triples:
                    members.append(triple)
                    seen_triples.add(triple)
        angle_groups.append(members)
        angle_group_arc_counts.append(idx)

    explicit_wedges = set(explicit_wedge_occurrence)
    implicit_bisectors = sorted(
        bisector_occurrence.items(), key=lambda item: (item[1], item[0])
    )
    for norm, _ in implicit_bisectors:
        if norm in explicit_wedges:
            continue
        triple = bisector_orientations.get(norm)
        if not triple:
            a, c = norm[1]
            triple = (a, norm[0], c)
        angle_groups.append([triple])
        angle_group_arc_counts.append(2)

    labels = list(point_labels.values()) + side_labels

    plan = RenderPlan(
        points=dict(coords),
        carriers=list(carriers.values()),
        aux_lines=aux_lines,
        circles=[(center, through, style) for (center, through), style in circles.items()],
        ticks=ticks,
        angles=angle_entries,
        labels=labels,
    )
    plan.tick_overflow_edges = tick_overflow_edges
    plan.helper_tick_edges = helper_tick_edges
    plan.tick_overlay_edges = tick_overlay_edges
    plan.carrier_lookup = carrier_lookup
    plan.angle_groups = angle_groups
    plan.angle_group_arc_counts = angle_group_arc_counts
    plan.right_angles = right_angle_marks
    plan.notes = diagnostics
    plan.polygon_vertices = polygon_vertices
    plan.circle_centers = circle_centers
    plan.special_points = special_points
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


def _normalize_wedge(triple: Tuple[str, str, str]) -> Tuple[str, Tuple[str, str]]:
    a, b, c = triple
    return (b, (a, c) if a <= c else (c, a))


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


def _extract_angle_bisectors(value: object) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []

    def visit(obj: object) -> None:
        if isinstance(obj, tuple):
            if len(obj) == 2 and obj[0] == "angle-bisector" and isinstance(obj[1], dict):
                payload = obj[1]
                triple = _angle_triple(payload.get("points"))
                if not triple:
                    triple = _angle_triple(payload.get("points_chain"))
                if triple:
                    triples.append(triple)
                    return
            for item in obj:
                visit(item)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)

    visit(value)
    return triples


def _angle_label_from_opts(opts: Mapping[str, object]) -> Tuple[Optional[str], Optional[float]]:
    if not opts:
        return None, None
    if "degrees" in opts:
        text = _format_measurement_value(opts.get("degrees"))
        numeric = _coerce_float(opts.get("degrees"))
        if text:
            text = text.strip()
            degree_token = "^\\circ"
            if text.startswith("$") and text.endswith("$"):
                inner = text[1:-1]
                if "\\circ" not in inner and "°" not in inner:
                    text = f"${inner}{degree_token}$"
            else:
                if "\\circ" not in text and "°" not in text:
                    text = f"${text}{degree_token}$"
                else:
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

    bbox = _coords_bbox(plan.points.values())
    span = _scene_span(plan.points.values())
    scene_diag = _scene_bbox_diag(bbox)
    length_epsilon = EPS_LEN_FACTOR * scene_diag
    tick_map = _build_tick_map(plan.ticks, plan.points, length_epsilon)

    lines.append("  \\begin{pgfonlayer}{main}")
    for start, end, style in plan.carriers:
        if start not in plan.points or end not in plan.points:
            continue
        key = _normalize_edge((start, end))
        style_tokens = ["carrier"]
        for idx in tick_map.get(key, []):
            style_tokens.append(f"tick{idx}")
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

    point_label_map = {
        label.target: label for label in plan.labels if label.kind == "point"
    }
    side_label_specs = [label for label in plan.labels if label.kind == "side"]

    pt_per_unit = max(layout_scale * PT_PER_CM, 1e-6)
    scene_diag_pt = scene_diag * pt_per_unit
    base_offset_pt = max(1.8 * GS_DOT_RADIUS_PT, 0.012 * scene_diag_pt)
    base_offset_units = base_offset_pt / pt_per_unit if pt_per_unit > 0 else 0.0
    side_offset_pt = max(1.2 * GS_DOT_RADIUS_PT, 0.008 * scene_diag_pt)

    segments = _collect_segments(plan)
    circle_geoms = _collect_circle_geoms(plan)
    placed_label_boxes: List[Tuple[Tuple[float, float, float, float], str]] = []

    lines.append("  \\begin{pgfonlayer}{fg}")
    for path, style in plan.aux_lines:
        aux_lines = _emit_aux_path(path, plan.points, span, style)
        for entry in aux_lines:
            lines.append("    " + entry)

    for key, oriented in plan.tick_overlay_edges.items():
        a, b = oriented
        if a not in plan.points or b not in plan.points:
            continue
        tokens = ["carrier", "draw opacity=0"]
        for idx in tick_map.get(key, []):
            tokens.append(f"tick{idx}")
        tokens.append("densely dashed")
        lines.append(
            "    \\draw[{styles}] ({a}) -- ({b});".format(
                styles=", ".join(tokens), a=a, b=b
            )
        )

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

    angle_arc_lines, angle_label_entries, placed_label_boxes = _render_angle_marks(
        plan,
        layout_scale,
        rules,
        segments,
        circle_geoms,
        pt_per_unit,
        base_offset_units,
        placed_label_boxes,
    )
    for arc_line in angle_arc_lines:
        lines.append("    " + arc_line)
    for entry in angle_label_entries:
        leader = entry.get("leader")
        if leader:
            lines.append("    " + leader)
        lines.append("    " + entry["node"])

    point_layouts, placed_label_boxes = _layout_point_labels(
        plan,
        point_label_map,
        bbox,
        base_offset_units,
        pt_per_unit,
        segments,
        circle_geoms,
        placed_label_boxes,
    )

    for name in sorted(plan.points.keys()):
        lines.append(f"    \\fill ({name}) circle (\\gsDotR);")
        info = point_layouts.get(name)
        if info and info.get("leader"):
            lines.append("    " + info["leader"])
        if info:
            anchor = info.get("anchor") or "above"
            formatted = info.get("text") or _format_label_text(name)
        else:
            anchor = _default_point_anchor(plan.points[name], bbox)
            formatted = _format_label_text(name)
        lines.append(
            f"    \\node[ptlabel,{anchor}] at ({name}) {{{formatted}}};"
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
        opts.extend(_side_label_shift_tokens(label.position, side_offset_pt))
        formatted = _format_label_text(label.text)
        lines.append(
            "    \\path ({a}) -- ({b}) node[{opts}] {{{text}}};".format(
                opts=", ".join(opts), a=a, b=b, text=formatted
            )
        )

    lines.append("  \\end{pgfonlayer}")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def _build_tick_map(
    ticks: Sequence[Tuple[str, str, int]],
    points: Mapping[str, Tuple[float, float]],
    length_epsilon: float,
) -> Dict[Tuple[str, str], List[int]]:
    mapping: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for a, b, idx in ticks:
        if a not in points or b not in points:
            continue
        if _distance(points[a], points[b]) <= length_epsilon:
            continue
        key = _normalize_edge((a, b))
        bucket = mapping[key]
        if idx not in bucket:
            bucket.append(idx)
    for key in list(mapping.keys()):
        mapping[key].sort()
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
    if kind == "segment":
        edge = _edge_from_data(data.get("points"))
        if not edge or not _points_present(edge, coords):
            return []
        return [
            "\\draw[{styles}] ({a}) -- ({b});".format(
                styles=", ".join(tokens), a=edge[0], b=edge[1]
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


def _scene_bbox_diag(bbox: Tuple[float, float, float, float]) -> float:
    min_x, max_x, min_y, max_y = bbox
    return math.hypot(max_x - min_x, max_y - min_y)


def _collect_segments(plan: RenderPlan) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    seen: set = set()
    for start, end, _ in plan.carriers:
        if start not in plan.points or end not in plan.points:
            continue
        key = _normalize_edge((start, end))
        if key in seen:
            continue
        seen.add(key)
        segments.append((plan.points[start], plan.points[end]))
    for _, oriented in plan.helper_tick_edges.items():
        a, b = oriented
        if a in plan.points and b in plan.points:
            segments.append((plan.points[a], plan.points[b]))
    return segments


def _collect_circle_geoms(plan: RenderPlan) -> List[Tuple[Tuple[float, float], float]]:
    circles: List[Tuple[Tuple[float, float], float]] = []
    for center, through, _ in plan.circles:
        if center not in plan.points or through not in plan.points:
            continue
        radius = _distance(plan.points[center], plan.points[through])
        if radius > 0:
            circles.append((plan.points[center], radius))
    return circles


def _wedge_key(a: str, b: str, c: str) -> Tuple[str, Tuple[str, str]]:
    return (b, tuple(sorted((a, c))))


def _angle_base_radius_pt(
    coords: Mapping[str, Tuple[float, float]],
    a: str,
    b: str,
    c: str,
    pt_per_unit: float,
) -> float:
    if not all(name in coords for name in (a, b, c)):
        return 8.0
    ba = _distance(coords[b], coords[a])
    bc = _distance(coords[b], coords[c])
    min_len_pt = min(ba, bc) * pt_per_unit if pt_per_unit > 0 else 0.0
    base = 0.12 * min_len_pt
    return max(7.0, min(14.0, base if base > 0 else 8.0))


def _format_pt_dimension(value: float) -> str:
    return f"{value:.2f}pt"


def _choose_angle_orientation(
    coords: Mapping[str, Tuple[float, float]],
    a: str,
    b: str,
    c: str,
    target_degrees: Optional[float],
) -> Optional[Tuple[str, str, float]]:
    orientation = _oriented_angle_degrees(coords, a, b, c)
    swapped = _oriented_angle_degrees(coords, c, b, a)
    if orientation is None or swapped is None:
        return None
    candidates = [
        (a, c, _normalise_angle_degrees(orientation)),
        (c, a, _normalise_angle_degrees(swapped)),
    ]
    if target_degrees is not None:
        target = _normalise_angle_degrees(target_degrees)
        best = min(
            candidates,
            key=lambda item: _angular_difference_degrees(item[2], target),
        )
        return best
    tol = 1e-4
    less_than = [cand for cand in candidates if cand[2] < 180.0 - tol]
    if less_than:
        return less_than[0]
    not_reflex = [cand for cand in candidates if cand[2] <= 180.0 + tol]
    if not_reflex:
        return min(not_reflex, key=lambda item: abs(item[2] - 180.0))
    return min(candidates, key=lambda item: item[2])


def _bisector_directions(
    coords: Mapping[str, Tuple[float, float]],
    a: str,
    b: str,
    c: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    v1 = _normalise_vector(_vector(coords[b], coords[a])) if a in coords and b in coords else None
    v2 = _normalise_vector(_vector(coords[b], coords[c])) if c in coords and b in coords else None
    if v1 is None or v2 is None:
        return (0.0, 1.0), (0.0, -1.0)
    internal = _normalise_vector((v1[0] + v2[0], v1[1] + v2[1]))
    external = _normalise_vector((v1[0] - v2[0], v1[1] - v2[1]))
    if internal is None:
        perp = _perp_vector(v1)
        internal = perp if perp is not None else (0.0, 1.0)
    if external is None:
        external = (-internal[0], -internal[1])
    return internal, external


def _approximate_text_length(text: str) -> int:
    cleaned = re.sub(r"\\[a-zA-Z]+", "", text)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("^", "")
    cleaned = cleaned.replace("_", "")
    cleaned = cleaned.strip()
    return max(len(cleaned), 1)


def _estimate_label_dimensions(text: str, pt_per_unit: float) -> Tuple[float, float]:
    length = _approximate_text_length(text)
    width_pt = LABEL_WIDTH_EM * FOOTNOTE_EM_PT * length
    height_pt = LABEL_HEIGHT_EM * FOOTNOTE_EM_PT
    if pt_per_unit <= 0:
        return width_pt, height_pt
    return (width_pt / pt_per_unit, height_pt / pt_per_unit)


def _rect_from_center(
    center: Tuple[float, float], width: float, height: float
) -> Tuple[float, float, float, float]:
    half_w = 0.5 * width
    half_h = 0.5 * height
    return (
        center[0] - half_w,
        center[0] + half_w,
        center[1] - half_h,
        center[1] + half_h,
    )


def _point_in_rect(point: Tuple[float, float], rect: Tuple[float, float, float, float]) -> bool:
    x, y = point
    return rect[0] - 1e-9 <= x <= rect[1] + 1e-9 and rect[2] - 1e-9 <= y <= rect[3] + 1e-9


def _rect_intersects_rect(
    r1: Tuple[float, float, float, float],
    r2: Tuple[float, float, float, float],
) -> bool:
    return not (r1[1] < r2[0] or r2[1] < r1[0] or r1[3] < r2[2] or r2[3] < r1[2])


def _orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> float:
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def _segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)
    if o1 == 0 and _point_in_rect(q1, (min(p1[0], p2[0]), max(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[1], p2[1]))):
        return True
    if o2 == 0 and _point_in_rect(q2, (min(p1[0], p2[0]), max(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[1], p2[1]))):
        return True
    if o3 == 0 and _point_in_rect(p1, (min(q1[0], q2[0]), max(q1[0], q2[0]), min(q1[1], q2[1]), max(q1[1], q2[1]))):
        return True
    if o4 == 0 and _point_in_rect(p2, (min(q1[0], q2[0]), max(q1[0], q2[0]), min(q1[1], q2[1]), max(q1[1], q2[1]))):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _segment_intersects_rect(
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
    rect: Tuple[float, float, float, float],
) -> bool:
    if _point_in_rect(seg_start, rect) or _point_in_rect(seg_end, rect):
        return True
    corners = [
        (rect[0], rect[2]),
        (rect[1], rect[2]),
        (rect[1], rect[3]),
        (rect[0], rect[3]),
    ]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(_segments_intersect(seg_start, seg_end, e1, e2) for e1, e2 in edges)


def _circle_rect_intersects(
    center: Tuple[float, float], radius: float, rect: Tuple[float, float, float, float]
) -> bool:
    closest_x = min(max(center[0], rect[0]), rect[1])
    closest_y = min(max(center[1], rect[2]), rect[3])
    distance = math.hypot(center[0] - closest_x, center[1] - closest_y)
    return distance <= radius + 1e-6


def _label_overlap_flags(
    rect: Tuple[float, float, float, float],
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    circles: Sequence[Tuple[Tuple[float, float], float]],
    placed_boxes: Sequence[Tuple[Tuple[float, float, float, float], str]],
) -> Tuple[bool, bool, bool]:
    overlaps_edges = any(_segment_intersects_rect(seg[0], seg[1], rect) for seg in segments)
    overlaps_circles = any(_circle_rect_intersects(center, radius, rect) for center, radius in circles)
    overlaps_labels = any(_rect_intersects_rect(rect, other) for other, _ in placed_boxes)
    return overlaps_edges, overlaps_circles, overlaps_labels


def _render_angle_marks(
    plan: RenderPlan,
    layout_scale: float,
    rules: Mapping[str, bool],
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    circle_geoms: Sequence[Tuple[Tuple[float, float], float]],
    pt_per_unit: float,
    base_offset_units: float,
    placed_boxes: List[Tuple[Tuple[float, float, float, float], str]],
) -> Tuple[List[str], List[Dict[str, str]], List[Tuple[Tuple[float, float, float, float], str]]]:
    arcs: List[str] = []
    label_entries: List[Dict[str, str]] = []
    boxes = list(placed_boxes)
    coords = plan.points

    wedge_groups: Dict[Tuple[str, Tuple[str, str]], int] = {}
    for idx, group in enumerate(plan.angle_groups):
        arc_count = plan.angle_group_arc_counts[idx] if idx < len(plan.angle_group_arc_counts) else len(group)
        for A, B, C in group:
            key = _wedge_key(A, B, C)
            wedge_groups[key] = max(wedge_groups.get(key, 0), arc_count)

    for idx, group in enumerate(plan.angle_groups):
        arc_count = plan.angle_group_arc_counts[idx] if idx < len(plan.angle_group_arc_counts) else len(group)
        for A, B, C in group:
            if not _points_present((A, B, C), coords):
                continue
            choice = _choose_angle_orientation(coords, A, B, C, None)
            if not choice:
                continue
            start, end, measure = choice
            if measure < 6.0:
                continue
            base_radius_pt = _angle_base_radius_pt(coords, A, B, C, pt_per_unit)
            for arc_idx in range(arc_count):
                radius_pt = base_radius_pt + arc_idx * GS_ANGLE_SEP_PT
                arcs.append(
                    f"\\path pic[draw, angle radius={_format_pt_dimension(radius_pt)}] {{angle={start}--{B}--{end}}};"
                )

    numeric_vertices: set = set()
    for entry in plan.angles:
        kind = entry.get("kind")
        A = entry.get("A")
        B = entry.get("B")
        C = entry.get("C")
        if not (isinstance(A, str) and isinstance(B, str) and isinstance(C, str)):
            continue
        if kind != "numeric":
            continue
        if B in numeric_vertices or not _points_present((A, B, C), coords):
            continue
        choice = _choose_angle_orientation(coords, A, B, C, entry.get("degrees"))
        if not choice:
            continue
        start, end, measure = choice
        base_radius_pt = _angle_base_radius_pt(coords, A, B, C, pt_per_unit)
        offset_count = wedge_groups.get(_wedge_key(A, B, C), 0)
        initial_radius_pt = base_radius_pt + offset_count * GS_ANGLE_SEP_PT
        radius_pt, label_entry, boxes = _place_numeric_angle_label(
            coords,
            A,
            B,
            C,
            initial_radius_pt,
            entry.get("label"),
            measure,
            segments,
            circle_geoms,
            boxes,
            pt_per_unit,
            base_offset_units,
        )
        arcs.append(
            f"\\path pic[draw, angle radius={_format_pt_dimension(radius_pt)}] {{angle={start}--{B}--{end}}};"
        )
        if label_entry:
            label_entries.append(label_entry)
        numeric_vertices.add(B)

    for A, B, C in plan.right_angles:
        if not _points_present((A, B, C), coords):
            continue
        arcs.append(f"\\path pic[draw, angle radius=\\gsAngR] {{right angle={A}--{B}--{C}}};")

    return arcs, label_entries, boxes


def _place_numeric_angle_label(
    coords: Mapping[str, Tuple[float, float]],
    a: str,
    b: str,
    c: str,
    initial_radius_pt: float,
    label_text: Optional[str],
    measure: float,
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    circles: Sequence[Tuple[Tuple[float, float], float]],
    placed_boxes: List[Tuple[Tuple[float, float, float, float], str]],
    pt_per_unit: float,
    base_offset_units: float,
) -> Tuple[float, Optional[Dict[str, str]], List[Tuple[Tuple[float, float, float, float], str]]]:
    if not isinstance(label_text, str) or not label_text.strip():
        return initial_radius_pt, None, placed_boxes

    width, height = _estimate_label_dimensions(label_text, pt_per_unit)
    internal_dir, external_dir = _bisector_directions(coords, a, b, c)
    label_offset_pt = ANGLE_LABEL_OFFSET_EM * FOOTNOTE_EM_PT

    radius_pt = initial_radius_pt
    final_center: Optional[Tuple[float, float]] = None
    final_rect: Optional[Tuple[float, float, float, float]] = None
    boxes = list(placed_boxes)
    overlaps_major = False

    for attempt in range(3):
        radius_pt = initial_radius_pt + attempt * GS_ANGLE_SEP_PT
        label_radius_pt = radius_pt + label_offset_pt
        label_radius_units = label_radius_pt / pt_per_unit if pt_per_unit > 0 else radius_pt
        center = (
            coords[b][0] + internal_dir[0] * label_radius_units,
            coords[b][1] + internal_dir[1] * label_radius_units,
        )
        rect = _rect_from_center(center, width, height)
        overlaps = _label_overlap_flags(rect, segments, circles, boxes)
        if not any(overlaps):
            final_center = center
            final_rect = rect
            break
        overlaps_major = overlaps_major or overlaps[0] or overlaps[1] or overlaps[2]

    leader_line: Optional[str] = None
    if final_center is None:
        overlaps_major = True
        for attempt in range(3):
            radius_pt = initial_radius_pt + (3 + attempt) * GS_ANGLE_SEP_PT
            label_radius_pt = radius_pt + label_offset_pt
            label_radius_units = label_radius_pt / pt_per_unit if pt_per_unit > 0 else radius_pt
            center = (
                coords[b][0] + external_dir[0] * label_radius_units,
                coords[b][1] + external_dir[1] * label_radius_units,
            )
            rect = _rect_from_center(center, width, height)
            overlaps = _label_overlap_flags(rect, segments, circles, boxes)
            if not any(overlaps):
                final_center = center
                final_rect = rect
                break
        if final_center is None:
            label_radius_pt = radius_pt + label_offset_pt
            label_radius_units = label_radius_pt / pt_per_unit if pt_per_unit > 0 else radius_pt
            final_center = (
                coords[b][0] + external_dir[0] * label_radius_units,
                coords[b][1] + external_dir[1] * label_radius_units,
            )
            final_rect = _rect_from_center(final_center, width, height)
        leader_length = base_offset_units * 0.6 if base_offset_units > 0 else 0.0
        if leader_length <= 0:
            leader_length = min(
                math.hypot(width, height) * 0.6,
                max(0.6, math.hypot(width, height)),
            )
        leader_tip = (
            coords[b][0] + external_dir[0] * leader_length,
            coords[b][1] + external_dir[1] * leader_length,
        )
        leader_line = "\\draw[aux] ({b}) -- ({x}, {y});".format(
            b=b,
            x=_format_float(leader_tip[0]),
            y=_format_float(leader_tip[1]),
        )

    if final_center is None or final_rect is None:
        return radius_pt, None, boxes

    node_line = "\\node[ptlabel] at ({x}, {y}) {{{text}}};".format(
        x=_format_float(final_center[0]),
        y=_format_float(final_center[1]),
        text=label_text,
    )
    boxes.append((final_rect, "angle"))
    entry: Dict[str, str] = {"node": node_line}
    if leader_line:
        entry["leader"] = leader_line
    return radius_pt, entry, boxes


def _points_present(names: Sequence[str], coords: Mapping[str, Tuple[float, float]]) -> bool:
    return all(isinstance(name, str) and name in coords for name in names)


def _build_incident_lines_map(plan: RenderPlan) -> Dict[str, List[Tuple[float, float]]]:
    mapping: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for start, end, _ in plan.carriers:
        if start not in plan.points or end not in plan.points:
            continue
        vec = _normalise_vector(_vector(plan.points[start], plan.points[end]))
        if vec is None:
            continue
        mapping[start].append(vec)
        mapping[end].append((-vec[0], -vec[1]))
    for _, oriented in plan.helper_tick_edges.items():
        a, b = oriented
        if a in plan.points and b in plan.points:
            vec = _normalise_vector(_vector(plan.points[a], plan.points[b]))
            if vec is None:
                continue
            mapping[a].append(vec)
            mapping[b].append((-vec[0], -vec[1]))
    return mapping


def _build_incident_circles_map(plan: RenderPlan) -> Dict[str, List[Tuple[float, float]]]:
    mapping: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for center, through, _ in plan.circles:
        if center not in plan.points or through not in plan.points:
            continue
        vec = _normalise_vector(_vector(plan.points[center], plan.points[through]))
        if vec is None:
            continue
        mapping[through].append(vec)
        mapping[center].append((-vec[0], -vec[1]))
    return mapping


def _anchor_direction(anchor: str) -> Optional[Tuple[float, float]]:
    key = anchor.strip().lower()
    return ANCHOR_DIRECTIONS.get(key)


def _point_priority_order(plan: RenderPlan) -> List[str]:
    order: List[str] = []
    seen: set = set()
    for name in sorted(plan.circle_centers):
        if name in plan.points and name not in seen:
            order.append(name)
            seen.add(name)
    for name in sorted(plan.polygon_vertices):
        if name in plan.points and name not in seen:
            order.append(name)
            seen.add(name)
    for name in sorted(plan.special_points):
        if name in plan.points and name not in seen:
            order.append(name)
            seen.add(name)
    for name in sorted(plan.points.keys()):
        if name not in seen:
            order.append(name)
            seen.add(name)
    return order


def _layout_point_labels(
    plan: RenderPlan,
    point_label_map: Mapping[str, LabelSpec],
    bbox: Tuple[float, float, float, float],
    base_offset_units: float,
    pt_per_unit: float,
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    circle_geoms: Sequence[Tuple[Tuple[float, float], float]],
    placed_boxes: List[Tuple[Tuple[float, float, float, float], str]],
) -> Tuple[Dict[str, Dict[str, Optional[str]]], List[Tuple[Tuple[float, float, float, float], str]]]:
    inc_lines = _build_incident_lines_map(plan)
    inc_circles = _build_incident_circles_map(plan)
    center_circle_geoms: Dict[str, List[Tuple[Tuple[float, float], float]]] = defaultdict(list)
    for center, through, _ in plan.circles:
        if center not in plan.points or through not in plan.points:
            continue
        radius = _distance(plan.points[center], plan.points[through])
        if radius <= 0:
            continue
        center_circle_geoms[center].append((plan.points[center], radius))
    boxes = list(placed_boxes)
    result: Dict[str, Dict[str, Optional[str]]] = {}
    offset_units = base_offset_units if base_offset_units > 0 else 0.05

    for name in _point_priority_order(plan):
        if name not in plan.points:
            continue
        point = plan.points[name]
        label_spec = point_label_map.get(name)
        raw_text = label_spec.text if label_spec else name
        formatted = _format_label_text(raw_text)
        explicit_anchor = label_spec.position if label_spec and label_spec.position else None
        width, height = _estimate_label_dimensions(raw_text, pt_per_unit)

        if explicit_anchor:
            direction = _anchor_direction(explicit_anchor) or (0.0, 1.0)
            center = (
                point[0] + direction[0] * offset_units,
                point[1] + direction[1] * offset_units,
            )
            rect = _rect_from_center(center, width, height)
            boxes.append((rect, "point"))
            result[name] = {"anchor": explicit_anchor, "text": formatted, "leader": None}
            continue

        effective_circle_geoms = circle_geoms
        skip_geoms = center_circle_geoms.get(name)
        if skip_geoms:
            effective_circle_geoms = [
                geom for geom in circle_geoms if geom not in skip_geoms
            ]
        anchor, center, w, h, leader_end, _ = _place_point_label(
            name,
            point,
            raw_text,
            offset_units,
            segments,
            effective_circle_geoms,
            boxes,
            inc_lines.get(name, []),
            inc_circles.get(name, []),
            plan.points,
            pt_per_unit,
        )
        rect = _rect_from_center(center, w, h)
        boxes.append((rect, "point"))
        leader_line = None
        if leader_end is not None:
            leader_line = "\\draw[aux] ({name}) -- ({x}, {y});".format(
                name=name,
                x=_format_float(leader_end[0]),
                y=_format_float(leader_end[1]),
            )
        result[name] = {"anchor": anchor, "text": formatted, "leader": leader_line}

    return result, boxes


def _place_point_label(
    name: str,
    point: Tuple[float, float],
    raw_text: str,
    base_offset_units: float,
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    circle_geoms: Sequence[Tuple[Tuple[float, float], float]],
    placed_boxes: Sequence[Tuple[Tuple[float, float, float, float], str]],
    inc_lines: Sequence[Tuple[float, float]],
    inc_circles: Sequence[Tuple[float, float]],
    all_points: Mapping[str, Tuple[float, float]],
    pt_per_unit: float,
) -> Tuple[str, Tuple[float, float], float, float, Optional[Tuple[float, float]], bool]:
    width, height = _estimate_label_dimensions(raw_text, pt_per_unit)
    offset = base_offset_units
    best_choice: Optional[Tuple[float, bool, str, Tuple[float, float], Tuple[float, float, float, float], Tuple[float, float]]] = None

    for attempt in range(3):
        candidates: List[Tuple[float, bool, str, Tuple[float, float], Tuple[float, float, float, float], Tuple[float, float]]] = []
        for anchor in ANCHOR_SEQUENCE:
            direction = _anchor_direction(anchor)
            if direction is None:
                continue
            center = (
                point[0] + direction[0] * offset,
                point[1] + direction[1] * offset,
            )
            rect = _rect_from_center(center, width, height)
            overlaps = _label_overlap_flags(rect, segments, circle_geoms, placed_boxes)
            inside_circle = any(_dot(direction, normal) < -0.05 for normal in inc_circles)
            crowded = _crowded_penalty(name, point, direction, all_points, offset)
            score = (
                12 * int(overlaps[0])
                + 10 * int(overlaps[1])
                + 8 * int(overlaps[2])
                + 4 * int(inside_circle)
                + 2 * crowded
                - _alignment_bonus(direction, inc_lines, inc_circles)
            )
            candidates.append((score, any(overlaps), anchor, center, rect, direction))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], ANCHOR_SEQUENCE.index(item[2])))
        best_choice = candidates[0]
        if not best_choice[1]:
            break
        offset *= 1.4

    if best_choice is None:
        anchor = "above"
        center = (point[0], point[1] + offset)
        rect = _rect_from_center(center, width, height)
        return anchor, center, width, height, None, False

    overlaps_major = bool(best_choice[1])
    anchor = best_choice[2]
    center = best_choice[3]
    rect = best_choice[4]
    direction = best_choice[5]

    leader_end: Optional[Tuple[float, float]] = None
    if overlaps_major:
        leader_length = max(0.8 * base_offset_units, 0.5 * max(width, height))
        leader_end = (
            point[0] + direction[0] * leader_length,
            point[1] + direction[1] * leader_length,
        )

    return anchor, center, width, height, leader_end, overlaps_major


def _crowded_penalty(
    name: str,
    point: Tuple[float, float],
    direction: Tuple[float, float],
    all_points: Mapping[str, Tuple[float, float]],
    offset: float,
) -> int:
    threshold = max(offset * 1.4, offset + 1e-3)
    dir_vec = _normalise_vector(direction) or direction
    for other_name, other in all_points.items():
        if other_name == name:
            continue
        vec = (other[0] - point[0], other[1] - point[1])
        dist = math.hypot(vec[0], vec[1])
        if dist < 1e-6:
            continue
        if dist < threshold:
            unit = (vec[0] / dist, vec[1] / dist)
            if _dot(unit, dir_vec) > 0.6:
                return 1
    return 0


def _alignment_bonus(
    direction: Tuple[float, float],
    inc_lines: Sequence[Tuple[float, float]],
    inc_circles: Sequence[Tuple[float, float]],
) -> float:
    bonus = 0.0
    for vec in inc_lines:
        if abs(vec[1]) < 0.35 and abs(direction[1]) >= abs(direction[0]):
            bonus += 1.0
        if abs(vec[0]) < 0.35 and abs(direction[0]) > abs(direction[1]):
            bonus += 1.0
    for normal in inc_circles:
        if _dot(direction, normal) > 0.2:
            bonus += 0.5
    return min(bonus, 2.0)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


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
