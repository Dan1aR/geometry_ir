from typing import Dict, Iterable, Tuple

from .ast import Program, Stmt
from .numbers import SymbolicNumber


def edge_str(edge: Tuple[str, str]) -> str:
    return f"{edge[0]}-{edge[1]}"


def angle_str(points: Tuple[str, str, str]) -> str:
    return f"{points[0]}-{points[1]}-{points[2]}"


def _format_opts(opts: Dict[str, object]) -> str:
    if not opts:
        return ""
    parts = []
    for key in sorted(opts.keys()):
        value = opts[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        elif isinstance(value, SymbolicNumber):
            rendered = str(value)
        else:
            rendered = value if (isinstance(value, str) and " " not in value) else f'"{value}"'
        parts.append(f"{key}={rendered}")
    return " [" + " ".join(parts) + "]"


def _require_edge(payload: object, *, kind: str) -> Tuple[str, str]:
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError(f"invalid {kind} payload {payload!r}")
    a, b = payload
    if not (isinstance(a, str) and isinstance(b, str)):
        raise ValueError(f"invalid {kind} payload {payload!r}")
    return str(a), str(b)


def path_str(path: Tuple[str, object]) -> str:
    kind, payload = path
    if kind in {"line", "ray", "segment"}:
        return f"{kind} {edge_str(_require_edge(payload, kind=kind))}"
    if kind == "circle":
        if not isinstance(payload, str):
            raise ValueError(f"invalid circle payload {payload!r}")
        return f"circle center {payload}"
    if kind == "angle-bisector":
        if not isinstance(payload, dict):
            raise ValueError(f"invalid angle-bisector payload {payload!r}")
        pts = payload.get("points")
        extra = " external" if payload.get("external") else ""
        if not isinstance(pts, (list, tuple)) or len(pts) != 3:
            raise ValueError(f"invalid angle-bisector points {pts!r}")
        return f"angle-bisector {angle_str(tuple(pts))}{extra}"
    if kind == "perpendicular":
        if not isinstance(payload, dict):
            raise ValueError(f"invalid perpendicular payload {payload!r}")
        at = payload.get("at")
        to_edge = payload.get("to")
        if not isinstance(at, str):
            raise ValueError(f"invalid perpendicular at-point {at!r}")
        edge = _require_edge(to_edge, kind="perpendicular to")
        return f"perpendicular at {at} to {edge_str(edge)}"
    if kind == "median":
        if not isinstance(payload, dict):
            raise ValueError(f"invalid median payload {payload!r}")
        frm = payload.get("frm")
        to_edge = payload.get("to")
        if not isinstance(frm, str):
            raise ValueError(f"invalid median source {frm!r}")
        edge = _require_edge(to_edge, kind="median to")
        return f"median from {frm} to {edge_str(edge)}"
    if kind == "perp-bisector":
        edge = _require_edge(payload, kind="perp-bisector")
        return f"perp-bisector of {edge_str(edge)}"
    if kind == "parallel":
        if not isinstance(payload, dict):
            raise ValueError(f"invalid parallel payload {payload!r}")
        through = payload.get("through")
        to_edge = payload.get("to")
        if not isinstance(through, str):
            raise ValueError(f"invalid parallel through-point {through!r}")
        edge = _require_edge(to_edge, kind="parallel to")
        return f"parallel through {through} to {edge_str(edge)}"
    raise ValueError(f"unknown path kind {kind!r}")


def _edge_chain(ids: Iterable[str]) -> str:
    return "-".join(ids)


def print_program(prog: Program, *, original_only: bool = False) -> str:
    lines = []
    for stmt in prog.stmts:
        if original_only and stmt.origin != "source":
            continue

        opts_suffix = _format_opts(stmt.opts)
        line: str

        if stmt.kind == "scene":
            line = f"scene \"{stmt.data['title']}\""
        elif stmt.kind == "layout":
            line = (
                f"layout canonical={stmt.data['canonical']} "
                f"scale={stmt.data['scale']}"
            )
        elif stmt.kind == "points":
            line = "points " + ", ".join(stmt.data["ids"])
        elif stmt.kind == "segment":
            line = f"segment {edge_str(stmt.data['edge'])}"
        elif stmt.kind == "ray":
            line = f"ray {edge_str(stmt.data['ray'])}"
        elif stmt.kind == "line":
            line = f"line {edge_str(stmt.data['edge'])}"
        elif stmt.kind == "line_tangent_at":
            line = (
                f"line {edge_str(stmt.data['edge'])} tangent to circle center "
                f"{stmt.data['center']} at {stmt.data['at']}"
            )
        elif stmt.kind == "circle_center_radius_through":
            line = (
                f"circle center {stmt.data['center']} radius-through "
                f"{stmt.data['through']}"
            )
        elif stmt.kind == "circle_through":
            line = "circle through (" + ", ".join(stmt.data["ids"]) + ")"
        elif stmt.kind == "circumcircle":
            line = f"circumcircle of {_edge_chain(stmt.data['ids'])}"
        elif stmt.kind == "incircle":
            line = f"incircle of {_edge_chain(stmt.data['ids'])}"
        elif stmt.kind == "perpendicular_at":
            line = (
                f"perpendicular at {stmt.data['at']} to {edge_str(stmt.data['to'])} "
                f"foot {stmt.data['foot']}"
            )
        elif stmt.kind == "parallel_through":
            line = (
                f"parallel through {stmt.data['through']} to "
                f"{edge_str(stmt.data['to'])}"
            )
        elif stmt.kind == "angle_bisector_at":
            points = stmt.data.get("points")
            if isinstance(points, (list, tuple)) and len(points) == 3:
                line = f"angle-bisector {angle_str(tuple(points))}"
            else:
                ray1, ray2 = stmt.data.get("rays", (None, None))
                if ray1 and ray2:
                    line = (
                        f"angle-bisector at {stmt.data.get('at', '')} rays "
                        f"{edge_str(ray1)} {edge_str(ray2)}"
                    )
                else:
                    line = "angle-bisector"
        elif stmt.kind == "median_from_to":
            line = (
                f"median from {stmt.data['frm']} to {edge_str(stmt.data['to'])} "
                f"midpoint {stmt.data['midpoint']}"
            )
        elif stmt.kind == "midpoint":
            line = (
                f"midpoint {stmt.data['midpoint']} of "
                f"{edge_str(stmt.data['edge'])}"
            )
        elif stmt.kind == "foot":
            line = (
                f"foot {stmt.data['foot']} from {stmt.data['from']} to "
                f"{edge_str(stmt.data['edge'])}"
            )
        elif stmt.kind == "angle_at":
            line = f"angle {angle_str(tuple(stmt.data['points']))}"
        elif stmt.kind == "right_angle_at":
            line = f"right-angle {angle_str(tuple(stmt.data['points']))}"
        elif stmt.kind == "equal_segments":
            lhs = ", ".join(edge_str(edge) for edge in stmt.data["lhs"])
            rhs = ", ".join(edge_str(edge) for edge in stmt.data["rhs"])
            line = f"equal-segments ({lhs} ; {rhs})"
        elif stmt.kind == "collinear":
            line = "collinear (" + ", ".join(stmt.data["points"]) + ")"
        elif stmt.kind == "concyclic":
            line = "concyclic (" + ", ".join(stmt.data["points"]) + ")"
        elif stmt.kind == "equal_angles":
            lhs = ", ".join(angle_str(tuple(ang)) for ang in stmt.data["lhs"])
            rhs = ", ".join(angle_str(tuple(ang)) for ang in stmt.data["rhs"])
            line = f"equal-angles ({lhs} ; {rhs})"
        elif stmt.kind == "ratio":
            left, right = stmt.data["edges"]
            a, b = stmt.data["ratio"]

            def _fmt_ratio_part(value: object) -> str:
                if isinstance(value, float) and value.is_integer():
                    return str(int(value))
                return str(value)

            line = (
                f"ratio ({edge_str(left)} : {edge_str(right)} = "
                f"{_fmt_ratio_part(a)} : {_fmt_ratio_part(b)})"
            )
        elif stmt.kind == "tangent_at":
            line = (
                f"tangent at {stmt.data['at']} to circle center "
                f"{stmt.data['center']}"
            )
        elif stmt.kind == "diameter":
            line = (
                f"diameter {edge_str(stmt.data['edge'])} to circle center "
                f"{stmt.data['center']}"
            )
        elif stmt.kind == "polygon":
            line = f"polygon {_edge_chain(stmt.data['ids'])}"
        elif stmt.kind in {
            "triangle",
            "quadrilateral",
            "parallelogram",
            "trapezoid",
            "rectangle",
            "square",
            "rhombus",
        }:
            line = f"{stmt.kind} {_edge_chain(stmt.data['ids'])}"
        elif stmt.kind == "point_on":
            line = f"point {stmt.data['point']} on {path_str(stmt.data['path'])}"
        elif stmt.kind == "intersect":
            at2 = f", {stmt.data['at2']}" if stmt.data['at2'] else ""
            line = (
                f"intersect ({path_str(stmt.data['path1'])}) with "
                f"({path_str(stmt.data['path2'])}) at {stmt.data['at']}{at2}"
            )
        elif stmt.kind == "label_point":
            line = f"label point {stmt.data['point']}"
        elif stmt.kind == "sidelabel":
            line = f"sidelabel {edge_str(stmt.data['edge'])} \"{stmt.data['text']}\""
        elif stmt.kind == "target_angle":
            line = f"target angle {angle_str(tuple(stmt.data['points']))}"
        elif stmt.kind == "target_length":
            line = f"target length {edge_str(stmt.data['edge'])}"
        elif stmt.kind == "target_point":
            line = f"target point {stmt.data['point']}"
        elif stmt.kind == "target_circle":
            line = f"target circle ({stmt.data['text']})"
        elif stmt.kind == "target_area":
            line = f"target area ({stmt.data['text']})"
        elif stmt.kind == "target_arc":
            line = (
                f"target arc {stmt.data['A']}-{stmt.data['B']} on circle center "
                f"{stmt.data['center']}"
            )
        elif stmt.kind == "parallel_edges":
            a, b = stmt.data["edges"]
            line = f"parallel-edges ({edge_str(a)} ; {edge_str(b)})"
        elif stmt.kind == "rules":
            if not stmt.opts:
                raise ValueError("rules statement requires at least one option")
            lines.append("rules" + _format_opts(stmt.opts))
            continue
        else:
            raise ValueError(f"unknown statement kind {stmt.kind!r}")

        if opts_suffix:
            line += opts_suffix
        lines.append(line)

    return "\n".join(lines) + "\n"


def format_stmt(stmt: Stmt) -> str:
    """Return a single-line representation of ``stmt``."""

    return print_program(Program(stmts=[stmt])).strip()
