from .ast import Program
from .numbers import SymbolicNumber
from typing import Tuple

def edge_str(e: Tuple[str,str]) -> str:
    return f'{e[0]}-{e[1]}'

def path_str(path: Tuple[str, object]) -> str:
    kind, payload = path
    if kind in {'line', 'ray', 'segment'} and isinstance(payload, (list, tuple)):
        return f'{kind} {edge_str(payload)}'
    if kind == 'circle' and isinstance(payload, str):
        return f'circle center {payload}'
    if kind == 'angle-bisector' and isinstance(payload, dict):
        at = payload.get('at', '')
        r1, r2 = payload.get('rays', (None, None))
        if isinstance(r1, (list, tuple)) and isinstance(r2, (list, tuple)):
            return f'angle-bisector at {at} rays {edge_str(r1)} {edge_str(r2)}'
        return f'angle-bisector at {at}'
    if kind == 'perpendicular' and isinstance(payload, dict):
        at = payload.get('at', '')
        to_edge = payload.get('to')
        if isinstance(to_edge, (list, tuple)):
            return f'perpendicular at {at} to {edge_str(to_edge)}'
        return f'perpendicular at {at}'
    if kind == 'median' and isinstance(payload, dict):
        frm = payload.get('frm', '')
        to_edge = payload.get('to')
        if isinstance(to_edge, (list, tuple)):
            return f'median from {frm} to {edge_str(to_edge)}'
        return f'median from {frm}'
    return f'# [unknown path {kind}]'

def print_program(prog: Program) -> str:
    lines = []
    for s in prog.stmts:
        o = ''
        if s.opts:
            parts = []
            for k in sorted(s.opts.keys()):
                v = s.opts[k]
                if isinstance(v, bool):
                    vv = 'true' if v else 'false'
                elif isinstance(v, (int,float)):
                    vv = str(v)
                elif isinstance(v, SymbolicNumber):
                    vv = str(v)
                else:
                    vv = v if (isinstance(v,str) and ' ' not in v) else f'"{v}"'
                parts.append(f'{k}={vv}')
            o = f' [{" ".join(parts)}]'
        if s.kind == 'scene':
            lines.append(f'scene "{s.data["title"]}"')
        elif s.kind == 'layout':
            lines.append(f'layout canonical={s.data["canonical"]} scale={s.data["scale"]}')
        elif s.kind == 'points':
            lines.append('points ' + ', '.join(s.data['ids']))
        elif s.kind == 'segment':
            lines.append(f'segment {edge_str(s.data["edge"])}')
        elif s.kind == 'ray':
            lines.append(f'ray {edge_str(s.data["ray"])}')
        elif s.kind == 'line':
            lines.append(f'line {edge_str(s.data["edge"])}')
        elif s.kind == 'line_tangent_at':
            lines.append(f'line {edge_str(s.data["edge"])} tangent to circle center {s.data["center"]} at {s.data["at"]}{o}'); continue
        elif s.kind == 'circle_center_radius_through':
            lines.append(f'circle center {s.data["center"]} radius-through {s.data["through"]}{o}'); continue
        elif s.kind == 'circle_center_tangent_sides':
            es = ', '.join(edge_str(e) for e in s.data['tangent_edges'])
            lines.append(f'circle center {s.data["center"]} tangent ({es}){o}'); continue
        elif s.kind == 'circle_through':
            ids = ', '.join(s.data['ids'])
            lines.append(f'circle through ({ids})')
        elif s.kind == 'circumcircle':
            chain = '-'.join(s.data['ids'])
            lines.append(f'circumcircle of {chain}')
        elif s.kind == 'incircle':
            chain = '-'.join(s.data['ids'])
            lines.append(f'incircle of {chain}')
        elif s.kind == 'perpendicular_at':
            lines.append(f'perpendicular at {s.data["at"]} to {edge_str(s.data["to"])}')
        elif s.kind == 'parallel_through':
            lines.append(f'parallel through {s.data["through"]} to {edge_str(s.data["to"])}')
        elif s.kind == 'angle_bisector_at':
            r1, r2 = s.data['rays']; lines.append(f'angle-bisector at {s.data["at"]} rays {edge_str(r1)} {edge_str(r2)}{o}'); continue
        elif s.kind == 'median_from_to':
            lines.append(f'median from {s.data["frm"]} to {edge_str(s.data["to"])}')
        elif s.kind == 'altitude_from_to':
            lines.append(f'altitude from {s.data["frm"]} to {edge_str(s.data["to"])}')
        elif s.kind == 'angle_at':
            r1, r2 = s.data['rays']; lines.append(f'angle at {s.data["at"]} rays {edge_str(r1)} {edge_str(r2)}{o}'); continue
        elif s.kind == 'right_angle_at':
            r1, r2 = s.data['rays']; lines.append(f'right-angle at {s.data["at"]} rays {edge_str(r1)} {edge_str(r2)}{o}'); continue
        elif s.kind == 'equal_segments':
            lhs = ', '.join(edge_str(e) for e in s.data['lhs'])
            rhs = ', '.join(edge_str(e) for e in s.data['rhs'])
            lines.append(f'equal-segments ({lhs} ; {rhs})')
        elif s.kind == 'tangent_at':
            lines.append(f'tangent at {s.data["at"]} to circle center {s.data["center"]}{o}'); continue
        elif s.kind == 'diameter':
            lines.append(
                f'diameter {edge_str(s.data["edge"])} to circle center {s.data["center"]}{o}'
            );
            continue
        elif s.kind == 'polygon':
            ids = '-'.join(s.data['ids']); lines.append(f'polygon {ids}{o}'); continue
        elif s.kind in ('triangle','quadrilateral','parallelogram','trapezoid','rectangle','square','rhombus'):
            ids = '-'.join(s.data['ids']); lines.append(f'{s.kind} {ids}{o}'); continue
        elif s.kind == 'point_on':
            lines.append(f'point {s.data["point"]} on {path_str(s.data["path"])}{o}'); continue
        elif s.kind == 'intersect':
            second = f', {s.data["at2"]}' if s.data['at2'] else ''
            lines.append(
                f'intersect ({path_str(s.data["path1"])}) with ({path_str(s.data["path2"])}) '
                f'at {s.data["at"]}{second}{o}'
            );
            continue
        elif s.kind == 'label_point':
            lines.append(f'label point {s.data["point"]}{o}'); continue
        elif s.kind == 'sidelabel':
            lines.append(f'sidelabel {edge_str(s.data["edge"])} "{s.data["text"]}"{o}'); continue
        elif s.kind == 'target_angle':
            r1, r2 = s.data['rays']; lines.append(f'target angle at {s.data["at"]} rays {edge_str(r1)} {edge_str(r2)}{o}'); continue
        elif s.kind == 'target_length':
            lines.append(f'target length {edge_str(s.data["edge"])}{o}'); continue
        elif s.kind == 'target_point':
            lines.append(f'target point {s.data["point"]}{o}'); continue
        elif s.kind == 'target_circle':
            lines.append(f'target circle ({s.data["text"]}){o}'); continue
        elif s.kind == 'target_area':
            lines.append(f'target area ({s.data["text"]}){o}'); continue
        elif s.kind == 'target_arc':
            lines.append(f'target arc {s.data["A"]}-{s.data["B"]} on circle center {s.data["center"]}{o}'); continue
        elif s.kind == 'parallel_edges':
            a, b = s.data['edges']
            lines.append(f'parallel-edges ({edge_str(a)} ; {edge_str(b)}){o}')
            continue
        elif s.kind == 'rules':
            parts = [f'{k}={"true" if s.opts[k] else "false"}' for k in sorted(s.opts.keys())]
            lines.append('rules ' + ' '.join(parts)); continue
        else:
            lines.append(f'# [unknown kind {s.kind}]'); continue
        if o and lines:
            lines[-1] += o
    return '\n'.join(lines) + "\n"
