from .ast import Program
from typing import Tuple

def edge_str(e: Tuple[str,str]) -> str:
    return f'{e[0]}-{e[1]}'

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
            a,b,c = s.data['tri']; lines.append(f'circumcircle of {a}-{b}-{c}')
        elif s.kind == 'incircle':
            a,b,c = s.data['tri']; lines.append(f'incircle of {a}-{b}-{c}')
        elif s.kind == 'perpendicular_at':
            lines.append(f'perpendicular at {s.data["at"]} to {edge_str(s.data["to"])}')
        elif s.kind == 'parallel_through':
            lines.append(f'parallel through {s.data["through"]} to {edge_str(s.data["to"])}')
        elif s.kind == 'bisector_at':
            lines.append(f'bisector at {s.data["at"]}')
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
        elif s.kind == 'polygon':
            ids = '-'.join(s.data['ids']); lines.append(f'polygon {ids}{o}'); continue
        elif s.kind in ('triangle','quadrilateral','parallelogram','trapezoid','rectangle','square','rhombus'):
            ids = '-'.join(s.data['ids']); lines.append(f'{s.kind} {ids}{o}'); continue
        elif s.kind == 'point_on':
            kind, val = s.data['path']
            pstr = f'circle center {val}' if kind=='circle' else f'{kind} {edge_str(val)}'
            lines.append(f'point {s.data["point"]} on {pstr}{o}'); continue
        elif s.kind == 'intersect':
            def p2s(p):
                k, v = p; return f'{k} {v if k=="circle" else edge_str(v)}'
            second = f', {s.data["at2"]}' if s.data['at2'] else ''
            lines.append(f'intersect ({p2s(s.data["path1"])}) with ({p2s(s.data["path2"])}) at {s.data["at"]}{second}{o}'); continue
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
            a,b = s.data['edges']; lines.append(f'# desugared: {edge_str(a)} ∥ {edge_str(b)}'); continue
        elif s.kind == 'rules':
            parts = [f'{k}={"true" if s.opts[k] else "false"}' for k in sorted(s.opts.keys())]
            lines.append('rules ' + ' '.join(parts)); continue
        else:
            lines.append(f'# [unknown kind {s.kind}]'); continue
        if o and lines:
            lines[-1] += o
    return '\n'.join(lines)