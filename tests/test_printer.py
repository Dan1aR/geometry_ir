import math

import pytest

from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.numbers import SymbolicNumber
from geoscript_ir.printer import print_program


def test_parallel_edges_prints_canonical_form():
    stmt = Stmt('parallel_edges', Span(1, 1), {'edges': [('A', 'B'), ('C', 'D')]})
    prog = Program([stmt])

    assert print_program(prog) == 'parallel-edges (A-B ; C-D)\n'


def test_segment_length_prints_symbolic_value():
    stmt = Stmt('segment', Span(1, 1), {'edge': ('A', 'B')}, {'length': SymbolicNumber('sqrt(19)', math.sqrt(19))})
    prog = Program([stmt])

    assert print_program(prog) == 'segment A-B [length=sqrt(19)]\n'


def test_diameter_prints_statement():
    stmt = Stmt('diameter', Span(1, 1), {'edge': ('A', 'B'), 'center': 'O'})
    prog = Program([stmt])

    assert print_program(prog) == 'diameter A-B to circle center O\n'


def test_rules_prints_bracketed_options():
    stmt = Stmt('rules', Span(1, 1), {}, {'allow_auxiliary': False, 'no_solving': True})
    prog = Program([stmt])

    assert print_program(prog) == 'rules [allow_auxiliary=false no_solving=true]\n'


def test_original_only_skips_generated_statements():
    original = Stmt('segment', Span(1, 1), {'edge': ('A', 'B')})
    generated = Stmt(
        'segment',
        Span(2, 1),
        {'edge': ('B', 'C')},
        origin='desugar(triangle)',
    )
    prog = Program([original, generated])

    assert print_program(prog, original_only=True) == 'segment A-B\n'


def test_midpoint_and_foot_print_primitives():
    midpoint = Stmt('midpoint', Span(1, 1), {'midpoint': 'M', 'edge': ('A', 'B')})
    foot = Stmt('foot', Span(2, 1), {'foot': 'H', 'from': 'C', 'edge': ('A', 'B')})
    prog = Program([midpoint, foot])

    assert print_program(prog) == 'midpoint M of A-B\nfoot H from C to A-B\n'


def test_unknown_kind_raises_value_error():
    stmt = Stmt('mystery', Span(1, 1), {})
    prog = Program([stmt])

    with pytest.raises(ValueError):
        print_program(prog)
