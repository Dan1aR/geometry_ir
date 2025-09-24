import math

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
