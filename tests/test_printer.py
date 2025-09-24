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


def test_diameter_prints_statement():
    stmt = Stmt('diameter', Span(1, 1), {'edge': ('A', 'B'), 'center': 'O'})
    prog = Program([stmt])

    assert print_program(prog) == 'diameter A-B to circle center O\n'
