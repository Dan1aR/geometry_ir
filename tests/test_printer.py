from geoscript_ir.ast import Program, Span, Stmt
from geoscript_ir.printer import print_program


def test_parallel_edges_prints_canonical_form():
    stmt = Stmt('parallel_edges', Span(1, 1), {'edges': [('A', 'B'), ('C', 'D')]})
    prog = Program([stmt])

    assert print_program(prog) == 'parallel-edges (A-B ; C-D)'
