from geoscript_ir.ast import Program, Span, Stmt


def test_program_source_stmts_filters_generated_statements():
    source_stmt = Stmt('segment', Span(1, 1), {'edge': ('A', 'B')})
    generated_stmt = Stmt(
        'segment',
        Span(2, 1),
        {'edge': ('B', 'C')},
        origin='desugar(triangle)',
    )

    program = Program([source_stmt, generated_stmt])

    assert program.source_stmts == [source_stmt]
