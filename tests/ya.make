PY3TEST()

PEERDIR(
    # geometry ir
    education/schoolbook/agents/geometry_agent/geometry_ir
)

TEST_SRCS(
    test_bnf.py
    test_consistency.py
    test_desugar.py
    test_printer.py
    test_reference.py
    test_smoke.py
    test_solver.py
    test_validation.py
    integrational/test_gir_scenes.py
)

END()