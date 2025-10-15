PY3_LIBRARY()

PEERDIR(
    # numpy scipy
    contrib/python/numpy
    contrib/python/scipy
)

PY_SRCS(
    TOP_LEVEL
    geoscript_ir/__init__.py
    geoscript_ir/__main__.py
    geoscript_ir/ast.py
    geoscript_ir/consistency.py
    geoscript_ir/demo.py
    geoscript_ir/desugar.py
    geoscript_ir/lexer.py
    geoscript_ir/parser.py
    geoscript_ir/printer.py
    geoscript_ir/reference.py
    geoscript_ir/reference_tikz.py
    geoscript_ir/solver.py
    geoscript_ir/validate.py
    geoscript_ir/numbers.py
    geoscript_ir/orientation.py
    geoscript_ir/ddc.py
    geoscript_ir/seeders.py
    geoscript_ir/tikz_codegen/__init__.py
    geoscript_ir/tikz_codegen/generator.py
    geoscript_ir/tikz_codegen/utils.py
    examples/solve_trapezoid.py
)

END()

RECURSE_FOR_TESTS(
    tests
)
