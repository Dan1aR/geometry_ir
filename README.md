# GeoScript IR

**GeoScript IR** is a parser, validator, and desugarer for a small DSL describing **2D Euclidean geometry scenes**.
It turns human-readable problem statements into a canonical intermediate representation that can later feed a numeric solver and TikZ generator.

---

## Features

* **Parser**: strict line-based parser for the GeoScript language (per BNF).
* **Validator**: semantic checks (distinct vertices, valid options, angle rays, etc.).
* **Desugarer**: expands high-level primitives into base constraints:

  * `polygon` → closed cycle of `segment`s
  * `triangle` → 3 sides (+ `isosceles` or `right` if given)
  * `parallelogram` → 2 parallel pairs
  * `trapezoid` → 1 parallel pair (+ equal legs if `isosceles=true`)
  * `rectangle` → 4 right angles
  * `square` → rectangle + all sides equal
  * `rhombus` → all sides equal
* **Stable Printer**: turns AST back into canonical text, ideal for snapshot tests.

Identifiers are **case-insensitive** and stored uppercase; underscores are allowed.
Options support booleans (`true|false`) and simple key=value forms.

---

## Example

Input GeoScript:

```geoscript
scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
points A, B, C, D
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
target angle B-A-D
rules no_solving=true
```

Parsed, validated, and desugared:

```geoscript
scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
points A, B, C, D
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
target angle B-A-D
rules no_solving=true
segment A-B
segment B-C
segment C-D
segment D-A
parallel-edges (A-D ; B-C)
equal-segments (A-D ; B-C)
```

---

## Usage

### Install

```bash
pip install -e .
```

To include the test tooling, install with the optional `test` extras:

```bash
pip install -e ".[test]"
```

### Python API

```python
from geoscript_ir import parse_program, validate, desugar, print_program

text = '''
scene "Square"
points A, B, C, D
square A-B-C-D
'''

prog = parse_program(text)
validate(prog)           # raises ValidationError on bad input
dz = desugar(prog)       # expand square to sides + right angles + equal segments
print(print_program(dz)) # canonical form
```

### Numeric solver (GeometryIR → SciPy)

The `geoscript_ir.solver` module compiles validated GeoScript into a
numeric model and optimizes the residuals with `scipy.optimize.least_squares`.

```python
from geoscript_ir import parse_program, validate, desugar
from geoscript_ir.solver import translate, solve, SolveOptions

text = """
scene "Right triangle"
points A, B, C
segment A-B [length=4]
segment A-C [length=3]
segment B-C [length=5]
right-angle B-A-C
"""

program = parse_program(text)
validate(program)
desugared = desugar(program)
model = translate(desugared)
solution = solve(model, SolveOptions())

print(solution.success, solution.max_residual)
print(solution.point_coords)
```

See `examples/solve_right_triangle.py` for a complete runnable sample.

### Grammar & LLM prompt

The canonical GeoScript grammar lives alongside the library so tooling can
consume it directly:

```python
from geoscript_ir.reference import BNF, LLM_PROMPT, get_llm_prompt

print(BNF)                       # raw Backus–Naur form
print(get_llm_prompt())          # default LLM instructions + BNF
print(get_llm_prompt(include_bnf=False))  # instructions only
```

`LLM_PROMPT` is a ready-to-use set of guardrails for agents that need to emit
GeoScript scenes. It repeats the "do" / "don't" guidance and, by default,
appends the grammar so the model always has the exact syntax available.

---

## Development

### Run demo

```bash
python -m geoscript_ir.demo
```

### Run tests

```bash
pip install -e ".[test]"
pytest -q
```

---

## Roadmap

* Residual generator for numeric solvers (`scipy.optimize.least_squares`).
* Label placement & aesthetics as a separate pass.
* TikZ exporter (`tkz-euclide` / `tkz-elements` preferred).
* More shape primitives (kites, regular n-gons, etc.).

---

## License

MIT — free to use, modify, and integrate.

---

Do you also want me to add a **language reference section** (all keywords and options from your BNF) inside the README, so new contributors can use it without opening the spec?
