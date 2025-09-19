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
target angle at A rays A-B A-D
rules no_solving=true
```

Parsed, validated, and desugared:

```geoscript
scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
points A, B, C, D
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
target angle at A rays A-B A-D
rules no_solving=true
segment A-B
segment B-C
segment C-D
segment D-A
# desugared: A-D ∥ B-C
equal-segments (A-D ; B-C)
```

---

## Usage

### Install

```bash
pip install -e .
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

---

## Development

### Run demo

```bash
python -m geoscript_ir.demo
```

### Run tests

```bash
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
