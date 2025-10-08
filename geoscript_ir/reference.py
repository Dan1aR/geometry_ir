"""Reference helpers for the GeoScript intermediate representation."""

from textwrap import dedent

BNF = dedent(
"""
```
Program   := { Stmt }
Stmt      := Scene | Layout | Points | Obj | Placement | Annot | Target | Rules | Comment

Scene     := 'scene' STRING
Layout    := 'layout' 'canonical=' ID 'scale=' NUMBER
Points    := 'points' ID { ',' ID }

Annot     := 'label point' ID Opts?
            | 'sidelabel' Pair STRING Opts?

Target    := 'target'
              ( 'angle' Angle3 Opts?
              | 'length' Pair Opts?
              | 'point' ID Opts?
              | 'circle' '(' STRING ')' Opts?
              | 'area' '(' STRING ')' Opts?
              | 'arc' ID '-' ID 'on' 'circle' 'center' ID Opts?
              )

Obj       := 'segment'      Pair Opts?
            | 'ray'          Pair Opts?
            | 'line'         Pair Opts?
            | 'circle' 'center' ID 'radius-through' ID Opts?
            | 'circle' 'through' '(' IdList ')' Opts?
            | 'circumcircle' 'of' IdChain Opts?
            | 'incircle'     'of' IdChain Opts?
            | 'perpendicular' 'at' ID 'to' Pair 'foot' ID Opts?
            | 'parallel'      'through' ID 'to' Pair Opts?
            | 'median'        'from' ID 'to' Pair 'midpoint' ID Opts?
            | 'angle'         Angle3 Opts?
            | 'right-angle'   Angle3 Opts?
            | 'equal-segments' '(' EdgeList ';' EdgeList ')' Opts?
            | 'parallel-edges' '(' Pair ';' Pair ')' Opts?
            | 'tangent' 'at' ID 'to' 'circle' 'center' ID Opts?
            | 'diameter'      Pair 'to' 'circle' 'center' ID
            | 'line' ID '-' ID 'tangent' 'to' 'circle' 'center' ID 'at' ID Opts?
            | 'polygon'       IdChain Opts?
            | 'triangle'      ID '-' ID '-' ID Opts?
            | 'quadrilateral' ID '-' ID '-' ID '-' ID Opts?
            | 'parallelogram' ID '-' ID '-' ID '-' ID Opts?
            | 'trapezoid'     ID '-' ID '-' ID '-' ID Opts?
            | 'rectangle'     ID '-' ID '-' ID '-' ID Opts?
            | 'square'        ID '-' ID '-' ID '-' ID Opts?
            | 'rhombus'       ID '-' ID '-' ID '-' ID Opts?
            | 'collinear' '(' IdList ')' Opts?
            | 'concyclic' '(' IdList ')' Opts?
            | 'equal-angles' '(' AngleList ';' AngleList ')' Opts?
            | 'ratio' '(' Pair ':' Pair '=' NUMBER ':' NUMBER ')' Opts?

Placement := 'point' ID 'on' Path Opts?
            | 'intersect' '(' Path ')' 'with' '(' Path ')' 'at' ID (',' ID)? Opts?
            | 'midpoint' ID 'of' Pair Opts?
            | 'foot' ID 'from' ID 'to' Pair Opts?

Path      := 'line'    Pair
            | 'ray'     Pair
            | 'segment' Pair
            | 'circle' 'center' ID
            | 'angle-bisector' Angle3 ('external')?
            | 'median'  'from' ID 'to' Pair
            | 'perpendicular' 'at' ID 'to' Pair
            | 'perp-bisector' 'of' Pair
            | 'parallel' 'through' ID 'to' Pair

Rules     := 'rules' Opts

Comment   := '#' { any-char }

EdgeList  := Pair { ',' Pair }
IdList    := ID { ',' ID }
IdChain   := ID '-' ID { '-' ID }
AngleList := Angle3 { ',' Angle3 }
Pair      := ID '-' ID
Angle3    := ID '-' ID '-' ID

Opts      := '[' KeyVal { (',' | ' ') KeyVal } ']'
KeyVal    := KEY '=' (NUMBER | STRING | BOOLEAN | ID | ID '-' ID | SQRT | PRODUCT)
            | 'choose' '=' ('near' | 'far' | 'left' | 'right' | 'cw' | 'ccw')
            | 'anchor' '=' ID
            | 'ref'     '=' Pair
SQRT      := 'sqrt' '(' NUMBER ')'
PRODUCT   := NUMBER '*' SQRT
BOOLEAN   := 'true' | 'false'
```
"""
).strip()


_PROMPT_CORE = dedent(
"""
**Role & scope**

You translate a geometry problem from natural language into **GeoScript IR**.
**Do not solve** anything; encode only the givens, constructions, and the ask using GeoScript syntax.

**Output policy**

* Output **only** a GeoScript program (no commentary).
* **Header order** (exactly this order):

  1. `scene "..."`
  2. `layout canonical=<id> scale=<number>`
  3. `points A, B, C, ...` (list every named point exactly once)
* Then: **one fact per line** (objects, placements, relations, annotations).
* Put **targets last**.
* **No unstated auxiliaries** unless the task explicitly allows it or you add `rules [allow_auxiliary=true]`.

---

## 🔧 Canonical mapping from prose → GeoScript objects

### 1) **Sides vs lines vs rays** (critical)

* “**сторона … / отрезок …**” → **`segment`**
  *Example:* “Стороны (BC) и (AD) пересекаются в точке (O)” →
  `intersect (segment B-C) with (segment A-D) at O`
* “**прямая … / прямые …**”, “**продолжение стороны …**” → **`line`**
* “**луч …**” → **`ray`**
* **Points on a side** → `point P on segment X-Y` (not `line`).

> Using `line` where a **side** is intended will change residuals and can misplace the picture. Prefer `segment` for polygon sides.

### 2) **Polygons & special quads**

* If the text says *triangle ABC*, *parallelogram ABCD*, *trapezoid ABCD*, *rectangle*, *square*, *rhombus*, **declare the high-level object**:

  ```
  triangle A-B-C
  parallelogram A-B-C-D
  trapezoid A-B-C-D [bases=A-D]        # 🔧 if bases named, set them here
  ```

  Then add extra facts (e.g., right angle, equal sides) if the prose states them.

### 3) **Diagonals & intersections**

* “Диагонали пересекаются в точке O” →

  ```
  segment A-C
  segment B-D
  intersect (segment A-C) with (segment B-D) at O
  ```

  🔧 Use `segment` for diagonals unless the text says *прямые*.

### 4) **Midlines (triangle/trapezoid)** 🔧

* **Triangle midline** between sides `AB` and `AC`:

  ```
  midpoint M of A-B
  midpoint N of A-C
  segment M-N
  ```
* **Trapezoid midline** connects **midpoints of the legs** (non-parallel sides). If bases are `[bases=A-D]`, the legs are `A-B` and `C-D`:

  ```
  midpoint M of A-B
  midpoint N of C-D
  segment M-N
  ```

### 5) **Distances to lines** (explicit foot)  — *MANDATORY*

When the text says “расстояние от точки `P` до прямой/стороны `XY` …”:

* If a number is given:

  ```
  foot H_P from P to X-Y
  segment P-H_P [length=<number|sqrt(...)]   # solver constraint
  ```
* If “find the distance”:

  ```
  foot H_P from P to X-Y
  target length P-H_P
  ```

🔧 Do **not** replace the foot with a generic “point on line” — perpendicularity is required.

### 6) **Bisectors, medians, altitudes** (construction)

* **Angle bisector at `B` hitting side `AC` at `D`**:

  ```
  intersect (angle-bisector A-B-C) with (segment A-C) at D [choose=...]
  ```
* **Median from `C` to side `AB`**:

  ```
  median from C to A-B midpoint M
  ```
* **Altitude** is encoded via **foot**:

  ```
  foot H from X to A-B
  ```

🔧 Never use an `altitude` keyword; it doesn’t exist in the DSL.

### 7) **“Ray between the sides of ∠COD”** 🔧

To place a ray **strictly inside** an angle:

```
point T on angle-bisector C-O-D         # interior direction
ray O-T
```

If a **side** is specified, additionally constrain `T`:

```
point T on angle-bisector C-O-D
point T on segment O-<the side’s vertex>   # optional to steer toward a side
ray O-T
```

If orientation matters (left/right), use soft selectors on a placement or intersection:
`[choose=left|right ref=C-D]` or `[choose=cw|ccw anchor=O]`.

### 8) **Tangency** (choose the right primitive)

* Tangent **at a known touchpoint** `T`:

  ```
  tangent at T to circle center O
  ```

  (Optionally add a carrier if the tangent line should be drawn: `line X-Y tangent ... at T`.)
* Tangent **from an external point** `A` with *unknown* touchpoint:

  ```
  line A-T tangent to circle center O at T [choose=...]
  ```

  (two roots; use `choose=near|far|left|right|cw|ccw` with `anchor`/`ref`)

### 9) **Circle–circle tangency** (no direct primitive; construct via line of centers)

* **External tangency** between centers `O1`, `O2`:

  ```
  circle center O1 radius-through K1
  circle center O2 radius-through K2
  intersect (line O1-O2) with (circle center O1) at T12 [choose=near anchor=O2]
  point T12 on circle center O2
  ```
* **Internal tangency** uses the **far** root along the same line.

### 10) **Concyclic / circumcircle / incircle**

* “A, B, C, D lie on one circle” → `concyclic (A, B, C, D)`
* Triangle circumcircle → `circumcircle of A-B-C` *or* `circle through (A, B, C)`
* Triangle incircle → `incircle of A-B-C`

### 11) **Ratios & equalities**

* Segment ratio: `ratio (A-B : C-D = p : q)` (with `p>0`, `q>0`).
* Equal segments/angles use the dedicated constructs (don’t rely on text labels):

  ```
  equal-segments (A-B ; C-D, E-F)
  equal-angles (A-B-C ; D-E-F)
  ```

---

## 🔧 Branch picking (two-root choices)

Any **line∩circle**, **circle∩circle**, tangent-from-external-point, or similar two-root step **must** include a soft selector:

* `choose=near|far anchor=<ID>`
* `choose=left|right ref=<A-B>`
* `choose=cw|ccw anchor=<ID> [ref=<A-B>]`

**Heuristics** (use when the prose doesn’t disambiguate in words):

* For **line∩circle** to pick the point “toward” some vertex `Q`: use `choose=near anchor=Q`.
* For “the intersection on the **left** of directed line `AB`”: `choose=left ref=A-B`.
* Around a circle (clockwise/counter-clockwise) relative to `anchor=Q`: `choose=cw|ccw anchor=Q`.

---

## Visual annotations (what to actually draw)

* If a numeric **length** is given for the solver, also **draw it** using a **side label**:

  ```
  segment A-B [length=5]          # solver constraint
  sidelabel A-B "5" [pos=below]   # 🔧 visible label in the figure
  ```

  *(The TikZ renderer ignores `[length=...]` for drawing; `sidelabel` makes it visible.)*
* If the problem is about “naming angles” or “marking equal angles”, declare **angle arcs** explicitly:

  ```
  angle H-O-K [label="$\\alpha$"]      # 🔧 wrap Greek in LaTeX to avoid encoding issues
  equal-angles (A-B-C ; D-E-F)
  ```
* Use `label point P [pos=left|right|above|below]` when the prose names or highlights a point.

---

## Targets (the ask)

* Use the most specific target that matches the text:

  ```
  target angle A-B-C [label="?"]
  target length A-B
  target point X
  target arc B-T on circle center O [label="?BT"]
  target area ("ABED")     # name string only; solver will ignore its text
  ```
* If the exercise is only “draw/name/mark” and **doesn’t** ask to find something, you may omit `target`, but **do** include the marks that make the drawing faithful (angle arcs, sidelabels, ticks).

---

## Boilerplate cheat-sheet

```text
# Header
scene "Short descriptive title"
layout canonical=<triangle_ABC|triangle_AB_horizontal|generic|generic_auto|triangle_ABO> scale=1
points A, B, C, ...

# Objects (one fact per line)
triangle A-B-C
trapezoid A-B-C-D [bases=A-D]
parallelogram A-B-C-D
right-angle A-C-B
angle A-B-C [degrees=30]
equal-angles (A-B-C ; D-E-F)
equal-segments (A-B ; C-D, E-F)
circle center O radius-through A
circumcircle of A-B-C
incircle of A-B-C
concyclic (A, B, D, E)
ratio (A-B : C-D = 2 : 3)

# Placements
point P on segment A-B
point Q on circle center O
point R on angle-bisector A-B-C
intersect (segment B-C) with (segment A-D) at O [choose=near anchor=B]
midpoint M of A-B
foot H from X to A-B
parallel through P to A-B
perp-bisector of A-B

# Visual labels (optional but recommended)
label point P [label="P" pos=above]
sidelabel A-B "5" [pos=below]
```

**BNF** (unchanged; keep your existing copy)

---

## Few-shot patterns (extra)

### A) Diagonals of a trapezoid cut its midline; find the larger sub-segment 🔧

```text
scene "Trapezoid: diagonal cuts the midline"
layout canonical=generic scale=1
points A, B, C, D, M, N, P
trapezoid A-B-C-D [bases=A-D]
segment A-D [length=10]
segment B-C [length=4]
midpoint M of A-B
midpoint N of C-D
segment M-N
segment A-C
intersect (segment A-C) with (segment M-N) at P [choose=near anchor=C]
# Optional labels for clarity
sidelabel A-D "10" [pos=below]
sidelabel B-C "4"  [pos=above]
target length P-N  # or M-P (choose one that the text asks for explicitly)
```

### B) Ray inside an angle 🔧

```text
scene "Ray a inside angle COD"
layout canonical=generic scale=1
points C, O, D, T
segment O-C
segment O-D
point T on angle-bisector C-O-D
ray O-T
target angle C-O-T
```

### C) Tangents from A to circle (both touchpoints) + disambiguation

```text
scene "Two tangents from A to (O)"
layout canonical=generic_auto scale=1
points A, O, B, C
circle center O radius-through B
line A-B tangent to circle center O at B [choose=left  ref=A-O]
line A-C tangent to circle center O at C [choose=right ref=A-O]
right-angle O-B-A
right-angle O-C-A
target length B-C
```

---

## **NON-negotiable rules (validator-safe)**

* **Always**: `scene` → `layout` → `points` (in that exact order at the top).
* **Always**: match **sides** with `segment`, **lines** with `line`, **rays** with `ray`.
* **Always**: add `choose=...` with `anchor`/`ref` on any two-root placement.
* **Always**: encode distances to lines via a **`foot`** placement.
* **Never**: use `altitude` as a keyword; express it as a `foot`.
* **Never**: mix `circle center O ...` and `circle through (...)` for the same circle.
* **Never**: invent auxiliary geometry unless the task allows it or `rules [allow_auxiliary=true]` is present.
* If bases are named for a trapezoid, **set** `[bases=...]` so renderers can orient the figure nicely.

---

"""
).strip()


def get_llm_prompt(*, include_bnf: bool = True) -> str:
    """Return the standard GeometryIR prompt for LLM agents."""
    sections = [_PROMPT_CORE]
    if include_bnf:
        sections.append("SYNTAX REFERENCE (BNF)\n" + BNF)
    return "\n\n".join(sections)


LLM_PROMPT = get_llm_prompt()

__all__ = ["BNF", "LLM_PROMPT", "get_llm_prompt"]
