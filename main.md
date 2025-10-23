# GeoScript IR — Project Documentation (Target State)

GeoScript IR is a compact DSL for 2D Euclidean geometry scenes. The toolchain
parses source text into an AST, validates geometric intent, desugars composite
objects, **compiles hard equalities to a CAD-grade solver (libslvs via
`python-solvespace`)**, and then **polishes** the result with a scale-aware
least-squares pass that enforces segment/ray semantics and aesthetics. A
deterministic derivation pass (DDC) cross-checks the final coordinates. This
document is the authoritative description of both the language and the
implementation.

> **Scope:** 2D planimetry.

---

## 1. Vision and Design Goals

1. **Readable problem statements.** Scripts mimic olympiad prose (e.g. "Trapezoid
   ABCD with base AD", "Circle with center O", "Find ∠DBE").
2. **Robust correctness + beauty.** Equality constraints are solved by a mature
   CAD solver; inequalities and aesthetic/readability goals are handled in a
   dedicated polishing step.
3. **Modular architecture.** Lexing/parsing, validation plus desugaring,
   CAD adapter, polish optimizer, DDC checks, and TikZ export are isolated
   modules with dedicated prompts.

---

## 2. Architecture Map

### 2.1 Data Flow Overview

```

source text ──▶ lexer ──▶ parser ──▶ AST ──▶ validator ──▶ desugarer
│
▼
CAD adapter (libslvs)
(solve hard equalities)
│
▼
polishing optimizer (least-squares)
(segment/ray clamps, guards, label layout)
│
┌─────────────────────┬────────────────┴───────────────┐
▼                     ▼                                ▼
DDC checker           renderers                       reporting
(deterministic           (printer /                        (logs,
derivations)              TikZ)                           metrics)

```

### 2.2 Module Index

| Component | Modules | Notes |
|-----------|---------|-------|
| Command-line tools | `geoscript_ir.__main__`, `geoscript_ir.demo` | Entry points for compiling or inspecting scenes. |
| Lexing & parsing | `geoscript_ir.lexer`, `geoscript_ir.parser`, `geoscript_ir.ast` | Convert source lines into AST statements with spans. |
| Numeric helpers | `geoscript_ir.numbers`, `geoscript_ir.orientation` | Symbolic numbers with numeric payloads, orientation utilities. |
| Validation | `geoscript_ir.validate`, `geoscript_ir.consistency` | Structural checks, option validation, adapter dry-run, and missing-support warnings with auto hot-fixes. |
| Desugaring | `geoscript_ir.desugar` | Expands polygons/traits into primitive facts; produces canonical carriers and helper entities. |
| **CAD adapter** | `geoscript_ir.cad.slvs_adapter` | Builds and solves a `python-solvespace` system for hard equalities; returns coordinates & diagnostics. |
| **Polish optimizer** | `geoscript_ir.polish` | Re-parameterizes along carriers and solves soft constraints (segment/ray, guards, labels). |
| **Solver façade** | `geoscript_ir.solver` | Orchestrates **CAD → Polish → DDC**; exposes options and diagnostics. |
| Deterministic derivation & cross-check | `geoscript_ir.ddc` | Recomputes derivable points from explicit rules and compares against the final numeric solution. |
| Rendering & exports | `geoscript_ir.printer`, `geoscript_ir.reference`, `geoscript_ir.reference_tikz`, `geoscript_ir.tikz_codegen` | Pretty-printing, example gallery, TikZ export helpers. |
| Tests & prompts | `tests/`, `.github/prompts/*.prompt.md` | Unit & integration suites and alignment prompts (`compile`, `lint`). |

---

## 3. Language Front-End

The parser consumes UTF-8 GeoScript text and produces a `Program` (see
§3.4). Tokens are case-insensitive identifiers, numbers, strings, and symbols as
specified below.

### 3.1 Lexical Conventions

* `ID` tokens are case-insensitive; they normalize to uppercase (`a-b` ≡ `A-B`).
* Strings use double quotes with C-style escapes.
* Numbers accept decimals and scientific notation. Symbolic values
  (`sqrt(...)`, `3*sqrt(2)`) become `SymbolicNumber` instances that carry both
  text and numeric value (`geoscript_ir.numbers`).
* `#` introduces a line comment.

### 3.2 Grammar Reference (BNF)

The BNF below defines the surface language accepted by the parser. Solver-facing
extensions include branch picking, `collinear`, `concyclic`, `equal-angles`,
`ratio`, and `perp-bisector` / `parallel through` path forms.

```

Program   := { Stmt }
Stmt      := Scene | Layout | Points | Obj | Placement | Annot | Target | Rules | Comment

Scene     := 'scene' STRING
Layout    := 'layout' 'canonical=' ID 'scale=' NUMBER
Points    := 'points' ID { ',' ID }

Annot     := 'label point' ID Opts?
| 'sidelabel' Pair STRING Opts?

Target    := 'target'
( 'angle'  Angle3 Opts?
| 'length' Pair   Opts?
| 'point'  ID     Opts?
| 'circle' '(' STRING ')' Opts?
| 'area'   '(' STRING ')' Opts?
| 'arc' ID '-' ID 'on' 'circle' 'center' ID Opts?
)

> **Note:** String payloads in target statements (e.g. `target area ("Find area of ABC")`)
> are treated purely as annotations. Their text is not scanned for point identifiers and
> introduces no solver variables or residuals.

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

`angle A-B-C` is only a visual mark until `degrees=` appears.

### 3.3 Supported Options & Keywords

The parser accepts arbitrary option keys, but the validator only recognizes the
entries listed here. Unknown options produce validation errors.

#### Global Rules (`rules [...]`)

* `no_equations_on_sides: bool`
* `no_solving: bool`
* `allow_auxiliary: bool`

#### Branch Selection (`point ... on ...`, `intersect (...) ... at ...`)

* `choose=near|far` with optional `anchor=P` → prefer the nearer/farther root
  relative to `P`.
* `choose=left|right` with `ref=A-B` → pick the left/right side of oriented line
  `AB`.
* `choose=cw|ccw` with optional `anchor=P` and optional `ref=A-B` → bias
  clockwise/counter-clockwise orientation around the anchor/reference.

Branch choices act as soft hints to resolve two-root ambiguities.

#### Angles & Arcs

* `angle A-B-C [degrees=NUMBER | label="..."]`
* `right-angle A-B-C [mark=square | label="..."]`
* `equal-angles (...) [label="..."]`
* `target angle A-B-C [label="..."]`
* `target arc P-Q on circle center O [label="?BT"]`

#### Segments & Polygons

* `segment A-B [length=NUMBER|SQRT|PRODUCT | label="..."]`
* `equal-segments (...) [label="..."]`
* `parallel-edges (...)`
* `polygon/triangle/... [isosceles=atA|atB|atC]`
* `triangle ... [right=atA|atB|atC]`
* `trapezoid [...] [bases=A-D]`
* `trapezoid [isosceles=true|false]`

#### Ratios

* `ratio (A-B : C-D = p : q)` with `p>0`, `q>0`.

#### Incidence Groups

* `collinear(A,B,C,...)` with ≥3 points.
* `concyclic(A,B,C,D,...)` with ≥4 distinct points.

#### Circles & Tangency

* `circle center O radius-through B`
* `circle through (A, B, C, ...)`
* `tangent at T to circle center O`
* `line X-Y tangent to circle center O at T`
* `diameter A-B to circle center O`

#### Annotations

* `label point P [label="..." pos=left|right|above|below]`
* `sidelabel A-B "..." [pos=left|right|above|below]` (renderers may add
  `mark=...`).

### 3.4 AST & Span Model (`geoscript_ir.ast`)

* `Span(line, col)` captures the primary source location for diagnostics.
* `Stmt(kind, span, data, opts, origin)` represents a normalized instruction.
  `origin` is `source` for author code or `desugar(<kind>)` / `hotfix(...)` for
  generated statements.
* `Program(stmts)` wraps the ordered list of statements and exposes
  `Program.source_program` to strip generated statements when needed.

### 3.5 Lexer & Parser Behavior

* `geoscript_ir.lexer.tokenize_line` tokenizes a single line and records
  line/column metadata for error reporting.
* `geoscript_ir.parser.Cursor` provides lookahead utilities, option parsing, and
  normalization of identifiers to uppercase.
* Error messages reuse `[line X, col Y]` spans. The parser preserves the original
  spelling in error messages while normalizing the AST to uppercase identifiers.
* Numeric option values are parsed into Python floats or `SymbolicNumber`
  instances; all other tokens remain strings until validation/desugaring.

---

## 4. Validation & Consistency Checks

### 4.1 Semantic Validation (`geoscript_ir.validate`)

* Ensures polygons, triangles, and quadrilaterals have the correct number of
  distinct vertices. Triangle options `isosceles=atX` and `right=atX` are
  validated explicitly.
* `trapezoid` checks confirm `bases=` picks an edge of the quadrilateral and
  `isosceles` is boolean.
* `polygon` requires ≥3 distinct vertices. `collinear` needs ≥3 distinct points;
  `concyclic` needs ≥4 distinct points.
* Angle and ratio statements require distinct vertices and positive numeric
  ratios. `equal-segments` demands non-empty lists on both sides; `equal-angles`
  requires matched triple lists.
* `rules [...]` only accepts the boolean flags listed in §3.3. `diameter`
  disallows options entirely.
* After structural checks, the validator deep-copies the program and invokes a
  **CAD adapter dry-run** to ensure constraints can be emitted (unknown
  constructs, missing carriers, etc.). Any emission error becomes a
  `ValidationError` annotated with source spans.

### 4.2 Consistency Hot-Fixes (`geoscript_ir.consistency`)

* Detects missing support segments/rays for `angle`, `right-angle`, and
  `equal-angles` statements. Suggested `segment` hot-fixes can be auto-inserted.
* Confirms that polygon-derived edges exist when required by later constraints.
* Warns about unsupported rays for branch-dependent constructs (parallel edges,
  tangents, etc.) and offers hot-fix statements tagged `origin='hotfix(consistency)'`.

---

## 5. Desugaring to Canonical Facts (`geoscript_ir.desugar`)

High-level constructs expand into primitive relations consumed by the **CAD
adapter**. Desugaring keeps helper statements deduplicated via canonical keys and
emits **carriers** (infinite lines / circle entities) where appropriate. **Range
semantics** (segment/ray intervals) are deferred to the polishing step.

* **`polygon A-B-C-...`** → segments along the perimeter (carriers).
* **`triangle A-B-C`** → carrier edges `AB`, `BC`, `CA`; optional
  `isosceles=atX` adds equality constraints; `right=atX` adds a right angle.
* **`quadrilateral A-B-C-D`** → segments around the quadrilateral.
* **`trapezoid A-B-C-D [bases=X-Y]`** → quadrilateral + `parallel-edges`
  between the chosen base pair and the opposite side; the named base anchors
  orientation/gauge.
* **`parallelogram A-B-C-D`** → `parallel-edges (A-B; C-D)` and
  `parallel-edges (B-C; A-D)`.
* **`rectangle`** → parallelogram + right angle; **`square`** → rectangle +
  equal segments; **`rhombus`** → equal segments + both parallel pairs.
* **`collinear (P1,...,Pn)`** → expands to carrier line plus point-on-line
  incidences.
* **`concyclic (P1,...,Pn)`** → introduces a latent center/radius and enforces
  point-on-circle incidences for all listed points.
* **`equal-angles (A-B-C, ... ; D-E-F, ...)`** → ties angles; prefer CAD equal-angle
  or numeric angle encoding; fallback handled in polish if required.
* **`ratio (A-B : C-D = p : q)`** → Length Ratio on the segments AB and CD.

`circle through (...)` and `circumcircle of ...` both introduce a shared latent
center/radius. The module may emit multiple variants where helper constructions
introduce branch choices; variants propagate to the CAD adapter.

---

## 6. Residual Library (Polishing Stage)

Let `v(P)` denote the 2D position of point `P`. The polishing stage keeps **CAD
equalities satisfied** by **re-parameterizing** points **on their carriers** and
optimizing only the remaining (soft) goals.

### 6.1 Re-parameterization along carriers

* **Point on line `A-B`** → scalar \(t\):  
  \( P = A + t\,\hat{u}_{AB}\), \(\hat{u}_{AB}=(B-A)/\|B-A\|\).
* **Point on circle center `O` radius `R`** → angle \(\theta\):  
  \( P = O + R(\cos\theta,\sin\theta)\).
* **Free points** (rare) may remain in \((x,y)\) with a weak anchor; prefer
  fixing gauge to avoid them.

### 6.2 Segment & Ray Membership (soft clamps)

Use a smooth hinge \( \operatorname{softplus}_k(x)=\frac{1}{k}\ln(1+e^{k x})\)
with \(k\in[15,40]\).

* **`point P on segment A-B`**: if \(L=\|AB\|\),
  \[
  r_{\text{seg}}(P;A,B) = \operatorname{softplus}(-t) + \operatorname{softplus}(t-L).
  \]
* **`point P on ray A-B`**:
  \[
  r_{\text{ray}}(P;A,B) = \operatorname{softplus}(-t).
  \]

### 6.3 Structural Guards (shape stabilizers; scale-aware)

Constants (defaults):  
\(\varepsilon_h=0.06\) (min altitude fraction), \(s_{\min}=0.10\) (min
\(|\sin\angle|\)), \(\varepsilon_A=0.02\) (area floor factor), and
\(w_{\text{shape}}=0.05\) (small weight).

* **Altitude floor** for side `(AB)` vs vertex `(C)`:
  \[
  h=\frac{|(B-A)\times(C-A)|}{\|B-A\|},\quad h_{\min}=\varepsilon_h\cdot\max(\|B-A\|,\|C-B\|),\quad
  r_h=\operatorname{softplus}(h_{\min}-h).
  \]
* **Adjacent-side angle cushion** at vertex \(B\) of \(\triangle ABC\):  
  \(s=|\sin\angle ABC|\), \(r_\angle=\operatorname{softplus}(s_{\min}-s)\).
* **Area floor** for polygon with area \(A\) and longest side \(L_{\max}\):  
  \( A_{\min}=\varepsilon_A L_{\max}^2\), \( r_A=\operatorname{softplus}(A_{\min}-A)\).

Guards are **aesthetic stabilizers**; they do **not** participate in DDC.

### 6.4 Equal-angles (fallback when not encoded in CAD)

Avoid `atan2` discontinuities. For angles \(\alpha,\beta\) minimize both
\(\Delta_c=\cos\alpha-\cos\beta\) and \(\Delta_s=\sin\alpha-\sin\beta\).

### 6.5 Residual aggregation

Polish residual set (normalized by a scene scale \(S\), e.g., max side length):

```

R_polish = w_seg * Σ r_seg/range + w_ray * Σ r_ray/range
+ w_shape * Σ (r_h + r_∠ + r_area)/S
+ w_readability * Σ r_labels
+ w_equalAngles * Σ (Δc^2 + Δs^2)

```

Weights are small (≪ 1 for hard facts). Equalities from CAD are preserved by
parameterization; any free-point equalities, if present, are added as stiff
terms or enforced by re-projection each iteration.

---

## 7. Solver Pipeline (`geoscript_ir.solver`)

### 7.1 Overview

We solve in two stages:

1) **CAD equality stage (libslvs via `python-solvespace`)**  
   Build a 2D system from the desugared program, apply **gauge** (fix 3 DoF of
   similarity), encode all *hard equalities*, bias discrete branches via initial
   placement and options (`choose=`), call `solve()`, read point coordinates.

2) **Polishing stage (in-house LSQ)**  
   Freeze carriers from the CAD result and **re-parameterize** points **along**
   those carriers; optimize only \(t,\theta\) (and any free \((x,y)\) if present)
   with soft constraints for **segment/ray bounds**, structural **guards**, and
   **readability** heuristics. Keep equalities satisfied by construction or
   stiff re-projection.

Finally, run **DDC** on the polished coordinates. The façade returns the final
solution, residuals, warnings, and a beauty score.

---

### 7.2 CAD Equality Stage (libslvs)

#### 7.2.1 Gauge & workplane

To prevent drift and improve conditioning:

- Create a 2D workplane: `wp = sys.create_2d_base()`.
- Choose a **gauge** automatically (from `layout canonical` or scene heuristics)
  or explicitly via options. Typical canonical gauge:

```

A = (0,0), B = (1,0); C_y > 0

````

Implement as fixed points (`dragged(A)`, `dragged(B)`); place `C` initially
with `y > 0`. This removes translation/rotation/scale.

#### 7.2.2 Entities

- **Points**: `P = sys.add_point_2d(x, y, wp)`; fix with `sys.dragged(P, wp)` when needed.
- **Lines (carriers)**: `lineAB = sys.add_line_2d(A, B, wp)` (infinite).
- **Circles**: create a circle centered at `O` with a radius seed; constrain points
onto it; radius may remain a variable if the scene demands it.

#### 7.2.3 DSL → `python-solvespace` mapping

Legend: ✅ native; 🟡 composition; 🔵 deferred to polish.

**Primitives & Incidence**

| DSL | CAD Encoding | Notes |
|---|---|---|
| `line A-B` | `add_line_2d(A,B)` | Carrier only. |
| `segment A-B` | `add_line_2d(A,B)` | 🔵 Bounds in polish. |
| `ray A-B` | `add_line_2d(A,B)` | 🔵 Bounds in polish. |
| `point P on line A-B` | `coincident(P, lineAB)` | ✅ |
| `point P on segment A-B` | `coincident(P, lineAB)` | 🔵 + clamp in polish. |
| `point P on ray A-B` | `coincident(P, lineAB)` | 🔵 + clamp in polish. |
| `point P on circle center O` | ensure circle(O); point-on-circle(P,circleO) | ✅ |
| `intersect(...) at X(,Y)` | build both carriers; `coincident(X, carrier1) & coincident(X, carrier2)`; seed near branch | 🟡 (`choose=` via seeding) |

**Metric & Angular**

| DSL | CAD Encoding | Notes |
|---|---|---|
| `segment A-B [length=L]` | `distance(A,B,L)` | ✅ |
| `equal-segments (...)` | equal length per pair | ✅ |
| `ratio (A-B : C-D = p : q)` | `ratio(AB, CD, p/q)` | ✅ (Length Ratio) |
| `right-angle A-B-C` | `perpendicular(BA, BC)` | ✅ |
| `angle A-B-C [degrees=θ]` | numeric angle constraint | ✅ (supplement via seeding) |
| `equal-angles (... ; ...)` | equal-angle if available; else encode numeric α and tie both; or defer | 🟡/🔵 |

**Parallel / Tangency / Midpoint**

| DSL | CAD Encoding | Notes |
|---|---|---|
| `parallel-edges (A-B ; C-D)` | `parallel(AB, CD)` | ✅ |
| `tangent at T to circle center O` | `point-on-circle(T,circleO)` + `tangent(lineXY, circleO)` + `coincident(T,lineXY)` | ✅ |
| `line X-Y tangent ... at T` | as above with shared endpoint `T` | ✅ |
| `midpoint M of A-B` | `coincident(M,lineAB)` + `distance(A,M)=distance(M,B)` | ✅ |

**Incidence Groups & Circles**

| DSL | CAD Encoding | Notes |
|---|---|---|
| `collinear(A,B,C,...)` | for each P: `coincident(P, lineAB)` | ✅ |
| `concyclic(P1,...,Pn)` | create circle (O,R) + `point-on-circle(Pi)` | ✅ |
| `circumcircle of A-B-C` | as concyclicity for A,B,C | ✅ |
| `incircle of A-B-C` | via bisectors/equal angles if available; else defer | 🟡/🔵 |

**Branch selection (`choose=`)**  
Realized by **initial placement** near the desired root and/or oriented
reference edges. Supplementary angle flips are handled by reseeding or by
rebuilding angle constraints with swapped vectors.

#### 7.2.4 Build, Solve, Read

```python
from python_solvespace import SolverSystem, ResultFlag

sys = SolverSystem()
wp  = sys.create_2d_base()

# Gauge (example): A=(0,0), B=(1,0)
A = sys.add_point_2d(0.0, 0.0, wp); sys.dragged(A, wp)
B = sys.add_point_2d(1.0, 0.0, wp); sys.dragged(B, wp)

# ... emit entities & constraints per mapping above ...

res = sys.solve()
if res != ResultFlag.OKAY:
  failures = sys.failures()
  dof      = sys.dof()
  # surface as adapter diagnostics

# Read point coordinates (robust across versions):
x, y = sys.params(P.params)
````

Diagnostics to surface from the adapter: `dof()` and `failures()`.

---

### 7.3 Polishing Stage (in-house LSQ)

#### 7.3.1 Parameterization

Freeze **carriers** from CAD (line directions, circle centers/radii) and
parameterize every “point on carrier” by a **single scalar** (`t` on a line) or
**angle** (`θ` on a circle). Any remaining free points may be left at ((x,y))
or weakly anchored.

#### 7.3.2 Soft constraints

Apply **segment/ray clamps**, **structural guards** (altitude, non-parallel
margin, area floor), and **readability** penalties (label/arc collisions, prefer
horizontal base if layout hints). Use **softplus** hinges and normalize by a
scene scale (S).

#### 7.3.3 Optimizer & projection discipline

Run LM/TRF on ({t,\theta}) (and any free ((x,y))). Because we optimize **on
carriers**, CAD equalities remain satisfied by construction. If a free point has
to be moved, add stiff equality terms or re-project to its carrier after each
iteration.

#### 7.3.4 Output

Return **polished coordinates**, a **beauty score**, and a residual breakdown
that lists the largest soft terms. These coordinates continue through DDC and
exports.

---

### 7.4 Deterministic Derivation & Cross-Check (DDC)

DDC runs **after polish**. It derives deterministic points from the final scene
and compares them to the polished coordinates. Reports include a derivation DAG
and normalized severities (`ok`, `warning`, `error`); `.passed` integrates with
CI.

---

### 7.5 Failure & Recovery

* **CAD stage fails** (inconsistent or under-constrained):
  surface `failures()` + `dof()`; retry with a tiny set of alternative seeds
  (flip supplementary, near/far as per `choose=`). If still failing, return a
  structured error with suggested hot-fixes (add gauge; relax a conflicting
  constraint; specify branch).

* **Polish fails** (clamps/guards dominate):
  report the dominating residuals and return CAD coordinates with warnings; do
  not invent geometry by over-weighting soft terms.

---

### 7.6 Options & Tunables

* **Gauge**: automatic from `layout canonical` or explicit (`--gauge A,B,C`).
* **Branch selection**: `choose=near|far|left|right|cw|ccw` biases initial
  placement.
* **Polish weights**: `w_shape`, label avoidance toggles, softplus steepness `k`.
* **Robust loss** (optional): Soft-L1/Huber in polish for extreme outliers.

---

### 7.7 Programmatic API (façade)

```python
from geoscript_ir.solver import solve_scene

res = solve_scene(program, options={
    "cad_solver": "slvs",
    "gauge": ("A","B","C"),    # fix A, B; keep C_y>0
    "polish": {
        "enable": True,
        "w_shape": 0.05,
        "label_avoid": True
    }
})

# res includes:
#   res.coords        -> dict[str, (x,y)]
#   res.beauty_score  -> float
#   res.cad_status    -> {"ok": bool, "dof": int, "failures": list[str]}
#   res.polish_report -> residual breakdown
#   res.ddc_report    -> pass/warn/error + notes
```

---

## 8. Deterministic Derivation & Cross-Check (DDC) (`geoscript_ir.ddc`)

* Implements §16 of the specification. Given the **polished** program solution,
  the DDC derives candidate coordinates for deterministically computable points
  and compares the result against the numeric solution.
* Rules are pure functions consuming known points/paths and returning candidate
  sets. Filters enforce ray/segment membership, perpendicular requirements, and
  soft selectors derived from options.
* `derive_and_check(program, solution)` emits a `DerivationReport` summarizing
  matches, mismatches, ambiguities, and unused facts. Reports include a derivation
  DAG for visualization.
* `DDCCheckResult` normalizes severity levels (`ok`, `warning`, `error`) and
  exposes `.passed` to integrate with CI.
* Circle caches, line specs, and orientation-aware matching ensure DDC tolerates
  floating-point noise when comparing with solver output.

---

## 9. Rendering & Export Pipelines

### 9.1 Textual Printer (`geoscript_ir.printer`)

* Pretty-prints programs, including desugared statements, with aligned columns.
* Respects option dictionaries and emits annotations for generated statements.

### 9.2 TikZ Export (`geoscript_ir.tikz_codegen`, `geoscript_ir.reference_tikz`)

* Generates TikZ code using **polished** solver output and metadata. Handles point
  labels, side labels, and optional styling marks from options.
* `reference_tikz` pairs with `docs/examples` to provide curated scene examples.

### 9.3 Reference Scenes (`geoscript_ir.reference`)

* Loads bundled GeoScript scenes for demos/tests. Offers quick inspection via
  `python -m geoscript_ir.demo`.

### 9.4 Orientation & Coordinate Normalization

* `apply_orientation(program, point_coords)` reorients coordinates for display.
  It prefers `trapezoid` declarations (especially those with `bases=`), falling
  back to isosceles triangles inferred from options or equal-segment groups;
  otherwise it returns the original coordinates with an identity transform.
* Trapezoid candidates are rotated so the averaged base direction becomes
  horizontal and reflected, if necessary, to keep the shorter base above the
  longer one. The routine records the applied matrix, translation, pivot, and the
  figure that triggered the transform inside an `OrientationResult`.
* `normalize_point_coords` (and `Solution.normalized_point_coords`) are exposed at
  the package level for deterministic min/max scaling when rendering or printing
  coordinates outside the CLI.

> Note: the **CAD stage** already fixes the similarity gauge for numerical
> stability; this section exists for presentation and export consistency.

---

## 10. Command-Line Interfaces & Tooling

* `python -m geoscript_ir` tokenizes, parses, validates, and solves scenes.
  Runtime flags include:

  * `--cad slvs` (select libslvs via `python-solvespace`);
  * `--no-polish` (return CAD coordinates without polish);
  * `--gauge A,B,C` (fix two points and orient a third above the x-axis);
  * logging level, seed, and optional TikZ export path.
* Each run logs CAD diagnostics (DoF, failures), polish statistics, beauty score,
  and prints both raw and normalized point coordinates before optionally writing a
  standalone TikZ document.
* `geoscript_ir.demo` launches an interactive prompt that lets users inspect
  bundled scenes.
* `.github/prompts/compile.prompt.md` enumerates CI steps: `pip install -e ".[test]"`,
  run `pytest`, and execute the integration suite (`tests/integrational/...`).

---

## 11. Testing Strategy

### 11.1 Unit — CAD adapter

* Points/lines/circles creation on a blank workplane.
* `coincident`, `parallel`, `distance`, `ratio`, `perpendicular`, `angle`
  constraints each in isolation (DoF=0, `solve()` OK).
* Coordinate reading via `sys.params(P.params)`.

### 11.2 Unit — Polish

* Segment clamp moves point inside `[0, L]` without breaking line incidence.
* Ray clamp respects `t ≥ 0`.
* Guards activate only near degeneracy; **scale-invariant** behavior (scale scene
  ×k → same decisions).
* Equal-angles fallback: paired rays yield small sin/cos residuals.

### 11.3 Integration

* **Parallel chords problem** → `BC≈8`, `DE≈12`, DDC pass (independent of the
  angle between the sides).
* Vary second side angle from 20° to 160° → stable result.
* Scenes with tangency, equal-segments, ratios, and collinearity groups.
* Under-constrained scene: CAD DoF>0 → meaningful diagnostics; polish doesn’t
  “fake” constraints.
* Supplementary angle branch test: reseed flips to the intended branch.

### 11.4 Regression

* Save failing cases; ensure deterministic reproduction (fixed RNG seeds).
* Compare `beauty_score` across versions (non-decreasing on reference gallery).

---

## 12. Change Log Notes

* **Migrated solver pipeline**: equality constraints solved in **CAD** (libslvs
  via `python-solvespace`), inequalities and aesthetics handled in **polishing**.
* **Desugaring emits carriers**; segment/ray **range semantics moved** to polish.
* **Structural guard** constants and hinges (formerly Appendix §6.5/S.*) are now
  detailed in §6.
* Validation rules continue to document triangle `right=atA|atB|atC` and
  `concyclic` ≥4-point requirements.
* New CLI flags: `--cad slvs`, `--no-polish`, `--gauge A,B,C`.

---

## Appendix A — Worked Example (Parallel chords in an angle)

**Task.** Стороны угла (A) пересечены параллельными прямыми (BC) и (DE),
(B,D) — на одной стороне, (C,E) — на другой. Известно (AB:BD=2:1),
(DE=12). Найти (BC).

**Sketch DSL.**

```text
points A,B,C,D,E
line A-X
line A-Y
parallel-edges (B-C; D-E)
ratio (A-B : B-D = 2 : 1)
segment D-E [length=12]
# placement: B,D on AX; C,E on AY; choose branches to split sides
```

**CAD stage (essential calls).**

```python
from python_solvespace import SolverSystem, ResultFlag
from math import sqrt

sys = SolverSystem(); wp = sys.create_2d_base()

# Gauge: A,B fixed; second side at 60°
A = sys.add_point_2d(0.0, 0.0, wp); sys.dragged(A, wp)
Bfix = sys.add_point_2d(1.0, 0.0, wp); sys.dragged(Bfix, wp)
Y = sys.add_point_2d(0.5, sqrt(3)/2, wp); sys.dragged(Y, wp)

# Carriers of the angle
lineAX = sys.add_line_2d(A, Bfix, wp)
lineAY = sys.add_line_2d(A, Y, wp)

# Unknowns on carriers (seeded on correct sides)
B = sys.add_point_2d(0.6, 0.0, wp);  sys.coincident(B, lineAX, wp)
D = sys.add_point_2d(1.0, 0.0, wp);  sys.coincident(D, lineAX, wp)
C = sys.add_point_2d(0.3, 0.52, wp); sys.coincident(C, lineAY, wp)
E = sys.add_point_2d(0.5, 0.87, wp); sys.coincident(E, lineAY, wp)

# Parallels and measures
BC = sys.add_line_2d(B, C, wp)
DE = sys.add_line_2d(D, E, wp)
sys.parallel(BC, DE, wp)

AB = sys.add_line_2d(A, B, wp)
BD = sys.add_line_2d(B, D, wp)
sys.ratio(AB, BD, 2.0, wp)     # AB : BD = 2 : 1
sys.distance(D, E, 12.0, wp)   # DE = 12

assert sys.solve() == ResultFlag.OKAY
Bx, By = sys.params(B.params); Cx, Cy = sys.params(C.params)
Dx, Dy = sys.params(D.params); Ex, Ey = sys.params(E.params)

# -> polishing: enforce segment/ray semantics if needed; here BC ≈ 8
```

**Result.** (BC = 8), invariant w.r.t. the angle between the sides (similarity).

```
