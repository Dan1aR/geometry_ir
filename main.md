# GeoScript IR — Technical Specification (for Codex code agent)

**GeoScript IR** is a tiny, human-readable DSL for 2D Euclidean geometry scenes.
It parses into an AST, validates author intent, optionally desugars to canonical facts, and compiles into a numeric model solved via `scipy.optimize.least_squares`.

---

## 1) Design goals

1. **Intuitive planimetry** — scripts should read like contest/olympiad problems (“Trapezoid ABCD with base AD…”, “Circle with center O…”, “Find ∠DBE”, etc.).
2. **Complete, well-posed constraint graph** — every statement contributes explicit residuals so a solver can position **nice** coordinates and keep figures non-degenerate. The translator adds robust guards: min separations, edge-length floors, tiny angular margins for near-parallels, and orientation gauges.
3. **Separation of concerns** — parsing/printing, validation & desugaring, solver translation, and TikZ rendering are orthogonal modules. Reference prompts exist for LLM agents and for TikZ generation.

---

## 2) Lexical & identifiers

* **Case**: `ID` tokens are case-insensitive and normalized to **uppercase** (e.g., `a-b` and `A-B` refer to the same segment).
* **Strings**: double-quoted with C-style escapes.
* **Numbers**: decimals and scientific notation; **symbolic square roots** via `sqrt(<non-negative number>)` and products like `3*sqrt(2)` are supported as **SymbolicNumber** (text + numeric value).
* **Comments**: `#` to end of line.

---

## 3) Grammar (BNF)

The Codex agent must emit scripts that conform to this grammar.
This version introduces five solver-oriented extensions: **branch picking**, **collinear**, **concyclic**, **equal-angles**, **segment ratios**, and two new **Path** forms (perp-bisector, parallel-through).

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

> **Angle marking vs constraint** — `angle A-B-C` **without** `degrees=` is a *visual mark only* (no residuals). With `degrees=...` it becomes a metric constraint.

---

## 4) Options (legal keys by statement)

Only the keys below are interpreted. The parser rejects malformed option syntax; the validator rejects unknown/ill-typed options.

### Global

* `rules [...]` → `no_unicode_degree`, `mark_right_angles_as_square`, `no_equations_on_sides`, `no_solving`, `allow_auxiliary` (booleans).

### Branch selection (for **Placement**: `point ... on ...`, `intersect(...) ... at ...`)

* `choose=near|far` with `anchor=P` → prefer the solution nearer/farther from `P`.
* `choose=left|right` with `ref=A-B` → prefer the point on the left/right of oriented line `AB`.
* `choose=cw|ccw` with `anchor=P` (and optionally `ref=A-B`) → prefer clockwise/counter-clockwise branch around an anchor/reference.

> These are **soft biases** implemented as small hinge residuals; they resolve two-root ambiguities without brittle hard constraints.

### Angles & arcs

* `angle A-B-C [degrees=NUMBER | label="..."]`
* `right-angle A-B-C [mark=square | label="..."]`
* `equal-angles (...) [label="..."]`
* `target angle A-B-C [label="..."]`
* `target arc P-Q on circle center O [label="..."]`

### Segments/Edges/Polygons

* `segment A-B [length=NUMBER|SQRT|PRODUCT | label="..."]`
* `equal-segments (...) [label="..."]`
* `parallel-edges (...)` (typically no extra keys)
* `polygon/triangle/... [isosceles=atA|atB|atC]` (triangle),
  `trapezoid [...] [bases=A-D]`, `trapezoid [isosceles=true|false]`

### Ratios

* `ratio (A-B : C-D = p : q)` with positive `p,q` (numbers).

### Incidence groups

* `collinear(A,B,C,...)` (≥3 points)
* `concyclic(A,B,C,D,...)` (≥3 points)

### Circles & tangency

* `circle center O radius-through B`
* `circle through (A, B, C, ...)`
* `tangent at T to circle center O`
* `line X-Y tangent to circle center O at T`
* `diameter A-B to circle center O` (no options)

### Annotations

* `label point P [label="..." pos=left|right|above|below]`
* `sidelabel A-B "..." [pos=left|right|above|below]`
  (Renderers may also interpret `mark=midpoint` etc.)

---

## 5) High-level objects → canonical facts (desugaring rules)

To keep authoring natural, several forms **desugar** to primitive relations that the solver understands.

* **triangle A-B-C** → carrier edges `AB, BC, CA`.
* **quadrilateral A-B-C-D** → `AB, BC, CD, DA`.
* **trapezoid A-B-C-D [bases=X-Y]** → quadrilateral + `parallel-edges (X-Y; opposite)` + a tiny **non-parallel margin** on legs; the declared base is preferred for orientation gauging.
* **parallelogram A-B-C-D** → `parallel-edges (A-B; C-D)` + `parallel-edges (B-C; A-D)`; optional equalities if author marks them.
* **rectangle** → parallelogram + `right-angle A-B-C`.
* **square** → rectangle + `equal-segments (A-B; B-C; C-D; D-A)`.
* **rhombus** → `equal-segments` on all sides + both pairs of parallels.
* **collinear (P1,...,Pn)** → expand to collinearity constraints among all (`n≥3`).
* **concyclic (P1,...,Pn)** → introduce latent center `O` and radius `R`; enforce equal radii to `O`.
* **equal-angles (A-B-C, ... ; D-E-F, ...)** → unify all listed angles to a representative angle using `atan2`-based residuals.
* **ratio (A-B : C-D = p : q)** → residual `q‖AB‖ − p‖CD‖ = 0`.

> `circle through (...)` and `circumcircle of ...` equivalently become “points share a circle” with a latent center/radius at translation time.

---

## 6) Semantic constraints (residuals)

Let `v(P)` be the 2D variable of point `P`; `AB := v(B)−v(A)`, `×` = 2D cross, `·` = dot, `‖·‖` = Euclidean norm.

### 6.1 Placement / incidence

* `point P on line A-B` → `cross(AB, AP) = 0`.
* `point P on ray A-B` → collinearity + forwardness hinge: `cross(AB, AP)=0`, `max(0, −AB·AP)=0`.
* `point P on segment A-B` → collinearity + clamping hinges: `cross(AB, AP)=0`, `max(0, −AB·AP)=0`, `max(0, AB·(AP−AB))=0`.
* `point P on circle center O` → `‖OP‖ − ‖OB0‖ = 0` (where `B0` is the circle’s `radius-through` witness).
* `intersect (path1) with (path2) at X(, Y)` → both `X` (and optionally `Y`) satisfy the incidence constraints of `path1` and `path2`.
* **Branch picking**:

  * `choose=near|far, anchor=Q` → add small bias term `w*(‖XQ‖ − target)` with `target` = min/max among discrete roots.
  * `choose=left|right, ref=A-B` → hinge on orientation sign: `max(0, s*(−sign) )` with `s=+1/-1`.
  * `choose=cw|ccw` → small angular preference around anchor/ref.

### 6.2 Metric relations

* `equal-segments (E... ; F...)` → pick representative edge `R`; for all edges `E` and `F`: `‖E‖ − ‖R‖ = 0`.
* `segment A-B [length=L]` → `‖AB‖ − L = 0` (supports symbolic `L`).
* `midpoint M of A-B` → `AM = MB` and `M` collinear with `AB`.
* `ratio (A-B : C-D = p : q)` → `q‖AB‖ − p‖CD‖ = 0`.

### 6.3 Angular / orthogonality / parallelism

* `right-angle A-B-C` → `(BA)·(BC) = 0`.
* `angle A-B-C [degrees=θ]` → `atan2( cross(BA, BC), BA·BC ) − θ = 0` (degrees→radians internally).
* `equal-angles (...)` → for each listed angle, equalize its `atan2` with the representative angle.
* `angle-bisector U-V-W` (as **Path**) → direction equidistant in angle; “external” flips the bisector.
* `parallel-edges (A-B; C-D)` → `cross(AB, CD) = 0` (with a small turn-sign guard to avoid 180° flips).
* `perpendicular at T to A-B foot H` → `(AB)·(TH) = 0` and `H` on `A-B`.
* `perp-bisector of A-B` (as **Path**) → passes through midpoint of `AB` and is perpendicular to `AB`.
* `parallel through P to A-B` (as **Path**) → line through `P` parallel to `AB`.

### 6.4 Circle-specific

* `circle center O radius-through B` → witness `B` defines radius; `point ... on circle center O` uses it.
* `circle through (...)` / `circumcircle of ...` → hidden `(Oc, Rc)` with equal-radius constraints to all listed points.
* `incircle of A-B-C` → incenter as intersection of bisectors; equal distances to sides via perpendicular feet.
* `tangent at T to circle center O` → `OT ⟂` tangent direction and `T` on that circle.
* `line X-Y tangent ... at T` → `X,Y,T` collinear and `OT ⟂ XY`.
* `diameter A-B to circle center O` → `O,A,B` collinear and `‖OA‖ = ‖OB‖`.

### 6.5 Polygons & structural guards

* Declared polygon cycles contribute **carrier edges** and receive **edge floors** and **area floors** to prevent collapse (e.g., trapezoid → segment). Non-polygon carriers get lighter floors.
* A **non-parallel margin** is added to trapezoid legs.
* Orientation gauges prefer a declared base or a canonical edge; a unit-span gauge is used if needed.

---

## 7) Gauges, layout, and scale

* **Layout**: `layout canonical=<id> scale=<number>` seeds the initial placement and fixes global similarity degrees of freedom. Canonical examples: `triangle_ABC`, `triangle_AB_horizontal`, `triangle_ABO`, `generic` / `generic_auto`.
* **Scale**: `scale` flows into the model; if no numeric scale is meaningful, a **unit-span gauge** is applied on an orientation edge.
* **Min-separation**: global pairwise min distances (stronger for declared collinear sets), polygon edge floors, and lighter carrier floors are enforced via hinge residuals.

---

## 8) Validation rules (reject early)

* **Arity & distinctness**

  * `triangle`: exactly three distinct points; quads/special quads: four distinct points; `polygon`: ≥3 distinct points.
  * `collinear`: ≥3 points; `concyclic`: ≥3 points.
  * `equal-angles`: both lists non-empty; all angle vertices distinct from their endpoints.
  * `circle through (...)`: ≥3 distinct points.
* **Options**

  * `diameter ...` accepts **no** options.
  * `trapezoid [bases=...]` must name one of its sides (either orientation); `isosceles` is boolean or absent.
  * `ratio (A-B : C-D = p : q)` requires `p>0`, `q>0`.
  * Branch picking:

    * `choose=near|far` requires `anchor=<ID>`.
    * `choose=left|right` requires `ref=<Pair>`.
    * `choose=cw|ccw` requires `anchor=<ID>` (and may use `ref` to disambiguate).
* **Rules**

  * `rules [...]` only admits known boolean flags listed above.

The validator runs a dry translation to ensure the residual builder can accept the program and reports precise source spans (`[line X, col Y]`).

---

## 9) AST & public API (summary)

* **AST**: `Program{ stmts: [Stmt] }`, `Stmt{ kind, span{line,col}, data{}, opts{}, origin: 'source'|'desugar()' }`.
  `Program.source_program` filters to source-only statements.
* **Core API**: `parse_program`, `validate`, `desugar`, `translate`, `solve*`, TikZ helpers, prompts `BNF`, `LLM_PROMPT`.

---

## 10) Authoring guidelines for the Codex agent

1. **Header first**: `scene`, then `layout`, then `points` (list every named point once).
2. **One fact per line**: translate givens into explicit statements (`segment`, `circle ...`, `parallel`, `right-angle`, `tangent`, `equal-segments`, `collinear`, `concyclic`, `equal-angles`, `ratio`, etc.).

   * Circles: choose **either** `center O radius-through B` **or** `through (A,B,C,...)` (never both).
   * Tangents in text → **explicit** tangency statements.
   * Avoid naked equations in labels; prefer structural statements or `sidelabel`.
3. **Construct, don’t infer**: use `point ... on ...` and `intersect (...) with (...)` for placements; do not invent helper geometry unless the prompt or `rules [allow_auxiliary=true]` permits.
4. **Resolve branches**: whenever an operation has two solutions (line–circle, circle–circle, angle bisector hits a segment, etc.), use `choose=...` with `anchor`/`ref` to select the intended branch.
5. **Targets last**: end with `target ...` lines matching the ask (“find ∠…”, length, point, circle, area, arc).
6. **Option hygiene**: only emit allowed keys/values; use `sqrt(...)` notation for radicals (e.g., `[length=3*sqrt(2)]`).

---

## 11) Examples

### A. Trapezoid with midline (unchanged shape, clearer semantics)

```
scene "Trapezoid with midline"
layout canonical=generic scale=1
points A, B, C, D, M
trapezoid A-B-C-D [bases=A-D]
segment B-D
point M on segment B-D [mark=midpoint]
parallel-edges (A-D; B-C)
sidelabel A-D "10" [pos=below]
target length A-M [label="?"]
```

### B. Right triangle: bisector & median; find the angle

```
scene "Right-angled triangle; ∠B=21°, find ∠(CD,CM)"
layout canonical=triangle_ABC scale=1
points A, B, C, D, M
triangle A-B-C
right-angle A-C-B [mark=square]
angle A-B-C [degrees=21]
intersect (angle-bisector A-C-B) with (segment A-B) at D [choose=left ref=A-B]
median from C to A-B midpoint M
target angle D-C-M [label="?"]
```

### C. Circle via diameter; highlight an arc

```
scene "Diameter AB; highlight arc BT"
layout canonical=generic scale=1
points A, B, C, D, O, T
segment A-B
circle center O radius-through A
diameter A-B to circle center O
point C on circle center O
point D on circle center O
point T on circle center O
target arc B-T on circle center O [label="?BT"]
```

### D. Equal angles and concyclicity stated directly

```
scene "Concyclic points with equal angles"
layout canonical=generic scale=1
points A, B, C, D, E
triangle A-B-C
concyclic (A, B, D, E)
equal-angles (A-B-C ; A-D-E) [label="∠ABC = ∠ADE"]
target point E
```

### E. Perpendicular bisector as a path; branch-picked intersection

```
scene "Circumcenter by perp bisectors"
layout canonical=triangle_ABC scale=1
points A, B, C, O
triangle A-B-C
intersect (perp-bisector of A-B) with (perp-bisector of B-C) at O [choose=near anchor=A]
target point O
```

### F. Segment ratio

```
scene "Given AB:CD = 2:3"
layout canonical=generic scale=1
points A, B, C, D
segment A-B
segment C-D
ratio (A-B : C-D = 2 : 3)
target length A-B
```

### G. Collinear chain expressed directly

```
scene "A, B, M are collinear; M is midpoint"
layout canonical=generic scale=1
points A, B, M
segment A-B
collinear (A, M, B)
midpoint M of A-B
target point M
```

---

## 12) Error model & messages

* **Lexical**: unexpected characters, unterminated strings, malformed `sqrt` → `[line x, col y]`.
* **Syntactic**: unexpected token/keyword with precise spans (missing separators inside `[ ... ]` also flagged).
* **Validation**: arity/distinctness/unknown options; illegal combinations (e.g., options on `diameter`), missing required branch-picking anchors/refs, non-positive ratio parts.

---

## 13) Solver contract

* **Input**: a validated `Program`.
* **Output**:
  `Solution { point_coords, success, max_residual, residual_breakdown, warnings }`, plus helpers to normalize coordinates for rendering.
* **Algorithm**: build
  `Model{ points, index, residuals, gauges, scale }`, then minimize all residual groups using LSQ (`method="trf"`); reseed if necessary; score candidates by success then residual magnitude.
* **Safety**: min-separation guards, edge floors, area floors, non-parallel margins for trapezoids, orientation gauges (prefer declared bases; otherwise unit-span).

---

## 14) Compatibility & versioning

* **Grammar stability**: future extensions should add new `Obj`/`Placement` variants or option keys; avoid changing existing productions.
* **Reserved words**: all top-level keywords shown in the grammar; identifiers beginning with `\` are accepted by the lexer but the backslash is stripped.

---

## 15) Appendix — Canonical layouts (non-normative hints)

Renderers and agents may use the following canonical seeds:

* `triangle_ABC`: place `A=(0,0)`, `B=(4,0)`, `C` above the base
* `triangle_AB_horizontal`: `A=(0,0)`, `B=(4,0)`, “third” vertex above
* `triangle_ABO`: `A=(0,0)`, `B=(4,0)`, `O` above AB
* `generic` / `generic_auto`: balanced unit-scale layout with non-degenerate orientation


---

## 16) Deterministic Derivation & Cross‑Check (DDC)

**Goal.** Even when LSQ “converges,” the chosen branch (mirror, wrong intersection, etc.) can be geometrically incorrect.
The **DDC** module deterministically **derives** coordinates for all points that are *uniquely computable* from other points and objects, **without** optimizing, then **compares** them to the solver’s output. It catches wrong branches, flipped orders, and inconsistent placements early.

### 16.1 Public API

```python
def derive_and_check(program: Program, solution: Solution, *, tol=None) -> DerivationReport: ...
```

* **Input**

  * `program`: validated & desugared GeoScript IR (`Program`).
  * `solution`: numeric result from the solver (`Solution.point_coords: Dict[str, (x,y)]`).
  * `tol` (optional): absolute distance tolerance for matches. If `None`, use `tol = 1e-6 * scene_scale`, where `scene_scale` is the diagonal of the solution’s bounding box (or the declared `layout.scale` if provided).

* **Output**: `DerivationReport`

  ```python
  class DerivationReport(TypedDict):
      status: Literal["ok","mismatch","ambiguous","partial"]
      summary: str
      points: Dict[str, DerivedPointReport]
      unused_facts: List[str]         # ids/labels of facts not consumed by any derivation
      graph: DerivationGraphExport    # nodes, edges, topo order (for visualization/debug)
  ```

  ```python
  class DerivedPointReport(TypedDict):
      rule: str                        # e.g. "foot(H; X→AB)", "intersect(line, circle)"
      inputs: List[str]                # dependency ids (points/paths)
      candidates: List[Tuple[float,float]]
      chosen_by: Literal["unique","opts","ray/segment filter","closest-to-solver","undetermined"]
      match: Literal["yes","no"]
      dist: float                      # min distance from solver coordinate to any candidate
      notes: List[str]                 # degeneracy warnings, filters applied
  ```

  Overall `status` rules:

  * `ok` if every derivable point matches within `tol`.
  * `partial` if some points are not derivable (no rule or missing inputs) but all derived ones match.
  * `ambiguous` if a point has >1 valid candidates and the solver picked one; report still `ok` unless a mismatch occurs, but mark ambiguity.
  * `mismatch` if any derived point does **not** match solver within `tol`.

---

### 16.2 Workflow

1. **Prep**

   * Use the **desugared** program (so high‑level shapes are already expanded to canonical facts).
   * Build a typed catalog of **objects** (lines/rays/segments, circles) and **facts** (e.g., “H is foot from X to AB”).

2. **Build the Derivation Graph**

   * Nodes are **points** (both given & to‑be‑derived).
   * Directed edges represent **dependencies** required to compute a point (e.g., `H ← {A,B,X}` for a foot from `X` to `AB`).
   * Include **path nodes** when a rule depends on a path (e.g., `line(A,B)`), but the graph is *topologically sorted on points*: path nodes don’t participate in ordering; they are recomputed on demand from their defining points.
   * If multiple rules can derive the same point, create parallel candidate nodes; the evaluator will reconcile them (they must agree within `tol` or we raise a consistency warning).

3. **Topological Sort**

   * **Base points**: those that **cannot** be uniquely derived from any rule given current knowledge (they must be provided by the solver). These appear at layer 0.
   * Higher layers contain points derivable from earlier layers. Cycles are broken by leaving involved points as base (no derivation).

4. **Evaluate (derive coordinates)**

   * For each point in topo order, execute its **derivation rule(s)** (see §16.4).
   * A rule may produce **1** or **2** candidates; for 2‑candidate rules, apply **filters**:

     * hard filters: ray/segment membership, `diameter` reflection, declared ordering constraints that are *hard* (e.g., “on segment”, “between”).
     * soft selectors from **options**: `choose=near|far` (w.r.t. `anchor=P`), `choose=left|right` (w.r.t. oriented `ref=A-B`), `choose=cw|ccw` (around `anchor`, optionally `ref`), see §4.
   * If >1 candidates remain, leave them all and set `chosen_by="closest-to-solver"` **only** for the comparison step (we do not collapse the set permanently).

5. **Compare against solver coordinates**

   * For each derived point `P`, compute `dist = min_{c∈candidates} ‖c - P_solved‖`.
   * `match="yes"` iff `dist ≤ tol`.
   * If no candidates (rule not applicable), mark as **not derivable** and exclude from pass/fail.

6. **Report & diagnostics**

   * Produce a human‑readable summary with per‑point info, a list of **unused facts** (good for catching under‑modeling), and a compact **graph export** (JSON) for visualization.

---

### 16.3 Geometry primitives used by DDC

To compute candidates deterministically, DDC needs exact parameterizations of basic objects. All formulas below use vectors in ℝ²; `⊥` denotes a 90° rotation (e.g., `R⊥(x,y)=(−y,x)`), and `·`, `×` are dot/cross (2D scalar cross).

* **Line(A,B)**: `ℓ(t) = A + t·(B−A)`.
* **Ray(A,B)**: same, with constraint `t ≥ 0`.
* **Segment(A,B)**: same, with `t ∈ [0,1]`.
* **Circle center O radius-through R**: `center=O`, `r = ‖R−O‖`.
* **Circle through (A,B,C)**: `center = intersect(PerpBisector(AB), PerpBisector(AC))`, `r = ‖center−A‖`.

**Path helpers** (used internally to form `Path` instances):

* `PerpBisector(A,B)`: passes through `M=(A+B)/2`, direction `(B−A)⊥`.
* `ParallelThrough(P; A,B)`: line through `P` with direction `B−A`.
* `PerpendicularAt(T; A,B)`: line through `T` with direction `(B−A)⊥`.
* `AngleBisector(U,V,W, external=False)`: at `V`, bisects oriented angle `(VU, VW)`; `external` flips direction.

---

### 16.4 Derivation rule library

Each rule is implemented as:

```python
@dataclass
class Rule:
    name: str
    inputs: Set[Symbol]   # required known points/paths
    produces: Symbol      # point being derived
    multiplicity: Literal[1,2]
    solver: Callable[..., List[Point]]  # returns candidate coordinates
    filters: List[Callable[..., None]]  # optional hard filters (segment/ray membership, etc.)
    soft_selectors: List[str]           # uses program opts: choose/anchor/ref
```

**Core rules (unique)**

1. **Midpoint** — `midpoint M of A-B`
   `M = (A + B) / 2`.

2. **Foot of perpendicular** — `foot H from X to A-B`
   Let `u = B−A`, `t = ((X−A)·u) / (u·u)`, `H = A + t·u`.

3. **Diameter reflection** — `diameter A-B to circle center O`
   If one endpoint known and `O` known, derive the other: `B = 2O − A` (or `A = 2O − B`).

4. **Incenter (triangle)** — from `incircle of A-B-C`
   `I = intersect(bisector at A, bisector at B)`, using normalized direction bisectors.

5. **Circumcenter (triangle)** — from `circumcircle of A-B-C` or `circle through (A,B,C)`
   `O = intersect(PerpBisector(AB), PerpBisector(AC))`.

**Intersection rules (may be 1 or 2 candidates)**

6. **Intersect(line, line)** — solve 2×2 linear system; unique unless parallel/collinear.

7. **Intersect(line, ray/segment)** — compute line–line intersection, then **hard filter** by parameter range on ray/segment.

8. **Intersect(ray/segment, ray/segment)** — idem.

9. **Intersect(line, circle)** — quadratic in `t` on line param; **0/1/2** candidates; ray/segment filters apply.

10. **Intersect(circle, circle)** — classic two‑circle intersection; 0/1/2 candidates.

11. **Angle‑bisector with line/ray/segment** — parametric line of bisector with filters.

**Tangency rules**

12. **Tangent from external point to known circle** —
    From `line A-P tangent to circle center O at P` derive `P`.
    Let `d = A−O`, `D2 = ‖d‖²`, `r = radius`. If `D2 ≤ r²` → no real tangent.
    Tangency points:

    ```
    k = r^2 / D2
    h = r*sqrt(D2 - r^2) / D2
    P± = O + k*d ± h * d⊥
    ```

    **Multiplicity 2**; apply `choose=...` soft selectors and ray/segment filters if present.

13. **Point on circle + perpendicular at touchpoint** —
    For `tangent at T to circle center O`, if the tangent line’s second point is known and `T` unknown but constrained to a line/ray/segment, intersect that carrier with the circle(s) as above and filter by perpendicularity.

**Equal‑angles / collinear / concyclic (used as filters/guards)**

* `collinear(P1,...,Pn)` — used to *reduce* candidate sets by checking cross( P1P2, P1Pi ) ≈ 0.
* `concyclic(P1,...,Pn)` — used to validate a constructed circle and prune candidates.
* `equal-angles(...)` — used to reject wrong branch if angle equality is present.

> The initial library above is sufficient to cover 95% of olympiad‑style derivations we encode today. Rules are **pluggable**; you can register more without changing the framework.

---

### 16.5 Hard vs. soft selection

* **Hard filters** (eliminate candidates):

  * Ray/segment membership (`t ≥ 0`, `0 ≤ t ≤ 1`).
  * Perpendicular/parallel requirements (e.g., `OT ⟂ AB` at tangency).
  * “Between” statements (e.g., `point A on segment B-C`).

* **Soft selectors** (bias only; do not eliminate unless inconsistent with all):

  * `choose=near|far anchor=Q`
  * `choose=left|right ref=A-B`
  * `choose=cw|ccw anchor=Q [ref=A-B]`

> If after applying **hard filters** more than one candidate remains and **no soft selector** is present, the point is **ambiguous**; DDC keeps all candidates and marks the point `ambiguous` (comparison may still succeed if solver picked any one of them).

---

### 16.6 Comparison policy & tolerances

* Compute `scene_scale = max(1.0, diag(bounding_box(points)))`.
* Default `tol = 1e-6 * scene_scale` (configurable).
* A point *matches* if its solver coordinate is within `tol` of any candidate.
* If multiple rules derive the same point, all must agree (pairwise distance ≤ `tol`) or the point is flagged `mismatch (conflicting rules)`.

---

### 16.7 Unused facts & coverage

* Any derivation‑eligible fact that **does not** participate in producing any candidate for any point is recorded in `unused_facts`.
  Common causes:

  * Under‑modeled programs (e.g., never using `equal-angles` that could disambiguate a branch).
  * Typos/IDs that don’t connect to the main graph.
  * Dead constraints (e.g., circle defined but no one uses it).

---

### 16.8 Failure modes & messages

* **Degenerate inputs** (parallel lines, concentric circles, `‖A−O‖ ≤ r` for tangency): emit a precise message alongside the rule name and the offending quantities.
* **Ambiguous** (multiple candidates survive hard filters and no soft selector): mark point `ambiguous`, include all candidates.
* **Mismatch** (solver point not within `tol` of any candidate): report the min distance and the candidate set.

---

### 16.9 CLI (optional)

```
geoscript-ir check path/to/problem.gs --dump-graph graph.json --tol 1e-7
```

* Prints a one‑line status and a table per point.
* Writes a JSON export of the derivation graph plus candidate coordinates.

---

### 16.10 Worked example (the tangent/tangent issue)

For

```
circle center O radius-through B
line A-B tangent to circle center O at B
line A-C tangent to circle center O at C
angle O-A-B [degrees=30]
segment A-B [length=5]
```

* **Derivable**: `C` from `(A, O, r)` via Rule 12 (two candidates).
* With no side selector, DDC marks `C` **ambiguous** but still **ok** if the solver’s `C` equals one of the candidates.
* If you add `point C on ray A-B [choose=ccw anchor=A]` (or `choose=left ref=A-B`), DDC treats `C` as **unique** (only one candidate survives), and any mirror solution becomes a **mismatch**.

---

### 16.11 Implementation notes

* Keep all computations in double precision; guard divisions by near‑zero with small eps (e.g., `1e-14 * scene_scale`).
* Never mutate solver coordinates; DDC is **read‑only** on the solution.
* The rule engine is deliberately small; each rule is < 30 lines and pure (no side effects).
* Unit tests: for every rule, craft fixtures with (a) normal, (b) degenerate, (c) ambiguous inputs; verify candidate counts and coordinates.

---

This module gives us a **deterministic oracle** for points that *should* be determined by the givens. When the solver converges to a wrong branch or mirror, DDC will flag it immediately, with an actionable trace (“which point, derived by which rule, from which inputs, disagreed by how much”).

### Notes for implementers

* Preserve the **symbolic** text for numbers like `sqrt(2)` / `3*sqrt(2)` for labels while using their numeric value in residuals.
* Keep the validator’s dry “translate” to surface early failures with source spans.
* Branch-picking options should compile to **small, scale-aware biases** (hinge terms) so they guide root selection without fighting metric constraints.

---
