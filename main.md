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

### Notes for implementers

* Preserve the **symbolic** text for numbers like `sqrt(2)` / `3*sqrt(2)` for labels while using their numeric value in residuals.
* Keep the validator’s dry “translate” to surface early failures with source spans.
* Branch-picking options should compile to **small, scale-aware biases** (hinge terms) so they guide root selection without fighting metric constraints.

---
