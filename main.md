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
* `target arc P-Q on circle center O [label="?BT"]`

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

**Anchor protection for seeding (initial guess).**
The solver’s initial-guess routine **must not** disturb the primary orientation gauge edge selected by the layout (or by the compiler when applying a unit-span gauge). Concretely:

1. Keep both endpoints of the primary gauge edge fixed at attempt `0`.
2. On reseeds (`attempt > 0`), do **not** random-rotate the entire configuration if a primary gauge edge is present; explore with jitter only.
3. Never jitter either endpoint of the primary gauge edge on any attempt.
   This guarantees that gauges start satisfied and avoids spurious flips on near-degenerate scenes.

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
* **Core API**:

  * `parse_program`, `validate`, `desugar`
  * **Pre-solve planning (default):**

    * `plan_derive(program: Program) -> DerivationPlan`
    * `compile_with_plan(program: Program, plan: DerivationPlan) -> Model`
  * **Solver**: `solve*(model_or_program, ...)`
  * **Post-solve DDC**: `derive_and_check(desugared: Program, solution: Solution, *, tol=None) -> DerivationReport`
  * TikZ helpers, prompts `BNF`, `LLM_PROMPT`.

```python
# NEW types (public sketch)
class FunctionalRule(TypedDict):
    name: str
    inputs: List[str]               # point ids needed
    eval: Callable[[Coords], Tuple[float,float]]  # pure function P = f(inputs)
    guard: Callable[[Coords], bool] # True if well-posed (not parallel/etc.)

class DerivationPlan(TypedDict):
    base_points: List[str]          # optimized by solver
    derived_points: Dict[str, FunctionalRule]  # computed on-the-fly
    ambiguous_points: List[str]     # remain variables (multi-root or branch)
    notes: List[str]
```

```python
# NEW: Seeding (initial guess) API
class SeedHint(TypedDict):
    kind: Literal[
        "on_path",          # point constrained to a Path
        "intersect",        # point is intersection of two Paths
        "length",           # numeric length for an edge
        "equal_length",     # group of equal-length edges
        "ratio",            # AB:CD = p:q
        "parallel",         # AB ∥ CD
        "perpendicular",    # AB ⟂ CD
        "tangent",          # line tangent to circle at T
        "concyclic"         # points share a circle (soft, optional)
    ]
    point: Optional[str]          # target point (if applies)
    path: Optional[PathSpec]      # normalized Path (see §16.3)
    path2: Optional[PathSpec]     # for 'intersect'
    payload: Dict[str, Any]       # choose/anchor/ref, numeric p/q, lengths, etc.

class SeedHints(TypedDict):
    by_point: Dict[str, List[SeedHint]]
    global_hints: List[SeedHint]  # equal_length groups, ratios, parallels, etc.

def build_seed_hints(program: Program, plan: Optional[DerivationPlan]) -> SeedHints: ...

def initial_guess(model: Model,
                  rng: np.random.Generator,
                  attempt: int,
                  *,
                  plan: Optional[DerivationPlan] = None) -> np.ndarray: ...
```

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

* **Inputs**:

  * a validated & desugared `Program`, and **optionally a `DerivationPlan`**.
* **Output**:
  `Solution { point_coords, success, max_residual, residual_breakdown, warnings }`, plus helpers to normalize coordinates for rendering.
* **Algorithm**:

  0. **Seeding**: compute an initial guess with the **seeding policy** defined in §18
     (layout/gauges respected; uses `build_seed_hints()` and `plan` when available).
  1. **Pre-solve planning (default)**:

     * If no explicit plan was provided, call `plan_derive(program)` to obtain a `DerivationPlan`.
     * The plan partitions points into:

       * `base_points` (decision variables),
       * `derived_points` (computed from functional rules),
       * `ambiguous_points` (kept as variables).
  2. **Model build**:

     * `compile_with_plan(program, plan)` produces
       `Model{ variables=base_points ∪ ambiguous_points, derived=derived_points, index, residuals, gauges, scale }`.
     * Residual evaluation **first** computes all `derived` via their `eval()` in topo order, guarded by `guard()`; then computes residuals using both optimized and derived coordinates.
  3. **Solve**: minimize all residual groups using LSQ (`method="trf"`); reseed if necessary; score candidates by success then residual magnitude.
* **Safety**: min-separation guards, edge floors, area floors, non-parallel margins for trapezoids, orientation gauges (prefer declared bases; otherwise unit-span).
* **Plan guards**: if any `guard()` fails at the initial seed (e.g., near-parallel lines in a derived intersection), the compiler emits a **plan degradation warning** and **promotes** the affected point(s) to `ambiguous_points` (variables), then recompiles **once**. Variable dimensionality is fixed thereafter.
* **Seeding guards**: the initial guess must respect primary gauge anchors (see §7) and avoid placing any polygon below its edge/area floors.

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

## 16) Deterministic Derivation — Plan & Check (DDC-Plan + DDC-Check)

**Goals.**

1. **DDC-Plan (pre-solve)**: statically identify points that are **functionally determined** (single-valued; no branch choice) by the givens, and compile them into **derived** nodes so the solver does **not** optimize them.
2. **DDC-Check (post-solve)**: for all deterministically derivable points (including multi-root situations filtered by hard constraints), compute candidates numerically and **cross-check** them against the solver’s output. It catches wrong branches, flipped orders, and inconsistent placements early.

### 16.0 DDC-Plan (pre-solve) — variable reduction

**What qualifies as functional (always single-valued):**

* Midpoint; Foot of perpendicular; Diameter reflection; Center from diameter.
* Incenter (intersection of internal bisectors); Circumcenter (intersection of perpendicular bisectors).
* Intersection of two **line-like** paths (Line, Ray, Segment, PerpBisector, PerpendicularAt, ParallelThrough, AngleBisector, MedianFrom, TangentLineAt) — unique except parallel/collinear (guarded).
* Tangency with a **known tangent line**: `line X-Y tangent to circle center O at T` ⇒ unique `T` by orthogonal projection (guarded by distance≈radius).
* **On∩On (line-like ∩ line-like)** synthesized from two `point P on ...` facts.

**What does *not* qualify (keep as variables):**

* Any rule with inherent **two roots** without a structural hard disambiguator at compile time:

  * `line ∩ circle`, `circle ∩ circle`, tangent from an external point (`line A-P tangent ... at P`), or any use of `choose=...` (branch selectors are numeric, not structural).
* On∩On with a circle unless the second path is a known tangent line (then use the tangent rule above).

**Plan construction (static, no coordinates needed):**

1. Build the **Derivation Graph** (see 16.2) using only rules that qualify as functional.
2. Topologically sort to partition points into `derived_points` and **base**. Cycles or rules depending on a non-functional dependency leave the target point as base.
3. Export `DerivationPlan{ base_points, derived_points, ambiguous_points, notes }`.

**Runtime evaluation:** In the compiled model, `derived_points` are evaluated *deterministically* each residual call; if a rule’s `guard()` fails at the initial seed, the compiler demotes that point to a variable and recompiles once (see §13, “Plan guards”).

---

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
      unused_facts: List[str]
      graph: DerivationGraphExport
  ```

  ```python
  class DerivedPointReport(TypedDict):
      rule: str
      inputs: List[str]
      candidates: List[Tuple[float,float]]
      chosen_by: Literal["unique","opts","ray/segment filter","closest-to-solver","undetermined"]
      match: Literal["yes","no"]
      dist: float
      notes: List[str]
  ```

  Overall `status` rules:

  * `ok` if every derivable point matches within `tol`.
  * `partial` if some points are not derivable (no rule or missing inputs) but all derived ones match.
  * `ambiguous` if a point has >1 valid candidates and the solver picked one; report still `ok` unless a mismatch occurs, but mark ambiguity.
  * `mismatch` if any derived point does **not** match solver within `tol`.

---

### 16.2 Workflow

1. **Prep**

   * Use the **desugared** program (so high-level shapes are already expanded to canonical facts).
   * Build a typed catalog of **objects** (lines/rays/segments, circles) and **facts** (e.g., “H is foot from X to AB”).

2. **Build the Derivation Graph**

   * Nodes are **points** (both given & to-be-derived).
   * Directed edges represent **dependencies** required to compute a point (e.g., `H ← {A,B,X}` for a foot from `X` to `AB`).
   * Include **path nodes** when a rule depends on a path (e.g., `line(A,B)`), but the graph is *topologically sorted on points*: path nodes don’t participate in ordering; they are recomputed on demand from their defining points.
   * If multiple rules can derive the same point, create parallel candidate nodes; the evaluator will reconcile them (they must agree within `tol` or we raise a consistency warning).

3. **Topological Sort**

   * **Base points**: those that **cannot** be uniquely derived from any rule given current knowledge (they must be provided by the solver). These appear at layer 0.
   * Higher layers contain points derivable from earlier layers. Cycles are broken by leaving involved points as base (no derivation).
   * **DDC-Plan vs DDC-Check**: the **Plan** uses only **functional** rules (no multi-root). The **Check** may use multi-root rules to generate candidate sets for verification.

**4. Evaluate (derive coordinates)**

* For each point in topo order, execute its **derivation rule(s)** (see §16.4).
* **In DDC-Plan** we only install rules that are **single-valued** (no candidate sets).
* **In DDC-Check** a rule may produce **1** or **2** candidates; for 2-candidate rules, apply **filters**:

  * hard filters: ray/segment membership, perpendicular/parallel/tangency guards, `diameter` reflection, declared ordering constraints that are *hard* (e.g., “on segment”, “between”).
  * **on∩on synthesis**: if a point is separately constrained to lie on two Paths (via two `point … on …` facts), synthesize an **intersection rule** for those two Paths on the fly (see §16.4, “On∩On synthetic rules”).
  * soft selectors from **options**: `choose=near|far`, `choose=left|right`, `choose=cw|ccw` (see §4).

5. **Compare against solver coordinates**

   * For each derived point `P`, compute `dist = min_{c∈candidates} ‖c - P_solved‖`.
   * `match="yes"` iff `dist ≤ tol`.
   * If no candidates (rule not applicable), mark as **not derivable** and exclude from pass/fail.

6. **Report & diagnostics**

   * Produce a human-readable summary with per-point info, a list of **unused facts** (good for catching under-modeling), and a compact **graph export** (JSON) for visualization.

---

### 16.3 Geometry primitives used by DDC

To compute candidates deterministically, DDC needs exact parameterizations of basic objects. All formulas below use vectors in ℝ²; `⊥` denotes a 90° rotation (e.g., `R⊥(x,y)=(−y,x)`), and `·`, `×` are dot/cross (2D scalar cross).

* **Line(A,B)**: `ℓ(t) = A + t·(B−A)`.
* **Ray(A,B)**: same, with constraint `t ≥ 0`.
* **Segment(A,B)**: same, with `t ∈ [0,1]`.
* **Circle center O radius-through R**: `center=O`, `r = ‖R−O‖`.
* **Circle through (A,B,C)**: `center = intersect(PerpBisector(AB), PerpBisector(AC))`, `r = ‖center−A‖`.

**Path helpers** (used internally to form `Path` instances):

* `Line(A,B)`: passes through A, direction `B−A`.
* `Ray(A,B)`: `Line(A,B)` with parameter `t≥0`.
* `Segment(A,B)`: `Line(A,B)` with parameter `t∈[0,1]`.
* `Circle(O; r)`: `center=O`, `r=‖R−O‖` where `R` is that circle’s `radius-through` witness.
* `PerpBisector(A,B)`: passes mid `M=(A+B)/2`, direction `(B−A)⊥`.
* `ParallelThrough(P; A,B)`: line through `P` with direction `B−A`.
* `PerpendicularAt(T; A,B)`: line through `T` with direction `(B−A)⊥`.
* `AngleBisector(U,V,W, external=False)`: at `V`, bisects oriented angle `(VU, VW)`; if `external=True` use external bisector. Direction `dir = ( (U−V)/‖U−V‖ + s*(W−V)/‖W−V‖ )`, with `s=+1` (internal) or `s=−1` (external); if `dir≈0`, declare degenerate.
* **`MedianFrom(P; A,B)`**: line through vertex `P` and midpoint `M=(A+B)/2`.
* **`TangentLineAt(T; O)`**: line through `T` with direction `(T−O)⊥` (well-defined if `T≠O`).

---

## 16.4 Derivation rule library

Each rule is implemented as:

```python
@dataclass
class Rule:
    name: str
    inputs: Set[Symbol]   # required known points/paths/facts
    produces: Symbol      # point being derived
    multiplicity: Literal[1,2]
    solver: Callable[..., List[Point]]  # returns candidate coordinates
    filters: List[Callable[..., None]]  # hard filters (segment/ray/tangency/parallel/etc.)
    soft_selectors: List[str]           # uses program opts: choose/anchor/ref
```

### A. Core unique rules

1. **Midpoint** — from `midpoint M of A-B`
   `M = (A + B) / 2`.

2. **Foot of perpendicular** — from `foot H from X to A-B`
   Let `u = B−A`, `t = ((X−A)·u) / (u·u)`, `H = A + t·u`.

3. **Diameter reflection** — from `diameter A-B to circle center O`
   If one endpoint and `O` are known: `B = 2O − A` (or symmetrically `A = 2O − B`).

4. **Center from diameter endpoints** — from `diameter A-B to circle center O`
   If `A` and `B` known: **derive** `O = (A + B) / 2`.

5. **Incenter (triangle)** — from `incircle of A-B-C`
   `I = intersect(AngleBisector(B,A,C), AngleBisector(A,B,C))` (internal bisectors).

6. **Circumcenter (triangle)** — from `circumcircle of A-B-C` or `circle through (A,B,C)`
   `O = intersect(PerpBisector(A,B), PerpBisector(A,C))`.

7. **Midpoint from equal segments + collinearity** — from
   `collinear(A,M,B)` **and** `equal-segments (A-M ; M-B)` (or included in a group)
   Derive `M = (A + B)/2`. If multiple equalities are present, verify consistency.

### B. Intersection rules (generic; may be 1 or 2 candidates)

These operate on any **line-like** path (Line, Ray, Segment, PerpBisector, ParallelThrough, PerpendicularAt, AngleBisector, MedianFrom, TangentLineAt), plus Circle:

8. **Intersect(line-like, line-like)** — unique unless parallel/collinear; apply **ray/segment** parameter filters.

9. **Intersect(line-like, circle)** — quadratic along the line parameter; **0/1/2** candidates; apply **ray/segment** filters.

10. **Intersect(circle, circle)** — classic two-circle intersection; **0/1/2** candidates.

> Special cases such as **Angle-bisector ∩ (line|circle)** or **Median ∩ …** are covered by 8–9 after parameterizing the path (see §16.3). No separate bespoke rules are required.

### C. Tangency rules (completed)

11. **Tangent from external point to circle** — from
    `line A-P tangent to circle center O at P` (A known, O known, circle has witness `r`)
    Let `d = A−O`, `D2 = ‖d‖²`. If `D2 ≤ r²`: **no real tangent**. Else

```
k = r^2 / D2
h = r*sqrt(D2 - r^2) / D2
P± = O + k*d ± h * d⊥
```

Multiplicity **2**; apply soft selectors (`choose=near|far|left|right|cw|ccw`) and membership filters if the “tangent line” is further constrained to a ray/segment.

12. **Touchpoint from known tangent line** — from
    `line X-Y tangent to circle center O at T` (X,Y,O known; r known)
    Compute the **foot of O on line XY**: let `v=Y−X`, `t = ((O−X)·v)/(v·v)`, `T0 = X + t·v`.
    Validate `|‖T0−O‖ − r| ≤ ε_tan` (scale-aware). If valid, **unique** candidate `T=T0`; else **degenerate** (report).

13. **Touchpoint from tangent direction at T** — from
    `tangent at T to circle center O` **and** a second constraint placing `T` on a **line-like** carrier `ℓ` (e.g., `point T on line A-B` / `ray` / `segment`)
    Compute `T = intersect(ℓ, Circle(O; r))`. Keep only solutions that satisfy `OT ⟂ ℓ`. Multiplicity **≤2**, then filtered to **≤1** by the perpendicular guard.

14. **Tangent line through a given point (line unknown)** — from
    `point A on tangent at T to circle center O` with unknown `T` (expressed together as: `line A-T tangent to circle center O at T`).
    This is a thin wrapper around Rule 11 producing the same `T±`. If the script additionally says `point T on (ray/segment A-?)`, apply hard membership filters.

### D. On∩On **synthetic** rules (new)

These rules are **synthesized** when the fact base contains **two independent `point P on <Path>`** constraints for the same `P` (even if the source script didn’t use an explicit `intersect(...)` statement):

15. **On∩On: line-like ∩ line-like** — produce unique intersection unless parallel/collinear; **ray/segment** filters apply.

16. **On∩On: line-like ∩ circle** — as in rule 9 (with ray/segment filters).

17. **On∩On: circle ∩ circle** — as in rule 10.

18. **On∩On with tangent guard** — if one of the two paths is a **TangentLineAt(T;O)** or equivalent constraint (from `tangent at T …`) and the other is **line-like**, keep only candidates that satisfy `OT ⟂` (other line’s direction at `T`).

> These synthetic rules are crucial because authors often write two `point … on …` lines instead of a single `intersect(...) … at …`. DDC should not force a particular authoring style.

### E. Filters & selectors (unchanged, expanded by context)

* **Hard filters**: ray/segment membership; perpendicular/parallel/tangency checks; “between” constraints; diameter reflection; circle membership with the correct witness `r`; **touchpoint validation** (`|‖OT‖−r|≤ε_tan`).
* **Soft selectors**: `choose=near|far anchor=Q`, `choose=left|right ref=A-B`, `choose=cw|ccw anchor=Q [ref=A-B]`. Tangency rules honor these to pick one of the two tangents.

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

> If after applying **hard filters** more than one candidate remains and **no soft selector** is present, the point is **ambiguous**; DDC keeps all candidates and marks the point `ambiguous` (comparison may still succeed if the solver picked any one of them). **DDC-Plan never installs such rules; those points remain variables.**

---

### 16.6 Comparison policy & tolerances

* Compute `scene_scale = max(1.0, diag(bounding_box(points)))`.
* Default `tol = 1e-6 * scene_scale`.
  Tangency validation uses `ε_tan = 5 * tol` (slightly looser to absorb projection noise).
* A point *matches* if its solver coordinate is within `tol` of any candidate.
* If multiple rules derive the same point, all must agree (pairwise distance ≤ `tol`) or the point is flagged `mismatch (conflicting rules)`.

---

### 16.7 Unused facts & coverage

* Any derivation-eligible fact that **does not** participate in producing any candidate for any point is recorded in `unused_facts`.
  Common causes:

  * Under-modeled programs (e.g., never using `equal-angles` that could disambiguate a branch).
  * Typos/IDs that don’t connect to the main graph.
  * Dead constraints (e.g., circle defined but no one uses it).

---

### 16.8 Failure modes & messages

* **Degenerate inputs**: parallel lines, concentric circles, `‖A−O‖ ≤ r` for tangency; for rule 12, `|dist(O, line XY) − r| > ε_tan`.
* **Ambiguous**: multiple candidates survive hard filters and no soft selector.
* **Mismatch**: solver point not within `tol` of any candidate.
* **On∩On unsupported pair**: if two `on` constraints reduce to the **same** path (e.g., both `point P on line A-B`) with no second independent path, mark **not derivable**.

---

### 16.9 CLI (optional)

```
geoscript-ir check path/to/problem.gs --dump-graph graph.json --tol 1e-7
```

* Prints a one-line status and a table per point.
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

* Keep all computations in double precision; guard divisions by near-zero with small eps (e.g., `1e-14 * scene_scale`).
* Never mutate solver coordinates; DDC is **read-only** on the solution.
* The rule engine is deliberately small; each rule is < 30 lines and pure (no side effects).
* Unit tests: for every rule, craft fixtures with (a) normal, (b) degenerate, (c) ambiguous inputs; verify candidate counts and coordinates.
* **Plan tests**: ensure each functional rule compiles into a `derived_point`, its `guard()` is exercised, and demotion to variable occurs on forced degeneracy (parallel). See §17.2 and §17.9.

---

This module gives us a **deterministic oracle** for points that *should* be determined by the givens. When the solver converges to a wrong branch or mirror, DDC will flag it immediately, with an actionable trace (“which point, derived by which rule, from which inputs, disagreed by how much”).

---

## 17) Integration-Test Flow (Solver + DDC Verification)

The integration test suite now runs in **three** phases:

1. **DDC-Plan compilation (pre-solve)** — compiles a plan and **reduces the variable set** by deriving functional points.
2. **Solver convergence test** — checks that `scipy.optimize.least_squares` terminates successfully and residuals fall below tolerance.
3. **DDC-Check (post-solve)** — verifies that all points that can be *derived numerically* match the solver’s coordinates within tolerance.

Only if **all** phases pass is a geometry test considered **successful**.

---

### 17.1 Test structure

Each test case contains:

| Field              | Meaning                                                                    |
| ------------------ | -------------------------------------------------------------------------- |
| `source`           | GeoScript program text                                                     |
| `expect_success`   | whether the problem is expected to be solvable                             |
| `expected_targets` | optional ground-truth values for `target ...` statements                   |
| `tol_solver`       | solver convergence tolerance (`max_residual`)                              |
| `tol_ddc`          | DDC geometric match tolerance                                              |
| `allow_ambiguous`  | whether ambiguous but consistent branches are acceptable (default `false`) |
| `expect_var_drop`  | expected reduction in variable count due to Plan (int or min ratio)        |
| `force_parallel`   | optional flag to seed near-parallel paths to exercise Plan demotion        |

---

### 17.2 Execution pipeline

> The DDC engine **auto-synthesizes** On∩On intersection rules (§16.4-D) and supports the complete tangency suite (§16.4-C). The **Plan** reduces variables before solving.

```
for case in test_cases:
    program = parse_program(case.source)
    validate(program)
    desugar(program)

    # --- NEW: pre-solve planning & compilation
    plan = plan_derive(program)
    if case.expect_var_drop is not None:
        assert variable_reduction(plan) >= case.expect_var_drop
    model = compile_with_plan(program, plan)
    # demotion-on-guard-failure occurs once inside compile_with_plan at first evaluation

    solution = solve(model, tol=case.tol_solver)
    assert solution.success, "Solver did not converge"
    assert solution.max_residual <= case.tol_solver

    report = derive_and_check(program, solution, tol=case.tol_ddc)
    evaluate_ddc(report, allow_ambiguous=case.allow_ambiguous)
```

---

### 17.3 DDC evaluation policy

| Condition                                                 | Result               | Notes                                                  |
| --------------------------------------------------------- | -------------------- | ------------------------------------------------------ |
| `report.status == "ok"`                                   | ✅ Pass               | All derivable points match solver coordinates.         |
| `report.status == "partial"`                              | ⚠️ Pass with warning | Some points not derivable, but all derived ones match. |
| `report.status == "ambiguous"` and `allow_ambiguous=True` | ⚠️ Pass with warning | Multiple valid branches consistent with constraints.   |
| `report.status == "mismatch"`                             | ❌ Fail               | At least one derived point differs > `tol_ddc`.        |

Each mismatch entry shows:

```
Point: C
Rule : tangent_from(A, circle O)
Dist : 0.04321
Chosen_by : closest-to-solver
Candidates:
  (2.1, 4.5)
  (-2.1, 4.5)
```

This helps identify mirror or wrong-branch errors.

---

### 17.4 Regression expectations

* **Plan** reduces variables in common scenes (midpoints, feet, in/circumcenters, line∩line intersections).

* **Check** must satisfy `report.status in {"ok","partial"}` (or `"ambiguous"` if explicitly allowed).

* The continuous-integration run will print both solver and DDC summaries:

  ```
  ✅ Plan: -4 variables (derived: M,H,O,X)
  ✅ Solver converged (residual 3.1e-9)
  ✅ DDC OK (12/12 points matched)
  ```

  or

  ```
  ✅ Plan: -2 variables
  ✅ Solver converged (residual 1.0e-8)
  ❌ DDC mismatch: Point C off by 0.04
  ```

* DDC must validate **touchpoints** from:

  * `line X-Y tangent to circle center O at T` (unique via foot-projection).
  * `line A-T tangent … at T` (two candidates from external point).

* DDC must derive points constrained by **two `on` statements** (line/line, line/circle, circle/circle).

* Bisector/median paths used in `intersect(...)` or via On∩On must be handled as **line-like** paths.

> Tangency validations use `ε_tan = 5×tol_ddc` internally for the “distance to circle equals r” check; the **point match** still uses `tol_ddc`.

---

### 17.5 Optional artifacts

When any case fails or warns:

* **`*.ddc.json`** — full derivation graph (`report.graph`) for visual inspection.
* **`*.scene.png`** — TikZ/Matplotlib snapshot with mismatched points highlighted in red.
* **`*.summary.txt`** — plain-text report with per-point distances and status.

These files are stored in `/tmp/geoscript_tests/<case_id>/` by default.

---

### 17.6 CI acceptance thresholds

| Metric                                  | Threshold            |
| --------------------------------------- | -------------------- |
| Solver success rate                     | ≥ 98 %               |
| DDC “ok/partial/allowed ambiguous” rate | ≥ 95 %               |
| Mean geometric deviation (`dist`)       | ≤ 1e-6 × scene_scale |
| Max geometric deviation                 | ≤ 5e-6 × scene_scale |

Any higher deviation fails the build and triggers a geometry-branch regression.

---

### 17.7 Developer workflow

1. Add or modify a `.gs` sample in `tests/scenes/`.
2. Run `pytest tests/test_integration.py --update-expected` to regenerate **plan snapshots**, **solver results**, and **DDC**.
3. Inspect `*.ddc.json` for new or ambiguous branches.
4. Commit only when DDC shows `"status": "ok"` or intentional `"ambiguous"` with justification.

---

### 17.8 Rationale

This layered testing ensures that
✔ we **reduce** the optimization problem (fewer variables),
✔ the solver **converges**,
✔ the geometry it converged to is **topologically correct**,
✔ mirror or wrong-branch solutions are caught automatically.

As a result, integration tests no longer silently accept “numerically nice but geometrically wrong” scenes, and they also avoid wasting iterations on deterministically derivable points.

---

### 17.9 Example plan tests (new)

**A. Midpoints demoted from variables**

```
scene "Two midpoints plan"
layout canonical=generic scale=1
points A, B, C, M, N
segment A-B
segment B-C
midpoint M of A-B
midpoint N of B-C
target point M
```

*Expectation*: `expect_var_drop >= 2`; `M,N ∈ derived_points`.

**B. Foot & line∩line**

```
scene "Foot + intersection"
layout canonical=generic scale=1
points A, B, C, D, H, X
line A-B
line C-D
foot H from C to A-B
intersect (line A-B) with (line C-D) at X
target point X
```

*Expectation*: `expect_var_drop >= 2` (`H` and `X` derived).

**C. Tangent with known tangent line (functional)**

```
scene "Projection touchpoint"
layout canonical=generic scale=1
points X, Y, O, T
line X-Y
circle center O radius-through X
line X-Y tangent to circle center O at T
target point T
```

*Expectation*: `T ∈ derived_points`.

**D. Degeneracy demotion (parallel)**
Use `force_parallel=True` to seed an almost-parallel `line A-B` and `line C-D`; compiler should demote the line∩line intersection to a variable and proceed.

---

### Notes for implementers

* Preserve the **symbolic** text for numbers like `sqrt(2)` / `3*sqrt(2)` for labels while using their numeric value in residuals.
* Keep the validator’s dry “translate” to surface early failures with source spans.
* Branch-picking options should compile to **small, scale-aware biases** (hinge terms) so they guide root selection without fighting metric constraints.

---

## 18) Initial Guess (Seeding) — Design & Requirements

> **Rationale.** The previous initializer biased only a few AST cases (e.g., `point … on segment`) and inspected raw residual statements. We replace it with a **pluggable, data-driven seeding policy** that:
>
> * derives hints from the **desugared program** and the **DerivationPlan** (not from raw residual text),
> * understands all **Path** forms (line, ray, segment, circle, perp-bisector, perpendicular-at, parallel-through, angle-bisector, median-from), and
> * respects **gauges**, **choose=…** options, and basic metric relations (length, equal-length, ratio).

### 18.0 Goals

1. Start in a **non-degenerate** configuration that already satisfies gauges and is geometrically plausible.
2. Use **deterministic derivations** (from **DDC-Plan**) to seed any single-valued points exactly; do **not** optimize what we can compute.
3. Handle common authoring patterns: `point … on <Path>`, `intersect(Path, Path)`, tangency, equal lengths/ratios, parallels, perpendiculars, rays/segments clamping.
4. Be **robust**: cheap O(n)–O(n log n) scans, no brittle branching on AST strings; work from normalized **PathSpec** and typed hints.
5. Keep the **primary gauge edge fixed** across attempts; jitter others modestly; escalate jitter per attempt without random global rotations (when a gauge is present).

### 18.1 Data sources

Seeding operates on:

* the **desugared** `Program`,
* the **DerivationPlan** (when available) to mark **derived_points** vs **variables**,
* the **normalized Path helpers** (§16.3),
* compiled metadata: primary **gauge edge** and `layout.scale`.

The compiler provides a `SeedHints` structure via `build_seed_hints()` that collects, per point:

* `on_path` hints from `point P on <Path>`,
* `intersect` hints from explicit `intersect(...) at P` **or** synthesized **On∩On** pairs (see §16.4-D),
* metric/global hints: `length`, `equal_length` groups, `ratio`, `parallel`, `perpendicular`, `tangent`, `concyclic` (optional).
  Soft selectors `choose=near|far|left|right|cw|ccw` with `anchor`/`ref` are attached in `payload`.

### 18.2 Seeding algorithm (single attempt)

Let `base = max(layout.scale, 1e-3)`. Let `(G1,G2)` be endpoints of the primary gauge edge if present.

**Stage A — Canonical scaffold.**

* Place `(G1,G2)` at canonical positions: `G1=(0,0)`, `G2=(base,0)`. If a third canonical vertex exists in the selected `layout` (e.g., `triangle_*`), place it above the base. Scatter other points on a small circle of radius `≈0.5·base` with blue-noise jitter (no global rotation).
* Mark `G1,G2` **protected**: they must not be moved or jittered by later stages or reseeds.

**Stage B — Deterministic derivations (Plan).**

* For every `P ∈ plan.derived_points`, evaluate its rule (guarded). If the guard fails at the seed, skip here (the compiler will demote once per §13). Otherwise **write** `P`’s coordinates and (if `P` is still a variable in the model) use them as the starting guess.

**Stage C — Path adherence for `on_path`.**
For each variable point `P` with `on_path` hints:

* **Line/Ray/Segment**: orthogonally project the current `P` onto the carrier line; for **Ray** clamp `t≥0`, for **Segment** clamp `t∈[0,1]`.
* **Circle(O; r)**: if `O` and a radius witness exist, place `P = O + r·û` where `û` is chosen by:

  * soft selectors (`choose=…` with `anchor`/`ref`),
  * otherwise the direction from `O` to the nearest already-placed neighbor of `P` (if any),
  * otherwise reuse `û` from the circle’s witness point.
* **Perp-bisector(A,B)**: place near the midpoint `M=(A+B)/2`.
* **PerpendicularAt(T; A,B)**, **ParallelThrough(P0; A,B)**, **MedianFrom(V; A,B)**, **AngleBisector(U,V,W)**: place on the carrier line through the appropriate anchor (use mid-t value for line-like paths).

**Stage D — Intersections.**
For each `intersect(Path1, Path2) at P` (including **On∩On** syntheses):

* Compute candidates using the generic intersection solvers (§16.4-B). Apply **hard filters** (segment/ray membership; tangency and perpendicular checks when applicable).
* If 2 candidates remain, apply **soft selectors** (`choose=…`) and, failing that, pick the candidate closest to `P`’s current coordinate. Write the chosen value into the guess.

**Stage E — Metric nudges (length/equal/ratio).**

* **Length** `‖AB‖=L`: if one endpoint is protected or already placed with high confidence, move the other endpoint onto the ray from the anchor by `L` (no movement if both endpoints protected).
* **Equal-length** groups: pick a reference edge with non-zero current length; for each other edge, if one endpoint is “anchored” (protected or appears in multiple hints), place the free endpoint to match the reference length, with direction taken from:

  * the averaged directions of any intersection/`on_path` targets involving that edge’s line-like carriers (if any),
  * otherwise the current edge direction,
  * otherwise the reference edge direction.
* **Ratio** `AB:CD = p:q`: when three endpoints are reasonably positioned, place the fourth along its current direction to satisfy the ratio approximately.

**Stage F — Parallels / Perpendiculars / Tangency.**

* If edges `AB ∥ CD` and `CD` has a stable direction, rotate `AB`’s direction to match; nudge the non-anchored endpoint.
* For `AB ⟂ CD`, rotate `AB` to the perpendicular of `CD`.
* For known tangent line `X–Y` to circle center `O` at `T`, seed `T` as the foot of the orthogonal projection of `O` onto line `XY` (validated by `|‖OT‖−r| ≤ ε_tan`).

**Stage G — Safety pass.**

* Enforce **min-separation** and polygon **edge/area floors** at the seed by spreading any colliding points slightly along existing directions (without moving `G1,G2`).

### 18.3 Reseed policy

* **No global rotation when a gauge edge exists.** (If no gauge is present, a random rotation is allowed on `attempt>0`.)
* Jitter is zero for `G1,G2` on **all** attempts. For other points:

  * `attempt==0`: very small iid jitter (`≈1%·base`) except for `G1,G2`.
  * `attempt>0`: Gaussian jitter with scale `σ_attempt = min(0.2, 0.05·(1+attempt))·base`, still clamped away from collisions by the safety pass.

### 18.4 Implementation notes

* **Do not** thread through raw `Stmt.kind` strings in the initializer. Build `SeedHints` during compilation:

  * consume `Placement` and `Obj` forms directly from the **desugared** program,
  * re-use Path normalization logic from §16.3,
  * synthesize **On∩On** intersection hints (§16.4-D).
* Prefer **constant-time** writes into the flat guess vector; maintain a scratch `coords[name]` map while seeding.
* Respect `plan.derived_points`: if a point is deterministically computed at runtime, use that coordinate as the seed (even if the model still optimizes it for robustness).
* Keep this module **pure**: no side effects on the `Model`; no randomness outside the passed `rng`.

### 18.5 Public surface & backwards compatibility

* `initial_guess(model, rng, attempt, *, plan=None)` becomes the single entry point used by the solver. Internally it calls `build_seed_hints()` if hints were not already attached to the `Model`.
* The prior initializer’s behavior (equilateral scaffold + ad-hoc segment/line intersection seeding) is subsumed by **Stages A–D**.

### 18.6 Tests (must-pass)

Add seeding tests to the integration flow (see §17):

1. **Gauge stability**

   * Scene: any triangle with `layout=triangle_ABC`.
   * Assert: on all attempts, the initial guess leaves `A,B` unchanged (within machine epsilon), no global rotation.

2. **On∩On synthesis (line ∩ circle)**

   ```
   points A, B, O, P
   line A-B
   circle center O radius-through A
   point P on line A-B
   point P on circle center O
   target point P
   ```

   * Assert: seed places `P` on (or very near) the analytical intersection, honoring `choose` if present.

3. **Known tangent touchpoint (functional)**

   ```
   points X, Y, O, T
   line X-Y
   circle center O radius-through X
   line X-Y tangent to circle center O at T
   ```

   * Assert: seed computes `T` by orthogonal projection; residual small before optimization.

4. **Equal-length propagation**

   ```
   points A, B, C, D
   segment A-B
   segment C-D
   segment A-B [length=3]
   equal-segments (A-B ; C-D)
   ```

   * Assert: the free endpoint of `C-D` is placed at distance `≈3` from its anchor and aligned by available direction hints (stable when we add an `on_path` for `C-D`).

5. **Ratio nudge**

   ```
   points A, B, C, D
   segment A-B
   segment C-D
   ratio (A-B : C-D = 2 : 3)
   ```

   * Assert: if three endpoints are reasonably placed by earlier stages, the fourth is nudged along its direction to satisfy the ratio approximately.

6. **Per-attempt jitter escalation**

   * Run `initial_guess` for attempts `0,1,2`.
   * Assert: `σ` increases, `A,B` remain fixed, no rotation when a gauge edge exists.

### 18.7 Migration guidance (for implementers)

* Remove initializer logic that inspects `spec.source.kind` directly; instead:

  * extend the compiler to output `SeedHints` by walking **desugared** statements,
  * normalize all `Path` payloads once (reuse §16.3 helpers),
  * synthesize **On∩On** intersection hints (§16.4-D).
* Replace bespoke `equal_segments` steering with the generic **Stage E** (reference edge + anchor-aware placement + direction averaging from hints).
* Fix the reseed behavior: do **not** random-rotate when a gauge edge is present; never jitter protected gauge endpoints.
* Keep the initial scaffold deterministic for `attempt==0`.

### 18.8 Out-of-scope (future)

* Using `concyclic` to estimate a circle center for seeding when no explicit center exists (possible via three-point least squares). Optional; not required for v1.
* Using `equal-angles` to orient directions at the seed (non-trivial; defer).

---

### 19) TikZ Rendering — Style & Codegen Contract

**19.0 Goals**

* **Visual clarity** with minimal ink: only draw declared carriers and essential construction lines.
* **Notation completeness**: equal‑segments ticks, equal‑angles arcs, and right‑angle squares are standardized.
* **Predictable layering** so labels/marks are legible.
* **Rule‑aware** rendering: respect `rules[...]` flags (`mark_right_angles_as_square`, `no_equations_on_sides`, `no_unicode_degree`, etc.).

---

**19.1 Preamble & libraries (minimal)**

```tex
\documentclass[border=2pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc,angles,quotes,intersections,decorations.markings,arrows.meta,positioning}
\tikzset{
  % global sizes (scale-aware; override per scene if needed)
  gs/dot radius/.store in=\gsDotR,       gs/dot radius=1.4pt,
  gs/line width/.store in=\gsLW,         gs/line width=0.8pt,
  gs/aux width/.store  in=\gsLWaux,      gs/aux width=0.6pt,
  gs/angle radius/.store in=\gsAngR,     gs/angle radius=8pt,
  gs/angle sep/.store   in=\gsAngSep,    gs/angle sep=2pt,
  gs/tick len/.store   in=\gsTick,       gs/tick len=4pt,
  point/.style={circle,fill=black,inner sep=0pt,minimum size=0pt},
  ptlabel/.style={font=\footnotesize, inner sep=1pt},
  carrier/.style={line width=\gsLW},
  circle/.style={line width=\gsLW},
  aux/.style={line width=\gsLWaux, dash pattern=on 3pt off 2pt},
  tick1/.style={postaction=decorate, decoration={markings,
      mark=at position 0.5 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);}}},
  tick2/.style={postaction=decorate, decoration={markings,
      mark=at position 0.4 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);},
      mark=at position 0.6 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);}}},
  tick3/.style={postaction=decorate, decoration={markings,
      mark=at position 0.35 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);},
      mark=at position 0.5  with {\draw (-\gsTick/2,0)--(\gsTick/2,0);},
      mark=at position 0.65 with {\draw (-\gsTick/2,0)--(\gsTick/2,0);}}},
}
% optional layers
\pgfdeclarelayer{bg}\pgfdeclarelayer{fg}\pgfsetlayers{bg,main,fg}
```

**19.2 Painter’s order**

1. **bg**: light fills (rare)
2. **main**: carriers — polygon sides, declared segments, circles
3. **fg**: construction lines (aux), angle/segment marks, labels, target highlights

**19.3 What to draw (and what **not** to draw)**

* **Polygons** (`triangle`, `quadrilateral`, `trapezoid`, …): draw *only* their carrier edges with `carrier` style (no diagonals unless declared or used for an explicit intersection to place a visible point).
* **Segments explicitly declared**: draw with `carrier`.
* **Lines/Rays**: draw as **auxiliary** (extend beyond endpoints; `aux` style). Rays get `-{Latex[length=2mm]}` heads.
* **Intersections**: if `X` is defined by `intersect(Path1, Path2)`, render both carriers as `aux` **only if** they are not already drawn elsewhere; otherwise, omit.
* **Circles**: draw with `circle` style. Do **not** auto-draw radii to all named points; draw a radius only if explicitly present, or needed for a right-angle mark at a tangent touchpoint.
* **Feet / medians / bisectors**: draw their carrier lines `aux` only if they appear in a `segment`/`ray`/`line` object or define a visible point; otherwise render only the **mark** (e.g., right‑angle square at the foot).

**19.4 Point symbols & labels**

* Each named point: `\fill (P) circle (\gsDotR);` then `\node[ptlabel,<anchor>] at (P){$P$};`
* Default **anchor heuristic**: choose `above` if the point is on the lower half of the bbox, `below` if on the upper half; break ties using local edge directions to avoid overlap. Honor explicit `label point P [pos=...]`.

**19.5 Side labels vs. ticks**

* If `rules[no_equations_on_sides=true]`, **suppress** all auto numeric side labels. Use `sidelabel A-B "..."` to force a sloped label (`node[midway, sloped, above|below]`).
* **Equal segments**: for each group, apply `tick1` / `tick2` / `tick3` to every segment in that group. If >3 groups, cycle ticks then add `densely dashed` to distinguish.
* Side labels are always emitted inside math mode. We normalise `sqrt(...)` products (e.g., `3*sqrt(2)`) **only** within that LaTeX context so radicals never appear as plain text.

**19.6 Angles**

* **Numeric angle** (`angle A-B-C [degrees=θ]`): choose the ray order whose counter-clockwise measurement best matches `θ`, then draw one arc at `B` via `pic`:

  ```tex
  \path pic[draw, angle radius=\gsAngR, "$\num{θ}$"{scale=0.9}] {angle=A--B--C};
  ```

  Respect `rules[no_unicode_degree]` by always using `^\circ`. When no `degrees=` metadata exists, pick the ordering whose counter-clockwise sweep is < `180^\circ` (or closest to it if the configuration is straight/reflex) so TikZ draws the minor arc.
* **Right angle** (`right-angle A-B-C`): always draw the square symbol at `B` via the TikZ `right angle` pic, never an arc or `$90^\circ$` label:

  ```tex
  \path pic[draw, angle radius=\gsAngR] {right angle=A--B--C};
  ```
* **Equal angles** (`equal-angles (A-B-C, ... ; D-E-F, ...)`): for each *group*, draw **n arcs** (n=group index: 1=single, 2=double, 3=triple) at the relevant vertices. Radii are `\gsAngR`, `\gsAngR+\gsAngSep`, `\gsAngR+2\gsAngSep` for the multiple arcs, no labels.

**19.7 Medians, bisectors, altitudes (marks)**

* **Median** `median from V to A-B midpoint M`: draw `V–M` if `segment`/`line` is present; otherwise omit the line. Optionally mark `M` with a small notch on `AB`.
* **Angle bisector** at `B`: draw the **double‑arc** mark at `B` (no line) unless explicitly requested as a `line`/`segment`.
* **Altitude** / **foot** `foot H from X to A-B`: draw a **right‑angle square** at `H` and draw `X–H` as `aux` *only if* altitude line is declared or needed to show a construction.

**19.8 Tangency**

* `tangent at T to circle center O`: draw tangent **line** (aux) through `T`, draw the **radius** `O–T` (carrier or aux), and a **right‑angle square** at `T`.
* `line X-Y tangent to circle center O at T`: same; if there’s no radius elsewhere, draw `O–T` thin to support the square.

**19.9 Targets & highlights**

* **Target angle/arc**: re‑draw that arc **on the foreground** with thicker width (`line width=\gsLW+0.2pt`) and (optionally) a mild color (`black!85`) and a label `"?"` if `label="?"` is set in the IR.
* **Target length / point / circle**: annotate minimally (e.g., halo box “?” near the object), but **do not** print equations on sides unless forced via `sidelabel`.

**19.10 Bounds & scaling**

* Compute bbox over all *drawn* primitives; add a 5–8 mm margin and avoid wrapper packages. Use `standalone` class borders to fit tightly.

**19.11 Implementation plan (changes to `tikz_codegen/generator.py`)**

> Keep generator responsibilities **separate** from solving. Build a small **RenderPlan** (like your DDC plan) and then emit TikZ from that plan.

**19.11.1 Build a `RenderPlan` (new)**

```python
@dataclass
class RenderPlan:
    points: Dict[str, Tuple[float,float]]
    carriers: List[Tuple[str,str,Dict]]            # edges to draw (style=carrier)
    aux_lines: List[Tuple[PathSpec, Dict]]         # lines/rays to draw (style=aux)
    circles: List[Tuple[str,str,Dict]]             # (center, witness) -> circle style
    ticks: List[Tuple[str,str,int]]                # (A,B, group_index 1..3)
    equal_angle_groups: List[List[Tuple[str,str,str]]]  # per group: [(A,B,C),...]
    right_angles: List[Tuple[str,str,str]]         # (A,B,C)
    angle_arcs: List[Tuple[str,str,str, Optional[float], Optional[str]]] # (A,B,C, degrees?, label)
    labels: List[LabelSpec]                        # point & side labels
    highlights: List[HighlightSpec]                # targets
```

Populate this by walking the **desugared** program + options:

* **Carriers**: all declared `segment` and polygon sides.
* **Aux**: `line`/`ray` used for *explicitly drawn* carriers or to make an intersection visible; also bisector/median/altitude carriers *only* if they were explicitly declared.
* **Circles**: only as declared.
* **Ticks**: expand `equal-segments` into grouped edges (index groups 1..3; wrap if >3).
* **Equal angles**: collect into groups.
* **Right angles**: all `right-angle` statements; also synthesize from tangency if present.
* **Angle arcs**: from `angle A-B-C [degrees=..]`.
* **Labels**: from `label point` and `sidelabel`; suppress numeric side labels if `rules[no_equations_on_sides]`.
* **Highlights**: from `target ...`.

**19.11.2 Suppress redundancy**

* **Do not** add any `O–P` radius unless: (a) explicitly declared as a `segment`/`line`, or (b) required for a tangency right‑angle square.
* **Do not** draw connectors between arbitrary point pairs that aren’t in `carriers` or `aux_lines`.

**19.11.3 Layered emission**

* Emit in painter’s order (19.2).
* Use the standardized styles from 19.1 exactly once per document.
* **Right-angle squares**: use TikZ `pic` right‑angle symbol when `mark_right_angles_as_square=true`.
* **Equal‑angles**: for group *g*, draw `g` arcs at radius `\gsAngR + (k-1)\gsAngSep` (`k=1..g`) with no labels.
* **Ticks**: apply `tick{g}` style to the **segment** draw command; if the segment is not otherwise drawn (e.g., it’s only an abstract equality), draw the segment **thin dashed** only for the tick mark, or place two small ticks floating near endpoints (simpler: lightly draw the segment).
* **Angle labels**: always `$\,^\circ$` (LaTeX degree), respecting `no_unicode_degree`.
* **Targets**: re‑draw highlighted arc/edge/point on `fg` with `line width=\gsLW+0.2pt`.

**19.11.4 Rules mapping**

* `rules[mark_right_angles_as_square]` → choose square vs numeric “90°”.
* `rules[no_equations_on_sides]` → drop numeric edge labels unless created by explicit `sidelabel`.
* `rules[no_unicode_degree]` → no Unicode “°”; always `^\circ`.
* `rules[allow_auxiliary]` → if `false`, don’t draw bisector/median/altitude carriers unless explicitly declared as `segment/line/ray`; draw only marks.

**19.11.5 Minimal preamble**
Replace the heavy preamble with the minimal block in 19.1; return a `standalone` document without `varwidth/adjustbox/pgfplots`.

---
