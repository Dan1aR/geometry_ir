# GeoScript IR — Technical Specification (agent view)

GeoScript IR is a compact DSL for 2D Euclidean scenes. The toolchain parses source text into an AST, validates intent, optionally desugars to canonical primitives, and compiles a nonlinear model solved with ` `.

---

## 1) Design goals

1. **Readable problem statements.** Scripts mimic olympiad prose (“Trapezoid ABCD with base AD…”, “Circle with center O…”, “Find ∠DBE”).
2. **Explicit, well-posed constraints.** Every statement yields residuals, while translators enforce min separations, carrier edge floors, near-parallel cushions, and orientation gauges.
3. **Modular architecture.** Parsing/printing, validation plus desugaring, solver compilation, and TikZ export remain isolated modules with dedicated prompts.

---

## 2) Lexical rules

* `ID` tokens are case-insensitive; they normalize to uppercase (`a-b` ≡ `A-B`).
* Strings use double quotes with C-style escapes.
* Numbers accept decimals and scientific notation. Symbolic values (`sqrt(...)`, `3*sqrt(2)`) become `SymbolicNumber` (text plus numeric value).
* `#` introduces a line comment.

---

## 3) Grammar (BNF)

Programs must satisfy this grammar. Solver-facing extensions include branch picking, `collinear`, `concyclic`, `equal-angles`, `ratio`, and the `perp-bisector` / `parallel through` path forms.

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

> **Note:** String payloads in target statements (e.g. `target area ("Find area of ABC")`) are treated purely as annotations. Their text is not scanned for point identifiers and introduces no solver variables or residuals.

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

---

## 4) Legal options

Only the keys below are interpreted. The parser enforces syntax; the validator rejects unknown or mistyped options.

### Global

* `rules [...]` admits `no_equations_on_sides`, `no_solving`, `allow_auxiliary` (booleans).

### Branch selection (`point ... on ...`, `intersect (...) ... at ...`)

* `choose=near|far` + `anchor=P` → prefer the nearer/farther root relative to `P`.
* `choose=left|right` + `ref=A-B` → pick the left/right side of oriented line `AB`.
* `choose=cw|ccw` + `anchor=P` (optional `ref=A-B`) → bias clockwise/counter-clockwise around the anchor/reference.

Branch choices act as soft hinges to resolve two-root ambiguities without brittle hard constraints.

### Angles & arcs

* `angle A-B-C [degrees=NUMBER | label="..."]`
* `right-angle A-B-C [mark=square | label="..."]`
* `equal-angles (...) [label="..."]`
* `target angle A-B-C [label="..."]`
* `target arc P-Q on circle center O [label="?BT"]`

### Segments / polygons

* `segment A-B [length=NUMBER|SQRT|PRODUCT | label="..."]`
* `equal-segments (...) [label="..."]`
* `parallel-edges (...)`
* `polygon/triangle/... [isosceles=atA|atB|atC]`
* `trapezoid [...] [bases=A-D]`
* `trapezoid [isosceles=true|false]`

### Ratios

* `ratio (A-B : C-D = p : q)` with `p>0`, `q>0`.

### Incidence groups

* `collinear(A,B,C,...)` with ≥3 points.
* `concyclic(A,B,C,D,...)` with ≥3 points.

### Circles & tangency

* `circle center O radius-through B`
* `circle through (A, B, C, ...)`
* `tangent at T to circle center O`
* `line X-Y tangent to circle center O at T`
* `diameter A-B to circle center O`

### Annotations

* `label point P [label="..." pos=left|right|above|below]`
* `sidelabel A-B "..." [pos=left|right|above|below]` (renderers may add `mark=...`).

---

## 5) High-level objects → canonical facts

High-level constructs desugar to primitive relations understood by the solver.

* **triangle A-B-C** → carrier edges `AB`, `BC`, `CA`.
* **quadrilateral A-B-C-D** → `AB`, `BC`, `CD`, `DA`.
* **trapezoid A-B-C-D [bases=X-Y]** → quadrilateral + `parallel-edges (X-Y; opposite)` + a non-parallel hinge on legs; the named base is the preferred orientation gauge.
* **parallelogram A-B-C-D** → `parallel-edges (A-B; C-D)` and `parallel-edges (B-C; A-D)`; optional equalities follow author options.
* **rectangle** → parallelogram + `right-angle A-B-C`.
* **square** → rectangle + `equal-segments (A-B; B-C; C-D; D-A)`.
* **rhombus** → `equal-segments` on all sides + both parallel pairs.
* **collinear (P1,...,Pn)** → expand to full collinearity constraints (`n≥3`).
* **concyclic (P1,...,Pn)** → introduce a latent center and radius; enforce equal radii.
* **equal-angles (A-B-C, ... ; D-E-F, ...)** → tie every listed angle to a representative using `atan2` residuals.
* **ratio (A-B : C-D = p : q)** → enforce `q‖AB‖ − p‖CD‖ = 0`.

`circle through (...)` and `circumcircle of ...` both introduce a shared latent center/radius.

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

  * `choose=near|far, anchor=Q` → bias toward the nearer/farther intersection relative to `Q`.
  * `choose=left|right, ref=A-B` → penalize the wrong orientation sign against oriented line `AB`.
  * `choose=cw|ccw, anchor=Q` (optional `ref=A-B`) → prefer clockwise / counter-clockwise rotation about the anchor/reference.

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

### #NEW 6.5 Polygons & structural guards

> This appendix **revises §6.5** (Polygons & structural guards) and **extends §18** (Seeding) to prevent “valid but squished” shapes (e.g., near-zero height parallelograms / trapezoids). All guards are **soft, scale-aware hinge residuals** that only act when a scene is under-constrained.
* **Polygons & structural guards.** Every declared polygon receives:
  (a) **edge floors** (to avoid zero-length sides) and **min-separation** hinges;
  (b) an **area floor** (r_{\text{area}}) with (A_{\min}=\varepsilon_A\cdot L_{\max}^2);
  (c) a **height floor** (r_{\text{height}}) as in **S.2** (triangles: all three with reduced weight; parallelograms: two independent heights; trapezoids: both bases);
  (d) an **adjacent-side angle cushion** (r_{\text{angle}}) as in **S.3** at each vertex.
  These guards are aggregated with a small **shape weight** (w_{\text{shape}}) so they never fight explicit metric constraints; they activate only to prevent needle-like degeneracies or near-parallel collapses in under-constrained scenes.

---

## S.1 Constants (defaults)

```
ε_h     = 0.06    # min altitude as a fraction of a nearby side length
s_min   = 0.10    # min |sin(angle)| cushion between adjacent edges (~5.7°)
ε_A     = 0.02    # area floor factor relative to longest side squared
w_shape = 0.05    # small weight for all "shape" residuals (≪ 1.0 for hard facts)
```

Implement these in the residual builder’s configuration; expose them as tunables.

---

## S.2 Height floor (altitude hinge)

For side (AB) and opposite vertex (C):
[
h(AB;C)=\frac{|(B{-}A)\times(C{-}A)|}{|B{-}A|},\quad
h_{\min}=\varepsilon_h\cdot \max{|B{-}A|,|C{-}B|}.
]
Residual:
[
r_{\text{height}}(A,B,C)=\max!\big(0,;h_{\min}-h(AB;C)\big).
]

**Where to apply**

* **Triangle (ABC)**: apply to *all three* altitudes with weight (w_{\text{shape}}/3) each (or to the smallest altitude once).
* **Parallelogram (ABCD)**: apply to **two independent** heights, e.g. (h(AB;C)) and (h(BC;D)).
* **Trapezoid (ABCD)**: apply to the **bases** (both heights from the non-base vertices).

---

## S.3 Adjacent-side angle cushion (non-parallel margin)

Let unit directions (u=\frac{B-A}{|B-A|}), (v=\frac{C-B}{|C-B|}), and (s=|u\times v|=|\sin\angle ABC|).
[
r_{\text{angle}}(A,B,C)=\max!\big(0,; s_{\min}-s\big).
]

**Where to apply**
At each **declared polygon** vertex (triangles, trapezoids, parallelograms, and special quads). For rectangles/squares the right-angle constraint dominates; the cushion rarely activates.

---

## S.4 Area floor (keep, but scale by longest edge)

Let (L_{\max}=\max) side length of the polygon, (A=) polygon area.
[
A_{\min}=\varepsilon_A\cdot L_{\max}^2,\quad
r_{\text{area}}=\max(0,;A_{\min}-A).
]

## S.6 Residual aggregation

Add the guards where polygons are expanded in the desugared program:

```
residuals += w_shape * [
  r_height(...), r_angle(...), r_area(...),  # per-object as applicable
  ...
]
```

Ensure these **do not** participate in DDC (§16) — they are not geometric facts, only aesthetic stabilizers.

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
  * `print_program(program, *, original_only=False)` re-serializes the full grammar (including placement primitives like `midpoint` and `foot`). It either prints a faithful statement or raises `ValueError` for unknown/malformed kinds; `format_stmt` is a one-line wrapper for single statements.
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
5. Keep the **primary gauge edge fixed** across attempts; freeze the attempt-0 gauge alignment as the baseline, and record cumulative jitter norms so each reseed increases the Sobol σ monotonically (no random global rotations when a gauge is present).

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

> **GraphMDS safety valve.** After classical MDS we rescale to `≈0.75·base`, clip coordinates to `±4·base`, and abort the MDS seed entirely when the positive eigenspectrum is too ill-conditioned (`cond > 1e6`), falling back to Sobol.

**Stage B — Deterministic derivations (Plan).**

* For every `P ∈ plan.derived_points`, evaluate its rule (guarded). If the guard fails at the seed, skip here (the compiler will demote once per §13). Otherwise **write** `P`’s coordinates and (if `P` is still a variable in the model) use them as the starting guess.

**Stage C — Path adherence for `on_path`.**
For each variable point `P` with `on_path` hints:

* **Line/Ray/Segment**: orthogonally project the current `P` onto the carrier line; for **Ray** clamp `t≥0`, for **Segment** clamp `t∈[0,1]`. Feet (`foot`, `perpendicular_at`) and right-angle anchors register the same projections so they always take part in this sweep.
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

* **Length** `‖AB‖=L`: apply a **step-limited nudge** toward the target length (`Δ=α·(L-‖AB‖)` with `α≈0.5`) and clamp the absolute step by `κ·scale` so hints never blow up the seed. Displacement per point accrues against a running cap; once a point travels `>κ_total·scale` we stop honoring further hard-length nudges in that cycle.
* **Equal-length** groups: pick a reference edge with non-zero current length; for each other edge, nudge along its current direction toward the target using the same capped step policy (with a softer `α`). If one endpoint is “anchored” (protected or appears in multiple hints), prefer moving the other endpoint; otherwise split the displacement conservatively. Directions are chosen from:

  * the averaged directions of any intersection/`on_path` targets involving that edge’s line-like carriers (if any),
  * otherwise the current edge direction,
  * otherwise the reference edge direction.
* **Ratio** `AB:CD = p:q`: when three endpoints are reasonably positioned, nudge the fourth along its current direction to satisfy the ratio under the same capped-step policy (so chains of ratios cannot eject points).

**Stage F — Parallels / Perpendiculars / Tangency.**

* If edges `AB ∥ CD` and `CD` has a stable direction, rotate `AB`’s direction to match; nudge the non-anchored endpoint.
* For `AB ⟂ CD`, rotate `AB` to the perpendicular of `CD`.
* For known tangent line `X–Y` to circle center `O` at `T`, seed `T` as the foot of the orthogonal projection of `O` onto line `XY` (validated by `|‖OT‖−r| ≤ ε_tan`) and retain the branch tags (`choose=left|right|cw|ccw`) in the hint payload so later nudges keep the intended tangent.

**Stage G — Safety pass.**

# NEW
* Enforce **min-separation** and polygon **edge/area floors** at the seed by spreading any colliding points slightly along existing directions (without moving `G1,G2`). During the very first solve attempt we smooth these guard residuals (`σ ≈ 0.25·min_sep`) so inequality walls do not dominate before other constraints engage.

**Stage H — Projection warm-start (POCS).**

* Build typed projection operators for every supported constraint (`point_on` carriers, `foot`/`perpendicular_at`, `parallel_edges`, `angle`/`right-angle`, `ratio`, `equal_segments`, `concyclic`, circle incidences). Each operator touches only the points involved in that statement.
* Run 3 alternating-projection sweeps with a conservative blend (`α≈0.45`). Updates are capped to the affected points, then the model realigns gauges and re-evaluates deterministic derivations so derived points stay coherent.
* Diagnostics capture how many times each constraint kind fired, the cumulative displacement magnitudes, and any guard failures observed during the warm-start so downstream logging pinpoints stubborn facts.

> **Parallelogram seed (new).** If a `parallelogram A-B-C-D` has no numeric angle/length fixing its shape, seed a comfortable slant to avoid needle starts:
>
> ```python
> # A=(0,0), B=(base,0) already placed by the layout
> φ0 = math.radians(60)      # any 45°..75° is fine
> s  = 0.6 * base            # initial guess for adjacent side length
> C  = B + s * rot(φ0)       # rot around B
> D  = A + (C - B)           # enforce AB ∥ CD and BC ∥ AD
> ```
>
> The shape guards will keep a healthy height; if the problem demands a different angle, the solver slides there easily.

---

## S.8 Optional: parallel-line gap (for stacked parallels)

For two declared **parallel** carriers (\ell_1,\ell_2) with representative span (L) (e.g., longer intercepted segment),
[
r_{\text{gap}}=\max!\big(0,;\varepsilon_{\parallel},L - \operatorname{dist}(\ell_1,\ell_2)\big),\quad \varepsilon_{\parallel}\approx 0.04,
]
weighted by (w_{\text{shape}}). This preserves a readable gap when many constructions place nearly coincident parallels.

---

## S.9 Tests (add to your suite)

1. **Bare parallelogram**: with only `parallelogram A-B-C-D`, assert after solve:

   * height ( \ge \varepsilon_h \cdot \max(|AB|,|BC|));
   * ( |\sin\angle ABC| \ge s_{\min} ).
2. **Trapezoid** (no numeric height): height of bases respects **S.2**; legs remain non-parallel (your existing margin + angle cushion).
3. **Legitimate skinny**: add a case with `angle A-B-C [degrees=3]`; verify guards back off (final angle ≈ 3°; residuals small; guards contribute near-zero).

---

**Result:** These additions keep polygons readable (no “squished” shapes) without altering mathematically correct solutions. The guards are soft, scale-aware, and kick in only when needed; the new parallelogram seed speeds convergence and reduces wrong local minima.

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
* **Rule‑aware** rendering: respect `rules[...]` flags (`no_equations_on_sides`, `allow_auxiliary`, etc.).

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
      mark=at position 0.5 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);}}},
  tick2/.style={postaction=decorate, decoration={markings,
      mark=at position 0.47 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);},
      mark=at position 0.53 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);}}},
  tick3/.style={postaction=decorate, decoration={markings,
      mark=at position 0.44 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);},
      mark=at position 0.5  with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);},
      mark=at position 0.56 with {\draw (0,-\gsTick/2) -- (0,\gsTick/2);}}},
}
% optional layers
\pgfdeclarelayer{bg}\pgfdeclarelayer{fg}\pgfsetlayers{bg,main,fg}
```

**19.2 Painter’s order**

1. **bg**: light fills (rare)
2. **main**: carriers — polygon sides, declared segments, circles
3. **fg**: construction lines (aux), angle/segment marks, labels

**19.3 What to draw (and what **not** to draw)**

* **Polygons** (`triangle`, `quadrilateral`, `trapezoid`, …): draw *only* their carrier edges with `carrier` style (no diagonals unless declared or used for an explicit intersection to place a visible point).
* **Segments explicitly declared**: draw with `carrier`.
* **Lines/Rays**: draw as **auxiliary** (extend beyond endpoints; `aux` style). Rays get `-{Latex[length=2mm]}` heads.
* **Intersections**: if `X` is defined by `intersect(Path1, Path2)`, render both carriers as `aux` **only if** they are not already drawn elsewhere; otherwise, omit.
* **Circles**: draw with `circle` style. Do **not** auto-draw radii to all named points; draw a radius only if explicitly present, or needed for a right-angle mark at a tangent touchpoint. A bare `point P on circle center O` is enough to queue the circle.
* **Feet**: always draw the altitude segment `X–H` as `aux` (honors `rules[allow_auxiliary]`). If `H` misses its declared base, also inject the carrier `base[0]–H` so the mismatch is visible.
* **Medians / bisectors**: draw their carrier lines `aux` only if they appear in a `segment`/`ray`/`line` object or define a visible point; otherwise render only the **mark**.

**19.4 Point symbols & labels**

* Each named point: `\fill (P) circle (\gsDotR);` then `\node[ptlabel,<anchor>] at (P){$P$};`
* Default **anchor heuristic**: choose `above` if the point is on the lower half of the bbox, `below` if on the upper half; break ties using local edge directions to avoid overlap. Honor explicit `label point P [pos=...]`.

**19.5 Side labels vs. ticks**

* `[length=...]` metadata on `segment` objects is ignored by the TikZ renderer; no automatic side equations are emitted. Use `sidelabel A-B "..."` to draw any edge text. (The legacy `rules[no_equations_on_sides]` flag no longer changes rendering but is accepted for forward compatibility.)
* **Equal segments**: for each group, apply `tick1` / `tick2` / `tick3` to every segment in that group. If >3 groups, cycle ticks then add `densely dashed` to distinguish.
* Side labels are always emitted inside math mode. We normalise `sqrt(...)` products (e.g., `3*sqrt(2)`) **only** within that LaTeX context so radicals never appear as plain text.
* When emitting `sidelabel`, place the label via a `\path (A) -- (B) node[...] {...};` helper, mark it `sloped`, and offset it by roughly one dot radius (≈1.2 pt, scaled by scene size) along the requested `pos` direction. This keeps the text snug to the edge instead of floating far from the segment while still following the side orientation.

**19.6 Angles**

* **Numeric angle** (`angle A-B-C [degrees=θ]`): choose the ray order whose counter-clockwise measurement best matches `θ`, then draw one arc at `B` via `pic`:

  ```tex
  \path pic[draw, angle radius=\gsAngR, "$\num{θ}$"{scale=0.9}] {angle=A--B--C};
  ```

  Always use `^\circ`. When no `degrees=` metadata exists, pick the ordering whose counter-clockwise sweep is < `180^\circ` (or closest to it if the configuration is straight/reflex) so TikZ draws the minor arc.
  Any symbolic `degrees=` metadata first runs through the same math normalisation (`sqrt(...)` → `\sqrt{...}`) before the degree token is appended, ensuring radicals and products render correctly inside the angle label.
* **Right angle** (`right-angle A-B-C`): always draw the square symbol at `B` via the TikZ `right angle` pic, never an arc or `$90^\circ$` label:

  ```tex
  \path pic[draw, angle radius=\gsAngR] {right angle=A--B--C};
  ```
* **Equal angles** (`equal-angles (A-B-C, ... ; D-E-F, ...)`): for each *group*, draw **n arcs** (n=group index: 1=single, 2=double, 3=triple) at the relevant vertices. Radii are `\gsAngR`, `\gsAngR+\gsAngSep`, `\gsAngR+2\gsAngSep` for the multiple arcs, no labels.

**19.7 Medians, bisectors, altitudes (marks)**

* **Median** `median from V to A-B midpoint M`: draw `V–M` if `segment`/`line` is present; otherwise omit the line. Optionally mark `M` with a small notch on `AB`.
* **Angle bisector** at `B`: draw the **double‑arc** mark at `B` (no line) unless explicitly requested as a `line`/`segment`.
* **Altitude** / **foot** `foot H from X to A-B`: draw a **right‑angle square** at `H` and always draw `X–H` as `aux` (still respecting `rules[allow_auxiliary]`). When `H` is off the supporting line `A-B`, also promote `A–H` (more generally, the first base endpoint joined with `H`) to a `carrier` to flag the inconsistency. If `H` lies on the supporting line but outside the segment `A-B`, add an `aux` segment with `densely dotted` style from the nearer endpoint of `A-B` to `H` to show the extension.

**19.8 Tangency**

* `tangent at T to circle center O`: draw tangent **line** (aux) through `T`, draw the **radius** `O–T` (carrier or aux), and a **right‑angle square** at `T`.
* `line X-Y tangent to circle center O at T`: same geometry, but render the segment `X–Y` as a **carrier** (solid) edge and extend it across the scene with a dotted **aux** line. If there’s no radius elsewhere, draw `O–T` thin to support the square.

**19.9 Targets & highlights**

*The TikZ renderer currently ignores `target ...` statements; no foreground question-mark overlays or emphasis marks are emitted.*

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
```

Populate this by walking the **desugared** program + options:

* **Carriers**: all declared `segment` and polygon sides.
* **Aux**: `line`/`ray` used for *explicitly drawn* carriers or to make an intersection visible; also bisector/median/altitude carriers *only* if they were explicitly declared.
* **Circles**: only as declared.
* **Ticks**: expand `equal-segments` into grouped edges (index groups 1..3; wrap if >3).
* **Equal angles**: collect into groups.
* **Right angles**: all `right-angle` statements; also synthesize from tangency if present.
* **Angle arcs**: from `angle A-B-C [degrees=..]`.
* **Labels**: from `label point` and explicit `sidelabel`. Ignore any `[length=...]` metadata carried by `segment` statements when building the plan.
*Targets are collected in the IR but presently skipped by the TikZ renderer.*

**19.11.2 Suppress redundancy**

* **Do not** add any `O–P` radius unless: (a) explicitly declared as a `segment`/`line`, or (b) required for a tangency right‑angle square.
* **Do not** draw connectors between arbitrary point pairs that aren’t in `carriers` or `aux_lines`.

**19.11.3 Layered emission**

* Emit in painter’s order (19.2).
* Use the standardized styles from 19.1 exactly once per document.
* **Right-angle squares**: use TikZ `pic` right‑angle symbol.
* **Equal‑angles**: for group *g*, draw `g` arcs at radius `\gsAngR + (k-1)\gsAngSep` (`k=1..g`) with no labels.
* **Ticks**: apply `tick{g}` style to the **segment** draw command; if the segment is not otherwise drawn (e.g., it’s only an abstract equality), draw the segment **thin dashed** only for the tick mark, or place two small ticks floating near endpoints (simpler: lightly draw the segment).
* **Angle labels**: always `$\,^\circ$` (LaTeX degree).
* *(Targets currently produce no additional drawing commands.)*

**19.11.4 Rules mapping**

* `rules[no_equations_on_sides]` → drop numeric edge labels unless created by explicit `sidelabel`.
* `rules[allow_auxiliary]` → if `false`, don’t draw bisector/median/altitude carriers unless explicitly declared as `segment/line/ray`; draw only marks.

**19.11.5 Minimal preamble**
Replace the heavy preamble with the minimal block in 19.1; return a `standalone` document without `varwidth/adjustbox/pgfplots`.


## 19.12 Orientation — Post‑solve “Nice” Figure Alignment (rigid only)

**Goal.** After the solver converges, but **before** TikZ generation, apply a *rigid* transform (rotation, optionally a mirror/reflection; plus a neutral translation around a pivot) to all points so that common figures appear with a visually “nice” orientation:

* **Trapezoid**: bases horizontal; the **smaller base is on top**.
* **Isosceles triangle**: the **base** is horizontal.

This pass does **not** change distances or angles; it only composes an isometry with the solved coordinates and then hands the re‑oriented coordinates to the TikZ renderer. No residuals are recomputed; DDC checks (§16) use the original solution.

> **Integrity guarantee:** The orientation transform is an orthogonal map `Q` (`QᵀQ=I`, `det(Q)=±1`) and a translation `t`. For every point `P`, we emit `P' = Q·(P − p₀) + p₀ + t` with a pivot `p₀` (see below). Distances, angles, incidences, parallelism, and perpendicularity are preserved.

---

### 19.12.1 Scope & when to apply

We act on **one** “main” figure if present; otherwise orientation is a no‑op.

**Priority order (pick the first that exists):**

1. A declared **trapezoid** (`trapezoid A-B-C-D`), preferring the one with an explicit `[bases=...]` option.
2. A declared **triangle** that is **isosceles** (explicit `isosceles=atA|atB|atC` or implied by `equal-segments` on two sides).

If multiple candidates tie within a class, choose the one with **largest area** (computed from solved coordinates); on ties, the **earliest** in source order.

> No automatic detection for non‑declared shapes. If neither (1) nor (2) exists, skip orientation.

---

### 19.12.2 Numerical tolerances

Let `scene_scale = diag(bbox(points))` from the solved coordinates.

```
ε_len   = 1e-9 * scene_scale     # length equality / base choice tolerance
ε_ang   = 1e-9                    # radian tolerance for horizontality checks
ε_para  = 1e-12 * scene_scale     # parallel test (cross magnitude threshold)
```

These tolerances are only used to break ties and avoid flip‑flop on nearly equal cases.

---

### 19.12.3 Trapezoid policy (bases horizontal; smaller base on top)

**Inputs.** A quadrilateral `A–B–C–D` that the desugarer marked as a trapezoid. If `trapezoid [...] [bases=X-Y]` is present, `XY` is the **declared base**; otherwise, identify the two parallel sides.

**Steps.**

1. **Identify the two bases**
   *If `[bases=X-Y]` present*: let `B₁ = (X,Y)`, `B₂ =` the opposite side parallel to `B₁`.
   *Else*: among edges `{AB, BC, CD, DA}`, pick the **two** whose direction vectors are parallel (`|cross(u,v)| ≤ ε_para·(‖u‖+‖v‖)`).

2. **Choose orientation direction (for horizontality)**
   Let `v = direction(B₁)` if `[bases=...]` was provided; otherwise let `v` be the **average** of the two base directions (normalized, same sign).

3. **Compute rotation to horizontal**
   Let `θ = -atan2(v_y, v_x)` and `R(θ)` the rotation matrix. This makes the bases *horizontal*.

4. **Apply rotation around a pivot**
   Use the **figure centroid** as pivot: `p₀ = mean({A,B,C,D})`. Set temporary coords `P* = R(θ)·(P − p₀) + p₀` for all points `P`.

5. **Decide which base is smaller and whether it is on top**
   Compute `L₁ = ‖B₁*‖`, `L₂ = ‖B₂*‖`.
   Let `B_small = argmin{L₁,L₂}`, `B_large = the other`.
   Let `y_small = y(midpoint(B_small*))`, `y_large = y(midpoint(B_large*))`.

6. **Flip if needed (mirror across the horizontal axis)**
   If `y_small ≤ y_large + ε_len`: apply a vertical mirror (flip across the x‑axis) about the same pivot:

   ```
   F = diag(1, -1)
   P' = F · (P* - p₀) + p₀
   ```

   else set `P' = P*`. Now the smaller base sits **above** the larger base.

7. **(Optional) Neutral recenter**
   No extra translation is required for TikZ. Renderers may shift by a small margin later when fitting the bbox (§19.10).

**Outcome invariants (assertions):**

* Both base directions are horizontal within `ε_ang`.
* `y(midpoint(B_small')) > y(midpoint(B_large')) − ε_len`.
* `‖P'_i − P'_j‖ = ‖P_i − P_j‖` for all named points.

---

### 19.12.4 Isosceles triangle policy (base horizontal)

**Inputs.** `triangle A-B-C` with an explicit `isosceles=atV` (V ∈ {A,B,C}) **or** an `equal-segments` group pairing two triangle sides.

**Steps.**

1. **Identify the base side**
   If `isosceles=atA`, base is `B–C` (opposite the equal legs `AB` and `AC`). Similarly for `atB`, `atC`.
   If inferred from `equal-segments`, pick the unique side **not** in the equality pair as the base.

2. **Rotate to horizontal**
   Let `v = direction(base)`, `θ = -atan2(v_y, v_x)`, `p₀ = mean({A,B,C})`.
   Apply `R(θ)` about `p₀`: `P' = R(θ)·(P − p₀) + p₀` for all points.

3. **No forced flip**
   We **do not** mandate “apex up”/“down”. The requirement is *only* “base horizontal”. (A future option could add `apex_up=true|false` if needed.)

**Outcome invariants:**

* Base direction horizontal within `ε_ang`.
* Isometry preserved.

---

### 19.12.5 Edge cases & tie‑breaking

* **Ambiguous bases (almost equal lengths).** If `|L₁−L₂| ≤ ε_len` in a trapezoid, treat the base named by `[bases=...]` as **authoritative**. If absent, prefer the base whose **midpoint** currently has the **smaller** y after rotation, then flip to enforce “smaller on top”. This yields stability across runs.
* **Multiple trapezoids/triangles.** The **largest‑area** candidate wins; on equal areas within `ε_len`, pick the one that appears **earliest** in the source program.
* **Already horizontal / already “small‑on‑top”.** The pass becomes the identity (no rotation; no flip).
* **Degenerate/near‑parallel failures.** If a trapezoid’s “bases” cannot be identified (no pair of parallel sides within `ε_para`), skip orientation with a warning note in diagnostics.

---

### 19.12.6 API & integration points

```python
@dataclass
class OrientationResult:
    Q: np.ndarray        # 2x2 orthogonal (rotation or rotation+reflection)
    t: np.ndarray        # 2-vector translation (usually 0)
    kind: Literal["identity","rotation","rotation+reflection"]
    pivot: Tuple[float,float]
    figure: Optional[Dict[str, Any]]   # {"kind":"trapezoid"/"triangle", "ids":[...], "notes":[...]}
    notes: List[str]
```

```python
def orient_for_rendering(program: Program,
                         solution: Solution) -> Tuple[Dict[str, Tuple[float,float]], OrientationResult]:
    """
    Returns: (coords_oriented, diagnostics) where coords_oriented maps every point id to oriented coordinates.
    This function never mutates 'solution'; DDC and solver stats continue to reference original coordinates.
    """
```

**Pipeline insertion (extends §17.2 & §19.11):**

```python
# after solving & (optionally) after DDC-Check
coords = solution.point_coords
coords, orient_diag = orient_for_rendering(program, solution)

# feed the oriented coords to RenderPlan/TikZ
render_plan = build_render_plan(program, coords)
tikz = emit_tikz(render_plan)
```

Add a renderer option `orient="auto"|"off"|"trapezoid"|"triangle"` (default `"auto"`). If `"off"`, skip this pass. If a specific kind is forced but missing, skip with a diagnostic.

---

### 19.12.7 Determinism requirements

* The chosen figure and transform must be **deterministic** given the program and solved coordinates. Choices rely on explicit metadata first; otherwise on stable geometric scores (area, y‑ordering), then source order.
* The transform is computed **once** per render and applied to **all** named points (including circle centers); radii are unchanged.

---

### 19.12.8 Pseudocode

```python
def orient_for_rendering(program, solution):
    pts = solution.point_coords.copy()

    # --- choose main figure
    traps = collect_declared_trapezoids(program)
    tris  = collect_isosceles_triangles(program)  # explicit isosceles or inferred from equal-segments
    main  = pick_main_figure(traps, tris, pts)    # priority: trapezoid > isosceles triangle

    if main is None:
        return pts, OrientationResult(Q=np.eye(2), t=np.zeros(2), kind="identity", pivot=(0,0), figure=None, notes=["no-op"])

    if main.kind == "trapezoid":
        B1, B2 = identify_bases(main, program, pts)              # respects [bases=...] if present
        vdir   = pick_base_direction(B1, B2, prefer_declared=True)
        p0     = centroid_of(main.vertex_ids, pts)
        θ      = -math.atan2(vdir[1], vdir[0])
        R      = rot2(θ)
        ptsR   = {k: R @ (np.array(p) - p0) + p0 for k,p in pts.items()}

        L1, L2 = seglen(ptsR, B1), seglen(ptsR, B2)
        small, large = (B1, B2) if L1 <= L2 + ε_len else (B2, B1)
        y_small = midpoint_y(ptsR, small)
        y_large = midpoint_y(ptsR, large)

        if y_small <= y_large + ε_len:
            F  = np.diag([1.0, -1.0])
            ptsF = {k: F @ (np.array(p) - p0) + p0 for k,p in ptsR.items()}
            Q = F @ R; kind = "rotation+reflection"; ptsO = ptsF
        else:
            Q = R; kind = "rotation"; ptsO = ptsR

        return ptsO, OrientationResult(Q=Q, t=np.zeros(2), kind=kind, pivot=tuple(p0), figure={"kind":"trapezoid","ids":main.vertex_ids}, notes=[])

    if main.kind == "triangle" and main.is_isosceles:
        base = find_isosceles_base(main, program, pts)
        vdir = direction(pts, base)
        p0   = centroid_of(main.vertex_ids, pts)
        θ    = -math.atan2(vdir[1], vdir[0])
        R    = rot2(θ)
        ptsO = {k: R @ (np.array(p) - p0) + p0 for k,p in pts.items()}
        return ptsO, OrientationResult(Q=R, t=np.zeros(2), kind="rotation", pivot=tuple(p0), figure={"kind":"triangle","ids":main.vertex_ids}, notes=[])
```

---

### 19.12.9 Tests (must‑pass)

1. **Trapezoid — small base on top**

   ```
   trapezoid A-B-C-D [bases=A-D]
   ```

   After orientation: `AD` and `BC` horizontal; `mid(BC).y < mid(AD).y` (or vice versa depending on naming) such that **smaller base has larger y**.

2. **Trapezoid — no [bases], detect parallels**
   Construct with bases unspecified; assert detection works and orientation matches policy.

3. **Isosceles triangle — base horizontal**

   ```
   triangle A-B-C [isosceles=atA]
   ```

   After orientation: `BC` horizontal within `ε_ang`. No reflection forced.

4. **No candidates**
   Scene without trapezoid or isosceles triangle → identity transform; TikZ unchanged.

5. **Determinism under near‑ties**
   Two trapezoids same area within `ε_len` → earliest in source order is chosen; results stable across runs.

6. **Distance & angle preservation**
   For random scenes with orientation applied, assert `‖P'−Q'‖ == ‖P−Q‖` and angle at triples unchanged within `1e-12`.

---

### 19.12.10 Non‑goals (v1)

* No translation normalization beyond pivoting; final framing is still handled by the TikZ bbox fit (§19.10).
* No special handling for non‑isosceles triangles or other polygons.
* No solver feedback; this is purely a rendering convenience.

---

## 19.13 Standard Notation — Equal Segments, Equal Angles & Derived Construction Marks

> This section **consolidates and formalizes** how the TikZ generator renders:
>
> * equal **segments** (single/double/... ticks),
> * equal **angles** (1/2/… arcs),
> * and what **altitudes**, **angle bisectors**, and **medians** *visually* generate (right‑angle squares, equal‑angle arcs, equal‑length ticks).
>   It complements §§19.5–19.7 and §20 by specifying **deterministic grouping, tie‑breaking, and emission** rules.

### 19.13.0 Scope & invariants

* A **segment** that belongs to an equality group **must** carry the same number of **ticks** as every other segment in that group (single/double/triple…).
* An **angle** that belongs to an equality group **must** be drawn with the same number of **arcs** as every other angle in that group.
* **Altitudes, bisectors, medians** always add their **canonical marks** even if the author did not redundantly add `equal-angles` / `equal-segments` statements:

  * **Altitude**: right‑angle **square** at the **foot**.
  * **Angle bisector** at vertex `V`: **double‑arc** at `V` (no numeric label implied).
  * **Median** to side `A–B` with midpoint `M`: apply equal‑length **ticks** to `A–M` and `M–B`.

Marks are emitted on the **visible stroke**; no auxiliary carriers are auto‑drawn unless explicitly declared (see §19.3, §19.11.2).

---

### 19.13.1 Inputs that create equality/marking intent

**Explicit equalities**

* `equal-segments (EdgeList ; EdgeList)`
* `equal-angles   (AngleList ; AngleList)`

**Implicit (derived) equalities/marks from construction objects**

* `right-angle A-B-C` → right‑angle square at `B`.
* `angle-bisector U-V-W` (path) → **double‑arc** at `V` (equal‑angle mark).
* `median from P to A-B midpoint M` → **equal‑length** ticks on `A–M` and `M–B`.
* `midpoint M of A-B` → **equal‑length** ticks on `A–M` and `M–B`.
* `point M on segment A-B [mark=midpoint]` → same midpoint tick pair on `A–M` and `M–B`.
* `triangle U-V-W [isosceles=atV]` → **equal‑length** ticks on the two **legs incident to `V`**.

> These implicit visuals are **marks only**. They do not add metric constraints (solver behavior is defined in §6); they exist to make standard constructions legible.

---

### 19.13.2 Canonicalization & grouping (deterministic)

The renderer builds **disjoint groups** for both segments and angles before emission. Determinism ensures stable tick/arc counts across runs.

#### A) Equal‑**segments** groups

1. **Normalize edges** as undirected, lexicographically ordered pairs: `edge(A,B) = (min(A,B), max(A,B))`.
2. Build a **union‑find (disjoint‑set)** structure:

   * For each `equal-segments (E... ; F...)`, **union** all edges in `E∪F`.
   * For each `midpoint M of A-B`, `point M on segment A-B [mark=midpoint]`, or `median ... midpoint M`, **union** `A–M` with `M–B` into an **implicit** group (unless either edge already belongs to an explicit group; see precedence below).
   * For each `triangle U-V-W [isosceles=atV]`, **union** the two **legs adjacent to `V`** into an implicit group (suppressed if either leg is already explicit).
3. Compute the **connected components**; each becomes one **segment equality group**.
4. **Ordering (stable):**

   * First, all groups that contain at least one edge mentioned in **explicit** `equal-segments` statements, ordered by the **first source occurrence** of any of their edges.
   * Then, purely **implicit** groups (midpoints/medians), ordered by the **first occurrence** of their defining statement.
5. Assign **group_index = 1,2,3,…** in that order.

#### B) Equal‑**angles** groups

1. **Normalize angles** as unordered wedges at the vertex: `∠(A,B,C)` is equal to `∠(C,B,A)` for **marking** purposes (we always draw the **minor** wedge; see §19.6).
2. Build a **union‑find** over normalized wedges:

   * For each `equal-angles (L ; R)`, union all angles in `L∪R`.
   * For each `angle-bisector U-V-W`, create an **implicit** 2‑member group at `V` (the two halves around the bisector). Implemented as a single **double‑arc** at `V` (no need to enumerate half‑wedges).
3. Components become **angle equality groups**. Order and **group_index** assignment mirror the segment rules: **explicit** groups first (by first occurrence), then **implicit bisector** marks.

> **Conflict handling (angles)**: If the same wedge appears in **two explicit** groups, mark a renderer diagnostic (non‑fatal) and keep the **earliest** group’s index for drawing.

---

### 19.13.3 Emission rules (ticks, arcs, right‑angle squares)

#### A) **Segments** — ticks

* Per §19.1 styles `tick1`, `tick2`, `tick3` exist. For **group_index g ≥ 1**:

  * Use `tick{ ((g-1) mod 3) + 1 }` as the **tick count**.
  * If **g > 3**, add the stylistic qualifier `densely dashed` **only to the tick carrier stroke** (see emission note below) to visually distinguish groups that cycle the same tick count.
* Apply the tick style to the **drawn segment**:

  * If the segment is already drawn as a **carrier** (polygon side or declared `segment`), attach the tick style to that path.
  * If it is **not otherwise drawn**, render a thin **dashed stub** of the segment solely to host the ticks (do **not** introduce new visible construction lines).
* **Zero/near‑zero length edges** (`‖AB‖ ≤ ε_len`, §19.12.2): skip ticks.

> **Tick orientation.** Each `tick*` style draws **perpendicular strokes** (local y-axis in the mark frame) so the dash is visible across the carrier instead of hiding along it.

**Precedence when a segment is both explicit and implicit (median/midpoint):**
Explicit `equal-segments` **wins** (its group_index/ticks are used). Implicit marks are suppressed to avoid double‑marking.

**Emission note (g>3):**
To avoid changing the visual of a primary carrier, emit a **second, overlay path** coincident with the segment using `draw opacity=0` + the tick decoration and `densely dashed`. This decorates ticks only, leaving the carrier’s solid stroke intact.

#### B) **Angles** — arcs

* For **group_index g**, draw **g concentric arcs** (no text). Radii follow §20.2:

  ```
  r_k = r0 + (k-1)*\gsAngSep,  k = 1..g,
  r0 = clamp(0.12 * min(|BA|, |BC|), 7pt, 14pt)
  ```
* Always draw the **minor** wedge at the vertex (choose the ray ordering whose CCW measure is < `180^\circ`; fallbacks per §19.6).
* If a numeric `angle A-B-C [degrees=θ|label="..."]` is **also** present for the same wedge, place its **numeric arc/label at radius `r0 + g*\gsAngSep`** (equal‑angle arcs stay **inside**).
* **Tiny wedges** (`∠ < 6°`): skip equal‑angle arcs to avoid clutter (numeric labels may move to the external side with a leader per §20.2.1).

#### C) **Right angles** — squares

* `right-angle A-B-C` and **altitude** footprints always render:

  ```
  \path pic[draw, angle radius=\gsAngR] {right angle=A--B--C};
  ```
* **Never** draw `$90^\circ$` text for right angles (see §19.6).

---

### 19.13.4 Interactions, precedence & diagnostics

* **Explicit vs implicit**
  Explicit `equal-segments` / `equal-angles` **override** implicit marks at the same geometry (median/bisector). Implicit marks are suppressed when they would duplicate an explicit equality mark.
* **Multiple explicit memberships (conflict)**
  If an edge (or wedge) is pulled into **two** explicit groups, the renderer keeps the **earliest** group (source order) and logs a **diagnostic note** in the `RenderPlan` (non‑fatal).
* **More than 3 groups** (segments)
  Tick count cycles every 3; for `g>3` an overlay with `densely dashed` is added to differentiate. Angle groups **do not** need this fallback (arcs stack naturally).
* **No carriers**
  Ticks/arcs are marks; they **do not** force drawing of otherwise absent construction lines (except the dashed **stub** for a non‑drawn equal segment as host for ticks).
* **Numeric angles with equal‑angles**
  Equal‑angle arcs draw **inside**; the numeric arc and label are placed **outside** the stack (radius `r0 + g*\gsAngSep`), with collision‑avoidance per §20.2.

---

### 19.13.5 Data flow additions (RenderPlan)

Extend the `RenderPlan` described in §19.11.1 with one optional field and clarifications:

```python
@dataclass
class RenderPlan:
    # (existing fields unchanged)
    ticks: List[Tuple[str,str,int]]              # (A,B, group_index 1..)
    equal_angle_groups: List[List[Tuple[str,str,str]]]  # groups of (A,B,C)
    right_angles: List[Tuple[str,str,str]]       # (A,B,C), includes altitudes
    notes: List[str]                             # renderer diagnostics (conflicts, suppressed implicit, g>3 overlays)
```

* Populate `ticks` from **explicit** groups first, then add **implicit** pairs `(A,M)` and `(M,B)` from **midpoints/medians/point-on midpoint marks**, plus the **isosceles legs** `(V,U)` and `(V,W)` that do **not** collide with explicit groups.
* Populate `equal_angle_groups` from explicit `equal-angles`; add **implicit** “bisector doubles” as single‑vertex marks (no need to enumerate two half‑wedges).
* Populate `right_angles` from explicit `right-angle` and from **altitude** footprints.

---

### 19.13.6 Pseudocode (grouping & emission)

```python
# --- segments (union-find)
edges = normalize_undirected_edges(program)
UFs = UnionFind(edges)

for stmt in equal_segments_statements:
    for e in stmt.edges_all_lists:
        UFs.union(e0, e)

implicit_seg_pairs = []   # from midpoint/median
for mid in midpoints_and_medians:
    e1, e2 = (A,M), (M,B) = normalized_pairs(mid)
    implicit_seg_pairs.append((e1,e2))

# precedence: mark which edges are already explicit
explicit_edges = set(flatten(UFs.components()))
# merge implicit only if neither edge is explicit
for (e1, e2) in implicit_seg_pairs:
    if e1 not in explicit_edges and e2 not in explicit_edges:
        UFs.union(e1, e2)

groups = order_components(UFs, explicit_first=True)  # by first source occurrence
for g_idx, comp in enumerate(groups, start=1):
    for (A,B) in comp:
        render_plan.ticks.append((A, B, g_idx))

# --- angles (union-find)
wedges = normalize_wedges(program)  # treat (A,B,C) == (C,B,A)
UFa = UnionFind(wedges)

for stmt in equal_angles_statements:
    for w in stmt.wedges_all_lists:
        UFa.union(w0, w)

# implicit from bisectors: add a "double arc at V" mark
for bis in bisector_statements:
    render_plan.equal_angle_groups.append([("∗", bis.V, "∗")])  # marker: drawn as double-arc at V

# order and assign group indices (explicit first)
angle_groups = order_components(UFa, explicit_first=True)
# emission later uses group index to choose arc count per-vertex

# --- altitudes: right-angle squares
for alt in altitude_like_statements:
    A,B,C = alt.as_right_angle_triple()
    render_plan.right_angles.append((A,B,C))
```

*(Actual emission follows §§19.11.3, 20.2 with radii, collision avoidance, and the overlay trick for `g>3` segment groups.)*

---

### 19.13.7 Examples (authoring → marks)

**A. Explicit equal segments (2 groups) + a midpoint (implicit)**

```
equal-segments (A-B, C-D ; E-F)     # group 1 → single tick on AB, CD, EF
equal-segments (G-H ; I-J, K-L)     # group 2 → double tick on GH, IJ, KL
midpoint M of P-Q                    # implicit → single tick on PM and MQ (group 3)
```

**B. Equal angles + numeric angle on one member**

```
equal-angles (A-B-C, D-E-F ; G-H-I) # group 1 → single-arc at B, E, H
angle A-B-C [degrees=30]            # numeric arc/label drawn OUTSIDE the group-1 arcs at B
```

**C. Bisector + Altitude + Median (implicit marks only)**

```
angle-bisector U-V-W                 # double-arc at V
foot H from X to A-B                 # right-angle square at H
median from C to A-B midpoint M      # ticks on AM and MB (equal segments implied)
```

---

### 19.13.8 Acceptance checks (renderer)

* **Segments**: Every edge in the same equality group uses the **same** tick count; explicit groups override implicit (median/midpoint).
* **Angles**: Every wedge in a group shows the **same** number of arcs; numeric labels sit **outside** equal‑arc stacks.
* **Constructions**: Altitude always shows a square; bisector a double‑arc; median/midpoint two equal‑length ticks.
* **Determinism**: Tick/arc counts are **stable** under re‑ordering of unrelated statements; conflicts are noted in `RenderPlan.notes`.
* **Clarity**: For `g>3` segment groups, an overlay with `densely dashed` is used to distinguish cyclic tick counts without altering the base carrier stroke.

---

# 20) Rendering — Points & Angle Labels (policy + algorithms)

This appendix extends §19 (Rendering Contract) with **deterministic, collision-aware placement** for point labels and angle labels, and removes current redundancies.

## 20.0 Ground rules

* **Only render what’s declared.** Numeric angles are drawn **only** when the program has `angle A-B-C [degrees=…]` or `target angle …`. Do **not** infer the third triangle angle.
* **No duplicates.** A segment appears at most once (carrier or aux). Ticks/marks are drawn **once**, on the visible stroke.
* **Right angle**: draw a square; never print `90^\circ`.
* **Degree symbol**: always use `^\circ` to avoid Unicode.

---

## 20.1 Point labels — collision-aware placement

**Inputs per point `P`:**

* `IncLines(P)`: incident **drawn** line-like carriers (segment/line/ray) with directions `u_i`.
* `IncCircles(P)`: incident **drawn** circles with centers `O_j` (outward normals `n_j = normalize(P−O_j)`).
* `PlacedLabels`: label boxes already placed (see below).

**Constants (scene-aware):**

```
d0 = max(1.8*gsDotR, 0.012 * scene_bbox_diag)   # base offset
label_box ≈ width: 0.52em*len(text), height: 0.9em (footnotesize)
anchors = [above, above right, right, below right, below, below left, left, above left]
```

**Scoring function for a candidate anchor `a` (lower is better):**

```
score(a) =
  12 * OverlapsEdges(a)        # label box intersects any drawn segment/ray
+ 10 * OverlapsCircles(a)      # label box cuts a drawn circle
+  8 * OverlapsLabels(a)       # box intersects a placed label
+  4 * InsideCircle(a)         # for boundary points, label points inward
+  2 * Crowded(a)              # too close to any nearby point
-  1 * AlignmentBonus(a)       # prefer above/below for ~horizontal, left/right for ~vertical,
                               # and outward wrt incident circles
```

**Algorithm P1 — PlacePointLabel(P):**

1. For each `a ∈ anchors`, compute offset `off(a)` of length `d0`, estimate label box, compute `score(a)`.
2. Pick `a*` with minimal score (break ties by anchor order).
3. **Escalation**: if `score(a*)` indicates an overlap, multiply the offset by `1.4` and recompute, up to **3** rounds.
4. If still overlapping, draw a tiny **leader** (aux line) of length `0.8*d0` from `P` in direction `off(a*)` and place the label at its tip.
5. Emit `\node[ptlabel,<a*>] at (P) {$P$};`.

**Priority order (stable greedy):**

1. circle centers; 2) polygon vertices; 3) special points (feet/tangency/intersections); 4) others.
   Points unused anywhere and without explicit `label point` **may be left unlabeled**.

---

## 20.2 Angle visuals — arcs, labels, and equal-angle stacks

### 20.2.1 Numeric angles (`angle A-B-C [degrees=θ|label="..."]`)

* Draw **one** arc at `B` via `pic`:

  ```
  r0 = clamp(0.12 * min(|BA|, |BC|), 7pt, 14pt)
  \path pic[draw, angle radius=r0] {angle=A--B--C};
  ```
* Label text:

  ```
  text = label if provided else f"{θ}^\circ"
  ```
* **Label placement**: center the label on the **angle bisector** at radius `r0 + 0.6em`.
  If the label overlaps a stroke/label, **increase radius** by `+gsAngSep` and retry (≤3 times).
  If the wedge is very narrow (`∠ABC < 10°`) and still collides, place the text just beyond the **external** bisector with a short leader.

### 20.2.2 Right angles (`right-angle A-B-C`)

* ALWAYS:

  ```
  \path pic[draw, angle radius=\gsAngR] {right angle=A--B--C};
  ```

  No numeric text.

### 20.2.3 Equal angles (`equal-angles (… ; …)`)

* Partition angles into **groups**. For group index `g = 1,2,3,…` draw **g arcs** (no text) at radii:

  ```
  r_k = r0 + (k-1)*gsAngSep,     k=1..g        (use same r0 formula as 20.2.1)
  ```
* If a numeric angle is **also** declared for the same wedge, draw the numeric arc at `r0 + g*gsAngSep` and put its label there; equal-angle arcs stay inside it.

### 20.2.4 Avoiding clutter

* Arcs/labels MUST NOT cross the polygon/carrier strokes at the vertex. If crossing occurs, **increase radius**; if impossible (tiny wedge), fall back to an **external** label with leader.
* Never draw multiple numeric labels at the same vertex; prefer the one explicitly present in source order.
* Do not place equal-angle arcs when the wedge is `< 6°` (skip quietly).

---

## 20.3 Ticks & side labels (consistency)

* **Equal segments**: apply `tick1/tick2/tick3` to every member of a group **on the visible stroke**. If a member segment isn’t drawn, render a short dashed stub centered on that segment for the ticks.
* **No auto numerics** on sides; only draw side text from `sidelabel` or an explicit segment `[label="…"]`.
* **Angle support**: `angle`, `right-angle`, `target-angle`, and `equal-angles` statements must have carriers for both rays. The consistency pass emits `segment` hotfixes for missing sides so downstream stages can rely on those edges existing.
* **Parallel carriers**: `parallel-edges (A-B ; C-D)` requires both edges to exist as segments. Missing carriers trigger consistency hotfixes so that downstream stages always have drawable strokes for each edge.

---

## 20.4 Emission order (layers)

1. **main**: carriers (segments/polygons), circles.
2. **fg**: aux lines (only when needed per §19/§20), **then** angle arcs/squares/ticks, **then** labels (angles first, then points).

---

## 20.5 Generator hooks (minimal)

* Extend the `RenderPlan` with:

  ```python
  AngleMark = Literal["numeric","right","equal"]
  render_plan.angles: List[Dict]  # {kind, A,B,C, degrees?, label?, group?}
  ```
* Before emitting labels, compute `scene_bbox_diag`, then:

  * place **angle labels/arcs** with the algorithm in §20.2,
  * place **point labels** with Algorithm P1.
* Ensure **no segment is re-drawn** in `fg` if already emitted in `main`.

---

# T) Globalized Solver + Loss‑Mode (Torch‑first, drop‑in)

> **Goal.** Make convergence **robust** (less sensitive to seeds and hinge kinks) by:
>
> 1. solving a **smoothed** version of the problem first (homotopy over σ),
> 2. using **autodiff (PyTorch)** to get exact gradients/Jacobians, and
> 3. running a **stage pipeline**: *Adam* → *L‑BFGS* → *LM/TRF*, with **deterministic multistart** reseeds.
>    The result still satisfies your solver contract (§13), works with DDC (§16), seeding (§18), and shape guards (§S).

> **Implementation note:** The pipeline now executes genuine Adam and L‑BFGS stages. Adam leverages `torch.optim.Adam`
> with central finite-difference gradients (step size `adam_fd_eps`), while the L‑BFGS stage calls `scipy.optimize.minimize`
> (`method="L-BFGS-B"`) on the same smoothed scalar loss. Torch autodiff wiring for residuals remains future work; when
> `autodiff="off"` the schedule falls back to the SciPy least-squares stages.

---

## T.1 Public API (non‑breaking additions)

```python
from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class SolverOptions:
    backend: Literal["scipy-trf","scipy-lm"] = "scipy-trf"
    xtol: float = 1e-12; ftol: float = 1e-12; gtol: float = 1e-12
    max_nfev: int = 5000

@dataclass
class LossModeOptions:
    enabled: bool = True
    autodiff: Literal["torch","off"] = "torch"   # Torch preferred
    # Homotopy (smoothing) schedule — multiplied by scene scale
    sigmas: Optional[List[float]] = None         # default [0.20, 0.10, 0.05, 0.02, 0.00]
    robust_losses: Optional[List[str]] = None    # default ["soft_l1","huber","linear","linear","linear"]
    # Stage per sigma (same length as sigmas)
    stages: Optional[List[str]] = None           # default ["adam","adam","lbfgs","lbfgs","lm"]
    # Deterministic reseeds per sigma
    restarts_per_sigma: Optional[List[int]] = None   # default [1,1,1,2,2]
    multistart_cap: int = 8
    # Adam
    adam_lr: float = 0.05
    adam_steps: int = 800
    adam_clip: float = 10.0
    adam_fd_eps: float = 1e-6
    # LBFGS
    lbfgs_maxiter: int = 500
    lbfgs_tol: float = 1e-9
    # Final SciPy pass
    lm_trf_max_nfev: int = 5000
    # Gates
    early_stop_factor: float = 1e-6   # relative improvement threshold per sigma stage
```

Entry point (extends §13):

```python
def solve*(model_or_program, *, solver_opts: Optional[SolverOptions]=None,
           loss_opts: Optional[LossModeOptions]=None, plan=None) -> Solution
```

If `loss_opts.enabled=False` or Torch is not available, fall back to the **SciPy‑only** path described in **T.8** (same σ schedule + restarts, no autodiff).

---

## T.2 Smoothing & robust aggregation (single scalar objective)

We minimize a scalar **loss** (L_\sigma(x)) that matches your original least‑squares when (\sigma!=!0) and `"linear"` robust loss is used.

**Smooth primitives** (use these everywhere you currently have hinges/clamps):

* Hinge (h(t)=\max(0,t)) → **softplus**
  (h_\sigma(t)=\sigma,\log(1+\exp(t/\sigma))) for (\sigma>0); exact hinge when (\sigma=0).
* Absolute value (|t|) → **pseudo‑Huber**
  (|t|_\sigma=\sqrt{t^2+\sigma^2}-\sigma) for (\sigma>0); exact (|t|) when (\sigma=0).

**Robust per‑residual map** (\rho(u)):

* `"soft_l1"`: (2(\sqrt{1+u^2}-1))
* `"huber"`: (\tfrac12 u^2) if (|u|\le 1) else (|u|-\tfrac12)
* `"linear"`: (\tfrac12 u^2)

**Scalar objective**
[
L_\sigma(x)=\sum_{i=1}^{m}\rho!\big(r_i^{(\sigma)}(x)\big),
]
where each (r_i^{(\sigma)}) is built from your residual primitives with softplus/pseudo‑Huber.

> **Important:** Shape guards (§S) **must** also use these smoothers, so one global `sigma` switch smooths the whole objective.

---

## T.3 Default schedules & stages

Let `scene_scale = max(layout.scale, diag(seed_bbox))`.

```
sigmas        = [0.20, 0.10, 0.05, 0.02, 0.00] * scene_scale
robust_losses = ["soft_l1","huber","huber","linear","linear"]
stages        = ["adam",   "adam",  "lbfgs", "lbfgs", "lm"   ]
restarts      = [1,        1,       1,       2,       2      ]
```

* **Adam (large σ)**: reach a good basin on a smooth landscape.
* **L‑BFGS (mid σ)**: quasi‑Newton polish while still smoothed.
* **LM/TRF (σ=0)**: finish on the **exact** least‑squares with a second‑order step.

Deterministic multistart: on each σ stage, do `restarts[k]` attempts using **§18 seeding** (no random rotation when a **gauge edge** exists; escalate jitter per attempt). Keep the **best** by (total loss, then max residual, then DDC status).

---

## T.4 Torch autodiff integration

**Contract:** residual builder is **pure**, maps flat `x ∈ R^n` ↔ point coordinates via `model.index`, and accepts a `sigma` float.

> **Status:** Numeric finite-difference gradients drive the Adam stage today. The scaffold below remains the roadmap for wiring
> full Torch autodiff so stages can consume exact gradients/Jacobians when available.

```python
# Set Torch defaults (double precision)
import torch
torch.set_default_dtype(torch.float64)

def residuals_torch(x_tensor, model, sigma):
    """
    Pure function: 1) unpack x→coords, 2) compute residual groups,
    3) use smooth 'hinge'/'abs' helpers when sigma>0, exact when sigma=0.
    Returns a 1D tensor of shape (m,).
    """
    return r

def loss_torch(x_tensor, model, sigma, robust: str):
    r = residuals_torch(x_tensor, model, sigma)
    u = robust_map(robust, r)      # elementwise map; see T.2
    return u.sum()                 # scalar tensor

def grad_torch(x_np, model, sigma, robust):
    x = torch.tensor(x_np, requires_grad=True)
    L = loss_torch(x, model, sigma, robust)
    L.backward()
    g = x.grad.detach().cpu().numpy()
    return float(L.detach().cpu().numpy()), g

def jacobian_torch(x_np, model, sigma):
    # For LM/TRF final pass (σ=0). If too big, fall back to numeric (2-point).
    x = torch.tensor(x_np, requires_grad=True)
    def r_fn(y): return residuals_torch(y, model, sigma)
    J = torch.autograd.functional.jacobian(r_fn, x)   # shape (m,n)
    return J.detach().cpu().numpy()
```

**Bounds in Torch stages:** after each optimizer step, **project**: `x.clamp_(low, high)`.

---

## T.5 Optimizer backends

### T.5.1 Adam (Torch)

Purpose: aggressive progress on smooth loss.

```
- steps: loss_opts.adam_steps (≈800)
- lr:    loss_opts.adam_lr
- grad clipping: clip to ±loss_opts.adam_clip (∞-norm)
- gradient: central finite differences over `_scalar_loss_numpy` (step `adam_fd_eps` scaled per variable)
- stop early if relative improvement < early_stop_factor over 50 steps
```

### T.5.2 L‑BFGS (SciPy)

Use `scipy.optimize.minimize(..., method="L-BFGS-B")` on the same scalar loss (`_scalar_loss_numpy`). Pass `lbfgs_maxiter`
and `lbfgs_tol` via the `options` dict; reuse the candidate even if the optimizer reports a soft failure, then optionally fall
back to the SciPy least-squares stage.

### T.5.3 Final exact pass: SciPy LM/TRF (σ = 0)

Use `scipy.optimize.least_squares` with **analytic Jacobian** if feasible:

```python
from scipy.optimize import least_squares

def run_final_trf(model, x0, bounds, max_nfev):
    def r_np(x): return residuals_torch(torch.tensor(x), model, sigma=0.0).detach().cpu().numpy()
    def j_np(x): return jacobian_torch(x, model, sigma=0.0)
    res = least_squares(r_np, x0, jac=j_np, bounds=bounds,
                        method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=max_nfev, loss="linear")
    return res
```

If Jacobian is too costly (very large `m×n`), set `jac=None` (SciPy uses 2‑point FD).

> Optionally, if no bounds are active, allow one **LM** polish: `method="lm"`.

---

## T.6 Stage orchestrator (homotopy + multistart)

```python
def solve_globalized(model, x0, solver_opts: SolverOptions, loss_opts: LossModeOptions, plan=None):
    # Defaults
    sigmas  = loss_opts.sigmas or [0.20, 0.10, 0.05, 0.02, 0.00]
    robusts = loss_opts.robust_losses or ["soft_l1","huber","huber","linear","linear"]
    stages  = loss_opts.stages or ["adam","adam","lbfgs","lbfgs","lm"]
    restarts= loss_opts.restarts_per_sigma or [1,1,1,2,2]

    scale = max(model.scale, 1.0)
    sigmas = [s*scale for s in sigmas]

    incumbent = (float("inf"), x0)  # (loss, x)

    for k, sigma in enumerate(sigmas):
        stage, robust = stages[k], robusts[k]
        attempts = max(1, restarts[k])

        for a in range(attempts):
            x_start = incumbent[1] if a == 0 else initial_guess(model, rng, attempt=a, plan=plan)  # §18 (no rotation if gauge)
            x = torch.tensor(x_start, requires_grad=True)

            if stage == "adam":
                x = run_adam_torch(x, model, sigma, robust, loss_opts, model.bounds)
            elif stage == "lbfgs":
                x = run_lbfgs_torch(x, model, sigma, robust, loss_opts, model.bounds)
            else:  # "lm" final pass (σ=0)
                res = run_final_trf(model, x.detach().cpu().numpy(), model.bounds, loss_opts.lm_trf_max_nfev)
                x_np = res.x; L = float((res.fun**2).sum())
                if L < incumbent[0]: incumbent = (L, x_np)
                # DDC final gate
                return package_solution(model, incumbent[1], res)

            # Evaluate scalar Torch loss for incumbent update
            L_val = float(loss_torch(x, model, sigma, robust).detach().cpu().numpy())
            if L_val + loss_opts.early_stop_factor * max(1.0, incumbent[0]) < incumbent[0]:
                incumbent = (L_val, x.detach().cpu().numpy())

        # Optional quick DDC‑gate at each σ: accept incumbent unless DDC "mismatch"
        if not ddc_ok(program=model.program, x=incumbent[1], tol=None):
            # one extra restart at this σ
            x_extra = initial_guess(model, rng, attempt=attempts+1, plan=plan)
            # rerun current stage once; keep the better of {incumbent, extra}
            # (omitted for brevity)

    # Safety: if loop exits without LM/TRF (shouldn't), do final TRF at σ=0
    res = run_final_trf(model, incumbent[1], model.bounds, loss_opts.lm_trf_max_nfev)
    return package_solution(model, res.x, res)
```

**Notes**

* `package_solution(...)` builds your standard `Solution` with `point_coords`, `success`, `max_residual`, and breakdown.
* Use a single RNG seeded from program hash; **never** random‑rotate when a gauge edge exists (§7/§18.3).
* After each stage (and finally), run **DDC‑Check** (§16). Accept only if `status in {"ok","partial"}` (or `"ambiguous"` when allowed).

---

## T.7 Residual builder hooks (single switch)

Add **two helpers** and thread a `sigma` float through all residual primitives:

```python
def hinge(t, sigma):
    if sigma == 0.0: return torch.clamp(t, min=0.0)
    z = t / sigma
    # numerically stable softplus
    return sigma * torch.nn.functional.softplus(z)

def abs_smooth(t, sigma):
    if sigma == 0.0: return torch.abs(t)
    return torch.sqrt(t*t + sigma*sigma) - sigma
```

Use them for:

* ray/segment clamping hinges,
* min‑separation floors,
* **shape guards** (§S: height/angle/area floors),
* any other `max(0,·)` or `|·|`‑based term.

---

## T.8 Fallback (no Torch or disabled Loss‑mode)

When `loss_opts.enabled=False` or Torch is unavailable:

1. Run **the same σ schedule** and **robust loss schedule** (T.3).
2. At each σ stage, call **SciPy `least_squares`** (`method="trf"`, same tolerances), with:

   * `loss` set to `"soft_l1"`/`"huber"`/`"linear"` per stage,
   * deterministic reseeds exactly as in §18 (no random rotations).
3. Carry forward the **best incumbent** by total cost, then max residual, then DDC status.
4. Final stage at `σ=0`, `loss="linear"`, exact least‑squares.

This gives you a robust path even without autodiff.

---

## T.9 Quality & scaling

* Use **float64** throughout Torch and SciPy.
* **Scale** decision variables by scene scale to balance Jacobian columns; unscale on I/O.
* **Gradient clipping** protects Adam early; keep `adam_clip` conservative (≈10).
* For very large `m×n`, consider numeric Jacobian (`jac=None`) in final TRF; or compute J only on active groups.

---

## T.10 Tests (must‑pass)

1. **Cold‑start robustness**
   On a hinge‑heavy parallelogram/trapezoid, 10 random attempts → **≥9/10** success with this pipeline; plain TRF single‑start may fail.
2. **Homotopy benefit**
   With `σ=0` only, at least one seed fails; with schedule, all succeed.
3. **DDC consistency**
   Final `Solution` yields `DDC status in {"ok","partial"}` (or `"ambiguous"` if allowed).
4. **Equivalence on easy scenes**
   For well‑posed triangles, final targets match direct TRF within `1e-9`.
5. **Determinism**
   Fixed seed + same program → identical numeric output.

---

## T.11 Implementation checklist

* [ ] Add `SolverOptions`, `LossModeOptions`.
* [ ] Implement `hinge` / `abs_smooth` and thread `sigma` through all residual primitives (including §S guards).
* [ ] Torch functions: `residuals_torch`, `loss_torch`, `grad_torch`, `jacobian_torch`.
* [ ] Optimizers: `run_adam_torch` (with projection), `run_lbfgs_torch` (closure + projection), `run_final_trf`.
* [ ] Stage orchestrator `solve_globalized(...)` with schedules from T.3 and DDC gates.
* [ ] Respect §18 seeding rules (gauge endpoints never rotated/jittered).
* [ ] Wire into your existing `solve*` entry point; preserve the `Solution` shape.
* [ ] Add tests from T.10 to CI.
