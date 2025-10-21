# GeoScript IR — Project Documentation

GeoScript IR is a compact DSL for 2D Euclidean geometry scenes. The toolchain
parses source text into an AST, validates geometric intent, desugars composite
objects, assembles nonlinear residuals, and solves them numerically while a
deterministic derivation pass (DDC) cross-checks the solution. This document is
the authoritative description of both the language and the implementation.

---

## 1. Vision and Design Goals

1. **Readable problem statements.** Scripts mimic olympiad prose (e.g. "Trapezoid
   ABCD with base AD", "Circle with center O", "Find ∠DBE").
2. **Explicit, well-posed constraints.** Each statement yields numeric residuals.
   Translators enforce min-separation guards, carrier edge floors, near-parallel
   cushions, and orientation gauges.
3. **Modular architecture.** Lexing/parsing, validation plus desugaring,
   solver compilation, DDC checks, and TikZ export are isolated modules with
   dedicated prompts.

---

## 2. Architecture Map

### 2.1 Data Flow Overview

```
source text ──▶ lexer ──▶ parser ──▶ AST ──▶ validator ──▶ desugarer
                               │                        │
                               │                        └──▶ canonical Program (variants)
                               ▼
                        residual builder
                               │
         ┌──────────────┬──────┴───────┬───────────────┐
         ▼              ▼              ▼               ▼
      solver       DDC checker     renderers      consistency
 (least-squares)   (deterministic  (printer /     hot-fixes
                    derivations)    TikZ)
```

### 2.2 Module Index

| Component | Modules | Notes |
|-----------|---------|-------|
| Command-line tools | `geoscript_ir.__main__`, `geoscript_ir.demo` | Entry points for compiling or inspecting scenes. |
| Lexing & parsing | `geoscript_ir.lexer`, `geoscript_ir.parser`, `geoscript_ir.ast` | Convert source lines into AST statements with spans. |
| Numeric helpers | `geoscript_ir.numbers`, `geoscript_ir.orientation` | Symbolic numbers with numeric payloads, orientation utilities. |
| Validation | `geoscript_ir.validate`, `geoscript_ir.consistency` | Structural checks, option validation, solver dry-run, and missing-support warnings with auto hot-fixes. |
| Desugaring | `geoscript_ir.desugar` | Expands polygons/traits into primitive facts; produces program variants. |
| Residual builder & solver | `geoscript_ir.solver` | Translates validated programs into least-squares residuals and solves them with SciPy. |
| Deterministic derivation & cross-check | `geoscript_ir.ddc` | Recomputes derivable points from explicit rules and compares against the numeric solution. |
| Rendering & exports | `geoscript_ir.printer`, `geoscript_ir.reference`, `geoscript_ir.reference_tikz`, `geoscript_ir.tikz_codegen` | Pretty-printing, example gallery, TikZ export helpers. |
| Tests & prompts | `tests/`, `.github/prompts/*.prompt.md` | Regression suite and alignment prompts (`compile`, `lint`). |

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

Branch choices act as soft hinges to resolve two-root ambiguities without hard
constraints.

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
* After structural checks, the validator deep-copies the program and invokes the
  solver translator (`translate`) to ensure residual generation succeeds. Any
  `ResidualBuilderError` or `ValueError` becomes a `ValidationError` annotated
  with source spans.

### 4.2 Consistency Hot-Fixes (`geoscript_ir.consistency`)

* Detects missing support segments/rays for `angle`, `right-angle`, and
  `equal-angles` statements. Suggested `segment` hot-fixes can be auto-inserted.
* Confirms that polygon-derived edges exist when required by later constraints.
* Warns about unsupported rays for branch-dependent constructs (parallel edges,
  tangents, etc.) and offers hot-fix statements tagged `origin='hotfix(consistency)'`.

---

## 5. Desugaring to Canonical Facts (`geoscript_ir.desugar`)

High-level constructs expand into primitive relations consumed by the solver.
Desugaring keeps helper statements deduplicated via canonical keys.

* **`polygon A-B-C-...`** → segments along the perimeter.
* **`triangle A-B-C`** → carrier edges `AB`, `BC`, `CA`; optional
  `isosceles=atX` adds equality constraints; `right=atX` adds a right angle.
* **`quadrilateral A-B-C-D`** → segments around the quadrilateral.
* **`trapezoid A-B-C-D [bases=X-Y]`** → quadrilateral + `parallel-edges`
  between the chosen base pair and the opposite side + non-parallel guards on
  legs; the named base anchors orientation gauges.
* **`parallelogram A-B-C-D`** → `parallel-edges (A-B; C-D)` and
  `parallel-edges (B-C; A-D)` with optional equalities based on user options.
* **`rectangle`** → parallelogram + right angle; **`square`** → rectangle +
  equal segments; **`rhombus`** → equal segments + both parallel pairs.
* **`collinear (P1,...,Pn)`** → expands to full collinearity constraints
  (requires `n ≥ 3`).
* **`concyclic (P1,...,Pn)`** → introduces a latent center/radius and enforces
  equal radii to each listed point.
* **`equal-angles (A-B-C, ... ; D-E-F, ...)`** → ties every listed angle to a
  representative using `atan2` residuals.
* **`ratio (A-B : C-D = p : q)`** → enforces `q‖AB‖ − p‖CD‖ = 0`.

`circle through (...)` and `circumcircle of ...` both introduce a shared latent
center/radius. The module can emit multiple variants when helper constructions
introduce branch choices; variants propagate to the solver for robust coverage.

---

## 6. Residual Library (Solver Primitives)

Let `v(P)` denote the 2D position of point `P`; `AB := v(B)−v(A)`. Cross products
use 2D scalar cross (`×`), dots use Euclidean dot (`·`), and `‖·‖` is the Euclidean
norm. Residuals are assembled by `geoscript_ir.solver.translate`.

### 6.1 Placement & Incidence Residuals

* `point P on line A-B` → `cross(AB, AP) = 0`.
* `point P on ray A-B` → collinearity + forwardness hinge: `cross(AB, AP)=0`,
  `max(0, −AB·AP)=0`.
* `point P on segment A-B` → collinearity + clamping hinges: `cross(AB, AP)=0`,
  `max(0, −AB·AP)=0`, `max(0, AB·(AP−AB))=0`.
* `point P on circle center O` → `‖OP‖ − ‖OB₀‖ = 0` (where `B₀` is the witness
  from `radius-through`).
* `intersect (path1) with (path2) at X(, Y)` → ensures each intersection point
  satisfies both path constraints.
* **Branch picking** (soft selectors):
  * `choose=near|far, anchor=Q` → bias nearer/farther intersection relative to
    anchor `Q`.
  * `choose=left|right, ref=A-B` → penalize wrong orientation sign against
    oriented line `AB`.
  * `choose=cw|ccw, anchor=Q` (optional `ref=A-B`) → prefer clockwise /
    counter-clockwise rotation about anchor/reference.

### 6.2 Metric Relations

* `equal-segments (E... ; F...)` → choose representative edge `R`; enforce
  `‖E‖ − ‖R‖ = 0` for each edge.
* `segment A-B [length=L]` → `‖AB‖ − L = 0` (supports symbolic `L`).
* `midpoint M of A-B` → `AM = MB` and `M` collinear with `AB`.
* `ratio (A-B : C-D = p : q)` → `q‖AB‖ − p‖CD‖ = 0`.

### 6.3 Angular, Parallel, and Orthogonality Constraints

* `right-angle A-B-C` → `(BA)·(BC) = 0`.
* `angle A-B-C [degrees=θ]` → `atan2( cross(BA, BC), BA·BC ) − θ = 0`
  (degrees converted to radians internally).
* `equal-angles (...)` → equalize `atan2` of each angle with the representative.
* `angle-bisector U-V-W` (as `Path`) → direction equidistant in angle; `external`
  flips the bisector.
* `parallel-edges (A-B; C-D)` → `cross(AB, CD) = 0` with a turn-sign guard to
  avoid 180° flips.
* `perpendicular at T to A-B foot H` → `(AB)·(TH) = 0` and `H` lies on `A-B`.
* `perp-bisector of A-B` (as `Path`) → passes through midpoint of `AB` and is
  perpendicular to `AB`.
* `parallel through P to A-B` (as `Path`) → line through `P` parallel to `AB`.

### 6.4 Circle-Specific Constraints

* `circle center O radius-through B` → witness `B` defines radius; reuse for all
  `point ... on circle center O` constraints.
* `circle through (...)` / `circumcircle of ...` → latent `(Oc, Rc)` with
  equal-radius constraints for all listed points.
* `incircle of A-B-C` → incenter from angle bisectors; equal distances to sides
  enforced via perpendicular feet.
* `tangent at T to circle center O` → `OT ⟂` tangent direction and `T` on circle.
* `line X-Y tangent ... at T` → `X,Y,T` collinear and `OT ⟂ XY`.
* `diameter A-B to circle center O` → `O,A,B` collinear and `‖OA‖ = ‖OB‖`.

---

## 7. Solver Pipeline (`geoscript_ir.solver`)

### 7.1 Derivation Planning

* `plan_derive(program)` inspects every statement to register deterministic
  constructions (midpoints, feet, diameters, tangents, line intersections) and
  to flag ambiguous points introduced by `choose=` options or circle
  intersections.【F:geoscript_ir/solver.py†L963-L1028】
* When multiple `point ... on ...` statements reference the same point without a
  branch selector, the planner synthesizes an intersection rule so the solver can
  derive that point instead of treating it as a variable.【F:geoscript_ir/solver.py†L1030-L1044】
* The resulting `DerivationPlan` separates `base_points`, automatically derived
  points, ambiguous points, and logs human-readable `notes` used for later
  reporting.【F:geoscript_ir/solver.py†L1046-L1074】

### 7.2 Model Compilation & Residual Assembly

* `compile_with_plan(program, plan)` applies the plan while building a numeric
  `Model`. It fixes point ordering, gathers polygon metadata, tracks carrier
  edges for guards, records layout hints, and picks an orientation gauge edge so
  subsequent solves are stable.【F:geoscript_ir/solver.py†L2465-L2537】
* `build_seed_hints` annotates the model with per-point and global hints for the
  initializer (circle radius fallbacks, tangent mirrors, diameter partners,
  concyclicity groups, etc.), which are later consumed by `initial_guess`.【F:geoscript_ir/solver.py†L294-L515】
* After the first seed, plan guards are evaluated; any derived point that
  violates incidence bounds is promoted back to a variable and tagged in the plan
  notes so the caller understands the relaxation.【F:geoscript_ir/solver.py†L3007-L3047】
* `Model` instances retain layout metadata (`layout_canonical`,
  `layout_scale`), plan notes, seed hints, polygon descriptors, and a copy of the
  residual configuration for later inspection.【F:geoscript_ir/solver.py†L2994-L3023】
* `ResidualBuilderConfig` exposes solver tunables (min-separation scale, edge
  floors, shape weights, etc.) and can be inspected or replaced via
  `get_residual_builder_config` / `set_residual_builder_config`. Residual
  builders raise `ResidualBuilderError` when rejecting unsupported statements.【F:geoscript_ir/solver.py†L1089-L1254】
* `translate(program)` is a thin wrapper that runs `plan_derive` followed by
  `compile_with_plan` on the validated program.【F:geoscript_ir/solver.py†L3052-L3056】

### 7.3 Seeding, Loss Modes & Solve Loop

* `initial_guess(model, rng, attempt, plan=...)` uses the stored seed hints and
  layout scale to place points, respecting tangency externals, median directions,
  and previously derived coordinates while reseeding each attempt.【F:geoscript_ir/solver.py†L3059-L3269】
* `SolveOptions` configure the base SciPy `least_squares` call. When
  `enable_loss_mode` is true, `_solve_with_loss_mode` executes a staged schedule
  of sigma-smoothing and robust losses (Soft L1, Huber, Levenberg–Marquardt) with
  multistart restarts before falling back to the classic solver path on
  failure.【F:geoscript_ir/solver.py†L1104-L1519】【F:geoscript_ir/solver.py†L4156-L4173】
* The standard solver performs multiple reseeds, automatically extends the
  attempt budget when all runs miss the tolerance, and optionally relaxes a small
  set of min-separation guards to salvage near-coincident configurations while
  recording warnings.【F:geoscript_ir/solver.py†L4180-L4344】
* `Solution` objects expose the solved coordinates, residual breakdown, warning
  log, and helpers like `normalized_point_coords`, which delegates to
  `normalize_point_coords` for deterministic scaling.【F:geoscript_ir/solver.py†L1195-L1269】

### 7.4 Structural Guards (#NEW)

The appendix formerly labelled §6.5/S.* is integrated here to prevent valid but
"squished" shapes. All guards are **soft, scale-aware hinge residuals** activated
only when a scene is under-constrained.

#### S.1 Constants (defaults)

```
ε_h     = 0.06    # min altitude as a fraction of a nearby side length
s_min   = 0.10    # min |sin(angle)| cushion between adjacent edges (~5.7°)
ε_A     = 0.02    # area floor factor relative to longest side squared
w_shape = 0.05    # small weight for all "shape" residuals (≪ 1.0 for hard facts)
```

Implement these in the residual builder configuration; expose as tunables (see
§7.2).

#### S.2 Height Floor (Altitude Hinge)

For side `(AB)` and opposite vertex `(C)`:

```
h(AB;C)=| (B−A)×(C−A) | / |B−A|
h_min = ε_h · max(|B−A|, |C−B|)
r_height(A,B,C) = max(0, h_min − h(AB;C))
```

* **Triangle (ABC)**: apply to all three altitudes with weight `w_shape/3`.
* **Parallelogram (ABCD)**: apply to two independent heights, e.g. `h(AB;C)` and
  `h(BC;D)`.
* **Trapezoid (ABCD)**: apply to heights from non-base vertices to each base.

#### S.3 Adjacent-Side Angle Cushion (Non-Parallel Margin)

Let unit directions `u = (B-A)/|B-A|`, `v = (C-B)/|C-B|`,
`s = |u×v| = |sin∠ABC|`.

```
r_angle(A,B,C) = max(0, s_min − s)
```

Apply at every declared polygon vertex (triangles, trapezoids, parallelograms,
rectangles, squares, rhombi). Right angles dominate in rectangles/squares so the
cushion rarely triggers.

#### S.4 Area Floor (Scaled by Longest Edge)

Let `L_max = max` side length of the polygon, `A =` polygon area.

```
A_min = ε_A · L_max²
r_area = max(0, A_min − A)
```

#### S.6 Residual Aggregation

Add guards where polygons expand in the desugared program:

```
residuals += w_shape * [
  r_height(...), r_angle(...), r_area(...),  # per-object as applicable
  ...
]
```

These guards **do not** participate in DDC (§8); they are aesthetic stabilizers
only.

### 7.5 Variant Utilities

* `score_solution` ranks solutions by convergence success and residual magnitude
  so CLI tools can pick the strongest candidate.【F:geoscript_ir/solver.py†L1231-L1234】
* `solve_best_model(models, options)` solves each compiled variant and returns
  the best-performing index + solution tuple, while `solve_with_desugar_variants`
  integrates desugaring, translation, and selection into one helper returning a
  `VariantSolveResult`.【F:geoscript_ir/solver.py†L4378-L4412】


---

## 8. Deterministic Derivation & Cross-Check (DDC) (`geoscript_ir.ddc`)

* Implements §16 of the specification. Given a solved program, the DDC derives
  candidate coordinates for deterministically computable points and compares the
  result against the numeric solution.
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

* Generates TikZ code using solver output and metadata. Handles point labels,
  side labels, and optional styling marks from options.
* `reference_tikz` pairs with `docs/examples` to provide curated scene examples.

### 9.3 Reference Scenes (`geoscript_ir.reference`)

* Loads bundled GeoScript scenes for demos/tests. Offers quick inspection via
  `python -m geoscript_ir.demo`.

### 9.4 Orientation & Coordinate Normalization

* `apply_orientation(program, point_coords)` reorients solved coordinates for
  display. It prefers source `trapezoid` declarations (especially those with an
  explicit `bases=` option), falling back to isosceles triangles inferred from
  options or equal-segment groups; otherwise it returns the original coordinates
  with an identity transform.【F:geoscript_ir/orientation.py†L42-L239】
* Trapezoid candidates are rotated so the averaged base direction becomes
  horizontal and reflected, if necessary, to keep the shorter base above the
  longer one. The routine records the applied matrix, translation, pivot, and the
  figure that triggered the transform inside an `OrientationResult`.【F:geoscript_ir/orientation.py†L240-L336】
* `normalize_point_coords` (and `Solution.normalized_point_coords`) are exposed at
  the package level for deterministic min/max scaling when rendering or printing
  coordinates outside the CLI.【F:geoscript_ir/solver.py†L1195-L1269】

---

## 10. Command-Line Interfaces & Tooling

* `python -m geoscript_ir` exposes a CLI that tokenizes, parses, validates, and
  solves scenes. Runtime flags cover logging level, solver seed, reseed attempt
  budget, and optional TikZ export path for the best-scoring variant.【F:geoscript_ir/__main__.py†L31-L57】
* Each run logs desugared variants, consistency hot-fixes, solver statistics, and
  chooses the winner via `score_solution`, printing both raw and normalized point
  coordinates before optionally writing a standalone TikZ document.【F:geoscript_ir/__main__.py†L79-L173】
* `geoscript_ir.demo` launches an interactive prompt that lets users inspect
  bundled scenes.
* `.github/prompts/compile.prompt.md` enumerates required CI steps: install with
  `pip install -e ".[test]"`, run `pytest`, and execute the integration suite
  (`tests/integrational/test_gir_scenes.py`).

---

## 11. Testing Strategy

* `tests/test_solver.py` covers residual assembly and solver behavior with unit
  scenes.
* `tests/integrational/test_gir_scenes.py` exercises end-to-end parsing → solver
  → DDC → TikZ for curated scenarios.
* Additional regression fixtures live under `tests/` alongside sample GeoScript
  snippets in `examples/`.

---

## 12. Change Log Notes

* Structural guard constants and hinges (formerly Appendix §6.5/S.*) are now in
  §7.3.
* Validation rules document triangle `right=atA|atB|atC` options and
  `concyclic` ≥4-point requirements to match the implemented validator behavior.

