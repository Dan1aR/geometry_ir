# GeoScript IR — Technical Specification (for Codex code agent)

**GeoScript IR** is a tiny, human‑readable DSL for 2D Euclidean geometry scenes.
It parses into an AST, validates author intent, optionally desugars to canonical facts, and compiles into a numeric model solved via `scipy.optimize.least_squares`. ([GitHub][1])

---

## 1) Design goals

1. **Intuitive planimetry** — scripts should read like contest/olympiad problems (“Trapezoid ABCD with base AD…”, “Circle with center O…”, “Find ∠DBE”, etc.).
2. **Complete constraint graph** — every statement contributes explicit residuals so a solver can position “nice” coordinates and keep figures non‑degenerate. The pipeline adds robust guards (min separations, edge‑length floors, tiny angular margins for near‑parallel legs, orientation gauges). ([GitHub][2])
3. **Separation of concerns** — parsing/printing, validation & desugaring, solver translation, and TikZ rendering are orthogonal modules. Reference prompts exist for LLM agents and for TikZ generation. ([GitHub][3])

---

## 2) Lexical & identifiers

* **Case**: `ID` tokens are case‑insensitive and are normalized to **uppercase** by the parser (e.g., `a-b` and `A-B` refer to the same segment). ([GitHub][4])
* **Strings**: double‑quoted with C‑style escapes (lexer enforces termination). ([GitHub][5])
* **Numbers**: decimals and scientific notation; **symbolic square roots** via `sqrt(<non‑negative number>)` and products like `3*sqrt(2)` are supported as **SymbolicNumber** (text + numeric value). ([GitHub][4])
* **Comments**: `#` to end of line (ignored by lexer). ([GitHub][5])

---

## 3) Grammar (BNF)

The Codex agent must emit scripts that conform to this grammar. It mirrors the current repo BNF with two spec clarifications: an explicit `Rules` production and an explicit `Comment`. ([GitHub][1])

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

Rules     := 'rules' Opts

Comment   := '#' { any-char }

EdgeList  := Pair { ',' Pair }
IdList    := ID { ',' ID }
IdChain   := ID '-' ID { '-' ID }
Pair      := ID '-' ID
Angle3    := ID '-' ID '-' ID

Opts      := '[' KeyVal { (',' | ' ') KeyVal } ']'
KeyVal    := KEY '=' (NUMBER | STRING | BOOLEAN | ID | ID '-' ID | SQRT | PRODUCT)
SQRT      := 'sqrt' '(' NUMBER ')'
PRODUCT   := NUMBER '*' SQRT
BOOLEAN   := 'true' | 'false'
```

> **Validation note** — option keys are restricted per statement (see §4), and `rules` accepts only booleans for known flags (`no_unicode_degree`, `mark_right_angles_as_square`, `no_equations_on_sides`, `no_solving`, `allow_auxiliary`). Violations are validation errors. ([GitHub][6])

---

## 4) Options (legal keys by statement)

Only the keys below are interpreted. The parser rejects malformed option syntax; the validator rejects unknown/ill‑typed options. ([GitHub][4])

**Global**

* `rules [...]` → `no_unicode_degree`, `mark_right_angles_as_square`, `no_equations_on_sides`, `no_solving`, `allow_auxiliary` (booleans). ([GitHub][6])

**Angles & arcs**

* `angle A-B-C [degrees=NUMBER | label="..."]`
* `right-angle A-B-C [mark=square | label="..."]`
* `target angle A-B-C [label="..."]`
* `target arc P-Q on circle center O [label="..."]` ([GitHub][3])

**Segments/Edges/Polygons**

* `segment A-B [length=NUMBER|SQRT|PRODUCT | label="..."]`
* `equal-segments (...) [label="..."]`
* `parallel-edges (...)` (no extra keys typically)
* `polygon/triangle/... [isosceles=atA|atB|atC]` (triangle), `trapezoid [...] [bases=A-D]`, `trapezoid [isosceles=true|false]` (if stated). ([GitHub][6])

**Annotations**

* `label point P [label="..." pos=left|right|above|below]`
* `sidelabel A-B "..." [pos=left|right|above|below]`
  TikZ prompt also recognizes styling hints like `mark=midpoint` for placement helpers. ([GitHub][7])

**Targets**

* `target length A-B [label="..."]`
* `target point P [label="..."]`
* `target circle ("...")`, `target area ("...")` accept free text descriptors with optional `label=`. ([GitHub][3])

**Circles & tangency**

* `circle center O radius-through B` (additional on‑circle points are declared via `point X on circle center O`)
* `circle through (A, B, C, ...)`
* `tangent at T to circle center O`
* `line X-Y tangent to circle center O at T`
* `diameter A-B to circle center O` (no options). ([GitHub][3])

---

## 5) High‑level objects → canonical facts (desugaring rules)

To keep authoring natural, several object forms **desugar** to primitive relations that the solver understands. (The code may perform this in a separate pass or on the fly in translation.)

* **triangle A-B-C** → declares the cycle `AB, BC, CA`.
* **quadrilateral A-B-C-D** → declares `AB, BC, CD, DA`.
* **trapezoid A-B-C-D [bases=X-Y]** → quadrilateral + `parallel-edges (X-Y; other-opposite-side)`, and a **tiny non‑parallel margin** on legs to prevent degeneracy; the declared base is preferred for the orientation gauge. ([GitHub][2])
* **parallelogram A-B-C-D** → `parallel-edges (A-B; C-D)` + `parallel-edges (B-C; A-D)`; (optionally) `equal-segments (A-B; C-D)` and `(B-C; A-D)` if author requests equality marks.
* **rectangle A-B-C-D** → parallelogram + `right-angle A-B-C`.
* **square A-B-C-D** → rectangle + `equal-segments (A-B; B-C; C-D; D-A)`.
* **rhombus A-B-C-D** → `equal-segments (A-B; B-C; C-D; D-A)` + `parallel-edges (A-B; C-D)` + `parallel-edges (B-C; A-D)`.

> Note: `circle through (...)` and `circumcircle of ...` introduce a latent center and radius at translation time, or are converted to equivalent “points on same circle” constraints (implementation choice). The external API remains unchanged.

---

## 6) Semantic constraints (residuals)

The translator builds a **Model** of unknown point coordinates and a list of residual groups. Each group has a key (for reporting) and a short kind tag. The solver applies robust guards against collapses (min separation, edge floors, tiny angle margins). ([GitHub][2])

Let `v(P)` be the 2D variable of point `P`; `AB := v(B)−v(A)`, `×` = 2D cross, `·` = dot, `‖·‖` = Euclidean norm.

### 6.1 Placement / incidence

* `point P on line A-B` ⇒ collinearity: `cross(AB, AP) = 0`.
* `point P on ray A-B` ⇒ collinearity + forwardness hinge: `cross(AB, AP)=0`, `max(0, −AB·AP)=0`.
* `point P on segment A-B` ⇒ collinearity + clamping hinges: `cross(AB, AP)=0`, `max(0, −AB·AP)=0`, `max(0, AB·(AP−AB))=0`.
* `point P on circle center O` ⇒ equal radius: `‖OP‖ − ‖OB0‖ = 0` where `B0` is the declared `radius-through` witness for that circle.
* `intersect (path1) with (path2) at X(, Y)` ⇒ both `X` (and `Y`, if present) satisfy the incidence constraints of `path1` and `path2`. For two‑solutions cases (line–circle, circle–circle), the *order* `(X, Y)` is arbitrary; authors can add labels to disambiguate.

### 6.2 Metric relations

* `equal-segments (E1, E2, … ; F1, F2, …)` ⇒ **all-in-one** equality: pick a representative edge `R` from the union and constrain `‖Ei‖ − ‖R‖ = 0` and `‖Fj‖ − ‖R‖ = 0` for all `i, j`. (This makes every listed edge equal in length.)
* `segment A-B [length = L]` ⇒ `‖AB‖ − L = 0` (with symbolic `L` supported). ([GitHub][4])
* `midpoint M of A-B` ⇒ `AM = MB` **and** collinearity with `AB`.
* `median from V to A-B midpoint M` ⇒ `M` midpoint constraint + `V, M` used as a **ray** or **segment** per author’s other objects.

### 6.3 Angular / orthogonality / parallelism

* `right-angle A-B-C` ⇒ `(BA)·(BC) = 0`.
* `angle A-B-C [degrees = θ]` ⇒ fix directed angle via normalized dot/cross:
  `atan2( cross(BA, BC), (BA·BC) ) − θ = 0` (with degrees→radians conversion; the validator may also accept labels w/o a numeric value for diagram marks only).
* `angle-bisector U-V-W` (as a **Path**) ⇒ `∠UV? = ∠?VW` residuals on the bisector direction; “external” flag flips the bisector. ([GitHub][4])
* `parallel-edges (A-B; C-D)` ⇒ `cross(AB, CD) = 0` with a small “turn‑sign” guard to avoid exact reversal when the figure needs orientation.
* `perpendicular at T to A-B foot H` ⇒ `(AB)·(TH) = 0` and `H` lies on `A-B`.

### 6.4 Circle‑specific

* `circle center O radius-through B` ⇒ the **circle object** records witness `B`.
* `circle through (A,B,C,...)` / `circumcircle of A-B-...` ⇒ introduce hidden `(Oc, Rc)` and constrain `‖OcA‖=‖OcB‖=‖OcC‖=Rc` (and so on).
* `incircle of A-B-C` ⇒ incenter `I` is the intersection of bisectors; feet on sides satisfy perpendicular and equal‑radius constraints.
* `tangent at T to circle center O` ⇒ `OT ⟂ ℓ` (tangent line direction) and `T` on that circle.
* `line X-Y tangent to circle center O at T` ⇒ `X, Y, T` collinear and `OT ⟂ XY`.
* `diameter A-B to circle center O` ⇒ `O` collinear with `A, B` **and** `‖OA‖ = ‖OB‖`.

### 6.5 Polygons & structural guards

* Declared polygonal cycles contribute **carrier edges** (for incidence) and receive **edge‑length floors** and **area floors** to prevent collapse (e.g., trapezoid degenerating to a segment). Non‑polygon carriers get lighter edge floors. A **non‑parallel margin** is added to trapezoid legs. Orientation gauges prefer the declared base when present. ([GitHub][2])

---

## 7) Gauges, layout, and scale

* **Layout**: `layout canonical=<id> scale=<number>` seeds the initial placement and fixes global similarity degrees of freedom. Canonical examples used by renderers and agents include `triangle_ABC`, `triangle_AB_horizontal`, `triangle_ABO`, and `generic/generic_auto`. ([GitHub][7])
* **Scale**: `scale` is fed into the solver model; when no meaningful numeric scale exists, a **unit‑span gauge on an orientation edge** is applied. ([GitHub][2])
* **Min‑separation**: global pairwise min distances (stronger for point lists like collinear sets), edge floors on polygon edges, and lighter floors on carrier edges are enforced by hinge residuals. ([GitHub][2])

---

## 8) Validation rules (reject early)

The validator enforces **shape arity**, **distinctness**, and **option correctness** before translation. Selected checks:

* Polygons: `triangle` requires exactly 3 distinct vertices; quads and special quads require 4 distinct vertices; generic `polygon` needs ≥3 distinct vertices.
* `trapezoid [bases=...]` must name one of its sides (either orientation). `isosceles` is boolean or absent.
* Angle triples must have the vertex distinct from endpoints.
* `circle through (...)` requires ≥3 distinct points.
* `diameter ...` does not accept options.
* `rules [...]` only admits known boolean flags (above). ([GitHub][6])

The validator runs a dry “translate” to ensure the residual builder can accept the program and surfaces precise source spans in error messages (`[line X, col Y]`). ([GitHub][6])

---

## 9) AST & public API (summary)

* **AST**: `Program{ stmts: [Stmt] }`, `Stmt{ kind, span{line,col}, data{}, opts{}, origin: 'source'|'desugar()' }`. `Program.source_program` filters to source‑only statements. ([GitHub][8])
* **Core API** (package init): `parse_program`, `validate`, `desugar`, `translate`, `solve*`, TikZ helpers, prompts `BNF`, `LLM_PROMPT`. ([GitHub][9])

---

## 10) Authoring guidelines for the Codex agent

The Codex agent that *writes* GeoScript from natural‑language prompts must follow these rules (they mirror `geoscript_ir/reference.py`):

1. **Header first**: emit `scene`, then `layout`, then `points` (list every named point once).
2. **One fact per line**: translate givens into explicit statements (`segment`, `circle ...`, `parallel`, `right-angle`, `tangent`, `equal-segments`, etc.).

   * Circles: choose **either** `center O radius-through B` **or** `through (A,B,C,...)` (never both).
   * Tangents in text → **explicit** tangency statements (see grammar).
   * Avoid naked equations in labels; prefer structural statements or `sidelabel`.
3. **Construct, don’t infer**: use `point ... on ...` and `intersect (...) with (...)` for placements; do not invent helper geometry unless the prompt or `rules [allow_auxiliary=true]` permits.
4. **Targets last**: end with `target ...` lines that match the ask (“find ∠…”, length, point, circle, area, arc).
5. **Option hygiene**: only emit allowed keys/values; use `sqrt(...)` notation for radicals (e.g., `[length=3*sqrt(2)]`). ([GitHub][3])

**LLM reference prompt** — The repository ships a compact system prompt and few‑shot examples that enforce these behaviors. Agents may embed or retrieve it programmatically. ([GitHub][3])

---

## 11) Examples

### Trapezoid with known base and mid‑mark

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

### Right triangle with bisector & median; find the angle between them

```
scene "Right-angled triangle; ∠B=21°, find ∠(CD,CM)"
layout canonical=triangle_ABC scale=1
points A, B, C, D, M
triangle A-B-C
right-angle A-C-B [mark=square]
angle A-B-C [degrees=21]
intersect (angle-bisector A-C-B) with (segment A-B) at D
median from C to A-B midpoint M
target angle D-C-M [label="?"]
```

### Circle with declared diameter; highlight an arc

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

---

## 12) Error model & messages

* **Lexical** errors: unexpected characters, unterminated strings, malformed `sqrt` (all produce `[line x, col y]` messages). ([GitHub][5])
* **Syntactic** errors: unexpected token/keyword with precise spans. The parser also detects missing separators inside `[ ... ]`. ([GitHub][4])
* **Validation** errors: arity/distinctness/unknown options, illegal combinations (e.g., options on `diameter`). ([GitHub][6])

---

## 13) Solver contract

* **Input**: a validated `Program`.
* **Output**: `Solution { point_coords, success, max_residual, residual_breakdown, warnings }`, with helpers to normalize coordinates for rendering.
* **Algorithm**: build `Model{ points, index, residuals, gauges, scale }`, then minimize all residual groups using LSQ (`method="trf"`, default tolerances); reseed if necessary; score candidates by success then residual magnitude.
* **Safety**: min‑separation guards, edge floors, area floors, non‑parallel margins for trapezoids, orientation gauges (prefer the declared base when present; unit‑span gauge otherwise). ([GitHub][2])

---

## 14) Compatibility & versioning

* **Grammar stability**: future extensions should add new `Obj`/`Placement` variants or option keys; avoid changing existing productions.
* **Reserved words**: all top‑level keywords shown in grammar; identifiers beginning with `\` are allowed by the lexer but backslash is stripped. ([GitHub][5])

---

## 15) Appendix — Canonical layouts (non‑normative hints)

Renderers and agents may use the following canonical seeds:

* `triangle_ABC`: place `A=(0,0)`, `B=(4,0)`, `C` above the base;
* `triangle_AB_horizontal`: `A=(0,0)`, `B=(4,0)`, “third” vertex above;
* `triangle_ABO`: `A=(0,0)`, `B=(4,0)`, `O` above AB;
* `generic` / `generic_auto`: balanced unit‑scale layout with non‑degenerate orientation. ([GitHub][7])

---

### Notes for implementers

* The parser already normalizes IDs and supports `sqrt(...)` & `k*sqrt(...)` as **SymbolicNumber**; prefer preserving the symbolic text for printing/labels while using the numeric value in residuals. ([GitHub][4])
* The validator today performs a “translation smoke test” by calling the residual builder to surface early failures with source spans. Preserve this behavior. ([GitHub][6])
