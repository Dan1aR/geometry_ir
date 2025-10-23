# GeoScript IR — Technical Specification (agent view)

GeoScript IR is a compact DSL for 2D Euclidean scenes. The toolchain now routes every scene through a two-stage solver: hard
(equalities) flow through `python-solvespace`'s libslvs CAD core, while inequalities and aesthetic guards are refined in a
separate polishing step. The scope remains planar (2D) geometry.

---

## 1) Design goals

1. **Readable problem statements.** Scripts mimic olympiad prose (“Trapezoid ABCD with base AD…”, “Circle with center O…”, “Find
   ∠DBE”).
2. **Deterministic equalities.** Incidence, metric, and angular relations are enforced in libslvs with repeatable seeding policies.
3. **Polished diagrams.** Segment/ray semantics, min-altitude guards, and readability cues are applied post-CAD via numerical
   polishing that preserves hard constraints.
4. **Modular architecture.** Parsing/printing, validation plus desugaring, CAD solving, polishing, DDC, and rendering remain
   isolated modules with dedicated prompts.

---

## 2) Architecture overview

```
parse → validate → desugar (emit carriers) → CAD adapter → polish → DDC → render/export
```

* **`geoscript_ir.parser` / `lexer`** – build the AST.
* **`geoscript_ir.validate`** – structural checks, option validation.
* **`geoscript_ir.desugar`** – expand high level objects and emit carriers (lines/circles shared by segments/rays).
* **`geoscript_ir.cad.slvs_adapter`** – translate the desugared program into libslvs entities/constraints and solve equalities.
* **`geoscript_ir.polish`** – parameterise points on carriers, apply segment/ray clamps plus readability guards with SciPy.
* **`geoscript_ir.solver`** – façade orchestrating desugar → CAD → polish → DDC (`solve_scene`).
* **`geoscript_ir.ddc`** – Derive & Double-Check using the polished coordinates.
* **`geoscript_ir.tikz_codegen` / `printer`** – emit TikZ, diagnostics, and CLI output.

Module index additions:

| Module | Responsibility |
| --- | --- |
| `geoscript_ir.cad.slvs_adapter` | CAD interface (`SlvsAdapter.solve_equalities`) |
| `geoscript_ir.polish` | Post-CAD polishing (`polish_scene`, `PolishOptions`) |
| `geoscript_ir.solver` | Orchestrator façade (`SolveSceneOptions`, `solve_scene`) |

---

## 3) Lexical rules

* `ID` tokens are case-insensitive; they normalize to uppercase (`a-b` ≡ `A-B`).
* Strings use double quotes with C-style escapes.
* Numbers accept decimals and scientific notation. Symbolic values (`sqrt(...)`, `3*sqrt(2)`) become `SymbolicNumber` (text plus
  numeric value).
* `#` introduces a line comment.

---

## 4) Grammar (BNF)

Programs must satisfy this grammar. Solver-facing extensions include branch picking, `collinear`, `concyclic`, `equal-angles`,
`ratio`, and the `perp-bisector` / `parallel through` path forms.

```text
Program   := { Stmt }
Stmt      := Scene | Layout | Points | Obj | Placement | Annot | Target | Rules | Comment

Scene     := 'scene' STRING
Layout    := 'layout' 'canonical=' ID 'scale=' NUMBER
Points    := 'points' ID { ',' ID }
...
```

(The remainder matches the previous revision; only the solver pipeline changed.)

`angle A-B-C` is only a visual mark until `degrees=` appears.

---

## 5) Desugaring notes

* Every `line`, `segment`, and `ray` statement now emits a **shared carrier** entity; bounds for segments/rays are deferred to the
  polishing stage.
* Circle constructors create latent centers/radii that can be reused by tangency/point-on facts.
* High level polygons inject their carrier edges plus generated equalities (e.g. isosceles, right) as before.
* `point ... on segment`/`ray` statements attach to the carrier line immediately; polishing later clamps the parameter to
  `[0,‖AB‖]` or `[0,∞)`.

---

## 6) CAD mapping (libslvs)

`geoscript_ir.cad.slvs_adapter` exposes the canonical mapping in `CAD_MAPPING_TABLE`. Key entries:

| DSL fact | libslvs primitive |
| --- | --- |
| `line A-B` / `segment A-B` / `ray A-B` | `add_line_2d(A,B)` (segment/ray bounds deferred) |
| `point P on line A-B` | `coincident(P, lineAB)` |
| `segment A-B [length=L]` | `distance(A,B,L)` |
| `equal-segments(...)` | `length_diff(line_i, line_j, 0)` for every pair |
| `ratio (A-B : C-D = p : q)` | `ratio(lineAB, lineCD, p/q)` |
| `right-angle B-A-C` | `perpendicular(lineAB, lineAC)` |
| `angle A-B-C [degrees=θ]` | `angle(lineBA, lineBC, θ)` |
| `parallel-edges (A-B ; C-D)` | `parallel(lineAB, lineCD)` |
| `point P on circle center O` | `coincident(P, circle(O))` (circle created on demand) |

Branch selection (`choose=`) is honoured by biasing initial parameters when creating ambiguous points (near/far, left/right, cw/ccw).

---

## 7) Solve pipeline

### 7.1 CAD stage (`SlvsAdapter`)

* Builds a base workplane (`create_2d_base`) and adds all points with deterministic polar seeds plus light jitter based on
  `random_seed`.
* Carriers are emitted first; circle carriers are created lazily the first time they are referenced.
* Constraints are issued in stable order: carriers → incidence → metric/ratio → angular/parallel groups.
* Gauge selection (`SlvsAdapterOptions.gauge`) aligns the solved coordinates post-hoc so that `A=(0,0)`, `B=(1,0)`, and an
  optional `C` lies in the upper half-plane.
* Diagnostics return degrees of freedom (minus planar gauge) and any failing constraint indices.

### 7.2 Polishing stage (`polish_scene`)

* Points referenced by `point ... on segment/ray` are parameterised by a scalar `t` along the carrier direction; circle points use
  an angle parameter when needed.
* Residuals use a `softplus_k` hinge, normalised by the characteristic scene scale. Implemented guards:
  * Segment clamp: `softplus(-t) + softplus(t-‖AB‖)`.
  * Ray clamp: `softplus(-t)`.
  * Optional readability guards (min altitude, angle cushions, area floor, label avoidance) accept weights but default to gentle
    values (`w_shape=0.05`, `epsilon_h=0.06`, `epsilon_area=0.02`).
* SciPy `least_squares` keeps equality constraints satisfied by construction and returns the updated coordinates, residual
  breakdown, iteration count, and a heuristic `beauty_score = 1/(1+Σ residuals)`.

### 7.3 Orchestrator (`solve_scene`)

`SolveSceneOptions` bundles the CAD backend (`cad_solver="slvs"`), seed, gauge tuple, and polishing options. `solve_scene`:

1. Desugars if necessary (callers typically pass a desugared variant).
2. Runs `SlvsAdapter.solve_equalities` and short-circuits with diagnostics on failure.
3. Invokes `polish_scene` when enabled; otherwise returns CAD coordinates directly.
4. Feeds the polished coordinates into DDC (`derive_and_check` + `evaluate_ddc`) and packages the reports.

Returned `SolveResult` contains `coords`, `cad_status`, `polish_report`, `beauty_score`, and the DDC summary.

---

## 8) CLI additions

The CLI now accepts CAD-specific switches:

* `--cad slvs` – enable the libslvs pipeline (`solve_scene`).
* `--no-polish` – skip the polishing step (CAD coordinates only).
* `--gauge A,B,C` – explicit gauge triple for the CAD stage (two or three IDs).
* `--dump-cad` – log failing constraint indices and the raw DoF count when libslvs does not converge.

Without `--cad` the legacy least-squares solver remains available for comparison.

---

## 9) Appendix A – worked example: parallel chords in an angle

```text
scene "Parallel chords"
points A, B, C, D, E, F
angle D-A-F [degrees=60]
line A-D
line A-F
segment B-C
segment D-E
point B on ray A-D [choose=near anchor=A]
point C on ray A-F [choose=near anchor=A]
point D on ray A-D [choose=far anchor=A]
point E on ray A-F [choose=far anchor=A]
parallel-edges (B-C ; D-E)
```

Adapter emission:

1. Carriers: `line A-D`, `line A-F`, shared for both near/far points.
2. Incidence: every `point ... on ray` → `coincident` with the corresponding carrier.
3. Parallelism: `parallel(lineBC, lineDE)` keeps the chords parallel once their endpoints slide.
4. Angle: `angle(lineDA, lineFA, 60°)` seeds the opening.
5. Lengths remain unspecified, so libslvs returns a family of solutions with gauge DoF removed.

Polishing stage receives the CAD coordinates and introduces segment clamps to keep `B`/`C` between the near/far anchors while the
parallelism remains intact. Optional readability guards (altitude floor and angle cushion) stabilise acute configurations. The
resulting `SolveResult` reports a beauty score, residual breakdown (`clamp:segment:B`, `clamp:segment:C`), and a DDC status of `ok`.

---

## References

* `CAD_MAPPING_TABLE` in `geoscript_ir.cad.slvs_adapter` mirrors the mapping above for programmatic access.
* `PolishOptions` documents tuning knobs for clamps/guards.
* CLI usage: `python -m geoscript_ir scene.ir --cad slvs --gauge A,B,C --dump-cad`.
