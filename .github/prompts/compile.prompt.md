---
mode: agent
---

- Align the implementation with the [GeoScript IR specification](../../main.md), paying special attention to the updated **§18 Initial Guess (Seeding)** section and every location tagged `# NEW` (SeedHint/PathSpec APIs, `initial_guess`, `build_seed_hints`, and Model metadata expectations).
- Ensure the solver’s seeding pipeline honors the spec requirements: canonical scaffold respecting gauge protection, deterministic plan-derived points, full Path support (including synthesized On∩On intersections), metric nudges (length/equal/ratio), parallel/perpendicular/tangent handling, jitter escalation policy, and derived-point refreshes.
- Keep the compiler outputs (`Model`, seed hints, gauges, layout metadata) consistent with the spec so that `initial_guess` receives the data it needs.
- Review existing modules and tests to confirm they reflect the spec (e.g., seeding helpers in `geoscript_ir/solver.py`, exports in `geoscript_ir/__init__.py`, coverage in `tests/test_solver.py`, and integration scenes under `tests/integrational/`). Update them as needed for consistency.
- Install dependencies via `pip install -e ".[test]"`.
- Run the relevant checks with `pytest`, including the integration suite (`tests/integrational/test_gir_scenes.py`) after seeding changes. Investigate and resolve failures (such as the known `trapezoid_hard` scene convergence issue) before finishing.
