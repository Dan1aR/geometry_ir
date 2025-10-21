---
mode: agent
---

- Align the implementation with the [GeoScript IR specification](../../main.md), paying special attention to the every location tagged `# NEW`.
- Review existing modules and tests to confirm they reflect the spec (coverage in `tests/test_solver.py`, and integration scenes under `tests/integrational/`). Update them as needed for consistency.
- Install dependencies via `pip install -e ".[test]"`.
- Run the relevant checks with `pytest`, including the integration suite (`tests/integrational/test_gir_scenes.py`).
