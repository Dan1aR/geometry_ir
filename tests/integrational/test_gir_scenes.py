from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pytest

from geoscript_ir import parse_program, validate
from geoscript_ir.ddc import evaluate_ddc
from geoscript_ir.solver import SolveOptions, solve_with_desugar_variants
from geoscript_ir.ddc import derive_and_check
from geoscript_ir.solver import Solution, VariantSolveResult


DATA_DIR = Path(__file__).resolve().parent / "gir"
ARTIFACT_ROOT = Path("/tmp/geoscript_tests")


@dataclass
class IntegrationCase:
    case_id: str
    source: str
    expect_success: bool = True
    expected_targets: Optional[Dict[str, float]] = None
    tol_solver: float = 1e-8
    tol_ddc: Optional[float] = None
    allow_ambiguous: bool = False

    @property
    def solver_options(self) -> SolveOptions:
        return SolveOptions(
            random_seed=123,
            reseed_attempts=10,
            tol=self.tol_solver,
        )


def _load_case_overrides(path: Path) -> Dict[str, object]:
    overrides_path = path.with_suffix(".json")
    if overrides_path.exists():
        with overrides_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"Overrides for {path.name} must be a JSON object")
            return data
    return {}


def _iter_cases() -> Iterable[IntegrationCase]:
    for scene_path in sorted(DATA_DIR.glob("*.gir")):
        text = scene_path.read_text()
        overrides = _load_case_overrides(scene_path)
        allow_ambiguous = bool(overrides.get("allow_ambiguous", False))
        tol_solver = float(overrides.get("tol_solver", 1e-8))
        tol_ddc = overrides.get("tol_ddc")
        if tol_ddc is not None:
            tol_ddc = float(tol_ddc)
        expect_success = bool(overrides.get("expect_success", True))
        expected_targets = overrides.get("expected_targets")
        if expected_targets is not None and not isinstance(expected_targets, dict):
            raise ValueError(
                f"expected_targets for {scene_path.name} must be a mapping"
            )
        case = IntegrationCase(
            case_id=scene_path.stem,
            source=text,
            expect_success=expect_success,
            expected_targets=expected_targets,
            tol_solver=tol_solver,
            tol_ddc=tol_ddc,
            allow_ambiguous=allow_ambiguous,
        )
        yield case


def _compute_target_lengths(program, solution: Solution) -> Dict[str, float]:
    lengths: Dict[str, float] = {}
    coords = solution.point_coords
    for stmt in program.body:
        if stmt.kind != "target_length":
            continue
        a, b = stmt.data["edge"]
        if a not in coords or b not in coords:
            continue
        ax, ay = coords[a]
        bx, by = coords[b]
        key = f"length:{a}-{b}"
        lengths[key] = math.hypot(bx - ax, by - ay)
    return lengths


def _write_artifacts(
    case: IntegrationCase,
    variant: VariantSolveResult,
    solution: Solution,
    report,
    message: str,
) -> None:
    case_dir = ARTIFACT_ROOT / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / f"{case.case_id}.ddc.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_lines = [
        f"Case: {case.case_id}",
        f"Solver success: {solution.success} (residual={solution.max_residual:.3e})",
        f"DDC status: {report.get('status')}",
        f"Message: {message}",
    ]
    (case_dir / f"{case.case_id}.summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    _render_scene_plot(
        case_dir / f"{case.case_id}.scene.png",
        variant.program,
        solution,
        report,
        case.case_id,
    )


_PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc````\x00\x00\x00\x05"
    b"\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _render_scene_plot(
    path: Path,
    program,
    solution: Solution,
    report,
    case_id: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        path.write_bytes(_PLACEHOLDER_PNG)
        return

    coords = solution.point_coords
    mismatched = {name for name, info in report.get("points", {}).items() if info.get("match") == "no"}
    ambiguous = {
        name
        for name, info in report.get("points", {}).items()
        if len(info.get("candidates", []) or []) > 1 and info.get("match") == "yes"
    }

    xs = [coord[0] for coord in coords.values()]
    ys = [coord[1] for coord in coords.values()]
    min_x, max_x = min(xs, default=0.0), max(xs, default=1.0)
    min_y, max_y = min(ys, default=0.0), max(ys, default=1.0)
    span = max(max_x - min_x, max_y - min_y, 1.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    for name, (x, y) in coords.items():
        if name in mismatched:
            color = "red"
        elif name in ambiguous:
            color = "orange"
        else:
            color = "#1f77b4"
        ax.scatter([x], [y], c=color, s=30)
        ax.text(x, y, name, fontsize=8, ha="left", va="bottom")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_x - 0.1 * span, max_x + 0.1 * span)
    ax.set_ylim(min_y - 0.1 * span, max_y + 0.1 * span)
    ax.set_title(case_id)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@pytest.mark.parametrize("case", list(_iter_cases()), ids=lambda case: case.case_id)
def test_gir_scene_passes_solver_and_ddc(case: IntegrationCase) -> None:
    program = parse_program(case.source)
    validate(program)

    variant = solve_with_desugar_variants(program, case.solver_options)
    solution = variant.solution

    if case.expect_success:
        assert solution.success, f"{case.case_id} solver failed"
        assert solution.max_residual <= case.tol_solver, (
            f"{case.case_id} exceeded residual threshold: {solution.max_residual}"
        )
    else:
        assert not solution.success or solution.max_residual > case.tol_solver

    report = derive_and_check(variant.program, solution, tol=case.tol_ddc)
    ddc_result = evaluate_ddc(report, allow_ambiguous=case.allow_ambiguous)

    if case.expected_targets:
        computed_lengths = _compute_target_lengths(variant.program, solution)
        for key, expected_value in case.expected_targets.items():
            assert key in computed_lengths, f"Missing target {key} in {case.case_id}"
            actual_value = computed_lengths[key]
            assert math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=case.tol_ddc or 1e-6), (
                f"{case.case_id}: target {key} expected {expected_value}, got {actual_value}"
            )

    if ddc_result.severity != "ok":
        _write_artifacts(case, variant, solution, report, ddc_result.message)

    assert ddc_result.passed, ddc_result.message


