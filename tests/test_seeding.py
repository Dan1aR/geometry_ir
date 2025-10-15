import numpy as np
import pytest

from geoscript_ir.desugar import desugar
from geoscript_ir.parser import parse_program
from geoscript_ir.solver import initial_guess, translate
from geoscript_ir.seeders import GraphMDSSeeder, SobolSeeder
from geoscript_ir.validate import validate


def _build_model(text: str):
    program = parse_program(text)
    validate(program)
    desugared = desugar(program)
    return translate(desugared)


def test_initial_guess_graph_mds_deterministic():
    model = _build_model(
        """
        scene "Triangle"
        points A, B, C
        segment A-B [length=4]
        segment A-C [length=3]
        segment B-C [length=5]
        """
    )

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    guess1 = initial_guess(model, rng1, 0)
    guess2 = initial_guess(model, rng2, 0)

    assert guess1.shape == (2 * len(model.points),)
    assert np.allclose(guess1, guess2)

    anchor = model.gauge_anchor or model.points[0]
    idx = model.index[anchor] * 2
    assert guess1[idx] == pytest.approx(0.0, abs=1e-9)
    assert guess1[idx + 1] == pytest.approx(0.0, abs=1e-9)


def test_initial_guess_fallback_sobol_deterministic():
    model = _build_model(
        """
        scene "Two"
        points A, B
        """
    )

    graph = GraphMDSSeeder()
    sobol = SobolSeeder()

    rng = np.random.default_rng(0)
    assert graph.seed(model, rng, 0) is None

    rng_a = np.random.default_rng(987)
    rng_b = np.random.default_rng(987)
    guess_a = initial_guess(model, rng_a, 0)
    guess_b = initial_guess(model, rng_b, 0)

    assert np.allclose(guess_a, guess_b)
    assert guess_a.shape == (2 * len(model.points),)

    rng_c = np.random.default_rng(987)
    sobol_direct = sobol.seed(model, rng_c, 0)
    assert sobol_direct is not None
    assert np.allclose(guess_a, sobol_direct)

    rng_d = np.random.default_rng(555)
    rng_e = np.random.default_rng(555)
    alt_a = initial_guess(model, rng_d, 1)
    alt_b = initial_guess(model, rng_e, 1)
    assert np.allclose(alt_a, alt_b)
