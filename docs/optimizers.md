# Robust Gradient-Based Solver (Adam + LM Hybrid)

This project now includes a robust, gradient-based solver that complements the existing SciPy
`least_squares` pipeline. The new solver uses an **Adam** optimizer on a **scalar robust loss**
built from the same residual blocks the system already assembles, then finishes with a classic
LM/TRF step for fast local convergence.

## High-level algorithm

For each desugared **Model**:

1. **Variables = free point coordinates** (points that are not deterministically derivable by DDC).
2. At each evaluation, we **derive** deterministic points from the current variables via the existing
   derivation plan (DDC-in-the-loop), then evaluate all residuals.
3. We compute a robust scalar objective using `soft_l1`/`huber` or `linear`.
4. We **optimize** the scalar objective using a simple **Adam** loop with gradient clipping and
   early stopping. Gradients are obtained via finite-difference on the scalar loss, which works
   with all existing residuals without re-implementing them for autograd.
5. A multi-stage schedule gradually reduces hinge smoothing `sigma` and switches to a precise
   local solver (`least_squares` with LM/TRF) for final polishing.

This approach greatly improves **basin-escape** while keeping compatibility with the current residual
library and plan derivations. It works for any well-posed GeoScript scene.

## Enabling from CLI

```bash
python -m geoscript_ir \
  --input path/to/scene.gir \
  --loss-mode \
  --adam-lr 0.05 \
  --adam-steps 800 \
  --robust soft_l1 \
  --sigma 0.05
```

If `--loss-mode` is omitted, the classic `least_squares` path is used.

## Enabling from Python

```python
from geoscript_ir.solver import SolveOptions, LossModeOptions, solve, translate
options = SolveOptions(enable_loss_mode=True, tol=1e-6, random_seed=0)
loss_opts = LossModeOptions(enabled=True)  # tweak schedule if desired
solution = solve(model, options, loss_opts=loss_opts)
```

## Notes

- Hinge-like residuals are smoothed during early Adam stages using `sigma` to avoid flat regions.
- Adam has gradient clipping, early stopping and multistart (reusing the existing seeding logic).
- You can still tune the final `least_squares` method (`--method`), robust loss, and tolerances.
- The implementation purposefully **reuses the current residual builder** and derivation plan; no
  duplication of geometry rules is required.
