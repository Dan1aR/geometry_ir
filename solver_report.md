# Solver Diagnostics Report

Generated from `solver_diagnostics.json` after running `pytest` on the GIR suite (random seed 123, 10 reseed attempts, tol=1e-8).

## Integration Cases

### circle
- **Status:** solver failed with max residual 7.55 despite 12 reseeds; best attempt was #9 (Sobol seeder).
- **Residual hotspots:** `diameter_midpoint(O;C-D)` at 7.55, `min_separation(C-D)` at 2.56, and circle-point residuals at 0.25, so the diameter and separation constraints never come close to satisfied.
- **Seeding notes:** global length hints moved point B by 10.76 (CB=13) and 5.83 (AB=16), overwhelming the initial layout.
- **Takeaway:** length hints reposition B far outside the circle before solving; the optimizer then cannot recover the intended diameter/separation geometry. We likely need diameter-aware seeding or to cap aggressive global length moves on circle-driven scenes.

### circle_tangent_lines_2
- **Status:** solver converged (max residual 8.3e-9) but DDC flagged a mismatch for tangent points T1/T2 (7.79 units away from the derived positions).
- **Residual hotspots:** only gauge/length/tangent constraints at noise level, so numerically the model is satisfied.
- **Seeding notes:** global length hints stretched OK by 3.5 units and OA by 8, indicating our seed ignores the circle radius and drags points along declared lengths.
- **Takeaway:** even with near-zero residuals, the derived tangent construction disagrees with DDC. Likely our orientation or tangent disambiguation picks the opposite tangent branch; need to inspect derivation plan / DDC expectations for these points.

### tikz_101
- **Status:** solver failed (max residual 0.116); best attempt #3 via Sobol.
- **Residual hotspots:** dominated by `min_separation` (A–C, B–F, A–D) and `point_on_segment` violations ~0.08, so the equal subdivision of AB never materializes.
- **Seeding notes:** no projection/global hints triggered; the seed never enforced segment division before solving.
- **Takeaway:** we need stronger per-point projection for equal-segment chains—currently both seeder families ignore those hints, leaving optimizer in a bad basin.

### tikz_104
- **Status:** solver failed (max residual 0.091); best attempt #10 (Sobol).
- **Residual hotspots:** 45°/30° angle constraints off by ~0.07–0.09 rad, foot guard also active (5e-3).
- **Seeding notes:** no projections/hints recorded.
- **Takeaway:** without angular projections, the optimizer never reaches the acute configuration; we should inject angle-based seed adjustments or better initial triangle placement.

### tikz_107
- **Status:** solver failed (max residual 7.32); best attempt #4 (Sobol).
- **Residual hotspots:** huge foot guard on H, and segment lengths AB, BH, AC miss targets by 4–6 units.
- **Seeding notes:** length hints moved B by 5.93, C by 2.23, H by 2.46—seeding is enforcing lengths but not the altitude construction.
- **Takeaway:** length hints alone pull the configuration into a stretched triangle; we need altitude-aware projections or to include foot constraints during seeding to avoid violent guard penalties.

### tikz_108
- **Status:** solver nearly succeeded but stalled at residual 9.07e-7 (>1e-8 tolerance); best attempt #12 (Sobol).
- **Residual hotspots:** `min_separation(B-C)` at 9.07e-7 plus segment lengths off by ~2e-7.
- **Seeding notes:** length hints made huge moves (EB by 10.95, CE by 4.87) and a concyclicity hint shifted all four points up to 0.80.
- **Takeaway:** after aggressive hint moves, B and C land nearly coincident, so the min-separation guard blocks convergence. Need to cap hint adjustments or reconsider how we enforce equal chords on this circle.

### tikz_177
- **Status:** solver failed (max residual 0.219); best attempt #2 (Sobol).
- **Residual hotspots:** foot guards for H_AD and H_AB ~0.219, matching foot residuals around 0.13, indicating altitude feet far from the lines.
- **Seeding notes:** hints applied equal-length and length constraints (moves up to 2.46), but no projection kept the feet on their carriers.
- **Takeaway:** despite aligning edge lengths, the seed leaves orthogonal drops misaligned; need perpendicular projections for these feet before solving.

### tikz_183
- **Status:** solver failed disastrously (max residual 26.2); best attempt #7 (Sobol).
- **Residual hotspots:** `point_on_segment_bounds(F,A-D)` at 26, right angle & foot guard at 17, equal segments at 17, FD length off by 15.
- **Seeding notes:** no projections or hints triggered.
- **Takeaway:** the seed never respects the constructed right angle/point on segment, so solver is far outside feasible region; we need to honor perpendicular and segment-bound hints in seeding.

### tikz_187
- **Status:** solver failed (max residual 66.7); best attempt #3 (Sobol).
- **Residual hotspots:** `point_on_circle(T,O)` and right angle at T both ~66, foot guards at 33, length CC1=11 off by 23.9.
- **Seeding notes:** no hint/projection activity.
- **Takeaway:** tangent-circle configuration is wildly wrong; without circle/tangent projections, optimization never recovers.

### tikz_22
- **Status:** solver failed (max residual 5.19e-3); best attempt #9 (Sobol).
- **Residual hotspots:** foot guard for H on BC at 5.19e-3, plus min separation FH and repeated point-on-segment residuals ~1.6e-3.
- **Seeding notes:** no hints triggered.
- **Takeaway:** failure is small but persistent—foot placement remains slightly off; may need more solver iterations or better projection accuracy for repeated point-on-segment constraints.

### tikz_31
- **Status:** solver failed (max residual 8.85e-3); best attempt #1 (GraphMDS).
- **Residual hotspots:** `point_on_line(C,P-Q)` at 8.85e-3 and multiple min-separation guards 7–8e-3.
- **Seeding notes:** only one projection event; no hints.
- **Takeaway:** the GraphMDS seed leaves the orthic foot chain collapsed; need to enforce line projections or strengthen min-separation handling.

### tikz_48
- **Status:** solver failed (max residual 30.1); best attempt #5 (Sobol).
- **Residual hotspots:** foot guard on C->H_AB hits 30, lengths AB/BC/CH_AB miss by 16, 22, 16.
- **Seeding notes:** no hints activated.
- **Takeaway:** seeding ignores trapezoid structure entirely; without projecting H_AB onto AB the solver flails.

### tikz_69
- **Status:** solver failed (max residual 14.96); best attempt #9 (Sobol).
- **Residual hotspots:** `point_on_segment_bounds(E,C-D)` at 14.96, DE length off by 3.96, EC by 0.834.
- **Seeding notes:** no hints.
- **Takeaway:** again we never projected E onto CD, so solver starts far off and cannot satisfy simultaneous length/segment constraints.

### tikz_73
- **Status:** solver failed (max residual 31.9); best attempt #9 (Sobol).
- **Residual hotspots:** min-separation constraints for point O vs A/C at 31, AC length off by 15.9, area guard 12.3.
- **Seeding notes:** no hints.
- **Takeaway:** the parallelogram/circle overlay collapses O onto existing vertices; min-separation guard reveals we need global structure-aware seeding.

### tikz_92
- **Status:** solver failed (max residual 2.56); best attempt #2 (Sobol).
- **Residual hotspots:** `min_separation(A-B)` at 2.56 and an edge floor of 0.102—points A and B collapse despite guard.
- **Seeding notes:** two projection events and a length hint moving A by 15 (OA=16).
- **Takeaway:** large length hints stretch OA while min-separation tries to keep A/B apart; we need to reconcile guard floors with heavy-length adjustments.

### trapezoid_hard
- **Status:** solver failed (max residual 0.026); best attempt #2 (Sobol).
- **Residual hotspots:** trapezoid convexity at 0.026, area guard 3.3e-3.
- **Seeding notes:** length hints moved D by 10.40 and M by 3.05, distorting the trapezoid before solving.
- **Takeaway:** enforcing raw lengths without respecting convexity pushes the seed into a self-intersecting shape, triggering convexity guard. Need better handling of trapezoid gauges and length hints.

### triangle_points
- **Status:** solver converged (2.56e-11), but DDC reports point D mismatched by 0.14 units.
- **Residual hotspots:** all residuals are tiny; issue is DDC disagreement, not solver failure.
- **Seeding notes:** equal-length hints moved A by 1.0 and C by 0.226—large but consistent with enforcing medians.
- **Takeaway:** despite numerical success, the derived point mapping disagrees with DDC (likely due to variant choice or ambiguous target). Need to inspect the derivation outputs for point D.

## Unit Test Regression

### `test_initial_guess_respects_gauge_and_jitter`
- Current implementation keeps the gauge edge (AB) fixed at (1,0) for all attempts; computed jitter distances are `diff01=1.818` and `diff12=1.113`, violating the monotonic-increase assertion.
- The layout-scale rescaling in `align_gauge` forces identical AB coordinates each time, so later Sobol attempts reuse the same baseline and only jitter point C. Because jitter variance is constant, the second jitter step moved C less than the first, failing the test. Need to revisit jitter scheduling or adjust the test expectation after rescaling change.
