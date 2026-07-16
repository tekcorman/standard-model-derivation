#!/usr/bin/env python3
"""
proofs/foundations/ML1d_selftest_regression_2026-07-13.py

[PUSH 3, HYGIENE -- W1 verify.py integration batch; ARCHITECT-CORRECTED FAST FORM 2026-07-13]

FAST verify-suite regression for the_net.py Section 9 (ML-1d) + Section 9b (ML-1d-b): calls the
UNDERLYING section functions directly at small sizes (architect instruction at the W1-integration
batch: the full ml1d_selftest_2026_07_12 / ml1db_selftest_2026_07_13 chain re-runs EVERY prior net
section's self-test -- incl. the minutes-heavy A2c/A2d weld checks, twice, since 9b calls 9
internally -- which is station-scale work, not suite regression; those full self-tests remain the
regression anchors as-written in the_net.py, unmodified, and are exercised by their own station
lineage). This file exercises BOTH sections structurally, well under 60 s.

This file is deliberately NOT proofs/foundations/ML1d_derived_horizon_2026-07-12.py (the full
ML-1d-b STATION runnable): that file's R2 hard gate exits 1 BY DESIGN on the amended instrument's
own second gate failure (INSTRUMENT-CLOSED, per internal research notes)
-- a station VERDICT, not a regression failure; wiring it would make a permanently-red suite entry.
The L=800 benchmark legs likewise belong to station runs, not suite regression, and are skipped
here (mirror_mode_pairing is checked on a SMALL genuine open chain instead).

WHAT THIS CHECKS (read-only imports; the_net.py is never modified):
  Section 9:  mirror_mode_pairing purity/mirror-norm identities on a genuine L=100 open-chain
              Dirac-sea projector · mode_position_expectation synthetic · near_surface_selection
              half-count/ordering · project_matrix orthonormal-U identity · k_boost_bond_matrix
              placement-shift linearity · boost_c_star planted-c0=2pi recovery ·
              distinct_spectrum_count on the L0b-shaped synthetic (+ growth) ·
              diamond_vertex_region nonempty/subset/growth on a small Patch.
  Section 9b: entanglement_carrying_selection boundary cases · k_boost_parabolic reflection
              symmetry + x->0 Rindler limit vs the linear template · positive_spectrum_dimension
              on the REAL 3-edge triangle region (the W3a forced zero mode) + an even synthetic.
  Plus the two permanent cheap anchors (anchor_cell_projector, anchor_tick_2pi).

Does NOT adjudicate any station verdict. Exits 0 iff all checks pass.
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402  the ONE master Layer-3 object; nothing rebuilt here

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    status = "PASS" if cond else "FAIL"
    if cond:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return cond


print("=" * 78)
print("ML-1d / ML-1d-b FAST REGRESSION (Section 9 + 9b functions directly, small sizes)")
print("=" * 78)

# --- the two permanent cheap anchors -----------------------------------------
check("anchor_cell_projector (M0 cell C-projector rank-3)", net.anchor_cell_projector())
check("anchor_tick_2pi (tick modular flow compact U(1), period 2pi)", net.anchor_tick_2pi())

# --- Section 9 ----------------------------------------------------------------
print("-" * 78)
print("Section 9 (ML-1d) readers, direct small-size checks")
print("-" * 78)

# mirror_mode_pairing: genuine finite Dirac-sea projector on a SMALL open chain (L=100, not 800).
Nc = 100
Hc = np.diag(np.ones(Nc - 1), 1) + np.diag(np.ones(Nc - 1), -1)
Ec, Vc = np.linalg.eigh(Hc)
cols = Vc[:, Ec < 0.0]
Cfull = cols @ cols.conj().T
A_idx = np.arange(0, Nc // 2)
Ap_idx = np.arange(Nc // 2, Nc)
C_A = Cfull[np.ix_(A_idx, A_idx)]
C_ApA = Cfull[np.ix_(Ap_idx, A_idx)]
mp = net.mirror_mode_pairing(C_A, C_ApA)
check("mirror_mode_pairing purity/mirror-norm identities on a genuine L=100 open-chain "
      "Dirac-sea projector",
      mp["mirror_norm_residual"] < 1e-6 and mp["purity_residual"] < 1e-6,
      f"mirror_norm_residual={mp['mirror_norm_residual']:.2e}, "
      f"purity_residual={mp['purity_residual']:.2e}")

# mode_position_expectation: synthetic sanity check.
vtest = np.zeros((5, 2))
vtest[3, 0] = 1.0
vtest[:, 1] = 1.0
postest = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
xe = net.mode_position_expectation(vtest, postest)
check("mode_position_expectation: localized mode -> exact position; uniform -> mean",
      abs(xe[0] - 3.0) < 1e-12 and abs(xe[1] - 2.0) < 1e-12, f"{xe}")

# near_surface_selection: half-count, smallest |eps| kept.
eps_test = np.array([5.0, -0.1, 3.0, 0.2, -4.0, 0.05])
idx_ns = net.near_surface_selection(eps_test, frac=0.5)
check("near_surface_selection: half-count, correct (smallest-|eps|) members",
      len(idx_ns) == 3 and set(idx_ns.tolist()) == {1, 3, 5}, f"idx={idx_ns}")

# project_matrix: orthonormal-U identity spot-check.
rng = np.random.default_rng(0)
Mtest = rng.normal(size=(6, 6))
Mtest = Mtest + Mtest.T
Qtest, _ = np.linalg.qr(rng.normal(size=(6, 6)))
Utest = Qtest[:, :3]
check("project_matrix == direct U^dagger.M.U",
      np.allclose(net.project_matrix(Mtest, Utest), Utest.T @ Mtest @ Utest, atol=1e-12))

# k_boost_bond_matrix: the placement-shift linearity identity the station relies on.
HAtest = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
xtest = np.array([0.0, 1.0, 2.0])
K0 = net.k_boost_bond_matrix(HAtest, xtest)
delta = 0.37
check("k_boost_bond_matrix: K_boost(delta) == K_boost(0) - delta*H_A EXACTLY",
      np.allclose(net.k_boost_bond_matrix(HAtest, xtest - delta), K0 - delta * HAtest,
                  atol=1e-12))

# boost_c_star: recovers a planted c0 = 2pi on h = c0*K + small noise.
c0 = 2 * math.pi
noise = 1e-6 * rng.normal(size=K0.shape)
noise = noise + noise.T
c_star, resid = net.boost_c_star(c0 * K0 + noise, K0)
check("boost_c_star recovers a planted c0=2pi",
      abs(c_star - c0) < 1e-3 and resid < 1e-3,
      f"c*={c_star:.6f}, residual={resid:.2e}")

# distinct_spectrum_count: the L0b 3-edge shape (0, +-eps cluster) + growth.
eps_l0b = np.array([0.0, 0.7, -0.7, 0.7000001, -0.6999998])
dcount = net.distinct_spectrum_count(eps_l0b, rel_tol=1e-3)
check("distinct_spectrum_count: the L0b 3-edge shape -> 2 distinct magnitudes",
      dcount == 2, f"got {dcount}")
eps_grown = np.concatenate([eps_l0b, [1.4, -1.4, 2.1, -2.1]])
dcount2 = net.distinct_spectrum_count(eps_grown, rel_tol=1e-3)
check("distinct_spectrum_count GROWS when well-separated magnitudes are added",
      dcount2 > dcount, f"{dcount} -> {dcount2}")

# diamond_vertex_region: nonempty, subset of the ambient patch, grows with depth (small Patch).
patch_d = net.Patch(M=6)
base_d = patch_d.central_dart()
Nv_all = len(patch_d.vertex_adjacency()[1])
reg2 = net.diamond_vertex_region(patch_d, base_d, 2)
reg4 = net.diamond_vertex_region(patch_d, base_d, 4)
check("diamond_vertex_region: nonempty subsets of the ambient patch, grow with depth",
      0 < len(reg2) <= len(reg4) < Nv_all
      and set(reg2.tolist()) <= set(range(Nv_all)) and set(reg4.tolist()) <= set(range(Nv_all)),
      f"|reg(2)|={len(reg2)}, |reg(4)|={len(reg4)}, N_ambient={Nv_all}")

# --- Section 9b ----------------------------------------------------------------
print("-" * 78)
print("Section 9b (ML-1d-b) corrected definitions, direct checks")
print("-" * 78)

# entanglement_carrying_selection: saturated excluded, fractional kept, boundary cases.
lam_test = np.array([1e-12, 0.3, 0.5, 1 - 1e-12, 1e-7, 1.0, 0.0, 1 - 1e-7])
sel = net.entanglement_carrying_selection(lam_test)
check("entanglement_carrying_selection keeps exactly the fractional modes",
      set(sel.tolist()) == {1, 2, 4, 7}, f"idx={sel.tolist()}")

# k_boost_parabolic: reflection symmetry + x->0 Rindler limit vs the linear template.
ell = 10.0
xa = np.array([0.0, 4.0, 10.0])
check("k_boost_parabolic: reflection symmetry w(m)=w(ell-m) elementwise",
      np.allclose(net.k_boost_parabolic(HAtest, xa, ell),
                  net.k_boost_parabolic(HAtest, ell - xa, ell), atol=1e-12))
x_small = np.array([0.0, 1e-6, 2e-6])
check("k_boost_parabolic: x->0 behavior == the linear (Rindler) template to O(x/ell)",
      np.allclose(net.k_boost_parabolic(HAtest, x_small, ell),
                  net.k_boost_bond_matrix(HAtest, x_small), rtol=1e-5, atol=1e-18))

# positive_spectrum_dimension: the REAL 3-edge triangle region (W3a forced zero mode) + even.
C6 = net.vacuum_covariance(sign=+1)
_, eps3, _ = net.region_data(C6, [0, 1, 3])
psd = net.positive_spectrum_dimension(eps3, region_dim=3)
check("positive_spectrum_dimension on the 3-edge triangle: n_positive=1, forced zero mode",
      psd["n_positive"] == 1 and psd["has_zero_mode"] and psd["zero_forced_by_odd_dim"],
      f"n_positive={psd['n_positive']}, min|eps|={psd['min_abs_eps']:.1e}")
psd2 = net.positive_spectrum_dimension(np.array([-2.0, -1.0, 1.0, 2.0]), region_dim=4)
check("positive_spectrum_dimension on an even no-zero spectrum: n_positive=2, no zero mode",
      psd2["n_positive"] == 2 and not psd2["has_zero_mode"]
      and not psd2["zero_forced_by_odd_dim"])

# --- summary -------------------------------------------------------------------
print("=" * 78)
print(f"SUMMARY: {passed} PASS / {failed} FAIL  (total {passed + failed})")
print("RESULT:", "ML-1d/ML-1d-b FAST REGRESSION PASSES" if failed == 0
      else "A CHECK FAILED -- inspect above")
print("=" * 78)
sys.exit(1 if failed else 0)
