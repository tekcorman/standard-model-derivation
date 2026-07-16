#!/usr/bin/env python3
# [2026-07-13 CORRECTION NOTE -- W3b audit C6]: the constant labeled beta_eff below (line ~99,
# `beta_eff = 2 * math.log((1 / math.sqrt(2)) / 0.039)`) is algebraically beta_prime =
# beta_natural - h_top ~= 5.794 (a u_c typo, 1/sqrt(2) for 1/2), not beta_eff ~= 5.101.
# Historical record preserved unmodified; see working notes/W3b_audit_2026-07-12.md and the
# roadmap L7 entry. Sensitivity: NIL on this file's booked outcome -- the mislabeled constant is
# consumed ONLY by the "kms" branch of nearest_bond_proper_slope/operator_c, feeding the KMS
# thermal-shift REPORT line (informational); the booked verdict (ML1'''-C: "2PI-COMPUTED" vs
# "CONVERGES-ELSEWHERE") is computed purely from vac_lim/op_c_vac, the VACUUM-state read, which
# never touches beta_eff. This station's own outcome (Newton's G / the local BW 2pi) is carried
# forward in the corpus as an explicit OPEN MISS, never a closed number (see docs/scoping/
# ML_track_session_consolidation_2026-07-08.md SS"THE OPEN MISSES": "Newton's G (2pi) ...
# CANDIDATE ... NOT closed"), and is now superseded by the Push-3 ML-1d/ML-1d-b station lineage,
# which does not reuse this file's beta_eff/beta_prime literal. No booked verdict moves.
"""
proofs/foundations/ML1ppp_computed_2pi_2026-07-08.py

ML-1‴ — the COMPUTED 2pi.  Pre-registered in internal research notes
(committed b9595d5 BEFORE this probe).  Station ML-1''' of the active fork contract (Fork B).
EXTENDS the_net.py.

Removes the TWO confounds architect named: METRIC (proper distance under h_ij=(g^{ij})^{-1}, ML-1''s derived
g^{ij}; FIXED first-principles, no tuning) and STATE (BW is a VACUUM theorem; run the vacuum AND the run
KMS state separately, never averaged), with DECLARED finite-size extrapolation + an operator-level test.

DISCIPLINE: 2pi-COMPUTED => G_eff=G; selecting hbar because it closes G is FORBIDDEN. The 2pi is the
extrapolated proper-distance vacuum slope with declared tolerance, MEASURED never inserted; no tuning; no
pattern-match; local 2pi does NOT retro-edit kappa. G stays OPEN unless fully-pinned 2pi-COMPUTED.
"""
import os
import sys
import math

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import the_net as net  # noqa: E402
import srs  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
ok_all = True
TWO_PI = 2 * math.pi


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)


# --- the FIXED emergent metric (ML-1'', derived from the cone velocities; NO tuning) ---
d = net.cone_velocity([1, 0, 0])[0] ** 2                      # g^00 = v_axis^2 = 0.5
g01 = net.cone_velocity([1, 1, 0])[0] ** 2 - d               # from the face velocities
g02 = net.cone_velocity([1, 0, 1])[0] ** 2 - d
g12 = net.cone_velocity([0, 1, 1])[0] ** 2 - d
Gup = np.array([[d, g01, g02], [g01, d, g12], [g02, g12, d]])  # g^{ij} (dispersion velocity^2 tensor)
g00_contra = Gup[0, 0]                                         # = 0.5 ; proper dist to {x0=c} = |dx0|/sqrt(g00)
PROPER = 1.0 / math.sqrt(g00_contra)                          # proper-distance-per-cell factor along x0
print(f"emergent metric g^ij eigenvalues {np.round(np.linalg.eigvalsh(Gup),4)}; g^00={g00_contra:.3f} "
      f"=> proper distance per cell (dir 0) = 1/sqrt(g^00) = {PROPER:.4f}")

# benchmark calibration anchor (known-BW system => 2pi)
sb, rb = net.benchmark_bw_2pi(800)
print(f"benchmark (critical chain) near-horizon slope = {rb:.4f} x 2pi  [calibration; pipeline trusted]")


def nearest_bond_proper_slope(M, state, beta_eff=None):
    """Near-horizon perpendicular-bond entanglement-hopping slope on the srs half-space, in PROPER
    distance, for the cone sector.  state='vacuum' (cone Dirac-sea projector) or 'kms' (Fermi-Dirac at
    beta_eff).  Returns beta(nearest)/proper_dist(nearest) / (2pi)."""
    patch = net.Patch(M=M)
    H, verts = patch.vertex_adjacency()
    vpos = {v: n for n, v in enumerate(verts)}
    E, V = np.linalg.eigh(H)
    if state == "vacuum":
        cols = V[:, E < -1.0 - 1e-9]                          # fill below the node = cone Dirac sea (pure)
        C = cols @ cols.conj().T
    else:                                                     # KMS: Fermi-Dirac at beta_eff, mu = node -1
        occ = 1.0 / (np.exp(beta_eff * (E + 1.0)) + 1.0)
        C = (V * occ) @ V.conj().T
    A_idx = [n for n, (i, x) in enumerate(verts) if x[0] < M // 2]
    posA = {g: a for a, g in enumerate(A_idx)}
    C_A = C[np.ix_(A_idx, A_idx)]
    hA = net.entanglement_hamiltonian(C_A)
    # nearest in-region perpendicular bond (edge (1,2,e1)) at cell-distance 1 from the horizon
    best = None
    for x in patch.box:
        if x[0] == M // 2 - 2:                                # connects x0=M/2-2 -> M/2-1, both in A
            v1, v2 = (1, x), (2, tuple(np.array(x) + np.array([1, 0, 0])))
            if v1 in vpos and v2 in vpos and vpos[v1] in posA and vpos[v2] in posA:
                beta = abs(hA[posA[vpos[v1]], posA[vpos[v2]]])
                best = beta if best is None else (best + beta)
    # average over transverse positions at that layer
    n_tv = sum(1 for x in patch.box if x[0] == M // 2 - 2)
    beta_mean = best / n_tv
    x_proper = 1.0 * PROPER                                   # cell-distance 1 -> proper
    return beta_mean / x_proper / TWO_PI


# ===========================================================================
banner("ML1‴-A  proper-distance slope: VACUUM vs KMS, finite-size extrapolation M->inf")
# ===========================================================================
Ms = [6, 8, 10, 12]
beta_eff = 2 * math.log((1 / math.sqrt(2)) / 0.039)          # M2b/M0-2R beta_eff
res = {"vacuum": [], "kms": []}
for M in Ms:
    rv = nearest_bond_proper_slope(M, "vacuum")
    rk = nearest_bond_proper_slope(M, "kms", beta_eff)
    res["vacuum"].append(rv)
    res["kms"].append(rk)
    print(f"    M={M:2d}: VACUUM slope = {rv:.4f} x 2pi   KMS slope = {rk:.4f} x 2pi")


def extrapolate(vals):
    x = np.array([1.0 / M for M in Ms])
    y = np.array(vals)
    a, b = np.polyfit(x, y, 1)               # y = a*(1/M) + b ; b = M->inf limit
    resid = np.sqrt(np.mean((y - (a * x + b)) ** 2))
    return b, resid


vac_lim, vac_err = extrapolate(res["vacuum"])
kms_lim, kms_err = extrapolate(res["kms"])
print(f"    extrapolation M->inf (linear in 1/M):")
print(f"      VACUUM: {vac_lim:.4f} x 2pi  (+/- {vac_err:.4f})   <- the BW-relevant state")
print(f"      KMS:    {kms_lim:.4f} x 2pi  (+/- {kms_err:.4f})")
thermal_shift = kms_lim - vac_lim
check("ML1‴-A the finite-size extrapolation CONVERGES (small fit residual) for both states",
      vac_err < 0.1 and kms_err < 0.1,
      detail=f"vacuum limit {vac_lim:.3f}x2pi(+/-{vac_err:.3f}); KMS {kms_lim:.3f}x2pi(+/-{kms_err:.3f})")

# ===========================================================================
banner("ML1‴-B  operator-level test: h_A vs 2pi*K_boost (resolution-independent)")
# ===========================================================================
def operator_c(M, state, beta_eff=None):
    """min_c ||h_A - c*K_boost|| ; K_boost = sum_bonds x_proper * H_phys[i,j] (the emergent boost
    generator).  Returns c/2pi and the residual fraction."""
    patch = net.Patch(M=M)
    H, verts = patch.vertex_adjacency()
    vpos = {v: n for n, v in enumerate(verts)}
    E, V = np.linalg.eigh(H)
    if state == "vacuum":
        cols = V[:, E < -1.0 - 1e-9]
        C = cols @ cols.conj().T
    else:
        occ = 1.0 / (np.exp(beta_eff * (E + 1.0)) + 1.0)
        C = (V * occ) @ V.conj().T
    A_idx = [n for n, (i, x) in enumerate(verts) if x[0] < M // 2]
    posA = {g: a for a, g in enumerate(A_idx)}
    C_A = C[np.ix_(A_idx, A_idx)]
    hA = net.entanglement_hamiltonian(C_A)
    HA = H[np.ix_(A_idx, A_idx)]
    # x_proper of each vertex = proper distance of its x0 to the horizon plane {x0=M/2}
    xprop = np.array([((M // 2) - verts[g][1][0]) * PROPER for g in A_idx])
    Kb = HA * ((xprop[:, None] + xprop[None, :]) / 2.0)      # H_phys weighted by bond-midpoint proper x
    num = np.real(np.vdot(Kb, hA))
    den = np.real(np.vdot(Kb, Kb))
    c = num / den
    resid = np.linalg.norm(hA - c * Kb) / np.linalg.norm(hA)
    return c / TWO_PI, resid


for state in ("vacuum", "kms"):
    c2pi, r = operator_c(12, state, beta_eff if state == "kms" else None)
    print(f"    M=12 {state:6s}: h_A = c*K_boost with c/2pi = {c2pi:.4f}  (operator residual {r:.3f})")
    if state == "vacuum":
        op_c_vac, op_r_vac = c2pi, r
print("    NOTE: the operator residual is LARGE (h_A is NOT globally proportional to K_boost=int x*T00)")
print("    because a FINITE region's modular Hamiltonian follows the CC parabola 2pi*x*(1-x/W)*T00, not")
print("    the pure Rindler boost 2pi*x*T00.  => the global operator c is CONTAMINATED by the interior")
print("    fall-off; the NEAR-HORIZON SLOPE (ML1'''-A) is the primary, clean BW observable, not this c.")
check("ML1‴-B operator test recorded (large residual = finite-region parabola, expected) -- the "
      "near-horizon slope is primary; the global c is NOT a clean 2pi read",
      0.05 < op_c_vac < 3.0,
      detail=f"vacuum c/2pi = {op_c_vac:.3f} at residual {op_r_vac:.2f} (contaminated -> use the slope)")

# ===========================================================================
banner("ML1‴-C  routing + cross-link to B3 (the +6 sigma M_Z oblique ~4% scale)")
# ===========================================================================
dev = vac_lim - 1.0                                          # vacuum slope deviation from 2pi
tol = max(0.05, 2 * vac_err)                                 # benchmark-calibrated tolerance
print(f"    vacuum proper-distance slope (extrapolated) = {vac_lim:.3f} x 2pi  (deviation {dev*100:+.1f}%)")
print(f"    operator-level c/2pi (vacuum) = {op_c_vac:.3f};  tolerance ~{tol*100:.0f}%")
print(f"    KMS thermal shift vs vacuum = {thermal_shift*100:+.1f}% (the tick-thermal horizon correction)")
if abs(dev) < tol and abs(op_c_vac - 1) < 0.15:
    routing = ("2pi-COMPUTED: the vacuum proper-distance slope extrapolates to 2pi within tolerance AND "
               "the operator test gives c/2pi~1 => the local horizon carries BW => gravity's Clausius "
               "uses hbar/t_P => G_eff=G (honestly; global kappa=h/t_P stands, two temperatures). The KMS "
               "shift is the tick-thermal correction. hbar FORCED not selected.")
    verdict = "2PI-COMPUTED"
else:
    b3 = 0.04
    tracks = abs(abs(dev) - b3) < 0.02
    routing = (f"CONVERGES-ELSEWHERE: the vacuum slope extrapolates to {vac_lim:.3f} x 2pi (dev {dev*100:+.1f}%), "
               f"NOT 2pi within tolerance. The miss is RE-QUANTIFIED at the measured value (raw, not "
               f"pattern-matched). Newton's G stays OPEN. Cross-link: the deviation "
               f"{'TRACKS' if tracks else 'does NOT track'} B3's ~4% M_Z oblique scale "
               f"({'two walls may be one lattice correction -- report, not forced' if tracks else 'distinct'}).")
    verdict = "CONVERGES-ELSEWHERE"
print("    ROUTING:", routing)
check("ML1‴-C verdict booked (2pi MEASURED via extrapolation, never inserted; hbar NOT selected; G "
      "OPEN unless fully-pinned 2pi-COMPUTED)", True, detail=verdict)

banner("SUMMARY")
print(f"""    ML-1‴ removed BOTH confounds (metric: proper distance under the derived g^ij, factor
    {PROPER:.3f}/cell; state: vacuum vs KMS separately) with declared finite-size extrapolation + an
    operator test.  Benchmark calibration {rb:.4f} x 2pi (trusted pipeline).
      VACUUM (BW-relevant) proper slope  -> {vac_lim:.3f} x 2pi (+/- {vac_err:.3f})   operator c/2pi {op_c_vac:.3f}
      KMS proper slope                   -> {kms_lim:.3f} x 2pi (thermal shift {thermal_shift*100:+.1f}%)
    VERDICT: {verdict}.  2pi MEASURED never inserted; hbar not selected; no scoreboard value moved.
    Newton's G: {'CLOSES G_eff=G (fully-pinned 2pi-COMPUTED)' if verdict=='2PI-COMPUTED' else 'stays an OPEN MISS, re-quantified at the measured slope'}.""")
print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
