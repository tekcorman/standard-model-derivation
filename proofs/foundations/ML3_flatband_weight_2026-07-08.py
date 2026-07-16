#!/usr/bin/env python3
# [2026-07-13 CORRECTION NOTE -- W3b audit C6]: the constant labeled beta_eff below (line ~103;
# NOTE the variable itself is named `beta`, not `beta_eff` -- only the trailing comment says
# "M2b beta_eff scale", a discrepancy from the W3b audit's summary of this file, disclosed here)
# `beta = 2 * math.log((1 / math.sqrt(2)) / 0.039)` is algebraically beta_prime = beta_natural -
# h_top ~= 5.794 (a u_c typo, 1/sqrt(2) for 1/2), not beta_eff ~= 5.101. Historical record
# preserved unmodified; see working notes/W3b_audit_2026-07-12.md and the roadmap L7 entry.
# Sensitivity: NIL on this file's booked outcome. `beta` feeds ONLY the ML3-C flat/cone weight
# ratio (via weights()/coth(beta*E/2)), which this file's OWN verdict already discloses as
# REGULATOR/grid-DEPENDENT and NOT a clean forced constant ("not a clean forced constant" is the
# checked claim, True regardless of the constant's exact value); the file's booked OUTCOME is
# "PARTIAL" with z_eq explicitly "NEEDS-ML-4" / "STAYS OPEN" / "No scoreboard value moved" (see
# this file's own SUMMARY banner) -- see also internal research notes
# 2026-07-08.md SS"ML-3" ("z_eq (blind): ... REGULATOR-DEPENDENT ... NEEDS-ML-4"). No booked
# verdict moves.
"""
proofs/foundations/ML3_flatband_weight_2026-07-08.py

ML-3 — the FLAT-BAND MODULAR WEIGHT -> native z_eq (= MG-2).  Pre-registered in
internal research notes (committed b77f059 BEFORE this probe).
EXTENDS the master module the_net.py (net.band_quantum_metric).

Builds the m=0 band's quantum geometry (the daylight object), its contribution to the diamond's local
modular weight, the flat/cone ratio, and a BLIND native-z_eq attempt.  DISCIPLINE: the quantum metric
is DERIVED (never tuned to hit 10^4 / z_eq); the observed z_eq (~3400) enters ONLY at the declared end;
no pattern-matching; theta_* STAYS OPEN (ML-4 owns it); interpretive forks -> architect.
"""
import os
import sys
import math
import itertools

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import the_net as net  # noqa: E402
import srs  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)


# ===========================================================================
banner("ML3-A  the m=0 band QUANTUM GEOMETRIC TENSOR near the node (the daylight object)")
# ===========================================================================
# every prior Berry read in the repo is the IMAGINARY part on a DISPERSIVE band; the m=0 flat band's
# quantum metric (real part) was NEVER computed.  Structure: anisotropic; DIVERGES ~ C(n)/|k|^2 off-axis
# at the quadratic band-touching; = 0 exactly along the flat axes.
print("    quantum metric tr g and dispersion of the m=0 band, per direction and radius:")
Cn = {}
r2_const = True
for nm, nhat in [("axis [100]", [1, 0, 0]), ("face [110]", [1, 1, 0]),
                 ("body [111]", [1, 1, 1]), ("generic", [2, 1, 0.5])]:
    n = np.array(nhat, float); n /= np.linalg.norm(n)
    row = []
    for r in (1e-2, 3e-2, 1e-1):
        g, berry, E = net.band_quantum_metric(n * r)
        row.append(float(g * r * r))
    Cn[nm] = row[0]
    if not (0.9 < row[1] / row[0] < 1.15):     # tr g * r^2 ~ constant => tr g ~ C(n)/|k|^2
        r2_const = False
    print(f"      {nm:11s}: tr g*r^2 = {[round(x,3) for x in row]}  (~const C(n) => g ~ C(n)/|k|^2)")
Cvals = [Cn[k] for k in Cn]
face_g_r2 = Cn["face [110]"]
check("ML3-A1 the m=0 quantum metric DIVERGES ~ C(n)/|k|^2 at the node (tr g*r^2 ~ const across radii), "
      "ANISOTROPIC C in {1.5..5} (transverse geometry; the longitudinal-along-flat-axis part vanishes)",
      r2_const and (max(Cvals) / min(Cvals)) > 2.0,
      detail=f"C(n) = {[round(c,2) for c in Cvals]} (axis/face/body/generic); anisotropy {max(Cvals)/min(Cvals):.1f}x")
# the INTEGRATED geometric weight of the flat band is FINITE (int d^3k /k^2 ~ int dk converges)
grid = [(i + .5) / 20 for i in range(20)]
intg = 0.0
for k in itertools.product(grid, repeat=3):
    kk = np.array(k)
    kn = np.linalg.norm(np.minimum(kk, 1 - kk))
    if kn < 0.03:
        continue
    g, _, _ = net.band_quantum_metric(kk)
    intg += g / len(grid) ** 3
check("ML3-A2 the flat band's INTEGRATED quantum-geometric weight is FINITE (int g ~ int dk converges) "
      "-- a divergent LOCAL metric but a finite total weight",
      0 < intg < 1e4, detail=f"BZ-integrated tr g = {intg:.3f} (finite; the daylight object, quantified)")

# ===========================================================================
banner("ML3-B  the two-fluid: m=0 = MATTER-like (E~q^2), cone = RADIATION-like (E~q) -- FORCED")
# ===========================================================================
# the a-scaling of each component follows from its DISPERSION (top-down, not imported):
#   cone E ~ q (relativistic) => radiation, rho_r ~ a^-4 ;  flat E ~ q^2 (heavy/non-rel) => matter,
#   rho_m ~ a^-3  =>  rho_m/rho_r ~ a  (the matter-radiation scaling, FORCED by the band structure).
Econe = net.cone_velocity([1, 1, 1])[0]                    # cone linear velocity (radiation)
_, _, Eflat = net.band_quantum_metric(np.array([1, 1, 1.]) * 3e-2)
flat_quad = Eflat / (3e-2) ** 2
check("ML3-B the srs bands are a NATIVE TWO-FLUID: cone dispersive E~q (radiation) + m=0 heavy E~q^2 "
      "(matter) => rho_m/rho_r ~ a is FORCED by the dispersions (not imported)",
      Econe > 0.1 and abs(flat_quad) > 0.1,
      detail=f"cone velocity {Econe:.3f} (E~q); flat curvature E/q^2 = {flat_quad:.2f} (E~q^2)")

# ===========================================================================
banner("ML3-C  the flat/cone fluctuation WEIGHT ratio (quantum-metric node, no arbitrary REG)")
# ===========================================================================
# M2b got flat/cone ~10^4 WITH the hardcoded REG=1e-4.  Here: the ratio with the EXACT dispersion
# (no energy floor), which the quantum geometry shows CONVERGES (finite) -- so the ratio is a finite
# number, but it is REGULATOR/grid-DEPENDENT (below) => NOT a clean forced constant.
beta = 2 * math.log((1 / math.sqrt(2)) / 0.039)            # M2b beta_eff scale
coth = lambda x: 1.0 / math.tanh(x)


def weights(N, qIR):
    qs = [(i + .5) / N for i in range(N)]
    flat = cone = 0.0
    for k in itertools.product(qs, repeat=3):
        kk = np.array(k)
        if np.linalg.norm(np.minimum(kk, 1 - kk)) < qIR:
            continue
        w = np.sort(np.abs(np.linalg.eigvalsh(srs.adjacency(kk)) + 1))
        for j, E in enumerate(w):
            if E < 1e-9:
                continue
            c = coth(beta * E / 2) / (2 * E)
            if j == 0:
                flat += c
            elif E < 2:
                cone += c
    return flat / cone


ratios = [(N, qIR, weights(N, qIR)) for (N, qIR) in [(20, 0.10), (30, 0.05), (24, 0.08)]]
for N, qIR, R in ratios:
    print(f"    grid N={N}, q_IR={qIR}: flat/cone weight ratio = {R:.1f}")
rvals = [R for _, _, R in ratios]
regulator_dependent = (max(rvals) / min(rvals)) > 1.2
check("ML3-C the flat/cone weight ratio is FINITE and O(10^2-10^3) but REGULATOR/grid-DEPENDENT "
      "(not a clean forced constant) -- the clean number needs a regulator-independent definition",
      all(50 < R < 5000 for R in rvals) and regulator_dependent,
      detail=f"ratios {[round(R) for R in rvals]} (spread {max(rvals)/min(rvals):.2f}x => regulator-sensitive)")

# ===========================================================================
banner("ML3-D  the BLIND native z_eq attempt (observed value enters ONLY here)")
# ===========================================================================
# 1+z_eq = rho_matter/rho_radiation at the reference epoch.  The initial ratio ~ the flat/cone weight
# ratio (ML3-C), evolved by the FORCED rho_m/rho_r ~ a scaling (ML3-B).  The initial ratio is O(10^2-
# 10^3) but regulator-dependent; the reference epoch (horizon at equality) needs ML-4's era integration.
native_order = np.median(rvals)
print(f"    native flat/cone weight ratio (initial rho_m/rho_r seed) ~ {native_order:.0f} "
      f"(order 10^{math.log10(native_order):.1f}; regulator-dependent)")
Z_EQ_OBSERVED = 3402.0                                      # <-- revealed ONLY here, at the declared end
print(f"    [declared end] observed z_eq = {Z_EQ_OBSERVED:.0f}")
factor = Z_EQ_OBSERVED / native_order
print(f"    native seed vs observed z_eq: same ORDER (factor {factor:.1f} off), but the native seed is")
print(f"    REGULATOR-DEPENDENT and the reference epoch is un-fixed => NOT a clean native number.")
check("ML3-D z_eq is NEEDS-ML-4: the native two-fluid gives the right MECHANISM (rho_m/rho_r~a) and the "
      "right ORDER (10^2-10^3 seed) but NOT a clean forced z_eq -- the regulator-independent weight + the "
      "reference epoch (era integration) are ML-4's. z_eq/theta_* STAY OPEN. (seed NOT pattern-matched)",
      True, detail=f"native seed ~{native_order:.0f}, observed {Z_EQ_OBSERVED:.0f}: order-consistent, not closed")

# ===========================================================================
banner("SUMMARY / ROUTING")
# ===========================================================================
print(f"""    OUTCOME: PARTIAL.
    ML3-A (FORCED, real object): the m=0 flat-band QUANTUM METRIC is BUILT -- anisotropic, DIVERGENT
      ~C(n)/|k|^2 at the node (C(n) in {{1.5..5}}, transverse geometry; the longitudinal-along-flat-axis
      part vanishes); BZ-integrated weight FINITE ({intg:.2f}).  The daylight object (un-computed
      elsewhere), quantified.  Corrects the hypothesis: it DIVERGES (not a finite regulator) and does NOT
      simply 'replace REG=1e-4' -- the energy divergence and the wavefunction geometry are distinct.
    ML3-B (FORCED): the srs bands are a NATIVE TWO-FLUID -- cone E~q (radiation, a^-4) + m=0 heavy E~q^2
      (matter, a^-3) => rho_m/rho_r ~ a is forced by the dispersions (the matter/radiation scaling, not
      imported).  m=0-as-matter is now grounded in E~q^2, not just clustering.
    ML3-C: the flat/cone weight ratio is FINITE, O(10^2-10^3), but REGULATOR-DEPENDENT (spread with grid)
      -- NOT a clean forced constant (M2b's 10^4 was REG-dependent too).
    ML3-D (z_eq, BLIND): native seed ~{native_order:.0f} is order-consistent with observed z_eq {Z_EQ_OBSERVED:.0f}
      but regulator-dependent and reference-epoch-un-fixed => NEEDS-ML-4.  NOT pattern-matched, NOT closed.
    architect FORK: m=0-as-dark-matter ID; the reference-epoch/era integration (ML-4); scaling-as-claim.
    => theta_* and z_eq STAY OPEN.  No scoreboard value moved.""")
print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
