#!/usr/bin/env python3
# ============================================================
# F8 gate, closure attempt: the g_A reduction 5/3 -> 1.2723 (factor 0.7634)
# ============================================================
#
# Scope: internal research notes §F8 open leg.
# Predecessors:
#   F8_gA_nucleon_spin_content_2026-05-31.py  -> g_A LEADING ORDER = 5/3 (SU(6),
#       from the color-singlet junction + spinor-return + Pauli). REAL result.
#   F8F9_bbn_endtoend_Yp_2026-05-31.py        -> the 5/3->1.2723 reduction is a
#       ~26 sigma lever on Y_p; the single highest-value open piece in the
#       baryon program. THIS is why we take the reduction on now.
#
# THE TARGET. Observed g_A = 1.2723 (PDG, +/- ~0.0023; experimental spread
# 1.2723-1.2762). SU(6) leading order = 5/3 = 1.66667. Reduction factor
#   r = g_A / (5/3) = 0.76338.
#
# WHAT THE REDUCTION PHYSICALLY IS (so we attack the right thing). g_A is the
# nucleon ISOVECTOR axial charge. Its reduction from the NR quark-model 5/3 is
# the RELATIVISTIC / Melosh (Wigner-rotation) renormalization of the isovector
# axial current in the bound state. It is MILD (~0.76) and is DISTINCT from the
# axial SINGLET "spin crisis" (Delta-Sigma ~ 0.3, factor 0.3) which carries the
# gluon/anomaly piece. So we are after the mild relativistic isovector factor,
# NOT the spin-crisis singlet suppression. (Conflating them is a known error.)
#
# THE FRAMEWORK STRUCTURE WE CAN USE. The axial current IS the chirality
# operator gamma^5, and in this framework CHIRALITY = the walk srs<->srs-z
# (theorem_V_Ram_Cl6_Fock_iso; V_Ram_Cl6_iso_T5_CLOSURE). The walker eigenvalue
# h = (sqrt3 + i*sqrt5)/2 is fixed by the Ihara-Bass quadratic, and the
# POSITIVE-IMAGINARY root is CHIRALITY-SELECTED (predictions/h_walker_eigenvalue.py:
# "the srs lattice is chiral... the positive-imaginary root is the physical
# enantiomer"). So Im(h) = sqrt5/2 is literally the chirality-carrying component
# of the walker spectrum. A chiral observable (g_A) plausibly involves Im(h) /
# sqrt5 / the golden ratio phi = 1/2 + Im(h). That hook is REAL, not arbitrary.
#
# DISCIPLINE (the logged F7 lesson: a clean number match is NOT a derivation).
# This probe (1) states the target + physics, (2) tests the clean candidates,
# (3) runs a look-elsewhere scan to gauge numerology density, and CRUCIALLY
# (4) attempts to DERIVE the factor from the framework's ACTUAL walk/chirality
# structure -- and reports honestly whether it closes. It does NOT promote a
# coincidence to a prediction.

import math
import itertools

SU6 = 5.0 / 3.0
G_A_OBS = 1.2723
G_A_SIG = 0.0023
R_OBS = G_A_OBS / SU6                       # 0.76338 target reduction

# framework spectral primitives (theorem-grade upstream)
K_STAR = 3
E_P = math.sqrt(K_STAR)                     # = sqrt3 = 2 Re(h)
RE_H = math.sqrt(K_STAR) / 2                # Re(h) = sqrt3/2 (chirality-PRESERVING propagation)
IM_H = math.sqrt(4 * (K_STAR - 1) - K_STAR) / 2   # Im(h) = sqrt5/2 (chirality-SELECTED)
ABS_H2 = K_STAR - 1                         # |h|^2 = 2 (Ramanujan saturation)
PHI = (1.0 + math.sqrt(5.0)) / 2.0          # golden ratio = 1/2 + Im(h)
SURV = (K_STAR - 1) / K_STAR                # 2/3 = non-backtracking survival per step


def dev(val):
    return (val - G_A_OBS) / G_A_OBS


def main():
    print("=" * 76)
    print(" F8 gate closure attempt — the g_A reduction 5/3 -> 1.2723")
    print("=" * 76)
    print(f"   target  g_A = {G_A_OBS} +/- {G_A_SIG}   (SU(6) LO = 5/3 = {SU6:.5f})")
    print(f"   reduction r = g_A/(5/3) = {R_OBS:.5f}")
    print(f"   physics: MILD relativistic/Melosh isovector renorm (NOT the")
    print(f"            singlet 'spin crisis' 0.3) -- attack the right object.")

    # -----------------------------------------------------------------------
    print("\n[1] clean candidates (the tempting matches):")
    cands = [
        ("sqrt(phi) = sqrt(1/2 + Im h)", math.sqrt(PHI)),
        ("(2/3)^(2/3) * 5/3  (survival^survival)", SURV ** SURV * SU6),
        ("4/pi   [Clause-9 EXCLUDED: pi]", 4.0 / math.pi),
    ]
    for name, val in cands:
        print(f"     {name:42s} = {val:.6f}   dev = {100*dev(val):+.3f}%")
    print("     => THREE different clean forms land within ~0.07% of g_A. A match")
    print("        at this density is a warning, not a result.")

    # -----------------------------------------------------------------------
    print("\n[2] look-elsewhere scan (algebraic framework grammar, NO pi):")
    prims = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "1/2": 0.5,
             "2/3": 2 / 3, "5/3": 5 / 3, "sqrt3": math.sqrt(3),
             "sqrt5": math.sqrt(5), "sqrt2": math.sqrt(2), "phi": PHI}
    cand = {}
    for n, v in prims.items():
        cand[n] = v
        if v > 0:
            cand["sqrt" + n] = math.sqrt(v)
    for (na, a), (nb, b) in itertools.product(prims.items(), prims.items()):
        for op, z in [("/", a / b), ("*", a * b), ("-", a - b), ("+", a + b)]:
            if z > 0:
                cand[f"{na}{op}{nb}"] = z
                cand[f"sqrt({na}{op}{nb})"] = math.sqrt(z)
    total = len({round(z, 5) for z in cand.values()})
    for w in (0.001, 0.005):
        hits = sorted({round(z, 5) for z in cand.values() if abs(z - G_A_OBS) / G_A_OBS < w})
        print(f"     within +/-{w*100:.1f}%: {len(hits)} of {total} distinct values  {hits}")
    print("     => within the ALGEBRAIC grammar sqrt(phi) is fairly distinguished")
    print("        (1 in ~445 at 0.1%); but fractional powers ((2/3)^(2/3)) and pi")
    print("        (4/pi) re-enter as competitors in the broader function space.")

    # -----------------------------------------------------------------------
    print("\n[3] DERIVATION ATTEMPTS from the framework's ACTUAL structure:")
    print("    (a) walk-return recursion = golden? Ihara-Bass: h^2 = E_P h - (k*-1)")
    print(f"        => h^2 = sqrt3 h - 2.  Golden needs x^2 = x + 1.")
    print(f"        framework |h|^2 = {ABS_H2} (Ramanujan), NOT phi = {PHI:.4f}.")
    print(f"        => the framework return recursion is NOT the golden recursion;")
    print(f"           there is no Fibonacci/continued-fraction fixed point here.")
    print(f"    (b) is g_A^2 = phi structurally?  phi = 1/2 + Im(h) = {0.5 + IM_H:.5f}.")
    print(f"        The '1/2' has NO framework origin (not Re(h)^2 = {RE_H**2:.3f}, not")
    print(f"        a survival, not |h|^2/4). So g_A^2 = 1/2 + Im(h) is unexplained.")
    print(f"    (c) direct chirality-overlap from h (the physical mechanism):")
    print(f"        chirality RETENTION = Re(h)-sector / |h|-sector. The natural ratios")
    ret1 = RE_H ** 2 / ABS_H2
    ret2 = RE_H / math.sqrt(ABS_H2)
    print(f"          Re(h)^2/|h|^2 = {ret1:.4f}  -> g_A = {SU6*ret1:.4f}  (way low)")
    print(f"          Re(h)/|h|     = {ret2:.4f}  -> g_A = {SU6*ret2:.4f}  (low)")
    print(f"        target retention r = {R_OBS:.4f} is BETWEEN these; no clean h-ratio")
    print(f"        lands on it. The chirality hook is real but does NOT yield 0.7634.")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(" VERDICT — F8 g_A reduction: INCONCLUSIVE, no closure (honest negative)")
    print("=" * 76)
    print(f"""  The framework does NOT derive the 0.7634 reduction. Specifically:

   - g_A = sqrt(phi) matches to 0.02% AND has a genuine hook (Im(h)=sqrt5/2 is
     the chirality-selected walker component; phi is built from sqrt5). It is
     also fairly distinguished in the algebraic grammar (~1 in 445 at 0.1%).
     This is the most intriguing single number in the open baryon program.
   - BUT every derivation route FAILS: (a) the framework's actual return
     recursion is h^2 = sqrt3*h - 2 (|h|^2 = 2, Ramanujan), NOT the golden
     x^2 = x+1 -- so 'g_A^2 = phi from walk-return' has no basis; (b) g_A^2 =
     1/2 + Im(h) leaves the 1/2 unexplained; (c) the direct, physically-correct
     chirality-overlap ratios from h (Re(h)^2/|h|^2 = 3/8, Re(h)/|h| = 0.612)
     bracket but do NOT hit 0.7634. And competitors ((2/3)^(2/3), 4/pi) sit
     equally close, so the bare numerical match cannot carry the claim.

  DISPOSITION (per the logged F7 discipline -- a clean match is not a
  derivation): the g_A LEADING ORDER (5/3) STANDS as the framework result; the
  0.76 reduction REMAINS the open wall. sqrt(phi) is logged as an UNEXPLAINED
  COINCIDENCE with a suggestive chirality hook -- a candidate to revisit, NOT a
  closure, and NOT to be promoted to a prediction.

  WHAT REAL CLOSURE WOULD REQUIRE (the genuine multi-step item this forecloses
  the shortcut for): the Melosh / Wigner-rotation average of the isovector axial
  charge over the framework's ACTUAL bound 3-walker momentum wavefunction (the
  relativistic reduction integral), OR a non-circular derivation of g_A^2 = phi
  from the srs<->srs-z chirality walk that explains the unexplained 1/2. Both are
  real bound-state-QCD work, not a one-line spectral identity.

  NET: taking the reduction on CLARIFIES the situation -- it pins the physics
  (mild relativistic isovector renorm, not the singlet crisis), forecloses the
  tempting sqrt(phi) shortcut as underived, and shows the framework's spectral
  primitives (h, |h|^2) do not contain the factor. The ~26 sigma Y_p lever
  stays genuinely open; g_A LO = 5/3 is the honest framework boundary.""")
    print("=" * 76)


if __name__ == "__main__":
    main()
