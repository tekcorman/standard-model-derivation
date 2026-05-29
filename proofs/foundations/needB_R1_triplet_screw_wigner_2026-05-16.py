#!/usr/bin/env python3
"""
needB_R1_triplet_screw_wigner_2026-05-16.py

Need-B research target, Route R1: specialize the solved lepton δ=2/9
screw-axis Wigner-D template to the color-TRIPLET (down) sector.

DISCIPLINE (anti-numerology, load-bearing). R1 either DERIVES δ_down
structurally by the SAME mechanism that gives δ_lepton=2/9 (sibling-
consistency), reproducing R4's re-pinned target (δ_down≈0.10 rad,
ε²_down≈2.5), or it is an HONEST NEGATIVE. No combination-hunting.

The lepton template (theorem_41_screw_wigner.md SS-1..SS-4) gives δ=2/9
THREE independent ways that COINCIDE:
  Route A : HM of the j=1 4₁-screw survival probs {4/9,1/9,4/9} = Q²/2
  Route B : Q_Koide / n_gen
  Direct  : Koide-inversion of (m_e,m_μ,m_τ)
The doc states the A=B coincidence "Q²/2 = Q/3 is EQUIVALENT to Q=2/3"
(SS-4 consistency note) — i.e. it is a Q=2/3-SPECIFIC accident, plus the
HM↔δ identification is itself an unproven physical argument (§6(i),
= Need-B). This probe tests whether "specialize the template" is even
well-posed for the triplet, where R4 found Q_down≈0.75≠2/3.
"""

import cmath, math
from fractions import Fraction

OMEGA = cmath.exp(2j * math.pi / 3)


def koide_Q(m):
    s = [math.sqrt(x) for x in m]
    return sum(m) / sum(s) ** 2


def koide_delta(m):
    """direct δ from the framework circulant-Koide form (R4 inversion)."""
    r = [math.sqrt(x) for x in m]
    F1 = sum(r[j] * OMEGA ** j for j in range(3))
    return (-cmath.phase(F1)) % (2 * math.pi / 3)


def routeA_HM(Q):
    """HM of the screw pattern {Q², Q²/4, Q²} = Q²/2 (the lepton form)."""
    P = [Q ** 2, Q ** 2 / 4, Q ** 2]
    return 3.0 / sum(1.0 / p for p in P)          # = Q²/2


def main():
    print("=" * 78)
    print("Need-B R1 — triplet-sector screw-axis Wigner-D (template transfer)")
    print("=" * 78)

    # ---- lepton self-check: the three routes MUST coincide at 2/9 -------
    m_e, m_mu, m_tau = 0.51099895e-3, 0.1056583755, 1.77686
    Ql = koide_Q([m_e, m_mu, m_tau])
    A_l = routeA_HM(2.0 / 3.0)                     # screw amp Q=2/3 (j=1)
    B_l = Ql / 3.0
    D_l = koide_delta([m_e, m_mu, m_tau])
    print("\n[SELF-CHECK] lepton — three routes must all = 2/9 = %.6f:"
          % (2 / 9))
    print(f"  Route A  HM{{4/9,1/9,4/9}} = Q²/2 = {A_l:.6f}")
    print(f"  Route B  Q_Koide/n_gen    = {Ql:.5f}/3 = {B_l:.6f}")
    print(f"  Direct   Koide-invert      = {D_l:.6f}")
    ok = max(abs(A_l-2/9), abs(B_l-2/9), abs(D_l-2/9)) < 5e-3
    print(f"  coincide at 2/9: {'PASS — template valid for leptons' if ok else 'FAIL'}")

    # ---- the coincidence is Q=2/3-SPECIFIC (pure algebra) ---------------
    print("\n[STRUCTURAL] Route A = Route B  ⟺  Q²/2 = Q/3  ⟺  Q = 2/3:")
    for Q in [Fraction(2, 3), Fraction(3, 4), Fraction(1, 2)]:
        a, b = Q * Q / 2, Q / 3
        print(f"  Q={str(Q):>4}:  Q²/2={float(a):.5f}  Q/3={float(b):.5f}  "
              f"{'EQUAL (only at 2/3)' if a == b else 'DIVERGE'}")

    # ---- triplet (down) sector: do the three routes still coincide? -----
    # R4 scenarios (PDG MS-bar; S4 = RG-clean GJ-textured leptons)
    downsets = {
        "S1 μ=2GeV":  (4.67e-3, 93.4e-3, 4.90),
        "S2 μ=m_b":   (2.82e-3, 55.0e-3, 4.18),
        "S4 GJ@GUT":  (3 * m_e, m_mu / 3, m_tau),
    }
    print("\n[TRIPLET] down sector — R4 found Q_down≈0.75≠2/3. The three")
    print("          lepton-coincident routes now DIVERGE:")
    print(f"  {'scenario':12s} {'Q_down':>7s} {'A:Q²/2':>8s} "
          f"{'B:Q/3':>7s} {'Direct δ':>9s}   spread")
    spreads = []
    for nm, ms in downsets.items():
        Qd = koide_Q(list(ms))
        A, B, D = Qd ** 2 / 2, Qd / 3, koide_delta(list(ms))
        sp = max(A, B, D) - min(A, B, D)
        spreads.append(sp)
        print(f"  {nm:12s} {Qd:7.4f} {A:8.4f} {B:7.4f} {D:9.4f}   "
              f"Δ={sp:.3f}")

    # ---- verdict -------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("• Lepton: Route A = Route B = Direct = 2/9 — but ONLY because")
    print("  Q=2/3 makes Q²/2 = Q/3 (algebraic identity, SS-4). The")
    print("  template's robustness is a Q=2/3-SPECIFIC accident.")
    print(f"• Triplet: Q_down≈0.75≠2/3 ⇒ the three routes DIVERGE")
    print(f"  (spread {min(spreads):.2f}–{max(spreads):.2f} rad across the")
    print("  band 0.10–0.28). There is NO single 'screw-Wigner-D δ' for")
    print("  the triplet — the lepton derivation had three coincident")
    print("  expressions; the triplet has three different numbers.")
    print("• Plus: §6(i) — even the lepton HM↔δ identification is an")
    print("  UNPROVEN physical argument (= Need-B). R1 inherits it: a")
    print("  triplet HM, even if clean, would NOT be theorem-grade δ.")
    print()
    print("⇒ R1 HONEST NEGATIVE (structurally informative, not a fail to")
    print("  hunt past): 'specialize the screw-Wigner-D template' is")
    print("  ILL-POSED for the triplet — its self-consistency was a")
    print("  Q=2/3 coincidence absent at Q_down≈0.75. The triplet δ is")
    print("  NOT the lepton-template analog.")
    print()
    print("  RE-SCOPE Need-B: the triplet δ must come from the framework's")
    print("  OWN quark-Koide structure (Row P37 koide_quark_ratio")
    print("  =(3g−2)/g, g=5 — the ε²≈2.5 family R4 surfaced) via R3 (the")
    print("  diagonal reading of the one G_NB), NOT R1/R2 (screw-Wigner-D")
    print("  analog) and NOT the refuted arg(h)/4 (R4). R2 ('derive the")
    print("  /4') is mooted with R1 (same template). Need-B narrows to:")
    print("  ONE route — R3 + the Row-P37 (3g−2)/g quark-Koide object.")
    print("=" * 78)


if __name__ == "__main__":
    main()
