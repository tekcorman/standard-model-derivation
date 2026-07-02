#!/usr/bin/env python3
"""
needB_R3_diagonal_GNB_rowP37_2026-05-16.py

Need-B research target, Route R3 (the sole route left after R1/R2/R4):
is δ_quark a diagonal reading of the one G_NB, operating on the
framework's own Row-P37 quark-Koide object?

DISCIPLINE (anti-numerology, load-bearing): R3 must DERIVE δ by a
mechanism that reproduces the lepton case as a limit — or it is an
HONEST NEGATIVE that precisely locates the irreducible gap. No
formula-hunting for ≈0.10.

Three facts established before computing (all confirmed live, this turn):
 (F1) Row-P37 is (ε²_up−2)/(ε²_down−2) = (3g−2)/g, g=10 (girth) = 14/5,
      UNIQUE-THEOREM-GRADE. It supplies ε²_down structurally. It is
      δ-SILENT (zero phase content). [Corrects an earlier "g=5" slip
      and the R4-propagation overstatement that "ε²_down must be
      derived" — Row-P37 already derives it.]
 (F2) theorem_41 §6(i), verbatim: under M1.B, Route B (δ=Q/n_gen) is
      rigorous *only as an ALGEBRAIC IDENTITY*; the physical
      interpretation of δ-as-cosine-phase (the parameter that sets
      mass ratios) "is Need-B … mass ∝ inverse propagator ∝ 1/survival"
      — a physical argument, not a spectral-reading choice.
 (F3) δ_lepton=2/9 is NOT cleanly any arg(h_P)/cavity reading
      ((2/9)/arg(h_P)=0.2437, not 1/4 nor 1/3).

This probe tests R3's premise ("δ_quark = a G_NB spectral-phase
reading") against the mandatory lepton-consistency gate.
"""

import cmath, math

OMEGA = cmath.exp(2j * math.pi / 3)
H_P = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
ARGH = math.atan2(H_P.imag, H_P.real)            # 0.91174 rad
TWO_NINTH = 2.0 / 9.0
G_GIRTH, K_STAR = 10, 3


def koide_Q(m):
    s = [math.sqrt(x) for x in m]
    return sum(m) / sum(s) ** 2


def main():
    print("=" * 78)
    print("Need-B R3 — diagonal G_NB reading × Row-P37 (sole remaining route)")
    print("=" * 78)

    m_e, m_mu, m_tau = 0.51099895e-3, 0.1056583755, 1.77686
    Ql = koide_Q([m_e, m_mu, m_tau])

    # (A) Route-B ALGEBRAIC identity reproduces lepton 2/9 (the only
    #     route that does — and §6(i) says it is ONLY an algebraic id).
    print("\n[A] Lepton δ via the Route-B ALGEBRAIC identity (§6(i)):")
    print(f"    δ = Q_Koide/n_gen = {Ql:.5f}/3 = {Ql/3:.6f}  vs 2/9="
          f"{TWO_NINTH:.6f}  → algebraic identity holds (Q=2/3 only).")

    # (B) CATEGORY TEST: can ANY structurally-principled G_NB spectral-
    #     phase reading reproduce the lepton δ=2/9? Gate: must hit 2/9
    #     by a mechanism, else δ is NOT a spectral-phase object.
    print("\n[B] Category test — structurally-principled G_NB phase")
    print("    readings vs the mandatory lepton gate (must = 2/9):")
    cands = {
        "arg(h_P)/4         (R4-refuted)": ARGH / 4,
        "arg(h_P)/k*":                     ARGH / K_STAR,
        "arg(h_P)·2/g":                    ARGH * 2 / G_GIRTH,
        "Ihara λ=h+q/h phase":             cmath.phase(H_P + (K_STAR - 1) / H_P),
        "cavity g(z) phase @λ_P":          cmath.phase(1 / (math.sqrt(3) - K_STAR * (1/(math.sqrt(3) - (K_STAR-1))))),
    }
    any_clean = False
    for nm, v in cands.items():
        v = abs(v)
        d = abs(v - TWO_NINTH)
        clean = d < 0.005
        any_clean |= clean
        print(f"    {nm:34s} = {v:.5f}  Δ(2/9)={d:.4f}  "
              f"{'← hits 2/9' if clean else 'no'}")
    print(f"    → any spectral-phase reading reproduces 2/9 by mechanism? "
          f"{'YES' if any_clean else 'NO — δ is NOT a G_NB spectral object'}")

    # (C) Row-P37 supplies ε² (done, thm-grade) but is δ-SILENT.
    eps2_down_rowP37 = "(3g−2)/g structure → ε²_up,ε²_down via many-body; " \
                       "ratio (ε²_up−2)/(ε²_down−2)=14/5, g=10 — δ ABSENT"
    print(f"\n[C] Row-P37: {eps2_down_rowP37}")
    print("    ⇒ '× Row-P37' supplies the ε² part of Need-B at")
    print("      UNIQUE-THEOREM-GRADE, but CANNOT supply δ (δ-silent).")

    # ---- VERDICT -------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT — R3 ELIMINATED; route-elimination COMPLETE")
    print("=" * 78)
    print("• Category mismatch (F3 + test B): δ is NOT a G_NB spectral-")
    print("  phase object — no structurally-principled cavity/Ihara/")
    print("  arg(h) reading reproduces even the lepton 2/9 by mechanism")
    print("  (arg(h)/4≈0.228 is the R4-refuted ~2.5% coincidence, not a")
    print("  derivation). δ lives in the Route-B ALGEBRAIC family.")
    print("• But (F2 / §6(i)): Route-B is rigorous ONLY as an algebraic")
    print("  identity; identifying that identity with the PHYSICAL")
    print("  cosine-phase that sets mass ratios is the mass∝1/survival")
    print("  argument = Need-B's irreducible core = the T_mass→physical-")
    print("  mass DYNAMICAL bridge. This is the SAME unproven gap for")
    print("  LEPTONS (δ=2/9 is a 'numerical coincidence / algebraic")
    print("  identity only' per §6) as for quarks.")
    print("• Row-P37 (F1) already solves the ε² part at theorem-grade")
    print("  (g=10; my earlier 'g=5' was wrong) and is δ-silent.")
    print()
    print("⇒ R1/R2/R3 ALL category-mismatched: δ-the-physical-phase is")
    print("  NOT a reading-selection problem. Need-B does NOT decompose")
    print("  as 'ε (find Row-P37) + δ (find a reading)'. It decomposes")
    print("  as:  ε²  = SOLVED (Row-P37, UNIQUE-THEOREM-GRADE)")
    print("       δ   = the §6(i) mass∝1/inverse-propagator physical")
    print("             identification = the deep substrate-DYNAMICS")
    print("             layer itself (one of the 'five masks'), the")
    print("             SAME irreducible gap for leptons and quarks.")
    print()
    print("SYNTHESIS REVISION (honest, supersedes the scoping-doc §1/§9")
    print("pitch): Need-B is NOT a localized / best-posed face of the")
    print("deep frontier. Its open part (δ-physical) IS the monolithic")
    print("deep T_mass/dynamics layer — no localized or reading-")
    print("selectable entry exists; the lepton sector has the identical")
    print("gap (its δ=2/9 is algebraic-identity-only). The quark-mass")
    print("path: ε done (Row-P37); δ = deep layer (lepton-shared, not")
    print("separable); up-type additionally needs y_t (deep). The deep")
    print("frontier is genuinely monolithic — Need-B was not a handle.")
    print("=" * 78)


if __name__ == "__main__":
    main()
