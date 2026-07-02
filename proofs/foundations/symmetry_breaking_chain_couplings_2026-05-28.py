"""
proofs/foundations/symmetry_breaking_chain_couplings_2026-05-28.py

Symmetry-breaking chain → gauge scale-dependence, via rep-theory invariants.

THE IDEA (user-led): α_GUT = 1/24 = 1/|S₄| is a rep-theory invariant of the
substrate's geometric symmetry. The gauge couplings' scale-dependence should
come from the SEQUENCE of rep-theory invariants as the symmetry breaks through
the F-fiber chain — NOT from borrowed logarithmic RG.

DISCIPLINE (hard): a match counts ONLY if the invariant is a genuine group-
theory quantity (subgroup order, # generators, Casimir, GQW trace) AND matches
without cherry-picking. Report non-matches honestly. NO forcing.

Tests:
  §1 — subgroup-order chain of S₄ vs inverse couplings
  §2 — GQW trace through PS→SM breaking (is sin²θ_W scale-dependent natively?)
  §3 — generator-count / Casimir invariants at each gauge stage
  §4 — honest verdict
"""

from __future__ import annotations

from fractions import Fraction


def banner(title, char="="):
    print(char * 100)
    print(title)
    print(char * 100)


# ============================================================================
# §1 — S₄ subgroup orders vs observed inverse couplings
# ============================================================================

def section_1_subgroup_orders():
    banner("§1 S₄ subgroup-order chain vs inverse couplings")
    print()
    print("α_GUT = 1/24 = 1/|S₄| (geometric symmetry order). If the coupling is")
    print("1/|effective symmetry|, the inverse coupling = |subgroup| at each scale.")
    print()
    subgroups = {
        "S₄ (full)": 24, "A₄": 12, "D₄": 8, "S₃": 6, "V₄ / C₄": 4,
        "C₃ (generation)": 3, "C₂": 2, "trivial": 1,
    }
    print("S₄ subgroup orders:", {k: v for k, v in subgroups.items()})
    print()

    # Observed inverse couplings (GUT-normalized) at M_Z
    inv_couplings = {
        "α_GUT⁻¹ (unif)": 24.0,
        "α_1⁻¹(M_Z) GUT-norm": 59.0,
        "α_2⁻¹(M_Z)": 29.6,
        "α_3⁻¹(M_Z)": 8.5,
        "α_EM⁻¹(M_Z)": 127.95,
    }
    print(f"  {'coupling':>24}  {'inv value':>10}  {'nearest |subgroup|':>20}  {'match?':>8}")
    print(f"  {'-'*24}  {'-'*10}  {'-'*20}  {'-'*8}")
    for name, inv in inv_couplings.items():
        # nearest subgroup order
        best = min(subgroups.items(), key=lambda kv: abs(kv[1] - inv))
        err = abs(best[1] - inv) / inv
        match = "✓" if err < 0.02 else ("~" if err < 0.10 else "✗")
        print(f"  {name:>24}  {inv:>10.2f}  {best[0]+f' ({best[1]})':>20}  {match:>8} ({err*100:.1f}%)")
    print()
    print("Reading: only α_GUT⁻¹ = 24 = |S₄| matches a subgroup order cleanly.")
    print("The M_Z couplings (59, 29.6, 8.5, 128) do NOT match subgroup orders.")
    print()


# ============================================================================
# §2 — GQW trace through the breaking chain
# ============================================================================

def gqw_trace(fermions):
    """sin²θ_W = Tr(T₃²) / Tr(Q²) over a list of (T_3, Q, multiplicity)."""
    num = sum(t3**2 * m for t3, q, m in fermions)
    den = sum(q**2 * m for t3, q, m in fermions)
    return Fraction(num).limit_denominator() / Fraction(den).limit_denominator()


def section_2_gqw_chain():
    banner("§2 GQW trace through PS→SM breaking — is sin²θ_W natively scale-dependent?")
    print()
    print("sin²θ_W = Tr(T₃²)/Tr(Q²). If the EFFECTIVE matter content changes through")
    print("the breaking chain, the trace changes → native scale-dependence. Test it.")
    print()

    # One SM generation, left-handed Weyl basis (T_3, Q, multiplicity)
    # Q_L: (u_L: T3=+1/2,Q=2/3) (d_L: T3=-1/2,Q=-1/3), ×3 colors
    # L_L: (ν_L: T3=+1/2,Q=0) (e_L: T3=-1/2,Q=-1)
    # u_R^c: T3=0, Q=-2/3, ×3 ; d_R^c: T3=0,Q=1/3,×3 ; e_R^c: T3=0,Q=1 ; ν_R^c: T3=0,Q=0
    sm_full = [
        (Fraction(1,2), Fraction(2,3), 3), (Fraction(-1,2), Fraction(-1,3), 3),  # Q_L
        (Fraction(1,2), Fraction(0), 1),   (Fraction(-1,2), Fraction(-1), 1),    # L_L
        (Fraction(0), Fraction(-2,3), 3),  (Fraction(0), Fraction(1,3), 3),      # u_R^c, d_R^c
        (Fraction(0), Fraction(1), 1),     (Fraction(0), Fraction(0), 1),        # e_R^c, ν_R^c
    ]
    sin2_full = gqw_trace(sm_full)
    print(f"  Full SM generation (incl. ν_R):  sin²θ_W = {sin2_full} = {float(sin2_full):.5f}")

    # Without ν_R (pure SM)
    sm_no_nuR = sm_full[:-1]
    sin2_no_nuR = gqw_trace(sm_no_nuR)
    print(f"  SM generation (no ν_R):          sin²θ_W = {sin2_no_nuR} = {float(sin2_no_nuR):.5f}")

    # Only left-handed (SU(2)-charged) — what if trace is over SU(2)-active only?
    sm_LH = sm_full[:4]
    sin2_LH = gqw_trace(sm_LH)
    print(f"  Left-handed doublets only:        sin²θ_W = {sin2_LH} = {float(sin2_LH):.5f}")
    print()
    print(f"  Observed at M_Z:                  sin²θ_W = 0.23121")
    print(f"  Observed at unification (framework): 3/8 = 0.37500")
    print()
    print("Reading: the GQW trace gives 3/8 for the full/no-ν_R SM content — SAME as")
    print("unification. The trace is SCALE-INDEPENDENT (same matter rep). It does NOT")
    print("produce the M_Z value 0.231. So the breaking chain does NOT give native")
    print("scale-dependence via the GQW trace — the matter rep doesn't change.")
    print()


# ============================================================================
# §3 — Generator counts / dimensions at each gauge stage
# ============================================================================

def section_3_generator_counts():
    banner("§3 Generator counts at each breaking stage vs substrate integers")
    print()
    stages = {
        "PS: SU(4)×SU(2)×SU(2)": 15 + 3 + 3,
        "SM: SU(3)×SU(2)×U(1)": 8 + 3 + 1,
        "EM: SU(3)×U(1)": 8 + 1,
    }
    print("Gauge generators per stage:")
    for name, n in stages.items():
        print(f"  {name:>28}: {n} generators")
    print()
    print("Substrate structural integers: 2|E|=12, |Aut|=24, |V|·|E|=24, g=10, k*=3")
    print()
    print("Note: SM has 12 generators = 2|E| (directed edges). PS has 21. EM has 9.")
    print("These are generator COUNTS; whether they give COUPLINGS requires a rule")
    print("(coupling = 1/generators? = 1/24 at PS? PS has 21 ≠ 24).")
    print()
    print("PS generators (21) ≠ |S₄| (24). So α_GUT = 1/24 is NOT 1/(gauge generators);")
    print("it's 1/|geometric symmetry|. The gauge generator count does not give the")
    print("coupling. Different invariant.")
    print()


# ============================================================================
# §4 — Honest verdict
# ============================================================================

def section_4_verdict():
    banner("§4 Honest verdict", "=")
    print()
    print("WHAT THE SYMMETRY-BREAKING CHAIN GIVES:")
    print("  • α_GUT⁻¹ = 24 = |S₄| at unification — CLEAN, unique. ✓")
    print("  • GQW trace = 3/8 at every stage (matter rep unchanged) — SCALE-INDEPENDENT.")
    print("  • M_Z inverse couplings (59, 29.6, 8.5, 128) — match NO subgroup order,")
    print("    NO clean rep-theory invariant.")
    print()
    print("HONEST CONCLUSION:")
    print("  The symmetry-breaking chain gives the UNIFICATION value natively (1/|S₄|,")
    print("  GQW = 3/8) but does NOT produce the M_Z scale-dependence. The reason is")
    print("  concrete: the rep-theory invariants (subgroup order, GQW trace) are")
    print("  PROPERTIES OF THE GROUP/REP, and the matter rep doesn't change through")
    print("  the breaking — so the invariants are scale-independent. The observed")
    print("  scale-dependence (3/8 → 0.231) is genuinely the logarithmic running,")
    print("  which is NOT a rep-theory invariant.")
    print()
    print("  This CONFIRMS the function-class argument: rep-theory invariants are")
    print("  algebraic and scale-independent; the running is transcendental. The")
    print("  breaking chain changes the GROUP but the invariants of the (unchanged)")
    print("  matter rep stay fixed. So the chain gives discrete native anchor values,")
    print("  NOT continuous running.")
    print()
    print("  WHERE THIS LEAVES IT: the framework natively predicts the coupling at the")
    print("  SYMMETRIC POINT (unification: 1/|S₄|, 3/8). The running between scales is")
    print("  not native — it's the borrowed log. The +4 gap lives in the borrowed log.")
    print("  The symmetry-breaking chain does NOT supply a native replacement, because")
    print("  the rep-theory invariants don't run.")
    print()
    print("  This is a clean negative for the 'breaking chain gives native running'")
    print("  hypothesis. The honest native gauge content remains: 2 invariants at one")
    print("  point. Everything else is borrowed log. NO forcing produced otherwise.")
    print()


def main():
    banner("Symmetry-breaking chain → gauge scale-dependence (honest, no forcing)", "#")
    print()
    section_1_subgroup_orders()
    print()
    section_2_gqw_chain()
    print()
    section_3_generator_counts()
    print()
    section_4_verdict()


if __name__ == "__main__":
    main()
