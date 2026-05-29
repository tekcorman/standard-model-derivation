#!/usr/bin/env python3
"""
R-G2b session 1 — does the framework's linear entropy S_total = N give coasting
via the Cai-Kim horizon first law?

Scoping: an internal working note. NOTE (2026-05-28): this probe honestly finds
G_eff = 2G and defers the factor of 2 to "session 4." That resolution attempt
concluded NOT-FORCED: the factor is the entropy count c_S, the framework's
accounting gives c_S = 1 → G_eff = 2G, and c_S = 2 is not forced (a later
"G_eff = G pinned" claim was an overclaim). The coupling does not close; gravity
is form-level. See cS_horizon_entropy_blind / cS_extent_vs_flux /
cS_2sphere_boundary_reopener (this directory).

THE CALCULATION
---------------
Cai-Kim 2005: applying the Clausius relation δQ = T dS to the flat-FRW apparent
horizon (radius R_A = 1/H), with matter energy flux δQ = A(ρ+p)H R_A dt, gives

    Ḣ = -4π(ρ+p) / [ T(R_A) · S'(R_A) ]            (S' = dS/dR_A)

Integrating with the continuity equation ρ̇ = -3H(ρ+p):
    • if T·S' = const C        ⇒  H² = (8π/3C)·ρ        (STANDARD Friedmann)
    • if T·S' = B·H (∝ 1/R_A)  ⇒  H³ = (4π/B)·ρ          (MODIFIED Friedmann)

So the Friedmann equation is fixed by the R_A-scaling of the PRODUCT T·S'.

THREE PAIRINGS
--------------
  standard:        S = A/4G = π R_A² M_Pl²   T = 1/(2πR_A)   [Gibbons-Hawking]
  framework + GH:  S = N    = R_A M_Pl        T = 1/(2πR_A)   [Gibbons-Hawking]
  framework + κ:   S = N    = R_A M_Pl        T = κ = M_Pl/2  [native Landauer/OEF]

(Framework entropy: S_total = N, and R_A = 1/H = N·t_P = N/M_Pl ⇒ N = R_A·M_Pl,
so S = R_A·M_Pl — LINEAR in R_A, i.e. S ∝ √A.  The Landauer temperature κ = M_Pl/2
is the framework's NATIVE first-law temperature: the OEF theorem is δE_obs = κ·dN,
i.e. δQ = T dS with T = κ. T3 fixed κ = M_Pl/2.)

This probe computes T·S' and the resulting Friedmann for each, symbolically.

Run:
    python3 proofs/cosmology/RG2b_modified_entropy_cai_kim_session1_2026-05-28.py
"""

from __future__ import annotations
import sympy as sp


def banner(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


# symbols
R, H, rho, t, MPl, G, kappa = sp.symbols("R_A H rho t M_Pl G kappa", positive=True)
pp = sp.symbols("rho_plus_p", real=True)   # (ρ+p)

banner("R-G2b SESSION 1 — Cai-Kim first law with the framework's linear entropy")
print("""
  Clausius on the flat-FRW apparent horizon (Cai-Kim 2005):
     Ḣ = -4π(ρ+p) / [ T(R_A)·S'(R_A) ],   R_A = 1/H
  Friedmann is set by the R_A-scaling of the product  P(R_A) ≡ T(R_A)·S'(R_A).
""")

# ---------------------------------------------------------------------------
# (1) Compute P = T·S' for each pairing
# ---------------------------------------------------------------------------
banner("(1) The product P(R_A) = T(R_A)·S'(R_A) for each pairing")

cases = {
    "standard (area S, GH T)":      (sp.pi * R**2 * MPl**2, 1/(2*sp.pi*R)),
    "framework + GH T":             (R * MPl,               1/(2*sp.pi*R)),
    "framework + Landauer T = κ":   (R * MPl,               kappa),
}

results = {}
print(f"  {'pairing':<32} {'S(R_A)':<16} {'T(R_A)':<14} {'P = T·S′':<16}")
print("  " + "-" * 76)
for name, (S, T) in cases.items():
    Sp = sp.diff(S, R)
    P = sp.simplify(T * Sp)
    results[name] = P
    print(f"  {name:<32} {str(S):<16} {str(T):<14} {str(P):<16}")

print("""
  Read-off:
    standard          → P = M_Pl²/2 ... let's see exact value below (const ⇒ H²∝ρ)
    framework + GH    → P ∝ 1/R_A      (⇒ H³∝ρ, MODIFIED — fails coasting)
    framework + κ     → P = κ·M_Pl = const (⇒ H²∝ρ, STANDARD — the linear entropy
                        is COMPENSATED by the constant Landauer temperature)
""")

# ---------------------------------------------------------------------------
# (2) Derive the Friedmann H(ρ) for each (symbolic integration)
# ---------------------------------------------------------------------------
banner("(2) Resulting Friedmann equation H(ρ) for each pairing")

# General: Ḣ = -4π(ρ+p)/P.  Use ρ+p = -ρ̇/(3H) (continuity) and integrate.
# If P = C const:        (1/2) d(H²)/dt = 4π ρ̇/(3C)  ⇒ H² = 8π ρ/(3C)
# If P = B/R_A = B·H:    (1/3) d(H³)/dt = 4π ρ̇/(3B)  ⇒ H³ = 4π ρ/B
def friedmann_from_P(P):
    """Return (power, expression) where H**power = expr·ρ."""
    Pr = sp.simplify(P)
    if sp.diff(Pr, R) == 0:                       # P constant ⇒ H²∝ρ
        C = Pr
        return 2, sp.simplify(8*sp.pi/(3*C))
    # P = B/R_A form: B = P*R_A
    B = sp.simplify(Pr * R)
    if sp.diff(B, R) == 0:                         # P ∝ 1/R_A ⇒ H³∝ρ
        return 3, sp.simplify(4*sp.pi/B)
    return None, None

for name, P in results.items():
    power, coeff = friedmann_from_P(P)
    if power == 2:
        print(f"  {name:<32}:  H² = ({sp.simplify(coeff)})·ρ      [STANDARD]")
    elif power == 3:
        print(f"  {name:<32}:  H³ = ({sp.simplify(coeff)})·ρ      [MODIFIED]")
    else:
        print(f"  {name:<32}:  (non-power-law P)")

# substitute G = 1/M_Pl² and κ = M_Pl/2 to compare couplings
print("\n  With G = 1/M_Pl² and κ = M_Pl/2:")
for name, P in results.items():
    power, coeff = friedmann_from_P(P)
    c = sp.simplify(coeff.subs(kappa, MPl/2))
    cG = sp.simplify(c.subs(MPl, 1/sp.sqrt(G)))   # express in G
    print(f"    {name:<32}: H^{power} = ({sp.simplify(cG)})·ρ")

# ---------------------------------------------------------------------------
# (3) Does framework + κ reproduce coasting with ρ_sub ∝ a⁻²?
# ---------------------------------------------------------------------------
banner("(3) Coasting check — framework + Landauer κ vs the GH alternative")
print("""
  Standard Friedmann H² = (8πG_eff/3)ρ with ρ = ρ_sub ∝ a⁻² (T3):
     H ∝ √ρ_sub ∝ a⁻¹  ⇒  ȧ/a ∝ 1/a  ⇒  ȧ = const  ⇒  a ∝ t   (COASTING ✓)
     and H = 1/t = 1/(N·t_P) is reproduced.  ← framework + Landauer κ lands here.

  Modified Friedmann H³ ∝ ρ (framework + GH temperature):
     coasting a ∝ t (H ∝ 1/t) would require H³ ∝ t⁻³ ∝ ρ ⇒ ρ ∝ a⁻³ (MATTER, w=0),
     NOT ρ_sub ∝ a⁻². Inconsistent with T3.  ← this branch FAILS.

  ⇒ The framework reproduces its coasting cosmology ONLY with its native
    Landauer temperature κ (constant), NOT the geometric Gibbons-Hawking
    temperature T_GH = 1/(2πR_A). The pre-registered NEGATIVE-ENTROPY-MISMATCH
    is AVOIDED — because the linear entropy × constant temperature gives the
    SAME T·S' = const as standard area-entropy × GH-temperature.
""")

# ---------------------------------------------------------------------------
# (4) the coupling factor and the κ-tension with T3
# ---------------------------------------------------------------------------
banner("(4) Coupling factor — G_eff = 2G, and the factor-2 κ tension with T3")
P_fw = sp.simplify(results["framework + Landauer T = κ"].subs(kappa, MPl/2))
P_std = sp.simplify(results["standard (area S, GH T)"])
print(f"  P(framework+κ) = T·S′ = {P_fw}     (= M_Pl²/2)")
print(f"  P(standard)    = T·S′ = {P_std}     (= M_Pl²)")
print(f"  ratio P_fw/P_std = {sp.simplify(P_fw/P_std)}")
print(f"""
  H² = (8π/3P)·ρ, so smaller P ⇒ larger coupling:
    standard:        H² = (8πG/3)·ρ          [P = M_Pl²,   G_eff = G]
    framework + κ:   H² = (16πG/3)·ρ          [P = M_Pl²/2, G_eff = 2G]

  The framework + native (linear S, Landauer κ=M_Pl/2) gives G_eff = 2G — a
  factor 2. T3 (volume reading ρ = E_obs/V_Hubble, (4π/3)R_H³) fixed κ = M_Pl/2;
  the horizon-FLUX reading here would want κ = M_Pl for G_eff = G. Both give
  κ ~ M_Pl (Planck scale — robust, forced); the O(1) factor differs between the
  VOLUME (T3, 4π/3) and AREA-FLUX (R-G2b) normalizations. Reconciling the two is
  a unified-first-law bookkeeping task (session 4), not a scale ambiguity.
""")

# ---------------------------------------------------------------------------
# (5) verdict
# ---------------------------------------------------------------------------
banner("VERDICT — R-G2b session 1: POSITIVE-COUPLING-ONLY (qualified)")
print("""
  RESULT:
   • Computed the Cai-Kim Friedmann for the framework's linear entropy
     S_total = N = R_A·M_Pl. The Friedmann form is set by T·S′(R_A):
       – framework + Gibbons-Hawking T  → T·S′ ∝ 1/R_A → H³∝ρ (MODIFIED; fails
         coasting — would need matter ρ∝a⁻³, not ρ_sub∝a⁻²).
       – framework + native Landauer T=κ → T·S′ = M_Pl²/2 = const → H²∝ρ
         (STANDARD Friedmann; reproduces coasting with ρ_sub∝a⁻², G_eff=2G).

   • KEY STRUCTURAL FINDING: the framework's (linear entropy)×(constant Landauer
     temperature) gives the SAME T·S′ = const as standard (area entropy)×(GH
     temperature). The two non-standard features compensate exactly at the level
     of the Friedmann equation. So the linear entropy is NOT a fatal mismatch —
     NEGATIVE-ENTROPY-MISMATCH is AVOIDED.

   • This GROUNDS the Friedmann coupling (H²∝ρ, "why E_obs gravitates") from the
     framework's NATIVE Clausius relation δE_obs = κ·dN — JUSTIFYING the standard
     Friedmann that T3 had to ASSUME. R-G2b + T3 together cover more of (G).

  RESIDUALS (why not POSITIVE-UNIFIED):
   • TEMPERATURE: the framework must use its Landauer κ (Planck-scale, constant),
     NOT the geometric Gibbons-Hawking T_GH = H/2π. Self-consistent, but DIVERGES
     from standard de Sitter horizon thermodynamics. Whether this is forced or a
     liability is the open NEGATIVE-TEMPERATURE question → session 4.
   • COUPLING: G_eff = 2G (factor 2 vs T3's κ=M_Pl/2 volume normalization) — an
     O(1) bookkeeping reconciliation, not a scale issue.
   • ρ_sub ∝ a⁻² still comes from T3's holographic V_Hubble reading, not derived
     independently here. R-G2b gives the FORM/COUPLING, T3 gives the SOURCE.

  Per W58: positive on the coupling, honest on the temperature residual. The
  Jacobson route is VIABLE for the framework (the entropy-mismatch is avoided);
  the decisive open question is now the horizon temperature (Landauer vs GH).
""")
