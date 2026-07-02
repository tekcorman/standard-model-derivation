#!/usr/bin/env python3
"""
Holographic-identification conjecture — first analysis (2026-05-28).

This is a SHARPENING (it explicitly claims no closure). NOTE (2026-05-28): the
coupling-magnitude headline that later built on this arc ("G_eff = G / mechanism
COMPLETE / parameter-free Newton's G") is RETRACTED — the framework's
horizon-entropy count is c_S = 1 → G_eff = 2G; gravity is form-level. See
cS_horizon_entropy_blind / cS_extent_vs_flux / cS_2sphere_boundary_reopener
(this directory) for the corrected, exhaustive treatment.

THE REMAINING OPEN PIECE
------------------------
After T2 (admissible) + T3 (positive-candidate), the substrate-thermal-coupling
MECHANISM has one open conjecture: the holographic identification
ρ_sub = E_obs / V_Hubble — "why is the framework holographic?" This analysis
SHARPENS that conjecture (it does not close it) by separating what is robustly
derived from what is genuinely posited.

THREE-WAY DECOMPOSITION
-----------------------
Write ρ_sub = E_obs / V = (κ·N) / V_causal. Three sub-claims:

  (S) SCALING ρ_sub ∝ a⁻².  Claim: this is NEARLY FREE — it follows from
      • E_obs = κ·N  with N ∝ t          [cascade: dN/dt = 1 per t_P, LINEAR info]
      • V_causal ∝ R_H^{d}  with R_H ∝ t  [a causal 3-volume; d = d_spatial = 3]
      ⇒ ρ_sub ∝ N / N^d = N^{1-d} ∝ a^{-(d-1)} = a⁻²  for d = 3.
      So w_sub = (d-4)/3, and d_spatial = 3 (theorem-grade) ⇒ w = −1/3 (coasting).
      The substrate EOS is TIED TO THE SPATIAL DIMENSION.

  (N) NORMALIZATION κ = M_Pl/2.  Fixed by self-consistency with Friedmann +
      the HUBBLE volume (V = (4π/3)H⁻³). Planck-scale, natural; matched not derived.

  (G) GRAVITATES.  The genuine residual posit: why the observer's temporal
      information energy E_obs is a UNIFORM gravitating energy density filling
      its causal patch. This is the real content of "why holographic."

BEKENSTEIN CHECK
----------------
Is the framework saturating the holographic (area) bound? S_max(Hubble horizon)
= Area/(4G) ∝ N². Cascade S_total = N. So S_total/S_max ∝ 1/N — the framework
sits a factor ~N BELOW the area bound. Its information is LINEAR/TEMPORAL
(one bit per Planck time along the worldline), NOT area-saturating/spatial.
This reframes the conjecture: it is not "the universe saturates the Bekenstein
bound" but "the observer's linear temporal record gravitates over its causal
3-volume."

HONESTY: sharpening only. (S) is reduced to {cascade, d_spatial=3}; (N) is a
matched constant; (G) is the open conjecture. No closure.

Run:
    python3 proofs/cosmology/holographic_identification_conjecture_first_analysis_2026-05-28.py
"""

from __future__ import annotations

import math

# --- constants (theorem-grade framework primitives + standard) --------------
M_PL = 1.220890e19        # Planck mass (GeV)
N_HUB = 8.394881e60       # predictions/N_hub.py
D_SPATIAL = 3             # predictions/d_spatial.py (theorem-grade)
G_NEWTON = 1.0 / M_PL ** 2
t_P = 1.0 / M_PL
kappa = M_PL / 2.0        # from T3 matching


def banner(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


# ===========================================================================
banner("HOLOGRAPHIC-IDENTIFICATION CONJECTURE — first analysis (sharpening)")
print(f"""
  Open conjecture (T3): ρ_sub = E_obs / V_Hubble, with E_obs = κ·N.
  Decompose into (S) scaling, (N) normalization, (G) gravitates.
  d_spatial = {D_SPATIAL} (theorem-grade), N_hub = {N_HUB:.3e}.
""")

# ---------------------------------------------------------------------------
# (S) SCALING from {linear info, d_spatial} — the EOS-dimension link
# ---------------------------------------------------------------------------
banner("(S) ρ_sub ∝ a^(-(d-1)); w_sub = (d-4)/3 — tied to d_spatial")
print("""
  E_obs = κ·N,  N ∝ t                 [cascade dN/dt = 1 per t_P — LINEAR info]
  V_causal ∝ R_H^d,  R_H ∝ t          [causal patch, d = d_spatial spatial dims]
  ρ_sub = E_obs/V ∝ N / N^d = N^{1-d}

  At the coasting attractor a ∝ N:  ρ_sub ∝ a^{1-d} = a^{-(d-1)}.
  From ρ ∝ a^{-3(1+w)}:  3(1+w) = d-1  ⇒  w_sub = (d-4)/3.
""")
for d in (2, 3, 4):
    rho_exp = -(d - 1)
    w = (d - 4) / 3.0
    tag = "  ← d_spatial (theorem-grade)" if d == D_SPATIAL else ""
    print(f"    d = {d}:  ρ_sub ∝ a^({rho_exp:+d})   w_sub = {w:+.3f}{tag}")
print(f"""
  ⇒ d_spatial = 3 ⇒ ρ_sub ∝ a⁻², w = −1/3 (coasting). The substrate EOS is a
    CONSEQUENCE of the spatial dimension + linear information growth — both
    theorem-grade. The SCALING half of the conjecture is reduced to existing
    framework facts (it is NOT an independent posit).
""")

# ---------------------------------------------------------------------------
# (N) NORMALIZATION — Hubble volume fixes κ = M_Pl/2 (recap from T3)
# ---------------------------------------------------------------------------
banner("(N) Normalization — the Hubble-volume identification fixes κ = M_Pl/2")
# ρ_sub = κN/((4π/3)R_H³), R_H = 1/H; Friedmann ⇒ H = 1/(2GκN); match cascade.
H_cascade = 1.0 / (N_HUB * t_P)
kappa_matched = 1.0 / (2.0 * G_NEWTON * M_PL)
print(f"""
  V = (4π/3)R_H³ (Hubble volume) + Friedmann ⇒ H = 1/(2GκN); matching the
  cascade H = 1/(N·t_P) fixes κ = 1/(2G·M_Pl) = M_Pl/2 = {kappa_matched:.4e} GeV.
  (Recap of T3 — Planck-scale, natural; a matched constant, not derived.)
  Sanity: κ used here = M_Pl/2 = {kappa:.4e} GeV; matches: {math.isclose(kappa, kappa_matched)}
""")

# ---------------------------------------------------------------------------
# Bekenstein bound — the framework is LINEAR/TEMPORAL, far below the area bound
# ---------------------------------------------------------------------------
banner("Bekenstein check — S_total = N vs area bound S_max ∝ N²")
# S_max = Area/(4G) for a sphere of radius R_H = N·t_P = N/M_Pl
# Area = 4π R_H²;  S_max = 4π R_H²/(4G) = π R_H² M_Pl² = π (N/M_Pl)² M_Pl² = π N²
def S_max_area(N):
    R_H = N * t_P
    area = 4.0 * math.pi * R_H ** 2
    return area / (4.0 * G_NEWTON)   # = π N²

print(f"\n  {'N':>12} {'S_total = N':>14} {'S_max = πN² (area)':>20} {'ratio':>12}")
print("  " + "-" * 60)
for N in (1e30, 1e45, N_HUB):
    smax = S_max_area(N)
    print(f"  {N:>12.2e} {N:>14.2e} {smax:>20.2e} {N/smax:>12.2e}")
print(f"""
  S_total/S_max = N/(πN²) = 1/(πN) → the framework's information sits a factor
  ~N BELOW the holographic area bound. Its content is LINEAR/TEMPORAL (one bit
  per Planck time along the observer worldline), not area-saturating/spatial.

  ⇒ The conjecture is NOT "the universe saturates Bekenstein." It is narrower:
    "the observer's linear temporal record (E_obs = κN) gravitates as a uniform
    density over its causal 3-volume." That is sub-claim (G) below.
""")

# ---------------------------------------------------------------------------
# (G) the genuine residual — what still must be derived from A1
# ---------------------------------------------------------------------------
banner("(G) The genuine residual posit — what 'derive from A1' now means")
print("""
  Sharpened conjecture (the ONLY open piece of the mechanism):

    Why is the observer's temporal information energy E_obs = κ·N a UNIFORM
    GRAVITATING energy density spread over its causal (Hubble) 3-volume?

  What is now NO LONGER mysterious (reduced to existing framework facts):
    • the a⁻² SCALING  → {cascade linear info} ÷ {d_spatial = 3 causal volume}
    • the w = −1/3 EOS  → w = (d_spatial − 4)/3, a dimension consequence
    • the κ = M_Pl/2 normalization → Hubble-volume self-consistency (matched)

  Candidate routes for (G) (future sessions):
    R-G1  Causal-patch argument: E_obs is information about the observer's
          causal past; it is necessarily distributed over the causal patch
          (Hubble volume). Needs: A1-level statement that observer information
          is spatially co-extensive with its light-cone interior.
    R-G2  Equivalence-principle / Jacobson route: derive Friedmann itself from
          the observer-entropy E_obs = κ·S_total across the causal horizon
          (Jacobson 1995 'Einstein eq of state'). Would turn (G) + Friedmann
          into ONE thermodynamic derivation rather than two posits.
    R-G3  MDL/A2-T spatial-extent: the observer's compressed model has a
          spatial support; show it fills the causal patch under A2-T.

  Per W58: this is a sharpening + route map, NOT a closure. (G) remains a
  conjecture. R-G2 (Jacobson) is the most promising — it could derive the
  Friedmann coupling AND the holographic identification together.
""")

banner("VERDICT — conjecture SHARPENED (scaling reduced; (G) is the residual)")
print(f"""
  • (S) SCALING ρ_sub ∝ a⁻² and w = −1/3 REDUCED to {{cascade linear-N, d_spatial=3}}
    via w = (d_spatial − 4)/3. Not an independent posit.
  • (N) κ = M_Pl/2 — matched via Hubble-volume self-consistency (T3).
  • Bekenstein: framework is ~N below the area bound → linear/temporal info,
    not area-holographic. Reframes the conjecture.
  • (G) RESIDUAL CONJECTURE: why E_obs gravitates uniformly over the causal
    3-volume. Most promising route = Jacobson 1995 (derive Friedmann from
    horizon entropy E_obs = κ·S_total), which would unify (G) + the Friedmann
    coupling. This is the next deep target.
""")
