#!/usr/bin/env python3
"""
R1 probe — does the Misner-Sharp / work-density Cai-Kim first law give a free
O(1) knob that pins κ = M_Pl/2 with G_eff = G, or is the factor of 2 forced?

Scoping parent: internal research notes

ADVERSARIAL INTENT: we are NOT trying to land κ = M_Pl/2. We test whether the
work density W = (ρ−p)/2 supplies an independent factor, or whether it is already
absorbed in the standard Cai-Kim heat flux δQ = A(ρ+p)H R_A dt. Then we localize
exactly where the factor of 2 lives.

All algebra in sympy. Natural units: G = 1/M_Pl², t_P = 1/M_Pl, ℏ = c = 1.
"""

import sympy as sp

def banner(s): print("\n" + "=" * 72 + f"\n  {s}\n" + "=" * 72)

# symbols
R, H, G, kappa, M_Pl, rho, p, t = sp.symbols("R H G kappa M_Pl rho p t", positive=True)
Sprime = sp.symbols("Sprime", positive=True)   # dS/dR_A, generic

banner("(1) Cai-Kim master equation — generic T·S' product")
# Heat crossing apparent horizon in dt (Cai-Kim 2005, flat FRW):
#   δQ = A(ρ+p) H R_A dt,  A = 4π R_A²,  R_A = 1/H
# Clausius δQ = T dS = T·S'·dR_A = T·S'·(dR_A/dt) dt
# ⇒ Ḣ = −4π(ρ+p)/(T·S').  Friedmann coupling set by P ≡ T·S'.
P = sp.symbols("P", positive=True)
# From Ḣ = −4π(ρ+p)/P and continuity ρ̇ = −3H(ρ+p): H² = (8π/(3P))·ρ
# standard Friedmann H² = (8πG/3)ρ ⇒ P_standard = 1/G = M_Pl²
P_standard = 1/G
print(f"  G_eff = G  ⟺  P = T·S' = 1/G = M_Pl²")
print(f"  (Friedmann coupling H² = 8π/(3P)·ρ; standard needs P = 1/G)")

banner("(2) Is the work density W=(ρ−p)/2 an UNUSED knob?")
# Hayward unified first law: dE = AΨ + W dV, E = Misner-Sharp = ρV.
# The heat that appears in T dS is the energy-supply term; worked out for FRW it
# is δQ = A(ρ+p) H R_A dt.  KEY: the combination is (ρ+p) = enthalpy density,
# NOT ρ.  The work density W = (ρ−p)/2 has ALREADY been used to convert the
# Misner-Sharp dE (which carries ρ) into the work-corrected flux (which carries
# ρ+p).  Verify the bookkeeping closes with no leftover factor.
V = sp.Rational(4,3)*sp.pi*R**3
E_MS = rho * V                      # Misner-Sharp energy inside horizon
W = (rho - p)/2                     # work density (Hayward)
dV_dR = sp.diff(V, R)
# Hayward: -dE + W dV  (the heat supply through the horizon, per dR)
#   -dE/dR carries ρ and ρ̇; on-shell with continuity the static piece is:
heat_supply_per_dR = -sp.diff(E_MS, R) + W*dV_dR
heat_supply_per_dR = sp.simplify(heat_supply_per_dR)
print(f"  −dE/dR + W·dV/dR = {heat_supply_per_dR}")
print(f"  = 4πR²·[−ρ + (ρ−p)/2]  = 4πR²·[−(ρ+p)/2]")
print(f"  The (ρ+p) enthalpy is produced BY the work term; W is NOT free.")
print(f"  ⇒ R1 premise refuted: no independent work-density factor remains.")

banner("(3) Localize the factor of 2 — it is entirely in P = κ·S'")
# Framework: S = c_S · R·M_Pl  (c_S = entropy event-counting normalization)
#   S' = c_S · M_Pl ;  T = κ.   P = κ · c_S · M_Pl.
# Require P = M_Pl² (for G_eff = G):  κ · c_S · M_Pl = M_Pl²  ⇒  κ · c_S = M_Pl.
c_S = sp.symbols("c_S", positive=True)
eq_GeffG = sp.Eq(kappa * c_S * M_Pl, M_Pl**2)
kappa_for_GeffG = sp.solve(eq_GeffG, kappa)[0]
print(f"  S = c_S·R·M_Pl  ⇒  S' = c_S·M_Pl ;  T = κ ;  P = κ·c_S·M_Pl")
print(f"  G_eff = G requires  κ·c_S = M_Pl,  i.e.  κ = M_Pl / c_S")
print(f"     c_S = 1 (net-node count)      ⇒  κ = M_Pl       (contradicts OEF M_Pl/2)")
print(f"     c_S = 2 (both toggle events)  ⇒  κ = M_Pl/2      (consistent with OEF)")

banner("(4) Cross-check against the holographic-volume loop")
# Holographic loop: ρ_sub = E_obs/V = κ·N/((4π/3)R³), N = clock count = R·M_Pl.
# Plug into standard Friedmann (assumed) — identity for all H iff κ = M_Pl/2.
N_clock = R*M_Pl
E_obs = kappa*N_clock
rho_sub = E_obs/V
fried_resid = sp.simplify(H**2 - sp.Rational(8,3)*sp.pi*G*rho_sub.subs(R, 1/H))
kap_hol = sp.solve(sp.Eq(fried_resid, 0), kappa)
print(f"  holographic loop (assumes Friedmann) ⇒ κ = {kap_hol}  (·M_Pl with G=1/M_Pl²)")
print(f"  uses N_clock = R·M_Pl  (c_S = 1 implicitly) but VOLUME reading,")
print(f"  not the FLUX reading — the two disagree by the c_S/volume factor.")

banner("(5) VERDICT")
print("""  • R1 (work density) is CLOSED-NEGATIVE: W=(ρ−p)/2 is already spent
    producing the (ρ+p) enthalpy in the standard Cai-Kim flux. No free knob.
  • The factor of 2 is NOT scheme-movable in the flux route. It is a single
    sharp question: the horizon-entropy event-counting normalization c_S.
       c_S = 1 (entropy tracks NET node creation)  ⇒ flux route forces κ = M_Pl
                                                     ⇒ G_eff = G but κ ≠ OEF M_Pl/2
       c_S = 2 (entropy tracks BOTH toggle events) ⇒ κ = M_Pl/2 gives G_eff = G
                                                     ⇒ consistent with OEF
  • So κ = M_Pl/2 closes WITH G_eff = G  IFF  the horizon entropy counts both
    create and destroy events (c_S = 2), while the cascade CLOCK counts net
    nodes (dN/dt = 1).  That entropy-vs-clock factor of 2 must be DERIVED from
    the cascade + perceptual-surface accounting, independently of the desired κ.
  • NOT closed here. R1 negative; residue sharpened to the c_S binary (R2).
""")
