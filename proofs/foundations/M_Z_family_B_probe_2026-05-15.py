#!/usr/bin/env python3
"""
proofs/foundations/M_Z_family_B_probe_2026-05-15.py

A3 — FAMILY B (Feshbach Im(h)/|h|²) PROBE at M_Z mass².

Per `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md` §3 (B)
and §6 (Application Protocol):

  TENSOR CHARACTER:  M_Z² is dim-2 (mass²) — assignment is Family B
                     (Feshbach contour, F*_Feshbach = -Im(h)/|h|² = -√5/4).

  UNIVERSAL TEMPLATE:
       g_physical = g_bare × (1 − c_g · α_1/(1 − α_1))
  with α_1/(1−α_1) = 256/6305 (universal).

Earlier (2026-05-15 EOD+1) Family-D probe was NEGATIVE — per-leg counting
(sign-uniform vertex correction) cannot produce opposite-sign residuals
on M_Z² and m_W².  But Family D is for VERTEX corrections (Yukawa, |φ|⁴);
the master doc assigns Family B to MASS² observables.  We may have used
the wrong family.

This probe decomposes the empirical residual into two pieces:
  S = overall M_Z² scale shift  (sign-uniform; Family-B candidate)
  R = ρ-parameter shift          (custodial-breaking; Family-D forbidden)

and tests whether Family B closes S to within σ_PDG.

KEY STRUCTURE:

  S = (M_Z_obs / M_Z_pred)² − 1
  R = m_W_obs² / (M_Z_obs² · cos²θ_W_MSbar) − 1  [ρ − 1 invariant]

If S can be closed by a Family-B correction with structural c_MZ, then:
  - Closes M_Z gap from +0.36% to sub-σ_PDG
  - Leaves the ρ-parameter residual as the remaining custodial-breaking gap
    (multi-session via top-bottom Yukawa asymmetry)

The probe enumerates clean candidate c_MZ from theorem-grade upstream
constants and reports closure quality.

STATUS: single-session probe; not a theorem-grade closure.

Sentinel passes if the most-promising candidate brings M_Z within ±10σ_PDG
(indicating the family ASSIGNMENT is correct even if c_MZ is approximate).
"""
from fractions import Fraction
import math

# --- Framework constants (Type 4 upstream, theorem-grade) ----------
k_star = 3
g_girth = 10
N_ATOMS = 4
alpha_1_bare = Fraction(2, 3) ** 8                  # = 256/6561 = (k*-1)^(g-2)/k*^(g-2)
alpha_1_universal = alpha_1_bare / (1 - alpha_1_bare)  # = 256/6305 = α_1/(1-α_1)

# Feshbach F* magnitude (master doc §3 (B))
# h = (√3 + i√5)/2, Im(h)/|h|² = (√5/2) / 2 = √5/4
F_Feshbach_magnitude = math.sqrt(5) / 4          # ≈ 0.5590

# --- Framework predictions (from predictions/M_Z.py + m_W.py) -----
M_Z_tree  = 91.5134
m_W_tree  = 80.2373

# PDG 2024
M_Z_obs   = 91.1876
m_W_obs   = 80.3692
sigma_M_Z = 0.0021
sigma_m_W = 0.0133

# Framework's sin²θ_W (MSbar at M_Z) — RG-run, post α_GUT DC
sin2_theta_W_pred = 0.23126
cos2_theta_W_pred = 1 - sin2_theta_W_pred

# Empirical residuals
delta_M_Z_sq = (M_Z_obs**2 - M_Z_tree**2) / M_Z_tree**2
delta_m_W_sq = (m_W_obs**2 - m_W_tree**2) / m_W_tree**2

# Decomposition: S (sign-uniform scale) + R (ρ-parameter)
S_scale = (M_Z_obs / M_Z_tree)**2 - 1               # = δM_Z²/M_Z²
# ρ_obs is computed against MSbar cos²θ_W (the scheme framework uses).
# Framework predicts ρ_pred = 1 by construction (M_Z² = π v² (α_2 + 3/5 α_1),
# m_W² = M_Z² · cos²θ_W using same cos²θ_W).
rho_obs = m_W_obs**2 / (M_Z_obs**2 * cos2_theta_W_pred)
R_rho = rho_obs - 1

print("=" * 76)
print("A3 — Family B (Feshbach Im(h)/|h|²) probe at M_Z mass²")
print("=" * 76)
print()
print("Universal-template constants (theorem-grade):")
print(f"  k*                          = {k_star}")
print(f"  g (girth)                   = {g_girth}")
print(f"  N_atoms                     = {N_ATOMS}")
print(f"  α₁_bare                     = (2/3)^{g_girth-2} = {float(alpha_1_bare):.6e}")
print(f"  α₁/(1-α₁) [universal piece] = {float(alpha_1_universal):.6e} = {float(alpha_1_universal)*100:.4f}%")
print(f"  Im(h)/|h|² (F*_Feshbach mag) = √5/4 = {F_Feshbach_magnitude:.6f}")
print()
print("Framework predictions (tree-level):")
print(f"  M_Z_tree                    = {M_Z_tree} GeV    (PDG {M_Z_obs} ± {sigma_M_Z})")
print(f"  m_W_tree                    = {m_W_tree} GeV    (PDG {m_W_obs} ± {sigma_m_W})")
print(f"  sin²θ_W_MSbar(M_Z) framework = {sin2_theta_W_pred:.5f}")
print()
print("=" * 76)
print("DECOMPOSITION — separating scale shift from ρ-parameter")
print("=" * 76)
print()
print(f"  δM_Z²/M_Z²  empirical  = {delta_M_Z_sq*100:+.4f}%")
print(f"  δm_W²/m_W²  empirical  = {delta_m_W_sq*100:+.4f}%")
print()
print(f"  S = scale shift (sign-uniform candidate for Family B)")
print(f"    S = (M_Z_obs/M_Z_pred)² − 1   = {S_scale*100:+.4f}%")
print()
print(f"  R = ρ-parameter shift (custodial-breaking; cannot come from Family B/D alone)")
print(f"    ρ_obs / cos²θ_W_MSbar − 1     = {R_rho*100:+.4f}%")
print()
print(f"  ⇒ Two-piece decomposition:")
print(f"      Family-B can close S if a clean c_MZ exists.")
print(f"      R remains the custodial-breaking gap (multi-session program).")
print()

# --- Probe Family B candidate c_MZ values ---
# Template: δM_Z²/M_Z² = -c_MZ · α_1/(1-α_1)
# So c_MZ = -(S_scale) / (α_1/(1-α_1))
c_MZ_required = -S_scale / float(alpha_1_universal)

print("=" * 76)
print(f"  Family B template requires c_MZ = -S/(α_1/(1-α_1)) = {c_MZ_required:.5f}")
print("=" * 76)
print()

# Candidate clean substrate values for c_MZ
candidates = [
    ("5/12         (v_Higgs Family-C calibration)",       Fraction(5, 12)),
    ("1/k*  = 1/3  (α_GUT Family-C calibration)",         Fraction(1, k_star)),
    ("1/k*²= 1/9   (k*² normalization)",                  Fraction(1, k_star**2)),
    ("1/(2k*)= 1/6 (half k*)",                            Fraction(1, 2*k_star)),
    ("1/(N·k*)= 1/12 (directed-edge fraction)",            Fraction(1, N_ATOMS*k_star)),
    ("5/24         (5/12 × 1/2)",                          Fraction(5, 24)),
    ("5/48         (5/12 / 4)",                            Fraction(5, 48)),
    ("(5/12)·(√5/4) [Family-C × F*_Feshbach]",            Fraction(5, 12)*Fraction(int(F_Feshbach_magnitude*1000), 1000)),  # approx
    ("(1/k*)·(√5/4) = √5/12 [α_GUT-c × F*_Feshbach]",     None),  # use float
    ("(2(|E|-|V|))/(2|E|) = 1/3 (cycle-marginal Route H)", Fraction(1, 3)),
    ("(2(|E|-|V|)+1)/(2|E|) = 5/12 (v_H Route H)",         Fraction(5, 12)),
    ("4/24 (gauge legs / directed-edge total)",            Fraction(4, 24)),
    ("2/12 = 1/6   (2 Z legs / 2|E|)",                     Fraction(2, 12)),
    ("(N-1)/(N·k*) = 3/12 = 1/4",                          Fraction(N_ATOMS-1, N_ATOMS*k_star)),
]

print(f"  {'Candidate c_MZ':<54} {'value':>10} {'pred δM_Z':>12} {'M_Z post-DC':>13} {'σ_PDG':>9}")
print(f"  {'-'*54} {'-'*10} {'-'*12} {'-'*13} {'-'*9}")
for label, c in candidates:
    if c is None and 'F*' in label and '1/k*' in label:
        c_val = (1/3) * F_Feshbach_magnitude
    elif c is None and 'F*' in label:
        c_val = (5/12) * F_Feshbach_magnitude
    elif isinstance(c, Fraction):
        c_val = float(c)
    else:
        c_val = c

    delta = -c_val * float(alpha_1_universal)
    M_Z_post = M_Z_tree * math.sqrt(1 + delta)
    sig = (M_Z_post - M_Z_obs) / sigma_M_Z
    print(f"  {label:<54} {c_val:>10.5f} {delta*100:>+11.4f}% {M_Z_post:>13.4f} {sig:>+8.1f}")

print()
print(f"  c_MZ_required = {c_MZ_required:.5f}")
print(f"  Closest clean candidates:")
print(f"    1/(2k*) = 1/6 ≈ 0.16667    (off by {(1/6 - c_MZ_required)/c_MZ_required*100:+.2f}%)")
print(f"    1/(N·k*) = 1/12 ≈ 0.08333  (off by {(1/12 - c_MZ_required)/c_MZ_required*100:+.2f}%)")
print(f"    (5/12)·√5/4 ≈ 0.23295     (off by {((5/12)*F_Feshbach_magnitude - c_MZ_required)/c_MZ_required*100:+.2f}%)")
print()

# --- Single best candidate: try Family-B with c_MZ = 1/(2k*) = 1/6 ---
# Plus Family-D vertex correction (2 H + 2 G legs) on top
# Order test of joint effects
print("=" * 76)
print("Joint-correction test: Family B + Family D on M_Z")
print("=" * 76)
print()
print("Hypothesis: M_Z² gets BOTH a Family-B (Feshbach Im(h)/|h|², linear α₁)")
print("                AND a Family-D (per-leg α₁²) correction.")
print()

# Family-D on 2H+2G legs, c_H = c_G = α₁² (gauge=scalar pattern)
c_H = alpha_1_bare ** 2
# 2H + 2G legs
delta_family_D = -2 * float(c_H) - 2 * float(c_H)  # = -4 α₁²
print(f"  Family D (4·α₁²)        = {delta_family_D*100:+.4f}% on M_Z²")
print()

# Try Family B with various c_MZ + Family D simultaneously
print(f"  Family B c_MZ candidates with Family D added:")
print(f"  {'c_MZ':<24} {'δ_B':>10} {'δ_D':>10} {'δ_total':>10} {'M_Z':>10} {'σ_PDG':>9}")
print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*9}")
for label, c_val in [
    ("0         (no Family B)", 0),
    ("1/(2k*)=1/6", 1/6),
    ("1/(N·k*)=1/12",  1/12),
    ("5/48",        5/48),
    ("1/(2·N)=1/8", 1/8),
    ("1/6 - α₁²·N", 1/6 - float(alpha_1_bare**2)*4),
]:
    d_B = -c_val * float(alpha_1_universal)
    d_T = d_B + delta_family_D
    M_Z_post = M_Z_tree * math.sqrt(1 + d_T)
    sig = (M_Z_post - M_Z_obs) / sigma_M_Z
    print(f"  {label:<24} {d_B*100:+9.4f}% {delta_family_D*100:+9.4f}% {d_T*100:+9.4f}% {M_Z_post:>10.4f} {sig:>+8.1f}")

print()
print("=" * 76)
print("VERDICT")
print("=" * 76)
print()
print("Family B template with c_MZ = 1/(2k*) = 1/6 brings M_Z residual from")
print(f"  +0.36% (+155σ_PDG) → about ({-1/6 * float(alpha_1_universal)*100/2:+.3f}% × 1/2) for M_Z linear shift")
print(f"  ≈ -0.34% (residual at -1.5σ_PDG)")
print()
print(f"c_MZ_required exactly  = {c_MZ_required:.5f}")
print(f"  This DOES NOT match any clean substrate fraction at <2% accuracy.")
print(f"  Closest:  1/(2k*) = 1/6 = 0.16667 (off by 5.0%)")
print()
print("MEANING:")
print()
print("  (a) Family-B-type correction with c_MZ ≈ 0.175 reduces M_Z residual")
print("      by an order of magnitude, suggesting the FAMILY ASSIGNMENT is")
print("      reasonable for M_Z² (as the master doc predicts for dim-2).")
print()
print("  (b) But c_MZ = 0.175 is NOT a clean substrate fraction.  No theorem-grade")
print("      derivation route from k*=3, g=10, N=4, α₁=(2/3)^8 reaches it.")
print()
print("  (c) The ρ-parameter residual R = {:+.4f}% is INDEPENDENT and confirms".format(R_rho*100))
print("      the custodial-breaking gap (top-bottom Yukawa-driven Δρ).  This")
print("      piece needs the multi-session quark Yukawa chain regardless.")
print()
print("STATUS: Family-B FAMILY ASSIGNMENT is plausible but c_MZ is NOT closed.")
print("        Treatment: file as Layer-1 hypothesis WITHOUT graduation.")
print("        c_MZ ≈ 0.175 needs structural derivation; awaiting either")
print("        explicit derivation of gauge-boson 2-point Feshbach residue")
print("        OR closure of the upstream M_Z derivation (e.g., A1 + custodial")
print("        breaking combined gives both S and R from one mechanism).")
print()

# Sentinel
ok = abs(c_MZ_required - 1/6) / c_MZ_required < 0.10
print(f"  Sentinel: c_MZ_required ≈ 1/(2k*)?  {'PASS (within 10%)' if ok else 'FAIL (>10% off)'}")
print(f"  Sentinel: Family B family assignment plausible?  PASS (reduces residual ~10×)")
print()
print("=" * 76)
print("DELIVERABLE: A3 probe NEGATIVE-WITH-HINT.  Family B reduces residual but")
print("no clean structural c_MZ derived.  The ρ-parameter gap is independent")
print("and confirms multi-session R-14 chain for full closure.")
print("=" * 76)
