#!/usr/bin/env python3
"""
Analytical calculation of baryon asymmetry η from framework parameters.

All inputs derived from toggle + MDL:
  - ε = 1/5 (Laves graph chirality: 9 CCW + 6 CW ten-cycles)
  - M_R = (2/3)^g × M_GUT (RH neutrino mass from girth attenuation)
  - g = 10 (Laves girth)
  - α_GUT = 1/24.1 (reconnection DL)
  - M_GUT from MSSM running of α_GUT

Standard leptogenesis formula (Buchmuller, Di Bari, Plumacher 2005):
  η_B = c_sph × ε₁ × κ / g_*

where:
  c_sph = 28/79 (sphaleron conversion factor, SM)
  ε₁ = CP asymmetry in RH neutrino decay
  κ = washout factor (efficiency)
  g_* = relativistic DOF at T = M_R

Also computes: N_eq, n_s, A_s for completeness.
"""

import math

print("=" * 80)
print("ANALYTICAL CALCULATION: BARYOGENESIS + COSMOLOGICAL PARAMETERS")
print("=" * 80)

# ============================================================================
# DERIVED FRAMEWORK PARAMETERS
# ============================================================================

k = 3                          # equilibrium valence
g = 10                         # Laves girth
n_g = 15                       # 10-cycles per vertex
n_cw = 6                       # clockwise
n_ccw = 9                      # counterclockwise
epsilon_chiral = (n_ccw - n_cw) / (n_ccw + n_cw)  # = 1/5

alpha_GUT = 2**(-math.log2(3) - 2 - 1)  # = 2^(-4.585) = 1/24.1

# Planck mass
M_P = 1.2209e19                # GeV (reduced: 2.435e18, but using non-reduced)
M_P_reduced = 2.435e18         # GeV

# GUT scale from MSSM running of α_GUT
M_GUT = 2.0e16                 # GeV (MSSM unification scale)

# RH neutrino mass from girth attenuation
M_R = ((k-1)/k)**g * M_GUT    # = (2/3)^10 × 2×10^16
M_R_alt = M_GUT / k**g * (k-1)**g

print("\n--- Framework inputs (all derived) ---")
print(f"  k* = {k}")
print(f"  g = {g}")
print(f"  ε_chiral = ({n_ccw}-{n_cw})/{n_g} = {epsilon_chiral:.4f} = 1/{1/epsilon_chiral:.0f}")
print(f"  α_GUT = 2^(-4.585) = 1/{1/alpha_GUT:.1f} = {alpha_GUT:.5f}")
print(f"  M_GUT = {M_GUT:.2e} GeV")
print(f"  M_R = (2/3)^{g} × M_GUT = {M_R:.3e} GeV")
print(f"  (2/3)^{g} = {((k-1)/k)**g:.6f}")

# ============================================================================
# 1. BARYOGENESIS VIA LEPTOGENESIS
# ============================================================================

print("\n" + "=" * 80)
print("1. BARYOGENESIS")
print("=" * 80)

# Standard leptogenesis (Fukugita-Yanagida 1986, Buchmuller+ 2005)
#
# The RH neutrino N₁ (mass M_R) decays: N₁ → l H and N₁ → l̄ H†
# CP asymmetry in the decay: ε₁
# Sphaleron conversion: lepton asymmetry → baryon asymmetry
# Washout: inverse decays + scattering reduce the asymmetry

# Step 1: CP asymmetry ε₁
# In the framework: ε₁ = ε_chiral = 1/5
# This is the RAW asymmetry from the Laves graph chirality.
# In standard leptogenesis, ε₁ comes from loop diagrams and is typically
# much smaller (~10⁻⁶). Our ε₁ = 1/5 is the PLANCK-SCALE asymmetry.
# The question is whether it survives to low energies.

epsilon_1 = epsilon_chiral
print(f"\n  CP asymmetry ε₁ = ε_chiral = {epsilon_1}")

# Step 2: Sphaleron conversion factor
# In SM: c_sph = 28/79 (converting L asymmetry to B asymmetry)
# In MSSM: c_sph = 8/23 (different due to extra Higgs doublet)
c_sph_SM = 28/79
c_sph_MSSM = 8/23
print(f"  Sphaleron conversion c_sph(SM) = 28/79 = {c_sph_SM:.4f}")
print(f"  Sphaleron conversion c_sph(MSSM) = 8/23 = {c_sph_MSSM:.4f}")

# Step 3: Relativistic DOF at T = M_R
# SM: g_* = 106.75
# MSSM: g_* = 228.75 (framework requires MSSM from α_GUT)
g_star_SM = 106.75
g_star_MSSM = 228.75
print(f"  g_*(SM) = {g_star_SM}")
print(f"  g_*(MSSM) = {g_star_MSSM}")

# Step 4: Washout factor κ
# Depends on the "effective neutrino mass" m̃₁ vs the "equilibrium mass" m_*
#
# m̃₁ = (Y_ν† Y_ν)₁₁ v² / M_R  (effective mass controlling washout)
# m_* = 16π^(5/2) √g_* v² / (3√5 M_P) ≈ 1.08×10⁻³ eV (equilibrium mass)
#
# From the framework: neutrino masses come from seesaw
#   m_ν₃ = y_t² v² / (2 M_R) ≈ 0.046 eV  (derived in completing_physics.md)
#
# The washout parameter K = m̃₁ / m_*
# For hierarchical RH neutrinos (M₁ << M₂ << M₃):
#   m̃₁ ~ m_ν₁ (lightest neutrino mass) for normal ordering

v = 246.22  # Higgs VEV in GeV

# Equilibrium neutrino mass (Davidson-Ibarra bound reference scale)
m_star = 16 * math.pi**(5/2) * math.sqrt(g_star_MSSM) * v**2 / (3 * math.sqrt(5) * M_P)
print(f"\n  Equilibrium mass m_* = {m_star:.3e} eV")
# Should be ~1.08e-3 eV

# Framework neutrino mass (from completing_physics.md)
m_nu3 = 0.046  # eV (derived: m_e × (2/3)^40 or from seesaw)
m_nu1 = 0.0    # lightest, normal ordering
m_nu2 = 0.0087 # eV (from Δm²₂₁)

# Effective washout mass: for normal ordering with m₁ ≈ 0
m_tilde_1 = m_nu1 if m_nu1 > 0 else 1e-4  # if m₁ ≈ 0, use a small value
# Actually, m̃₁ is NOT m_ν₁. It's the (1,1) element of Y_ν† Y_ν v²/M_R
# For a specific Yukawa texture, m̃₁ can differ from m_ν₁.
# In the framework's seesaw: m̃₁ ≈ m_ν₃ × |U_e3|² ≈ 0.046 × sin²θ₁₃ ≈ 0.046 × 0.022 = 0.001 eV

# From θ₁₃ = 8.54° (observed) or V_us/√2 = 9.1° (framework):
theta_13 = math.radians(8.54)
m_tilde_1_est = m_nu3 * math.sin(theta_13)**2
print(f"  m̃₁ ≈ m_ν₃ × sin²θ₁₃ = {m_nu3} × {math.sin(theta_13)**2:.4f} = {m_tilde_1_est:.4e} eV")

K = m_tilde_1_est / m_star
print(f"  Washout parameter K = m̃₁/m_* = {K:.3f}")

# Efficiency factor κ(K)
# For K << 1 (weak washout): κ ≈ K (efficiency proportional to K)
# For K >> 1 (strong washout): κ ≈ 0.3/(K × (ln K)^0.6)
# For K ~ 1: κ ~ 0.1-0.2

if K < 1:
    kappa = K  # weak washout
    regime = "weak"
elif K < 10:
    kappa = 1 / (2 * math.sqrt(K**2 + 9))  # interpolation
    regime = "intermediate"
else:
    kappa = 0.3 / (K * math.log(K)**0.6)  # strong washout
    regime = "strong"

print(f"  Washout regime: {regime}")
print(f"  Efficiency κ = {kappa:.4f}")

# Step 5: Baryon-to-photon ratio
# η_B = c_sph × ε₁ × κ / g_*
# Factor of 1/g_* from entropy dilution (number density / entropy density)

# But wait: the standard formula is:
# Y_B = n_B/s = c_sph × ε₁ × κ × (n_N1/s)|_eq
# where (n_N1/s)|_eq = 135 ζ(3) / (4π⁴ g_*) ≈ 3.9×10⁻³ / g_*
# η_B = n_B/n_γ = Y_B × s/n_γ = Y_B × 7.04

prefactor = 135 * 1.202 / (4 * math.pi**4)  # 135 ζ(3) / (4π⁴) ≈ 0.0417
n_over_s_eq = prefactor / g_star_MSSM
s_over_ngamma = 7.04  # entropy per photon

print(f"\n  (n_N₁/s)_eq = {n_over_s_eq:.4e}")

# Using MSSM parameters
Y_B = c_sph_MSSM * epsilon_1 * kappa * n_over_s_eq
eta_B = Y_B * s_over_ngamma

print(f"\n  Y_B = c_sph × ε₁ × κ × (n_N₁/s)_eq")
print(f"      = {c_sph_MSSM:.4f} × {epsilon_1:.4f} × {kappa:.4f} × {n_over_s_eq:.4e}")
print(f"      = {Y_B:.4e}")
print(f"\n  η_B = Y_B × s/n_γ = {Y_B:.4e} × {s_over_ngamma}")
print(f"      = {eta_B:.4e}")
print(f"\n  Observed: η_B = (6.12 ± 0.04) × 10⁻¹⁰")
print(f"  Ratio predicted/observed: {eta_B / 6.12e-10:.3f}")

# Sensitivity analysis
print("\n--- Sensitivity to m̃₁ ---")
for m_tilde in [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]:
    K_test = m_tilde / m_star
    if K_test < 1:
        kappa_test = K_test
    elif K_test < 10:
        kappa_test = 1 / (2 * math.sqrt(K_test**2 + 9))
    else:
        kappa_test = 0.3 / (K_test * math.log(K_test)**0.6)
    Y_test = c_sph_MSSM * epsilon_1 * kappa_test * n_over_s_eq
    eta_test = Y_test * s_over_ngamma
    print(f"  m̃₁ = {m_tilde:.1e} eV  K = {K_test:>8.3f}  κ = {kappa_test:.4f}  "
          f"η = {eta_test:.3e}  ratio = {eta_test/6.12e-10:.3f}")

# ============================================================================
# 2. N_eq (graph size at leptogenesis)
# ============================================================================

print("\n" + "=" * 80)
print("2. N_eq (GRAPH SIZE AT LEPTOGENESIS)")
print("=" * 80)

# Hubble parameter at T = M_R
# H = √(8π³g_*/90) × T²/M_P  (radiation domination)
H_at_MR = math.sqrt(8 * math.pi**3 * g_star_MSSM / 90) * M_R**2 / M_P
print(f"\n  H(T=M_R) = {H_at_MR:.3e} GeV")

# Convert to Planck units
H_planck = H_at_MR / M_P  # H in units of M_P
N_eq = 1 / H_planck  # N = 1/H in Planck units
print(f"  H in Planck units = {H_planck:.3e}")
print(f"  N_eq = 1/H = {N_eq:.3e}")

# Compare to completing_physics.md estimate
N_today = 8.5e60  # Hubble time in Planck times
print(f"  N_today = {N_today:.2e}")
print(f"  N_eq/N_today = {N_eq/N_today:.3e}")

# Temperature check
T_from_N = M_P / math.sqrt(N_eq * math.sqrt(90 / (8 * math.pi**3 * g_star_MSSM)))
print(f"  T_eq ≈ {M_R:.3e} GeV = {M_R/1e6:.1f} × 10⁶ GeV")

# ============================================================================
# 3. SPECTRAL INDEX n_s
# ============================================================================

print("\n" + "=" * 80)
print("3. SPECTRAL INDEX n_s")
print("=" * 80)

# From MSSM running: M_GUT → N_e → n_s
# Standard formula: N_e = 62 - ln(10¹⁶/M_GUT) - ln(V_end^1/4 / 10¹⁶)
# Simplified: N_e ≈ 62 for M_GUT ~ 2×10¹⁶

# More precisely: N_e depends on reheating temperature T_rh
# For T_rh ~ M_R ~ 3.5×10¹⁴ GeV:
# N_e = 62 - (1/4)ln(M_P⁴/(ρ_end)) + (1/4)ln(T_rh²M_P²/ρ_end)
# ≈ 62 + (1/2)ln(T_rh/M_GUT) for high-scale inflation

T_rh = M_R  # reheating at leptogenesis scale
N_e = 62 - 0.5 * math.log(M_GUT / T_rh)
# Actually the standard formula for single-field slow-roll:
# N_e = 67 - ln(V^{1/4}/10^16) + ln(V^{1/4}/T_rh) - 1/3 ln(T_rh V^{1/4}/...)
# Simplify: for V^{1/4} ~ M_GUT and T_rh ~ M_R:
N_e_simple = 62 - math.log(1e16 / M_GUT) + 0.5 * math.log(M_GUT / T_rh)
# Actually let me use the simpler version from the paper:
N_e_paper = 62.7  # from complete_physics_derivations.md

n_s_paper = 1 - 2/N_e_paper

# With T_rh = M_R:
N_e_with_Trh = 55 + (2/3) * math.log(T_rh / 1e9)  # approximate for T_rh >> 10⁹
n_s_with_Trh = 1 - 2/N_e_with_Trh

print(f"\n  From paper (N_e = 62.7): n_s = 1 - 2/{N_e_paper:.1f} = {n_s_paper:.4f}")
print(f"  With T_rh = M_R = {M_R:.2e} GeV:")
print(f"    N_e ≈ 55 + (2/3)ln(T_rh/10⁹) = 55 + {(2/3)*math.log(T_rh/1e9):.1f} = {N_e_with_Trh:.1f}")
print(f"    n_s = 1 - 2/{N_e_with_Trh:.1f} = {n_s_with_Trh:.4f}")
print(f"\n  Observed: n_s = 0.9649 ± 0.0042")
print(f"  Paper prediction: {n_s_paper:.4f} ({(n_s_paper - 0.9649)/0.0042:.1f}σ)")
print(f"  With M_R reheating: {n_s_with_Trh:.4f} ({(n_s_with_Trh - 0.9649)/0.0042:.1f}σ)")

# ============================================================================
# 4. CMB AMPLITUDE A_s
# ============================================================================

print("\n" + "=" * 80)
print("4. CMB AMPLITUDE A_s")
print("=" * 80)

# From complete_physics_derivations.md:
# A_s = α_GUT × (2/3)^g × (M_GUT/M_P)²
A_s_pred = alpha_GUT * ((k-1)/k)**g * (M_GUT/M_P)**2

print(f"\n  A_s = α_GUT × (2/3)^g × (M_GUT/M_P)²")
print(f"      = {alpha_GUT:.4f} × {((k-1)/k)**g:.4e} × ({M_GUT/M_P:.3e})²")
print(f"      = {alpha_GUT:.4f} × {((k-1)/k)**g:.4e} × {(M_GUT/M_P)**2:.3e}")
print(f"      = {A_s_pred:.3e}")
print(f"\n  Observed: A_s = (2.10 ± 0.03) × 10⁻⁹")
print(f"  Ratio: {A_s_pred / 2.10e-9:.3f}")

# ============================================================================
# 5. TENSOR-TO-SCALAR RATIO
# ============================================================================

print("\n" + "=" * 80)
print("5. TENSOR-TO-SCALAR RATIO r")
print("=" * 80)
print(f"\n  Predicted: r = 0 (no inflaton field, no coherent tensor source)")
print(f"  Current bound: r < 0.036 (BICEP/Keck 2021)")
print(f"  Future: LiteBIRD sensitivity r ~ 10⁻³ (2032)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: COSMOLOGICAL PARAMETERS FROM FRAMEWORK")
print("=" * 80)

print(f"""
  All from: k=3 (toggle+MDL), g=10 (Laves), α_GUT=1/24.1 (reconnection DL)

  | Parameter     | Framework          | Observed           | Match  |
  |---------------|--------------------|--------------------|--------|
  | ε (CP asym)   | 1/5 = 0.2000       | —                  | derived|
  | M_R           | {M_R:.3e} GeV  | ~10¹⁴ GeV (seesaw) | ✓      |
  | η_B           | {eta_B:.3e}      | 6.12 × 10⁻¹⁰      | {eta_B/6.12e-10:.2f}×  |
  | n_s           | {n_s_paper:.4f}          | 0.9649 ± 0.0042    | {abs(n_s_paper-0.9649)/0.0042:.1f}σ    |
  | A_s           | {A_s_pred:.2e}     | 2.10 × 10⁻⁹       | {A_s_pred/2.10e-9:.1%}   |
  | r             | 0                  | < 0.036            | ✓      |
  | Λ             | 3/N² = 2.83e-122   | 2.85e-122          | 1%     |
  | Ω_DM/Ω_m     | 0.849              | 0.846              | 0.4%   |

  Key: η_B depends on m̃₁ (effective washout mass). The table above uses
  m̃₁ = m_ν₃ × sin²θ₁₃ = {m_tilde_1_est:.4e} eV.
  See sensitivity table for the range.
""")

# Flag if eta is off
if abs(math.log10(eta_B / 6.12e-10)) > 1:
    print(f"  ⚠ η_B is off by {eta_B/6.12e-10:.1f}×. The washout mass m̃₁ may need")
    print(f"    revision. The framework predicts m̃₁ from the neutrino Yukawa")
    print(f"    texture, which depends on the Laves lattice Green's function")
    print(f"    (Tier 1 item #3).")
