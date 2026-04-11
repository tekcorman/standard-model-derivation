#!/usr/bin/env python3
"""
GUT-scale baryogenesis from the Laves graph (srs net).

KEY INSIGHT: Baryon-violating reconnections are IN equilibrium (Gamma_B >> H)
all the way down to T ~ alpha_GUT * M_P ~ 5e17 GeV, which is ABOVE M_GUT.
Standard leptogenesis at T = M_R fails due to extreme washout (K ~ 10^8).
Topological suppression also fails (local reconnections always create short cycles).

Therefore: the asymmetry must freeze out at or near the GUT/Planck scale.

Framework parameters (all derived from toggle + MDL + Laves graph):
  epsilon = 1/5          (CP asymmetry: 9 CCW + 6 CW ten-cycles per vertex)
  alpha_GUT = 1/24.1     (reconnection probability from description length)
  M_GUT = 2e16 GeV       (MSSM unification)
  M_P = 1.22e19 GeV      (Planck mass)
  g_* = 228.75           (MSSM relativistic DOF)

Three approaches computed:
  1. Freeze-out baryogenesis (Gamma_B = H defines T_freeze)
  2. Gravitational baryogenesis (chiral graph = gravitational CP violation)
  3. GUT-scale out-of-equilibrium decay (direct analog of GUT baryogenesis)
"""

import math

# =============================================================================
# FRAMEWORK PARAMETERS
# =============================================================================

k = 3                                  # equilibrium valence
g = 10                                 # Laves girth
n_ccw, n_cw = 9, 6                    # ten-cycles per vertex
epsilon = (n_ccw - n_cw) / (n_ccw + n_cw)  # = 1/5

alpha_GUT = 2**(-math.log2(3) - 2 - 1)     # = 1/24.1
M_P = 1.2209e19                              # GeV (non-reduced Planck mass)
M_P_red = M_P / math.sqrt(8 * math.pi)      # reduced Planck mass
M_GUT = 2.0e16                               # GeV
g_star = 228.75                              # MSSM DOF

# Seesaw parameters
M_R = ((k - 1) / k)**g * M_GUT              # RH neutrino mass

# Sphaleron conversion
c_sph = 8 / 23                               # MSSM

# Observed
eta_obs = 6.12e-10

print("=" * 78)
print("GUT-SCALE BARYOGENESIS FROM THE LAVES GRAPH")
print("=" * 78)

print(f"""
Framework parameters:
  epsilon  = {epsilon}  (CP asymmetry from Laves chirality)
  alpha_GUT = {alpha_GUT:.5f} = 1/{1/alpha_GUT:.1f}
  M_P      = {M_P:.4e} GeV
  M_GUT    = {M_GUT:.2e} GeV
  M_R      = {M_R:.3e} GeV  [(2/3)^10 * M_GUT]
  g_*      = {g_star}  (MSSM)
  c_sph    = {c_sph:.4f} = 8/23  (MSSM)
""")

# =============================================================================
# 0. WHY STANDARD LEPTOGENESIS FAILS
# =============================================================================

print("=" * 78)
print("0. WHY STANDARD LEPTOGENESIS AT T = M_R FAILS")
print("=" * 78)

# Hubble rate at T = M_R
H_MR = math.sqrt(8 * math.pi**3 * g_star / 90) * M_R**2 / M_P
print(f"\n  H(T = M_R) = {H_MR:.3e} GeV")

# Reconnection rate at T = M_R: Gamma_B ~ n_reconnect * alpha_GUT * T
# On the Laves graph, each vertex has k=3 edges, so the reconnection
# site density is ~ T^3 (one per thermal volume).
# Rate per unit volume: Gamma_B ~ alpha_GUT * T^4
# Rate per Hubble volume: Gamma_B ~ alpha_GUT * T
Gamma_B_MR = alpha_GUT * M_R
K_washout = Gamma_B_MR / H_MR
print(f"  Gamma_B(T = M_R) ~ alpha_GUT * T = {Gamma_B_MR:.3e} GeV")
print(f"  K = Gamma_B / H = {K_washout:.2e}")
print(f"\n  K >> 1: EXTREME washout. Any asymmetry generated at T = M_R")
print(f"  is washed out by ~exp(-K) ~ exp(-{K_washout:.0e}) = 0.")
print(f"  Standard leptogenesis FAILS.")

# =============================================================================
# 1. FREEZE-OUT BARYOGENESIS
# =============================================================================

print(f"\n{'=' * 78}")
print("1. FREEZE-OUT BARYOGENESIS (Gamma_B = H defines T_freeze)")
print("=" * 78)

# The baryon-violating rate per Hubble volume:
#   Gamma_B = alpha_GUT * T
#
# The Hubble rate:
#   H = sqrt(8*pi^3*g_*/90) * T^2 / M_P = 1.66 * sqrt(g_*) * T^2 / M_P
#
# Note: 1.66 * sqrt(g_*) = sqrt(8*pi^3*g_*/90) * M_P / M_P
# Let's verify: sqrt(8*pi^3/90) = sqrt(8*31.006/90) = sqrt(2.756) = 1.660
# Yes, H = 1.66 * sqrt(g_*) * T^2 / M_P

H_prefactor = 1.66 * math.sqrt(g_star)  # = 1.66 * 15.12 = 25.10

# Freeze-out: Gamma_B(T_f) = H(T_f)
#   alpha_GUT * T_f = H_prefactor * T_f^2 / M_P
#   T_f = alpha_GUT * M_P / H_prefactor

T_freeze = alpha_GUT * M_P / H_prefactor
print(f"\n  Freeze-out condition: Gamma_B = H")
print(f"    alpha_GUT * T_f = 1.66 * sqrt(g_*) * T_f^2 / M_P")
print(f"    T_f = alpha_GUT * M_P / (1.66 * sqrt(g_*))")
print(f"    T_f = {alpha_GUT:.5f} * {M_P:.4e} / {H_prefactor:.2f}")
print(f"    T_f = {T_freeze:.4e} GeV")
print(f"\n  Compare: M_GUT = {M_GUT:.2e} GeV")
print(f"           T_f / M_GUT = {T_freeze / M_GUT:.1f}")
print(f"           T_f / M_P   = {T_freeze / M_P:.4f}")

# Verify: Gamma_B = H at T_freeze
H_at_Tf = H_prefactor * T_freeze**2 / M_P
Gamma_at_Tf = alpha_GUT * T_freeze
print(f"\n  Check: H(T_f) = {H_at_Tf:.4e} GeV")
print(f"         Gamma(T_f) = {Gamma_at_Tf:.4e} GeV")
print(f"         Ratio = {Gamma_at_Tf / H_at_Tf:.6f}")

# The asymmetry generated at freeze-out:
# Y_B = n_B / s where s = (2*pi^2/45) * g_* * T^3
# At freeze-out, the asymmetry per entropy is:
#   Y_B = epsilon * n_eq / s
# where n_eq = zeta(3) * T^3 / pi^2 (for each bosonic species)
# and s = (2*pi^2/45) * g_* * T^3
# So n_eq / s = 45 * zeta(3) / (2 * pi^4 * g_*)

zeta3 = 1.20206
n_over_s = 45 * zeta3 / (2 * math.pi**4 * g_star)
print(f"\n  n_eq/s = 45*zeta(3) / (2*pi^4*g_*) = {n_over_s:.5e}")

# But this is the equilibrium abundance at freeze-out.
# The asymmetry is epsilon * (departure from equilibrium).
# At freeze-out (Gamma = H), the departure is O(1),
# so Y_B ~ epsilon * n_eq/s
# More precisely, the Boltzmann equation gives:
# Y_B = epsilon * n_eq/s * (H/Gamma) evaluated at T slightly above T_f
# Since H/Gamma = 1 at T_f, and the asymmetry is generated over
# Delta T ~ T_f (one Hubble time), we get Y_B ~ epsilon * n_eq/s

Y_B_freeze = epsilon * n_over_s
eta_B_freeze = Y_B_freeze * 7.04  # s/n_gamma = 7.04
print(f"\n  Y_B = epsilon * n_eq/s = {epsilon} * {n_over_s:.5e}")
print(f"      = {Y_B_freeze:.5e}")
print(f"\n  eta_B = Y_B * (s/n_gamma) = {Y_B_freeze:.5e} * 7.04")
print(f"        = {eta_B_freeze:.5e}")
print(f"\n  Observed: eta_B = {eta_obs:.2e}")
print(f"  Ratio pred/obs: {eta_B_freeze / eta_obs:.2f}")

print(f"\n  PROBLEM: This gives eta_B ~ 1/(5 * g_*) ~ 1e-3, WAY too large.")
print(f"  The freeze-out is at T ~ 10^{math.log10(T_freeze):.1f} GeV, "
      f"too high for sufficient dilution.")

# =============================================================================
# 1b. FREEZE-OUT WITH SPHALERON CONVERSION
# =============================================================================

print(f"\n{'=' * 78}")
print("1b. FREEZE-OUT WITH SPHALERON CONVERSION AND ENTROPY DILUTION")
print("=" * 78)

# The raw asymmetry Y_B ~ epsilon / g_* needs additional suppression.
# Possible sources:
# (a) Sphaleron conversion: factor c_sph = 8/23
# (b) The reconnection creates a LEPTON asymmetry, not baryon directly
#     -> needs sphaleron redistribution
# (c) Entropy dilution from GUT-scale phase transition or inflaton decay

# With sphaleron:
Y_B_sph = c_sph * epsilon * n_over_s
eta_B_sph = Y_B_sph * 7.04
print(f"\n  With sphaleron: Y_B = c_sph * epsilon * n_eq/s")
print(f"    = {c_sph:.4f} * {epsilon} * {n_over_s:.5e} = {Y_B_sph:.5e}")
print(f"    eta_B = {eta_B_sph:.5e}  (ratio = {eta_B_sph/eta_obs:.1f})")

# Additional dilution from the number of reconnection species
# On the Laves graph, each vertex has k=3 edges. The number of
# reconnection channels that violate baryon number vs total:
# In GUT theories, B-violation comes from X,Y boson exchange.
# The SU(5) has 12 X,Y bosons among 24 gauge bosons.
# Fraction: 12/24 = 1/2. But we need the fraction that produces
# net baryon number, considering all channels.

# In the Laves framework: each edge carries a gauge quantum number.
# B-violating reconnections are those that change the cycle structure
# (topology change). Not all reconnections change baryon number.

# The key suppression: at T = T_freeze >> M_GUT, ALL gauge bosons are
# in equilibrium. The B-violating processes must DECOUPLE from
# the B-conserving thermal bath. The efficiency is:
#   kappa = (Gamma_B_violating / Gamma_total) at T_freeze

# On the Laves graph: each vertex has 3 edges, so 3 reconnection
# channels per vertex. Of these, the B-violating ones involve
# changing the winding of ten-cycles. The number of ten-cycles
# per vertex is 15, and each reconnection affects ~2 of them.
# But the TOTAL scattering rate is alpha_GUT * T (same as B-violating).
# So kappa ~ 1 (no kinematic suppression at T >> M_GUT).

print(f"\n  No kinematic suppression at T >> M_GUT: kappa ~ 1")
print(f"  The ~ 10^3 ratio must come from elsewhere.")

# =============================================================================
# 2. GRAVITATIONAL BARYOGENESIS
# =============================================================================

print(f"\n{'=' * 78}")
print("2. GRAVITATIONAL BARYOGENESIS")
print("=" * 78)

# Reference: Davoudiasl, Kitano, Kribs, Murayama, Steinhardt (2004)
# The idea: a coupling (1/M_*^2) * (d_mu R) * J^mu_B generates baryon
# asymmetry during the radiation-dominated era when R-dot != 0.
#
# On the Laves graph: the chirality of the graph provides a natural
# coupling between the Ricci scalar and the baryon current.
# The coupling scale is M_* ~ M_P / epsilon (where epsilon = 1/5 is
# the chiral asymmetry).
#
# Standard result: Y_B = -15 * g_b / (4 * pi^2 * g_*) * T_D / M_*^2 * dR/dt
# where T_D is the decoupling temperature and g_b = 1 (baryon species).

# In radiation domination: R = 0 (conformally flat).
# BUT: during a phase transition (GUT or EW), R-dot != 0.
# For a first-order GUT phase transition:
#   R ~ H^2 ~ T^4 / M_P^2
#   dR/dt ~ H * R ~ T^6 / M_P^3

# The chiral Laves graph provides M_* = M_P / sqrt(epsilon)
# (the chirality enters at the Planck scale)
M_star = M_P / math.sqrt(epsilon)

# Decoupling temperature for B-violation:
T_D = T_freeze  # same as before

# Y_B from gravitational baryogenesis (Davoudiasl et al. 2004, eq. 11):
# Y_B = (15 * g_b) / (4 * pi^2 * g_*) * R_dot / (M_*^2 * H * T)
# evaluated at T = T_D
# R_dot during radiation: R = 6(H_dot + 2H^2) ≈ -6H^2 (radiation era)
# R_dot = -12 * H * H_dot = -12 * H * (-H^2) = 12 * H^3
# Wait: in radiation era, a ~ t^{1/2}, H = 1/(2t), H_dot = -1/(2t^2) = -2H^2
# So R = 6(H_dot + 2H^2) = 6(-2H^2 + 2H^2) = 0. Exactly zero!
# R is exactly zero in pure radiation domination.

# The asymmetry requires departure from pure radiation:
# During the GUT phase transition, the trace anomaly gives R != 0.
# R ~ g_* * T_GUT^4 / M_P^2 * (Delta_g / g_*) where Delta_g is the
# change in DOF across the transition.

# For MSSM GUT transition: Delta_g ~ 24 (the X,Y bosons becoming massive)
# going from g_* = 228.75 to g_* = 228.75 - 24 = 204.75
Delta_g = 24  # X,Y boson DOF
T_GUT = M_GUT

# During the transition, R ~ Delta_g/g_* * T^4/M_P^2 * (something)
# More carefully: the trace of T_mu_nu = rho - 3p
# For massive particles at T ~ M: rho - 3p ~ M^2 * T^2 / (2*pi^2)
# R = 8*pi*G * (rho - 3p) = (rho - 3p) / M_P_red^2

# The derivative: dR/dt ~ R * H ~ Delta_g/g_* * T^4/M_P^2 * H
# = Delta_g/g_* * T^4/M_P^2 * T^2/M_P
# = Delta_g/g_* * T^6/M_P^3

R_dot = (Delta_g / g_star) * T_GUT**6 / M_P**3
H_GUT = H_prefactor * T_GUT**2 / M_P

g_b = 1  # baryon species

Y_B_grav = (15 * g_b) / (4 * math.pi**2 * g_star) * R_dot / (M_star**2 * H_GUT * T_GUT)
eta_B_grav = Y_B_grav * 7.04

print(f"\n  M_* = M_P / sqrt(epsilon) = {M_star:.4e} GeV")
print(f"  T_D = T_freeze = {T_D:.4e} GeV")
print(f"  Delta_g = {Delta_g} (X,Y boson DOF)")
print(f"  R_dot ~ (Delta_g/g_*) * T_GUT^6 / M_P^3 = {R_dot:.4e} GeV^4")
print(f"  H(T_GUT) = {H_GUT:.4e} GeV")
print(f"\n  Y_B = 15/(4*pi^2*g_*) * R_dot / (M_*^2 * H * T)")
print(f"      = {Y_B_grav:.4e}")
print(f"  eta_B = {eta_B_grav:.4e}")
print(f"  Ratio pred/obs: {eta_B_grav / eta_obs:.2e}")

print(f"\n  Gravitational baryogenesis gives a TINY number.")
print(f"  The (T_GUT/M_P)^4 suppression is too severe.")

# =============================================================================
# 3. GUT-SCALE OUT-OF-EQUILIBRIUM DECAY
# =============================================================================

print(f"\n{'=' * 78}")
print("3. GUT-SCALE OUT-OF-EQUILIBRIUM DECAY (X,Y boson analog)")
print("=" * 78)

# Classic GUT baryogenesis (Yoshimura 1978, Weinberg 1979):
# Heavy gauge bosons X,Y (mass ~ M_GUT) decay with CP violation.
# If they decouple while still abundant, their decays generate Y_B.
#
# On the Laves graph: the "X,Y bosons" are edge excitations at
# the GUT scale. Their mass is M_GUT. Their decay channels have
# CP asymmetry epsilon = 1/5 from the graph chirality.
#
# Y_B = epsilon * n_X/s at T ~ M_GUT
# n_X/s = (45 * zeta(3)) / (2 * pi^4 * g_*) per species (if in equilibrium)
# Number of X,Y species: in SU(5), there are 12 (X, X-bar, Y, Y-bar,
# times 3 colors). On the Laves graph with k=3, there are k*(k-1)/2 = 3
# independent edge-pair channels per vertex.

# But wait: the CRITICAL question is whether X,Y decouple BEFORE they
# decay. The condition is: Gamma_decay < H at T ~ M_GUT.
# Gamma_decay ~ alpha_GUT * M_GUT (perturbative decay rate)
# H(M_GUT) = H_prefactor * M_GUT^2 / M_P

Gamma_decay_X = alpha_GUT * M_GUT
H_at_MGUT = H_prefactor * M_GUT**2 / M_P

print(f"\n  Gamma_X = alpha_GUT * M_GUT = {Gamma_decay_X:.4e} GeV")
print(f"  H(M_GUT) = {H_at_MGUT:.4e} GeV")
print(f"  Gamma_X / H(M_GUT) = {Gamma_decay_X / H_at_MGUT:.2f}")

# If Gamma_X / H > 1: decays are fast, X bosons decay in equilibrium
# and no net asymmetry survives (washout).
# If Gamma_X / H < 1: X bosons are still around when they decouple,
# their out-of-equilibrium decays produce the asymmetry.

ratio_GH = Gamma_decay_X / H_at_MGUT

if ratio_GH > 1:
    print(f"\n  Gamma_X / H = {ratio_GH:.1f} > 1: X decays are in equilibrium.")
    print(f"  Need washout suppression factor.")

    # Even though decays are fast, there IS a departure from equilibrium
    # proportional to H/Gamma. The Boltzmann equation gives:
    # Y_B = epsilon * (H / Gamma_X) * n_eq/s
    # This is the "weak washout" regime applied at the GUT scale.

    suppression = H_at_MGUT / Gamma_decay_X
    Y_B_gut = epsilon * suppression * n_over_s
    eta_B_gut = Y_B_gut * 7.04

    print(f"\n  Departure from equilibrium: delta = H/Gamma = {suppression:.4e}")
    print(f"  Y_B = epsilon * (H/Gamma) * n_eq/s")
    print(f"      = {epsilon} * {suppression:.4e} * {n_over_s:.5e}")
    print(f"      = {Y_B_gut:.5e}")
    print(f"  eta_B = {eta_B_gut:.5e}")
    print(f"  Ratio pred/obs: {eta_B_gut / eta_obs:.3f}")
else:
    # Out of equilibrium: full epsilon * n/s
    Y_B_gut = epsilon * n_over_s
    eta_B_gut = Y_B_gut * 7.04
    print(f"\n  Gamma_X / H = {ratio_GH:.4f} < 1: X decays OUT of equilibrium!")
    print(f"  Y_B = epsilon * n_eq/s = {Y_B_gut:.5e}")
    print(f"  eta_B = {eta_B_gut:.5e}")
    print(f"  Ratio pred/obs: {eta_B_gut / eta_obs:.3f}")

# =============================================================================
# 3b. FULL GUT BARYOGENESIS FORMULA
# =============================================================================

print(f"\n{'=' * 78}")
print("3b. FULL FORMULA: GUT BARYOGENESIS WITH ALL FACTORS")
print("=" * 78)

# The complete formula for GUT baryogenesis:
#   Y_B = epsilon_CP * kappa_washout * n_eq/s * N_species / g_*_dilution
#
# Let's be more careful about each factor:

# (a) CP asymmetry: epsilon = 1/5 (from Laves chirality)
# (b) Number of B-violating species: N_X
#     In SU(5): 12 X,Y gauge bosons. On Laves: k*(k-1) = 6 reconnection
#     channels per vertex (ordered pairs of distinct edges).
#     But each produces baryon number +-1/3 (fractional, from color).
#     Net: N_eff = 6 * (1/3) = 2 per vertex.
#     Actually let's use the standard SU(5) value: N_X = 12.
#     Per species: n_X/s = (45*zeta(3))/(2*pi^4*g_*) = n_over_s

# (c) Washout: kappa = min(1, H/Gamma_decay)
#     When Gamma >> H (our case): kappa ~ H/Gamma

# (d) Dilution from subsequent entropy production
#     If there's a GUT phase transition releasing entropy, dilution factor D.
#     For MSSM GUT with first-order transition:
#     D ~ (g_*(T_GUT) / g_*(T_below))^{1/3}
#     This is O(1) for a continuous transition.
#     For inflaton decay reheating to T_rh < M_GUT, D ~ (T_rh/M_GUT)^3.
#     We'll consider both cases.

# Case A: No additional dilution (continuous transition or T_rh > M_GUT)
print(f"\n  Case A: No additional entropy dilution")
kappa_A = min(1.0, H_at_MGUT / Gamma_decay_X)
Y_B_A = epsilon * kappa_A * n_over_s
eta_B_A = Y_B_A * 7.04
print(f"    kappa = H/Gamma = {kappa_A:.4e}")
print(f"    Y_B = {epsilon} * {kappa_A:.4e} * {n_over_s:.5e} = {Y_B_A:.5e}")
print(f"    eta_B = {eta_B_A:.5e}")
print(f"    Ratio: {eta_B_A / eta_obs:.3f}")

# Case B: Dilution from reheating at T_rh = M_R (seesaw scale)
# If inflation ends and reheating is at T_rh ~ M_R:
T_rh = M_R
D_rh = (T_rh / M_GUT)  # dilution ~ T_rh/T_production for radiation era
# Actually: entropy dilution from a late decay is D = (T_rh / T_prod)^3
# if the entropy is produced by inflaton decay.
# But if the asymmetry is produced at T_GUT and we just have radiation
# down to T_rh, there is NO additional dilution (entropy is conserved).
# The dilution only happens if there is a matter-dominated epoch.

# For reheating after inflation:
# Y_B is computed at T = T_rh (when radiation domination starts),
# and the asymmetry produced at T_GUT is diluted by the entropy
# from inflaton decay. D = T_rh / T_GUT (approximately).
D_infl = T_rh / T_freeze  # dilution from inflaton entropy
print(f"\n  Case B: Post-inflationary dilution to T_rh = M_R = {M_R:.3e} GeV")
print(f"    D = T_rh / T_freeze = {D_infl:.4e}")
Y_B_B = epsilon * kappa_A * n_over_s * D_infl
eta_B_B = Y_B_B * 7.04
print(f"    Y_B = {Y_B_B:.5e}")
print(f"    eta_B = {eta_B_B:.5e}")
print(f"    Ratio: {eta_B_B / eta_obs:.3f}")

# =============================================================================
# 4. THE CORRECT APPROACH: Boltzmann equation at GUT scale
# =============================================================================

print(f"\n{'=' * 78}")
print("4. BOLTZMANN EQUATION AT T ~ M_GUT")
print("=" * 78)

# The Boltzmann equation for Y_B in the expanding universe:
#   dY_B/dt = epsilon * Gamma_prod * (Y_eq - Y_X) - Gamma_wash * Y_B
#
# At T >> M_X (above GUT scale): Y_X = Y_eq, so dY_B/dt = -Gamma_wash * Y_B
# -> asymmetry washed out.
#
# At T ~ M_X: X bosons start becoming Boltzmann-suppressed.
# Y_eq(T < M_X) ~ (M_X/T)^{3/2} * exp(-M_X/T)
# The departure from equilibrium: Delta_Y = Y_X - Y_eq ~ Y_eq * H/Gamma
# This generates:
#   Y_B ~ epsilon * (Y_eq * H/Gamma) evaluated at T ~ M_X
#
# More precisely, integrating the Boltzmann equation:
#   Y_B = epsilon * n_eq/s * f(K)
# where K = Gamma/H at T = M_X and f(K) is the efficiency function.
#
# For K >> 1 (strong washout): f(K) ~ 1/K
# For K << 1 (weak washout): f(K) ~ 1
# For K ~ 1 (optimal): f(K) ~ 0.3

K_GUT = Gamma_decay_X / H_at_MGUT
print(f"\n  K = Gamma_X / H at T = M_GUT = {K_GUT:.2f}")

if K_GUT > 10:
    f_K = 0.3 / (K_GUT * math.log(K_GUT)**0.6)
    regime = "strong"
elif K_GUT > 1:
    f_K = 1 / (2 * math.sqrt(K_GUT**2 + 9))
    regime = "intermediate"
else:
    f_K = K_GUT
    regime = "weak"

print(f"  Regime: {regime}")
print(f"  f(K) = {f_K:.5f}")

Y_B_boltz = epsilon * n_over_s * f_K
eta_B_boltz = Y_B_boltz * 7.04

print(f"\n  Y_B = epsilon * (n_eq/s) * f(K)")
print(f"      = {epsilon} * {n_over_s:.5e} * {f_K:.5f}")
print(f"      = {Y_B_boltz:.5e}")
print(f"  eta_B = {eta_B_boltz:.5e}")
print(f"  Ratio pred/obs: {eta_B_boltz / eta_obs:.3f}")

# =============================================================================
# 5. PUTTING IT TOGETHER: The graph-derived formula
# =============================================================================

print(f"\n{'=' * 78}")
print("5. THE GRAPH-DERIVED FORMULA")
print("=" * 78)

# The cleanest formula uses only framework parameters:
#
# eta_B = 7.04 * c_sph * epsilon * f(K) * 45*zeta(3) / (2*pi^4*g_*)
#
# where:
#   epsilon = 1/5  (Laves chirality)
#   K = alpha_GUT * M_P / (1.66 * sqrt(g_*) * M_GUT)
#     = alpha_GUT * M_P / (H_prefactor * M_GUT)  <-- NOTE: this is Gamma/H at M_GUT
#   c_sph = 8/23  (MSSM sphaleron conversion)
#   g_* = 228.75  (MSSM)

K_formula = alpha_GUT * M_P / (H_prefactor * M_GUT)
print(f"\n  K = alpha_GUT * M_P / (1.66 * sqrt(g_*) * M_GUT)")
print(f"    = {alpha_GUT:.5f} * {M_P:.4e} / ({H_prefactor:.2f} * {M_GUT:.2e})")
print(f"    = {K_formula:.4f}")

# For K ~ 1, use the interpolation formula
if K_formula > 10:
    f_K_formula = 0.3 / (K_formula * math.log(K_formula)**0.6)
elif K_formula > 0.1:
    f_K_formula = 1 / (2 * math.sqrt(K_formula**2 + 9))
else:
    f_K_formula = K_formula

print(f"  f(K) = {f_K_formula:.5f}")

eta_B_formula = 7.04 * c_sph * epsilon * f_K_formula * n_over_s
print(f"\n  eta_B = 7.04 * c_sph * epsilon * f(K) * 45*zeta(3)/(2*pi^4*g_*)")
print(f"       = 7.04 * {c_sph:.4f} * {epsilon} * {f_K_formula:.5f} * {n_over_s:.5e}")
print(f"       = {eta_B_formula:.5e}")
print(f"\n  Observed: eta_B = {eta_obs:.2e}")
print(f"  Ratio pred/obs: {eta_B_formula / eta_obs:.4f}")

# =============================================================================
# 6. SENSITIVITY ANALYSIS
# =============================================================================

print(f"\n{'=' * 78}")
print("6. SENSITIVITY ANALYSIS")
print("=" * 78)

print(f"\n  Varying alpha_GUT (the least certain parameter):")
print(f"  {'alpha_GUT':>12} {'1/alpha':>8} {'K':>8} {'f(K)':>10} "
      f"{'eta_B':>12} {'ratio':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")

for alpha_test in [1/30, 1/25, alpha_GUT, 1/24, 1/20, 1/15, 1/10]:
    K_test = alpha_test * M_P / (H_prefactor * M_GUT)
    if K_test > 10:
        f_test = 0.3 / (K_test * math.log(K_test)**0.6)
    elif K_test > 0.1:
        f_test = 1 / (2 * math.sqrt(K_test**2 + 9))
    else:
        f_test = K_test
    eta_test = 7.04 * c_sph * epsilon * f_test * n_over_s
    print(f"  {alpha_test:>12.5f} {1/alpha_test:>8.1f} {K_test:>8.2f} {f_test:>10.5f} "
          f"{eta_test:>12.4e} {eta_test/eta_obs:>8.3f}")

print(f"\n  Varying M_GUT:")
print(f"  {'M_GUT (GeV)':>14} {'K':>8} {'f(K)':>10} {'eta_B':>12} {'ratio':>8}")
print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")

for M_test in [5e15, 1e16, 2e16, 5e16, 1e17]:
    K_test = alpha_GUT * M_P / (H_prefactor * M_test)
    if K_test > 10:
        f_test = 0.3 / (K_test * math.log(K_test)**0.6)
    elif K_test > 0.1:
        f_test = 1 / (2 * math.sqrt(K_test**2 + 9))
    else:
        f_test = K_test
    eta_test = 7.04 * c_sph * epsilon * f_test * n_over_s
    print(f"  {M_test:>14.2e} {K_test:>8.2f} {f_test:>10.5f} "
          f"{eta_test:>12.4e} {eta_test/eta_obs:>8.3f}")

# =============================================================================
# 7. WHAT WOULD MAKE IT WORK
# =============================================================================

print(f"\n{'=' * 78}")
print("7. WHAT WOULD MAKE eta_B MATCH OBSERVATION")
print("=" * 78)

# We need eta_B = 6.12e-10
# eta_B = 7.04 * c_sph * epsilon * f(K) * n_eq/s
# = 7.04 * (8/23) * (1/5) * f(K) * 45*zeta(3)/(2*pi^4 * g_*)
# = 7.04 * 0.3478 * 0.2 * f(K) * 1.825e-4
# = 8.963e-5 * f(K)

coeff = 7.04 * c_sph * epsilon * n_over_s
print(f"\n  eta_B = {coeff:.4e} * f(K)")
print(f"  Need f(K) = {eta_obs / coeff:.5e}")

f_needed = eta_obs / coeff
print(f"\n  Required f(K) = {f_needed:.5e}")
print(f"  Current  f(K) = {f_K_formula:.5e} at K = {K_formula:.2f}")
print(f"  Need f to be {f_needed / f_K_formula:.4f} of current value")

# In the strong washout regime: f(K) ~ 0.3 / (K * ln(K)^0.6)
# Need K such that 0.3 / (K * ln(K)^0.6) = f_needed
# K ~ 0.3 / f_needed ~ 0.3 / 6.8e-6 ~ 44000
# Check: f(44000) = 0.3 / (44000 * ln(44000)^0.6) = 0.3 / (44000 * 5.3) = 1.3e-6
# Hmm, need to solve numerically.

# Binary search for K
lo, hi = 1.0, 1e10
for _ in range(100):
    mid = math.sqrt(lo * hi)
    f_mid = 0.3 / (mid * math.log(mid)**0.6)
    if f_mid > f_needed:
        lo = mid
    else:
        hi = mid
K_needed = math.sqrt(lo * hi)
f_check = 0.3 / (K_needed * math.log(K_needed)**0.6)

print(f"\n  Need K = {K_needed:.0f} (currently K = {K_formula:.2f})")
print(f"  Verify: f({K_needed:.0f}) = {f_check:.5e}")

# What alpha_GUT gives this K?
# K = alpha_GUT * M_P / (H_prefactor * M_GUT)
alpha_needed = K_needed * H_prefactor * M_GUT / M_P
print(f"\n  This requires alpha_GUT = {alpha_needed:.4f} = 1/{1/alpha_needed:.1f}")
print(f"  Current alpha_GUT = {alpha_GUT:.5f} = 1/{1/alpha_GUT:.1f}")
print(f"  Or equivalently, Gamma_B must be {K_needed/K_formula:.0f}x larger.")

# Alternative: additional dilution factor
D_needed = eta_B_formula / eta_obs
print(f"\n  Or: additional entropy dilution D = {D_needed:.1f}")
print(f"  e.g., from moduli decay or late inflaton decay")

# =============================================================================
# 8. THE (2/3)^10 CONNECTION
# =============================================================================

print(f"\n{'=' * 78}")
print("8. THE SEESAW ATTENUATION (2/3)^g AS DILUTION")
print("=" * 78)

# Interesting: M_R = (2/3)^10 * M_GUT
# (2/3)^10 = 0.01734
# If we interpret this as a dilution factor for the asymmetry:
# eta_B = coeff * f(K) * (2/3)^g

attenuation = ((k - 1) / k)**g
eta_B_with_atten = coeff * f_K_formula * attenuation
print(f"\n  (2/3)^{g} = {attenuation:.5e}")
print(f"  eta_B = {coeff:.4e} * {f_K_formula:.5e} * {attenuation:.5e}")
print(f"        = {eta_B_with_atten:.5e}")
print(f"  Ratio pred/obs: {eta_B_with_atten / eta_obs:.4f}")

# Physical interpretation: the asymmetry is generated at T ~ M_GUT
# but the B-violating interactions remain in equilibrium down to
# T ~ M_R = (2/3)^10 * M_GUT. Between M_GUT and M_R, the asymmetry
# is partially washed out by a factor of (2/3)^g.
#
# This is the girth attenuation: each step along the 10-cycle
# path reduces the coherent asymmetry by (k-1)/k = 2/3.

print(f"\n  Physical picture: the asymmetry is generated at T ~ M_GUT")
print(f"  with epsilon = 1/5, but it propagates through g = {g} steps of")
print(f"  the Laves graph before becoming a physical baryon number.")
print(f"  Each step attenuates by (k-1)/k = 2/3 (branching ratio at")
print(f"  each trivalent vertex). Total: (2/3)^{g} = {attenuation:.5e}.")

# =============================================================================
# 9. COMBINED: GUT BARYOGENESIS WITH GIRTH ATTENUATION
# =============================================================================

print(f"\n{'=' * 78}")
print("9. COMBINED RESULT: epsilon * (2/3)^g * f(K)")
print("=" * 78)

# Full formula:
#   eta_B = 7.04 * c_sph * [epsilon * (2/3)^g] * f(K) * n_eq/s
#
# The effective CP asymmetry is NOT epsilon = 1/5, but
#   epsilon_eff = epsilon * (2/3)^g = (1/5) * (2/3)^10
#
# This is exactly the physical CP asymmetry after propagation
# through the girth-length cycle.

epsilon_eff = epsilon * attenuation
print(f"\n  epsilon_eff = epsilon * (2/3)^g = {epsilon} * {attenuation:.5e}")
print(f"             = {epsilon_eff:.5e}")

eta_B_combined = 7.04 * c_sph * epsilon_eff * f_K_formula * n_over_s
print(f"\n  eta_B = 7.04 * c_sph * epsilon_eff * f(K) * n_eq/s")
print(f"       = 7.04 * {c_sph:.4f} * {epsilon_eff:.5e} * {f_K_formula:.5f} * {n_over_s:.5e}")
print(f"       = {eta_B_combined:.5e}")
print(f"\n  Observed: {eta_obs:.2e}")
print(f"  Ratio: {eta_B_combined / eta_obs:.4f}")
print(f"  log10(ratio): {math.log10(eta_B_combined / eta_obs):.2f}")

# =============================================================================
# 10. EXACT MATCHING: What K gives the right answer?
# =============================================================================

print(f"\n{'=' * 78}")
print("10. EXACT MATCH CONDITION")
print("=" * 78)

coeff_eff = 7.04 * c_sph * epsilon_eff * n_over_s
f_needed_eff = eta_obs / coeff_eff
print(f"\n  eta_B = {coeff_eff:.5e} * f(K)")
print(f"  Need f(K) = {f_needed_eff:.5e}")

if f_needed_eff < 1:
    # Might be in intermediate regime: f(K) = 1/(2*sqrt(K^2+9))
    # 1/(2*sqrt(K^2+9)) = f_needed
    # sqrt(K^2+9) = 1/(2*f_needed)
    # K^2 + 9 = 1/(4*f_needed^2)
    # K^2 = 1/(4*f_needed^2) - 9
    K_sq = 1 / (4 * f_needed_eff**2) - 9
    if K_sq > 0:
        K_exact = math.sqrt(K_sq)
        f_verify = 1 / (2 * math.sqrt(K_exact**2 + 9))
        print(f"  Intermediate formula: K = {K_exact:.3f}")
        print(f"  Verify: f({K_exact:.3f}) = {f_verify:.5e}")

        # What alpha gives this K?
        alpha_exact = K_exact * H_prefactor * M_GUT / M_P
        print(f"  Requires alpha_GUT = {alpha_exact:.6f} = 1/{1/alpha_exact:.1f}")
    else:
        print(f"  f_needed > 1/(2*3) = 0.167: need K < 0 (unphysical)")
        print(f"  -> The model OVERSHOOTS even with f(K)=1")

    # Strong washout: f(K) = 0.3/(K * ln(K)^0.6)
    lo, hi = 1.0, 1e6
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        f_mid = 0.3 / (mid * math.log(mid)**0.6)
        if f_mid > f_needed_eff:
            lo = mid
        else:
            hi = mid
    K_exact_strong = math.sqrt(lo * hi)
    f_verify_strong = 0.3 / (K_exact_strong * math.log(K_exact_strong)**0.6)
    alpha_exact_strong = K_exact_strong * H_prefactor * M_GUT / M_P
    print(f"\n  Strong washout formula: K = {K_exact_strong:.1f}")
    print(f"  Verify: f({K_exact_strong:.1f}) = {f_verify_strong:.5e}")
    print(f"  Requires alpha_GUT = {alpha_exact_strong:.5f} = 1/{1/alpha_exact_strong:.1f}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'=' * 78}")
print("SUMMARY")
print("=" * 78)

print(f"""
APPROACH 1 (Freeze-out, Gamma_B = H):
  T_freeze = {T_freeze:.3e} GeV  (~{T_freeze/M_GUT:.0f} * M_GUT)
  eta_B = epsilon / g_* ~ {eta_B_freeze:.3e}
  FAILS: 10^6 too large (no washout suppression)

APPROACH 2 (Gravitational baryogenesis):
  eta_B = {eta_B_grav:.3e}
  FAILS: (T/M_P)^4 suppression too severe

APPROACH 3 (GUT X,Y boson decay at T = M_GUT):
  K = Gamma_X / H = {K_formula:.2f}
  f(K) = {f_K_formula:.5f}
  eta_B = {eta_B_formula:.3e}
  OFF by factor {eta_B_formula/eta_obs:.1f}

APPROACH 3 + GIRTH ATTENUATION (best candidate):
  epsilon_eff = (1/5) * (2/3)^10 = {epsilon_eff:.5e}
  eta_B = {eta_B_combined:.3e}
  OFF by factor {eta_B_combined/eta_obs:.2f}
  log10(deviation) = {math.log10(eta_B_combined/eta_obs):.2f}

KEY FORMULA (all from graph parameters):
  eta_B = 7.04 * (8/23) * (1/5) * (2/3)^10 * f(K) * 45*zeta(3)/(2*pi^4 * 228.75)
  where K = alpha_GUT * M_P / (1.66 * sqrt(g_*) * M_GUT)

PARAMETER VALUES:
  epsilon     = 1/5                     (Laves chirality)
  (2/3)^10    = {attenuation:.5e}         (girth attenuation)
  alpha_GUT   = 1/{1/alpha_GUT:.1f}               (reconnection DL)
  M_GUT       = {M_GUT:.1e} GeV          (MSSM unification)
  M_P         = {M_P:.4e} GeV          (Planck mass)
  g_*         = {g_star}               (MSSM DOF)
  c_sph       = 8/23                    (MSSM sphaleron)
  K           = {K_formula:.2f}
  f(K)        = {f_K_formula:.5f}

STATUS: The combination epsilon*(2/3)^g gives the right ORDER OF MAGNITUDE
  but is off by a factor of {eta_B_combined/eta_obs:.1f}. This is {abs(math.log10(eta_B_combined/eta_obs)):.1f} orders
  of magnitude, which is {'encouraging' if abs(math.log10(eta_B_combined/eta_obs)) < 2 else 'problematic'} given that the
  raw epsilon = 1/5 is 8 orders too large.

  The remaining factor of ~{eta_B_combined/eta_obs:.0f} could come from:
  (a) More precise washout computation (our f(K) interpolation is crude)
  (b) Additional entropy dilution from moduli or gravitino decay
  (c) The factor of N_species (we used 1, could be different)
  (d) Running of alpha_GUT from M_GUT to the actual freeze-out scale
""")
