#!/usr/bin/env python3
"""
SRS Quark Koide Verification: Are the 5 quark masses theorem-grade?

This script checks every link in the derivation chain for the quark sector
Koide relations, grading each honestly as THEOREM, CONJECTURE, or FITTED.

The charged lepton Koide is theorem-grade:
  eps = sqrt(2) (water-filling on Z3 irreps, theorem)
  Q = 2/3 (toggle + MDL, theorem)
  delta = 2/9 (Wigner D1 + HM uniqueness, theorem)
  => m_e, m_mu, m_tau from M0 + these three parameters

For quarks, the claim is:
  eps^2(n) = 2 + 6*alpha_1*n*f(n) with alpha_1 = (5/3)*(2/3)^8
  delta(n) = 2/(9*(n+1))
  M0(sector) anchored to heaviest mass in each sector

This script evaluates the derivation status of each parameter and mass.
"""

import numpy as np
from numpy import linalg as la
from scipy.optimize import brentq, minimize

PI = np.pi
sqrt = np.sqrt
cos = np.cos

# =====================================================================
# PDG MASSES (MeV, MS-bar at 2 GeV for light quarks, m_q for heavy)
# =====================================================================
m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86  # MeV, pole
m_d, m_s, m_b = 4.67, 93.4, 4180.0     # MeV, MS-bar
m_u, m_c, m_t = 2.16, 1270.0, 172760.0  # MeV, MS-bar (m_t = pole)

# MS-bar m_t for waterfall comparison
m_t_msbar = 162500.0  # MeV (PDG: 162.5 +/- 1.1 GeV)

# Graph constants
k_star = 3
girth = 10
alpha_1 = (5.0 / 3.0) * (2.0 / 3.0) ** 8  # = 1280/19683

omega_z3 = np.exp(2j * PI / 3)

# =====================================================================
# KOIDE MACHINERY
# =====================================================================

def koide_params(masses):
    """Extract Koide epsilon and delta from three masses."""
    sq = sqrt(np.array(masses, dtype=float))
    c0 = np.mean(sq)
    c1 = np.mean(sq * np.array([1, omega_z3**(-1), omega_z3**(-2)]))
    eps = 2 * abs(c1) / c0
    delta = -np.angle(c1)
    Q = np.sum(masses) / np.sum(sq)**2
    return eps, delta, Q

def koide_masses(M0, eps, delta):
    """Compute 3 masses from Koide parametrization.
    sqrt(m_k) = M0 * (1 + eps*cos(2*pi*k/3 + delta)), k=0,1,2."""
    ks = np.arange(3)
    sq = M0 * (1 + eps * cos(2*PI*ks/3 + delta))
    return sq**2

def koide_Q(m1, m2, m3):
    """Koide Q parameter."""
    s = sqrt(m1) + sqrt(m2) + sqrt(m3)
    return (m1 + m2 + m3) / s**2

def eps_sq_pred(n):
    """Predicted eps^2 for Fock sector n."""
    if n == 0:
        return 2.0
    f_n = 1 + (n - 1) * (girth - 2) / (2 * girth)
    return 2 + 6 * alpha_1 * n * f_n

def delta_pred(n):
    """Predicted Koide phase for Fock sector n."""
    return 2.0 / (9 * (n + 1))


# =====================================================================
print("=" * 78)
print("  SRS QUARK KOIDE VERIFICATION")
print("  Goal: honest grade for each of the 5 quark masses")
print("=" * 78)


# =====================================================================
# SECTION 1: LEPTON KOIDE BASELINE (THEOREM-GRADE)
# =====================================================================
print("\n" + "=" * 78)
print("SECTION 1: LEPTON KOIDE BASELINE")
print("=" * 78)

eps_l, delta_l, Q_l = koide_params([m_e, m_mu, m_tau])

# The DFT extraction convention puts k=0 at lightest mass.
# The Koide formula with delta=2/9 has k=0 as heaviest.
# The extracted delta relates to the "true" delta by: delta_true = 2*pi/3 - delta_extracted
# (generation relabeling). But this is just a convention issue.
# Let's verify the actual mass predictions.

# Using theorem parameters: eps=sqrt(2), delta=2/9, anchor at m_tau
eps_l_th = sqrt(2.0)
delta_l_th = 2.0 / 9.0

# Find M0 from m_tau (k=0 is heaviest with delta=2/9)
M0_l = sqrt(m_tau) / (1 + eps_l_th * cos(delta_l_th))
masses_l_pred = koide_masses(M0_l, eps_l_th, delta_l_th)
masses_l_pred = np.sort(masses_l_pred)

print(f"\n  Lepton Koide (ALL THEOREM):")
print(f"    eps = sqrt(2) = {eps_l_th:.6f}  (obs: {eps_l:.6f})")
print(f"    Q = 2/3 = {2/3:.6f}  (obs: {Q_l:.6f})")
print(f"    delta = 2/9 = {2/9:.6f}")
print(f"\n  {'Particle':<10} {'Predicted':>12} {'Observed':>12} {'Error':>8}")
print("  " + "-" * 44)
for name, pred, obs in zip(["electron", "muon", "tau"],
                           masses_l_pred, [m_e, m_mu, m_tau]):
    err = (pred - obs) / obs * 100
    print(f"  {name:<10} {pred:>12.4f} {obs:>12.4f} {err:>7.2f}%")


# =====================================================================
# SECTION 2: QUARK KOIDE PARAMETERS - DERIVATION STATUS
# =====================================================================
print("\n" + "=" * 78)
print("SECTION 2: QUARK KOIDE PARAMETERS - DERIVATION STATUS")
print("=" * 78)

# --- 2a: eps^2(n) for quarks ---
print("\n--- 2a: eps^2(n) = 2 + 6*alpha_1*n*f(n) ---")
print(f"  alpha_1 = (5/3)*(2/3)^8 = 1280/19683 = {alpha_1:.6f}")

# Observed eps^2 values
eps_d_obs, delta_d_obs, Q_d_obs = koide_params([m_d, m_s, m_b])
eps_u_obs, delta_u_obs, Q_u_obs = koide_params([m_u, m_c, m_t])

for label, n, eps_obs in [("Down quarks (n=1)", 1, eps_d_obs),
                           ("Up quarks (n=2)", 2, eps_u_obs)]:
    eps2_p = eps_sq_pred(n)
    eps2_o = eps_obs**2
    dev_p = eps2_p - 2
    dev_o = eps2_o - 2
    print(f"\n  {label}:")
    print(f"    eps^2 predicted = {eps2_p:.6f}")
    print(f"    eps^2 observed  = {eps2_o:.6f}")
    print(f"    eps^2-2: pred = {dev_p:.6f}, obs = {dev_o:.6f}, err = {(dev_p-dev_o)/dev_o*100:+.2f}%")

# The deviation RATIO is the strongest test (zero free parameters)
dev_d = eps_d_obs**2 - 2
dev_u = eps_u_obs**2 - 2
ratio_obs = dev_u / dev_d
ratio_pred = (eps_sq_pred(2) - 2) / (eps_sq_pred(1) - 2)
print(f"\n  Deviation ratio (eps^2-2)_up / (eps^2-2)_down:")
print(f"    Predicted: 14/5 = {14/5} = {ratio_pred:.6f}")
print(f"    Observed:  {ratio_obs:.6f}")
print(f"    Match: {abs(ratio_pred - ratio_obs) / ratio_obs * 100:.2f}%")

print(f"""
  GRADE for eps^2(n):
    - alpha_1 = (n_g/k*)*(k*-1/k*)^(g-2): CONJECTURE (B+)
      The 10-cycle counting is well-motivated but not proven from pure
      graph theory. It relies on the rate-distortion / equal-information
      interpretation of mass.
    - f(n) = 1 + (n-1)*(g-2)/(2*g): CONJECTURE (B)
      Girth-dependent correction, physically motivated but not derived
      from first principles.
    - The RATIO 14/5 depends only on girth g=10: TESTABLE (0.55% match)
      This is the best zero-parameter prediction.""")

# --- 2b: delta(n) for quarks ---
print("\n--- 2b: delta(n) = 2/(9*(n+1)) ---")

# Compare observed vs predicted delta
# The DFT extraction convention with k=0=lightest gives
# delta_extracted such that delta_true = 2*pi/3 - delta_extracted
# when delta_extracted is in range [pi, 2*pi], or we can use
# the absolute value of the fitted delta directly.
# Best approach: fit delta directly from the masses using the Koide formula.

def extract_true_delta(masses):
    """Fit the Koide parametrization to extract the true delta."""
    sq = np.sqrt(np.sort(masses)[::-1])  # heaviest first (k=0)
    def residual(params):
        M0, eps, d = params
        pred = np.array([M0*(1 + eps*cos(2*PI*k/3 + d)) for k in range(3)])
        return np.sum((pred - sq)**2)
    res = minimize(residual, x0=[np.mean(sq), sqrt(2), 0.15],
                   method='Nelder-Mead', options={'xatol':1e-14, 'fatol':1e-20})
    return abs(res.x[2])

delta_l_true = extract_true_delta([m_e, m_mu, m_tau])
delta_d_true = extract_true_delta([m_d, m_s, m_b])
delta_u_true = extract_true_delta([m_u, m_c, m_t])

for label, n, d_true in [("Leptons (n=0)", 0, delta_l_true),
                          ("Down quarks (n=1)", 1, delta_d_true),
                          ("Up quarks (n=2)", 2, delta_u_true)]:
    d_pred = delta_pred(n)
    err_pct = abs(d_true - d_pred) / d_pred * 100
    print(f"  {label}:")
    print(f"    delta predicted = {d_pred:.6f}")
    print(f"    delta observed  = {d_true:.6f}")
    print(f"    error = {err_pct:.2f}%")

print(f"""
  GRADE for delta(n):
    - delta(0) = 2/9: THEOREM (A)
      Derived from Wigner D^1 survival probabilities P = (4/9, 1/9, 4/9),
      harmonic mean uniquely selected by three independent arguments.
    - delta(n) = delta(0)/(n+1): CONJECTURE (C+)
      The "capacity sharing" argument -- screw phase budget distributed
      among (n+1) channels -- is an INFORMATION-THEORETIC HEURISTIC,
      not a mathematical derivation. The numerical agreement (0.2% for
      down, 0.6% for up) is encouraging but does not constitute proof.
    - The 1/(n+1) factor has NO group-theoretic derivation.
      The cos(beta_n) values that would produce delta(n) through the
      Wigner HM formula do NOT have simple algebraic forms.""")


# =====================================================================
# SECTION 3: QUARK MASS PREDICTIONS
# =====================================================================
print("\n" + "=" * 78)
print("SECTION 3: QUARK MASS PREDICTIONS")
print("=" * 78)

# --- 3a: Down quarks using predicted parameters, anchored to m_b ---
print("\n--- 3a: Down quarks (d, s, b) ---")
print(f"  Anchor: m_b = {m_b:.1f} MeV (OBSERVED INPUT)")

eps_d_p = sqrt(eps_sq_pred(1))
delta_d_p = delta_pred(1)

# Find M0 from m_b anchor
factors_d = np.array([1 + eps_d_p * cos(2*PI*k/3 + delta_d_p) for k in range(3)])
k_heavy_d = np.argmax(factors_d)
M0_d = sqrt(m_b) / factors_d[k_heavy_d]
masses_d_pred = np.sort(np.array([(M0_d * f)**2 for f in factors_d]))

print(f"  eps = {eps_d_p:.6f} (from eps^2 = {eps_sq_pred(1):.6f})")
print(f"  delta = 1/9 = {delta_d_p:.6f}")
print(f"  M0 = {M0_d:.4f} sqrt(MeV)")
print(f"\n  {'Particle':<10} {'Predicted':>12} {'Observed':>12} {'Error':>8}")
print("  " + "-" * 44)
for name, pred, obs in zip(["down", "strange", "bottom"],
                           masses_d_pred, [m_d, m_s, m_b]):
    err = (pred - obs) / obs * 100
    print(f"  {name:<10} {pred:>12.4f} {obs:>12.4f} {err:>7.2f}%")

Q_d_pred = koide_Q(masses_d_pred[0], masses_d_pred[1], masses_d_pred[2])
print(f"\n  Q(d,s,b) predicted = {Q_d_pred:.6f}")
print(f"  Q(d,s,b) observed  = {Q_d_obs:.6f}")

# --- 3b: Up quarks using predicted parameters, anchored to m_t ---
print("\n--- 3b: Up quarks (u, c, t) ---")
print(f"  Anchor: m_t = {m_t:.1f} MeV (OBSERVED INPUT)")

eps_u_p = sqrt(eps_sq_pred(2))
delta_u_p = delta_pred(2)

factors_u = np.array([1 + eps_u_p * cos(2*PI*k/3 + delta_u_p) for k in range(3)])
k_heavy_u = np.argmax(factors_u)
M0_u = sqrt(m_t) / factors_u[k_heavy_u]
masses_u_pred = np.sort(np.array([(M0_u * f)**2 for f in factors_u]))

print(f"  eps = {eps_u_p:.6f} (from eps^2 = {eps_sq_pred(2):.6f})")
print(f"  delta = 2/27 = {delta_u_p:.6f}")
print(f"  M0 = {M0_u:.4f} sqrt(MeV)")
print(f"\n  {'Particle':<10} {'Predicted':>12} {'Observed':>12} {'Error':>8}")
print("  " + "-" * 44)
for name, pred, obs in zip(["up", "charm", "top"],
                           masses_u_pred, [m_u, m_c, m_t]):
    err = (pred - obs) / obs * 100
    print(f"  {name:<10} {pred:>12.4f} {obs:>12.4f} {err:>7.2f}%")

Q_u_pred = koide_Q(masses_u_pred[0], masses_u_pred[1], masses_u_pred[2])
print(f"\n  Q(u,c,t) predicted = {Q_u_pred:.6f}")
print(f"  Q(u,c,t) observed  = {Q_u_obs:.6f}")


# =====================================================================
# SECTION 4: GJ = 3 AND INTER-SECTOR SCALE
# =====================================================================
print("\n" + "=" * 78)
print("SECTION 4: GEORGI-JARLSKOG FACTOR GJ = 3")
print("=" * 78)

# GJ = 3 is a theorem on Q3 hypercube: |Sigma(0)|/|Sigma(1)| = 3
# This gives m_b/m_tau = 3 at GUT scale.
# At low energies, QCD running modifies this.

# Check at low energies (MS-bar):
ratio_b_tau = m_b / m_tau
print(f"\n  m_b/m_tau (low energy) = {ratio_b_tau:.4f}")
print(f"  GJ prediction at GUT scale: 3.000")
print(f"  At low energy with QCD running: ~2.35 expected")
print(f"  (QCD enhances quark masses relative to leptons by ~28% at m_b scale)")

# Does GJ = 3 determine M0(down)/M0(lepton)?
# At GUT scale: m_b = 3 * m_tau
# In Koide: sqrt(m_b) = M0_d * f_max(eps_d, delta_d)
#           sqrt(m_tau) = M0_l * f_max(eps_l, delta_l)
# So: M0_d/M0_l = sqrt(m_b/m_tau) * f_max(l) / f_max(d)
# This requires knowing eps and delta for BOTH sectors.

f_max_l = 1 + sqrt(2.0) * cos(2.0/9)
f_max_d = max(factors_d)
scale_ratio = sqrt(ratio_b_tau) * f_max_l / f_max_d
print(f"\n  M0(down)/M0(lepton) = sqrt(m_b/m_tau) * f_max(l)/f_max(d)")
print(f"    = sqrt({ratio_b_tau:.4f}) * {f_max_l:.4f} / {f_max_d:.4f}")
print(f"    = {scale_ratio:.4f}")

# The GJ factor connects DOWN quarks to LEPTONS.
# For UP quarks, we need m_t or another anchor.
print(f"""
  GRADE for GJ = 3:
    - |Sigma(0)|/|Sigma(1)| = 3 on Q_3: THEOREM (A)
      Exact result from DL Laplacian on binary hypercube.
    - m_b/m_tau = 3 at GUT scale: THEOREM (A)
      Direct consequence of GJ = 3.
    - RG running from GUT to m_b scale: STANDARD QCD (A)
      Well-understood perturbative running.
    - BUT: GJ = 3 determines m_b GIVEN m_tau, NOT the absolute quark scale.
      It requires the GUT scale and SUSY threshold corrections to be
      precisely specified to extract m_b(MS-bar) from m_tau(pole).""")


# =====================================================================
# SECTION 5: RIVERO WATERFALL Q(c,b,t) = 2/3
# =====================================================================
print("\n" + "=" * 78)
print("SECTION 5: RIVERO WATERFALL Q(c,b,t) = 2/3")
print("=" * 78)

# This uses Q = 2/3 on the HEAVY quark triple (c, b, t) with MS-bar masses
Q_cbt_obs = koide_Q(m_c/1000, m_b/1000, m_t_msbar/1000)  # GeV
print(f"\n  Q(c, b, t) with MS-bar masses:")
print(f"    m_c = {m_c/1000:.3f} GeV, m_b = {m_b/1000:.3f} GeV, m_t(MS-bar) = {m_t_msbar/1000:.1f} GeV")
print(f"    Q = {Q_cbt_obs:.6f}")
print(f"    Target: 2/3 = {2/3:.6f}")
print(f"    Deviation: {abs(Q_cbt_obs - 2/3)/Q_cbt_obs*100:.3f}%")

# Solve for m_t from Q(c,b,t) = 2/3
def eq_mt(mt):
    return koide_Q(m_c/1000, m_b/1000, mt) - 2.0/3.0
m_t_waterfall = brentq(eq_mt, 50.0, 500.0)  # GeV
print(f"\n  Rivero waterfall: m_t(MS-bar) = {m_t_waterfall:.2f} GeV")
print(f"  Observed: m_t(MS-bar) = {m_t_msbar/1000:.1f} GeV")
print(f"  Match: {(m_t_waterfall - m_t_msbar/1000)/(m_t_msbar/1000)*100:+.2f}%")

print(f"""
  GRADE for Rivero waterfall:
    - Q(c,b,t) = 2/3: OBSERVATION, not derivation.
      WHY does Q = 2/3 apply to (c,b,t) instead of (u,c,t) or (d,s,b)?
      The framework says all three sectors should satisfy Koide with
      sector-dependent eps and delta, but Q = 2/3 is only approximately
      satisfied for the heavy triple. This is an EMPIRICAL OBSERVATION
      that is CONSISTENT with the framework but not DERIVED from it.
    - MS-bar vs pole: MDL argument for MS-bar is THEOREM-grade (A).
    - The waterfall gives m_t given (m_c, m_b): CONDITIONAL (B+).
      It's a theorem IF you accept Q(c,b,t) = 2/3, but the reason
      WHY Q=2/3 holds for (c,b,t) rather than other triples is unclear.""")


# =====================================================================
# SECTION 6: WHAT ABOUT USING OBSERVED delta vs PREDICTED delta?
# =====================================================================
print("\n" + "=" * 78)
print("SECTION 6: SENSITIVITY TO delta CHOICE")
print("=" * 78)

# Anchored to heaviest mass in each sector, compare predictions
# using predicted delta(n) vs best-fit delta from data

# Fit delta from observed masses
def fit_delta(masses_obs, eps, anchor_mass):
    """Find delta that minimizes mass prediction error."""
    def obj(d):
        factors = np.array([1 + eps*cos(2*PI*k/3 + d[0]) for k in range(3)])
        k_h = np.argmax(factors)
        M0 = sqrt(anchor_mass) / factors[k_h]
        m_pred = np.sort(np.array([(M0*f)**2 for f in factors]))
        return np.sum(((m_pred - np.sort(masses_obs))/np.sort(masses_obs))**2)
    res = minimize(obj, x0=[0.15], method='Nelder-Mead',
                   options={'xatol': 1e-14, 'fatol': 1e-20})
    return res.x[0]

delta_d_fit = fit_delta([m_d, m_s, m_b], eps_d_p, m_b)
delta_u_fit = fit_delta([m_u, m_c, m_t], eps_u_p, m_t)

print(f"\n  Down quarks:")
print(f"    delta predicted (1/9)  = {1/9:.6f}")
print(f"    delta best-fit         = {delta_d_fit:.6f}")
print(f"    Difference             = {abs(delta_d_fit - 1/9):.6f} ({abs(delta_d_fit - 1/9)/(1/9)*100:.2f}%)")

print(f"\n  Up quarks:")
print(f"    delta predicted (2/27) = {2/27:.6f}")
print(f"    delta best-fit         = {delta_u_fit:.6f}")
print(f"    Difference             = {abs(delta_u_fit - 2/27):.6f} ({abs(delta_u_fit - 2/27)/(2/27)*100:.2f}%)")

# Now compare mass predictions with fitted vs predicted delta
print(f"\n  Mass predictions: predicted delta vs best-fit delta")
print(f"\n  DOWN QUARKS:")
print(f"  {'Particle':<10} {'delta=1/9':>12} {'delta=fit':>12} {'Observed':>12}")
print("  " + "-" * 48)

factors_d_fit = np.array([1 + eps_d_p*cos(2*PI*k/3 + delta_d_fit) for k in range(3)])
k_h_df = np.argmax(factors_d_fit)
M0_d_fit = sqrt(m_b) / factors_d_fit[k_h_df]
masses_d_fit = np.sort(np.array([(M0_d_fit*f)**2 for f in factors_d_fit]))

for name, pred_th, pred_fit, obs in zip(["down", "strange", "bottom"],
                                         masses_d_pred, masses_d_fit, [m_d, m_s, m_b]):
    print(f"  {name:<10} {pred_th:>12.4f} {pred_fit:>12.4f} {obs:>12.4f}")

print(f"\n  UP QUARKS:")
print(f"  {'Particle':<10} {'delta=2/27':>12} {'delta=fit':>12} {'Observed':>12}")
print("  " + "-" * 48)

factors_u_fit = np.array([1 + eps_u_p*cos(2*PI*k/3 + delta_u_fit) for k in range(3)])
k_h_uf = np.argmax(factors_u_fit)
M0_u_fit = sqrt(m_t) / factors_u_fit[k_h_uf]
masses_u_fit = np.sort(np.array([(M0_u_fit*f)**2 for f in factors_u_fit]))

for name, pred_th, pred_fit, obs in zip(["up", "charm", "top"],
                                         masses_u_pred, masses_u_fit, [m_u, m_c, m_t]):
    print(f"  {name:<10} {pred_th:>12.4f} {pred_fit:>12.4f} {obs:>12.4f}")


# =====================================================================
# SECTION 7: DOES GJ = 3 FULLY DETERMINE QUARK KOIDE?
# =====================================================================
print("\n" + "=" * 78)
print("SECTION 7: DOES GJ = 3 FULLY DETERMINE QUARK KOIDE?")
print("=" * 78)

print("""
  Given the lepton Koide (theorem-grade), does GJ = 3 determine everything?

  What GJ = 3 gives:
    m_b(GUT) = 3 * m_tau(GUT)                         ... THEOREM
    m_b(m_b) via QCD running from GUT scale            ... requires M_GUT, alpha_s(GUT)

  What GJ = 3 does NOT give:
    1. The Koide parameters (eps, delta) for quarks     ... requires alpha_1, delta(n)
    2. The quark Koide SCALE M0_d and M0_u             ... requires m_b anchor
    3. The up-sector anchor (m_t or M0_u)              ... requires waterfall or separate input

  The derivation chain for each quark mass:

  m_b: LEPTON ANCHOR (m_tau) + GJ=3 + QCD running
       Status: THEOREM (GJ) x STANDARD (QCD) x FITTED (M_GUT, alpha_s(GUT) details)
       Grade: B+ (the GJ factor is exact, but RG running to m_b introduces
       sensitivity to SUSY threshold corrections and M_GUT)

  m_s: KOIDE(d,s,b) with m_b anchor, eps^2(1), delta(1)
       Status: depends on eps^2(1) [B+] and delta(1) [C+]
       Grade: C+ (delta(1) is the weakest link)

  m_d: KOIDE(d,s,b) with m_b anchor, eps^2(1), delta(1)
       Status: same chain as m_s, but m_d is the LIGHTEST mass
       so it's most sensitive to parameter errors
       Grade: C (lightest mass = worst numerical stability)

  m_t: RIVERO WATERFALL Q(c,b,t) = 2/3 with m_c, m_b MS-bar
       Status: empirical observation, MDL justification for MS-bar is theorem
       Grade: B+ (waterfall works to ~0.5%, but WHY is unclear)
       Alternative: KOIDE(u,c,t) with eps^2(2) and delta(2)
       Grade: C+ (same delta(n) weakness)

  m_c: KOIDE(u,c,t) with m_t anchor, eps^2(2), delta(2)
       Status: same as m_s
       Grade: C+
""")


# =====================================================================
# SECTION 8: THE FREE PARAMETER COUNT
# =====================================================================
print("=" * 78)
print("SECTION 8: FREE PARAMETER AUDIT")
print("=" * 78)

print("""
  Parameters used to predict 5 quark masses:

  THEOREM-GRADE (0 free parameters each):
    eps(n=0) = sqrt(2)           from water-filling on Z3 irreps
    Q(n=0) = 2/3                 from toggle + MDL
    delta(n=0) = 2/9             from Wigner D + HM uniqueness
    GJ = 3                       from Q3 DL Laplacian

  CONJECTURED (grade B to C):
    alpha_1 = 1280/19683         from srs girth-cycle counting       [B+]
    f(n) = 1 + (n-1)*(g-2)/(2g) from girth correction               [B]
    delta(n) = 2/(9*(n+1))       from capacity sharing argument      [C+]

  OBSERVED INPUTS (not derived):
    m_b = 4180 MeV               anchor for down sector
    m_t = 172760 MeV             anchor for up sector
    (or equivalently: m_tau + GJ for m_b, waterfall for m_t)

  BOTTOM LINE: The framework has 3 conjectured parameters (alpha_1, f(n),
  delta(n)) and 2 observed anchors. For 5 predicted quark masses, that's
  5 predictions from 2 inputs + 3 conjectures = 5 free numbers for 5 masses.
  Not overconstrained. The fact that it WORKS is non-trivial (wrong alpha_1
  or wrong delta(n) would give bad ratios), but it's not theorem-grade.
""")


# =====================================================================
# SECTION 9: HONEST FINAL GRADES
# =====================================================================
print("=" * 78)
print("SECTION 9: HONEST FINAL GRADES")
print("=" * 78)

grades = [
    ("m_b (bottom)", "B+",
     "GJ=3 (theorem) + QCD running (standard) + M_GUT input",
     "GJ exactly connects to m_tau; QCD running is reliable but needs M_GUT"),
    ("m_s (strange)", "C+",
     "Koide(d,s,b) with m_b anchor + eps^2(1) [B+] + delta(1)=1/9 [C+]",
     "delta(1) capacity-sharing argument is heuristic, not mathematical"),
    ("m_d (down)", "C",
     "Koide(d,s,b) with m_b anchor + eps^2(1) [B+] + delta(1)=1/9 [C+]",
     "Lightest mass, most sensitive to delta error; C+ chain minus stability"),
    ("m_t (top)", "B+",
     "Rivero Q(c,b,t)=2/3 in MS-bar + QCD pole conversion",
     "Works to 0.5% but WHY Q=2/3 for (c,b,t) is empirical, not derived"),
    ("m_c (charm)", "C+",
     "Koide(u,c,t) with m_t anchor + eps^2(2) [B+] + delta(2)=2/27 [C+]",
     "Same delta(n) weakness as m_s"),
]

print(f"\n  {'Mass':<15} {'Grade':>5}  Derivation chain")
print("  " + "-" * 72)
for name, grade, chain, note in grades:
    print(f"  {name:<15} {grade:>5}  {chain}")
    print(f"  {'':15} {'':5}  NOTE: {note}")
    print()

# =====================================================================
# SECTION 10: WHAT WOULD MAKE THEM THEOREM-GRADE?
# =====================================================================
print("=" * 78)
print("SECTION 10: WHAT WOULD UPGRADE TO THEOREM?")
print("=" * 78)

print("""
  To upgrade from C+/B+ to THEOREM (A), each conjectured parameter needs
  a mathematical derivation from graph topology + MDL alone:

  1. alpha_1 = (n_g/k*) * ((k*-1)/k*)^(g-2):
     NEEDED: Prove that the rate-distortion argument for chirality coupling
     uniquely produces this formula. Currently it's a COUNTING argument
     (girth-10 cycles on the srs net) that has not been proven optimal
     under MDL. The 14/5 ratio prediction (0.55% match) is strong evidence
     but not a proof.

  2. delta(n) = 2/(9*(n+1)):
     NEEDED: Either (a) a group-theoretic derivation from the srs net
     automorphism group that shows how n occupied edges dilute the screw
     phase by exactly 1/(n+1), or (b) an MDL optimality proof showing
     that 2/(9*(n+1)) minimizes description length for the mass spectrum
     at Fock number n.
     The current "capacity sharing" argument is an analogy, not a proof.
     This is the SINGLE BIGGEST GAP in the quark sector.

  3. m_b anchor: GJ=3 gives m_b(GUT)/m_tau(GUT) = 3 (theorem).
     NEEDED: Precise RG running from GUT to m_b, including SUSY threshold
     corrections. This is standard physics but introduces sensitivity to
     M_GUT and M_SUSY.

  4. m_t anchor: The Rivero waterfall Q(c,b,t) = 2/3 needs a derivation
     of WHY this particular triple satisfies Q = 2/3.
     ALTERNATIVE: if eps^2(2), delta(2), AND M0_u could all be derived
     from the graph, m_t would follow without the waterfall.

  MINIMUM PATH TO THEOREM:
    Derive delta(n) = 2/(9*(n+1)) from graph theory. Everything else
    is already B+ or better. The delta derivation is the bottleneck.
""")


# =====================================================================
# SECTION 11: NUMERICAL SUMMARY
# =====================================================================
print("=" * 78)
print("SECTION 11: NUMERICAL SUMMARY TABLE")
print("=" * 78)

print(f"\n  {'Mass':<12} {'Predicted':>12} {'PDG':>12} {'Error':>8} {'Grade':>6}")
print("  " + "-" * 54)

all_results = [
    ("m_d (MeV)", masses_d_pred[0], m_d, "C"),
    ("m_s (MeV)", masses_d_pred[1], m_s, "C+"),
    ("m_b (MeV)", m_b, m_b, "B+"),  # anchor
    ("m_u (MeV)", masses_u_pred[0], m_u, "C"),
    ("m_c (MeV)", masses_u_pred[1], m_c, "C+"),
    ("m_t (MeV)", m_t, m_t, "B+"),  # anchor
]

for name, pred, obs, grade in all_results:
    err = (pred - obs) / obs * 100
    anchor = " (anchor)" if abs(err) < 0.01 else ""
    print(f"  {name:<12} {pred:>12.2f} {obs:>12.2f} {err:>+7.2f}% {grade:>6}{anchor}")

print(f"""
  VERDICT: None of the 5 quark masses are theorem-grade.

  The bottleneck is delta(n) = 2/(9*(n+1)). The 1/(n+1) dilution factor
  is supported by:
    - Numerical agreement (0.2% for n=1, 0.6% for n=2)
    - A plausible information-theoretic argument
  but NOT by:
    - A mathematical proof from graph theory
    - An MDL optimality derivation
    - A group-theoretic computation

  Best-graded: m_b (B+) and m_t (B+) through GJ=3 and Rivero waterfall.
  Worst-graded: m_d and m_u (C) due to delta(n) + numerical sensitivity.

  The SINGLE upgrade that would change everything: prove delta(n) = 2/(9*(n+1))
  from the srs net automorphism group or from MDL on the Fock sectors.
  That would promote all 5 quark masses from C/C+ to B+/A-.
""")
