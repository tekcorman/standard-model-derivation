#!/usr/bin/env python3
"""
proofs/cosmology/Lambda_CC_path_A_session2_cmb_theta_star.py

Λ_CC PATH A — SESSION 2: CMB acoustic θ* test under coasting cosmology.

Setup
-----
Session 1 (`Lambda_CC_path_A_session1_coasting_lcdm_fit.py`) showed that
the simple factor-of-2 hypothesis ("ΛCDM-fitting of coasting SN1a data
recovers Ω_m ≈ 1/3") FAILS on the SN1a-only channel: ΛCDM-fit of coasting
mock recovers Ω_m ≈ 0.53, not 1/3. The wCDM and q_0 sub-tests confirmed
coasting IS detectable in Pantheon+ (correctly recovering w → -1/3,
q_0 → 0), so the issue is specifically that ΛCDM-w-enforced misfit lands
at the wrong Ω_m magnitude.

Reframe: if the empirical Planck Ω_m ≈ 0.315 ≈ 1/3 is to be explained
by ΛCDM-mis-extraction of coasting data, the constraint must come from
CMB+BAO (which dominate the joint Planck Ω_m σ), not SN1a.

Session 2 attacks the strongest single CMB constraint: the acoustic
peak angular scale θ_*. Planck measures this to 0.03% precision:

  100·θ_* = 1.04109 ± 0.00030     (Planck 2018)

θ_* = r_s(z*) / D_M(z*) where:
  r_s(z*) = comoving sound horizon at last scattering (z* ≈ 1090)
  D_M(z*) = comoving angular diameter distance to last scattering

If coasting predicts the same θ_* (or a small adjustment that closes
the factor-of-2 picture), Path A survives. If coasting predicts θ_*
wildly different from Planck, the factor-of-2 hypothesis is dead at
the CMB level.

Three quantitative tests this session
-------------------------------------
1. D_M(z*) under coasting vs ΛCDM. Compute the ratio. (Coasting
   D_M is analytical: D_M_coast = (c/H_0) ln(1+z*).)

2. r_s(z*) under coasting vs ΛCDM. Coasting H(z) = H_0(1+z) at all
   epochs (cascade theorem N = t/t_P). At high z, this is much smaller
   than ΛCDM H(z) ∝ √(Ω_r(1+z)⁴), leading to a divergent (or huge)
   r_s_coast that requires a UV cutoff.

3. θ_* under coasting (with Planck-epoch UV cutoff) vs Planck. If the
   ratio is O(1), factor-of-2 closes; if it's orders of magnitude off,
   Path A is falsified at CMB.

Selection grammar discipline: the comparison is channel_select between
two physically distinct cosmologies (coasting vs ΛCDM); both above the
A2-T waterline as candidate models; observation (Planck θ_*) is the
selector.
"""

import sys
import os
import math

import numpy as np
from scipy import integrate

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =============================================================================
# §0. Constants and framework predictions
# =============================================================================
c_km_s = 2.99792458e5

# Framework
H0_SUBSTRATE = 68.19            # km/s/Mpc, cascade H = 1/(N t_P)
H0_OBSERVER = (16.0 / 15.0) * H0_SUBSTRATE  # = 72.74 (D2-extended)

# Planck 2018 (TT,TE,EE+lowE+lensing baseline)
PLANCK_H0 = 67.36
PLANCK_OMEGA_M = 0.3153
PLANCK_OMEGA_LAMBDA = 1.0 - PLANCK_OMEGA_M
PLANCK_OMEGA_BH2 = 0.02237
PLANCK_OMEGA_CH2 = 0.1200
PLANCK_OMEGA_R_H2 = 4.18e-5      # photons + 3 massless ν approximation; Ω_r h² ≈ 4.18e-5
PLANCK_T_CMB = 2.7255             # K
PLANCK_THETA_STAR_TIMES_100 = 1.04109   # Planck 2018 best-fit
PLANCK_THETA_STAR_TIMES_100_SIGMA = 0.00030
PLANCK_R_S_MPC = 147.05           # Planck 2018 sound horizon in Mpc
PLANCK_D_M_STAR_MPC = 13869.61    # Planck 2018 comoving distance to z*
PLANCK_Z_STAR = 1089.92            # Planck 2018 redshift of last scattering

# Note: in coasting, the framework hasn't published an independent T_CMB
# or Ω_b derivation that would change these; we use Planck values for
# Ω_b h², Ω_γ h² and check whether coasting can reproduce θ* given them.
# (If coasting fails even with input scrupulously matched to Planck, the
# failure is inherent to the H(z) profile, not to early-universe input.)

print("=" * 78)
print("Λ_CC PATH A — SESSION 2: CMB acoustic θ* test under coasting")
print("=" * 78)
print(f"  Framework H_0 (substrate, cascade): {H0_SUBSTRATE:.4f} km/s/Mpc")
print(f"  Framework H_0 (observer, ×16/15):   {H0_OBSERVER:.4f} km/s/Mpc")
print()
print(f"  Planck 2018 baseline:")
print(f"    H_0 = {PLANCK_H0:.4f} km/s/Mpc")
print(f"    Ω_m = {PLANCK_OMEGA_M:.4f}")
print(f"    z*  = {PLANCK_Z_STAR}")
print(f"    100·θ* = {PLANCK_THETA_STAR_TIMES_100:.5f} ± {PLANCK_THETA_STAR_TIMES_100_SIGMA:.5f}")
print(f"    r_s(z*) = {PLANCK_R_S_MPC:.2f} Mpc  (comoving sound horizon)")
print(f"    D_M(z*) = {PLANCK_D_M_STAR_MPC:.2f} Mpc  (comoving angular distance)")
print()


# =============================================================================
# §1. D_M(z*) under coasting vs ΛCDM
# =============================================================================
print("§1. Comoving angular diameter distance D_M(z*)")
print("-" * 78)

# Coasting (a ∝ t, H = H_0(1+z), all epochs)
# D_M_coast(z) = (c/H_0) ln(1+z)   [analytical]
def D_M_coasting(z, H0):
    return (c_km_s / H0) * math.log(1.0 + z)

# ΛCDM
def E_lcdm(z, Om, OL, Or):
    return math.sqrt(Or * (1+z)**4 + Om * (1+z)**3 + OL)

def D_M_lcdm(z, H0, Om, OL, Or):
    integrand = lambda zp: 1.0 / E_lcdm(zp, Om, OL, Or)
    chi, _ = integrate.quad(integrand, 0.0, z, limit=500)
    return (c_km_s / H0) * chi

# Compute at z* under both
Or_LCDM = PLANCK_OMEGA_R_H2 / (PLANCK_H0/100.0)**2
DM_coast_sub = D_M_coasting(PLANCK_Z_STAR, H0_SUBSTRATE)
DM_coast_obs = D_M_coasting(PLANCK_Z_STAR, H0_OBSERVER)
DM_LCDM = D_M_lcdm(PLANCK_Z_STAR, PLANCK_H0, PLANCK_OMEGA_M, PLANCK_OMEGA_LAMBDA, Or_LCDM)

print(f"  D_M(z*) coasting (substrate H_0={H0_SUBSTRATE:.2f}):  {DM_coast_sub:.2f} Mpc")
print(f"  D_M(z*) coasting (observer  H_0={H0_OBSERVER:.2f}): {DM_coast_obs:.2f} Mpc")
print(f"  D_M(z*) ΛCDM      (Planck   H_0={PLANCK_H0:.2f}):     {DM_LCDM:.2f} Mpc")
print(f"  D_M(z*) Planck published:                            {PLANCK_D_M_STAR_MPC:.2f} Mpc")
print()
print(f"  Ratio D_M_coast(observer) / D_M_LCDM = {DM_coast_obs/DM_LCDM:.4f}")
print(f"  Ratio D_M_coast(substrate) / D_M_LCDM = {DM_coast_sub/DM_LCDM:.4f}")
print()
print("  **First observation:** the coasting D_M is roughly 2× the ΛCDM D_M.")
print("  Tantalizingly close to the empirical Λ_LCDM/Λ_substrate ≈ 2.06.")
print("  If r_s were the same in both cosmologies, this alone would shift θ*")
print("  by a factor of 2 in the ΛCDM-fit. But r_s under coasting is wildly")
print("  different — see §2.")
print()


# =============================================================================
# §2. r_s(z*) under coasting vs ΛCDM
# =============================================================================
print("§2. Comoving sound horizon r_s(z*)")
print("-" * 78)

# Sound speed in baryon-photon plasma:
#   c_s² = c² / [3(1 + R)]
#   R = (3 ρ_b)/(4 ρ_γ) = (3 Ω_b h²) / (4 Ω_γ h² × (1+z))
# Here Ω_γ h² ≈ 2.47e-5 (Planck CMB temperature-derived)
OMEGA_GAMMA_H2 = 2.47e-5

def R_baryon_photon(z):
    return (3.0 * PLANCK_OMEGA_BH2) / (4.0 * OMEGA_GAMMA_H2 * (1.0 + z))

def cs_kms(z):
    R = R_baryon_photon(z)
    cs2 = (c_km_s ** 2) / (3.0 * (1.0 + R))
    return math.sqrt(cs2)

# Coasting: H(z) = H_0(1+z)
def Hz_coast(z, H0):
    return H0 * (1.0 + z)

# ΛCDM: H(z) = H_0 E(z)
def Hz_lcdm(z, H0, Om, OL, Or):
    return H0 * E_lcdm(z, Om, OL, Or)

# Comoving sound horizon: r_s(z*) = ∫_{z*}^{∞} c_s(z) / H(z) dz
# In ΛCDM, integrand at high z falls as ∝ 1/(1+z)² (radiation-dominated H ∝ (1+z)²
# dominates the rapid c_s saturation), so integral is finite.
# In coasting, integrand at high z falls only as ∝ 1/(1+z) (since H_coast ∝ (1+z)
# only), so integral is logarithmically divergent without a UV cutoff.

# Test: compute r_s ΛCDM (numerical integration to large z_max)
def r_s_lcdm(H0, Om, OL, Or, z_star, z_max=1e6):
    integrand = lambda z: cs_kms(z) / Hz_lcdm(z, H0, Om, OL, Or)
    rs, _ = integrate.quad(integrand, z_star, z_max, limit=500)
    return rs

# Test: compute r_s coasting at various UV cutoffs (z_max)
def r_s_coast(H0, z_star, z_max):
    integrand = lambda z: cs_kms(z) / Hz_coast(z, H0)
    rs, _ = integrate.quad(integrand, z_star, z_max, limit=500)
    return rs

# Compute ΛCDM r_s
rs_LCDM = r_s_lcdm(PLANCK_H0, PLANCK_OMEGA_M, PLANCK_OMEGA_LAMBDA, Or_LCDM,
                   PLANCK_Z_STAR, z_max=1e8)
print(f"  r_s(z*) ΛCDM (integrated to z=1e8):     {rs_LCDM:.4f} Mpc")
print(f"  r_s(z*) Planck published:               {PLANCK_R_S_MPC:.4f} Mpc")
print(f"  Match: ratio {rs_LCDM/PLANCK_R_S_MPC:.4f}  (sanity check)")
print()

# Compute coasting r_s at various UV cutoffs
print("  r_s(z*) coasting (observer H_0=72.74) at various UV cutoffs z_max:")
print(f"    {'z_max':>14} {'r_s_coast (Mpc)':>20}")
for z_max in [1e3, 1e4, 1e5, 1e6, 1e10, 1e30, 1e60]:
    rs = r_s_coast(H0_OBSERVER, PLANCK_Z_STAR, z_max)
    print(f"    {z_max:>14.2e} {rs:>20.4f}")
print()
print("  Coasting r_s grows logarithmically with z_max — it's UV divergent.")
print("  Physical UV cutoff: z_max = 1/a_P ≈ t_now/t_P ≈ 8e60 (Planck epoch).")

# Use Planck-epoch cutoff
N_HUB = 8.4e60   # framework cascade scale (≈ t_now/t_P, theorem-grade)
z_planck_cutoff = N_HUB
rs_coast_planck_cutoff = r_s_coast(H0_OBSERVER, PLANCK_Z_STAR, z_planck_cutoff)
print(f"  r_s_coast (z_max = N_hub ≈ {z_planck_cutoff:.1e}):  {rs_coast_planck_cutoff:.2f} Mpc")
print()
print(f"  Ratio r_s_coast / r_s_LCDM = {rs_coast_planck_cutoff/rs_LCDM:.2f}")
print()
print("  r_s under coasting (with Planck-epoch UV cutoff) is ~3 ORDERS OF")
print("  MAGNITUDE larger than ΛCDM r_s. This is because at high z the coasting")
print("  H(z) ∝ (1+z) is much smaller than the ΛCDM H(z) ∝ (1+z)² (radiation-")
print("  dominated), so sound waves have FAR more time to propagate.")
print()


# =============================================================================
# §3. θ*(coasting) vs Planck θ*
# =============================================================================
print("§3. Acoustic peak angular scale θ_* = r_s / D_M")
print("-" * 78)

theta_star_LCDM = rs_LCDM / DM_LCDM
theta_star_coast_obs_planck_cutoff = rs_coast_planck_cutoff / DM_coast_obs
theta_star_coast_sub_planck_cutoff = r_s_coast(H0_SUBSTRATE, PLANCK_Z_STAR,
                                                 z_planck_cutoff) / DM_coast_sub

print(f"  θ_* ΛCDM (computed):                       100·θ_* = {100*theta_star_LCDM:.5f}")
print(f"  θ_* Planck published:                      100·θ_* = {PLANCK_THETA_STAR_TIMES_100:.5f}")
print(f"  Match: ratio {theta_star_LCDM/(PLANCK_THETA_STAR_TIMES_100/100):.4f} (sanity check)")
print()
print(f"  θ_* coasting (observer, Planck cutoff):    100·θ_* = {100*theta_star_coast_obs_planck_cutoff:.5f}")
print(f"  θ_* coasting (substrate, Planck cutoff):   100·θ_* = {100*theta_star_coast_sub_planck_cutoff:.5f}")
print()
print(f"  Ratio θ_*_coast / θ_*_Planck (observer) = {theta_star_coast_obs_planck_cutoff*100/PLANCK_THETA_STAR_TIMES_100:.2f}")
print(f"  Discrepancy in σ_obs:                    {(theta_star_coast_obs_planck_cutoff*100 - PLANCK_THETA_STAR_TIMES_100)/PLANCK_THETA_STAR_TIMES_100_SIGMA:+.0f}σ")
print()


# =============================================================================
# §4. What this means for Path A
# =============================================================================
print("§4. What this means for Path A")
print("-" * 78)
print(f"""
  Coasting θ_* (with the natural Planck-epoch UV cutoff) is ~{theta_star_coast_obs_planck_cutoff*100/PLANCK_THETA_STAR_TIMES_100:.0f}× larger
  than Planck-measured θ_*. This is a falsification at ~10⁵σ_obs — many
  orders of magnitude beyond any reasonable "factor-of-2 mis-extraction"
  closure.

  The mechanism:
    - D_M_coast(z*) = (c/H_0) ln(1+z*) ≈ 2× D_M_LCDM(z*).  (Tantalizing
      factor-of-2.)
    - r_s_coast(z*) ≈ {rs_coast_planck_cutoff/rs_LCDM:.0f}× r_s_LCDM(z*) due to UV-divergent integral
      with no radiation-domination regulator.
    - θ_*_coast / θ_*_LCDM ≈ {(rs_coast_planck_cutoff/rs_LCDM)/(DM_coast_obs/DM_LCDM):.0f}×.

  The factor-of-2 in D_M is "right shape" for the empirical Λ_LCDM/Λ_substrate
  ratio, but the divergent r_s overwhelms it by ~3 orders of magnitude.

  Implications for the factor-of-2 hypothesis:

  - The simple form ("coasting at all epochs + ΛCDM-fit → empirical
    (Ω_m, Ω_Λ) = (1/3, 2/3)") is structurally falsified at CMB level.
    Coasting does NOT reproduce CMB acoustic peaks, period.

  - For the factor-of-2 hypothesis to survive, the framework must specify
    a NON-COASTING early-universe regime that regulates r_s. This is
    pre-recombination physics that the framework's current cosmology
    arc has not closed (cf. `proofs/cosmology/early_universe_k_rundown.py`
    sketch for k-cooling, but no quantitative r_s prediction from it).

  - The empirical match Ω_m_LCDM ≈ 0.315 ≈ 1/3 (within 5%) cannot be
    explained by ΛCDM mis-extraction of pure-coasting data. It must
    either be:
      (a) coincidence at percent level (factor-of-2 is structural illusion);
      (b) explained by a more elaborate framework cosmology with non-
          coasting early-universe phase that the framework hasn't published;
      (c) some other mechanism not in scope of Path A as defined.

  Path A SUMMARY (Sessions 1+2):
    - Session 1 (SN1a): factor-of-2 prediction (Ω_m → 1/3) misses by 6σ_fit.
    - Session 2 (CMB θ_*): coasting falsified by ~10⁵σ at the strongest
      single observable.
    - Honest verdict: Path A's data-side refit closure of P24/P23 is
      BLOCKED. The factor-of-2 cannot be reproduced by coasting + ΛCDM-
      fitting at any data channel I've tested.
""")


# =============================================================================
# §5. Selection grammar disclosure
# =============================================================================
print("§5. Selection grammar disclosure")
print("-" * 78)
print("""
  This session's comparison is channel_select between two cosmological
  candidates (coasting vs ΛCDM). Both are physically realizable above
  the A2-T waterline; observation (Planck θ_*) selects the K-candidate
  that matches the data. Per `theorem_lattice_coupling_general.md` §2,
  this is NOT canonical_encoding (the channels predict different θ_*
  values; they are physically distinct objects).

  The discrepancy ~10⁵σ between coasting θ_* and Planck θ_* is too large
  to attribute to a "different observational channel" in any natural
  sense. Either:
    (a) the coasting prediction is wrong at high z (early-universe physics
        is non-coasting), or
    (b) the framework's prediction H = 1/t at all epochs needs revision.

  Both options put Path A's structural premise into question. The
  cosmology arc's roadmap for closing P24 should not assume Path A is
  viable as currently scoped.
""")


# =============================================================================
# §6. Verdict
# =============================================================================
print("§6. Verdict — Path A status after Session 2")
print("=" * 78)
print(f"""
  Coasting H = 1/t at all epochs (the framework's cascade-theorem
  prediction) is incompatible with Planck-measured CMB acoustic peak
  position by ~10⁵σ. Combined with Session 1's SN1a-only failure (Ω_m
  mis-extraction lands at 0.53 not 1/3), the simple factor-of-2 hypothesis
  ("ΛCDM-fitting of coasting data recovers (1/3, 2/3)") does NOT close
  P24 or Row P23.

  Path A status: **BLOCKED.** Same outcome as Path B (which was blocked
  on framework prerequisites for substrate w_eff mixing).

  This means BOTH proposed closures of the Λ_CC factor-of-2 are now
  blocked:
    - Path B: blocked on framework's substrate-FLRW T^ab bridge (g1a O3.1-2,
      O4.1-2; cosmology Item 2 Session 2 honest negative)
    - Path A: blocked on framework's pre-coasting early-universe physics
      (this session)

  Implications for ledger:
    - Row P23 (absolute Ω_DM, Ω_b) stays THEOREM-GRADE-CONDITIONAL on P24,
      with neither candidate closure (A or B) currently viable. The +2.2σ
      (Ω_b) and +2.6σ (Ω_m, Ω_Λ, Ω_DM) Planck residuals stay as known
      systematic floor reflecting the unresolved factor-of-2 reorganization.
    - P24 (Λ_CC) stays UNIQUE-THEOREM-GRADE-CONDITIONAL on the factor-of-2
      decomposition residue. Both candidate closures blocked means closure
      requires a NEW path not in the current roadmap.

  Possible new path candidates (not yet scoped):
    - Path D — substrate cosmology with non-coasting early-universe phase
      (k-cooling era? framework's `early_universe_k_rundown.py` sketch).
      Multi-session.
    - Path E — accept factor-of-2 as percent-level structural coincidence;
      not a true closure. Honest concession.
    - Path F — re-examine whether the factor-of-2 in Lambda_CC_factor_two
      decomposition is a real structural relationship or a numerical
      coincidence; revisit at the level of "what does ΛCDM extraction
      from the framework's TRUE cosmology actually give?".

  Recommended next session: stop attacking Path A; instead audit whether
  the factor-of-2 is structurally meaningful at all (Path F-like). If
  it is, the closure path is open research. If it's coincidence, P24
  stays open with no bounded closure path.
""")
print("=" * 78)
print("DONE: Session 2 of Λ_CC Path A. Path A BLOCKED.")
print("=" * 78)
