#!/usr/bin/env python3
"""
Path D probe (D.5) — Sound-speed c_s(z) modification (2026-05-15 EOD+1).

HYPOTHESIS
----------
The CMB θ_* tension might close through a substrate-derived c_s(z) different
from ΛCDM's c_s ≈ c/√(3(1+R)) — without modifying H(z) from coasting.

If a substrate-derived c_s gives r_s ~ 150 Mpc under coasting H(z) = H_0(1+z),
then Item 5 closes via c_s modification rather than H(z) modification.

APPROACH
--------
1. Compute r_s under coasting H(z) as a function of c_s.
2. Determine what c_s would be needed for r_s ≈ 147 Mpc (Planck value).
3. Check whether any framework-derived c_s candidate matches.

PRE-COMPUTATION EXPECTATION
---------------------------
Under coasting, conformal time η = ∫dt/a = t_0·ln(t/t_min) diverges
logarithmically as t_min → 0.  The framework's natural cutoff at T_srs
gives ln(t_*/t_srs) ≈ 48.

r_s = c_s · t_0 · ln(t_*/t_srs) ≈ (c_s/c) · 4400 Mpc · 48

For r_s = 147 Mpc, need c_s/c ≈ 0.0007 — far below physical photon-baryon
plasma sound speed c/√(3(1+R)) ≈ 0.4 to 0.58.  Negative likely.
"""

from __future__ import annotations
import math
import sys
import os

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============================================================================
# Constants and framework primitives
# ============================================================================
K_STAR = 3
G_GIRTH = 10
ALPHA_1 = (2.0 / 3.0) ** (G_GIRTH - 2)
N_G = 15
C_SRS_BITS = N_G * math.log2(1.0 / ALPHA_1)
LN2 = math.log(2)
T_SRS_PLANCK = C_SRS_BITS / LN2

T_PLANCK_K = 1.416784e32
T_CMB_TODAY_K = 2.7255
H_0_KMSMPC = 67.4   # Planck-extracted H_0 (used for r_s comparison vs Planck r_s)
MPC_IN_KM = 3.086e19
c_KMS = 2.998e5
HUBBLE_RADIUS_MPC = c_KMS / H_0_KMSMPC      # ~4448 Mpc

# Recombination
Z_STAR = 1089.0
T_STAR_K = T_CMB_TODAY_K * (1 + Z_STAR)

# Planck's measured comoving sound horizon at recombination
R_S_PLANCK_MPC = 147.06


# ============================================================================
# § A. r_s under coasting H(z), as function of c_s
# ============================================================================
def r_s_under_coasting(c_s_over_c: float, t_min_over_t0: float) -> float:
    """
    r_s = ∫_0^{a_*} c_s · dt/a · ... under coasting.

    Under coasting a(t) = t/t_0:
      η = ∫dt/a = t_0 · ln(t/t_min)  (logarithmic, divergent at t_min → 0)

    r_s = c_s · ∫_0^{η_*} dη = c_s · η_*
        = c_s · t_0 · ln(t_*/t_min)
        = (c_s/c) · c·t_0 · ln(t_*/t_min)
        = (c_s/c) · Hubble_radius · ln(t_*/t_min)

    t_*/t_0 = 1/(1+z_*) under coasting, so:
      t_* = t_0/(1+z_*)

    Cutoff: t_min < t_*.  Required for finite r_s.

    Parameters
    ----------
    c_s_over_c : dimensionless sound speed (e.g., 1/√3 ≈ 0.577)
    t_min_over_t0 : cutoff time relative to t_0

    Returns
    -------
    r_s in Mpc.
    """
    t_star_over_t0 = 1.0 / (1.0 + Z_STAR)
    ln_ratio = math.log(t_star_over_t0 / t_min_over_t0)
    return c_s_over_c * HUBBLE_RADIUS_MPC * ln_ratio


# ============================================================================
# § B. Survey natural cutoff scales + standard sound speed
# ============================================================================
print("=" * 80)
print(" Path D probe (D.5) — sound-speed c_s(z) modification")
print("=" * 80)
print()
print(f"  Target: Planck r_s ≈ {R_S_PLANCK_MPC} Mpc at recombination (z = {Z_STAR})")
print(f"  Hubble radius today: c/H_0 = {HUBBLE_RADIUS_MPC:.1f} Mpc")
print()
print(f"  Substrate-stability cutoff (Planck units):")
print(f"    T_srs ≈ {T_SRS_PLANCK:.2f}, T_srs in K ≈ {T_SRS_PLANCK * T_PLANCK_K:.2e}")
print(f"    z_srs ≈ T_srs / T_CMB_today ≈ {T_SRS_PLANCK * T_PLANCK_K / T_CMB_TODAY_K:.2e}")
print()
print("-" * 80)
print("§B. r_s under coasting H(z) with various cutoffs + c_s = c/√3")
print("-" * 80)
print()
print(f"  Standard photon-baryon plasma: c_s² = c²/(3(1+R)), R ~ 0.6 at recomb")
print(f"    c_s/c ≈ 1/√(3·1.6) ≈ 0.456 (slightly below 1/√3)")
print(f"    For simplicity, use c_s = c/√3 ≈ 0.577 (negligible baryon loading)")
print()
print(f"  {'Cutoff scale':<40} {'t_min/t_0':<15} {'ln(t_*/t_min)':<18} {'r_s (Mpc)':<15}")
print(f"  {'-'*40} {'-'*15} {'-'*18} {'-'*15}")

cutoffs = [
    ("T_srs (substrate-stability)",   1.0 / (T_SRS_PLANCK * T_PLANCK_K / T_CMB_TODAY_K)),
    ("Electroweak (T = v_higgs)",     1.0 / (1.0 + 246.22 / 1.22089e19 * T_PLANCK_K / T_CMB_TODAY_K)),
    ("QCD transition",                1.0 / (1.0 + 0.2 / 1.22089e19 * T_PLANCK_K / T_CMB_TODAY_K)),
    ("BBN (T ~ 0.1 MeV)",             1.0 / (1.0 + 1e-4 / 1.22089e19 * T_PLANCK_K / T_CMB_TODAY_K)),
    ("Matter-rad equality (z=3400)",  1.0 / 3400.0),
]

c_s_standard = 1.0 / math.sqrt(3)
for name, t_min_t0 in cutoffs:
    t_star_t0 = 1.0 / (1.0 + Z_STAR)
    ln_ratio = math.log(t_star_t0 / t_min_t0)
    r_s = r_s_under_coasting(c_s_standard, t_min_t0)
    print(f"  {name:<40} {t_min_t0:<15.3e} {ln_ratio:<18.2f} {r_s:<15.1f}")

print()
print(f"  Planck observed r_s = {R_S_PLANCK_MPC} Mpc")
print()
print(f"  Verdict: with standard c_s = c/√3 and ANY reasonable cutoff (BBN, EW,")
print(f"  or T_srs), r_s under coasting is FAR too large (3000-130000 Mpc range).")
print(f"  None match Planck's 147 Mpc.")
print()


# ============================================================================
# § C. What c_s would give r_s = Planck under coasting?
# ============================================================================
print("-" * 80)
print("§C. Required c_s for Planck r_s = 147 Mpc under coasting")
print("-" * 80)
print()
print(f"  Solving r_s = (c_s/c) · c·t_0 · ln(t_*/t_min) = {R_S_PLANCK_MPC} Mpc:")
print()
print(f"  {'Cutoff':<40} {'ln(t_*/t_min)':<18} {'Required c_s/c':<18}")
print(f"  {'-'*40} {'-'*18} {'-'*18}")
for name, t_min_t0 in cutoffs:
    t_star_t0 = 1.0 / (1.0 + Z_STAR)
    ln_ratio = math.log(t_star_t0 / t_min_t0)
    c_s_over_c_req = R_S_PLANCK_MPC / (HUBBLE_RADIUS_MPC * ln_ratio)
    print(f"  {name:<40} {ln_ratio:<18.2f} {c_s_over_c_req:<18.6f}")

print()
print(f"  Even with the most extreme cutoff (T_srs giving ln ≈ 80),")
print(f"  required c_s/c is ~4×10⁻⁴, i.e. c_s ≈ 100 km/s.")
print(f"  For natural cutoffs at BBN or EW, required c_s/c is ~10⁻³ to 10⁻⁴.")
print()
print(f"  STANDARD photon-baryon plasma c_s ≈ 0.4-0.58 c.")
print(f"  Required c_s is ~3 orders of magnitude smaller.")
print(f"  No physical sound speed in any thermal plasma is this small.")
print()


# ============================================================================
# § D. Test framework-derived c_s candidates
# ============================================================================
print("-" * 80)
print("§D. Framework-derived c_s candidates — do any match required value?")
print("-" * 80)
print()
print(f"  Test substrate-natural dimensionless quantities as candidate c_s/c:")
print()

candidates = [
    ("1/√3 (photon-baryon plasma)",       1.0 / math.sqrt(3)),
    ("1/k* = 1/3",                        1.0/3.0),
    ("α_1_bare = (2/3)^8",                ALPHA_1),
    ("α_1_full = α_1/(1-α_1)",            ALPHA_1/(1-ALPHA_1)),
    ("(5/12)·α_1/(1-α_1)",                (5.0/12.0)*ALPHA_1/(1-ALPHA_1)),
    ("α_1²",                              ALPHA_1**2),
    ("α_1²/k*",                           ALPHA_1**2 / K_STAR),
    ("α_1² / (N_atoms·k*)",                ALPHA_1**2 / (4*K_STAR)),
    ("(v_higgs/M_Pl)²",                   (246.22/1.22089e19)**2),
    ("v_higgs/M_Pl",                      246.22/1.22089e19),
]

print(f"  {'Candidate':<40} {'c_s/c value':<18} {'r_s (T_srs cutoff)':<22}")
print(f"  {'-'*40} {'-'*18} {'-'*22}")
t_min_T_srs = 1.0 / (T_SRS_PLANCK * T_PLANCK_K / T_CMB_TODAY_K)
for name, c_s_val in candidates:
    r_s_val = r_s_under_coasting(c_s_val, t_min_T_srs)
    match = " ← matches Planck" if abs(r_s_val - R_S_PLANCK_MPC) < 0.1 * R_S_PLANCK_MPC else ""
    print(f"  {name:<40} {c_s_val:<18.4e} {r_s_val:<22.1f}{match}")

print()
print(f"  Verdict: no framework-natural quantity gives c_s/c ≈ 4×10⁻⁴ required.")
print(f"  α_1² ≈ 1.5×10⁻³ is closest in magnitude but still factor 4 off, and")
print(f"  there's no derivation that would identify α_1² with a sound speed.")
print(f"  Pattern-matching α_1²/N_atoms·k* ≈ 1.3×10⁻⁴ to required ~4×10⁻⁴ is")
print(f"  NOT acceptable per parameter_linter §'NOT acceptable':")
print(f"    'Any step that selects between alternatives by comparing to data.'")
print()


# ============================================================================
# § E. Why the framework can't have an arbitrarily small c_s
# ============================================================================
print("-" * 80)
print("§E. Why physical c_s must be of order c/√3 at recombination")
print("-" * 80)
print()
print(f"  Sound speed in a fluid c_s² = ∂p/∂ρ at fixed entropy.")
print(f"  At recombination, the photon-baryon plasma has:")
print(f"    p_γ = ρ_γ/3  (photon equation of state w_γ = 1/3)")
print(f"    p_b ≈ 0      (cold baryons; pressureless dust)")
print(f"    ρ_total = ρ_γ + ρ_b")
print()
print(f"  c_s² = (∂p/∂ρ)_S = p_γ' / (ρ_γ' + ρ_b')")
print(f"       = (4/3)·ρ_γ·H / (4·ρ_γ·H + 3·ρ_b·H)")
print(f"       = c²/(3·(1 + (3·ρ_b)/(4·ρ_γ)))")
print(f"       = c²/(3·(1 + R))")
print(f"  With R = 3ρ_b/(4ρ_γ) ≈ 0.6 at recombination, c_s ≈ 0.46·c.")
print()
print(f"  This is BASIC FLUID DYNAMICS — independent of framework's substrate-")
print(f"  level physics.  As long as the framework's emergent gravity reproduces")
print(f"  GR at cosmological scales, photon-baryon plasma at recombination has")
print(f"  the standard c_s.  No substrate-derived modification can change this")
print(f"  without breaking the emergent-gravity match.")
print()
print(f"  The only way to get c_s ~ 4×10⁻⁴·c is to have the framework's")
print(f"  substrate-level sound speed be DIFFERENT from the photon-baryon plasma")
print(f"  sound speed — i.e., the 'sound horizon' observed at the CMB is set by")
print(f"  a SUBSTRATE-LEVEL propagation mode, not by photon-baryon hydrodynamics.")
print()
print(f"  But the framework's substrate-level propagation speed (Hashimoto walker")
print(f"  group velocity at P-point) is O(c) in lattice units — not O(10⁻⁴·c).")
print()


# ============================================================================
# § F. Verdict
# ============================================================================
print("=" * 80)
print("VERDICT — Path D probe (D.5)")
print("=" * 80)
print()
print(f"  Sound-speed c_s(z) modification CANNOT close Item 5 under coasting H(z).")
print()
print(f"  QUANTITATIVE FINDING:")
print(f"    Required c_s ≈ 4×10⁻⁴·c ≈ 100 km/s to give r_s = 147 Mpc under")
print(f"    coasting with T_srs cutoff.  This is ~3 orders of magnitude below")
print(f"    standard photon-baryon plasma c_s ≈ 0.46·c at recombination.")
print()
print(f"  No physical sound speed in any thermal plasma is this small.")
print(f"  No framework-natural quantity matches the required value without")
print(f"  pattern-matching to data.")
print()
print(f"  STRUCTURAL CONCLUSION:")
print(f"    Framework's emergent gravity at cosmological scales must reproduce GR")
print(f"    (Row P60 G_N·M_Pl² = 1 theorem-grade); photon-baryon plasma at")
print(f"    recombination has standard hydrodynamics with c_s² = c²/(3(1+R)).")
print(f"    Substrate-derived modifications to c_s would break the emergent-gravity")
print(f"    match, which is itself theorem-grade.")
print()
print(f"  Path D probe (D.5) is CLOSED-NEGATIVE.")
print()
print(f"  PATH D DECISION TREE STATUS (FINAL):")
print(f"    (D.1) CLOSED-NEGATIVE — thermal MDL acceptance suppression negligible")
print(f"    (D.4) CLOSED-NEGATIVE — D3 has no sub-T_srs structural transition")
print(f"    (D.5) CLOSED-NEGATIVE — c_s modification can't rescue r_s (this probe)")
print(f"    (D.2) BLOCKED by Need A of MS.1 (multiway formalization)")
print(f"    (D.3) BLOCKED by partial-negative in early_universe_k_rundown.py")
print()
print(f"  ALL THREE BOUNDED PATH D PROBES NOW CLOSED-NEGATIVE.")
print()
print(f"  HONEST CONCLUSION (Item 5):")
print(f"    Path D is research-level only via (D.2)/(D.3), and both are blocked")
print(f"    on Need A of MS.1 (multiway formalization).  Item 5 has no near-term")
print(f"    bounded closure path.")
print()
print(f"    The CMB θ_* tension is a real structural incompleteness in the")
print(f"    framework's pre-recombination story.  Late-time predictions (Rows")
print(f"    P17, P19, P20, P22, P24) remain insulated.  High-z derived predictions")
print(f"    (P25, P26, P29) remain insulated.  But the framework cannot currently")
print(f"    predict CMB acoustic-peak structure without invoking pre-recombination")
print(f"    machinery that isn't structurally derived.")
print()
print(f"  RECOMMENDED PATH FORWARD:")
print(f"    Accept Item 5 as a multi-session research direction.  Document Λ_CC")
print(f"    factor-of-2 and Ω_DM absolute as THEOREM-GRADE-CONDITIONAL on Item 5")
print(f"    in their ledger rows.  The structural prediction Λ_substrate = 1/N²")
print(f"    in the coasting frame is theorem-grade and ships in predictions/Lambda_CC.py.")
print(f"    The factor-of-2 vs ΛCDM-fit Planck Λ_LCDM is OPEN — and this probe")
print(f"    confirms that no bounded mechanism rescues it.")
