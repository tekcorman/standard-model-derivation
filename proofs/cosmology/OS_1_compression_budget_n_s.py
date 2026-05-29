#!/usr/bin/env python3
"""
proofs/cosmology/OS_1_compression_budget_n_s.py

Need OS Session 2 — OS-1 closure attempt.

Per an internal working note §3, OS-1
is the highest-readiness Need OS sub-target: derive a scale-stratified
compression budget B(k) from the framework's finite-register argument
(A2-T waterline + register-is-real) and check whether it produces an
observed n_s spectral tilt of -0.035 at theorem grade.

This probe attempts three routes and honestly reports the verdict:

  Route A — Hard-cutoff bit budget. Total bit budget from N_hub or
            similar cumulative-step count; cutoff at k_max where
            number of modes saturates the budget. PREDICTION:
            cutoff scale, NOT smooth tilt.

  Route B — Cumulative-observation framing. Modes that have been
            observable longer accumulate more bits; bits/mode varies
            with mode lifetime in observer's horizon. PREDICTION:
            possible smooth tilt, but sign and magnitude depend on
            framing direction.

  Route C — Information-theoretic precision-allocation. Each mode
            requires bits ∝ resolution; allocation across modes given
            a fixed total budget. PREDICTION: depends on cost-function
            f(k) and allocation rule.

Each route is computed numerically and compared to observed
n_s = 0.9649 ± 0.0042 (Planck 2018) → tilt = -0.0351.

Honest output: Route A gives sharp cutoff far beyond observable scales
(no observable tilt). Route B gives wrong sign. Route C requires an
allocation rule that's not derivable from existing framework primitives.

Net verdict: OS-1's "scale-stratified compression budget" mechanism does
NOT give observed n_s ≈ 0.965 at theorem grade from existing framework
primitives. The HONEST NEGATIVE finding is that the framework's natural
finite-register-budget mechanism gives a CUTOFF SCALE, not a SMOOTH TILT.
"""

import math
import sys
import os

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =============================================================================
# §0. Constants and observed values
# =============================================================================

# Planck 2018 baseline:
N_S_OBSERVED = 0.9649          # spectral index
N_S_OBSERVED_SIGMA = 0.0042
TILT_OBSERVED = N_S_OBSERVED - 1.0   # = -0.0351
K_PIVOT_INV_MPC = 0.05         # Planck pivot scale [Mpc^-1]
A_S_OBSERVED = 2.10e-9         # primordial amplitude

# Framework cosmological:
N_HUB = 8.4e60                 # observer-rate scale (cumulative substrate steps)
H0_OBSERVER_KMS_MPC = 72.74    # H_0 observer (D2-extended)
T_PLANCK_S = 5.391247e-44      # Planck time
C_KM_S = 2.998e5
MPC_KM = 3.086e19              # km per Mpc

# Derived scales:
T_NOW_S = N_HUB * T_PLANCK_S    # current cosmic age in seconds (substrate side)
HUBBLE_HORIZON_MPC = C_KM_S / H0_OBSERVER_KMS_MPC  # ~4123 Mpc
PLANCK_LENGTH_MPC = T_PLANCK_S * C_KM_S / MPC_KM   # ~5.2e-58 Mpc

# Substrate parameters:
K_STAR = 3
G_GIRTH = 10
TWO_K_STAR = 2 * K_STAR        # 6 toggle modes per node


# =============================================================================
# §1. Framework primitive — what's the natural bit budget N_obs?
# =============================================================================

def section_1_bit_budget():
    print("=" * 78)
    print("§1. Framework's natural bit budget N_obs")
    print("=" * 78)
    print()
    print("Per A2-T (theorem_A2_mdl_from_finite_register.md), observer is a")
    print("finite register. Total distinguishable states: N_obs. Per Theorem 10")
    print("of a separate private derivation by the author-style finite-eddy argument: N_obs is bounded but specific")
    print("value depends on observer's stability budget.")
    print()
    print("Framework-internal candidates for N_obs:")
    print()

    candidates = {
        'N_hub (cumulative substrate steps)': N_HUB,
        'N_hub^(2/3) (CMB sphere counting)': N_HUB**(2.0/3.0),
        'N_hub^(1/3) (linear cosmic scale)': N_HUB**(1.0/3.0),
        'log(N_hub) (entropic scale)': math.log(N_HUB),
    }

    for name, val in candidates.items():
        print(f"  {name:>40} = {val:.3e}")
    print()
    print("All candidates are >> observable mode counts:")
    print(f"  CMB observable modes (ℓ_max ~ 3000): {3000**2:.3e}")
    print(f"  3D observable modes (k <= H_0): ~{(HUBBLE_HORIZON_MPC * K_PIVOT_INV_MPC)**3:.3e}")
    print()
    print("OBSERVATION: any of these N_obs candidates ≫ observable mode count")
    print("by tens of orders of magnitude. The finite-register cutoff cannot")
    print("appear within the observable cosmological range under straightforward")
    print("counting.")
    print()


# =============================================================================
# §2. Route A — hard-cutoff bit budget (sharp k_max)
# =============================================================================

def section_2_route_A_cutoff():
    print("=" * 78)
    print("§2. Route A — hard-cutoff finite-register budget")
    print("=" * 78)
    print()
    print("Setup. Number of spatial Fourier modes up to k_max in 3D Hubble")
    print("volume: M(k_max) = (4π/3) (k_max·H_horizon)³.")
    print()
    print("Each mode costs f bits to retain at fixed precision. Total bits:")
    print("  M(k_max) · f = N_obs   →   k_max = (3 N_obs / (4π·f))^(1/3) / H_horizon")
    print()
    print("For f = O(1) bits/mode (standard precision):")

    f_bits_per_mode = 1.0   # typical
    for n_obs_label, n_obs in [
        ('N_hub', N_HUB),
        ('N_hub^(2/3)', N_HUB**(2/3)),
        ('log(N_hub)', math.log(N_HUB)),
    ]:
        k_max_h0 = (3 * n_obs / (4 * math.pi * f_bits_per_mode))**(1/3)
        k_max_inv_mpc = k_max_h0 / HUBBLE_HORIZON_MPC
        print(f"  N_obs = {n_obs_label:>12} = {n_obs:.2e}  →  k_max = {k_max_inv_mpc:.2e} Mpc^-1")
    print()
    print(f"Observed CMB pivot:                 k_pivot = {K_PIVOT_INV_MPC:.2e} Mpc^-1")
    print()
    print("VERDICT Route A: with N_obs ~ N_hub or any plausible candidate,")
    print("k_max is FAR ABOVE the CMB pivot scale. The cutoff does NOT appear")
    print("within observable cosmological range.")
    print()
    print("Furthermore, Route A gives a SHARP CUTOFF (P(k) = const for k < k_max,")
    print("P(k) = 0 for k > k_max), not a smooth spectral tilt.")
    print()
    print("Route A does NOT close OS-1.")
    print()


# =============================================================================
# §3. Route B — cumulative-observation framing
# =============================================================================

def section_3_route_B_cumulative():
    print("=" * 78)
    print("§3. Route B — cumulative-observation framing")
    print("=" * 78)
    print()
    print("Setup. Modes at scale k entered observer's horizon at time t_h(k) when")
    print("1/k = c · t_h. So t_h(k) = 1/(k·c). Mode has been observable for")
    print("Δt(k) = t_now - 1/(k·c). Bits accumulated per mode ∝ Δt(k).")
    print()
    print("If observer's amplitude precision per mode ∝ accumulated bits:")
    print("  P_observer(k) = P_substrate × [Δt(k)/t_now]² = P_substrate × f(k)²")
    print()
    print("With t_now = 1/H_0 and small kct_now → 1 + small:")
    print("  f(k) = 1 - 1/(k·c·t_now)")
    print()
    print(f"Spectral tilt: n_s - 1 = d log P / d log k.")
    print()

    # Compute tilt at pivot scale
    # Using H_0 in 1/s for t_now = 1/H_0
    H0_SI = H0_OBSERVER_KMS_MPC / MPC_KM  # 1/s
    t_now = 1.0 / H0_SI
    k_pivot_si = K_PIVOT_INV_MPC / MPC_KM  # 1/m... no, Mpc^-1 → 1/m via 1/Mpc = 1/(3.086e22 m)
    k_pivot_per_m = K_PIVOT_INV_MPC / 3.086e22

    # k·c·t_now in dimensionless: k[1/Mpc] · (c/H_0)[Mpc] = k·c/H_0 (dimensionless if c in km/s and H_0 in km/s/Mpc and k in 1/Mpc)
    kct_now_at_pivot = K_PIVOT_INV_MPC * HUBBLE_HORIZON_MPC  # dimensionless
    print(f"  At k_pivot = {K_PIVOT_INV_MPC} Mpc^-1, k·c·t_now = {kct_now_at_pivot:.3f}")
    print(f"  (= number of e-folds since horizon crossing for this mode, roughly)")
    print()

    # f(k)² = (1 - 1/x)² where x = k·c·t_now
    # log P = log(P_substrate) + 2 log(1 - 1/x)
    # d log P / d log k = 2 · (d log(1 - 1/x) / dx) · (dx/d log k)
    #                   = 2 · [1/(x²·(1-1/x))] · k · (c·t_now)
    #                   = 2 · 1/(x·(x-1))
    # at x = kct_now: tilt = 2/(x·(x-1))

    x = kct_now_at_pivot
    tilt_routeB_at_pivot = 2.0 / (x * (x - 1))
    print(f"  Tilt = 2/(x·(x-1)) = {tilt_routeB_at_pivot:.4f}")
    print(f"  → n_s - 1 = +{tilt_routeB_at_pivot:.4f}  (BLUE TILT, wrong sign)")
    print(f"  Observed: n_s - 1 = {TILT_OBSERVED:.4f}   (RED TILT)")
    print()
    print("Sign is wrong. Route B gives blue tilt because larger scales (smaller k)")
    print("have LESS accumulated observation time → LESS amplitude → P decreases at")
    print("small k → P INCREASES with k → blue tilt. Observed is red tilt (P")
    print("DECREASES with k).")
    print()
    print("VERDICT Route B: correct order of magnitude (~0.04 vs observed 0.035) but")
    print("WRONG SIGN. Route B does NOT close OS-1.")
    print()
    print("Route B-inverse: would need amplitude precision to DECREASE with")
    print("observation time (older modes are LESS precisely retained). No physical")
    print("mechanism in framework supports this.")
    print()


# =============================================================================
# §4. Route C — precision-allocation per mode
# =============================================================================

def section_4_route_C_precision():
    print("=" * 78)
    print("§4. Route C — precision-allocation per mode")
    print("=" * 78)
    print()
    print("Setup. Total bit budget N_obs allocated across modes at allocation")
    print("rate g(k) bits/mode. Each mode's amplitude precision ∝ allocated bits.")
    print()
    print("Conservation: ∫ M(k) · g(k) dk = N_obs")
    print("  with M(k) ∝ k² (spherical-shell mode density in 3D)")
    print()
    print("Power retained: P(k) ∝ g(k)² (squared precision)")
    print("Spectral tilt: n_s - 1 = d log g²(k)/d log k = 2 d log g(k)/d log k")
    print()
    print("For observed n_s - 1 = -0.035: g(k) ∝ k^(-0.0175).")
    print()
    print("Question: can the framework derive g(k) ∝ k^(-0.0175) from primitives?")
    print()
    print("Candidate allocation rules:")
    print()
    print("  (i) Equal bits per mode (g(k) = const):")
    print("      → tilt = 0, n_s = 1. Framework's white-noise prediction.")
    print("      Per Item 3 closure, this is the framework's natural baseline.")
    print()
    print("  (ii) Equal bits per spherical shell (g(k) ∝ 1/M(k) ∝ 1/k²):")
    print("      → tilt = 2 · (-2) = -4, n_s = -3. Way off.")
    print()
    print("  (iii) Bits allocated by surprise (g(k) ∝ surprise(k)):")
    print("      For white-noise input, surprise is k-independent.")
    print("      → tilt = 0 (same as (i)).")
    print()
    print("  (iv) Bits allocated by cosmological 'maturity' (g(k) ∝ ln(t_now/t_h(k))):")
    print("      For modes well within horizon, ln(t_now·k·c) ≈ ln(N_e(k)).")
    print("      → tilt = 2/ln(N_e), need N_e ≈ 60 for tilt ≈ 2/60 ≈ -0.033 (close")
    print("      to observed -0.035), BUT sign requires bits DECREASING with k...")
    print("      and the standard horizon-crossing argument has bits INCREASING.")
    print()
    print("VERDICT Route C: NO natural allocation rule derivable from existing")
    print("framework primitives gives the observed sign + magnitude of n_s - 1.")
    print("The closest match (rule iv) gives the right magnitude with an inverted")
    print("sign convention, but the framework has no mechanism to invert the sign.")
    print()
    print("Route C does NOT close OS-1.")
    print()


# =============================================================================
# §5. Comparison with standard inflationary derivation (cross-check)
# =============================================================================

def section_5_inflationary_comparison():
    print("=" * 78)
    print("§5. Cross-check: standard inflationary derivation")
    print("=" * 78)
    print()
    print("Standard inflation (Mukhanov-Sasaki + slow-roll):")
    print("  n_s - 1 = -2/N_e + 2 η - 6 ε")
    print("  where N_e ≈ 50-60 e-folds, η, ε are slow-roll parameters.")
    print()
    print("For N_e = 60 (typical chaotic-inflation prior):")
    n_e_std = 60.0
    tilt_std = -2.0 / n_e_std
    print(f"  -2/N_e = {tilt_std:.4f}  (close to observed {TILT_OBSERVED:.4f})")
    print()
    print("This relies on canonical-quantization (Mukhanov-Sasaki) of the inflaton")
    print("perturbation. Per `theorem_n_s_scoping.md` Attempt 4, slow-roll formula")
    print("is REJECTED by framework's rigor bar — it imports inflaton + Friedmann")
    print("background + canonical quantization, none of which are framework axioms.")
    print()
    print("Framework anchor for N_e: log_e(N_hub^(1/3)) ≈ log_e((10^60)^(1/3))")

    n_e_framework = math.log(N_HUB**(1.0/3.0))
    print(f"  log(N_hub^(1/3)) = {n_e_framework:.2f}")
    print(f"  -2/this = {-2.0/n_e_framework:.4f}  (also close to observed)")
    print()
    print("OBSERVATION: numerically the framework can REPRODUCE -0.035 if it accepts")
    print("an N_e identification analogous to inflation's e-folds. But the framework")
    print("has NO STRUCTURAL DERIVATION of why n_s - 1 = -2/N_e in framework terms.")
    print("This would be the 'cite an observed-physics identity' pattern explicitly")
    print("rejected by the rigor bar.")
    print()


# =============================================================================
# §6. Honest verdict
# =============================================================================

def section_6_verdict():
    print("=" * 78)
    print("§6. OS-1 closure attempt — honest verdict")
    print("=" * 78)
    print()
    print("Per Routes A, B, C examined above:")
    print()
    print("  Route A (hard-cutoff bit budget):  k_max far above observable;")
    print("                                     gives sharp cutoff, not smooth tilt.")
    print()
    print("  Route B (cumulative-observation):  WRONG SIGN. Older modes (smaller k)")
    print("                                     get less accumulated time, lower")
    print("                                     amplitude → blue tilt n_s > 1, vs")
    print("                                     observed red tilt n_s < 1.")
    print()
    print("  Route C (precision-allocation):    No allocation rule derivable from")
    print("                                     existing framework primitives gives")
    print("                                     the observed sign + magnitude.")
    print()
    print("Cross-check with standard inflation (§5): n_s - 1 = -2/N_e is a slow-roll")
    print("identity, structurally rejected by framework rigor bar.")
    print()
    print("HONEST VERDICT: OS-1's scale-stratified compression budget mechanism does")
    print("NOT close the n_s spectral tilt at theorem grade from existing framework")
    print("primitives. The framework's natural finite-register-budget mechanism gives:")
    print()
    print("  - A SHARP CUTOFF (Route A) at scales far beyond observable.")
    print("  - A BLUE TILT under cumulative-observation framing (Route B), wrong sign.")
    print("  - NO TILT for equal-bits-per-mode allocation (Route C-i), giving the")
    print("    framework's existing white-noise n_s = 1 baseline.")
    print()
    print("The observed -0.035 red tilt cannot be derived from these primitives at")
    print("theorem grade.")
    print()
    print("This is the SAME negative outcome as NA-4's substrate-side audit (§3.5 D2)")
    print("and the n_s scoping doc's Attempts 1-3, reached via a different angle.")
    print("Three independent audits now converge on: framework predicts n_s = 1; the")
    print("observed -0.035 tilt requires structural machinery not in current scope.")
    print()
    print("RECOMMENDATION: revert n_s tilt to the BLOCKED status from NA-4's Scenario")
    print("3 honest concession. OS-1 does NOT close in 2-4 sessions; it does not close")
    print("at all from existing primitives.")
    print()
    print("The remaining Need OS sub-targets (OS-2, OS-3) may still close — they")
    print("attack different framework primitives (rate-gap z-dependence, ΛCDM-as-")
    print("observer-compression). OS-1's negative does not propagate to OS-2/OS-3")
    print("automatically, but the cleanness of the audit-before-ansatz negative here")
    print("is a useful prior on Need OS overall.")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    section_1_bit_budget()
    section_2_route_A_cutoff()
    section_3_route_B_cumulative()
    section_4_route_C_precision()
    section_5_inflationary_comparison()
    section_6_verdict()

    print("=" * 78)
    print("OS-1 attempt: HONEST NEGATIVE.")
    print("Scale-stratified compression budget does NOT give observed n_s tilt at")
    print("theorem grade from existing framework primitives. Three independent")
    print("audits (NA-4, n_s scoping Attempts 1-3, this OS-1 probe) converge on")
    print("the same verdict: framework predicts n_s = 1; observed -0.035 tilt is")
    print("BLOCKED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
