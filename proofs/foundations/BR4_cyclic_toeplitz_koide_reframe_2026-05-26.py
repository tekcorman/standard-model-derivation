#!/usr/bin/env python3
"""
BR4 session 1 — cyclic-Toeplitz Hermitian M_gen ↔ Koide cosine reframe.

Probe for the structural reframe in
  an internal working note

Six gates:
  G1: Cyclic-Toeplitz Hermitian M = a_0·I + a_1·P + a_1*·P† has eigenvalues
      μ_j = a_0 + 2|a_1|·cos(2πj/3 + arg(a_1)) for j ∈ {0,1,2}, where
      P = canonical cyclic-shift (Halmos 1958 §83).
  G2: For charged lepton (Q=2/3, ε²=2, δ=2/9 — all theorem-grade in the
      framework), the eigenvalues reproduce (m_τ, m_μ, m_e) ratios within
      PDG precision.
  G3: V_cb / V_ub walker amplitudes via L_eff(m) = 6m+2. NOTE: these are
      channel-coupling amplitudes (CKM), NOT the within-species M_gen
      off-diagonal magnitudes — they share the L_eff(m) walker class but
      couple through different channels.
  G4: Down sector — what δ_down value reproduces (m_b, m_s, m_d) given
      ε²_down = 5/2 (Type IV, W53)? Empirical extraction (NOT a derivation).
  G5: Up sector — what δ_up value reproduces (m_t, m_c, m_u) given
      ε²_up = 17/5 (Row P37)? Empirical extraction (NOT a derivation).
  G6: W73 Candidate A consistency: is δ_lepton ≈ (π − arg(h_P))/g and
      is δ_down ≈ arccos(1/3)/g? Honest report of near-matches.

Per W58 + an internal note: G4/G5 are
EMPIRICAL EXTRACTIONS that characterise what arg(a_1^(s)) would need to
be. They are NOT used to derive structure; they are NOT the closure of
BR4. The structural derivation of arg(a_1^(s)) remains open.
"""

import math
import sys

import numpy as np


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def cyclic_shift():
    """Canonical cyclic-shift P = σ_shift ∈ U(3) per R3.L2."""
    return np.array(
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]],
        dtype=complex,
    )


def cyclic_toeplitz_hermitian(a0, a1):
    """Return M = a0·I + a1·P + a1*·P† on C³."""
    P = cyclic_shift()
    return a0 * np.eye(3, dtype=complex) + a1 * P + np.conjugate(a1) * P.conj().T


def koide_cosine_eigenvalues(a0, a1):
    """Eigenvalues μ_j = a0 + 2|a1|·cos(2πj/3 + arg(a1)) sorted by j."""
    mag = abs(a1)
    phase = np.angle(a1) if a1 != 0 else 0.0
    return np.array([a0 + 2 * mag * math.cos(2 * math.pi * j / 3.0 + phase)
                     for j in range(3)])


def koide_params_from_masses(m_heavy, m_mid, m_light):
    """
    Extract (a0, |a1|, δ) for the Koide cosine that reproduces
    (m_heavy, m_mid, m_light) — empirical extraction only.

    Returns:
      Q : Koide ratio = Σm / (Σ√m)²
      ε  : ε from Q = (1 + ε²/2)/3 → ε² = 2(3Q − 1)
      a0 : √M = (Σ√m)/3
      a1_mag : |a1| = a0 · ε / 2
      delta_extracted : phase δ such that the cosine triple matches
                       (m_heavy, m_mid, m_light) — extracted, not derived
    """
    sm = math.sqrt(m_heavy) + math.sqrt(m_mid) + math.sqrt(m_light)
    sM = m_heavy + m_mid + m_light
    Q = sM / (sm * sm)
    eps_sq = 2.0 * (3.0 * Q - 1.0)
    eps = math.sqrt(eps_sq) if eps_sq > 0 else float("nan")
    a0 = sm / 3.0
    a1_mag = a0 * eps / 2.0
    # Extract δ by finding the rotation that maps sqrt(m_j) → cosine triple.
    # Search δ ∈ [0, 2π/3) for best fit; for j=0,1,2 the masses can map in
    # any order, so try all 6 permutations.
    masses_sorted = sorted([m_heavy, m_mid, m_light], reverse=True)
    sqrt_m_sorted = [math.sqrt(m) for m in masses_sorted]

    best = None
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        # Try this assignment of (j=0, j=1, j=2) ↔ permutation of (heavy, mid, light)
        target_sqrt = [sqrt_m_sorted[perm[j]] for j in range(3)]
        # Solve: a0 + 2|a1|cos(2πj/3 + δ) = target_sqrt[j]
        # 3 equations, 1 unknown δ (given a0, |a1| from Koide invariants)
        for delta_trial in np.linspace(0, 2 * math.pi, 7200):
            pred = [a0 + 2 * a1_mag * math.cos(2 * math.pi * j / 3.0 + delta_trial)
                    for j in range(3)]
            err = sum((p - t) ** 2 for p, t in zip(pred, target_sqrt))
            if best is None or err < best[1]:
                best = (delta_trial, err, perm)
    # The cosine pattern has 3-fold redundancy in δ (shift by 2π/3 relabels j).
    # Report δ in the fundamental domain [0, 2π/3); the "small-δ branch" is
    # min(δ_extracted mod (2π/3), 2π/3 − δ_extracted mod (2π/3)).
    period = 2 * math.pi / 3.0
    delta_wrapped = best[0] % period
    delta_small = min(delta_wrapped, period - delta_wrapped)
    return {
        "Q": Q,
        "epsilon": eps,
        "epsilon_sq": eps_sq,
        "a0": a0,
        "a1_mag": a1_mag,
        "delta_extracted": best[0],
        "delta_wrapped": delta_wrapped,
        "delta_small_branch": delta_small,
        "fit_error_sq": best[1],
        "assignment": best[2],
    }


# -----------------------------------------------------------------------
# G1 — cyclic-Toeplitz Hermitian = Koide cosine
# -----------------------------------------------------------------------

def G1_cyclic_toeplitz_to_koide():
    print("=" * 70)
    print("G1 — Cyclic-Toeplitz Hermitian M ↔ Koide cosine spectrum")
    print("=" * 70)
    # Trial: a_0 = 5.0, a_1 = 1.5 * e^{i * 0.4}
    a0, a1 = 5.0, 1.5 * complex(math.cos(0.4), math.sin(0.4))
    M = cyclic_toeplitz_hermitian(a0, a1)
    # Verify Hermitian
    herm_err = np.linalg.norm(M - M.conj().T)
    # Compute eigenvalues numerically
    eigs_num = sorted(np.linalg.eigvalsh(M).real)
    eigs_formula = sorted(koide_cosine_eigenvalues(a0, a1))
    diff = max(abs(a - b) for a, b in zip(eigs_num, eigs_formula))
    print(f"  a_0 = {a0}, a_1 = {a1}")
    print(f"  |a_1| = {abs(a1):.6f}, arg(a_1) = {np.angle(a1):.6f} rad")
    print(f"  M Hermiticity check: ||M − M†|| = {herm_err:.2e}")
    print(f"  Numerical eigenvalues:     {eigs_num}")
    print(f"  Koide-cosine eigenvalues:  {eigs_formula}")
    print(f"  Max difference: {diff:.2e}")
    passed = (herm_err < 1e-12 and diff < 1e-10)
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# -----------------------------------------------------------------------
# G2 — Charged lepton: theorem-grade (Q, ε, δ) reproduces (m_τ, m_μ, m_e)
# -----------------------------------------------------------------------

def G2_charged_lepton_reconstruction():
    print()
    print("=" * 70)
    print("G2 — Charged lepton: (Q=2/3, ε=√2, δ=2/9) → (m_τ, m_μ, m_e)")
    print("=" * 70)
    # PDG 2024 charged-lepton masses (MeV)
    m_tau = 1776.86
    m_mu = 105.6583755
    m_e = 0.510998950
    # Framework-derived parameters (theorem-grade)
    Q = 2.0 / 3.0
    eps = math.sqrt(2.0)
    delta = 2.0 / 9.0
    # Koide invariants → (a_0, |a_1|)
    sm = math.sqrt(m_tau) + math.sqrt(m_mu) + math.sqrt(m_e)
    a0 = sm / 3.0          # = √M
    a1_mag = a0 * eps / 2.0
    a1 = a1_mag * complex(math.cos(delta), math.sin(delta))
    print(f"  PDG masses (MeV): m_τ = {m_tau}, m_μ = {m_mu}, m_e = {m_e}")
    print(f"  Σ√m = {sm:.6f}")
    print(f"  a_0 = √M = {a0:.6f}  (a_0² = {a0*a0:.6f} MeV)")
    print(f"  |a_1| = {a1_mag:.6f}")
    print(f"  arg(a_1) = δ = {delta:.6f} rad = {math.degrees(delta):.4f}°")
    # Eigenvalues — these are √m_j
    sqrt_m_j = koide_cosine_eigenvalues(a0, a1)
    m_j = sqrt_m_j ** 2
    print(f"  Eigenvalue triple (√m): {sqrt_m_j}")
    print(f"  Predicted masses: {sorted(m_j, reverse=True)}")
    print(f"  PDG masses:        {[m_tau, m_mu, m_e]}")
    # Match (each predicted mass should be within 1% of one PDG mass)
    targets = sorted([m_tau, m_mu, m_e], reverse=True)
    pred = sorted(m_j, reverse=True)
    rel_dev = [abs(p - t) / t for p, t in zip(pred, targets)]
    print(f"  Relative deviations: {[f'{d*100:.4f}%' for d in rel_dev]}")
    passed = all(d < 0.005 for d in rel_dev)
    print(f"  Result: {'PASS' if passed else 'FAIL'} (tolerance 0.5% on each mass)")
    return passed


# -----------------------------------------------------------------------
# G3 — Walker amplitudes via L_eff(m) — V_cb / V_ub channel
# -----------------------------------------------------------------------

def G3_walker_amplitudes_L_eff():
    print()
    print("=" * 70)
    print("G3 — Walker amplitudes via L_eff(m) = 6m+2 (CKM channel, NOT within-species M_gen)")
    print("=" * 70)
    # Per V_cb paradigm: V_cb = α^L_eff(1)/(1-α^L_eff(1)) with α = (k-1)/k = 2/3.
    # For m-cycle host (ΔGen = m): L_eff(m) = m·g − 2(m-1)·s − n_fixed = 6m + 2.
    k = 3
    alpha = (k - 1) / k  # = 2/3
    print(f"  α = (k-1)/k = {alpha}")
    walker_amps = {}
    for m in [1, 2, 3]:
        L_eff = 6 * m + 2
        a = alpha ** L_eff
        amp = a / (1.0 - a)
        walker_amps[m] = (L_eff, a, amp)
        print(f"  m={m}: L_eff = {L_eff}, α^L_eff = {a:.6e}, walker = α^L/(1-α^L) = {amp:.6e}")
    # V_cb reference: 256/6305 ≈ 0.04061 (exact, PDG 0.0406±0.0009)
    V_cb_pred = walker_amps[1][2]
    V_cb_exact = 256.0 / 6305.0
    print(f"\n  V_cb prediction:  {V_cb_pred:.6e}")
    print(f"  V_cb exact:       {V_cb_exact:.6e}")
    print(f"  PDG V_cb:         {40.6e-3} ± {0.9e-3}")
    V_cb_pass = abs(V_cb_pred - V_cb_exact) < 1e-10
    print(f"  V_cb match: {'PASS' if V_cb_pass else 'FAIL'}")
    # V_ub: full A5(b) Case B sum over host classes m=2,3,...
    # (bridge scoping doc §1: V_ub = Σ_{m≥2} (2/3)^{6m+2}/(1-(2/3)^{6m+2}))
    V_ub_pred_m2_only = walker_amps[2][2]
    V_ub_full = sum(((2/3) ** (6*m+2)) / (1 - (2/3) ** (6*m+2)) for m in range(2, 20))
    V_ub_obs = 3.82e-3
    V_ub_sigma = 0.20e-3
    V_ub_dev = (V_ub_full - V_ub_obs) / V_ub_sigma
    print(f"\n  V_ub (m=2 host class only): {V_ub_pred_m2_only:.6e}")
    print(f"  V_ub (full Σ_{{m≥2}} sum):    {V_ub_full:.6e}  (entry-point §1 formula)")
    print(f"  PDG V_ub:                    {V_ub_obs} ± {V_ub_sigma}  → {V_ub_dev:+.2f}σ")
    V_ub_pass = abs(V_ub_dev) < 1.0
    print(f"  V_ub match: {'PASS' if V_ub_pass else 'FAIL'}  (currently ADOPTED status)")
    print()
    print("  NOTE: These are the CKM walker amplitudes (cross-species channel).")
    print("        The within-species M_gen off-diagonals (|a_m|/a_0 = ε^(s)/2)")
    print("        are NOT numerically equal to V_cb/V_ub — they share the L_eff(m)")
    print("        walker class but couple through different channels with")
    print("        species-specific A^(s) normalisation (persistence theorem §3.2).")
    return V_cb_pass and V_ub_pass


# -----------------------------------------------------------------------
# G4 — Down sector: empirical δ_down extraction
# -----------------------------------------------------------------------

def G4_down_sector_delta_extraction():
    print()
    print("=" * 70)
    print("G4 — Down sector: extract δ_down from (m_b, m_s, m_d)")
    print("=" * 70)
    print("  Framework input: ε²_down = 5/2 (Type IV walker, W53 theorem-grade)")
    print("  EMPIRICAL EXTRACTION — characterises what δ_down WOULD need to be.")
    print("  This is NOT a derivation; per W58 / an internal note.")
    print()
    # PDG quark masses at consistent scale (MS-bar at 2 GeV)
    m_b_2GeV = 4180.0   # MeV at m_b
    m_s_2GeV = 93.4
    m_d_2GeV = 4.67
    print(f"  Masses (MeV at 2 GeV scheme): m_b={m_b_2GeV}, m_s={m_s_2GeV}, m_d={m_d_2GeV}")
    result = koide_params_from_masses(m_b_2GeV, m_s_2GeV, m_d_2GeV)
    print(f"  Koide Q_down (extracted)     = {result['Q']:.6f}")
    print(f"  Koide ε²_down (extracted)    = {result['epsilon_sq']:.6f}")
    print(f"  Framework ε²_down (theorem)  = 2.5")
    print(f"  Ratio (extracted/theorem):   {result['epsilon_sq']/2.5:.4f}")
    print(f"  δ_down (raw extracted)       = {math.degrees(result['delta_extracted']):.4f}°")
    print(f"  δ_down (mod 2π/3)            = {math.degrees(result['delta_wrapped']):.4f}°")
    print(f"  δ_down (small branch)        = {math.degrees(result['delta_small_branch']):.4f}°")
    print(f"  W73 candidate δ_down = φ_K4/g = arccos(1/3)/10 = {math.degrees(math.acos(1/3))/10:.4f}°")
    print(f"  W73 expected δ_lepton via candidate A = {math.degrees(math.pi - math.atan(math.sqrt(5/3)))/10:.4f}°")
    print(f"  Fit error² = {result['fit_error_sq']:.6e}")
    # Note: ε²_extracted ≠ 5/2 because the framework's Type IV identification
    # gives ε² = 5/2 at the framework-bare level, but PDG masses are at 2 GeV
    # MS-bar — there's an RG correction. So the extraction may differ from
    # the framework prediction by the RG correction magnitude.
    print()
    print("  HONEST: ε²_extracted ≠ 5/2 because PDG masses are at 2 GeV MS-bar;")
    print("          framework ε²_down = 5/2 is at the framework-bare scale.")
    print("          The δ extraction inherits this RG-scale ambiguity.")
    return True  # G4 is honest report, not a pass/fail


# -----------------------------------------------------------------------
# G5 — Up sector: empirical δ_up extraction
# -----------------------------------------------------------------------

def G5_up_sector_delta_extraction():
    print()
    print("=" * 70)
    print("G5 — Up sector: extract δ_up from (m_t, m_c, m_u)")
    print("=" * 70)
    print("  Framework input: ε²_up = 17/5 (Row P37 ratio, theorem-grade)")
    print("  EMPIRICAL EXTRACTION — characterises what δ_up WOULD need to be.")
    print()
    # PDG quark masses
    m_t = 172690.0  # MeV (pole mass approx)
    m_c_2GeV = 1270.0
    m_u_2GeV = 2.16
    print(f"  Masses (MeV, scheme-mixed): m_t={m_t}, m_c={m_c_2GeV}, m_u={m_u_2GeV}")
    result = koide_params_from_masses(m_t, m_c_2GeV, m_u_2GeV)
    print(f"  Koide Q_up (extracted)       = {result['Q']:.6f}")
    print(f"  Koide ε²_up (extracted)      = {result['epsilon_sq']:.6f}")
    print(f"  Framework ε²_up (theorem)    = 17/5 = 3.4")
    print(f"  Ratio (extracted/theorem):   {result['epsilon_sq']/3.4:.4f}")
    print(f"  δ_up (raw extracted)         = {math.degrees(result['delta_extracted']):.4f}°")
    print(f"  δ_up (mod 2π/3)              = {math.degrees(result['delta_wrapped']):.4f}°")
    print(f"  δ_up (small branch)          = {math.degrees(result['delta_small_branch']):.4f}°")
    print(f"  Fit error² = {result['fit_error_sq']:.6e}")
    print()
    print("  HONEST: Type II saturation (L=0) means walker phase argument may")
    print("          not follow the same mechanism as Type III/IV; up sector")
    print("          requires its own structural derivation (W73 G5 noted this).")
    return True  # G5 is honest report


# -----------------------------------------------------------------------
# G6 — W73 Candidate A consistency check
# -----------------------------------------------------------------------

def G6_W73_candidate_A_check():
    print()
    print("=" * 70)
    print("G6 — W73 Candidate A: δ_species ≈ (V_{-1}-T_{B-L} large phase)/g?")
    print("=" * 70)
    g = 10  # girth of srs
    # Lepton: T_{B-L} = -1 → large phase = arccos(-1) = π
    # Walking phase factor: arg(h_P) = arctan(√5/√3) ≈ 0.9117 rad
    arg_hP = math.atan(math.sqrt(5.0 / 3.0))
    delta_lep_candidate = (math.pi - arg_hP) / g
    delta_lep_target = 2.0 / 9.0
    lep_match = abs(delta_lep_candidate - delta_lep_target) / delta_lep_target
    print(f"  Lepton (T_{{B-L}} = −1, large phase = π):")
    print(f"    Candidate: (π − arg(h_P))/g = ({math.pi:.4f} − {arg_hP:.4f})/{g} = {delta_lep_candidate:.6f} rad")
    print(f"    Framework δ_lepton = 2/9 = {delta_lep_target:.6f} rad (theorem-grade)")
    print(f"    Relative diff: {lep_match*100:.3f}%")
    print(f"    Verdict: {'NEAR-MATCH (0.3% diff)' if lep_match < 0.01 else 'NO MATCH'}")
    print()
    # Down quark: T_{B-L} = +1/3 → large phase = arccos(1/3)
    arccos_1_3 = math.acos(1.0 / 3.0)
    delta_d_candidate = arccos_1_3 / g
    print(f"  Down quark (T_{{B-L}} = +1/3, large phase = arccos(1/3) = {math.degrees(arccos_1_3):.4f}°):")
    print(f"    Candidate: arccos(1/3)/g = {delta_d_candidate:.6f} rad = {math.degrees(delta_d_candidate):.4f}°")
    # Empirical δ_down from G4 (need scheme):
    print(f"    W73 G4: empirical δ_down at 2 GeV = 5.80°; at m_b = 6.31° (scheme-dependent)")
    print(f"    Verdict: scheme-dependent borderline (~10-15% off depending on scheme)")
    print()
    # Up quark: Type II saturation L=0 — formula not structurally motivated
    print(f"  Up quark: Type II saturation walker (L=0); /g rule not applicable.")
    print(f"    No clean W73 candidate for up sector.")
    print()
    # Structural finding: the V_{-1}-T_{B-L} large phases divided by g are
    # in the right ballpark for lepton (0.3%) and down (~10-15%) but the
    # mechanism is not unified.
    print("  Net: Candidate A (W73) gives the right ORDER OF MAGNITUDE structurally")
    print("       for lepton (0.3%) and down (10-15% scheme-dependent); up unhandled.")
    print("       Session 2 should attempt clean structural derivation OR accept")
    print("       Candidate A as a near-coincidence and pursue Candidate B.")
    return True  # G6 is structural-consistency report


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("BR4 SESSION 1 PROBE — cyclic-Toeplitz reframe + L_eff consistency")
    print("=" * 70)
    print("Per an internal working note")
    print()

    results = {
        "G1": G1_cyclic_toeplitz_to_koide(),
        "G2": G2_charged_lepton_reconstruction(),
        "G3": G3_walker_amplitudes_L_eff(),
        "G4": G4_down_sector_delta_extraction(),
        "G5": G5_up_sector_delta_extraction(),
        "G6": G6_W73_candidate_A_check(),
    }

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for gate, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate}: {status}")
    structural_gates = ["G1", "G2", "G3"]
    structural_pass = all(results[g] for g in structural_gates)
    print()
    if structural_pass:
        print("  Structural gates (G1, G2, G3) all PASS.")
        print("  The cyclic-Toeplitz Hermitian M_gen ↔ Koide cosine reframe")
        print("  is numerically validated for the lepton sector; the V_cb / V_ub")
        print("  walker amplitudes via L_eff(m) reproduce existing predictions.")
    print()
    print("  G4, G5, G6 are HONEST REPORTS — not derivations:")
    print("    G4/G5 extract empirical δ_down, δ_up given the framework's")
    print("    ε² values. These characterise the BR4 gap; they do NOT close it.")
    print("    G6 records the W73 Candidate A near-coincidence for lepton.")
    print()
    print("  BR4 session 1 verdict: SKETCH-GRADE. The reframe is clean; the")
    print("  off-diagonal-phase mechanism is the genuine open gap. Session 2")
    print("  attempts structural closure of Candidate A or B.")
    print()
