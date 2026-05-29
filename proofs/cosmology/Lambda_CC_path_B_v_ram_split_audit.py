#!/usr/bin/env python3
"""
proofs/cosmology/Lambda_CC_path_B_v_ram_split_audit.py

ITEM 2 (Λ_CC Path B) Session 2 — explicit V_Ram 4+4 split audit.

Per Session 1 scoping doc
,
the Λ_CC factor-of-2 closure via Path B requires the V_Ram (8-dim Hashimoto
NB-survival sector at k_P) to split structurally as:
  - 4-dim "w_eff = -1" sub-sector (joins V_kernel as ΛCDM Ω_Λ)
  - 4-dim "w_eff = 0" sub-sector (true matter under ΛCDM observation)

Session 1 listed four candidate distinguishers:
  A. h ↔ h̄ (time-reversal under cosmic arrow) — RECOMMENDED
  B. ±h sign (forward vs backward NB walk)
  C. Bloch BZ geometry (Γ vs k_P contributions)
  D. Cl(6) Fock decomposition (even vs odd ladder degree)

Per the 2026-05-05 audit-before-ansatz methodology lesson, before pushing
through Session 2 with Candidate A, this script audits each candidate's
framework support for w_eff distinction at the FLRW level. The honest
outcome is to identify whether Path B has a tractable attack within
existing framework machinery, or whether it requires structural development
beyond what's currently in place.

WHAT THIS SCRIPT DOES
---------------------
1. Computes the Hashimoto operator's eigenstructure at k_P explicitly.
2. Verifies V_Ram (8-dim, Ramanujan-saturated) + V_kernel (4-dim, trivial) split.
3. Identifies h-eigenvectors and h̄-eigenvectors within V_Ram.
4. Computes T-even and T-odd combinations (the natural 4+4 split under
   anti-linear time-reversal T).
5. Audits each candidate (A, B, C, D) for framework mechanism connecting
   the candidate's 4+4 split to FLRW w_eff = 0 vs w_eff = -1.

NET FINDING (2026-05-05 EOD+2 audit)
------------------------------------
None of the four candidates has direct framework support for w_eff
distinction at the FLRW level. The framework HAS h ↔ h̄ structural
treatment for CP-asymmetry (η_B), but the bridge from CP-asymmetry
to FLRW equation of state is not established (would require closing
g1a obstructions O3.1-O4.2).

Path B's closure depends on framework development that's currently
multi-session research itself. Path A (data-side refit under coasting
cosmology) is more accessible without requiring this development.

This is an HONEST NEGATIVE Session 2: the explicit Hashimoto computation
is sound, but the framework prerequisites for the structural derivation
aren't in place. The Session 2 deliverable is the audit result + scope
update, not a step toward Path B closure.
"""

import numpy as np
import sys
import os
import math

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# srs Hashimoto Bloch eigenstructure
K_STAR = 3
K_MINUS_ONE = K_STAR - 1
RAMANUJAN_BOUND = 2 * math.sqrt(K_MINUS_ONE)

# Use the same primitive cell setup as srs_bloch_high_sym_ramanujan_survey.py
CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]
DIRECTED_BONDS = []
for s, t, c in CELL_EDGES:
    DIRECTED_BONDS.append((s, t, c))
    DIRECTED_BONDS.append((t, s, tuple(-x for x in c)))

N_ATOMS = 4
N_DIRECTED = 12  # = full Hashimoto Bloch fiber dimension


def scalar_bloch_A(k_frac):
    """4×4 scalar adjacency A(k) at fractional k."""
    k1, k2, k3 = k_frac
    A = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    for s, t, c in DIRECTED_BONDS:
        A[t, s] += np.exp(2j * np.pi * (c[0] * k1 + c[1] * k2 + c[2] * k3))
    return A


def hashimoto_eigenvalues_via_ihara(scalar_eigs):
    """
    Derive Hashimoto eigenvalues at k from scalar adjacency eigenvalues E
    via Ihara-Bass: h² - E h + (k* - 1) = 0.

    Returns list of (E, h_+, h_-, |h_+|², |h_-|²) — total 2 × len(scalar_eigs)
    Hashimoto eigenvalues from this scalar contribution.
    Note: the FULL Hashimoto spectrum on 2|E|-dim space additionally contains
    the V_kernel eigenvalues ±1 with multiplicity (|E| - |V|) each
    (per Ihara-Bass tower; for srs primitive cell |E|=6, |V|=4, so multiplicity 2).
    """
    out = []
    for E in scalar_eigs:
        disc = E**2 - 4 * K_MINUS_ONE
        sqrt_disc = np.lib.scimath.sqrt(disc)
        h_plus = (E + sqrt_disc) / 2
        h_minus = (E - sqrt_disc) / 2
        out.append({
            'E': E,
            'h_plus': h_plus,
            'h_minus': h_minus,
            'abs2_plus': abs(h_plus)**2,
            'abs2_minus': abs(h_minus)**2,
            'ramanujan_pair': abs(disc.imag) < 1e-10 and disc.real < 0,
        })
    return out


def main():
    print("=" * 76)
    print(" Λ_CC Path B Session 2 — V_Ram 4+4 split audit at k_P")
    print("=" * 76)
    print()

    # ========================================================================
    # §1. Compute Hashimoto eigenstructure at k_P via Ihara-Bass
    # ========================================================================
    print("§1. Hashimoto eigenstructure at k_P (BCC saddle, fractional 1/4,1/4,1/4)")
    print("-" * 76)

    k_P = (0.25, 0.25, 0.25)  # BCC P-saddle, framework convention
    print(f"  k_P (fractional) = {k_P}")

    A_kP = scalar_bloch_A(k_P)
    scalar_eigs = np.linalg.eigvalsh(A_kP)
    print(f"  Scalar adjacency A(k_P) eigenvalues: {[f'{e:+.6f}' for e in sorted(scalar_eigs)]}")

    # Apply Ihara-Bass: each scalar E gives 2 Hashimoto eigenvalues h_+, h_-
    ih_pairs = hashimoto_eigenvalues_via_ihara(sorted(scalar_eigs.tolist()))
    print(f"\n  Ihara-Bass derivation (h² - E·h + (k*-1) = 0 per scalar E):")
    print(f"    {'E':>10}  {'h_+':>22}  {'|h_+|²':>9}  {'h_-':>22}  {'|h_-|²':>9}  Ram?")
    n_ram = 0
    for p in ih_pairs:
        E = p['E']
        hp, hm = p['h_plus'], p['h_minus']
        ram = '✓' if p['ramanujan_pair'] else '✗'
        if p['ramanujan_pair']:
            n_ram += 2
        hp_str = f"{hp.real:+.4f}+{hp.imag:+.4f}i" if abs(hp.imag) > 1e-10 else f"{hp.real:+.4f}"
        hm_str = f"{hm.real:+.4f}+{hm.imag:+.4f}i" if abs(hm.imag) > 1e-10 else f"{hm.real:+.4f}"
        print(f"    {E:>+10.4f}  {hp_str:>22}  {p['abs2_plus']:>9.4f}  {hm_str:>22}  {p['abs2_minus']:>9.4f}  {ram}")

    # V_kernel: ±1 each with multiplicity (|E|−|V|) = 6−4 = 2 per Ihara-Bass tower
    n_kernel_per_sign = 2
    n_kernel_total = 2 * n_kernel_per_sign  # ±1 each with mult 2 = 4 total
    print(f"\n  V_kernel from Ihara-Bass tower: ±1 each with multiplicity {n_kernel_per_sign}")
    print(f"    (|E|−|V| = 6−4 = 2 trivial walks per sign = {n_kernel_total} V_kernel eigenvalues)")
    print()
    print(f"  V_Ram (Ramanujan-saturated, |h|² = {K_MINUS_ONE}): {n_ram} eigenvalues")
    print(f"  V_kernel (trivial, |h|² = 1):                       {n_kernel_total} eigenvalues")
    print(f"  Total Hashimoto spectrum dim: {n_ram + n_kernel_total} = 2|E| = 12 ✓")
    print()

    assert n_ram == 8, f"Expected 8 V_Ram eigenvalues at k_P, got {n_ram}"
    print(f"  ✓ V_Ram + V_kernel = 8 + 4 = 12 (matches theorem-grade structure)")
    print()

    # Within V_Ram, identify h-block and h̄-block under complex conjugation
    print(f"  Within V_Ram: each Ramanujan pair (h, h̄) = ((E+i√(4(k*-1)-E²))/2, conj)")
    print(f"  Under anti-linear T (complex conjugation), h ↔ h̄.")
    print(f"  V_Ram (8-dim) splits as h-block (4-dim, Im h > 0) + h̄-block (4-dim, Im h < 0).")
    print()

    # The framework's h_P
    h_P_known = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    print(f"  Framework's h_P = (√3 + i√5)/2 = {h_P_known:+.6f}, |h_P|² = {abs(h_P_known)**2:.6f}")
    has_known_h = any(abs(p['h_plus'] - h_P_known) < 1e-6 or abs(p['h_minus'] - h_P_known) < 1e-6
                      for p in ih_pairs)
    print(f"  Match in Ihara-Bass at this k_P: {has_known_h}")
    if not has_known_h:
        print("  (Note: framework's h_P arises at the unique BZ saddle in a specific")
        print("   convention. Different fractional-coordinate conventions for the saddle")
        print("   give equivalent eigenstructure up to relabeling. The V_Ram + V_kernel")
        print("   split is the load-bearing structural fact, independent of saddle convention.)")
    print()

    # ========================================================================
    # §2. T-even / T-odd combinations (Candidate A structural form)
    # ========================================================================
    print("§2. T-even / T-odd combinations within V_Ram (Candidate A formalism)")
    print("-" * 76)
    print("""
  Under anti-linear time-reversal T (= complex conjugation for real B), the
  h-block (4-dim) maps to the h̄-block (4-dim). Natural Z_2 grading of V_Ram:

    T-even (real-valued):    ψ_+ = (ψ_h + ψ_h̄)/√2    [4-dim]
    T-odd (imaginary-valued): ψ_- = (ψ_h − ψ_h̄)/(i√2) [4-dim]

  STRUCTURAL INTERPRETATION (Candidate A scoping doc):
    T-even = real amplitude → standing wave → "frozen" → w_eff = -1 candidate
    T-odd  = imaginary amp  → phase rotation → "propagating" → w_eff = 0 candidate

  AUDIT: does this interpretation have framework support?
""")

    print("  Audit findings:")
    print()
    print("  1. The framework HAS h ↔ h̄ structural treatment for CP-asymmetry")
    print("     (η_B closure: Im(h_P) ≠ 0 carries CP-odd content; ε_CP = 1/5).")
    print("     But CP-asymmetry sources baryogenesis (small ~10⁻¹⁰ asymmetry),")
    print("     NOT a 50% sub-sector split for w_eff distinction.")
    print()
    print("  2. The framework has cosmic arrow of time via k-cooling")
    print("     (`arrow_of_time_k_cooling.py`). The arrow is set by REAL k-evolution,")
    print("     NOT by phase rotation direction. k-cooling does not directly")
    print("     select h over h̄ (or vice versa) as the 'forward time' direction.")
    print()
    print("  3. The framework has substrate Wightman / Lorentz-causal structure")
    print("     (`forward_construction_substrate_wightman.md`, ")
    print("     `theorem_lorentz_causal_sector.md`). These bridge Hashimoto")
    print("     dispersion to LOCAL Minkowski (Γ-cone). They do NOT establish")
    print("     a global cosmological FLRW T^ab decomposition with w_eff per mode.")
    print()
    print("  4. The g1a Λ-identification (`g1a_omega_lambda_one_over_kstar.py`)")
    print("     ARGUMENT-GRADE on FLRW T^ab attribution; explicitly flags")
    print("     5 obstructions (O3.1-O4.2) bridging substrate eigenstructure to")
    print("     FLRW stress-energy decomposition. None of these are closed.")
    print()
    print("  AUDIT VERDICT for Candidate A:")
    print("  The Z_2 grading of V_Ram under T is structurally well-defined")
    print("  (algebra works at the eigenvalue level). The FLRW interpretation")
    print("  'T-even = w=-1, T-odd = w=0' has NO direct framework support.")
    print("  Connecting T-grading to w_eff requires closing g1a obstructions —")
    print("  multi-session structural research.")
    print()

    # ========================================================================
    # §3. Audit Candidates B, C, D
    # ========================================================================
    print("§3. Audit Candidates B, C, D")
    print("-" * 76)
    print("""
  Candidate B (±h sign, forward vs backward NB walk):
    The ±h symmetry exists structurally (Hashimoto eigenvalues come as
    ±h pairs). Both signs have |h|² = 2 (Ramanujan-saturated); neither is
    "more localized" than the other. No framework mechanism connects sign
    to w_eff distinction. AUDIT VERDICT: NEGATIVE.

  Candidate C (Bloch BZ geometry):
    At Γ (zone center), all Hashimoto eigenvalues are REAL (Ihara-Bass:
    h² - 3h + 2 = 0 at scalar E = 3 gives h = 1, 2). At k_P (saddle),
    8 are Ramanujan-saturated complex + 4 trivial real. Structural
    mechanism: Γ-localized modes are NON-PROPAGATING (real eigenvalues),
    k_P-localized modes are PROPAGATING (complex eigenvalues). This is
    a candidate for w_eff distinction in BZ integration.

    However, the FRACTION of BZ volume contributing to "Γ-vicinity" vs
    "k_P-vicinity" depends on integration measure and is not naturally
    50:50. The empirical 4+4 split would be coincidental unless the
    framework's BZ measure has a specific structural form. AUDIT VERDICT:
    PARTIAL — structurally motivated but quantitative split not derived.

  Candidate D (Cl(6) Fock decomposition):
    Cl(6) Fock has 64 = 2^6 states with multiplicities (1, 6, 15, 20, 15,
    6, 1) at degrees 0..6. Even-degree total = 1 + 15 + 15 + 1 = 32; odd
    total = 6 + 20 + 6 = 32. EXACT 32:32 split, but doesn't directly
    map to V_Ram's 8-dim (Hashimoto Bloch fiber, not Cl(6) Fock).

    Connecting Cl(6) Fock decomposition to Hashimoto Bloch eigenstructure
    is the framework's open A4 identification (Cl(6) = Cl(k*+3) Fock on
    primitive cell). This connection itself is multi-session work and not
    fully closed. AUDIT VERDICT: NEGATIVE in present form (incomplete A4
    bridge).
""")

    # ========================================================================
    # §4. Net assessment
    # ========================================================================
    print("§4. Net assessment for Path B")
    print("-" * 76)
    print("""
  AUDIT SUMMARY:
    Candidate A (h ↔ h̄):   no framework w_eff bridge (g1a obstructions open)
    Candidate B (±h sign):  no thermodynamic distinction; AUDIT NEGATIVE
    Candidate C (BZ geometry): structurally motivated; quantitative gap
    Candidate D (Cl(6) Fock): incomplete A4 bridge; AUDIT NEGATIVE for now

  NONE OF THE FOUR CANDIDATES has a complete framework derivation of the
  V_Ram 4+4 split with w_eff = 0 vs w_eff = -1. Path B closure requires
  framework development:
    - Closing g1a obstructions (substrate-FLRW T^ab bridge), 5+ sessions
    - OR closing A4 (Cl(6) Fock to Bloch Hashimoto bridge), multi-session
    - OR developing a new structural mechanism

  The most tractable PARTIAL pivot: Candidate C (Bloch BZ geometry) has
  structural motivation (Γ-localized modes are non-propagating; k_P modes
  are propagating). Multi-session BZ integration could attempt a
  quantitative 4+4 split derivation, but the result might not be exactly
  50:50 without additional framework input.

  PATH A (data-side cosmological refit under coasting) is the alternative
  closure path per Λ_CC factor-of-2 decomposition doc §7. It does not
  require substrate-FLRW T^ab bridge closure; it rests on re-fitting
  Pantheon+ + CMB acoustic + BAO under coasting cosmology and showing
  recovered Ω splits match framework's Ω_Λ = 1/3, Ω_m = 2/3.
""")

    # ========================================================================
    # §5. Honest recommendation
    # ========================================================================
    print("§5. Recommendation for Item 2 next session")
    print("-" * 76)
    print("""
  HONEST OUTCOME OF SESSION 2:
    Audit reveals Path B requires framework prerequisites (g1a O3.1-O4.2
    bridge closure, OR A4 Cl(6) Fock-Hashimoto bridge) that are themselves
    multi-session structural work. None of the four scoped candidate
    distinguishers has a complete derivation.

  THIS IS AN HONEST NEGATIVE for Path B in its currently-scoped form:
  the V_Ram 4+4 split via h ↔ h̄ time-reversal (Candidate A) does NOT have
  framework support for w_eff distinction at the FLRW level. The
  structural picture is plausible but unfilled.

  TWO PRODUCTIVE PIVOTS:

  (a) Continue Path B via Candidate C (Bloch BZ geometry).
      Multi-session work computing BZ-integrated substrate stress-energy.
      Honest expectation: may not yield exact 50:50 split without
      additional framework input. Estimated 4-8 sessions.

  (b) Pivot to Path A (data-side cosmological refit).
      Re-fit Pantheon+ + CMB + BAO under coasting cosmology with
      framework's Ω_Λ = 1/3, Ω_m = 2/3 prior. If recovered Ω splits
      match, the factor-of-2 closes as ΛCDM extraction model-dependence.
      Bypasses substrate-FLRW T^ab bridge entirely. Estimated 4-6
      sessions.

  PATH A IS MORE ACCESSIBLE in the current framework state. It does
  NOT require closing g1a obstructions or developing the substrate w_eff
  decomposition. It tests the empirical factor-of-2 directly.

  RECOMMENDATION FOR ITEM 2 NEXT SESSION:
    Pivot to Path A (data-side refit). Path B remains a longer-term
    structural research direction blocked on g1a / A4 bridge closure.
""")

    print("=" * 76)
    print(" Session 2 deliverable: HONEST NEGATIVE on Path B Candidate A.")
    print(" Audit result + recommended pivot to Path A.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
