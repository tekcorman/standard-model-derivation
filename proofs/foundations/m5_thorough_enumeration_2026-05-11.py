"""
proofs/foundations/m5_thorough_enumeration_2026-05-11.py

M5 enumeration REDO (Task #8) — thorough search anchored on KNOWN unclosed
framework gaps (residue register R-N entries + master plan Priority 3/4).

This supersedes `m5_candidate_enumeration_2026-05-11.py` (Task #6), which was
correctly criticized as a textbook Lie-algebra list rather than the
framework's actual specific content.

Methodology:
1. List every known OPEN residue from `structural_residue_register.md`.
2. List every Priority 3/4 unclosed parameter from `master_plan.md`.
3. For each, identify the structural shape (E, L_M, max_L_r, dim) where
   the framework's existing scoping doc names it.
4. Compute W(M, N_hub) for each.
5. Tabulate WITHOUT premature verdict — flag pattern-matches to known
   unclosed observable values where they appear.

WHAT THIS DOES NOT DO:
- Compute β-function loop contributions (separate calculation).
- Verify that each candidate's W signature uniquely determines an
  observable (would need structural derivation).
- Declare MSSM-derivation verdict — that requires the loop calculation
  + matching exercise as a separate piece.

WHAT THIS DOES:
- Make the actual candidate set EXPLICIT (vs. my prior textbook list).
- Confirm retention at N_hub for each.
- Flag structural patterns the user can interpret.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# ============================================================================
# Formula primitives (validated in Task #7 sanity check)
# ============================================================================

def L_elias(m):
    if m == float('inf') or m <= 1:
        return 1.0
    return 1 + 2 * math.floor(math.log2(m))


def F_inv_log_count(E, N):
    if N == 0 or E == 0:
        return 0.0
    if E == 1:
        return 1.0 if N >= 1 else 0.0
    if E == 2:
        return math.log2(2 * N + 1) if N > 0 else 0.0
    return N * math.log2(E - 1) + math.log2(E / (E - 2))


def Phi_compression(E, dim, N):
    f_log = F_inv_log_count(E, N)
    w_log = math.log2(dim) if dim > 0 else 0.0
    return max(0.0, f_log - min(f_log, w_log))


def freq_factor(E, max_L_r, N):
    if N <= 0 or E <= 0:
        return float('-inf')
    return math.log2(N) - max_L_r * math.log2(E)


def W_combined(Phi, L, freq):
    return Phi - L + min(freq, 0.0)


# ============================================================================
# Candidate catalog — every known open framework gap
# ============================================================================
#
# Each entry:
#   source: which register/master-plan entry it came from
#   name: short description
#   structural_shape: (E, L_M, max_L_r, dim) where derivable
#   would_close: which observable/parameter this candidate would close if M5
#                produced it
#   open_question: what's not yet known about it
#   matches_signature_of: numerical pattern the candidate's structure could
#                         produce that would match a known unclosed value
#
# Where the structural shape is NOT clearly derivable from existing scoping,
# we note "scoping doc names this but doesn't give explicit (E, L_M, ...);
# parameters listed are estimates from analogous structures."

candidates = [
    # ============================================================
    # FROM RESIDUE REGISTER — OPEN entries
    # ============================================================
    {
        'source': 'R-9',
        'name': 'srs-z chiral-net sub-leading retention (2.56 bit gap)',
        'E': 3, 'L_M': L_elias(3) + L_elias(3) + 4, 'max_L_r': 6, 'dim': 24,
        'would_close': 'V_us at sub-3σ structural-DL margin (load-bearing for all framework predictions)',
        'open_question': '2.56-bit structural-DL discrimination between srs and srs-z; three closure directions in srs_vs_srs_z_dl_audit.py',
        'matches_signature_of': 'V_us = 9/40 numerical pattern; structurally chiral-net retention',
    },
    {
        'source': 'R-13',
        'name': 'hyperbolic Cayley quotient class (out-of-scope Bloch-decomposable)',
        'E': 3, 'L_M': L_elias(3) + 8, 'max_L_r': 10, 'dim': 32,  # representative hyperbolic Coxeter
        'would_close': 'substrate scope question; affects all framework predictions if relevant',
        'open_question': 'numerically bounded ≤ 2.7e-10 contribution to C4 (dark/cosmological) observables; below sensitivity but OPEN',
        'matches_signature_of': 'no specific observable; bounded',
    },
    {
        'source': 'R-14',
        'name': 'PS quark/lepton sector-dependent formula',
        'E': 6, 'L_M': L_elias(6) + L_elias(4) + 4, 'max_L_r': 4, 'dim': 4,  # PS quartet acting on Cl(6)
        'would_close': 'δ_CP_PMNS (P34), δ_CP_CKM (P15), quark masses (P39 cluster), sector-differentiation residue cluster',
        'open_question': 'W-vertex 4-walk Jarlskog phase derivation on K_4 (cos = T_BL eigenvalue identity needs geometric derivation)',
        'matches_signature_of': 'δ_CP_PMNS = arccos(-1) = 180° at 0.15σ; δ_CP_CKM = arccos(1/3) = 70.529°',
    },
    {
        'source': 'R-15',
        'name': 'm_ν1 = 0 derivation (C³_gen mass operator)',
        'E': 3, 'L_M': L_elias(3) + 2, 'max_L_r': 3, 'dim': 3,
        'would_close': 'm_ν1 = 0 as derivation rather than convention (Sprint 11 B7.3a.v)',
        'open_question': 'C³_gen mass operator structure (need_D3 territory; closely tied to Need-A2 closure)',
        'matches_signature_of': 'lightest neutrino mass = 0 (one specific eigenvalue of C³_gen mass op)',
    },
    {
        'source': 'R-16',
        'name': 'N_hub absolute value (5th dimensionless relation)',
        'E': 2, 'L_M': L_elias(2) + 3, 'max_L_r': 4, 'dim': 1,
        'would_close': 'N ≈ 10^61 anchor → reduce from 2 anchors to 1 across the framework',
        'open_question': 'no closure path identified; "deep research question, multi-session conceptual work"',
        'matches_signature_of': 'N_hub ≈ 8.4×10^60 from R∞ + G_N anchor pair',
    },

    # ============================================================
    # FROM MASTER PLAN — Priority 3 (research-level)
    # ============================================================
    {
        'source': 'MP-3.5',
        'name': 'continuum-limit existence proof (Stage 3 premise)',
        'E': 3, 'L_M': L_elias(3) + 5, 'max_L_r': 6, 'dim': 24,
        'would_close': 'removes Stage 3 premise; tightens Lorentz arc',
        'open_question': 'Sunada + causal-set theory; research-level',
        'matches_signature_of': '4D smooth continuum from srs lattice',
    },

    # ============================================================
    # FROM MASTER PLAN — Priority 4 (not started)
    # ============================================================
    {
        'source': 'MP-4.1 / SUSY spectrum',
        'name': 'gluino mass (sub-leading SU(3) sector mass eigenmode)',
        'E': 8, 'L_M': L_elias(8) + L_elias(3) + 4, 'max_L_r': 4, 'dim': 8,
        'would_close': 'm_gluino (one of 8 SUSY spectrum rows; MSSM-partner directly)',
        'open_question': 'specific mass scale; MSSM benchmark ~ 1-3 TeV',
        'matches_signature_of': 'spin-1/2 fermion in color-octet adjoint at ~TeV scale',
    },
    {
        'source': 'MP-4.1 / SUSY spectrum',
        'name': 'squark mass (sub-leading V_Ram eigenmode in quark sector)',
        'E': 6, 'L_M': L_elias(6) + L_elias(3) + 4, 'max_L_r': 4, 'dim': 8,
        'would_close': 'm_squark (MSSM-partner: spin-0 boson with quark color + Y)',
        'open_question': 'spin/statistics must flip from quark partner; framework has no current mechanism',
        'matches_signature_of': 'spin-0 boson in (3, 2, 1/6) or (3̄, 1, -2/3) rep at ~TeV',
    },
    {
        'source': 'MP-4.1 / SUSY spectrum',
        'name': 'slepton mass (sub-leading V_Ram in lepton sector)',
        'E': 6, 'L_M': L_elias(6) + L_elias(1) + 4, 'max_L_r': 4, 'dim': 2,
        'would_close': 'm_slepton (MSSM-partner)',
        'open_question': 'same as squark — statistics flip not in framework',
        'matches_signature_of': 'spin-0 boson in (1, 2, -1/2) or (1, 1, -1) rep at ~TeV',
    },
    {
        'source': 'MP-4.1 / SUSY spectrum',
        'name': 'neutralino mass (fermion partner of neutral bosons)',
        'E': 4, 'L_M': L_elias(4) + L_elias(3) + 4, 'max_L_r': 4, 'dim': 4,
        'would_close': 'm_neutralino (4 mass eigenstates from gaugino/higgsino mixing)',
        'open_question': 'requires fermion partner to bosons',
        'matches_signature_of': 'spin-1/2 fermion mixing bino/wino/higgsino content at ~100s of GeV',
    },
    {
        'source': 'MP-4.1 / SUSY spectrum',
        'name': 'chargino mass (charged fermion partners of W and Higgs)',
        'E': 4, 'L_M': L_elias(4) + 4, 'max_L_r': 4, 'dim': 2,
        'would_close': 'm_chargino (2 charged mass eigenstates)',
        'open_question': 'same family as neutralino',
        'matches_signature_of': 'charged spin-1/2 fermions at ~100s GeV',
    },
    {
        'source': 'MP-4.1 / SUSY spectrum',
        'name': 'tan β (Higgs VEV ratio in 2-Higgs sector)',
        'E': 2, 'L_M': L_elias(2) + 3, 'max_L_r': 4, 'dim': 2,
        'would_close': 'tan β (MSSM benchmark ~10)',
        'open_question': 'requires 2-Higgs sector beyond framework\'s current single doublet',
        'matches_signature_of': 'dimensionless ratio O(10)',
    },
    {
        'source': 'MP-4.1 / SUSY spectrum',
        'name': 'm_H, m_A, m_H± (extended Higgs sector — 3 additional masses)',
        'E': 4, 'L_M': L_elias(4) + 4, 'max_L_r': 4, 'dim': 4,
        'would_close': 'MSSM extended Higgs sector (3 additional Higgs masses)',
        'open_question': 'requires 2-Higgs sector',
        'matches_signature_of': '3 additional Higgs masses at TeV scale',
    },
    {
        'source': 'MP-4.2',
        'name': 'θ_13_PMNS PS embedding step (1-2 sessions)',
        'E': 6, 'L_M': L_elias(6) + L_elias(2) + 3, 'max_L_r': 4, 'dim': 4,
        'would_close': 'θ_13_PMNS theorem grade (currently sub-class part conditional)',
        'open_question': 'PS embedding step for dark correction',
        'matches_signature_of': 'PMNS θ_13 ≈ 8.57°',
    },
    {
        'source': 'MP-4.3',
        'name': 'θ_12_PMNS dark corrections (1-2 sessions)',
        'E': 3, 'L_M': L_elias(3) + L_elias(2) + 3, 'max_L_r': 4, 'dim': 3,
        'would_close': 'θ_12_PMNS theorem grade',
        'open_question': 'apply dark corrections to TBM θ_12 = 35.26°',
        'matches_signature_of': 'PMNS θ_12 ≈ 33°',
    },
    {
        'source': 'MP-4.4 / Step 2.1',
        'name': 'Feshbach analog on λ_Higgs (m_H residual)',
        'E': 2, 'L_M': L_elias(2) + L_elias(4) + 3, 'max_L_r': 4, 'dim': 4,
        'would_close': 'm_H 3.43σ_PDG residual; bridge convention for tree-level couplings',
        'open_question': 'graph-QFT loop machinery for substrate λ_Higgs Feshbach analog; 3 naive paths falsified session 25',
        'matches_signature_of': 'universal QFT 1/(16π²) prefactor (matches λ_obs/λ_tree to 0.033%)',
    },
    {
        'source': 'MP-4.4 / Step 2.2',
        'name': 'Feshbach analog on y_τ (sub-leading lepton sector)',
        'E': 8, 'L_M': L_elias(8) + L_elias(2) + 4, 'max_L_r': 4, 'dim': 16,
        'would_close': 'y_τ +0.13% residual via fermion-Higgs vertex analog',
        'open_question': 'not yet investigated under bridge convention',
        'matches_signature_of': 'small sub-leading correction ~0.13%',
    },

    # ============================================================
    # FROM MASTER PLAN — Open frontier
    # ============================================================
    {
        'source': 'frontier / Λ_CC residual',
        'name': 'z_eff cosmological observation scale (under (γ) framing)',
        'E': 2, 'L_M': L_elias(2) + 3, 'max_L_r': 4, 'dim': 4,
        'would_close': 'Λ_CC factor-of-2 residual (and Ω partition absolute); via z_eff ≈ 1.92',
        'open_question': 'data-side derivation of effective observation redshift',
        'matches_signature_of': 'z_eff ≈ 1.92 from bias function 𝓑(z) = (Ω_m, Ω_Λ)',
    },
    {
        'source': 'frontier / quark Yukawa',
        'name': 'quark Yukawa hierarchy (Row P39 family)',
        'E': 6, 'L_M': L_elias(6) + L_elias(6) + 4, 'max_L_r': 4, 'dim': 8,
        'would_close': 'm_u, m_d, m_s, m_c, m_b, m_t MSSM threshold absolute values',
        'open_question': 'Pati-Salam quark/lepton differentiation residue (same blocker as R-14)',
        'matches_signature_of': 'specific mass hierarchy spanning 14 orders of magnitude',
    },
    {
        'source': 'frontier / β cosmic birefringence',
        'name': 'β cosmic birefringence Pathway-4 unit phasor',
        'E': 6, 'L_M': L_elias(6) + L_elias(4) + 4, 'max_L_r': 4, 'dim': 8,
        'would_close': 'β observable (ADOPTED-DARK-MAP narrow scope)',
        'open_question': 'theorem_cosmic_birefringence β.A1 (Pathway-4) or β.A2 (Pathway-2)',
        'matches_signature_of': 'β ≈ 0.30° from CMB polarization rotation',
    },
    {
        'source': 'frontier / Majorana phases',
        'name': 'α_21, α_31 Majorana phases via R-9 + mass-ordering',
        'E': 3, 'L_M': L_elias(3) + L_elias(2) + 3, 'max_L_r': 4, 'dim': 3,
        'would_close': 'α_21_PMNS, α_31_PMNS (Priority 2.2 ~1-2 sessions tractable)',
        'open_question': 'arg(h^n) selection for n; minimal-girth-loop holonomy + analytical Feshbach',
        'matches_signature_of': 'PMNS Majorana phases ≈ specific arg(h^n) values',
    },

    # ============================================================
    # FROM SUSY SCOPING — paths attempted but no closure
    # ============================================================
    {
        'source': 'theorem_susy_requirement_scoping',
        'name': 'Witten SUSY pair (χ̃ bipartite SUSY-pair on srs-z, B1+D5 NEGATIVE)',
        'E': 6, 'L_M': L_elias(6) + 4, 'max_L_r': 4, 'dim': 8,
        'would_close': 'partial MSSM matter via χ̃ pairing',
        'open_question': 'observables are χ̃-invariant (no observable consequence); Path E uplift blocked',
        'matches_signature_of': '— observables χ̃-invariant by construction',
    },
]


# ============================================================================
# Compute + tabulate
# ============================================================================

def main():
    print("=" * 130)
    print("M5 thorough enumeration — every known open framework gap encoded as M5 candidate")
    print("=" * 130)
    print()
    print(f"Candidates: {len(candidates)} (vs. 12 in prior textbook enumeration)")
    print()
    print("Sources catalogued:")
    sources = sorted(set(c['source'] for c in candidates))
    for s in sources:
        print(f"  - {s}")
    print()
    print("Formula (validated in Task #7): W(M, N) = Φ − L + min(freq_factor, 0)")
    print()

    N_hub = 10 ** 60

    print(f"{'source':<22} {'name':<60} {'|E|':>4} {'L_M':>5} {'dim':>10} {'W@N_hub':>12}")
    print("-" * 130)

    retained = []
    not_retained = []
    for c in candidates:
        E = c['E']
        L = c['L_M']
        max_Lr = c['max_L_r']
        Phi = Phi_compression(E, c['dim'], N_hub)
        ff = freq_factor(E, max_Lr, N_hub)
        W = W_combined(Phi, L, ff)
        c['W_at_Nhub'] = W

        dim_str = f"{c['dim']:.2e}" if c['dim'] > 10000 else str(c['dim'])
        W_str = _fmt(W)
        print(f"{c['source']:<22} {c['name'][:60]:<60} {E:>4} {L:>5.1f} {dim_str:>10} {W_str:>12}")

        if W > 0:
            retained.append(c)
        else:
            not_retained.append(c)

    print()
    print("-" * 130)
    print(f"Retained at N_hub: {len(retained)} / {len(candidates)}")
    if not_retained:
        print(f"NOT retained: {len(not_retained)}")
        for c in not_retained:
            print(f"  - {c['name']}: W = {c['W_at_Nhub']:.2e}")

    # ============================================================
    # Group by what they would close
    # ============================================================
    print()
    print("=" * 130)
    print("Grouped by what each candidate would close")
    print("=" * 130)
    print()

    # MSSM-relevant candidates
    print("MSSM-RELEVANT candidates (directly affect MSSM derivation question):")
    print("-" * 130)
    mssm_relevant = [c for c in retained if any(k in c['name'].lower()
                                                  for k in ['squark', 'slepton', 'gluino', 'neutralino', 'chargino', 'tan β', 'higgs sector', 'witten', 'sector'])]
    for c in mssm_relevant:
        print(f"  [{c['source']}] {c['name']}")
        print(f"     Would close: {c['would_close']}")
        print(f"     Open: {c['open_question']}")
        print(f"     Signature: {c['matches_signature_of']}")
        print()

    print("OTHER candidates (other framework gaps, possibly M5 content):")
    print("-" * 130)
    other = [c for c in retained if c not in mssm_relevant]
    for c in other:
        print(f"  [{c['source']}] {c['name']}")
        print(f"     Would close: {c['would_close']}")
        print(f"     Open: {c['open_question']}")
        print(f"     Signature: {c['matches_signature_of']}")
        print()

    # ============================================================
    # Honest assessment
    # ============================================================
    print("=" * 130)
    print("Honest assessment — what this enumeration shows vs doesn't")
    print("=" * 130)
    print()
    print(f"  RETENTION: {len(retained)}/{len(candidates)} candidates retained at N_hub. The W(M, N)")
    print(f"  formula gives positive weight to every candidate with sensible (E, L_M, max_L_r, dim)")
    print(f"  parameters. This is expected — N_hub = 10^60 gives essentially every named structure")
    print(f"  astronomical margin.")
    print()
    print(f"  WHAT RETENTION ALONE DOES NOT TELL US:")
    print(f"  - Whether the framework's M5 mechanism actually PRODUCES the candidate.")
    print(f"  - Whether the candidate's substrate primitives match the named observable.")
    print(f"  - Whether the candidate's quantum numbers / loop contributions match what's needed.")
    print(f"  All of these are separate structural questions.")
    print()
    print(f"  WHAT'S CLEAR FROM THE ENUMERATION:")
    print(f"  - The framework has {len(candidates)} explicitly-named open gaps with at least partial")
    print(f"    structural scoping. Any of them could be M5-content candidates.")
    print(f"  - 5 OPEN residue-register entries (R-9, R-13, R-14, R-15, R-16) + the SUSY spectrum,")
    print(f"    quark Yukawas, cosmology residuals, and Feshbach analogs constitute the actual M5")
    print(f"    candidate set, NOT the textbook Lie algebra list of the prior enumeration.")
    print(f"  - For MSSM specifically: candidates from MP-4.1 (SUSY spectrum) are the natural")
    print(f"    candidates; whether the framework's M5 mechanism produces them with right quantum")
    print(f"    numbers is the actual MSSM-derivation question, and the prior 'no clean match' verdict")
    print(f"    was based on a narrower enumeration that didn't include these.")
    print()
    print(f"  WHAT'S NEEDED NEXT (per-candidate, not done in this script):")
    print(f"  - For each MSSM-relevant candidate, compute substrate quantum numbers from its")
    print(f"    structural shape (Cl(6) Fock decomposition, gauge action, etc.)")
    print(f"  - Compare to MSSM partner reps. Match → potential M5 content; mismatch → not.")
    print(f"  - For matched candidates, compute loop contributions to β-functions and compare to")
    print(f"    MSSM-minus-SM delta (5/2, 25/6, 4).")
    print(f"  - This is per-candidate structural work, not a single-session probe.")
    print()
    print(f"  REVISED VERDICT on prior M5 closure:")
    print(f"  - The prior 'triply-closed' verdict (Task #6 + earlier) overstated. Specifically,")
    print(f"    the third leg (M5 candidate enumeration) was based on a candidate set that did")
    print(f"    NOT include the framework's actual open SUSY/sector/Yukawa gaps as M5 candidates.")
    print(f"  - ADOPTED-MSSM-Sb is still the linter-grade position by default (no derived closure),")
    print(f"    but the structural argument for 'forced adoption' is weaker than I claimed.")
    print(f"  - The MSSM question is genuinely OPEN with multiple candidate paths via the gaps")
    print(f"    enumerated above. Whether any closes requires the per-candidate structural work.")


def _fmt(W):
    if abs(W) > 1e15:
        sign = '+' if W > 0 else '-'
        mag = int(math.log10(abs(W)))
        return f"{sign}10^{mag:>2}"
    return f"{W:>+10.2f}"


if __name__ == "__main__":
    main()
