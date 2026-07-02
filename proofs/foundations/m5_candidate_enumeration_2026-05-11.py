"""
proofs/foundations/m5_candidate_enumeration_2026-05-11.py

M5 candidate enumeration (Task #6): apply the framework's W(M, N) formula,
validated in Task #7 sanity check, to STANDALONE substrate objects that are
NOT in the dominant visible alphabet — testing whether any retained
candidate produces content with MSSM-partner quantum numbers.

Background:
- M5 ("dark sector / convergent-emergence access mechanism") is UNCONNECTED
  per M_mechanisms_synthesis_2026-05-07: it could in principle access
  subdominant zoo retention but its formal definition is open.
- The user clarified: convergent persistent structures emerge in the
  multiway via the toggle Markov even without observer prior anchoring.
- Sanity check confirms: dominant alphabet members clear W(M, N_hub) by
  astronomical margins. Smallest visible margin: ~192 bits (Cl(0,2)).
- This script enumerates SUBDOMINANT standalone candidates and asks:
    (a) Do they clear W(M, N_hub) > 0?
    (b) Do their algebraic representations contain MSSM-partner content?

CANDIDATE CLASSES:

  Class I — Subdominant zoo Lie algebras (formally retained at N_hub
    per saturated_symmetry_zoo doc; access mechanism = M5):
      F_4 (52-dim, via Albert algebra J_3(𝕆))
      E_6 (78-dim)
      E_7 (133-dim)
      E_8 (248-dim)

  Class II — Alternative vertex algebras (non-dominant):
      𝕆 (octonion, non-associative, Cayley-Dickson depth 3)
      Cl(8,0) (next associative Clifford)
      Cl(10,0)

  Class III — Alternative edge algebras:
      Cl(0,4)
      Cl(0,6)

  Class IV — Composites of dominant content (Possibility B):
      Cl(6)⊗Cl(6) (diquark candidate)
      Cl(6)⊗Cl(0,2) (fermion-Higgs candidate)

DECLARED PREDICTIONS (before computation):

  P1. Class I (Lie algebras): all retained at N_hub (W > 0). Adjoint
      decompositions under SM gauge group contain extra reps beyond SM.
      Whether those extra reps match MSSM-partner quantum numbers is an
      open question handled by branching-rule lookup (Slansky 1981).

  P2. Class II (alternative vertex algebras): retention likely. Octonion
      non-associativity could carry triality structure → bosons + fermions.

  P3. Class III (alternative edges): probably retained.

  P4. Class IV (composites): likely retained but spin/statistics analysis
      shows composites have OPPOSITE statistics to constituents (2-fermion
      = boson; fermion+boson = fermion). Whether quantum numbers match
      MSSM partners requires explicit check.

FAILURE MODES TO AVOID:
  N1: numerical match without mechanism (M5 itself is unformalized)
  N2: relabeling existing content as M5 content
  N3: vague claims about subdominant structures without specific quantum
      number content
  N4: matching quantum numbers without checking that the framework's M5
      mechanism actually produces those reps (vs. just being content of
      the Lie algebra)
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# ============================================================================
# Formula primitives (same as Task #7 sanity check)
# ============================================================================

def L_elias(m):
    if m == float('inf'):
        return 1.0
    if m <= 1:
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


def N_attest(E, max_L_r):
    return E ** max_L_r


def W_combined(Phi, L, freq):
    return Phi - L + min(freq, 0.0)


# ============================================================================
# M5 candidates with quantum-number content
# ============================================================================
# For each candidate:
#   - E: number of generators (must be ≥ 2 for the formula to apply)
#   - L_M: description length of M (Elias-encoded)
#   - max_L_r: longest defining relation
#   - dim: model dim (vector-space dim of algebra, or |Weyl group| for Lie alg)
#   - sm_content: list of SM gauge representations contained in this object's
#     adjoint / fundamental, with their statistics. Standard branching rules
#     from Slansky 1981 / Lie algebra references.
#   - matches_mssm: list of MSSM-partner reps the candidate contains, if any.

candidates = [
    # ========== Class I — Exceptional Lie algebras (subdominant zoo) ==========
    # Branching rules: Slansky 1981 "Group theory for unified model building"
    # F_4 → SO(9) → ... → SM. 52-dim adjoint contains relatively narrow SM content.
    {
        'class': 'I-Lie',
        'name': 'F_4 (rank 4, dim 52, via J_3(𝕆) Albert algebra)',
        'E': 4,
        'L_M': L_elias(4) + L_elias(52) + 3,
        'max_L_r': 6,
        'dim': 1152,  # |W(F_4)| = 1152 (Weyl group order)
        'sm_content_summary': 'F_4 adjoint = 52; under SO(9): 36+16. SO(9)→SM has limited rep variety.',
        'mssm_partner_candidates': [],  # no clean superpartner-shaped reps
    },
    {
        'class': 'I-Lie',
        'name': 'E_6 (rank 6, dim 78)',
        'E': 6,
        'L_M': L_elias(6) + L_elias(78) + 3,
        'max_L_r': 6,
        'dim': 51840,  # |W(E_6)|
        'sm_content_summary': 'E_6 fundamental 27 → SU(5): 10+5̄+1. 78 adjoint under SM: contains gauge + Higgs-like reps but no clean (3,1,1/6)-style sfermion.',
        # The 27 of E_6 has well-known content: under SM gauge group, contains
        # SM fermions per generation (16 of SO(10) embedded) plus extras.
        # NOT MSSM partners specifically.
        'mssm_partner_candidates': [],
    },
    {
        'class': 'I-Lie',
        'name': 'E_7 (rank 7, dim 133)',
        'E': 7,
        'L_M': L_elias(7) + L_elias(133) + 3,
        'max_L_r': 6,
        'dim': 2903040,
        'sm_content_summary': 'E_7 56-rep → E_6 27+27̄+1+1. Contains 2 copies of E_6 27 reps. Doubling structure is suggestive but not specifically MSSM-shaped.',
        # E_7 56 under E_6: 27 + 27̄ + 1 + 1. Two copies of matter content.
        # The "doubling" might look superpartner-like at first but it's
        # particle/antiparticle, not boson/fermion partnering.
        'mssm_partner_candidates': [],
    },
    {
        'class': 'I-Lie',
        'name': 'E_8 (rank 8, dim 248)',
        'E': 8,
        'L_M': L_elias(8) + L_elias(248) + 3,
        'max_L_r': 6,
        'dim': 696729600,
        'sm_content_summary': 'E_8 248 adjoint contains SU(3)×E_6 with 248 = (8,1)+(1,78)+(3,27)+(3̄,27̄). Vast extra content beyond SM. Heterotic string uses E_8×E_8 with SUSY ADDED; without SUSY, no clean partner structure.',
        # Critical observation: heterotic string THEORY uses E_8 + SUPERSYMMETRY
        # to derive MSSM. The SUSY is ADDED, not derived from E_8 itself.
        # In the framework's context (no automatic SUSY), E_8 retention
        # gives extra Lie content but NOT superpartner structure.
        'mssm_partner_candidates': [],
    },

    # ========== Class II — Alternative vertex algebras ==========
    {
        'class': 'II-vertex',
        'name': '𝕆 (octonion algebra at vertex, non-associative)',
        'E': 3,  # 3 imaginary unit generators (e_1, e_2, e_4 in standard basis)
        'L_M': L_elias(8) + L_elias(3) + 4,  # dim + generators + "non-assoc" marker
        'max_L_r': 6,  # associator relation
        'dim': 8,
        'sm_content_summary': 'Aut(𝕆) = G_2; 8-dim rep splits as 1+7 under G_2. Triality structure: three 8-dim reps of Spin(8) related by triality. Could host fermions+bosons via different triality channels.',
        # Octonion triality is suggestive — three 8-dim reps of Spin(8)
        # (vector V, two spinor reps S+, S-) related by triality. The
        # framework has Cl(6) → Spin(6) → SU(4) which is bosonic structure.
        # Octonion triality would give DIFFERENT statistics from Cl(6).
        # If 𝕆 retained, could give boson+fermion partner structure.
        'mssm_partner_candidates': ['triality-shifted partners possible (needs M5 mechanism for triality access)'],
    },
    {
        'class': 'II-vertex',
        'name': 'Cl(8,0) at vertex (k=4, next associative after Cl(6))',
        'E': 8,
        'L_M': L_elias(8) + L_elias(0) + 2,
        'max_L_r': 4,
        'dim': 2 ** 8,
        'sm_content_summary': 'Cl(8,0) → Spin(8); 16-dim spinor rep splits under triality into S+, S-, V (each 8-dim). Hosts SO(8) gauge — bigger than SM SU(3)×SU(2)×U(1) but not MSSM-shaped.',
        'mssm_partner_candidates': [],
    },
    {
        'class': 'II-vertex',
        'name': 'Cl(10,0) at vertex (k=5, SO(10) GUT-like)',
        'E': 10,
        'L_M': L_elias(10) + L_elias(0) + 2,
        'max_L_r': 4,
        'dim': 2 ** 10,
        'sm_content_summary': 'Cl(10,0) → Spin(10) ≅ SO(10) GUT. 16-dim spinor = one fermion generation under SU(5). Standard SO(10) GUT content — same matter as SM, no MSSM partners.',
        'mssm_partner_candidates': [],
    },

    # ========== Class III — Alternative edge algebras ==========
    {
        'class': 'III-edge',
        'name': 'Cl(0,4) at edge (extended weak-sector algebra)',
        'E': 4,
        'L_M': L_elias(0) + L_elias(4) + 2,
        'max_L_r': 4,
        'dim': 2 ** 4,
        'sm_content_summary': 'Cl(0,4) ≅ ℍ(2) = quaternion 2×2 matrices, dim 16. Spin(4) ≅ SU(2)×SU(2) — same as PS Spin(4) part but doubled. Doesn\'t generate new sector beyond PS.',
        'mssm_partner_candidates': [],
    },
    {
        'class': 'III-edge',
        'name': 'Cl(0,6) at edge (would give SU(4) structure at edge)',
        'E': 6,
        'L_M': L_elias(0) + L_elias(6) + 2,
        'max_L_r': 4,
        'dim': 2 ** 6,
        'sm_content_summary': 'Cl(0,6) ≅ ℝ(8) = real 8×8 matrices. Spin(6) ≅ SU(4) at edge. Redundant with vertex Cl(6)\'s SU(4) — same gauge content at different layer.',
        'mssm_partner_candidates': [],
    },

    # ========== Class IV — Composites of dominant content (Possibility B) ==========
    {
        'class': 'IV-composite',
        'name': 'Diquark Cl(6)⊗Cl(6) composite (spin-0 squark-like)',
        'E': 12,  # tensor product: 6 + 6 generators
        'L_M': L_elias(6) + L_elias(6) + L_elias(2) + 4,  # 2 factors + tensor marker
        'max_L_r': 4,
        'dim': 2 ** 12,  # = 4096 (tensor algebra dim)
        'sm_content_summary': 'Two-fermion composite: spin-1/2 ⊗ spin-1/2 = spin 0 or 1. Color: 3 ⊗ 3 = 3̄ + 6 (or in color singlet channel). Charge: 2/3+2/3=4/3 (uu) or 2/3-1/3=1/3 (ud) or others. None matches squark (3,2,1/6) cleanly.',
        # Diquark has charge -1/3 or 4/3 for color-3̄ channel; squark has
        # the SAME charge as the quark partner (1/6 hypercharge for Q_L).
        # Charge mismatch: composite has Q_quark+Q_quark, squark has Q_quark.
        'mssm_partner_candidates': [],
    },
    {
        'class': 'IV-composite',
        'name': 'Fermion-Higgs Cl(6)⊗Cl(0,2) (spin-1/2 higgsino-like)',
        'E': 8,  # 6 + 2
        'L_M': L_elias(6) + L_elias(2) + L_elias(2) + 4,
        'max_L_r': 4,
        'dim': 2 ** 8,
        'sm_content_summary': 'Fermion ⊗ Higgs-doublet composite: spin-1/2 (fermion) ⊗ spin-0 (Higgs) = spin 1/2. SU(2) doublet (from Higgs). Hypercharge: Y_fermion + Y_Higgs. If fermion is lepton-L (Y=-1/2), result has Y=0 — NOT higgsino (Y=±1/2).',
        # Higgsino is (1,2,1/2) — spin-1/2 SU(2)-doublet hypercharge +1/2.
        # Fermion + Higgs composite has wrong hypercharge unless we pick
        # a fermion that adds to the right total — but fermions have
        # specific Y values fixed by Cl(6) Fock decomposition.
        # CHARGE MISMATCH for the natural composite.
        'mssm_partner_candidates': [],
    },
    {
        'class': 'IV-composite',
        'name': 'Lepton-pair Cl(6)⊗Cl(6)bar (slepton-like)',
        'E': 12,
        'L_M': L_elias(6) + L_elias(6) + L_elias(2) + 4,
        'max_L_r': 4,
        'dim': 2 ** 12,
        'sm_content_summary': 'Lepton-antilepton: spin 0 or 1. Charge: -1+1=0 or -1+0=-1. Color: 1⊗1=1. Could match (1,1,0) slepton-like — but charge 0 slepton would be sneutrino-like, not selectron. Doublet structure not natural from anti-pair.',
        # Lepton+anti-lepton bound state has total lepton number 0 (which
        # selectrons don't). Doesn't match sneutrino either (sneutrinos
        # have lepton number ±1).
        'mssm_partner_candidates': [],
    },
]


# ============================================================================
# Computation + verdict
# ============================================================================

def main():
    print("=" * 110)
    print("M5 candidate enumeration — does the subdominant zoo contain MSSM-partner content?")
    print("=" * 110)
    print()
    print("Formula (validated in Task #7 sanity check):")
    print("  W(M, N) = Φ(M, N) − L(M) + min(freq_factor(M, N), 0)")
    print()
    print("Declared predictions:")
    print("  - Class I-Lie algebras: retained but extra content not specifically MSSM-shaped")
    print("  - Class II-vertex alternatives: retained; 𝕆 triality is the only one with")
    print("    plausible boson+fermion partner structure")
    print("  - Class III-edge alternatives: retained but no novel content")
    print("  - Class IV-composites: retained but quantum numbers don't match MSSM partners")
    print()
    print("Failure modes monitored: N1-N4 declared in script docstring")
    print()

    N_hub = 10 ** 60
    N_values = [10**2, 10**4, 10**6, 10**10, 10**30, 10**60]

    header_N = "  ".join(f"W@10^{int(math.log10(N)):>2}" for N in N_values)
    print(f"{'name':<60} {'class':<14} {'|E|':>4} {'L_M':>5} {'dim':>10}  {header_N}")
    print("-" * 110)

    retained_at_Nhub = []
    for c in candidates:
        E = c['E']
        L = c['L_M']
        max_Lr = c['max_L_r']
        Ws = []
        for N in N_values:
            Phi = Phi_compression(E, c['dim'], N)
            ff = freq_factor(E, max_Lr, N)
            W = W_combined(Phi, L, ff)
            Ws.append(W)
        Ws_str = "  ".join(_fmt(W) for W in Ws)
        # Show truncated dim for display
        dim_str = f"{c['dim']:.2e}" if c['dim'] > 10000 else str(c['dim'])
        print(f"{c['name'][:60]:<60} {c['class']:<14} {E:>4} {L:>5.1f} {dim_str:>10}  {Ws_str}")
        if Ws[-1] > 0:
            retained_at_Nhub.append(c)

    print()
    print("-" * 110)
    print(f"Retained at N_hub: {len(retained_at_Nhub)} / {len(candidates)} candidates")
    print()

    # Verdict: which retained candidates have MSSM partner content?
    print("=" * 110)
    print("MSSM-partner content analysis (retained candidates only)")
    print("=" * 110)
    print()
    print("MSSM partner reps to match:")
    print("  - Squarks: spin 0 boson, (3, 2, 1/6) or (3, 1, 2/3) or (3, 1, -1/3)")
    print("  - Sleptons: spin 0 boson, (1, 2, -1/2) or (1, 1, -1)")
    print("  - Gauginos: spin 1/2 fermion, adjoint of SM gauge group")
    print("  - Higgsinos: spin 1/2 fermion, (1, 2, ±1/2)")
    print()
    print(f"{'name':<60} {'SM content / MSSM match':<50}")
    print("-" * 110)
    matches_found = 0
    for c in retained_at_Nhub:
        match_str = "; ".join(c['mssm_partner_candidates']) if c['mssm_partner_candidates'] else "— (none cleanly match)"
        print(f"{c['name'][:60]:<60} {match_str[:50]:<50}")
        if c['mssm_partner_candidates']:
            matches_found += 1

    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    print()
    if matches_found == 0:
        print(f"  Status: NO retained M5 candidate produces structures with MSSM-partner")
        print(f"  quantum numbers cleanly.")
        print()
        print(f"  Retained candidates ({len(retained_at_Nhub)} of {len(candidates)}) all have")
        print(f"  either:")
        print(f"  - Extra Lie-algebra content unrelated to MSSM partner structure (Class I)")
        print(f"  - Extended gauge content but standard fermion statistics (Class II-vertex,")
        print(f"    except 𝕆 triality which is open-question)")
        print(f"  - Redundant gauge content at different substrate layer (Class III)")
        print(f"  - Wrong quantum numbers for MSSM partners (Class IV-composites)")
        print()
        print(f"  CONCLUSION: at the level of standalone-subdominant-content quantum")
        print(f"  number matching, the framework's M5 candidates do NOT cleanly produce")
        print(f"  MSSM-partner spectrum. ADOPTED-MSSM-Sb position confirmed at this level.")
    elif matches_found == 1 and retained_at_Nhub[0]['name'].startswith('𝕆'):
        print(f"  Status: ONLY 𝕆 (octonion) triality is a residual open candidate")
        print(f"  for MSSM-partner-shaped content via M5.")
    else:
        print(f"  Status: {matches_found} retained candidates have MSSM-partner-candidate content.")
        print(f"  These need detailed quantitative analysis (β-function loop contributions)")
        print(f"  before claiming MSSM-derivation.")

    print()
    print("=" * 110)
    print("Notes on what this does NOT close")
    print("=" * 110)
    print()
    print("  - M5 is still UNCONNECTED (no formal mechanism for how convergent")
    print("    structures access the subdominant zoo). This enumeration assumes")
    print("    the formal mechanism exists; if it doesn't, the candidates remain")
    print("    formally retained but observably inaccessible.")
    print("  - Loop contributions to β-functions are NOT computed here. Even if")
    print("    a candidate had MSSM-partner quantum numbers, matching (5/2, 25/6, 4)")
    print("    requires explicit loop integral computation.")
    print("  - The branching-rule analysis is based on standard Lie algebra")
    print("    decompositions (Slansky 1981) — well-established but interpretation")
    print("    in the framework requires substrate-side derivation.")
    print("  - 𝕆 triality is the only 'maybe' surviving this enumeration. Triality")
    print("    structurally gives three 8-dim reps of Spin(8) (V, S+, S-) which")
    print("    have different transformation properties under reflections.")
    print("    Whether the framework's M5 mechanism would naturally select")
    print("    triality-shifted partners for MSSM-style content is open research")
    print("    (likely needs explicit substrate-octonion access mechanism).")


def _fmt(W):
    if abs(W) > 1e15:
        sign = '+' if W > 0 else '-'
        mag = int(math.log10(abs(W)))
        return f"{sign}10^{mag:>2}"
    return f"{W:>+9.2f}"


if __name__ == "__main__":
    main()
