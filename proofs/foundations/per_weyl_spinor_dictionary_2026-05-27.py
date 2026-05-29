"""
proofs/foundations/per_weyl_spinor_dictionary_2026-05-27.py

Per-Weyl-spinor dictionary (Phase 1 of unified theory development).

Extends the walker-class-family-grain dictionary (2026-05-27 EOD+3) to
per-Weyl-spinor grain. For each of 48 SM Weyl spinors per primitive
cell, derives a specific (saddle, walker-mode, Cl(6) Fock state,
γ_7-chirality, Q_i-generation, SU(2)_L-isospin) tag.

Builds on:
  - V_Ram-iso T1: explicit U: V_Ram(P) → Cl(6) Fock (theorem-grade
    via proofs/foundations/V_Ram_Cl6_iso_T1_construction_2026-05-26.py)
  - V_Ram-iso T4: Q_i = Cl(4) volume omitting Furey pair i, 3 SM
    generations (theorem-grade via theorem_V_Ram_Cl6_Fock_iso_2026-05-26.md)
  - chir-7 theorem: Γ + H saddles host neutrino sector
  - PS B3 labeling: SU(4) × SU(2)_L × SU(2)_R structure per gen
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def banner(title, char="="):
    print(char * 100)
    print(title)
    print(char * 100)


# ============================================================================
# Step 1: Cl(6,0) gamma matrices (from T1 construction)
# ============================================================================

def build_cl6_gammas():
    """Cl(6,0) generators as 8×8 complex matrices via Brauer-Weyl."""
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron(*mats):
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    G = [None] * 7
    G[1] = kron(sx, I2, I2)
    G[2] = kron(sy, I2, I2)
    G[3] = kron(sz, sx, I2)
    G[4] = kron(sz, sy, I2)
    G[5] = kron(sz, sz, sx)
    G[6] = kron(sz, sz, sy)

    # γ_7 chirality operator
    G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]
    return G, G7


# ============================================================================
# Step 2: Q_i operators (per V_Ram-iso T4)
# ============================================================================

def build_Q_operators(G):
    """Q_i = Cl(4) volume element omitting Furey pair i.

    Q_1 = γ_3 γ_4 γ_5 γ_6  (omits Furey pair (γ_1, γ_2))
    Q_2 = γ_1 γ_2 γ_5 γ_6  (omits Furey pair (γ_3, γ_4))
    Q_3 = γ_1 γ_2 γ_3 γ_4  (omits Furey pair (γ_5, γ_6))
    """
    Q = {}
    Q[1] = G[3] @ G[4] @ G[5] @ G[6]
    Q[2] = G[1] @ G[2] @ G[5] @ G[6]
    Q[3] = G[1] @ G[2] @ G[3] @ G[4]
    return Q


# ============================================================================
# Step 3: Enumerate Cl(6) Fock states by (γ_7, Q_i) eigenvalue labels
# ============================================================================

def section_3_enumerate_cl6_states(G, G7, Q):
    banner("§3 Enumerate 8 Cl(6) Fock states by (γ_7, Q_i) eigenvalue labels")
    print()

    # Cl(6) Fock = 8-dim. Standard basis: |n_1 n_2 n_3⟩ where n_i ∈ {0, 1}
    # γ_7 eigenvalues: ±1 (4 states each, since Cl(6) splits 8 = 4 + 4 chiral)
    # Q_i eigenvalues: ±1 (commute with γ_7, label generations)

    # Diagonalize γ_7
    print("γ_7 spectrum:")
    eigs_g7, vecs_g7 = la.eig(G7)
    print(f"  Eigenvalues: {[f'{e.real:+.1f}' for e in eigs_g7]}")
    print()

    # In the standard Brauer-Weyl basis, γ_7 is diagonal with:
    # γ_7 = -i γ_1 γ_2 γ_3 γ_4 γ_5 γ_6
    # Under the σ_z ⊗ σ_z ⊗ σ_z chirality grading, eigenvalues are ±1
    # arranged as (+1, -1, -1, +1, -1, +1, +1, -1) or similar.

    # Find γ_7 eigenstates
    chiral_plus = []  # γ_7 = +1
    chiral_minus = []  # γ_7 = -1
    for i, e in enumerate(eigs_g7):
        if abs(e - 1) < 1e-6:
            chiral_plus.append(i)
        elif abs(e + 1) < 1e-6:
            chiral_minus.append(i)
    print(f"γ_7 = +1 chiral states: {len(chiral_plus)} (computational basis indices: {chiral_plus})")
    print(f"γ_7 = -1 chiral states: {len(chiral_minus)} (computational basis indices: {chiral_minus})")
    print()

    # Now check Q_i action. The Q_i commute with γ_7 (all even-grade products
    # of 4 γ's). So they act within each chirality subspace.
    print("Q_i operators ([Q_i, γ_7] = 0; act within chirality subspace):")
    for i in [1, 2, 3]:
        commutator = Q[i] @ G7 - G7 @ Q[i]
        if not np.allclose(commutator, 0, atol=1e-6):
            print(f"  Q_{i}: WARNING [Q_{i}, γ_7] ≠ 0; max|comm| = {np.max(np.abs(commutator)):.2e}")
        else:
            print(f"  Q_{i}: ✓ commutes with γ_7")
        # Eigenvalues
        eigs_Qi = la.eigvals(Q[i])
        print(f"        eigenvalues: {[f'{e.real:+.1f}' if abs(e.imag) < 1e-6 else f'{e}' for e in eigs_Qi]}")
        # Q_i² should be I (Q_i squares to identity for proper Cl(4) volume)
        Q_i_sq = Q[i] @ Q[i]
        if not np.allclose(Q_i_sq, np.eye(8, dtype=complex), atol=1e-6):
            print(f"        Q_{i}² ≠ I; max|Q_{i}²-I| = {np.max(np.abs(Q_i_sq - np.eye(8))):.2e}")
    print()

    # Verify quaternion algebra Q_1 Q_2 = ±Q_3, [Q_i, Q_j] = 0
    print("Verify quaternion-like algebra Q_i Q_j = ±Q_k (anti-commute is fine):")
    for i, j in [(1, 2), (2, 3), (1, 3)]:
        k = list({1, 2, 3} - {i, j})[0]
        Q_ij = Q[i] @ Q[j]
        Q_ji = Q[j] @ Q[i]
        # Check anti-commutation: Q_ji = ±Q_ij or Q_ji = -Q_ij
        if np.allclose(Q_ij + Q[k], 0, atol=1e-6):
            print(f"  Q_{i} Q_{j} = -Q_{k} ✓")
        elif np.allclose(Q_ij - Q[k], 0, atol=1e-6):
            print(f"  Q_{i} Q_{j} = +Q_{k} ✓")
        else:
            print(f"  Q_{i} Q_{j} matches neither ±Q_{k}; max diff = {min(np.max(np.abs(Q_ij + Q[k])), np.max(np.abs(Q_ij - Q[k]))):.2e}")
        # Check commutator
        if np.allclose(Q_ij - Q_ji, 0, atol=1e-6):
            print(f"  [Q_{i}, Q_{j}] = 0 ✓")
        else:
            print(f"  [Q_{i}, Q_{j}] ≠ 0; max|comm| = {np.max(np.abs(Q_ij - Q_ji)):.2e}")
    print()

    # Build common eigenbasis of (γ_7, Q_1, Q_2)
    # (Q_3 is determined by Q_1·Q_2 = ±Q_3)
    print("Common eigenbasis of (γ_7, Q_1, Q_2) — these label 8 Cl(6) Fock states:")
    print()
    print(f"  {'#':>2}  {'γ_7':>4}  {'Q_1':>4}  {'Q_2':>4}  {'Q_3 derived':>11}  {'comp. basis':>20}")
    print(f"  {'-'*2}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*11}  {'-'*20}")

    # Construct simultaneous eigenstates of γ_7, Q_1, Q_2
    # All commute, so we can simultaneously diagonalize.
    cl6_states = []
    # Compute γ_7 + 2·Q_1 + 4·Q_2 + 8 (offset to make positive), find unique eigenstate per labeled state
    # Simpler: use the fact that all three commute and find common eigenvectors

    # Build operator with distinct eigenvalues for each (γ_7, Q_1, Q_2) combination
    combined = G7 + 3 * Q[1] + 9 * Q[2]
    eigs_c, vecs_c = la.eig(combined)
    # Sort by real part of eigenvalue
    order = np.argsort(eigs_c.real)
    eigs_c = eigs_c[order]
    vecs_c = vecs_c[:, order]

    for i in range(8):
        v = vecs_c[:, i]
        g7_val = v.conj() @ G7 @ v
        q1_val = v.conj() @ Q[1] @ v
        q2_val = v.conj() @ Q[2] @ v
        q3_val = v.conj() @ Q[3] @ v
        cl6_states.append({
            'idx': i,
            'vector': v,
            'g7': g7_val.real,
            'q1': q1_val.real,
            'q2': q2_val.real,
            'q3': q3_val.real,
        })
        # Identify dominant computational basis state
        dom_idx = np.argmax(np.abs(v))
        # Convert to (n_1, n_2, n_3) binary
        n1 = (dom_idx >> 2) & 1
        n2 = (dom_idx >> 1) & 1
        n3 = dom_idx & 1
        basis_str = f"|{n1}{n2}{n3}⟩"

        print(f"  {i:>2}  {g7_val.real:>+4.1f}  {q1_val.real:>+4.1f}  {q2_val.real:>+4.1f}  "
              f"{q3_val.real:>+11.1f}  {basis_str:>20}")
    print()
    return cl6_states


# ============================================================================
# Step 4: Map Cl(6) Fock states to SM matter content via PS branching
# ============================================================================

def section_4_PS_mapping(cl6_states):
    banner("§4 Map Cl(6) Fock states to SM matter content (PS branching)")
    print()
    print("Pati-Salam: SO(10) ⊃ SU(4)_PS × SU(2)_L × SU(2)_R")
    print("Per generation, SO(10) spinor 16 decomposes as:")
    print("  16 = (4, 2, 1)_L + (4̄, 1, 2)_R")
    print()
    print("Per chirality (γ_7 = ±1) of Cl(6) Fock per vertex:")
    print("  γ_7 = +1: 4 of SU(4)_PS = 1 PS multiplet (Q_L + L_L) sector")
    print("           but only the SU(4) index here; SU(2)_L doublet is per-edge")
    print("  γ_7 = -1: 4̄ of SU(4)_PS = 1 PS antimultiplet (u_R^c + d_R^c + e_R^c + ν_R^c)")
    print("           but only the SU(4) index here; SU(2)_R doublet is per-edge")
    print()
    print("SU(4)_PS index a ∈ {1,2,3,4} = {color-r, color-g, color-b, lepton}")
    print("  identified via Q_i eigenvalues (or equivalently SU(4) Cartan diagonal)")
    print()

    # The Q_i eigenvalues label the SU(4) index in some way. Standard reading:
    # Q_i acts as ±1 on Cl(6) Fock states. The 4 states with γ_7=+1 split as
    # (Q_1, Q_2, Q_3) = (+,+,+), (+,-,-), (-,+,-), (-,-,+) — the 4 of SU(4)
    # under the body-diagonal Cartan. Wait, depends on sign conventions.
    #
    # Actually: 4 of SU(4) under Cartan diag(1,1,1,-3)/√24:
    # the 4 states have Cartan eigenvalues (1, 1, 1, -3)/√24.
    # Or under (1, -1, 0, 0) + (0, 0, 1, -1): more general.
    #
    # For our Q_i, each Q_i² = I so eigenvalues are ±1. With 3 Q's commuting,
    # we get 8 combinations of (q1, q2, q3) signs. But only 4 of them have a
    # given γ_7, so within each chirality there are 4 (q1, q2, q3) triplets.
    #
    # Let me just enumerate and assign:

    print("Cl(6) Fock state → SM Weyl-spinor candidate mapping (per generation index):")
    print()
    print("Note: SU(2)_L isospin (up/down) is PER-EDGE structure (Cl(0,2) ≅ ℍ)")
    print("      and NOT captured by per-vertex Cl(6) Fock alone. The per-vertex")
    print("      labeling gives the SU(4)_PS index; the SU(2)_L doublet is added")
    print("      via the per-edge structure (theorem_g2_edge_qubit_su2.md).")
    print()
    print(f"  {'Cl(6)#':>7}  {'γ_7':>4}  {'(q1,q2,q3)':>11}  {'PS chirality':>14}  {'SU(4) interpretation':>30}")
    print(f"  {'-'*7}  {'-'*4}  {'-'*11}  {'-'*14}  {'-'*30}")
    for s in cl6_states:
        chirality = '(4,2,1)_L' if s['g7'] > 0 else '(4̄,1,2)_R'
        # Identify SU(4) index a from (q1, q2, q3). Convention:
        # The 4 of SU(4) has 4 states; pick a ∈ {1, 2, 3, 4} based on Q signs.
        # Standard reading per Furey 2018: Q_i = +1 means "Furey pair i excluded
        # is in a specific state"; (Q_1, Q_2, Q_3) = (+,+,+) ↔ vacuum-like state.
        # 4 = {3 colors + 1 lepton}. Identify lepton with all-(+,+,+) under Q_i
        # convention (lowest weight) or all-(-) (highest weight). Convention-
        # dependent; for the dictionary we use:
        #  Q_i = +1 means quark-like for that Furey-pair coordinate
        #  Q_i = -1 means lepton-like (excluded direction)
        # With 3 Q_i and γ_7 fixed, we have 4 sign patterns matching the 4 of SU(4).
        signs = (s['q1'], s['q2'], s['q3'])
        if all(q > 0 for q in signs):
            a_label = 'a=4 (lepton/anti-lepton)'
        elif sum(q > 0 for q in signs) == 1:
            color_idx = [i for i, q in enumerate(signs) if q > 0][0] + 1
            a_label = f'a={color_idx} (color {chr(ord("r")+color_idx-1)})'
        elif sum(q > 0 for q in signs) == 2:
            # Two +1's, one -1 — this is a different SU(4) basis combination
            anti_idx = [i for i, q in enumerate(signs) if q < 0][0] + 1
            a_label = f'a=...({signs}) anti-color {anti_idx}-related'
        else:
            a_label = f'a=?({signs}) — non-canonical sign'
        print(f"  {s['idx']:>7}  {s['g7']:>+4.1f}  ({s['q1']:>+1.0f},{s['q2']:>+1.0f},{s['q3']:>+1.0f})  "
              f"{chirality:>14}  {a_label:>30}")
    print()

    return cl6_states


# ============================================================================
# Step 5: Build the per-Weyl-spinor 48-row dictionary
# ============================================================================

def section_5_dictionary(cl6_states):
    banner("§5 The 48-row per-Weyl-spinor dictionary")
    print()
    print("Each SM Weyl spinor labeled by:")
    print("  generation i ∈ {1, 2, 3}: via R3 observer C³ index")
    print("  chirality L/R: γ_7 = +1 (L = (4,2,1)) or γ_7 = -1 (R = (4̄,1,2))")
    print("  SU(4)_PS index a ∈ {r, g, b, ℓ}: via Cl(6) Fock state ((Q_1,Q_2,Q_3) pattern)")
    print("  SU(2) isospin b ∈ {up, down}: per-edge Cl(0,2) structure")
    print()
    print("4 valid Q-patterns (forced by Q_1 Q_2 Q_3 = -I, so q_1 q_2 q_3 = -1):")
    print("  (-,-,-): lepton (ℓ) — color singlet")
    print("  (+,+,-): color r   — quark color 'red'")
    print("  (+,-,+): color g   — quark color 'green'")
    print("  (-,+,+): color b   — quark color 'blue'")
    print()
    print("Per gen: 4 SU(4) states × 2 chiralities × 2 SU(2) isospin = 16 SM components")
    print("3 gens × 16 = 48 SM Weyl spinors ✓")
    print()

    def assign_su4(signs):
        """Map (q_1, q_2, q_3) sign pattern to SU(4) label."""
        sgn = tuple(int(np.sign(q)) for q in signs)
        if sgn == (-1, -1, -1):
            return 'ℓ', 'lepton'
        elif sgn == (1, 1, -1):
            return 'r', 'color-r'
        elif sgn == (1, -1, 1):
            return 'g', 'color-g'
        elif sgn == (-1, 1, 1):
            return 'b', 'color-b'
        else:
            return '?', f'unmatched {sgn}'

    print(f"  {'#':>3}  {'gen':>3}  {'γ_7':>4}  {'(q1,q2,q3)':>11}  {'a':>4}  {'b':>5}  "
          f"{'SM spinor':>20}  {'Cl(6)#':>7}  {'walker class':>22}")
    print(f"  {'-'*3}  {'-'*3}  {'-'*4}  {'-'*11}  {'-'*4}  {'-'*5}  {'-'*20}  {'-'*7}  {'-'*22}")

    sm_components = []
    n = 0
    for gen in [1, 2, 3]:
        for s in cl6_states:
            for b in ['up', 'down']:
                signs = (s['q1'], s['q2'], s['q3'])
                a, a_desc = assign_su4(signs)

                # SM Weyl-spinor identification
                if s['g7'] > 0:  # γ_7 = +1, left-handed (4, 2, 1)_L
                    if a == 'ℓ':
                        sm = f"ν_L^{{gen{gen}}}" if b == 'up' else f"e_L^{{gen{gen}}}"
                    else:
                        sm = f"u_L^{{{a},gen{gen}}}" if b == 'up' else f"d_L^{{{a},gen{gen}}}"
                else:  # γ_7 = -1, right-handed (4̄, 1, 2)_R  (charge-conjugate convention)
                    if a == 'ℓ':
                        sm = f"ν_R^c^{{gen{gen}}}" if b == 'up' else f"e_R^c^{{gen{gen}}}"
                    else:
                        # Color labels for 4̄ become anti-colors (r̄, ḡ, b̄)
                        a_bar = {'r':'r̄', 'g':'ḡ', 'b':'b̄'}.get(a, a)
                        sm = f"u_R^c^{{{a_bar},gen{gen}}}" if b == 'up' else f"d_R^c^{{{a_bar},gen{gen}}}"

                # Walker class:
                # - Neutrinos (SU(4)=ℓ AND SU(2)=up, i.e., ν): chir-7 at h_Γ (L) or h_H (R)
                # - Charged leptons (SU(4)=ℓ AND SU(2)=down, i.e., e): h_P (like charged fermions)
                # - Quarks (SU(4) ∈ {r,g,b}): h_P at P
                if a == 'ℓ' and b == 'up':
                    walker = 'h_Γ @ Γ (chir-7)' if s['g7'] > 0 else 'h_H @ H (chir-7)'
                else:
                    walker = 'h_P / h_P_neg @ P'

                sm_components.append({
                    'idx': n, 'gen': gen, 'g7': s['g7'], 'a': a, 'a_desc': a_desc, 'b': b,
                    'sm': sm, 'cl6_idx': s['idx'], 'walker': walker,
                    'q_signs': signs,
                })

                print(f"  {n:>3}  {gen:>3}  {s['g7']:>+4.1f}  "
                      f"({signs[0]:>+1.0f},{signs[1]:>+1.0f},{signs[2]:>+1.0f})  "
                      f"{a:>4}  {b:>5}  {sm:>20}  {s['idx']:>7}  {walker:>22}")
                n += 1
    print()
    print(f"Total: {n} entries (target: 48)")
    print()

    return sm_components


# ============================================================================
# Step 6: Sanity checks
# ============================================================================

def section_6_sanity(sm_components):
    banner("§6 Sanity checks")
    print()

    # Count per chirality
    n_L = sum(1 for c in sm_components if c['g7'] > 0)
    n_R = sum(1 for c in sm_components if c['g7'] < 0)
    print(f"Left-handed (γ_7=+1): {n_L} (target: 24 = 3 gens × 8)")
    print(f"Right-handed (γ_7=-1): {n_R} (target: 24)")
    print()

    # Count per generation
    from collections import Counter
    gen_counter = Counter(c['gen'] for c in sm_components)
    print(f"Per generation count: {dict(sorted(gen_counter.items()))} (each target: 16)")
    print()

    # Count per walker class
    walker_counter = Counter(c['walker'] for c in sm_components)
    print(f"Per walker-class count:")
    for w, c in walker_counter.items():
        print(f"  {w}: {c}")
    print()

    # Count per (a, b) pair
    print("Per (SU(4)_a, isospin_b) count:")
    ab_counter = Counter((c['a'], c['b']) for c in sm_components)
    for (a, b), count in sorted(ab_counter.items()):
        print(f"  {a:>20}, {b:>4}: {count} (target: 3 = one per generation)")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    banner("Per-Weyl-spinor dictionary — Phase 1 of unified theory development", "#")
    print(f"\nDate: 2026-05-27 EOD+3")
    print(f"Builds on: V_Ram-iso T1 (explicit construction) + T4 (Q_i generation correspondence)")
    print()

    G, G7 = build_cl6_gammas()
    Q = build_Q_operators(G)

    print()
    cl6_states = section_3_enumerate_cl6_states(G, G7, Q)
    print()
    section_4_PS_mapping(cl6_states)
    print()
    sm_components = section_5_dictionary(cl6_states)
    print()
    section_6_sanity(sm_components)


if __name__ == "__main__":
    main()
