#!/usr/bin/env python3
"""
σ_S compatibility with Q_Koide — does substituting σ_S for C₃_body in the
Born-rule Fourier derivation preserve Q = 2/3?

CONTEXT
-------
`predictions/Q_Koide.py` derives Q = 2/3 via:

  Step 4 (ADOPTED-J): the three C₃-Fourier outputs (j ∈ {0, 1, 2}) of the
                       (4, 2, 2) isotypic decomposition of V_Ram(P) under
                       C₃_body are matched to three generation labels.
                       This is a dimensional-matching adoption.

  Step 5: amp_j = √4 + √2·ω^j + √2·ω^{-j} = 2 + 2√2 cos(2πj/3)
                  giving (amp_0, amp_1, amp_2) = (2 + 2√2, 2 - √2, 2 - √2).

  Step 6 (A3 + CDP 2011): m_j = |amp_j|².
  Step 7: Σ m_j = 24, Σ √m_j = 6, Q = 24/36 = 2/3.

The choke-point work (2026-04-25) identified σ_S as a candidate for the
generation Z_3 (distinct from C₃_body which labels color per session 25
sin²θ_W). σ_S has the same isotypic (4, 2, 2) on V_Ram(P) ≅ S, but is a
different element of Spin(6) (they generate 2I = SL(2, 5)).

QUESTION
--------
If σ_S replaces C₃_body in the Q_Koide step 5 Fourier decomposition,
does the predicted Q remain 2/3?

  (A) Yes — σ_S substitution is invariant under the algebra (only (4,2,2)
      multiplicities matter). σ_S = generation candidate is COMPATIBLE
      with Q_Koide.
  (B) No — σ_S gives a different decomposition that breaks Q.

If (A), substitution is consistent but does NOT automatically derive
ADOPTED-J: an independent argument that σ_S labels generations is still
needed.

COMPUTATION
-----------
1. Build V_Ram(P), C₃_body | V_Ram, σ_S transported to V_Ram (via
   gamma7_chirality intertwiner).
2. Decompose V_Ram(P) into σ_S-isotypic (4, 2, 2) — confirm same dims.
3. Compute amp_j_σS = Σ_α √μ_α(σ_S) · ω^{jα} for j = 0, 1, 2.
4. Compute Q_σS = Σ |amp_j|² / (Σ |amp_j|)².
5. Same for C₃_body as a control.
6. Verify Q_σS = Q_C₃body = 2/3.

VERDICT (filled by run output)
------------------------------
[Reported by run.]

Run with:
    PYTHONPATH=. python3 proofs/foundations/sigma_S_Q_koide_compat.py

Upstream:
    proofs/foundations/sigma_S_phase_walks.py
    proofs/foundations/gamma7_chirality.py
    predictions/Q_Koide.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
)
from proofs.foundations.matching_brauer_weyl_sigma import (
    brauer_weyl_gammas,
    build_sigma_S,
    sigma_permutation_on_gammas,
)
from proofs.foundations.gamma7_chirality import (
    build_U_C3_S,
    c3_isotypic_basis,
    classify_c3,
)


TOL = 1e-8
K_P = (0.25, 0.25, 0.25)
omega = np.exp(2j * np.pi / 3)


def banner(s):
    print()
    print("=" * 76)
    print(s)
    print("=" * 76)


def compute_Q_from_isotypic_dims(mu_trivial, mu_omega, mu_omega_bar):
    """Q_Koide formula given (4,2,2) isotypic multiplicities.

    amp_j = √μ_0 + √μ_1·ω^j + √μ_2·ω^{-j}
    m_j = |amp_j|²
    Q = (Σ m_j) / (Σ √m_j)²

    Per Q_Koide.py step 5, with (μ_0, μ_1, μ_2) = (4, 2, 2):
        amp_0 = 2 + 2√2
        amp_1 = amp_2 = 2 - √2
        m_j = (2 + 2√2)², (2 - √2)², (2 - √2)²
        Q = (sum m_j)/(sum √m_j)² = 2/3
    """
    s0 = math.sqrt(mu_trivial)
    s1 = math.sqrt(mu_omega)
    s2 = math.sqrt(mu_omega_bar)
    amps = []
    for j in range(3):
        amp = s0 + s1 * (omega ** j) + s2 * (omega ** (-j))
        amps.append(amp)
    m = [abs(a) ** 2 for a in amps]
    sqrt_m = [math.sqrt(mi) for mi in m]
    Q = sum(m) / (sum(sqrt_m) ** 2)
    return Q, m, amps


def main():
    banner("σ_S compatibility with Q_Koide — Born-rule Fourier substitution test")

    # ---- Build infrastructure -----------------------------------------
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    B_P = bloch_hashimoto(K_P, directed)
    U_C3_12 = build_c3_on_directed_edges(directed)

    # Extract V_Ram(P)
    evals, evecs = la.eig(B_P)
    ram_idx = [i for i, ev in enumerate(evals) if abs(abs(ev) ** 2 - 2.0) < 1e-5]
    assert len(ram_idx) == 8
    V_Ram, _ = la.qr(evecs[:, ram_idx])
    V_Ram = V_Ram[:, :8]
    U_C3_VRam = V_Ram.conj().T @ U_C3_12 @ V_Ram

    # Build σ_S on Cl(6,0) spinor and transport to V_Ram via C₃-intertwiner
    Gs = brauer_weyl_gammas()
    perm = sigma_permutation_on_gammas()
    sigma_S = build_sigma_S(Gs, perm)
    U_C3_S = build_U_C3_S(directed)

    bases_S = c3_isotypic_basis(U_C3_S)
    bases_VR = c3_isotypic_basis(U_C3_VRam)
    A = np.zeros((8, 8), dtype=complex)
    for label in ['1', 'w', 'w2']:
        A += bases_VR[label] @ bases_S[label].conj().T
    sigma_S_V = A @ sigma_S @ A.conj().T

    # ---- Decompose V_Ram under each C₃ -------------------------------
    print("\nC₃_body isotypic dims on V_Ram(P):")
    bases_C3body = c3_isotypic_basis(U_C3_VRam)
    dims_C3body = (
        bases_C3body['1'].shape[1],
        bases_C3body['w'].shape[1],
        bases_C3body['w2'].shape[1],
    )
    print(f"  (m_1, m_ω, m_ω²) = {dims_C3body}")
    assert dims_C3body == (4, 2, 2)

    print("\nσ_S^V isotypic dims on V_Ram(P):")
    bases_sV = c3_isotypic_basis(sigma_S_V)
    dims_sV = (
        bases_sV['1'].shape[1],
        bases_sV['w'].shape[1],
        bases_sV['w2'].shape[1],
    )
    print(f"  (m_1, m_ω, m_ω²) = {dims_sV}")
    assert dims_sV == (4, 2, 2)

    # ---- Compute Q under each Fourier choice -------------------------
    print()
    print("Q_Koide formula via (4, 2, 2) Fourier decomposition:")
    print("  amp_j = √4 + √2·ω^j + √2·ω^{-j}")
    print("        = 2 + 2√2·cos(2πj/3)")
    print()

    Q_C3, m_C3, amps_C3 = compute_Q_from_isotypic_dims(*dims_C3body)
    print(f"  Using C₃_body isotypic (4, 2, 2):")
    print(f"    amp = {[f'{a.real:+.4f}{a.imag:+.4f}i' for a in amps_C3]}")
    print(f"    m_j = {[f'{mi:.4f}' for mi in m_C3]}")
    print(f"    Q   = {Q_C3:.10f}   (expected 2/3 = {2/3:.10f})")
    print()

    Q_sV, m_sV, amps_sV = compute_Q_from_isotypic_dims(*dims_sV)
    print(f"  Using σ_S^V isotypic (4, 2, 2):")
    print(f"    amp = {[f'{a.real:+.4f}{a.imag:+.4f}i' for a in amps_sV]}")
    print(f"    m_j = {[f'{mi:.4f}' for mi in m_sV]}")
    print(f"    Q   = {Q_sV:.10f}   (expected 2/3 = {2/3:.10f})")
    print()

    # ---- Verify invariance --------------------------------------------
    diff = abs(Q_C3 - Q_sV)
    print(f"  |Q_C₃body - Q_σS^V| = {diff:.2e}   (expected 0 — same algebra)")
    assert diff < 1e-12

    # ---- Subtler test: do the underlying amplitudes match? ------------
    # The Q value invariance is purely combinatorial in (4,2,2) and gives
    # the same numerical answer regardless of which C₃ is used.
    # The actual EIGENSTATES corresponding to "trivial sector" differ
    # between C₃_body and σ_S, so the *physical* identification of
    # "j=0 generation" with specific states differs.

    # Compute overlap between σ_S^V trivial-sector eigenspace and
    # C₃_body trivial-sector eigenspace.
    print()
    print("Subtler test: how much do the σ_S^V and C₃_body isotypic decompositions")
    print("of V_Ram(P) overlap? (If 100%, σ_S^V = C₃_body restricted; if not,")
    print("the 'generation labels' are physically different states.)")
    print()
    for label in ['1', 'w', 'w2']:
        Q_a = bases_C3body[label]
        Q_b = bases_sV[label]
        # Frobenius norm of overlap matrix: ||Q_a^† Q_b||_F²/min(dim_a, dim_b)
        # = average squared overlap.
        overlap_matrix = Q_a.conj().T @ Q_b
        avg_overlap_sq = la.norm(overlap_matrix) ** 2 / min(Q_a.shape[1], Q_b.shape[1])
        print(f"  Sector {label}: avg |⟨C₃_body | σ_S^V⟩|² = {avg_overlap_sq:.4f}")

    # ---- Verdict ------------------------------------------------------
    banner("VERDICT")

    print()
    print(f"σ_S substitution preserves Q_Koide: Q_σS^V = Q_C₃body = 2/3 ✓")
    print()
    print(f"This is INVARIANT under the C₃ choice: only the multiplicities")
    print(f"(4, 2, 2) matter for the Fourier algebra, and both C₃_body and")
    print(f"σ_S_V have those multiplicities (same Spin(6) conjugacy class).")
    print()
    print(f"What this DOES tell us:")
    print(f"  σ_S = generation hypothesis is COMPATIBLE with Q_Koide's")
    print(f"  Q = 2/3 prediction. Substituting σ_S for C₃_body in the")
    print(f"  Fourier step gives the same numerical answer.")
    print()
    print(f"What this does NOT tell us:")
    print(f"  σ_S = generation is NOT derived just because the substitution")
    print(f"  works. The (4, 2, 2) algebra is invariant under any C₃ in the")
    print(f"  same Spin(6) conjugacy class. To DERIVE ADOPTED-J (the j ↔")
    print(f"  generation matching), we still need an independent structural")
    print(f"  argument that σ_S (or C₃_body, or some other C₃) is THE")
    print(f"  generation Z₃.")
    print()
    print(f"Bottom line: Q_Koide's adopted residual J survives unchanged.")
    print(f"σ_S = generation candidate clears the Q_Koide compatibility check")
    print(f"but doesn't automatically promote ADOPTED-J to derived.")

    return {
        'Q_C3body': Q_C3,
        'Q_sigma_S_V': Q_sV,
        'invariant': diff < 1e-12,
    }


if __name__ == "__main__":
    main()
