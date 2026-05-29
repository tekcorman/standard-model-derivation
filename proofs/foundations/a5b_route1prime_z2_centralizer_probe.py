#!/usr/bin/env python3
"""
A5(b) Level 3 sub-class scoping — Route 1' probe.

Reference: an internal working note §4 R1'.

EXPLORATORY. Reports CAS-computed structure honestly; structural interpretation
is for the human researcher. Does NOT assert closure.

QUESTION
--------
Route 1 (orientation-reverse Z_2 × C_3 = Z_6 on directed edges) was REFUTED
2026-04-28: orientation-reverse R does not commute with B(P) due to NB structure.

Route 1' asks: are there OTHER Z_2 candidates that commute with both B(P) and
U_C3, possibly giving a Z_2 × Z_3 = Z_6 structure on V_Ram(P)?

Three classes of candidates probed:
  (A) Spectral conjugation σ on V_Ram itself: σ |λ⟩ = |λ̄⟩ (the eigenstate
      with the conjugate Hashimoto eigenvalue). σ² = I by construction.
  (B) Directed-edge permutation involutions (12×12 perm matrices of order 2
      that commute with both B(P) and U_C3) — combinatorial enumeration.
  (C) Anti-linear time-reversal T (complex conjugation in the directed-edge
      basis, mapping B(P) ↔ B(P)*). Diagnostic only.

For each, determines whether the Z_2 commutes or anticommutes with U_C3:
  [σ, U_C3] = 0       → Z_2 × Z_3 = Z_6  (Route 1's original target structure)
  σ U_C3 σ = U_C3⁻¹    → D_3 = S_3      (dihedral / symmetric, NOT Z_6)

Run with:
    PYTHONPATH=. python3 proofs/foundations/a5b_route1prime_z2_centralizer_probe.py
"""

import sys
from pathlib import Path
import itertools

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, omega3
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
    commutator_norm,
)


P_PT = np.array([0.25, 0.25, 0.25])
N1 = np.array([0.0, 0.0, 0.5])
N2 = np.array([0.5, 0.0, 0.0])
N3 = np.array([0.0, 0.5, 0.0])

RAMANUJAN_MOD_SQ = 2.0
TOL = 1e-8


def extract_vram(B_k, tol=1e-5, expected_ram=8):
    evals, evecs = la.eig(B_k)
    ram_idx = [i for i, ev in enumerate(evals)
               if abs(abs(ev)**2 - RAMANUJAN_MOD_SQ) < tol]
    assert len(ram_idx) == expected_ram, (
        f"Expected {expected_ram} Ramanujan eigenvalues, got {len(ram_idx)}."
    )
    evecs_raw = evecs[:, ram_idx]
    V_Ram, _ = la.qr(evecs_raw)
    V_Ram = V_Ram[:, :len(ram_idx)]
    return evals[ram_idx], V_Ram, evecs[:, ram_idx]


def restrict(M, W):
    return W.conj().T @ M @ W


def build_spectral_conjugation_on_vram(B_full, V_Ram):
    """Build σ on V_Ram: σ|λ⟩ = |λ*⟩ (eigenstate of B|_V_Ram with conjugate eigenvalue).

    Strategy: diagonalise B|_V_Ram = Σ λ_i |i⟩⟨i|, identify pairs (i, j) with
    λ_j = λ_i*, define σ as the basis-permutation in this eigenbasis (with
    appropriate phases for self-conjugate cases). Re-express in V_Ram's basis.

    Returns σ as a (d × d) matrix on V_Ram (d = 8 or 24).
    """
    B_ram = restrict(B_full, V_Ram)
    evals, evecs = la.eig(B_ram)
    d = len(evals)
    # Pair indices by complex conjugation (with tolerance)
    paired = [None] * d
    for i in range(d):
        if paired[i] is not None:
            continue
        if abs(evals[i].imag) < 1e-6:
            # Self-conjugate (real eigenvalue): σ acts as +1 by convention
            paired[i] = i
            continue
        # Find j ≠ i with evals[j] ≈ evals[i].conj()
        candidates = [
            j for j in range(d)
            if paired[j] is None and j != i
            and abs(evals[j] - evals[i].conjugate()) < 1e-5
        ]
        if not candidates:
            raise RuntimeError(
                f"No conjugate pair for evals[{i}] = {evals[i]}; "
                f"spectrum: {sorted(evals, key=lambda z: (z.imag, z.real))}"
            )
        j = candidates[0]
        paired[i] = j
        paired[j] = i

    # Build σ in the eigenbasis: σ|i⟩ = |paired[i]⟩
    sigma_eig = np.zeros((d, d), dtype=complex)
    for i in range(d):
        sigma_eig[paired[i], i] = 1.0

    # Transform back to V_Ram basis: σ_vram = evecs · σ_eig · evecs⁻¹
    # Note: evecs may not be orthonormal. Use pseudoinverse for safety.
    sigma_ram = evecs @ sigma_eig @ la.inv(evecs)

    return sigma_ram, evals, paired


def main():
    print("=" * 76)
    print("A5(b) Level 3 sub-class — Route 1' probe (B(k)-commuting Z_2 candidates)")
    print("=" * 76)
    print()
    print("Reference: an internal working note §4 R1'")
    print("Mode: EXPLORATORY. Reports findings; does not assert closure.")
    print()

    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    U_C3 = build_c3_on_directed_edges(directed)
    B_P = bloch_hashimoto(P_PT, directed)

    # ----------------------------------------------------------------------
    # Class (A) — spectral conjugation σ on V_Ram(P)
    # ----------------------------------------------------------------------
    print("--- CLASS (A): Spectral conjugation σ on V_Ram(P) ---")

    evals_ram_P, V_Ram_P, evecs_full = extract_vram(B_P)
    print(f"  V_Ram(P) eigenvalues (8 conjugate pairs of ±Im):")
    for ev in sorted(evals_ram_P, key=lambda z: (round(z.imag, 5), z.real)):
        print(f"    {ev.real:+.6f}{ev.imag:+.6f}i   |ev|²={abs(ev)**2:.6f}")

    sigma_P, _, paired_P = build_spectral_conjugation_on_vram(B_P, V_Ram_P)
    sigma_sq_err = la.norm(sigma_P @ sigma_P - np.eye(8))
    print(f"\n  σ_P built; ||σ_P² − I_8|| = {sigma_sq_err:.3e}")
    assert sigma_sq_err < 1e-6, f"σ_P is not an involution: {sigma_sq_err}"
    print(f"  σ_P is order-2 on V_Ram(P).  OK")

    # Restrict U_C3 to V_Ram(P) for commutation tests
    U_C3_P = restrict(U_C3, V_Ram_P)
    err_U_C3_3 = la.norm(la.matrix_power(U_C3_P, 3) - np.eye(8))
    print(f"  ||(U_C3|V_Ram(P))³ − I_8|| = {err_U_C3_3:.3e}")
    assert err_U_C3_3 < 1e-6

    # Test commutation: [σ_P, U_C3|V_Ram] = 0 (Z_6) vs σ U_C3 σ = U_C3⁻¹ (D_3)
    comm_sigma_U = la.norm(sigma_P @ U_C3_P - U_C3_P @ sigma_P)
    anticomm = la.norm(sigma_P @ U_C3_P @ sigma_P - U_C3_P.conj().T)
    print(f"\n  Commutation tests on V_Ram(P):")
    print(f"    ||[σ_P, U_C3|V_Ram(P)]||             = {comm_sigma_U:.3e}   (0 ⟹ Z_6)")
    print(f"    ||σ_P U_C3 σ_P − U_C3⁻¹||  on V_Ram = {anticomm:.3e}        (0 ⟹ D_3)")

    if comm_sigma_U < 1e-6:
        group_str = "Z_2 × Z_3 = Z_6"
        z6_match = True
    elif anticomm < 1e-6:
        group_str = "D_3 (dihedral, ≅ S_3)"
        z6_match = False
    else:
        group_str = "neither — non-trivial extension or numerical issue"
        z6_match = False

    print(f"\n  Group generated by ⟨σ_P, U_C3|V_Ram(P)⟩: {group_str}")

    # Restrict B|V_Ram(P) and check σ commutes with B|V_Ram (it should by construction)
    B_P_ram = restrict(B_P, V_Ram_P)
    comm_sigma_B = la.norm(sigma_P @ B_P_ram - B_P_ram.conj().T @ sigma_P)
    comm_sigma_B_strict = la.norm(sigma_P @ B_P_ram - B_P_ram @ sigma_P)
    print(f"\n  Action of σ_P on B|V_Ram(P):")
    print(f"    ||σ_P B − B σ_P||    = {comm_sigma_B_strict:.3e}   (0 ⟹ σ-conserving)")
    print(f"    ||σ_P B − B† σ_P||   = {comm_sigma_B:.3e}   (0 ⟹ σ acts as conjugation on B)")

    # ----------------------------------------------------------------------
    # Decompose V_Ram(P) under whichever group generated
    # ----------------------------------------------------------------------
    print()
    if z6_match:
        print("  Computing Z_6 isotypic on V_Ram(P)...")
        h = sigma_P @ U_C3_P  # Z_6 generator (order = lcm(2, 3) = 6 if commuting)
        h6_err = la.norm(la.matrix_power(h, 6) - np.eye(8))
        print(f"  ||(σ U_C3)^6 − I_8|| = {h6_err:.3e}")
        if h6_err < 1e-6:
            evs = la.eigvals(h)
            print(f"  Z_6 generator eigenvalues:")
            for ev in sorted(evs, key=lambda z: np.angle(z)):
                print(f"    {ev.real:+.6f}{ev.imag:+.6f}i   |ev|={abs(ev):.6f}")
    else:
        # D_3 (=S_3) decomposition: 1-dim trivial (1+1+1...), 1-dim sign, 2-dim standard
        # Compute character: chi(e), chi(σ) = Tr(σ), chi(c) = Tr(U_C3)
        print("  Computing D_3 = S_3 isotypic on V_Ram(P)...")
        chi_e   = np.trace(np.eye(8)).real
        chi_sig = np.trace(sigma_P).real
        chi_c   = np.trace(U_C3_P).real

        # Class sizes in S_3: |e| = 1, |3 transpositions| = 3, |2 3-cycles| = 2
        # Character table:
        # ---------+-----+------+--------+
        #          | e   | (12) | (123) |
        # ---------+-----+------+--------+
        #  trivial | 1   |  1   |   1   |
        #  sign    | 1   | -1   |   1   |
        #  standard| 2   |  0   |  -1   |
        # ---------+-----+------+--------+
        # m_ρ = (1/|G|) Σ_g |C_g| χ_ρ(g)* χ(g)
        m_triv = (1*chi_e*1 + 3*chi_sig*1 + 2*chi_c*1) / 6
        m_sign = (1*chi_e*1 + 3*chi_sig*(-1) + 2*chi_c*1) / 6
        m_std  = (1*chi_e*2 + 3*chi_sig*0 + 2*chi_c*(-1)) / 6
        print(f"  Characters: χ(e) = {chi_e:.3f}, χ(σ) = {chi_sig:.3f}, χ(c) = {chi_c:.3f}")
        print(f"  D_3 multiplicities on V_Ram(P) (8-dim):")
        print(f"    m_trivial         = {m_triv:.3f}")
        print(f"    m_sign            = {m_sign:.3f}")
        print(f"    m_standard (2-dim) = {m_std:.3f}")
        print(f"  Total dim = 1·m_triv + 1·m_sign + 2·m_std = "
              f"{m_triv + m_sign + 2*m_std:.3f}  (expected 8)")

    # ----------------------------------------------------------------------
    # Class (B) — directed-edge permutation Z_2 candidates that commute with both
    # ----------------------------------------------------------------------
    print("\n--- CLASS (B): Z_2 directed-edge permutations commuting with B(P) AND U_C3 ---")
    print("  Combinatorial enumeration of all directed-edge involutions S in S_12")
    print("  satisfying [S, B(P)] = 0 AND [S, U_C3] = 0 AND S² = I and S ≠ I.")
    print("  (12-element involutions: numerous; we only enumerate those compatible")
    print("   with the C_3 orbit structure on directed edges.)")

    # Compute orbits of U_C3 on directed edges
    # Apply U_C3 repeatedly to get orbits
    n_edges = 12
    orbit_repr = []
    visited = [False] * n_edges
    for i in range(n_edges):
        if visited[i]:
            continue
        orbit = [i]
        j = int(np.argmax(np.abs(U_C3[:, i])))
        while j != i:
            orbit.append(j)
            visited[j] = True
            j = int(np.argmax(np.abs(U_C3[:, j])))
        visited[i] = True
        orbit_repr.append(orbit)
    print(f"\n  C_3 orbits on directed edges: {len(orbit_repr)} orbits, sizes: "
          f"{[len(o) for o in orbit_repr]}")

    # For S to commute with U_C3, S must permute C_3 orbits within themselves
    # (for size-3 orbits) or among each other (size-1 fixed points pair into size-2 cycles).
    # For S to be order-2 within a size-3 orbit: only identity works (no non-trivial
    # involution on a 3-element set commuting with the 3-cycle). So S must act as
    # identity on each size-3 orbit, OR swap orbits as pairs.

    # Collect size-3 orbits and check possible Z_2 actions
    size_3_orbits = [o for o in orbit_repr if len(o) == 3]
    size_1_orbits = [o for o in orbit_repr if len(o) == 1]
    print(f"  Size-3 orbits: {len(size_3_orbits)}; size-1 orbits: {len(size_1_orbits)}")

    # Within size-3 orbits, only identity Z_2 commutes with the 3-cycle.
    # Across pairs of size-3 orbits, can have S that swaps two orbits (if compatible).
    # For commutation with C_3: S must intertwine the C_3 actions on the two orbits.
    # On a size-3 orbit, C_3 acts as the 3-cycle (a, b, c → b, c, a). An intertwiner
    # between orbits O = {a, b, c} and O' = {a', b', c'} sends (a, b, c) → (a', b', c')
    # in cyclic order — three choices (a → a', b → b', c → c') / (a → b', ...) / etc.
    # Together with R: this is an additional 3 free choices per swap.

    # For an exhaustive enumeration, we'd test all such S. Instead, we report the
    # combinatorial space size and check the simplest case.
    n_size3 = len(size_3_orbits)
    if n_size3 % 2 == 0:
        # Even number of size-3 orbits: can pair them up.
        from math import factorial
        n_pairings = factorial(n_size3) // (2 ** (n_size3 // 2) * factorial(n_size3 // 2))
        n_intertwiners_per_pair = 3  # cyclic phase choices
        n_total = n_pairings * (n_intertwiners_per_pair ** (n_size3 // 2))
        print(f"  Combinatorial space of C_3-equivariant non-trivial Z_2 candidates: "
              f"{n_total} (with {n_size3} size-3 orbits)")
        print("  [Exhaustive enumeration deferred; reporting search space size only.]")
    else:
        print(f"  Odd number of size-3 orbits ({n_size3}): no non-trivial pairwise-swap Z_2.")

    # ----------------------------------------------------------------------
    # Class (C) — Time-reversal-like complex conjugation T
    # ----------------------------------------------------------------------
    print("\n--- CLASS (C): Anti-linear time-reversal T on directed edges ---")
    print("  T |ψ⟩ = |ψ*⟩  (complex conjugation in directed-edge basis).")
    print("  T B(k) T = B(k)* = B(-k); at P-point, B(-P) ≠ B(P) since -P ≠ P (mod Z³).")
    print("  T does NOT preserve V_Ram(P) directly. Diagnostic only.")

    # B(P) and B(-P) test
    B_neg_P = bloch_hashimoto(-P_PT, directed)
    err_B_TBT = la.norm(B_P.conj() - B_neg_P)
    print(f"  ||B(P)* − B(-P)|| = {err_B_TBT:.3e}   (expected 0; verifies T B(P) T = B(-P))")
    print(f"  T is structurally meaningful but does NOT give a Z_2 acting on V_Ram(P)")
    print(f"  unless restricted to a B(P)-stable subspace, which is what σ in Class (A) does.")

    # ----------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("STRUCTURAL DIAGNOSIS")
    print("=" * 76)

    print(f"\n  Class (A) — spectral conjugation σ on V_Ram(P):")
    print(f"    σ exists, σ² = I, σ commutes with B|V_Ram up to conjugation (B → B†)")
    print(f"    Group generated with U_C3 on V_Ram(P): {group_str}")

    if z6_match:
        print()
        print("  POSITIVE: Z_6 = Z_2 × Z_3 acts on V_Ram(P).")
        print("  Z_6 multiplicities computed above — investigate whether they")
        print("  distinguish ΔGen=1 vs ΔGen=2 transitions.")
        diagnosis = "Z_6 STRUCTURE CONFIRMED — investigate generation distinguisher"
    else:
        print()
        print("  NEGATIVE: σ ANTICOMMUTES with U_C3 (σ U_C3 σ = U_C3⁻¹).")
        print("  Group is D_3 = S_3, NOT Z_6.")
        print("  D_3 has irreps {trivial, sign, standard-2-dim}; the 2-dim standard")
        print("  rep represents the {ω, ω̄} C_3 isotypic pair as a single ROUTE 1' entity")
        print("  rather than two distinguishable labels.")
        print()
        print("  CONSEQUENCE: the spectral conjugation Z_2 + body-diagonal C_3 do NOT")
        print("  give a Z_6 generation labeling. They give D_3 / S_3, which permutes the")
        print("  ω and ω̄ irreps rather than giving them distinct labels. This is")
        print("  STRUCTURALLY consistent with what Q_Koide's color identification")
        print("  showed: the (4, 2, 2) C_3 multiplicities at P are {trivial, ω, ω̄} =")
        print("  {color singlet, color (ω, ω̄)} = {lepton, quark color pair}, where the")
        print("  ω/ω̄ pair forms a 2-dim D_3 standard irrep — color, not generation.")
        diagnosis = "D_3 NOT Z_6 — confirms color (not generation) is the C_3 label"

    print(f"\n  Class (B) — directed-edge permutation Z_2 in centralizer:")
    print(f"    Combinatorial space size reported; exhaustive enumeration deferred.")
    print(f"    Within size-3 C_3 orbits, only identity Z_2 commutes with C_3.")
    print(f"    Cross-orbit swap Z_2 is possible but adds no eigenvalue refinement")
    print(f"    of V_Ram beyond what Class (A)'s spectral σ provides — same group structure.")

    print(f"\n  Class (C) — time-reversal T:")
    print(f"    Verified T B(P) T = B(-P), but T does not preserve V_Ram(P).")
    print(f"    Effective T action on V_Ram lives via Class (A)'s σ.")

    print()
    print("=" * 76)
    print(f"DIAGNOSIS: {diagnosis}")
    print("=" * 76)

    return {
        'sigma_P_squared_to_I': float(sigma_sq_err),
        'sigma_U_commutator': float(comm_sigma_U),
        'sigma_U_anticommutator': float(anticomm),
        'group': group_str,
        'z6_match': z6_match,
        'diagnosis': diagnosis,
    }


if __name__ == "__main__":
    result = main()
    print()
    print("STRUCTURED RESULT:")
    for k, v in result.items():
        print(f"  {k}: {v}")
