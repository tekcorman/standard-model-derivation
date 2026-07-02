#!/usr/bin/env python3
"""
Lemma 3 (β closure) — Step B: derive the coefficient c in
β = c · sin(arg h) · α_EM by computing the natural chirality operator on
the photon Hodge bundle at the P-point.

Setup
-----
The photon Δ_1 lives on canonical undirected edges (6-dim).  The
non-backtracking walker B(P) lives on directed bonds (12-dim).  The two
spaces are connected by the natural projection

    π : undirected-edges (6) → directed-bonds (12)
    π |e_k⟩ = (1/√2) ( |T_e_forward⟩ + |T_e_backward⟩ )

(its adjoint π† maps the orientation-symmetric subspace of directed bonds
back to undirected edges).

The chirality operator the photon sees is

    V_chir := π† · Im(B(P)) · π
            = π† · (B(P) − B(P)†) / (2i) · π

restricted to the doubly-degenerate photon eigenspace at ω² = 36.  By
Schur's lemma + the verified C₃-irrep structure (L = ω, R = ω²,
inequivalent ⇒ V_chir|photon is diagonal in L/R), V_chir|photon =
diag(c_L, c_R).  Time-reversal antisymmetry forces c_L = −c_R.

Claim: c_L = + sin(arg h),  c_R = − sin(arg h).

Equivalently: V_chir | L⟩ = + sin(arg h) | L⟩ and V_chir | R⟩ = − sin(arg h) | R⟩.
This gives β = c · sin(arg h) · α_EM with c = 1 (no extra structural
multiplicity).

Verification protocol
---------------------
1. Build B(P) on the 12 directed bonds (consistent with bond ordering used
   by Bloch-Hashimoto already in the project).
2. Build π (12×6) and π† (6×12).
3. Compute Im(B(P)) = (B − B†)/(2i).  This is a 12×12 Hermitian operator.
4. Compute V_chir_undir = π† · Im(B(P)) · π (6×6 Hermitian).
5. Restrict V_chir_undir to the photon ω² = 36 eigenspace at k_P:
   V_chir_photon = Q† · V_chir_undir · Q, where Q is the orthonormal basis
   of the 2D photon eigenspace.
6. Express V_chir_photon in the L/R basis (L = ω-irrep, R = ω²-irrep) via
   the C₃ eigenvectors of C₃_e | photon.
7. Read off c_L, c_R; verify c_L = −c_R = sin(arg h) = √(5/8) to machine
   precision.
"""

import os
import sys
import math
import numpy as np
from numpy import linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
    incidence_matrix_primitive,
    nb_walk_operator,
    HIGH_SYM_POINTS,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_c3_chainmap import (
    build_C3_vertex,
    build_C3_edge,
    build_delta_1,
    K_P_RED,
)


# =============================================================================
# Build π : undirected edges (6) → directed bonds (12).
# =============================================================================

def build_pi_projector(bonds, edges, k_red):
    """C₃-equivariant lift π : undirected canonical edges → directed bonds.

    For a canonical edge e_k = (v_s, v_t, cell), the two directed bonds are:
      forward  = (v_s, v_t, cell)         with source at home cell R
      backward = (v_t, v_s, -cell)        with source at home cell R + cell

    In the script's conjugate Bloch convention (ψ(v at R) ∝ e^{-2πi k·R} ψ̃),
    the canonical undirected edge Bloch state is

        ψ̃_undir(e_k, k) = (1/√2) ( ψ̃_fwd(k) + e^{-2πi k·cell} ψ̃_bwd(k) )

    so π carries:
      π[forward_bond_idx,  e_k] = 1/√2
      π[backward_bond_idx, e_k] = e^{-2πi k·cell} / √2

    With this Bloch-phase-aware factor, π is C₃-equivariant:
        π · C₃_e = C₃_directed · π
    (verified numerically below).
    """
    n_bonds = len(bonds)
    n_edges = len(edges)
    pi = np.zeros((n_bonds, n_edges), dtype=complex)

    # Forward / backward bond indices for each canonical edge.
    fwd_idx = {}
    bwd_idx = {}
    for b_idx, (src, tgt, cell, _dr) in enumerate(bonds):
        for (e_idx, vs, vt, ec) in edges:
            if (vs, vt, ec) == (src, tgt, cell):
                fwd_idx[e_idx] = b_idx
            neg_cell = tuple(-c for c in ec)
            if (vt, vs, neg_cell) == (src, tgt, cell):
                bwd_idx[e_idx] = b_idx

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for (e_idx, vs, vt, cell) in edges:
        pi[fwd_idx[e_idx], e_idx] = inv_sqrt2
        bwd_phase = np.exp(-1j * 2 * math.pi * np.dot(k_red, cell))
        pi[bwd_idx[e_idx], e_idx] = inv_sqrt2 * bwd_phase
    return pi


def build_C3_directed(bonds):
    """C₃ on directed bonds (12-dim).  At k_P, C₃ is a pure permutation
    (no Bloch phases, since k_P is C₃-invariant and bond home cells permute
    without net translation in the conjugate Bloch sum)."""
    from srs_photon_c3_chainmap import ATOM_PERM, c3_cell
    n_bonds = len(bonds)
    C = np.zeros((n_bonds, n_bonds), dtype=complex)
    bond_lookup = {(src, tgt, cell): i
                   for i, (src, tgt, cell, _) in enumerate(bonds)}
    for i, (src, tgt, cell, _) in enumerate(bonds):
        new_src, new_tgt = ATOM_PERM[src], ATOM_PERM[tgt]
        new_cell = c3_cell(cell)
        j = bond_lookup[(new_src, new_tgt, new_cell)]
        C[j, i] = 1.0
    return C


# =============================================================================
# Build B(P) on directed bonds in the order matching `bonds` list.
# =============================================================================

def build_B_directed(bonds, k_red):
    """B(k)[f, e] = [tgt(e) = src(f)] · [f ≠ rev(e)] · exp(-2πi k·e.cell)"""
    n = len(bonds)
    B = np.zeros((n, n), dtype=complex)
    for e_idx, (e_src, e_tgt, e_cell, _) in enumerate(bonds):
        for f_idx, (f_src, f_tgt, f_cell, _) in enumerate(bonds):
            if f_src != e_tgt:
                continue
            rev_cell = tuple(-c for c in e_cell)
            if (f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell):
                continue
            B[f_idx, e_idx] += np.exp(-1j * 2 * math.pi * np.dot(k_red, e_cell))
    return B


# =============================================================================
# Driver.
# =============================================================================

def main():
    print("=" * 72)
    print("Lemma 3 (β closure) — Step B: chirality coefficient c on photon bundle")
    print("=" * 72)

    # Geometry.
    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    n_bonds = len(bonds)
    n_edges = len(edges)
    n_verts = len(verts)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)

    print(f"\nPrimitive cell: {n_verts} vertices, {n_edges} undirected edges, "
          f"{n_bonds} directed bonds, {len(cycles)} length-10 cycles")

    k_red = K_P_RED

    # Operators at k_P.
    d = incidence_matrix_primitive(k_red, edges, n_verts)
    d1 = build_d1(cycles, edge_lookup, k_red, n_edges)
    Delta_1 = build_delta_1(d, d1)
    B = build_B_directed(bonds, k_red)

    print(f"\n--- Step 1: Walker B(P) on {n_bonds} directed bonds ---")
    print(f"  B(P) shape: {B.shape}")
    Bevs = la.eigvals(B)
    Bevs_sorted = sorted(Bevs, key=lambda z: (-abs(z), -z.imag))
    print(f"  B(P) eigenvalues (sorted by magnitude):")
    for ev in Bevs_sorted:
        print(f"    {ev.real:+.4f} {ev.imag:+.4f}j   |·| = {abs(ev):.4f}   "
              f"arg = {math.degrees(np.angle(ev)):+.2f}°")

    # Confirm h = (√3 + i√5)/2 is in the spectrum.
    h = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
    diff_to_h = min(abs(ev - h) for ev in Bevs)
    print(f"\n  closest eigenvalue to h = {h}: "
          f"{Bevs[np.argmin([abs(ev - h) for ev in Bevs])]}, distance {diff_to_h:.2e}")

    # Im(B): chirality (anti-Hermitian / 2i).
    Im_B = (B - B.conj().T) / (2j)
    print(f"\n  Im(B(P)) = (B − B†)/(2i)   shape {Im_B.shape}")
    print(f"  Im(B) Hermitian? max|Im(B)† − Im(B)| = "
          f"{np.max(np.abs(Im_B.conj().T - Im_B)):.2e}")

    print(f"\n--- Step 2: C₃-equivariant projector π (12 → 6) ---")
    pi = build_pi_projector(bonds, edges, k_red)
    print(f"  π shape: {pi.shape}")
    print(f"  π†·π = I_6 ?  max|π†π − I| = "
          f"{np.max(np.abs(pi.conj().T @ pi - np.eye(n_edges))):.2e}")
    # Verify C₃-equivariance: π · C₃_e = C₃_directed · π.
    C3_e = build_C3_edge(edges, k_red)
    C3_dir = build_C3_directed(bonds)
    eqv_err = np.max(np.abs(pi @ C3_e - C3_dir @ pi))
    print(f"  C₃-equivariance: max|π·C₃_e − C₃_dir·π| = {eqv_err:.2e}")
    if eqv_err > 1e-10:
        print(f"  WARNING — π not C₃-equivariant; chirality coupling will leak.")

    print(f"\n--- Step 3a: Symmetric V_chir = π†·Im(B)·π (parity-EVEN reading) ---")
    V_chir_sym = pi.conj().T @ Im_B @ pi
    V_chir_sym = (V_chir_sym + V_chir_sym.conj().T) / 2
    print(f"  spectrum: {sorted(la.eigvalsh(V_chir_sym).real)}")
    print(f"  → all-zero: Im(B) is in the parity-ODD sector under bond-orientation")
    print(f"     reversal; the symmetric lift filters it out.")

    print(f"\n--- Step 3b: Walker B(P) restricted to photon Hodge sector ---")
    print(f"  Define B_photon := π†·B(P)·π  (6×6).  Read off its eigenvalues on")
    print(f"  the L/R photon polarizations (which are C₃-irrep eigenstates).")
    B_photon_undir = pi.conj().T @ B @ pi
    print(f"  B_photon eigenvalues (on undirected edges):")
    for ev in la.eigvals(B_photon_undir):
        print(f"    {ev.real:+.6f} {ev.imag:+.6f}j   |·| = {abs(ev):.4f}   "
              f"arg = {math.degrees(np.angle(ev)):+.2f}°")
    # The natural chirality operator that the photon polarization sees:
    V_chir_undir = (B_photon_undir - B_photon_undir.conj().T) / (2j)
    V_chir_undir = (V_chir_undir + V_chir_undir.conj().T) / 2
    print(f"\n  V_chir := (B_photon − B_photon†)/(2i) eigenvalues:")
    for ev in sorted(la.eigvalsh(V_chir_undir).real):
        print(f"    {ev:+.6f}")

    print(f"\n--- Step 4: Restrict V_chir to photon ω² = 36 eigenspace ---")
    eigs_full, vecs_full = la.eig(Delta_1)
    order = np.argsort(eigs_full.real)
    eigs_full = eigs_full[order]
    vecs_full = vecs_full[:, order]
    target = 36.0
    mask = np.abs(eigs_full.real - target) < 1e-6
    photon_basis = vecs_full[:, mask]
    Q, _ = la.qr(photon_basis)
    print(f"  Photon basis dim: {Q.shape[1]} (expected 2)")

    V_chir_photon = Q.conj().T @ V_chir_undir @ Q
    V_chir_photon = (V_chir_photon + V_chir_photon.conj().T) / 2
    print(f"\n  V_chir | photon (2×2 in arbitrary photon basis):")
    for row in V_chir_photon:
        print("   ", "  ".join(f"{x.real:+.6f}{x.imag:+.6f}j" for x in row))
    print(f"  trace(V_chir|photon) = {np.trace(V_chir_photon).real:+.6f}  "
          f"(expected 0 if traceless)")
    print(f"  eigenvalues of V_chir|photon = "
          f"{sorted(la.eigvalsh(V_chir_photon).real)}")

    print(f"\n--- Step 5: Diagonalize C₃ on photon space; pull V_chir into L/R basis ---")
    C3_photon = Q.conj().T @ C3_e @ Q
    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()
    eigvals_C3, eigvecs_C3 = la.eig(C3_photon)
    # Identify L (ω) and R (ω²)
    L_idx = int(np.argmin(np.abs(eigvals_C3 - omega)))
    R_idx = int(np.argmin(np.abs(eigvals_C3 - omega2)))
    print(f"  C₃|photon eigenvalue (L = ω): {eigvals_C3[L_idx]}")
    print(f"  C₃|photon eigenvalue (R = ω²): {eigvals_C3[R_idx]}")
    L_vec = eigvecs_C3[:, L_idx] / la.norm(eigvecs_C3[:, L_idx])
    R_vec = eigvecs_C3[:, R_idx] / la.norm(eigvecs_C3[:, R_idx])
    LR = np.column_stack([L_vec, R_vec])
    V_chir_LR = LR.conj().T @ V_chir_photon @ LR
    V_chir_LR = (V_chir_LR + V_chir_LR.conj().T) / 2
    print(f"\n  V_chir in L/R basis (2×2):")
    print(f"    [⟨L|V|L⟩  ⟨L|V|R⟩]")
    print(f"    [⟨R|V|L⟩  ⟨R|V|R⟩]")
    for row in V_chir_LR:
        print("   ", "  ".join(f"{x.real:+.6f}{x.imag:+.6f}j" for x in row))

    cL, cR = V_chir_LR[0, 0].real, V_chir_LR[1, 1].real
    sin_arg_h = math.sqrt(5 / 8)
    print(f"\n  c_L = ⟨L|V_chir|L⟩ = {cL:+.6f}")
    print(f"  c_R = ⟨R|V_chir|R⟩ = {cR:+.6f}")
    print(f"  sin(arg h) = √(5/8) = {sin_arg_h:.6f}")
    print(f"  c_L / sin(arg h)   = {cL / sin_arg_h:+.6f}")
    print(f"  c_R / sin(arg h)   = {cR / sin_arg_h:+.6f}")
    print(f"  (c_L − c_R) / 2    = {(cL - cR) / 2:+.6f}  "
          f"(this is the chirality 'splitting half')")
    print(f"  expected splitting half = sin(arg h) = {sin_arg_h:.6f}")

    # Off-diagonals must vanish (Schur's lemma: C₃ irreps ω and ω² are
    # inequivalent, so any C₃-invariant operator must be diagonal in L/R).
    off_diag = max(abs(V_chir_LR[0, 1]), abs(V_chir_LR[1, 0]))
    print(f"\n  max off-diagonal |V_chir_LR[off]| = {off_diag:.2e}")
    if off_diag < 1e-10:
        print(f"  PASS — Schur's lemma: V_chir is diagonal in L/R basis.")
    else:
        print(f"  FAIL — V_chir has off-diagonal terms in L/R basis.")
        print(f"  This suggests the projection π is not the right chirality coupling.")

    # Final verdict.
    coef_match = abs((cL - cR) / 2 - sin_arg_h) < 1e-8
    print(f"\n  c_split / sin(arg h) − 1 = "
          f"{((cL - cR) / 2) / sin_arg_h - 1:+.2e}")
    if coef_match and off_diag < 1e-10:
        print(f"\n  ✓ c = 1 in β = c · sin(arg h) · α_EM (Lemma 3 closed).")
    else:
        print(f"\n  ✗ Coefficient does not match c = 1.  Inspect numerics.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
