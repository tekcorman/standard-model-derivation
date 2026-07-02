#!/usr/bin/env python3
"""
proofs/foundations/srs_cycles_su4_explicit_iso.py

PURPOSE
-------
Construct an explicit linear isomorphism

    B : V_15(cycles) ──→ su(4)_PS_adjoint

between the 15-dimensional cycle space at srs vertex 0 and the 15-dimensional
adjoint of SU(4)_PS, with the property that B intertwines:

  (i)  the C_3 actions on both sides (cycle's [111]-rotation permutation
       <-> Ad_g conjugation by g = diag(1, 1, ω, ω²) on Mat(4, ℂ)
       restricted to traceless matrices);
  (ii) the chirality grading on cycles (9 chiral ⊕ 6 P-sym) <-> Pati-Salam
       grading on su(4) (9 unbroken (SU(3)_color × U(1)_{B-L}) ⊕ 6 broken
       (leptoquarks SU(4) / (SU(3) × U(1)_{B-L}))).

This upgrades the prior multiplicity / block-pattern match (verified in
`srs_cycle_C3_irrep_decomposition.py` and `srs_cycles_PS_unbroken_broken_match.py`)
from "5+5+5 dimension match plus 9+6 block-pattern match" to "explicit
basis-level intertwining map verified at machine precision".

WHAT IS / IS NOT VERIFIED HERE
------------------------------
VERIFIED:
  - Both sides decompose as 3×5 = (1, ω, ω²) eigenvalue × (5 dim per eigenvalue).
  - Within each C_3 eigenvalue subspace, both sides further split as
    3 unbroken / chiral ⊕ 2 broken / P-sym.
  - An explicit invertible map B is constructed cell-by-cell; it is automatically
    a C_3-module isomorphism because it pairs eigenspaces at the same eigenvalue.
  - The chirality grading is also intertwined by construction.
  - All sanity checks pass at machine precision.

NOT VERIFIED (and explicitly NOT claimed):
  - This is NOT a Lie-algebra isomorphism. The cycle space carries no
    canonical Lie bracket; only the su(4) side has one. The iso is at the
    level of C_3-modules + chirality grading.
  - The iso is not unique: it has (3×3)^3 × (2×2)^3 block-diagonal freedom
    (one invertible matrix per cell of the 5×3 grid). The probe records
    this freedom and uses the natural diagonal cell-wise pairing.
  - This does NOT close ADOPTED-B3 (Pati-Salam labeling); B3 is about
    spinor labeling, not gauge content. This finding provides a substrate-
    level structural anchor for the SU(4)_PS gauge group complementary to B3.
  - The Bloch-eigenmode lift (Option B in the scoping doc §5) remains open.

WHAT THIS PROBE VERIFIES (numerical)
------------------------------------
  E1. 15 girth-10 cycles enumerated; chirality classification (9+6) verified.
  E2. C_3 permutation matrix on cycles built; C_3³ = I (machine precision).
  E3. Cell decomposition: 5 orbits × 3 cycles, with 3 chiral orbits + 2 P-sym
      orbits, all C_3-eigenvalue invariant.
  E4. Cycle eigenvectors: explicit construction of the 15 eigenvectors
      v_{(s, λ)} for species s ∈ {chiral_1, chiral_2, chiral_3, psym_1, psym_2}
      and color λ ∈ {1, ω, ω²}. P_C3 v_{(s, λ)} = λ v_{(s, λ)} at machine precision.
  E5. su(4)_PS adjoint basis: explicit construction of 15 traceless 4×4 matrices
      with B6's C_3-eigenvalue labels (5 + 5 + 5 split) and PS grading
      (9 unbroken + 6 broken split).
  E6. Eigenvalue match: cycle 5+5+5 ↔ su(4) 5+5+5 cell-by-cell.
  E7. Chirality / PS match: cycle (3 chiral + 2 P-sym) per λ ↔ su(4)
      (3 unbroken + 2 leptoquark) per λ cell-by-cell.
  E8. Construction of B as 15×15 block-diagonal map (one block per
      (chirality, eigenvalue) cell). Verified invertible (det ≠ 0).
  E9. Intertwining: B · ρ_cycle(C_3) = Ad_g · B at machine precision, where
      ρ_cycle is the cycle space's C_3 representation in the eigenvector basis
      and Ad_g is the conjugation by g = diag(1, 1, ω, ω²) on Mat(4, ℂ)
      restricted to traceless matrices, in the su(4) adjoint basis.
  E10. Chirality intertwining: B(V_chiral) = V_unbroken, B(V_psym) = V_leptoquark
       (verified by checking B applied to chirality projector matches PS
       grading projector).
  E11. The iso is NOT unique: documented (3×3)^3 × (2×2)^3 cell-internal freedom.

GATE STATUS
-----------
Closes the (a) sub-item flagged in `srs_cycles_su4_adjoint_identification_2026-05-03.md`:

  > (a) An explicit linear isomorphism between the 15-cycle space and the
  >     15-dim adjoint of su(4)_PS that intertwines the C_3 actions.

The (b) sub-item (Bloch-momentum / Hashimoto eigenmode verification) remains
open; it would supply additional canonical structure that could fix the
cell-internal basis freedom and elevate the iso to "canonical" rather than
"existence-up-to-cell-internal-rotation".

CROSS-REFERENCES
----------------
  - `proofs/foundations/srs_cycle_C3_irrep_decomposition.py` (cycle C_3 character)
  - `proofs/foundations/srs_cycles_PS_unbroken_broken_match.py` (block-pattern match)
  - `proofs/foundations/srs_chirality_orbit_decomposition.py` (orbit decomposition)
  - `framework/B3_B6_reconciliation.md` (B6 C_3 element diag(1,1,ω,ω²))
  - `audits/registers/adoption_register.md` ADOPTED-B3 (Pati-Salam labeling)
"""

import os
import sys
from itertools import product

import numpy as np
from numpy import linalg as la

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


# =============================================================================
# srs structure (matches existing probes: srs_chirality_orbit_decomposition.py)
# =============================================================================

A_PRIM = np.array([[-0.5,  0.5,  0.5],
                   [ 0.5, -0.5,  0.5],
                   [ 0.5,  0.5, -0.5]])
ATOMS = np.array([[1/8, 1/8, 1/8],
                  [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8],
                  [5/8, 3/8, 7/8]])
N_ATOMS = 4
GIRTH = 10
SUPERCELL = 3

omega = np.exp(2j * np.pi / 3)


def frac_to_cart(frac):
    return A_PRIM.T @ np.asarray(frac, dtype=float)


def find_bonds():
    tol, NN = 0.02, np.sqrt(2) / 4
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = la.norm(rj - ATOMS[i])
                if d < tol:
                    continue
                if abs(d - NN) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds


def vertex_cart(atom, cell):
    frac = ATOMS[atom] + cell[0]*A_PRIM[0] + cell[1]*A_PRIM[1] + cell[2]*A_PRIM[2]
    return frac_to_cart(frac)


# =============================================================================
# Enumerate 15 girth-10 cycles + classify chirality
# =============================================================================

def build_cycles_and_chirality():
    bonds = find_bonds()

    def get_nbrs(atom, cell):
        out = []
        for src, tgt, dc in bonds:
            if src != atom:
                continue
            nc = (cell[0]+dc[0], cell[1]+dc[1], cell[2]+dc[2])
            if all(abs(c) <= SUPERCELL for c in nc):
                out.append((tgt, nc))
        return out

    start = (0, (0, 0, 0))
    cycles_ordered = []

    def dfs(path, current, depth):
        atom, cell = current
        prev = path[-2] if depth >= 1 else None
        for tgt, nc in get_nbrs(atom, cell):
            if prev is not None and (tgt, nc) == prev:
                continue
            if depth == GIRTH - 1:
                if (tgt, nc) == start and start not in path[1:]:
                    cycles_ordered.append(path[:])
            elif depth < GIRTH - 1:
                if (tgt, nc) == start:
                    continue
                dfs(path + [(tgt, nc)], (tgt, nc), depth + 1)

    dfs([start], start, 0)

    def unoriented_edge(v1, v2):
        return tuple(sorted([v1, v2]))

    def cycle_edge_set(path):
        return frozenset(unoriented_edge(path[i], path[(i+1) % len(path)])
                         for i in range(len(path)))

    seen = {}
    for path in cycles_ordered:
        es = cycle_edge_set(path)
        if es not in seen:
            seen[es] = path
    cycles_unique = list(seen.values())
    assert len(cycles_unique) == 15, f"expected 15 cycles, got {len(cycles_unique)}"

    # Chirality via signed projected area on perp([111])
    axis = A_PRIM.T @ np.array([1.0, 1.0, 1.0]); axis /= la.norm(axis)
    ref = np.array([1.0, 0.0, 0.0])
    e1 = ref - np.dot(ref, axis) * axis; e1 /= la.norm(e1)
    e2 = np.cross(axis, e1)
    origin = vertex_cart(0, (0, 0, 0))

    def project_2d(v):
        rel = v - origin
        return np.array([np.dot(rel, e1), np.dot(rel, e2)])

    def signed_area(pts):
        n = len(pts); s = 0.0
        for i in range(n):
            x1, y1 = pts[i]; x2, y2 = pts[(i+1) % n]
            s += x1*y2 - x2*y1
        return s / 2

    chirality_label = []
    for path in cycles_unique:
        pts = [project_2d(vertex_cart(a, c)) for (a, c) in path]
        sa = signed_area(pts)
        chirality_label.append('chiral' if abs(sa) > 1e-10 else 'psym')

    return cycles_unique, chirality_label, cycle_edge_set


def build_C3_perm(cycles_unique, cycle_edge_set):
    def C3_cart(v):
        return np.array([v[2], v[0], v[1]])

    def apply_C3_to_vertex(atom_cell):
        a, c = atom_cell
        v_rot = C3_cart(vertex_cart(a, c))
        best = None; best_d = float('inf')
        for ap in range(N_ATOMS):
            for n1, n2, n3 in product(range(-SUPERCELL-1, SUPERCELL+2), repeat=3):
                cp = (n1, n2, n3)
                d = la.norm(vertex_cart(ap, cp) - v_rot)
                if d < best_d:
                    best_d = d; best = (ap, cp)
        return best if best_d < 1e-6 else None

    def apply_C3_to_cycle(path):
        new = []
        for v in path:
            nv = apply_C3_to_vertex(v)
            if nv is None:
                return None
            new.append(nv)
        return new

    def cycle_match(path):
        target = cycle_edge_set(path)
        for i, p in enumerate(cycles_unique):
            if cycle_edge_set(p) == target:
                return i
        return None

    P_C3 = np.zeros((15, 15), dtype=complex)
    for i, path in enumerate(cycles_unique):
        rotated = apply_C3_to_cycle(path)
        j = cycle_match(rotated)
        P_C3[j, i] = 1.0
    return P_C3


def find_orbits(subset_idx, P_C3):
    perm = {src: int(np.argmax(np.abs(P_C3[:, src]))) for src in subset_idx}
    visited = set(); orbits = []
    for src in subset_idx:
        if src in visited:
            continue
        orb = [src]; visited.add(src)
        nxt = perm[src]
        while nxt != src:
            orb.append(nxt); visited.add(nxt)
            nxt = perm[nxt]
        orbits.append(orb)
    return orbits


# =============================================================================
# su(4)_PS adjoint basis with B6 C_3 = diag(1, 1, ω, ω²)
#
# Index assignment (per parent doc and srs_cycles_PS_unbroken_broken_match.py):
#     i = 1  →  lepton  (g_1 = 1)
#     i = 2  →  color-1 (g_2 = 1)
#     i = 3  →  color-2 (g_3 = ω)
#     i = 4  →  color-3 (g_4 = ω²)
# =============================================================================

g4 = np.diag([1.0, 1.0, omega, omega**2])
g4_inv = np.diag([1.0, 1.0, np.conj(omega), np.conj(omega**2)])


def E(i, j, n=4):
    """Matrix unit E_{ij} (i, j ∈ 1..n) of size n×n."""
    M = np.zeros((n, n), dtype=complex)
    M[i-1, j-1] = 1.0
    return M


def Ad_g(M):
    """Conjugation by g = diag(1, 1, ω, ω²): Ad_g(M) = g M g^{-1}."""
    return g4 @ M @ g4_inv


def build_su4_basis():
    """
    Build a basis of 15 traceless 4×4 matrices, organised into a 5×3 grid by
    (Pati-Salam component, C_3 eigenvalue).

    Cells (5 PS species × 3 C_3 eigenvalues):

      Eigenvalue 1 (5 dim):
        Unbroken (chiral side, 3):
          T3_color   = diag(0, 0, 1, -1)/√2   (SU(3) Cartan, "T_3" of SU(2)-color)
          T8_color   = diag(0, 0, 1, 1)/√2 - 2/√6 · diag(0, 1, 0, 0)   no, see below
          B_L        = diag(-3, 1, 1, 1)/(2√3)   (B-L, traceless)
        Broken (P-sym side, 2):
          E_{12}, E_{21}                       (lepton-color1 leptoquark Cartan)

      Eigenvalue ω (5 dim):
        Unbroken (chiral side, 3): E_{32}, E_{43}, E_{24} (gluon raisers)
        Broken (P-sym side, 2):    E_{31}, E_{14}        (lepton-color leptoquarks)

      Eigenvalue ω² (5 dim):
        Unbroken (chiral side, 3): E_{23}, E_{34}, E_{42} (gluon lowerers)
        Broken (P-sym side, 2):    E_{13}, E_{41}        (lepton-color leptoquarks)

    For the eigenvalue-1 unbroken subspace, we use the standard SU(3) Cartan +
    B-L basis:

      H1 (= "T_3 of color")   = diag(0,  0,  1, -1) / √2
      H2 (= "T_8 of color")   = diag(0, -2,  1,  1) / √6
      H3 (= B-L)              = diag(-3, 1,  1,  1) / (2√3)

    All three are traceless and have eigenvalue 1 under Ad_g (they are diagonal,
    so Ad_g leaves them invariant).
    """
    basis = {}  # (PS_species, eigval_label) → list of matrices

    # -----------------------------------------------------------------
    # Eigenvalue 1: unbroken (3 generators) + broken (2 generators)
    # -----------------------------------------------------------------
    H1 = np.diag([0,  0,  1, -1]).astype(complex) / np.sqrt(2)
    H2 = np.diag([0, -2,  1,  1]).astype(complex) / np.sqrt(6)
    H3 = np.diag([-3, 1,  1,  1]).astype(complex) / (2 * np.sqrt(3))
    # Sanity: traceless
    assert abs(np.trace(H1)) < 1e-12
    assert abs(np.trace(H2)) < 1e-12
    assert abs(np.trace(H3)) < 1e-12
    # Sanity: Ad_g eigenvalue = 1
    for H, name in [(H1, "H1"), (H2, "H2"), (H3, "H3")]:
        assert la.norm(Ad_g(H) - 1.0 * H) < 1e-12, f"{name} not Ad_g-invariant"

    basis[("unbroken", "1")] = [H1, H2, H3]
    basis[("broken",   "1")] = [E(1, 2), E(2, 1)]

    # -----------------------------------------------------------------
    # Eigenvalue ω: unbroken (gluon raisers) + broken (LQ raisers)
    # -----------------------------------------------------------------
    # Gluon raisers: E_{ij} with g_i/g_j = ω, i, j ∈ {2, 3, 4}
    #   E_{32}: g_3/g_2 = ω/1 = ω ✓
    #   E_{43}: g_4/g_3 = ω²/ω = ω ✓
    #   E_{24}: g_2/g_4 = 1/ω² = ω ✓
    basis[("unbroken", "ω")] = [E(3, 2), E(4, 3), E(2, 4)]
    # Leptoquarks at eigenvalue ω: E_{ij} with i = 1 or j = 1, g_i/g_j = ω
    #   E_{31}: g_3/g_1 = ω/1 = ω ✓
    #   E_{14}: g_1/g_4 = 1/ω² = ω ✓
    basis[("broken",   "ω")] = [E(3, 1), E(1, 4)]

    # -----------------------------------------------------------------
    # Eigenvalue ω²: unbroken (gluon lowerers) + broken (LQ lowerers)
    # -----------------------------------------------------------------
    basis[("unbroken", "ω²")] = [E(2, 3), E(3, 4), E(4, 2)]
    basis[("broken",   "ω²")] = [E(1, 3), E(4, 1)]

    return basis


def verify_su4_eigenvalues(basis):
    """For each cell, verify all matrices have the expected Ad_g eigenvalue."""
    expected = {"1": 1.0, "ω": omega, "ω²": omega**2}
    for (ps, ev_lbl), mats in basis.items():
        target = expected[ev_lbl]
        for k, M in enumerate(mats):
            assert la.norm(Ad_g(M) - target * M) < 1e-12, \
                f"Cell ({ps}, {ev_lbl})[{k}] Ad_g eigenvalue mismatch"


def flatten_su4_basis(basis, ordering):
    """Concatenate cell-wise basis into a single ordered list of 15 matrices."""
    out = []
    for cell in ordering:
        out.extend(basis[cell])
    return out


def vec_su4_basis(basis_list):
    """Stack 15 traceless 4×4 matrices into a 16×15 column matrix (vec)."""
    cols = [M.reshape(16, order='F') for M in basis_list]
    return np.column_stack(cols)


def Ad_g_matrix_in_basis(basis_list):
    """
    Return the 15×15 matrix of Ad_g acting in the given su(4) basis.

    For a basis-ordered list of generators T_a, the matrix is defined by
    Ad_g(T_a) = Σ_b A[b, a] T_b (i.e., column a contains the Ad_g-image of T_a
    expanded in the basis).
    """
    n = len(basis_list)
    A = np.zeros((n, n), dtype=complex)
    # Vec the basis (stack as columns of a matrix M_basis of shape 16×n)
    M_basis = vec_su4_basis(basis_list)
    # Solve M_basis @ A = (Ad_g images of basis)
    M_imgs = np.column_stack([Ad_g(M).reshape(16, order='F') for M in basis_list])
    # Least-squares; with 15 linearly independent columns of M_basis (rank 15)
    # the solve is exact.
    A_sol, *_ = la.lstsq(M_basis, M_imgs, rcond=None)
    return A_sol


# =============================================================================
# Construction of cycle eigenvectors and the iso B
# =============================================================================

def build_cycle_eigenvectors(orbits, P_C3, label):
    """
    For each orbit (a 3-cycle under C_3), construct the three C_3 eigenvectors
    at eigenvalues 1, ω, ω². Return a dict
      result[(species_name, eigval_label)] = numpy array length 15
    """
    result = {}
    for k, orb in enumerate(orbits):
        a, b, c = orb  # P_C3 e_a = e_b, P_C3 e_b = e_c, P_C3 e_c = e_a
        species = f"{label}_{k+1}"
        # Eigenvector at eigenvalue λ: v_λ = e_a + λ̄ e_b + λ̄² e_c (verified in §0).
        for ev_lbl, lam in [("1", 1.0), ("ω", omega), ("ω²", omega**2)]:
            v = np.zeros(15, dtype=complex)
            v[a] = 1.0
            v[b] = np.conj(lam)
            v[c] = np.conj(lam) ** 2
            v /= np.sqrt(3)   # unit normalisation
            result[(species, ev_lbl)] = v
            # Verify eigenvalue
            assert la.norm(P_C3 @ v - lam * v) < 1e-10, \
                f"Eigenvector ({species}, {ev_lbl}) not at eigenvalue λ = {lam}"
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Explicit C_3-module isomorphism: cycles → su(4)_PS adjoint")
    print("=" * 80)

    # ----- E1 + E2: cycles + chirality + C_3 permutation -----
    print("\n[E1] 15 girth-10 cycles + chirality classification")
    cycles_unique, chirality_label, cycle_edge_set = build_cycles_and_chirality()
    n_chiral = chirality_label.count('chiral')
    n_psym   = chirality_label.count('psym')
    assert n_chiral == 9 and n_psym == 6
    print(f"     ✓ 9 chiral + 6 P-sym cycles ({n_chiral + n_psym} total)")

    print("\n[E2] C_3 permutation matrix on cycles")
    P_C3 = build_C3_perm(cycles_unique, cycle_edge_set)
    P_C3_cubed = la.matrix_power(P_C3, 3)
    err_e2 = la.norm(P_C3_cubed - np.eye(15))
    print(f"     ||P_C3³ - I|| = {err_e2:.2e}    (machine precision)")
    assert err_e2 < 1e-10

    # ----- E3: orbits within chirality classes -----
    print("\n[E3] Orbit decomposition under C_3")
    chiral_idx = [i for i, c in enumerate(chirality_label) if c == 'chiral']
    psym_idx   = [i for i, c in enumerate(chirality_label) if c == 'psym']
    orbits_chiral = find_orbits(chiral_idx, P_C3)
    orbits_psym   = find_orbits(psym_idx,   P_C3)
    print(f"     chiral orbits: {len(orbits_chiral)} (sizes {[len(o) for o in orbits_chiral]})")
    print(f"     P-sym  orbits: {len(orbits_psym)} (sizes {[len(o) for o in orbits_psym]})")
    assert len(orbits_chiral) == 3 and all(len(o) == 3 for o in orbits_chiral)
    assert len(orbits_psym)   == 2 and all(len(o) == 3 for o in orbits_psym)

    # ----- E4: cycle eigenvectors -----
    print("\n[E4] Cycle C_3-eigenvectors per (species, color) cell")
    chiral_eigvecs = build_cycle_eigenvectors(orbits_chiral, P_C3, "chiral")
    psym_eigvecs   = build_cycle_eigenvectors(orbits_psym,   P_C3, "psym")
    cycle_eigvecs = {**chiral_eigvecs, **psym_eigvecs}
    assert len(cycle_eigvecs) == 15
    print(f"     ✓ 15 eigenvectors built; each verified P_C3 v = λ v")

    # ----- E5: su(4)_PS basis -----
    print("\n[E5] su(4)_PS adjoint basis with B6 C_3 = diag(1, 1, ω, ω²)")
    su4_basis = build_su4_basis()
    verify_su4_eigenvalues(su4_basis)
    total_su4 = sum(len(v) for v in su4_basis.values())
    assert total_su4 == 15
    print(f"     ✓ 15 generators, all Ad_g eigenvalues match cell labels")

    # ----- E6: Eigenvalue cell match -----
    print("\n[E6] Cell match — cycle 5+5+5 ↔ su(4) 5+5+5")
    cell_table = {}
    for ev in ["1", "ω", "ω²"]:
        cyc_count = sum(1 for (sp, e) in cycle_eigvecs if e == ev)
        su4_count = (len(su4_basis[("unbroken", ev)]) +
                     len(su4_basis[("broken",   ev)]))
        print(f"     λ = {ev:>3}: {cyc_count} cycles, {su4_count} su(4) gens")
        assert cyc_count == 5 and su4_count == 5

    # ----- E7: Chirality / PS match -----
    print("\n[E7] Chirality / PS match per cell")
    print(f"     {'cell':<22}  {'cycles':>7}  {'su(4) gens':>10}")
    for ev in ["1", "ω", "ω²"]:
        # chiral cycles at this ev = 3 species × 1 = 3
        n_cyc_chiral = sum(1 for (sp, e) in chiral_eigvecs if e == ev)
        n_cyc_psym   = sum(1 for (sp, e) in psym_eigvecs   if e == ev)
        n_su4_unbr   = len(su4_basis[("unbroken", ev)])
        n_su4_brok   = len(su4_basis[("broken",   ev)])
        print(f"     ({ev:>3}, chiral / unbroken):  {n_cyc_chiral:>7}  {n_su4_unbr:>10}")
        print(f"     ({ev:>3}, P-sym  / broken  ):  {n_cyc_psym:>7}  {n_su4_brok:>10}")
        assert n_cyc_chiral == n_su4_unbr == 3
        assert n_cyc_psym   == n_su4_brok == 2

    # ----- E8: Construct B as cell-wise diagonal map -----
    print("\n[E8] Construction of B (cell-wise diagonal pairing)")
    # Order cells consistently for cycle and su(4) sides:
    cycle_order = [
        # (species, eigval) for the 15 cycle eigenvectors, in 5 species × 3 colors order
        ("chiral_1", "1"), ("chiral_2", "1"), ("chiral_3", "1"),  # 3 chiral × ev=1
        ("chiral_1", "ω"), ("chiral_2", "ω"), ("chiral_3", "ω"),  # 3 chiral × ev=ω
        ("chiral_1", "ω²"), ("chiral_2", "ω²"), ("chiral_3", "ω²"),
        ("psym_1",   "1"), ("psym_2",   "1"),
        ("psym_1",   "ω"), ("psym_2",   "ω"),
        ("psym_1",   "ω²"), ("psym_2",   "ω²"),
    ]
    su4_order_cells = [
        ("unbroken", "1"), ("unbroken", "ω"), ("unbroken", "ω²"),
        ("broken",   "1"), ("broken",   "ω"), ("broken",   "ω²"),
    ]
    su4_order = []
    for cell in su4_order_cells:
        for k in range(len(su4_basis[cell])):
            su4_order.append((cell, k))

    # The cycle order is: 3 chiral × 3 evs (in λ-major order: 1, ω, ω², per chiral species)
    # Wait — let me redo cycle_order to put things in (chirality, eigval, species-within-cell) order
    # to match the su4_order grouping more naturally.
    cycle_order = []
    for ps_eqv in [("unbroken", "1"), ("unbroken", "ω"), ("unbroken", "ω²"),
                   ("broken",   "1"), ("broken",   "ω"), ("broken",   "ω²")]:
        ps, ev = ps_eqv
        if ps == "unbroken":
            for k in range(3):  # chiral_1, chiral_2, chiral_3
                cycle_order.append((f"chiral_{k+1}", ev))
        else:
            for k in range(2):  # psym_1, psym_2
                cycle_order.append((f"psym_{k+1}", ev))
    assert len(cycle_order) == 15 and len(su4_order) == 15

    # B is the matrix that sends cycle eigvec at position i to su(4) gen at position i.
    # Express both as 15×15 matrices in some external basis, then B = Su4_M @ Cyc_M^{-1}.
    # External basis for su(4): vec into 16-dim, restricted to the 15-dim traceless subspace.
    # External basis for cycles: standard basis on 15 cycles.

    # Cycle eigvecs as columns of a 15×15 matrix
    Cyc_M = np.column_stack([cycle_eigvecs[(sp, ev)] for (sp, ev) in cycle_order])
    # Each su(4) gen as a 16-vec (col-major flatten); then 16×15
    Su4_M16 = np.column_stack([su4_basis[cell][k].reshape(16, order='F')
                               for (cell, k) in su4_order])

    # B sends cycle eigvec basis to su(4) gen basis. B as a map V_cycles → su(4)_adj
    # in COORDINATE form (cycle standard basis → su(4) generator coordinates):
    #   B: ℂ^15 → ℂ^15
    # If x = sum α_i (cycle eigvec_i), then B(x) = sum α_i (su(4) gen_i).
    # So B has matrix Su4_coords @ Cyc_M^{-1} in the cycle/su(4)-coord representation,
    # where Su4_coords is just the identity in the su(4) generator basis (since we
    # express B as a 15×15 matrix in (cycle standard) → (su(4) generator) coordinates).
    # I.e., B: cycle vector → su(4)-coord vector, B = (Cyc_M)^{-1} when reading
    # cycle vectors in their standard basis on the input side, and writing su(4)
    # generator coefficients on the output side (since the i-th eigvec maps to the
    # i-th generator).
    B = la.inv(Cyc_M)
    det_B = la.det(B)
    print(f"     |det B| = {abs(det_B):.4e}    (must be ≠ 0)")
    assert abs(det_B) > 1e-6

    # ----- E9: Intertwining check B · ρ_cycle(C_3) = Ad_g · B -----
    print("\n[E9] Intertwining: B · ρ_cycle(C_3) = Ad_g · B")
    # ρ_cycle(C_3) on cycle standard basis is just P_C3.
    # Ad_g on su(4) generator basis: this is the matrix representation of Ad_g
    # in the su4_basis ordering (su4_order). Compute it.
    su4_basis_list = [su4_basis[cell][k] for (cell, k) in su4_order]
    Ad_g_mat = Ad_g_matrix_in_basis(su4_basis_list)
    # Verify Ad_g_mat is diagonal (within each cell) at eigenvalues
    is_diag = la.norm(Ad_g_mat - np.diag(np.diag(Ad_g_mat))) < 1e-10
    print(f"     Ad_g matrix is diagonal: {is_diag}")
    print(f"     diag(Ad_g) = {np.round(np.diag(Ad_g_mat), 4)}")

    # Test: B P_C3 = Ad_g_mat B
    lhs = B @ P_C3
    rhs = Ad_g_mat @ B
    intertwine_err = la.norm(lhs - rhs)
    print(f"     ||B · P_C3 - Ad_g · B|| = {intertwine_err:.2e}    (must be 0)")
    assert intertwine_err < 1e-10

    # ----- E10: Chirality intertwining -----
    print("\n[E10] Chirality / PS grading intertwining")
    # Chirality projector on cycles: diag(1 if chiral else 0) in standard cycle basis
    P_chir_cyc = np.diag([1.0 if c == 'chiral' else 0.0 for c in chirality_label]).astype(complex)
    P_psym_cyc = np.eye(15, dtype=complex) - P_chir_cyc
    # PS grading projector on su(4) basis: 1 on unbroken (first 9 of su4_order), 0 on broken
    P_unbr_su4 = np.diag([1.0 if cell[0] == "unbroken" else 0.0
                          for (cell, _) in su4_order]).astype(complex)
    P_brok_su4 = np.eye(15, dtype=complex) - P_unbr_su4

    # Intertwining: B · P_chir_cyc · B^{-1} = P_unbr_su4
    Binv = la.inv(B)
    lhs_c = B @ P_chir_cyc @ Binv
    err_c = la.norm(lhs_c - P_unbr_su4)
    print(f"     ||B · P_chir · B^{{-1}} - P_unbroken|| = {err_c:.2e}    (must be 0)")
    assert err_c < 1e-10
    lhs_p = B @ P_psym_cyc @ Binv
    err_p = la.norm(lhs_p - P_brok_su4)
    print(f"     ||B · P_psym · B^{{-1}} - P_broken||   = {err_p:.2e}    (must be 0)")
    assert err_p < 1e-10

    # ----- E11: Document cell-internal freedom -----
    print("\n[E11] Cell-internal freedom of the iso")
    print("     The iso B is unique up to one invertible matrix per cell of the")
    print("     5×3 grid (block-diagonal in the (species, eigenvalue) decomposition).")
    print("     Specifically:")
    print("       3 unbroken cells × GL(3, ℂ) freedom each → (3×3)^3 freedom")
    print("       3 broken   cells × GL(2, ℂ) freedom each → (2×2)^3 freedom")
    print("     Total cell-internal freedom: GL(3, ℂ)^3 × GL(2, ℂ)^3.")
    print("     The diagonal pairing chosen here is one canonical representative;")
    print("     elevating to a unique iso requires additional structure (Bloch-")
    print("     eigenmode lift — Option B in the parent doc §5).")

    # ----- Summary -----
    print()
    print("=" * 80)
    print("THEOREM (proven)")
    print("=" * 80)
    print()
    print("  The 15-dim cycle space at srs vertex 0 (with C_3 lattice symmetry")
    print("  and the chirality grading 9 chiral ⊕ 6 P-sym) is C_3-module-")
    print("  isomorphic to the 15-dim adjoint of su(4)_PS (with Ad_g action of")
    print("  g = diag(1, 1, ω, ω²) and the Pati-Salam grading 9 unbroken ⊕ 6 broken)")
    print("  via an explicit linear map B. The iso intertwines:")
    print()
    print("    B · P_C3        = Ad_g · B    (machine precision)")
    print("    B(V_chiral)     = V_unbroken   (machine precision)")
    print("    B(V_P-sym)      = V_broken    (machine precision)")
    print()
    print("  This upgrades the prior multiplicity / block-pattern match of")
    print("  `srs_cycles_su4_adjoint_identification_2026-05-03.md` from")
    print("  '5+5+5 dimension match + 9+6 block-pattern match' to 'explicit")
    print("  basis-level intertwining map verified at machine precision'.")
    print()
    print("  The iso has GL(3, ℂ)^3 × GL(2, ℂ)^3 cell-internal freedom; the")
    print("  diagonal pairing here is one canonical representative.            ∎")
    print()
    print("OPEN ITEMS (unchanged)")
    print("----------------------")
    print("  - This is NOT a Lie-algebra isomorphism (cycle space carries no")
    print("    canonical Lie bracket).")
    print("  - Bloch-eigenmode verification (parent doc §5 Option B) remains open.")
    print("  - Internal SU(2) × U(1)^2 fiber identification (parent doc §")
    print("    'Internal structure') remains a research-level question.")
    print("=" * 80)


if __name__ == "__main__":
    main()
