"""
Shared infrastructure for srs lattice computations.

Extracted from srs_generation_c3.py. All proofs that need the srs Bloch
Hamiltonian, C3 decomposition, or lattice constants import from here.

The srs net (space group I4_132) has a BCC real-space lattice with 4 atoms
per primitive cell. Its Brillouin zone is a truncated octahedron with a
P point at (1/4,1/4,1/4) where the little group contains C3.
"""

import numpy as np
from numpy import linalg as la
from itertools import product

# ======================================================================
# CONSTANTS
# ======================================================================

omega3 = np.exp(2j * np.pi / 3)       # C3 eigenvalue omega
NN_DIST = np.sqrt(2) / 4              # srs NN distance (a=1)
ARCCOS_1_3 = np.degrees(np.arccos(1/3))

# Key physics constants from k*=3
K_STAR = 3                             # optimal coordination number
GIRTH = 10                             # girth of srs net
ALPHA_1 = (2/3)**8                     # NB walk survival: ((k-1)/k)^(g-2)

# Hashimoto eigenvalue at P
h_P = (np.sqrt(3) + 1j * np.sqrt(5)) / 2   # |h| = sqrt(k*-1) = sqrt(2)

# BCC primitive vectors (conventional cubic a=1)
A_PRIM = np.array([
    [-0.5,  0.5,  0.5],   # a1
    [ 0.5, -0.5,  0.5],   # a2
    [ 0.5,  0.5, -0.5],   # a3
])

# 4 atoms in Cartesian coordinates (Wyckoff 8a, x=1/8, base set only)
ATOMS = np.array([
    [1/8, 1/8, 1/8],   # v0 -- on C3 axis
    [3/8, 7/8, 5/8],   # v1 -- permuted by C3
    [7/8, 5/8, 3/8],   # v2
    [5/8, 3/8, 7/8],   # v3
])
N_ATOMS = 4

# C3 permutation matrix: (x,y,z) -> (z,x,y)
# v0->v0, v1->v3, v2->v1, v3->v2  (cycle: 1->3->2->1)
# P[i,j]=1 means C3 maps atom j to atom i
C3_PERM = np.zeros((4, 4), dtype=complex)
C3_PERM[0, 0] = 1
C3_PERM[3, 1] = 1
C3_PERM[1, 2] = 1
C3_PERM[2, 3] = 1

# C3 eigenstates on the 4-atom basis
C3_ESTATES = {
    'trivial_0': np.array([1, 0, 0, 0], dtype=complex),
    'trivial_s': np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3),
    'gen_w':     np.array([0, 1, omega3, omega3**2], dtype=complex) / np.sqrt(3),
    'gen_w2':    np.array([0, 1, omega3**2, omega3], dtype=complex) / np.sqrt(3),
}


# ======================================================================
# CONNECTIVITY
# ======================================================================

def find_bonds():
    """
    Find NN bonds in the primitive cell.

    Searches atom j at r_j + n1*a1 + n2*a2 + n3*a3 for |n_k| <= 2.
    Returns: list of (src, tgt, (n1,n2,n3)) tuples.
    """
    tol = 0.02
    bonds = []

    for i in range(N_ATOMS):
        ri = ATOMS[i]
        nbrs = []
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                dist = la.norm(rj - ri)
                if dist < tol:
                    continue
                if abs(dist - NN_DIST) < tol:
                    nbrs.append((j, (n1, n2, n3), rj - ri))

        assert len(nbrs) == 3, f"Atom {i} has {len(nbrs)} NN (expected 3)"
        for j, cell, dr in nbrs:
            bonds.append((i, j, cell))

    return bonds


# ======================================================================
# BLOCH HAMILTONIAN
# ======================================================================

def bloch_H(k_frac, bonds):
    """
    4x4 Bloch Hamiltonian at fractional k = (k1, k2, k3).
    Phase: exp(2*pi*i*(k1*n1 + k2*n2 + k3*n3)).
    """
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for src, tgt, cell in bonds:
        phase = np.exp(2j * np.pi * np.dot(k, cell))
        H[tgt, src] += phase
    return H


def diag_H(k_frac, bonds):
    """Diagonalize H(k), return sorted eigenvalues and eigenvectors."""
    H = bloch_H(k_frac, bonds)
    evals, evecs = la.eigh(H)
    idx = np.argsort(np.real(evals))
    return np.real(evals[idx]), evecs[:, idx]


# ======================================================================
# C3 DECOMPOSITION
# ======================================================================

def c3_decompose(k_frac, bonds, degen_tol=1e-8):
    """
    Simultaneously diagonalize H(k) and C3.

    Since [H(k), C3] = 0 on the Gamma-P axis, they share eigenstates.
    For degenerate H eigenvalues, diagonalize C3 within the subspace
    to get proper C3 quantum numbers.

    Returns: eigenvalues, eigenvectors, C3 eigenvalues (labels),
             off-diagonal magnitude (quality check)
    """
    evals, evecs = diag_H(k_frac, bonds)

    # Group degenerate bands
    groups = []
    i = 0
    while i < N_ATOMS:
        grp = [i]
        while i + 1 < N_ATOMS and abs(evals[i+1] - evals[i]) < degen_tol:
            i += 1
            grp.append(i)
        groups.append(grp)
        i += 1

    new_evecs = evecs.copy()
    c3_diag = np.zeros(N_ATOMS, dtype=complex)

    for grp in groups:
        if len(grp) == 1:
            b = grp[0]
            c3_diag[b] = np.conj(evecs[:, b]) @ C3_PERM @ evecs[:, b]
        else:
            sub = evecs[:, grp]
            C3_sub = np.conj(sub.T) @ C3_PERM @ sub
            c3_evals, c3_evecs = la.eig(C3_sub)

            order = np.argsort(np.angle(c3_evals))
            c3_evals = c3_evals[order]
            c3_evecs = c3_evecs[:, order]

            new_sub = sub @ c3_evecs
            for ig, b in enumerate(grp):
                new_evecs[:, b] = new_sub[:, ig]
                c3_diag[b] = c3_evals[ig]

    C3_new = np.conj(new_evecs.T) @ C3_PERM @ new_evecs
    offdiag = la.norm(C3_new - np.diag(np.diag(C3_new)))

    return evals, new_evecs, c3_diag, offdiag


def label_c3(c3_val):
    """Classify a C3 eigenvalue as 1, omega, or omega^2."""
    if abs(c3_val - 1.0) < 0.3:
        return '1'
    elif abs(c3_val - omega3) < 0.3:
        return 'w'
    elif abs(c3_val - omega3**2) < 0.3:
        return 'w2'
    else:
        return '?'
