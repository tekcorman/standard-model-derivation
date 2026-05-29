#!/usr/bin/env python3
"""
Generation quantum numbers from C₃ irreps at the P point of the srs BZ.

KEY INSIGHT: The srs lattice (I4₁32) has a BCC real-space lattice.
Its Brillouin zone (truncated octahedron) has a P point at fractional
reciprocal coords (1/4,1/4,1/4) where the little group contains C₃
(3-fold rotation along (111)).

The 4-band primitive cell decomposes at P into C₃ irreps:
  2 × trivial (eigenvalue 1) + ω + ω²
where ω = e^{2πi/3}. The C₃ eigenvalue IS the generation quantum number.

Previous scripts used:
  - 8-atom conventional cell (doubled bands, obscured structure)
  - Simple cubic BZ (wrong high-symmetry points, P point missing)
  - Z₄ Fourier modes for generations (screw axis ≠ generation symmetry)

This script fixes all three.

Symmetry verification:
  - C₃: (x,y,z)→(z,x,y) is symmorphic in I4₁32, centered at v₀=(1/8,1/8,1/8)
  - Permutes primitive vectors: a₁→a₂→a₃→a₁, hence b₁→b₂→b₃
  - At P=(1/4,1/4,1/4): k invariant under C₃ → [H(k_P), P_C₃] = 0
  - Time reversal: -P ≢ P (since 2k_P is NOT a reciprocal lattice vector)
    → TR does NOT force ω/ω² degeneracy → generations CAN split at P
"""

import numpy as np
from numpy import linalg as la
from itertools import product
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=6, linewidth=120)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

omega3 = np.exp(2j * np.pi / 3)       # C₃ eigenvalue ω
NN_DIST = np.sqrt(2) / 4              # srs NN distance (a=1)
ARCCOS_1_3 = np.degrees(np.arccos(1/3))

# BCC primitive vectors (conventional cubic a=1)
A_PRIM = np.array([
    [-0.5,  0.5,  0.5],   # a₁
    [ 0.5, -0.5,  0.5],   # a₂
    [ 0.5,  0.5, -0.5],   # a₃
])

# 4 atoms in Cartesian coordinates (Wyckoff 8a, x=1/8, base set only)
ATOMS = np.array([
    [1/8, 1/8, 1/8],   # v₀ — on C₃ axis
    [3/8, 7/8, 5/8],   # v₁ — permuted by C₃
    [7/8, 5/8, 3/8],   # v₂
    [5/8, 3/8, 7/8],   # v₃
])
N_ATOMS = 4

# C₃ permutation matrix: (x,y,z)→(z,x,y)
# v₀→v₀, v₁→v₃, v₂→v₁, v₃→v₂  (cycle: 1→3→2→1)
# P[i,j]=1 means C₃ maps atom j to atom i
C3_PERM = np.zeros((4, 4), dtype=complex)
C3_PERM[0, 0] = 1
C3_PERM[3, 1] = 1
C3_PERM[1, 2] = 1
C3_PERM[2, 3] = 1

# C₃ eigenstates on the 4-atom basis
# {v₁,v₂,v₃} subspace: C₃ cycle (1→3→2→1) has matrix [[0,1,0],[0,0,1],[1,0,0]]
# acting on (v₁,v₂,v₃) as: v₁→v₃ means the COLUMN for v₁ puts a 1 in the v₃ ROW
# Wait: P|v₁⟩ = |v₃⟩, so in the {v₁,v₂,v₃} subblock:
#   P_sub|1⟩ = |3⟩, P_sub|2⟩ = |1⟩, P_sub|3⟩ = |2⟩
# P_sub = [[0,1,0],[0,0,1],[1,0,0]] (the cyclic shift that sends (a,b,c)→(c,a,b))
# No wait: P_sub[row,col] = 1 at (3,1),(1,2),(2,3) in the FULL matrix
# In the {1,2,3} subspace indices (0-indexed):
#   P_sub[2,0]=1 (v₁→v₃), P_sub[0,1]=1 (v₂→v₁), P_sub[1,2]=1 (v₃→v₂)
# P_sub = [[0,1,0],[0,0,1],[1,0,0]]
# Eigenvectors of [[0,1,0],[0,0,1],[1,0,0]]:
#   λ=1: (1,1,1)/√3
#   λ=ω: Verify: P_sub (1,ω,ω²) = (ω, ω², 1) = ω(1,ω,ω²)?
#         ω·1=ω, ω·ω=ω², ω·ω²=1. Yes! ✓
#   λ=ω²: P_sub (1,ω²,ω) = (ω²,ω,1) = ω²(1,ω²,ω)?
#          ω²·1=ω², ω²·ω²=ω, ω²·ω=1. Yes! ✓

C3_ESTATES = {
    'trivial_0': np.array([1, 0, 0, 0], dtype=complex),
    'trivial_s': np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3),
    'gen_w':     np.array([0, 1, omega3, omega3**2], dtype=complex) / np.sqrt(3),
    'gen_w2':    np.array([0, 1, omega3**2, omega3], dtype=complex) / np.sqrt(3),
}


# ══════════════════════════════════════════════════════════════════════
# 1. CONNECTIVITY
# ══════════════════════════════════════════════════════════════════════

def find_bonds():
    """
    Find NN bonds in the primitive cell.

    Searches atom j at r_j + n₁a₁+n₂a₂+n₃a₃ for |nₖ| ≤ 2.
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


# ══════════════════════════════════════════════════════════════════════
# 2. BLOCH HAMILTONIAN
# ══════════════════════════════════════════════════════════════════════

def bloch_H(k_frac, bonds):
    """
    4×4 Bloch Hamiltonian at fractional k = (k₁,k₂,k₃).
    Phase: exp(2πi(k₁n₁ + k₂n₂ + k₃n₃)).
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


# ══════════════════════════════════════════════════════════════════════
# 3. C₃ DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════

def c3_decompose(k_frac, bonds, degen_tol=1e-8):
    """
    Simultaneously diagonalize H(k) and C₃.

    Since [H(k), C₃] = 0 on the Γ-P axis, they share eigenstates.
    For degenerate H eigenvalues, diagonalize C₃ within the subspace
    to get proper C₃ quantum numbers.

    Returns: eigenvalues, eigenvectors, C₃ eigenvalues (labels),
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
            # Diagonalize C₃ within the degenerate subspace
            sub = evecs[:, grp]
            C3_sub = np.conj(sub.T) @ C3_PERM @ sub
            c3_evals, c3_evecs = la.eig(C3_sub)

            # Sort by C₃ phase (angle)
            order = np.argsort(np.angle(c3_evals))
            c3_evals = c3_evals[order]
            c3_evecs = c3_evecs[:, order]

            new_sub = sub @ c3_evecs
            for ig, b in enumerate(grp):
                new_evecs[:, b] = new_sub[:, ig]
                c3_diag[b] = c3_evals[ig]

    # Quality check
    C3_new = np.conj(new_evecs.T) @ C3_PERM @ new_evecs
    offdiag = la.norm(C3_new - np.diag(np.diag(C3_new)))

    return evals, new_evecs, c3_diag, offdiag


def label_c3(c3_val):
    """Classify a C₃ eigenvalue as 1, ω, or ω²."""
    if abs(c3_val - 1.0) < 0.3:
        return '1'
    elif abs(c3_val - omega3) < 0.3:
        return 'w'
    elif abs(c3_val - omega3**2) < 0.3:
        return 'w2'
    else:
        return '?'


# ══════════════════════════════════════════════════════════════════════
# 4. ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def verify_gamma(bonds):
    """Γ-point spectrum must be {3, -1, -1, -1} (K₄ adjacency)."""
    evals, _ = diag_H([0, 0, 0], bonds)
    expected = np.array([-1, -1, -1, 3], dtype=float)
    err = la.norm(evals - expected)

    print("=" * 70)
    print("  GAMMA-POINT VERIFICATION")
    print("=" * 70)
    print(f"  H(Gamma) eigenvalues:  {evals}")
    print(f"  Expected (K4):         {expected}")
    print(f"  Error: {err:.2e}  {'PASS' if err < 0.01 else 'FAIL'}")
    return err < 0.01


def verify_c3_at_P(bonds):
    """Check [H(k_P), C₃] = 0."""
    k_P = [0.25, 0.25, 0.25]
    H = bloch_H(k_P, bonds)
    comm = C3_PERM @ H - H @ C3_PERM
    cn = la.norm(comm)

    print("\n" + "=" * 70)
    print("  C3 SYMMETRY AT P = (1/4, 1/4, 1/4)")
    print("=" * 70)
    print(f"  ||[H(k_P), C3]|| = {cn:.2e}  {'PASS' if cn < 1e-10 else 'FAIL'}")

    # Also check C₃² (should also commute)
    C3_sq = C3_PERM @ C3_PERM
    comm2 = C3_sq @ H - H @ C3_sq
    print(f"  ||[H(k_P), C3^2]|| = {la.norm(comm2):.2e}")

    # Verify C₃³ = I
    C3_cube = C3_sq @ C3_PERM
    print(f"  ||C3^3 - I|| = {la.norm(C3_cube - np.eye(4)):.2e}")

    return cn < 1e-10


def verify_TR_at_P(bonds):
    """Verify -P ≢ P (TR doesn't constrain spectrum at P)."""
    k_P = [0.25, 0.25, 0.25]
    k_mP = [-0.25, -0.25, -0.25]

    evals_P, _ = diag_H(k_P, bonds)
    evals_mP, _ = diag_H(k_mP, bonds)

    # H(k)* = H(-k) for TR, so eigenvalues of H(-k) = eigenvalues of H(k)
    # But the C₃ labels at k and -k can differ
    print(f"\n  Time-reversal check:")
    print(f"  E(P)  = {evals_P}")
    print(f"  E(-P) = {evals_mP}")
    print(f"  ||E(P) - E(-P)|| = {la.norm(evals_P - evals_mP):.2e}")

    if la.norm(evals_P - evals_mP) < 1e-10:
        print(f"  E(P) = E(-P) -- but this is because H(-k) = H(k)* (TR)")
        print(f"  The C3 labels CAN still differ if P != -P in BZ")

    # Check P ≡ -P?  Need 2k_P = reciprocal lattice vector
    # 2k_P = (1/2,1/2,1/2) fractional. NOT integer → -P ≢ P
    print(f"  2k_P in fractional = (0.5, 0.5, 0.5) -- NOT a reciprocal lattice vector")
    print(f"  Therefore -P is NOT equivalent to P.  TR does NOT constrain P spectrum.")


def analyze_P(bonds):
    """Full C₃ decomposition at the P point."""
    k_P = [0.25, 0.25, 0.25]
    evals, evecs, c3_diag, offdiag = c3_decompose(k_P, bonds)

    print("\n" + "=" * 70)
    print("  C3 DECOMPOSITION AT P POINT")
    print("=" * 70)

    print(f"\n  Eigenvalues: {evals}")
    print(f"  C3 off-diagonal magnitude: {offdiag:.2e} (0 = perfect C3 quantum numbers)")

    labels = []
    for b in range(N_ATOMS):
        lab = label_c3(c3_diag[b])
        labels.append(lab)
        c3v = c3_diag[b]
        # Show as polar
        mag, ang = abs(c3v), np.degrees(np.angle(c3v))
        print(f"  Band {b}: E={evals[b]:+.8f}  C3={c3v:.4f} (|{mag:.4f}| ∠{ang:+.1f}°)  label={lab}")

    # Overlaps with analytic C₃ eigenstates
    print(f"\n  Overlaps with analytic C3 eigenstates:")
    for name, state in C3_ESTATES.items():
        for b in range(N_ATOMS):
            ov = abs(np.dot(np.conj(state), evecs[:, b]))**2
            if ov > 0.1:
                print(f"    |<{name}|band_{b}>|^2 = {ov:.6f}")

    # Key: ω vs ω² splitting
    w_idx = [b for b, l in enumerate(labels) if l == 'w']
    w2_idx = [b for b, l in enumerate(labels) if l == 'w2']
    t_idx = [b for b, l in enumerate(labels) if l == '1']

    print(f"\n  {'='*50}")
    if w_idx and w2_idx:
        Ew = evals[w_idx[0]]
        Ew2 = evals[w2_idx[0]]
        split = Ew - Ew2
        print(f"  E(w)  = {Ew:.10f}")
        print(f"  E(w2) = {Ew2:.10f}")
        print(f"  GENERATION SPLITTING: dE = {split:.10f}")

        if abs(split) > 1e-10:
            print(f"\n  >>> GENERATIONS SPLIT AT P <<<")
            print(f"  The C3 eigenvalue defines a non-degenerate generation quantum number.")
        else:
            print(f"\n  Generations DEGENERATE at P (additional symmetry).")
            print(f"  Splitting must come from moving AWAY from P along non-C3 direction.")

    if len(t_idx) >= 2:
        print(f"\n  Trivial bands: E = {evals[t_idx[0]]:.8f}, {evals[t_idx[1]]:.8f}")
        print(f"  Singlet splitting: {evals[t_idx[1]] - evals[t_idx[0]]:.8f}")

    return evals, labels


def gamma_P_line(bonds, n_pts=400):
    """
    Band structure along Γ→P where C₃ is preserved at every point.

    k(t) = t * (1/4, 1/4, 1/4), t ∈ [0, 1].
    C₃ maps k(t)→k(t) for all t, so bands carry fixed C₃ labels.

    Tracks bands by C₃ LABEL (not energy index) to handle crossings correctly.
    At Γ: triplet is degenerate, ω and ω² both at E=-1, split=0.
    At P: ω at E=+√3, ω² at E=-√3, split=2√3.
    """
    print("\n" + "=" * 70)
    print("  GAMMA-P LINE (C3 preserved throughout)")
    print("=" * 70)

    ts = np.linspace(0, 1, n_pts)
    E_all = np.zeros((n_pts, N_ATOMS))
    c3_all = np.zeros((n_pts, N_ATOMS), dtype=complex)
    offdiag_all = np.zeros(n_pts)

    # Track by C₃ label at each k-point independently
    E_by_label = {'w': np.zeros(n_pts), 'w2': np.zeros(n_pts),
                  '1_lo': np.zeros(n_pts), '1_hi': np.zeros(n_pts)}

    for i, t in enumerate(ts):
        k = [t * 0.25, t * 0.25, t * 0.25]
        evals, evecs, c3d, od = c3_decompose(k, bonds)
        E_all[i] = evals
        c3_all[i] = c3d
        offdiag_all[i] = od

        # Assign energies by C₃ label
        trivials = []
        for b in range(N_ATOMS):
            lab = label_c3(c3d[b])
            if lab == 'w':
                E_by_label['w'][i] = evals[b]
            elif lab == 'w2':
                E_by_label['w2'][i] = evals[b]
            elif lab == '1':
                trivials.append(evals[b])

        if len(trivials) == 2:
            trivials.sort()
            E_by_label['1_lo'][i] = trivials[0]
            E_by_label['1_hi'][i] = trivials[1]

    # Labels from near-P
    labels_P = [label_c3(c3_all[-1, b]) for b in range(N_ATOMS)]
    print(f"  Band labels (from P): {labels_P}")
    print(f"  Max C3 off-diagonal along line: {offdiag_all.max():.2e}")

    split = E_by_label['w'] - E_by_label['w2']
    print(f"\n  Generation splitting E(w) - E(w2)  [tracked by C3 label]:")
    print(f"    At Gamma (t=0):  {split[0]:.10f}")
    print(f"    At t=0.25:       {split[n_pts//4]:.10f}")
    print(f"    At t=0.5:        {split[n_pts//2]:.10f}")
    print(f"    At t=0.75:       {split[3*n_pts//4]:.10f}")
    print(f"    At P (t=1):      {split[-1]:.10f}")
    print(f"    Max |split|:     {np.max(np.abs(split)):.10f}")

    # Band crossing: where does ω band cross a trivial band?
    cross_w = E_by_label['w'] - E_by_label['1_hi']
    cross_idx = None
    for i in range(1, n_pts):
        if cross_w[i-1] * cross_w[i] < 0:
            cross_idx = i
            break
    if cross_idx:
        t_cross = ts[cross_idx]
        print(f"\n  Band crossing: w band crosses trivial at t ~ {t_cross:.4f}")
        print(f"    E(w) = {E_by_label['w'][cross_idx]:.6f}, E(1_hi) = {E_by_label['1_hi'][cross_idx]:.6f}")

    # Physical structure
    print(f"\n  Physical picture along Gamma -> P:")
    print(f"    Gamma: triplet {{1,w,w2}} at E=-1, singlet {{1}} at E=+3")
    print(f"    P:     lower {{1,w2}} at E=-sqrt(3), upper {{1,w}} at E=+sqrt(3)")
    print(f"    The w band RISES from -1 to +sqrt(3) (crosses through trivial)")
    print(f"    The w2 band FALLS from -1 to -sqrt(3)")
    print(f"    This band crossing is a topological feature of the srs lattice")

    return ts, E_all, c3_all, labels_P, E_by_label


def extended_P_line(bonds, n_pts=400):
    """
    Continue past P to see where the splitting goes.
    k(t) = t * (1/4, 1/4, 1/4), t ∈ [0, 4] (P at t=1, Γ' at t=4).
    """
    ts = np.linspace(0, 4, n_pts)
    E_all = np.zeros((n_pts, N_ATOMS))
    c3_all = np.zeros((n_pts, N_ATOMS), dtype=complex)

    for i, t in enumerate(ts):
        k = [t * 0.25, t * 0.25, t * 0.25]
        evals, _, c3d, _ = c3_decompose(k, bonds)
        E_all[i] = evals
        c3_all[i] = c3d

    return ts, E_all, c3_all


def full_bz_bands(bonds, n_pts=200):
    """
    Band structure along BCC high-symmetry path: Γ-H-N-Γ-P-H | P-N
    """
    print("\n" + "=" * 70)
    print("  FULL BCC BZ BAND STRUCTURE")
    print("=" * 70)

    # BCC BZ high-symmetry points (fractional reciprocal coords)
    G = np.array([0.0, 0.0, 0.0])
    H = np.array([0.5, -0.5, 0.5])
    N = np.array([0.0, 0.0, 0.5])
    P = np.array([0.25, 0.25, 0.25])

    segments = [(G, H, 'G', 'H'), (H, N, 'H', 'N'), (N, G, 'N', 'G'),
                (G, P, 'G', 'P'), (P, H, 'P', 'H')]

    all_k = []
    all_E = []
    all_c3 = []
    ticks = [0.0]
    tlabels = ['G']

    cum = 0.0
    for k_s, k_e, ls, le in segments:
        seg_len = la.norm(k_e - k_s)
        for i in range(n_pts):
            t = i / n_pts
            k = k_s + t * (k_e - k_s)
            evals, _, c3d, _ = c3_decompose(k, bonds)
            all_k.append(cum + t * seg_len)
            all_E.append(evals)
            all_c3.append(c3d)
        cum += seg_len
        ticks.append(cum)
        tlabels.append(le)

    all_E = np.array(all_E)
    all_c3 = np.array(all_c3)
    all_k = np.array(all_k)

    # Report bands at high-symmetry points
    for label, k in [('G', G), ('H', H), ('N', N), ('P', P)]:
        evals, _ = diag_H(k, bonds)
        print(f"  E({label}) = {evals}")

    return all_k, all_E, all_c3, ticks, tlabels


def bz_average_splitting(bonds, n_grid=30):
    """
    Compute BZ-averaged splitting between C₃-labeled bands.

    Only meaningful on the Γ-P axis where C₃ is a good quantum number.
    Off-axis: track via overlap with P-point eigenstates.
    """
    print("\n" + "=" * 70)
    print("  BZ-AVERAGED GENERATION ANALYSIS")
    print("=" * 70)

    # Get P-point eigenstates as reference
    _, evecs_P = diag_H([0.25, 0.25, 0.25], bonds)

    # BZ grid
    splittings = []
    c3_quality = []

    for n1, n2, n3 in product(range(n_grid), repeat=3):
        k = np.array([n1, n2, n3], dtype=float) / n_grid
        evals, evecs, c3d, od = c3_decompose(k, bonds)

        # Overlap with P-point states to track which band is which
        overlap = np.abs(np.conj(evecs_P.T) @ evecs)**2  # [P_band, k_band]

        # Assignment: each P-band maps to the k-band with max overlap
        assignment = np.argmax(overlap, axis=1)  # P_band_i → k_band[assignment[i]]

        E_by_P_label = evals[assignment]
        splittings.append(E_by_P_label)
        c3_quality.append(od)

    splittings = np.array(splittings)
    c3_quality = np.array(c3_quality)

    # Identify which P-bands are ω and ω²
    _, _, c3_P, _ = c3_decompose([0.25, 0.25, 0.25], bonds)
    labels_P = [label_c3(c3_P[b]) for b in range(N_ATOMS)]

    w_idx = [b for b, l in enumerate(labels_P) if l == 'w']
    w2_idx = [b for b, l in enumerate(labels_P) if l == 'w2']

    if w_idx and w2_idx:
        gen_split = splittings[:, w_idx[0]] - splittings[:, w2_idx[0]]
        print(f"  BZ-averaged |E(w) - E(w2)|: {np.mean(np.abs(gen_split)):.8f}")
        print(f"  RMS E(w) - E(w2):           {np.sqrt(np.mean(gen_split**2)):.8f}")
        print(f"  Max |E(w) - E(w2)|:         {np.max(np.abs(gen_split)):.8f}")
        print(f"  Fraction with split > 0.01: {np.mean(np.abs(gen_split) > 0.01):.4f}")

    print(f"  Mean C3 off-diagonal (mixing): {np.mean(c3_quality):.4f}")
    print(f"  Max C3 off-diagonal:           {np.max(c3_quality):.4f}")

    return splittings, labels_P


# ══════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════════════════

def plot_all(ts_GP, E_GP, labels_P, E_by_label,
             bz_k, bz_E, bz_c3, bz_ticks, bz_tlabels,
             ts_ext, E_ext, c3_ext):
    """Generate all plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0): Γ→P line, bands tracked by C₃ label
    ax = axes[0, 0]
    ax.plot(ts_GP, E_by_label['w'], color='red', linewidth=2.5, label='ω (gen)')
    ax.plot(ts_GP, E_by_label['w2'], color='blue', linewidth=2.5, label='ω² (gen)')
    ax.plot(ts_GP, E_by_label['1_lo'], color='black', linewidth=1.5, ls='--', label='trivial (lo)')
    ax.plot(ts_GP, E_by_label['1_hi'], color='black', linewidth=1.5, ls=':', label='trivial (hi)')
    ax.set_xlabel('t  (Γ at 0, P at 1)')
    ax.set_ylabel('Energy')
    ax.set_title('Γ → P: C₃-labeled bands (ω=red, ω²=blue)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color='green', ls='--', alpha=0.4)

    # (0,1): Full BZ band structure
    ax = axes[0, 1]
    for b in range(N_ATOMS):
        ax.plot(bz_k, bz_E[:, b], color='steelblue', linewidth=1.2)
    for t in bz_ticks:
        ax.axvline(t, color='gray', ls='--', lw=0.5)
    ax.set_xticks(bz_ticks)
    ax.set_xticklabels(bz_tlabels)
    ax.set_ylabel('Energy')
    ax.set_title('SRS Band Structure (4-atom primitive cell, BCC BZ)')
    ax.grid(True, alpha=0.3)

    # (1,0): Generation splitting tracked by C₃ label
    ax = axes[1, 0]
    split = E_by_label['w'] - E_by_label['w2']
    ax.plot(ts_GP, split, color='purple', linewidth=2)
    ax.set_xlabel('t  (Γ at 0, P at 1)')
    ax.set_ylabel('E(ω) − E(ω²)')
    ax.set_title('Generation splitting along Γ → P (by C₃ label)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)

    # (1,1): Extended line through P
    ax = axes[1, 1]
    for b in range(N_ATOMS):
        ax.plot(ts_ext, E_ext[:, b], color='steelblue', linewidth=1.2)
    ax.axvline(1.0, color='green', ls='--', alpha=0.4, label='P')
    ax.axvline(0.0, color='orange', ls='--', alpha=0.4, label='Γ')
    ax.axvline(2.0, color='orange', ls='--', alpha=0.4)
    ax.axvline(4.0, color='orange', ls='--', alpha=0.4)
    ax.set_xlabel('t  (k = t·(1/4,1/4,1/4))')
    ax.set_ylabel('Energy')
    ax.set_title('Extended (111) line: periodicity check')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'srs_generation_c3.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")

    # Second figure: C₃ label quality across BZ
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    c3_mag = np.abs(bz_c3)
    for b in range(N_ATOMS):
        ax2.plot(bz_k, c3_mag[:, b], linewidth=1, label=f'Band {b}')
    for t in bz_ticks:
        ax2.axvline(t, color='gray', ls='--', lw=0.5)
    ax2.set_xticks(bz_ticks)
    ax2.set_xticklabels(bz_tlabels)
    ax2.set_ylabel('|C₃ eigenvalue|')
    ax2.set_title('C₃ label magnitude across BZ (1.0 = good quantum number)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(OUTDIR, 'srs_generation_c3_quality.png')
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Plot saved: {path2}")


# ══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SRS GENERATION DEFINITION VIA C3 IRREPS AT BCC BZ P POINT")
    print("  4-atom primitive cell | correct BZ | correct symmetry")
    print("=" * 70)

    # Build connectivity
    print("\nFinding bonds...")
    bonds = find_bonds()
    print(f"  {len(bonds)} bonds (4 atoms x 3 neighbors = 12)")
    for src, tgt, cell in bonds:
        rj = ATOMS[tgt] + cell[0]*A_PRIM[0] + cell[1]*A_PRIM[1] + cell[2]*A_PRIM[2]
        dr = rj - ATOMS[src]
        print(f"    v{src}->v{tgt} cell=({cell[0]:+d},{cell[1]:+d},{cell[2]:+d})  "
              f"dr=({dr[0]:+.4f},{dr[1]:+.4f},{dr[2]:+.4f})  d={la.norm(dr):.4f}")

    # Verifications
    ok1 = verify_gamma(bonds)
    ok2 = verify_c3_at_P(bonds)
    verify_TR_at_P(bonds)

    if not ok1 or not ok2:
        print("\n  VERIFICATION FAILED — stopping.")
        return

    # P-point analysis
    evals_P, labels_P = analyze_P(bonds)

    # Γ-P line
    ts_GP, E_GP, c3_GP, labels_GP, E_by_label = gamma_P_line(bonds, n_pts=400)

    # Extended line
    ts_ext, E_ext, c3_ext = extended_P_line(bonds, n_pts=400)

    # Full BZ
    bz_k, bz_E, bz_c3, bz_ticks, bz_tlabels = full_bz_bands(bonds, n_pts=150)

    # BZ-averaged analysis
    bz_split, bz_labels = bz_average_splitting(bonds, n_grid=25)

    # Plots
    plot_all(ts_GP, E_GP, labels_GP, E_by_label,
             bz_k, bz_E, bz_c3, bz_ticks, bz_tlabels,
             ts_ext, E_ext, c3_ext)

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  Primitive cell: 4 atoms, BCC lattice")
    print(f"  BZ: truncated octahedron (FCC reciprocal)")
    print(f"  Gamma spectrum: [-1, -1, -1, 3]  (K4 adjacency)")
    print(f"  [H(P), C3] = 0:  C3 is exact symmetry at P")
    print(f"  -P not equiv P:  TR does not force degeneracy")
    print(f"  P-point eigenvalues: {evals_P}")
    print(f"  C3 labels: {labels_P}")

    w_idx = [b for b, l in enumerate(labels_P) if l == 'w']
    w2_idx = [b for b, l in enumerate(labels_P) if l == 'w2']
    if w_idx and w2_idx:
        split_P = evals_P[w_idx[0]] - evals_P[w2_idx[0]]
        print(f"\n  GENERATION SPLITTING AT P: {split_P:.10f}")
        print(f"  = 2*sqrt(3) = {2*np.sqrt(3):.10f}")

        if abs(split_P) > 1e-10:
            print(f"\n  >>> RESULT: Generations are non-degenerate at P.")
            print(f"  >>> The C3 eigenvalue IS the generation quantum number.")
            print(f"  >>> Generation states = pure Fourier modes (0,1,w^n,w^2n)/sqrt(3)")
            print(f"  >>> on the K4 vertex orbit {{v1,v2,v3}}.")
        else:
            print(f"  Generations degenerate at P.")

        # Label-tracked splitting along Gamma-P
        split_line = E_by_label['w'] - E_by_label['w2']
        print(f"\n  Splitting along Gamma-P (label-tracked):")
        print(f"    Gamma: {split_line[0]:.10f}  (should be 0 — triplet degenerate)")
        print(f"    P:     {split_line[-1]:.10f}  (= 2*sqrt(3))")
        print(f"    Max:   {np.max(np.abs(split_line)):.10f}")


if __name__ == '__main__':
    main()
