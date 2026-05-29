#!/usr/bin/env python3
"""
CKM matrix elements from generation mixing off the C₃ axis on the srs lattice.

At the P point k_P = (1/4,1/4,1/4), the 4-band Bloch Hamiltonian commutes
with C₃ and the eigenstates carry definite generation labels {1, 1, ω, ω²}.
Away from the (111) axis, C₃ is broken and generations MIX.

The CKM-like mixing matrix V_mn measures the transition amplitude between
generation m and generation n when probed at a general k-point.

Multiple extraction methods:
  1. BZ-averaged spectral overlap: V_mn ~ ∫ dk |⟨gen_m(P)|ψ_n(k)⟩|²
  2. Resolvent method: V_mn ~ ∫ dk |⟨gen_m|G(E_n)|gen_n⟩| with suitable E
  3. Direct overlap in degenerate subspaces
  4. Connection to NB walk result: V_us = (2/3)^{2+√3}

KEY: On K₄ alone, S₃ symmetry makes all off-diagonal V_mn equal.
     The full srs lattice breaks S₃ (confirmed splitting ratio 3.87),
     so Bloch mixing should give UNEQUAL CKM elements.

PDG reference: V_us = 0.22500, V_cb = 0.04182, V_ub = 0.00369
               δ_CP = 68.5°, J = 3.08e-5
"""

import numpy as np
from numpy import linalg as la
from itertools import product
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=8, linewidth=140)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS  (shared with srs_generation_c3.py)
# ══════════════════════════════════════════════════════════════════════

omega3 = np.exp(2j * np.pi / 3)
NN_DIST = np.sqrt(2) / 4

A_PRIM = np.array([
    [-0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
])

ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])
N_ATOMS = 4

# C₃ permutation: v₀→v₀, v₁→v₃, v₂→v₁, v₃→v₂
C3_PERM = np.zeros((4, 4), dtype=complex)
C3_PERM[0, 0] = 1
C3_PERM[3, 1] = 1
C3_PERM[1, 2] = 1
C3_PERM[2, 3] = 1

# PDG reference values
PDG_Vus = 0.22500
PDG_Vcb = 0.04182
PDG_Vub = 0.00369
PDG_delta_CP = 68.5  # degrees
PDG_J = 3.08e-5

# NB walk prediction
NB_Vus = (2/3)**(2 + np.sqrt(3))


# ══════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE (from srs_generation_c3.py)
# ══════════════════════════════════════════════════════════════════════

def find_bonds():
    """Find NN bonds in the primitive cell."""
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


def bloch_H(k_frac, bonds):
    """4×4 Bloch Hamiltonian at fractional k."""
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


def c3_decompose(k_frac, bonds, degen_tol=1e-8):
    """Simultaneously diagonalize H(k) and C₃."""
    evals, evecs = diag_H(k_frac, bonds)

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
    """Classify a C₃ eigenvalue."""
    if abs(c3_val - 1.0) < 0.3:
        return '1'
    elif abs(c3_val - omega3) < 0.3:
        return 'w'
    elif abs(c3_val - omega3**2) < 0.3:
        return 'w2'
    else:
        return '?'


# ══════════════════════════════════════════════════════════════════════
# P-POINT GENERATION EIGENSTATES
# ══════════════════════════════════════════════════════════════════════

def get_generation_states(bonds):
    """
    Get the three generation eigenstates at the P point.

    Returns dict with keys 'gen1' (trivial on orbit), 'gen2' (ω), 'gen3' (ω²)
    and the full set of 4 eigenstates with labels.

    Convention: gen1=d (trivial), gen2=s (ω²), gen3=b (ω)
    ordered by mass (energy splitting at P): gen1 lightest, gen3 heaviest
    """
    k_P = [0.25, 0.25, 0.25]
    evals, evecs, c3_diag, offdiag = c3_decompose(k_P, bonds)

    labels = [label_c3(c3_diag[b]) for b in range(N_ATOMS)]

    # Find the generation-carrying bands (ω and ω²)
    gen_states = {}
    gen_energies = {}

    for b in range(N_ATOMS):
        if labels[b] == 'w':
            gen_states['w'] = evecs[:, b]
            gen_energies['w'] = evals[b]
        elif labels[b] == 'w2':
            gen_states['w2'] = evecs[:, b]
            gen_energies['w2'] = evals[b]

    # Also get the trivial states
    trivials = [(evals[b], evecs[:, b]) for b in range(N_ATOMS) if labels[b] == '1']
    trivials.sort(key=lambda x: x[0])

    return {
        'evals': evals,
        'evecs': evecs,
        'labels': labels,
        'c3_diag': c3_diag,
        'gen_w': gen_states['w'],       # ω eigenstate (E = +√3)
        'gen_w2': gen_states['w2'],     # ω² eigenstate (E = -√3)
        'triv_lo': trivials[0][1],      # lower trivial
        'triv_hi': trivials[1][1],      # upper trivial
        'E_w': gen_energies['w'],
        'E_w2': gen_energies['w2'],
    }


# ══════════════════════════════════════════════════════════════════════
# METHOD 1: BZ-AVERAGED SPECTRAL OVERLAP
# ══════════════════════════════════════════════════════════════════════

def method_spectral_overlap(bonds, gen_info, n_grid=40):
    """
    V_mn = (1/N_k) Σ_k Σ_α |⟨gen_m(P)|ψ_α(k)⟩|² · |⟨gen_n(P)|ψ_α(k)⟩|²

    This measures the probability that generation m at P "becomes" generation n
    when transported to a general k-point, summed over all intermediate bands α.

    The off-diagonal elements V_{w,w2} measure how much ω mixes into ω² across the BZ.
    """
    print("\n" + "=" * 70)
    print("  METHOD 1: BZ-AVERAGED SPECTRAL OVERLAP")
    print("  V_mn = (1/N_k) Σ_k Σ_α |⟨gen_m|ψ_α(k)⟩|² |⟨gen_n|ψ_α(k)⟩|²")
    print("=" * 70)

    # Generation states at P
    states = {
        'w': gen_info['gen_w'],
        'w2': gen_info['gen_w2'],
        'triv_lo': gen_info['triv_lo'],
        'triv_hi': gen_info['triv_hi'],
    }

    # For the 3-generation CKM, use: triv_hi (on same energy as ω at P), ω, ω²
    # Actually: the 3 generations in the K₄ orbit subspace are {trivial_s, ω, ω²}
    # The trivial_s = (0,1,1,1)/√3 state
    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
    gen_labels = ['1_s', 'w', 'w2']
    gen_vecs = [triv_s, gen_info['gen_w'], gen_info['gen_w2']]

    n_gen = len(gen_labels)
    V_raw = np.zeros((n_gen, n_gen))
    n_k = 0

    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                evals, evecs = diag_H(k, bonds)
                n_k += 1

                # Overlaps: O[m, α] = |⟨gen_m|ψ_α(k)⟩|²
                O = np.zeros((n_gen, N_ATOMS))
                for m in range(n_gen):
                    for a in range(N_ATOMS):
                        O[m, a] = abs(np.dot(np.conj(gen_vecs[m]), evecs[:, a]))**2

                # V_mn = Σ_α O[m,α] · O[n,α]
                V_raw += O @ O.T

    V = V_raw / n_k

    print(f"\n  Grid: {n_grid}³ = {n_k} k-points")
    print(f"\n  Raw overlap matrix (3×3, gen labels: {gen_labels}):")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{V[m,n]:12.8f} "
        print(row)

    # Normalize: make each row sum to 1 (doubly stochastic if symmetric)
    V_norm = V / V.sum(axis=1, keepdims=True)
    print(f"\n  Row-normalized overlap matrix:")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{V_norm[m,n]:12.8f} "
        print(row)

    # Extract |V_us|, |V_cb|, |V_ub| by identifying off-diagonal elements
    # Convention: gen1=1_s (d-like), gen2=w2 (s-like), gen3=w (b-like)
    # So V_us ~ V[0,1], V_cb ~ V[1,2], V_ub ~ V[0,2]
    print(f"\n  CKM-like extraction (V_norm off-diag):")
    print(f"    |V_us| ~ V[1_s, w2] = {V_norm[0,1]:.6f}  (PDG: {PDG_Vus})")
    print(f"    |V_cb| ~ V[w2, w]   = {V_norm[1,2]:.6f}  (PDG: {PDG_Vcb})")
    print(f"    |V_ub| ~ V[1_s, w]  = {V_norm[0,2]:.6f}  (PDG: {PDG_Vub})")

    # Also try sqrt of off-diagonal (since V² ~ probability)
    print(f"\n  sqrt(V_norm) off-diagonal:")
    print(f"    sqrt(V_us) = {np.sqrt(V_norm[0,1]):.6f}")
    print(f"    sqrt(V_cb) = {np.sqrt(V_norm[1,2]):.6f}")
    print(f"    sqrt(V_ub) = {np.sqrt(V_norm[0,2]):.6f}")

    return V, V_norm, gen_labels


# ══════════════════════════════════════════════════════════════════════
# METHOD 2: BZ-AVERAGED AMPLITUDE (not probability)
# ══════════════════════════════════════════════════════════════════════

def method_amplitude_overlap(bonds, gen_info, n_grid=40):
    """
    V_mn = (1/N_k) Σ_k Σ_α ⟨gen_m|ψ_α(k)⟩ ⟨ψ_α(k)|gen_n⟩

    Complex-valued BZ average. This can carry a phase (CP violation source).
    """
    print("\n" + "=" * 70)
    print("  METHOD 2: BZ-AVERAGED AMPLITUDE OVERLAP")
    print("  V_mn = (1/N_k) Σ_k Σ_α ⟨gen_m|ψ_α(k)⟩⟨ψ_α(k)|gen_n⟩")
    print("=" * 70)

    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
    gen_labels = ['1_s', 'w', 'w2']
    gen_vecs = [triv_s, gen_info['gen_w'], gen_info['gen_w2']]

    n_gen = len(gen_labels)
    V_complex = np.zeros((n_gen, n_gen), dtype=complex)
    n_k = 0

    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                evals, evecs = diag_H(k, bonds)
                n_k += 1

                # ⟨gen_m|ψ_α⟩
                amps = np.zeros((n_gen, N_ATOMS), dtype=complex)
                for m in range(n_gen):
                    for a in range(N_ATOMS):
                        amps[m, a] = np.dot(np.conj(gen_vecs[m]), evecs[:, a])

                # V_mn = Σ_α amps[m,α] * conj(amps[n,α])
                V_complex += amps @ np.conj(amps.T)

    V_complex /= n_k

    print(f"\n  Grid: {n_grid}³ = {n_k} k-points")
    print(f"\n  Complex overlap matrix |V_mn|:")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{abs(V_complex[m,n]):12.8f} "
        print(row)

    print(f"\n  Phases arg(V_mn) [degrees]:")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{np.degrees(np.angle(V_complex[m,n])):+12.4f} "
        print(row)

    # V_complex should be identity at P (completeness).
    # The deviation from identity IS the mixing.
    V_dev = V_complex - np.eye(n_gen)
    print(f"\n  Deviation from identity (V - I):")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{abs(V_dev[m,n]):12.8f} "
        print(row)

    # Note: Σ_α |ψ_α⟩⟨ψ_α| = I (completeness) so Σ_k V_mn = δ_mn
    # This means V_complex IS the identity. The mixing comes from
    # restricting the sum to SUBSETS of bands.
    print(f"\n  NOTE: By completeness, BZ-averaged V = δ_mn exactly.")
    print(f"  Mixing must come from BAND-RESTRICTED sums (see Method 3).")

    return V_complex


# ══════════════════════════════════════════════════════════════════════
# METHOD 3: BAND-RESTRICTED OVERLAP (the physical one)
# ══════════════════════════════════════════════════════════════════════

def method_band_restricted(bonds, gen_info, n_grid=40):
    """
    For each k, the 4 bands split into groups (by energy proximity to P-point values).
    The "valence" (lower 2 bands) and "conduction" (upper 2 bands) can be considered
    separately. The generation mixing within a band group gives the physical CKM.

    V^(group)_mn = (1/N_k) Σ_k Σ_{α∈group} ⟨gen_m|ψ_α(k)⟩⟨ψ_α(k)|gen_n⟩

    At P: lower bands contain {trivial, ω²}, upper contain {trivial, ω}.
    Off-axis: the generation content of each band group changes → mixing.
    """
    print("\n" + "=" * 70)
    print("  METHOD 3: BAND-RESTRICTED OVERLAP (physical CKM)")
    print("  V^(lo/hi)_mn = BZ-avg of generation projector within band group")
    print("=" * 70)

    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
    gen_labels = ['1_s', 'w', 'w2']
    gen_vecs = [triv_s, gen_info['gen_w'], gen_info['gen_w2']]
    n_gen = len(gen_labels)

    # Band groups: lower 2 bands vs upper 2 bands
    V_lo = np.zeros((n_gen, n_gen), dtype=complex)
    V_hi = np.zeros((n_gen, n_gen), dtype=complex)

    # Also track: for each k, what fraction of ω is in the lower bands?
    w_in_lo = []
    w2_in_hi = []
    n_k = 0

    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                evals, evecs = diag_H(k, bonds)
                n_k += 1

                amps = np.zeros((n_gen, N_ATOMS), dtype=complex)
                for m in range(n_gen):
                    for a in range(N_ATOMS):
                        amps[m, a] = np.dot(np.conj(gen_vecs[m]), evecs[:, a])

                # Lower 2 bands (indices 0,1), upper 2 bands (indices 2,3)
                V_lo += amps[:, :2] @ np.conj(amps[:, :2].T)
                V_hi += amps[:, 2:] @ np.conj(amps[:, 2:].T)

                # Track ω content
                w_lo = sum(abs(amps[1, a])**2 for a in range(2))
                w_in_lo.append(w_lo)
                w2_hi = sum(abs(amps[2, a])**2 for a in range(2, 4))
                w2_in_hi.append(w2_hi)

    V_lo /= n_k
    V_hi /= n_k

    print(f"\n  Grid: {n_grid}³ = {n_k} k-points")

    for label, V in [("LOWER 2 bands", V_lo), ("UPPER 2 bands", V_hi)]:
        print(f"\n  {label} — |V_mn|:")
        for m in range(n_gen):
            row = "    "
            for n in range(n_gen):
                row += f"{abs(V[m,n]):12.8f} "
            print(row)

        # Off-diagonal elements
        print(f"    Off-diagonal magnitudes:")
        print(f"      |V[1_s,w]|  = {abs(V[0,1]):.8f}")
        print(f"      |V[1_s,w2]| = {abs(V[0,2]):.8f}")
        print(f"      |V[w,w2]|   = {abs(V[1,2]):.8f}")

    # The KEY test: are the off-diagonal elements UNEQUAL?
    offdiag_lo = [abs(V_lo[0,1]), abs(V_lo[0,2]), abs(V_lo[1,2])]
    offdiag_hi = [abs(V_hi[0,1]), abs(V_hi[0,2]), abs(V_hi[1,2])]

    print(f"\n  S₃ BREAKING TEST:")
    print(f"    Lower band off-diag: {offdiag_lo[0]:.8f}, {offdiag_lo[1]:.8f}, {offdiag_lo[2]:.8f}")
    print(f"    Ratio max/min (lower): {max(offdiag_lo)/max(min(offdiag_lo), 1e-15):.4f}")
    print(f"    Upper band off-diag: {offdiag_hi[0]:.8f}, {offdiag_hi[1]:.8f}, {offdiag_hi[2]:.8f}")
    print(f"    Ratio max/min (upper): {max(offdiag_hi)/max(min(offdiag_hi), 1e-15):.4f}")

    print(f"\n  ω in lower bands (BZ avg): {np.mean(w_in_lo):.6f}")
    print(f"  ω² in upper bands (BZ avg): {np.mean(w2_in_hi):.6f}")

    return V_lo, V_hi


# ══════════════════════════════════════════════════════════════════════
# METHOD 4: ENERGY-WEIGHTED TRANSITION AMPLITUDE
# ══════════════════════════════════════════════════════════════════════

def method_energy_weighted(bonds, gen_info, n_grid=40):
    """
    Weight the overlap by the energy difference from the P-point value.
    Bands far from the P-point energy of a generation contribute less.

    V_mn = (1/N_k) Σ_k Σ_α f(E_α - E_m^P) |⟨gen_m|ψ_α⟩|² · |⟨gen_n|ψ_α⟩|²

    where f is a Lorentzian or Gaussian centered at E_m^P.
    """
    print("\n" + "=" * 70)
    print("  METHOD 4: ENERGY-WEIGHTED TRANSITION AMPLITUDE")
    print("=" * 70)

    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
    gen_labels = ['1_s', 'w', 'w2']
    gen_vecs = [triv_s, gen_info['gen_w'], gen_info['gen_w2']]
    n_gen = len(gen_labels)

    # P-point energies for each generation
    # triv_s is in the orbit subspace: at P it could be in either trivial band
    # Compute explicitly
    E_P = gen_info['evals']
    labels_P = gen_info['labels']
    E_gen = np.zeros(n_gen)

    # 1_s overlaps with both trivial bands; use the one with larger overlap
    for b in range(N_ATOMS):
        ov = abs(np.dot(np.conj(triv_s), gen_info['evecs'][:, b]))**2
        if ov > 0.4:
            E_gen[0] = E_P[b]
            break
    E_gen[1] = gen_info['E_w']
    E_gen[2] = gen_info['E_w2']

    print(f"  P-point generation energies: {E_gen}")

    # Try multiple widths
    for sigma_label, sigma in [("narrow (0.5)", 0.5), ("medium (1.0)", 1.0),
                                ("wide (2.0)", 2.0), ("bandwidth (3.46)", 2*np.sqrt(3))]:
        V = np.zeros((n_gen, n_gen))
        n_k = 0

        for n1 in range(n_grid):
            for n2 in range(n_grid):
                for n3 in range(n_grid):
                    k = np.array([n1, n2, n3], dtype=float) / n_grid
                    evals, evecs = diag_H(k, bonds)
                    n_k += 1

                    O = np.zeros((n_gen, N_ATOMS))
                    for m in range(n_gen):
                        for a in range(N_ATOMS):
                            O[m, a] = abs(np.dot(np.conj(gen_vecs[m]), evecs[:, a]))**2

                    for m in range(n_gen):
                        for n in range(n_gen):
                            for a in range(N_ATOMS):
                                # Gaussian weight centered at E_gen[m]
                                w = np.exp(-0.5 * ((evals[a] - E_gen[m]) / sigma)**2)
                                V[m, n] += O[m, a] * O[n, a] * w

        V /= n_k
        # Normalize rows
        V_norm = V / V.sum(axis=1, keepdims=True)

        print(f"\n  sigma = {sigma_label}:")
        print(f"    |V_us| ~ {V_norm[0,2]:.8f}  |V_cb| ~ {V_norm[1,2]:.8f}  |V_ub| ~ {V_norm[0,1]:.8f}")
        print(f"    Ratios: V_us/V_cb = {V_norm[0,2]/max(V_norm[1,2],1e-15):.4f}  "
              f"V_cb/V_ub = {V_norm[1,2]/max(V_norm[0,1],1e-15):.4f}")

    return


# ══════════════════════════════════════════════════════════════════════
# METHOD 5: RESOLVENT / GREEN'S FUNCTION
# ══════════════════════════════════════════════════════════════════════

def method_resolvent(bonds, gen_info, n_grid=40):
    """
    G_mn(E) = (1/N_k) Σ_k ⟨gen_m| (E - H(k))^{-1} |gen_n⟩

    The CKM element is related to the off-diagonal resolvent at the
    generation's own energy:
    V_mn ~ Im G_mn(E_m + iη) / Im G_mm(E_m + iη)
    """
    print("\n" + "=" * 70)
    print("  METHOD 5: RESOLVENT / GREEN'S FUNCTION")
    print("  G_mn(E) = BZ-avg ⟨gen_m|(E-H)^{-1}|gen_n⟩")
    print("=" * 70)

    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
    gen_labels = ['1_s', 'w', 'w2']
    gen_vecs = [triv_s, gen_info['gen_w'], gen_info['gen_w2']]
    n_gen = len(gen_labels)

    E_gen = np.zeros(n_gen)
    for b in range(N_ATOMS):
        ov = abs(np.dot(np.conj(triv_s), gen_info['evecs'][:, b]))**2
        if ov > 0.4:
            E_gen[0] = gen_info['evals'][b]
            break
    E_gen[1] = gen_info['E_w']
    E_gen[2] = gen_info['E_w2']

    eta = 0.1  # broadening

    for e_idx, E_probe in enumerate(E_gen):
        G = np.zeros((n_gen, n_gen), dtype=complex)
        z = E_probe + 1j * eta
        n_k = 0

        for n1 in range(n_grid):
            for n2 in range(n_grid):
                for n3 in range(n_grid):
                    k = np.array([n1, n2, n3], dtype=float) / n_grid
                    H = bloch_H(k, bonds)
                    Ginv = z * np.eye(N_ATOMS) - H
                    Gk = la.inv(Ginv)
                    n_k += 1

                    for m in range(n_gen):
                        for n in range(n_gen):
                            G[m, n] += np.conj(gen_vecs[m]) @ Gk @ gen_vecs[n]

        G /= n_k

        print(f"\n  E_probe = {E_probe:.4f} (gen {gen_labels[e_idx]}), eta = {eta}")
        print(f"    |G_mn|:")
        for m in range(n_gen):
            row = "      "
            for n in range(n_gen):
                row += f"{abs(G[m,n]):12.8f} "
            print(row)

        # Ratio of off-diagonal to diagonal
        print(f"    Mixing ratios |G_mn/G_mm|:")
        for m in range(n_gen):
            for n in range(n_gen):
                if m != n:
                    ratio = abs(G[m,n]) / abs(G[m,m]) if abs(G[m,m]) > 1e-15 else 0
                    print(f"      |G[{gen_labels[m]},{gen_labels[n]}]/G[{gen_labels[m]},{gen_labels[m]}]| = {ratio:.8f}")

    return


# ══════════════════════════════════════════════════════════════════════
# METHOD 6: SPECTRAL FLOW ALONG HIGH-SYMMETRY LINES
# ══════════════════════════════════════════════════════════════════════

def method_spectral_flow(bonds, gen_info, n_pts=200):
    """
    Track generation content along paths AWAY from the (111) axis.
    The rate at which C₃ breaks determines the mixing strength.

    Paths:
      P → H: leaves (111) axis, C₃ breaks
      P → N: another symmetry-breaking path
    """
    print("\n" + "=" * 70)
    print("  METHOD 6: SPECTRAL FLOW (generation mixing off C₃ axis)")
    print("=" * 70)

    P = np.array([0.25, 0.25, 0.25])
    H = np.array([0.5, -0.5, 0.5])
    N = np.array([0.0, 0.0, 0.5])
    G = np.array([0.0, 0.0, 0.0])

    gen_w = gen_info['gen_w']
    gen_w2 = gen_info['gen_w2']
    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)

    paths = [
        (P, H, 'P->H'),
        (P, N, 'P->N'),
        (P, G, 'P->G (on C3 axis, control)'),
    ]

    for k_start, k_end, path_label in paths:
        print(f"\n  Path: {path_label}")
        ts = np.linspace(0, 1, n_pts)
        mix_w_w2 = np.zeros(n_pts)     # |⟨ω|ψ_α⟩|² where α tracks ω² character
        mix_w_triv = np.zeros(n_pts)

        for i, t in enumerate(ts):
            k = k_start + t * (k_end - k_start)
            evals, evecs = diag_H(k, bonds)

            # For each band, compute generation content
            ov_w = np.array([abs(np.dot(np.conj(gen_w), evecs[:, a]))**2 for a in range(N_ATOMS)])
            ov_w2 = np.array([abs(np.dot(np.conj(gen_w2), evecs[:, a]))**2 for a in range(N_ATOMS)])
            ov_ts = np.array([abs(np.dot(np.conj(triv_s), evecs[:, a]))**2 for a in range(N_ATOMS)])

            # Band with most ω² character: how much ω does it also have?
            band_w2 = np.argmax(ov_w2)
            mix_w_w2[i] = ov_w[band_w2]  # ω content of the "ω² band"

            # Band with most ω character: how much triv_s?
            band_w = np.argmax(ov_w)
            mix_w_triv[i] = ov_ts[band_w]

        print(f"    At start (t=0): ω content of ω² band = {mix_w_w2[0]:.8f}")
        print(f"    At t=0.25:      ω content of ω² band = {mix_w_w2[n_pts//4]:.8f}")
        print(f"    At t=0.5:       ω content of ω² band = {mix_w_w2[n_pts//2]:.8f}")
        print(f"    At end (t=1):   ω content of ω² band = {mix_w_w2[-1]:.8f}")
        print(f"    Max mixing:     {np.max(mix_w_w2):.8f}")
        print(f"    triv_s content of ω band at end: {mix_w_triv[-1]:.8f}")

    return


# ══════════════════════════════════════════════════════════════════════
# METHOD 7: NB WALK CONNECTION
# ══════════════════════════════════════════════════════════════════════

def method_nb_walk_connection(bonds, gen_info, n_grid=40):
    """
    The NB walk result: V_us = (2/3)^{2+√3} ≈ 0.2253
    uses λ₁ = 2-√3 (Ihara zeta spectral gap of K₄) and L_us = 1/λ₁ = 2+√3.

    Connection to Bloch bands:
    The NB walk matrix on K₄ has eigenvalues {2, ω·2, ω²·2, -1, -ω, -ω²}
    = {2, 2ω, 2ω², -1, -ω, -ω²} (6 eigenvalues for 6 directed edges).

    The Ihara zeta function relates to the Bloch Hamiltonian via:
    det(I - uT) = (1-u²)^{E-V} det(I - uA + u²(D-I))
    where A is the adjacency matrix.

    For K₄: A has eigenvalues {3,-1,-1,-1}, D = 3I.
    det factor = det(I - uA + 2u²I) = Π_i (1 - λ_i u + 2u²)

    At the Γ point of srs: H(Γ) = A(K₄) exactly.
    At general k: H(k) is a phase-twisted version of A(K₄).

    The spectral gap of the BZ-averaged resolvent should reproduce λ₁.
    """
    print("\n" + "=" * 70)
    print("  METHOD 7: NB WALK / IHARA ZETA CONNECTION")
    print("=" * 70)

    print(f"\n  NB walk prediction: V_us = (2/3)^{{2+sqrt(3)}} = {NB_Vus:.8f}")
    print(f"  PDG value:                                        {PDG_Vus:.8f}")
    print(f"  Ratio: {NB_Vus/PDG_Vus:.6f}")

    # Ihara zeta for K₄
    # T = NB walk matrix, eigenvalues: ±1, ±ω, ±ω² scaled by degree-1=2
    # Actually for K₄ (3-regular, 4 vertices, 6 edges → 12 directed edges):
    # NB walk matrix is 12×12. Ihara: det(I-uT) = (1-u²)^{E-V} Πᵢ(1-λᵢu+2u²)
    # where λᵢ are eigenvalues of A(K₄) = {3,-1,-1,-1}
    # So det factor = (1-3u+2u²)(1+u+2u²)³
    # = (1-u)(1-2u) · (1+u+2u²)³

    # Poles of 1/det(I-uT): u = 1/2 (from 1-2u), u = 1 (from 1-u)
    # u = (-1±i√7)/(4) (from 1+u+2u²)
    # Spectral radius of T = 2, so 1/ρ = 1/2.
    # λ₁ = 2-√3 comes from... the NB walk on the INFINITE srs graph, not K₄!

    # BZ-averaged spectral gap
    print(f"\n  Ihara zeta of K₄:")
    print(f"    A(K₄) eigenvalues: 3, -1, -1, -1")
    print(f"    Spectral gap: lambda_1 = 3 - (-1) = 4 (as adjacency gap)")
    print(f"    But NB walk lambda_1 = 2-sqrt(3) = {2-np.sqrt(3):.8f}")
    print(f"    This is the spectral gap of T (NB walk) on INFINITE srs,")
    print(f"    not on the finite K₄ graph.")

    # Compute BZ-averaged density of states
    n_E = 500
    E_range = np.linspace(-3.5, 3.5, n_E)
    dos = np.zeros(n_E)
    eta = 0.05
    n_k = 0

    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                evals, _ = diag_H(k, bonds)
                n_k += 1
                for ev in evals:
                    dos += eta / ((E_range - ev)**2 + eta**2) / np.pi

    dos /= n_k

    # Find band edges from DOS
    threshold = dos.max() * 0.01
    in_band = dos > threshold

    print(f"\n  DOS computed on {n_grid}³ grid")
    print(f"  Band edges (DOS > 1% of max):")
    edges = []
    for i in range(1, n_E):
        if in_band[i] != in_band[i-1]:
            edges.append(E_range[i])
            print(f"    E = {E_range[i]:.4f} ({'start' if in_band[i] else 'end'})")

    # Bandwidth
    evals_all = []
    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                evals, _ = diag_H(k, bonds)
                evals_all.extend(evals)
    evals_all = np.array(evals_all)

    print(f"\n  Band statistics:")
    print(f"    Global min E: {evals_all.min():.8f}")
    print(f"    Global max E: {evals_all.max():.8f}")
    print(f"    Bandwidth: {evals_all.max() - evals_all.min():.8f}")

    # Attempt: relate BZ spectral gap to NB walk spectral gap
    # For srs (3-regular): T_NB has spectral radius q = degree-1 = 2
    # Ramanujan bound: |λ| ≤ 2√(q-1) = 2 for q=2
    # Spectral gap: gap = q - |λ₁| where λ₁ is the second eigenvalue of T
    # For srs: the Bloch bands give H(k) eigenvalues in some range.
    # The NB walk matrix on the infinite graph has spectrum:
    # ζ^{-1}(u) = 0 iff det(I - uH(k) + 2u²I) = 0 for some k in BZ
    # i.e. u² - u·λᵢ(k)/(2) + 1 = 0 ... no, let me be more careful.

    # Ihara: det(I - uT) = Π_k det(I - u·H(k) + (q-1)u²·I) where q=degree-1=2
    # Zeros of det factor for band i: 1 - u·Eᵢ(k) + 2u² = 0
    # u = (Eᵢ(k) ± √(Eᵢ(k)²-8)) / 4

    # The smallest |u| zero (= 1/spectral_radius of T) comes from max |E|
    E_max = evals_all.max()  # should be 3 (at Gamma)
    u_min_candidate = (E_max - np.sqrt(max(E_max**2 - 8, 0))) / 4

    print(f"\n  Ihara zeta connection:")
    print(f"    For 3-regular graph: q = degree-1 = 2")
    print(f"    Ihara factor per band: 1 - u·E_i(k) + 2u² = 0")
    print(f"    u = (E ± sqrt(E²-8))/4")
    print(f"    E_max = {E_max:.4f} → u = ({E_max} - sqrt({E_max**2-8:.4f}))/4 = {u_min_candidate:.8f}")
    print(f"    1/u = {1/u_min_candidate:.8f} (spectral radius of T)")

    # The SECOND eigenvalue determines the gap
    # Need to find the second-largest |1/u| across the BZ
    # For each k, each band gives two reciprocal eigenvalue pairs of T
    all_u = []
    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                evals, _ = diag_H(k, bonds)
                for E in evals:
                    disc = E**2 - 8
                    if disc >= 0:
                        u1 = (E + np.sqrt(disc)) / 4
                        u2 = (E - np.sqrt(disc)) / 4
                        if abs(u1) > 1e-10:
                            all_u.append(1/u1)
                        if abs(u2) > 1e-10:
                            all_u.append(1/u2)
                    else:
                        # Complex conjugate pair
                        u_re = E / 4
                        u_im = np.sqrt(-disc) / 4
                        r = np.sqrt(u_re**2 + u_im**2)
                        if r > 1e-10:
                            all_u.append(1/r)  # magnitude of reciprocal

    all_u = np.array(sorted(set(np.round(np.abs(all_u), 8)), reverse=True))

    print(f"\n  NB walk eigenvalue magnitudes (from Ihara, top 10):")
    for i, u in enumerate(all_u[:10]):
        print(f"    |T eigenvalue| #{i}: {u:.8f}")

    if len(all_u) >= 2:
        rho = all_u[0]
        lam1 = all_u[1]
        gap = rho - lam1
        print(f"\n  Spectral radius ρ = {rho:.8f}")
        print(f"  Second eigenvalue |λ₁| = {lam1:.8f}")
        print(f"  Spectral gap ρ - |λ₁| = {gap:.8f}")
        print(f"  Target: 2 - sqrt(3) = {2-np.sqrt(3):.8f}")
        print(f"  Ratio: {gap/(2-np.sqrt(3)):.6f}")

        # V_us from this gap
        if gap > 0:
            L_us = 1/gap if gap < 1 else rho/gap
            V_us_bloch = (2/3)**L_us
            print(f"  L_us = {L_us:.8f}")
            print(f"  V_us = (2/3)^L_us = {V_us_bloch:.8f}")

    return dos, E_range


# ══════════════════════════════════════════════════════════════════════
# METHOD 8: DIRECT C₃ BREAKING MATRIX
# ══════════════════════════════════════════════════════════════════════

def method_c3_breaking(bonds, gen_info, n_grid=40):
    """
    Compute the C₃ breaking perturbation directly.

    At each k off the (111) axis, [H(k), C₃] ≠ 0.
    The commutator [H(k), C₃] in the generation basis gives
    the mixing Hamiltonian. BZ-average this.
    """
    print("\n" + "=" * 70)
    print("  METHOD 8: DIRECT C₃ BREAKING (commutator method)")
    print("  V_mix = BZ-avg |⟨gen_m|[H(k),C₃]|gen_n⟩|")
    print("=" * 70)

    gen_w = gen_info['gen_w']
    gen_w2 = gen_info['gen_w2']
    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)
    triv_0 = np.array([1, 0, 0, 0], dtype=complex)

    # Use all 4 P-point states as basis
    basis_labels = ['triv_0', 'triv_s', 'w', 'w2']
    basis = [triv_0, triv_s, gen_w, gen_w2]
    n_b = len(basis)

    comm_avg = np.zeros((n_b, n_b), dtype=complex)
    comm2_avg = np.zeros((n_b, n_b))
    n_k = 0

    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                H = bloch_H(k, bonds)
                comm = H @ C3_PERM - C3_PERM @ H
                n_k += 1

                for m in range(n_b):
                    for n in range(n_b):
                        val = np.conj(basis[m]) @ comm @ basis[n]
                        comm_avg[m, n] += val
                        comm2_avg[m, n] += abs(val)**2

    comm_avg /= n_k
    comm2_avg /= n_k

    print(f"\n  Grid: {n_grid}³ = {n_k} k-points")

    print(f"\n  BZ-averaged |[H,C3]| in generation basis:")
    for m in range(n_b):
        row = "    "
        for n in range(n_b):
            row += f"{abs(comm_avg[m,n]):12.8f} "
        print(f"    {basis_labels[m]:8s} " + row)

    print(f"\n  RMS |[H,C3]| in generation basis:")
    for m in range(n_b):
        row = "    "
        for n in range(n_b):
            row += f"{np.sqrt(comm2_avg[m,n]):12.8f} "
        print(f"    {basis_labels[m]:8s} " + row)

    # The generation-generation block
    gen_block_rms = np.sqrt(comm2_avg[2:, 2:])
    print(f"\n  Generation subblock (w, w2) RMS [H,C3]:")
    print(f"    |[H,C3]|_ww   = {gen_block_rms[0,0]:.8f}")
    print(f"    |[H,C3]|_ww2  = {gen_block_rms[0,1]:.8f}")
    print(f"    |[H,C3]|_w2w2 = {gen_block_rms[1,1]:.8f}")

    # S₃ breaking: are the off-diagonal elements between different pairs UNEQUAL?
    print(f"\n  Off-diagonal RMS values:")
    print(f"    triv_s ↔ w:   {np.sqrt(comm2_avg[1,2]):.8f}")
    print(f"    triv_s ↔ w2:  {np.sqrt(comm2_avg[1,3]):.8f}")
    print(f"    w ↔ w2:       {np.sqrt(comm2_avg[2,3]):.8f}")
    vals = [np.sqrt(comm2_avg[1,2]), np.sqrt(comm2_avg[1,3]), np.sqrt(comm2_avg[2,3])]
    if min(vals) > 1e-10:
        print(f"    Ratio max/min: {max(vals)/min(vals):.6f}")
    print(f"    (If > 1, S₃ is broken → unequal CKM elements)")

    return comm_avg, comm2_avg


# ══════════════════════════════════════════════════════════════════════
# METHOD 9: PERTURBATIVE CKM FROM BAND MIXING
# ══════════════════════════════════════════════════════════════════════

def method_perturbative_ckm(bonds, gen_info, n_grid=40):
    """
    Treat the off-axis Hamiltonian as a perturbation on the P-point Hamiltonian.

    H(k) = H(P) + δH(k),  where δH = H(k) - H(P)

    First-order mixing: V_mn^(1) = ⟨gen_m|δH|gen_n⟩ / (E_m - E_n)
    BZ-average: V_mn = (1/N_k) Σ_k |⟨gen_m|δH(k)|gen_n⟩| / |E_m - E_n|
    """
    print("\n" + "=" * 70)
    print("  METHOD 9: PERTURBATIVE CKM (1st order band mixing)")
    print("  V_mn = BZ-avg |⟨gen_m|δH(k)|gen_n⟩| / |E_m - E_n|")
    print("=" * 70)

    H_P = bloch_H([0.25, 0.25, 0.25], bonds)

    gen_w = gen_info['gen_w']
    gen_w2 = gen_info['gen_w2']
    triv_s = np.array([0, 1, 1, 1], dtype=complex) / np.sqrt(3)

    gen_labels = ['1_s', 'w', 'w2']
    gen_vecs = [triv_s, gen_w, gen_w2]
    n_gen = len(gen_labels)

    # Energy denominators
    E_gen = np.zeros(n_gen)
    for b in range(N_ATOMS):
        ov = abs(np.dot(np.conj(triv_s), gen_info['evecs'][:, b]))**2
        if ov > 0.4:
            E_gen[0] = gen_info['evals'][b]
            break
    E_gen[1] = gen_info['E_w']
    E_gen[2] = gen_info['E_w2']

    print(f"  Generation energies at P: {E_gen}")
    for m in range(n_gen):
        for n in range(m+1, n_gen):
            print(f"    |E_{gen_labels[m]} - E_{gen_labels[n]}| = {abs(E_gen[m] - E_gen[n]):.8f}")

    # BZ average
    dH_mn_avg = np.zeros((n_gen, n_gen))
    dH_mn_rms = np.zeros((n_gen, n_gen))
    V_pert = np.zeros((n_gen, n_gen))
    n_k = 0

    for n1 in range(n_grid):
        for n2 in range(n_grid):
            for n3 in range(n_grid):
                k = np.array([n1, n2, n3], dtype=float) / n_grid
                H = bloch_H(k, bonds)
                dH = H - H_P
                n_k += 1

                for m in range(n_gen):
                    for n in range(n_gen):
                        val = np.conj(gen_vecs[m]) @ dH @ gen_vecs[n]
                        dH_mn_avg[m, n] += abs(val)
                        dH_mn_rms[m, n] += abs(val)**2
                        if m != n:
                            dE = abs(E_gen[m] - E_gen[n])
                            if dE > 1e-10:
                                V_pert[m, n] += abs(val) / dE

    dH_mn_avg /= n_k
    dH_mn_rms = np.sqrt(dH_mn_rms / n_k)
    V_pert /= n_k

    print(f"\n  BZ-averaged |⟨m|δH|n⟩|:")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{dH_mn_avg[m,n]:12.8f} "
        print(f"    {gen_labels[m]:4s} " + row)

    print(f"\n  RMS ⟨m|δH|n⟩:")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{dH_mn_rms[m,n]:12.8f} "
        print(f"    {gen_labels[m]:4s} " + row)

    print(f"\n  Perturbative mixing V_mn = avg |⟨m|δH|n⟩|/|E_m-E_n|:")
    for m in range(n_gen):
        row = "    "
        for n in range(n_gen):
            row += f"{V_pert[m,n]:12.8f} "
        print(f"    {gen_labels[m]:4s} " + row)

    # CKM identification
    print(f"\n  CKM identification (perturbative):")
    print(f"    V_us ~ V[1_s,w2] = {V_pert[0,2]:.8f}  (PDG: {PDG_Vus})")
    print(f"    V_cb ~ V[w2,w]   = {V_pert[2,1]:.8f}  (PDG: {PDG_Vcb})")
    print(f"    V_ub ~ V[1_s,w]  = {V_pert[0,1]:.8f}  (PDG: {PDG_Vub})")

    # Check hierarchy
    vals = sorted([V_pert[0,2], V_pert[2,1], V_pert[0,1]], reverse=True)
    if all(v > 1e-10 for v in vals):
        print(f"\n  Hierarchy check:")
        print(f"    Largest:  {vals[0]:.8f}")
        print(f"    Middle:   {vals[1]:.8f}")
        print(f"    Smallest: {vals[2]:.8f}")
        print(f"    Ratios: {vals[0]/vals[1]:.4f} : 1 : {vals[2]/vals[1]:.4f}")
        print(f"    PDG ratios: {PDG_Vus/PDG_Vcb:.4f} : 1 : {PDG_Vub/PDG_Vcb:.4f}")

    return V_pert


# ══════════════════════════════════════════════════════════════════════
# SUMMARY AND COMPARISON
# ══════════════════════════════════════════════════════════════════════

def summary():
    """Print reference values and NB walk prediction."""
    print("\n" + "=" * 70)
    print("  REFERENCE VALUES")
    print("=" * 70)
    print(f"  PDG CKM magnitudes:")
    print(f"    |V_us| = {PDG_Vus}")
    print(f"    |V_cb| = {PDG_Vcb}")
    print(f"    |V_ub| = {PDG_Vub}")
    print(f"    delta_CP = {PDG_delta_CP} deg")
    print(f"    J = {PDG_J}")
    print(f"\n  NB walk prediction:")
    print(f"    V_us = (2/3)^{{2+sqrt(3)}} = {NB_Vus:.8f}")
    print(f"    lambda_1 = 2-sqrt(3) = {2-np.sqrt(3):.8f}")
    print(f"    L_us = 1/lambda_1 = 2+sqrt(3) = {2+np.sqrt(3):.8f}")
    print(f"\n  Wolfenstein parametrization:")
    lam = PDG_Vus
    A = PDG_Vcb / lam**2
    print(f"    lambda = {lam}")
    print(f"    A = V_cb/lambda^2 = {A:.4f}")
    print(f"    lambda^2 = {lam**2:.6f}")
    print(f"    lambda^3 = {lam**3:.6f}  (V_ub ~ lambda^3 A)")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CKM FROM C₃ MIXING ON THE SRS LATTICE (I4₁32)")
    print("  Generation = C₃ eigenvalue at P = (1/4,1/4,1/4)")
    print("  CKM = generation mixing off the (111) axis")
    print("=" * 70)

    summary()

    # Build
    print("\n\nBuilding lattice...")
    bonds = find_bonds()
    print(f"  {len(bonds)} bonds found")

    # Get P-point generation states
    gen_info = get_generation_states(bonds)
    print(f"\n  P-point eigenvalues: {gen_info['evals']}")
    print(f"  P-point C₃ labels:  {gen_info['labels']}")
    print(f"  E(ω) = {gen_info['E_w']:.8f}, E(ω²) = {gen_info['E_w2']:.8f}")
    print(f"  Generation splitting: {gen_info['E_w'] - gen_info['E_w2']:.8f} = 2√3 = {2*np.sqrt(3):.8f}")

    # Run all methods
    V1, V1_norm, gen_labels = method_spectral_overlap(bonds, gen_info, n_grid=30)
    V2 = method_amplitude_overlap(bonds, gen_info, n_grid=30)
    V3_lo, V3_hi = method_band_restricted(bonds, gen_info, n_grid=30)
    method_energy_weighted(bonds, gen_info, n_grid=25)
    method_resolvent(bonds, gen_info, n_grid=25)
    method_spectral_flow(bonds, gen_info, n_pts=200)
    dos, E_range = method_nb_walk_connection(bonds, gen_info, n_grid=25)
    comm_avg, comm2_avg = method_c3_breaking(bonds, gen_info, n_grid=30)
    V_pert = method_perturbative_ckm(bonds, gen_info, n_grid=30)

    # ═══════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON: ALL METHODS vs PDG")
    print("=" * 70)

    print(f"\n  {'Method':<40s} {'V_us':>10s} {'V_cb':>10s} {'V_ub':>10s} {'Hier?':>8s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'PDG':<40s} {PDG_Vus:10.6f} {PDG_Vcb:10.6f} {PDG_Vub:10.6f} {'YES':>8s}")
    print(f"  {'NB walk (2/3)^(2+sqrt3)':<40s} {NB_Vus:10.6f} {'---':>10s} {'---':>10s} {'---':>8s}")

    # Method 1
    v_us = V1_norm[0,2]
    v_cb = V1_norm[1,2]
    v_ub = V1_norm[0,1]
    hier = "YES" if v_us > v_cb > v_ub else "NO"
    print(f"  {'M1: Spectral overlap (norm)':<40s} {v_us:10.6f} {v_cb:10.6f} {v_ub:10.6f} {hier:>8s}")

    # Method 3 (lower bands)
    v_us = abs(V3_lo[0,2])
    v_cb = abs(V3_lo[1,2])
    v_ub = abs(V3_lo[0,1])
    hier = "YES" if v_us > v_cb > v_ub else "NO"
    print(f"  {'M3: Band-restricted (lower)':<40s} {v_us:10.6f} {v_cb:10.6f} {v_ub:10.6f} {hier:>8s}")

    # Method 3 (upper bands)
    v_us = abs(V3_hi[0,2])
    v_cb = abs(V3_hi[1,2])
    v_ub = abs(V3_hi[0,1])
    hier = "YES" if v_us > v_cb > v_ub else "NO"
    print(f"  {'M3: Band-restricted (upper)':<40s} {v_us:10.6f} {v_cb:10.6f} {v_ub:10.6f} {hier:>8s}")

    # Method 9 (perturbative)
    v_us = V_pert[0,2]
    v_cb = V_pert[2,1]
    v_ub = V_pert[0,1]
    hier = "YES" if v_us > v_cb > v_ub else "NO"
    print(f"  {'M9: Perturbative mixing':<40s} {v_us:10.6f} {v_cb:10.6f} {v_ub:10.6f} {hier:>8s}")

    # C₃ breaking (method 8)
    v1 = np.sqrt(comm2_avg[1,2])  # triv_s ↔ w
    v2 = np.sqrt(comm2_avg[1,3])  # triv_s ↔ w2
    v3 = np.sqrt(comm2_avg[2,3])  # w ↔ w2
    print(f"  {'M8: C3 breaking [H,C3] (RMS)':<40s} {v2:10.6f} {v3:10.6f} {v1:10.6f} {'---':>8s}")

    print(f"\n  KEY QUESTION: Does any method show V_us >> V_cb >> V_ub?")
    print(f"  (The Wolfenstein hierarchy: ~0.22 >> ~0.04 >> ~0.004)")

    # Plot DOS
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(E_range, dos, 'b-', linewidth=1.5)
    ax.set_xlabel('Energy')
    ax.set_ylabel('DOS (arb. units)')
    ax.set_title('SRS density of states (BZ-averaged)')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.sqrt(3), color='red', ls='--', alpha=0.5, label='E(ω)=√3')
    ax.axvline(-np.sqrt(3), color='blue', ls='--', alpha=0.5, label='E(ω²)=-√3')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'srs_ckm_dos.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  DOS plot saved: {path}")


if __name__ == '__main__':
    main()
