"""
srs substrate data — the framework's MDL-dominant Cayley graph.

srs is the (10,3) chiral 3-coordination crystal net at Wyckoff 8a positions
in space group I4_132. Per the framework's existing apparatus, srs is the
unique MDL-dominant substrate at framework scale N_hub ~ 10^60 (subdominant
substrates are below waterline at this scale).

Structural primitives (all derived from substrate; not parameters):
  k*    = 3   (coordination number per vertex)
  |V|   = 4   (atoms per primitive cell, K_4 quotient)
  |E|   = 6   (undirected edges per primitive cell)
  2|E|  = 12  (directed edges, Hashimoto domain dimension)
  g     = 10  (girth — smallest cycle length on the lattice)

This module wraps the existing substrate machinery in proofs/common.py and
proofs/cosmology/srs_photon_bloch_primitive.py rather than re-implementing.
The kernel layer (kernel.py) provides counting-first primitives on top of
this substrate data.
"""

import sys
import math
from pathlib import Path
from fractions import Fraction
from functools import cached_property

import numpy as np
from numpy import linalg as la

# Wire up to the existing framework apparatus
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import (
    K_STAR, GIRTH, N_ATOMS, A_PRIM, ATOMS, NN_DIST,
    omega3, h_P, C3_PERM, C3_ESTATES,
    find_bonds, bloch_H, diag_H, c3_decompose, label_c3,
)


# ============================================================================
# srs SUBSTRATE — fixed structural data + lazy-loaded derived objects
# ============================================================================

class SrsSubstrate:
    """The framework's MDL-dominant substrate.

    All structural counts are fixed (derived from substrate, not parameters).
    Derived objects (bonds, Bloch operators) are lazy-loaded from the existing
    proofs/common.py apparatus.

    Usage:
        substrate = SrsSubstrate()
        substrate.k_star  # 3
        substrate.n_atoms  # 4
        substrate.adjacency_at_k('P')  # 4x4 complex matrix
        substrate.hashimoto_at_k('P')  # 12x12 complex matrix
    """

    # --- Structural counts (immutable; derived from substrate per framework) ---
    K_STAR = K_STAR        # 3 — coordination per vertex
    N_ATOMS = N_ATOMS      # 4 — |V| per primitive cell
    N_EDGES = 6            # |E| per primitive cell (undirected)
    N_DIRECTED = 12        # 2|E| — directed edges (Hashimoto domain)
    GIRTH = GIRTH          # 10 — smallest cycle on the lattice
    D_SPATIAL = 3          # 3 spatial dimensions (from k* via Coxeter)

    # --- High-symmetry k-points in the BCC primitive Brillouin zone ---
    K_POINTS = {
        'Gamma': (0.0, 0.0, 0.0),
        'P':     (0.25, 0.25, 0.25),
        'N':     (0.0, 0.5, 0.0),
        'H':     (-0.5, 0.5, 0.5),
    }

    # --- Class-dependent closure rates at the Ramanujan saddle h ---
    # Per dark-extraction map and theorem_dark_map_class2_closure
    # h = (√3 + i√5)/2 with |h|² = k* − 1 = 2
    @cached_property
    def ramanujan_eigenvalue_at_P(self):
        """h = (√3 + i√5)/2 — substrate Hashimoto eigenvalue at P-point."""
        return complex(math.sqrt(3) / 2, math.sqrt(5) / 2)

    @cached_property
    def closure_rate_amplitude(self):
        """ν_amp = |Im(h)|/|h|² = √5/4 — Class-1 amplitude-class closure rate."""
        h = self.ramanujan_eigenvalue_at_P
        return abs(h.imag) / (abs(h) ** 2)

    @cached_property
    def closure_rate_mass_squared(self):
        """ν_mass² = tan²(arg h) = 5/3 — Class-2 mass²-class closure rate."""
        h = self.ramanujan_eigenvalue_at_P
        arg_h = math.atan2(h.imag, h.real)
        return math.tan(arg_h) ** 2

    @cached_property
    def closure_rate_edge_local(self):
        """ν_edge = 1 — Class-3 edge-local closure rate."""
        return 1.0

    # --- Derived spectral data ---
    @cached_property
    def adjacency_perron(self):
        """λ_max(A) = k* = 3 — adjacency Perron eigenvalue."""
        return self.K_STAR

    @cached_property
    def hashimoto_perron(self):
        """λ_max(B) = k* − 1 = 2 — Hashimoto Perron eigenvalue.
        Asymptotic NB-walk survival ratio per step."""
        return self.K_STAR - 1

    @cached_property
    def nb_survival_per_step(self):
        """Per-step NB walk survival probability = (k* − 1)/k* = 2/3."""
        return Fraction(self.K_STAR - 1, self.K_STAR)

    # --- Lazy-loaded bond structure (from proofs/common) ---
    @cached_property
    def bonds(self):
        """List of NN bonds in the primitive cell as (src, tgt, cell_offset).

        Each bond is (atom_i, atom_j, (n1, n2, n3)) where (n1, n2, n3) is
        the Bloch translation offset to atom_j's primitive cell.
        |bonds| = 12 (each undirected edge counted both ways).
        """
        return find_bonds()

    @cached_property
    def n_bonds_directed(self):
        """Number of directed bonds = 2 |E| = 12 for srs."""
        return len(self.bonds)

    # --- Bloch operators ---
    def adjacency_at_k(self, k_label_or_frac):
        """4x4 Bloch adjacency A(k) at fractional momentum or named point."""
        k_frac = self._resolve_k(k_label_or_frac)
        return bloch_H(k_frac, self.bonds)

    def hashimoto_at_k(self, k_label_or_frac):
        """12x12 Bloch Hashimoto B(k) at fractional momentum or named point.

        B(k) acts on the directed-edge space. Per the Bloch lift:
          B(k)[f, e] = [e.target = f.source] · [f ≠ rev(e)] · exp(-2πi k·e.cell)
        """
        k_frac = self._resolve_k(k_label_or_frac)
        bonds = self.bonds
        n = self.n_bonds_directed
        B = np.zeros((n, n), dtype=complex)
        k = np.asarray(k_frac, dtype=float)
        for e_idx, (e_src, e_tgt, e_cell) in enumerate(bonds):
            for f_idx, (f_src, f_tgt, f_cell) in enumerate(bonds):
                if f_src != e_tgt:
                    continue
                # Non-backtracking condition: f is not the reverse of e
                rev_cell = tuple(-c for c in e_cell)
                if (f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell):
                    continue
                phase = np.exp(-1j * 2 * np.pi * np.dot(k, np.array(e_cell)))
                B[f_idx, e_idx] += phase
        return B

    def adjacency_spectrum_at_k(self, k_label_or_frac):
        """Sorted real eigenvalues of A(k)."""
        k_frac = self._resolve_k(k_label_or_frac)
        evals, _ = diag_H(k_frac, self.bonds)
        return evals

    @cached_property
    def _bond_displacements_exact(self):
        """Exact rational displacement vectors r_tgt - r_src + cell·a_prim per bond.

        Used for Cartesian-k Bloch construction (cf.
        proofs/foundations/lorentz_sig_h_lv_coefficients.py). The atom and
        primitive-vector positions are the I4_132 Wyckoff-8a srs realization.
        """
        atoms_exact = [
            (Fraction(1, 8), Fraction(1, 8), Fraction(1, 8)),
            (Fraction(3, 8), Fraction(7, 8), Fraction(5, 8)),
            (Fraction(7, 8), Fraction(5, 8), Fraction(3, 8)),
            (Fraction(5, 8), Fraction(3, 8), Fraction(7, 8)),
        ]
        a_prim_exact = [
            (Fraction(-1, 2), Fraction(1, 2), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(-1, 2), Fraction(1, 2)),
            (Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2)),
        ]
        out = []
        for src, tgt, cell in self.bonds:
            r = []
            for d in range(3):
                v = atoms_exact[tgt][d] - atoms_exact[src][d]
                for i in range(3):
                    v += cell[i] * a_prim_exact[i][d]
                r.append(v)
            out.append(tuple(r))
        return out

    def adjacency_at_k_cartesian(self, k_cart):
        """4×4 Bloch H(k) at Cartesian wavevector k_cart (lattice-constant units).

        Uses the displacement gauge (cf.
        proofs/foundations/lorentz_sig_h_lv_coefficients.py):
            H[tgt, src] = sum_{(src,tgt,cell)} exp(i k_cart · r_disp).
        This is the convention in which the Perron-band Taylor coefficients
        are D_H = 1/16, D4_iso^H = -1/1024, D4_aniso^H = +1/1536 in the
        framework's published symbolic computation.
        """
        kx, ky, kz = float(k_cart[0]), float(k_cart[1]), float(k_cart[2])
        H = np.zeros((self.N_ATOMS, self.N_ATOMS), dtype=complex)
        for i, (src, tgt, _cell) in enumerate(self.bonds):
            r = self._bond_displacements_exact[i]
            phase = np.exp(1j * (kx * float(r[0]) + ky * float(r[1]) + kz * float(r[2])))
            H[tgt, src] += phase
        return H

    def adjacency_at_k_cartesian_mp(self, k_cart, prec=200):
        """High-precision (mpmath) version of adjacency_at_k_cartesian.

        Returns an mpmath matrix; precision in bits. Used by the kernel's
        bloch_taylor_at_gamma primitive so that Taylor coefficients can be
        extracted at enough precision to recognize their exact rational form.
        """
        import mpmath as mp
        mp.mp.prec = prec
        H = mp.matrix(self.N_ATOMS, self.N_ATOMS)
        for i, (src, tgt, _cell) in enumerate(self.bonds):
            r = self._bond_displacements_exact[i]
            r_mp = [mp.mpf(rr.numerator) / mp.mpf(rr.denominator) for rr in r]
            arg = sum(k_cart[d] * r_mp[d] for d in range(3))
            phase = mp.exp(mp.mpc(0, 1) * arg)
            H[tgt, src] = H[tgt, src] + phase
        return H

    def c3_isotypic_decomposition_at_P(self):
        """C₃-isotypic decomposition of V_Ram at the P-point.

        DERIVED, not hardcoded: diagonalizes the C₃ permutation matrix on
        the adjacency eigenspaces at P via the simultaneous diagonalization
        machinery in proofs/common.py:c3_decompose, then doubles to V_Ram
        per the Stark-Terras factorization.

        For srs: the 4-dim adjacency at P has C₃ multiplicities (2, 1, 1);
        V_Ram is 8-dim = 2× the adjacency basis (each adjacency mode
        contributes both directed-edge versions per Stark-Terras), so V_Ram
        has C₃ multiplicities (4, 2, 2).
        """
        # Resolve k point as fractional (P = (1/4, 1/4, 1/4))
        k_frac = self._resolve_k('P')
        evals, evecs, c3_diag, _offdiag = c3_decompose(k_frac, self.bonds)

        # Count C₃ eigenvalue multiplicities across the adjacency basis (4-dim)
        labels = [label_c3(c3_val) for c3_val in c3_diag]
        mu_trivial_adj = labels.count('1')
        mu_omega_adj = labels.count('w')
        mu_omega_bar_adj = labels.count('w2')

        # V_Ram is 8-dim = 2 × 4-dim adjacency (Stark-Terras factorization
        # gives 2 directed-edge copies per adjacency mode). The doubling is
        # structural and preserves C₃ multiplicities.
        return (
            2 * mu_trivial_adj,
            2 * mu_omega_adj,
            2 * mu_omega_bar_adj,
        )

    # --- Helpers ---
    def _resolve_k(self, k_label_or_frac):
        """Accept either a high-symmetry name ('Gamma', 'P', etc.) or a tuple."""
        if isinstance(k_label_or_frac, str):
            return self.K_POINTS[k_label_or_frac]
        return tuple(k_label_or_frac)


# ============================================================================
# Convenience constants exposed at module level
# ============================================================================

K_STAR_SRS = K_STAR        # 3
N_ATOMS_SRS = N_ATOMS      # 4
N_EDGES_SRS = 6            # |E| undirected
GIRTH_SRS = GIRTH          # 10
H_RAMANUJAN = h_P          # (√3 + i√5)/2
