"""
srs-z substrate data — the (10,3)-b net, srs's nearest competitor (R-9).

PURPOSE: this is NOT a production-substrate module. It exists so the
*entire* simulator/ + match/ stack (the enumerated-dynamics machinery) can
be run on srs-z exactly as it is run on srs, to settle R-9 by computation
rather than by abstract MDL accounting. See
`proofs/foundations/r9_srsz_simulator_run.py` for the driver, and
`docs/audits/registers/structural_residue_register.md` R-9.

WHAT srs-z IS (verified — see r9_srsz_simulator_run.py):
  * RCSR net "srs-z" = the (10,3)-b net, space group P4_132 (#213), cubic,
    8 atoms at Wyckoff 8c (x,x,x) with x ≈ 0.6607, 12 undirected edges,
    3-regular, girth 10.
  * Same coordination sequence and vertex symbol as srs (this is why it is
    the canonical competitor).
  * Primitive-cell quotient graph ≅ Q_3 (the 3-cube), NOT K_4 (srs's
    quotient). srs-z is BIPARTITE; srs is not. The bipartite 2-colouring is
    the Z_2 grading χ̃ that the framework's `srs_z_chi_*` probes found —
    srs-z is "the bipartite cousin of srs", which is exactly the structure
    that gives it a Witten-SUSY-QM grading srs lacks.

EVERYTHING spectral is COMPUTED from srs-z's own Bloch operator. Nothing
is inherited from srs — in particular srs's Ramanujan saddle
h_P = (√3+i√5)/2, its closure rates √5/4 and 5/3, and its C_3 multiplicities
are NOT assumed; the srs-z analogues are recomputed and may differ.

The bond list (i, j, c) — vertex i in the home cell bonds to vertex j in
cell c ∈ Z^3 of srs-z's *primitive cubic* lattice — is constructed once
from spglib's verified P4_132 operations + RCSR's Wyckoff data, in
`_build_srsz_bonds()`.
"""

import math
from fractions import Fraction
from functools import cached_property
from itertools import product

import numpy as np
from numpy import linalg as la


# ============================================================================
# srs-z net construction (spglib P4_132 ops + RCSR Wyckoff 8c / 12d)
# ============================================================================

_X_VERT = 0.6607          # RCSR srs-z vertex 8c free parameter
_E_CENTER = (0.9643, 0.2143, 0.625)   # RCSR srs-z edge-orbit representative (12d)


def _p4132_ops():
    """24 (rotation, translation) operations of P4_132 (#213) from spglib."""
    import spglib
    sym = spglib.get_symmetry_from_database(509)   # hall 509 = P4_132 (#213)
    return list(sym['rotations']), list(sym['translations'])


def _orbit(p, rots, trans, tol=1e-5):
    pts = []
    for R, t in zip(rots, trans):
        q = (R @ np.asarray(p, float) + t) % 1.0
        q = np.round(q, 8) % 1.0
        if not any(np.allclose(q, e, atol=tol) for e in pts):
            pts.append(q)
    return pts


def _build_srsz_bonds():
    """Return (verts, bonds, c3_perm, bipartite_colors).

    verts          : list of 8 fractional positions (the Wyckoff 8c orbit)
    bonds          : list of (i, j, (c1,c2,c3)) directed bonds, vertex i in
                     home cell, vertex j in cell c. |bonds| = 24.  Symmetric
                     (reverse of every bond present).
    c3_perm        : 8x8 permutation matrix of the [111] 3-fold on the verts
    bipartite_colors : length-8 array of 0/1 (the Q_3 2-colouring)
    """
    rots, trans = _p4132_ops()
    verts = _orbit([_X_VERT, _X_VERT, _X_VERT], rots, trans)
    ecenters = _orbit(_E_CENTER, rots, trans)
    ec_mod1 = [np.round(e, 6) % 1.0 for e in ecenters]

    def is_ecenter(p):
        p = np.round(p, 6) % 1.0
        return any(np.allclose(p, e, atol=1e-3) for e in ec_mod1)

    shifts = list(product(range(-2, 3), repeat=3))
    bonds = []
    for i in range(8):
        nbrs = []
        for j in range(8):
            for s in shifts:
                vj = verts[j] + np.array(s)
                if is_ecenter((verts[i] + vj) / 2.0):
                    nbrs.append((j, tuple(int(c) for c in s),
                                 la.norm(vj - verts[i])))
        if nbrs:
            dmin = min(n[2] for n in nbrs)
            nbrs = [n for n in nbrs if abs(n[2] - dmin) < 1e-3]
        assert len(nbrs) == 3, f"vertex {i}: {len(nbrs)} bonds (expected 3)"
        for (j, s, _d) in nbrs:
            bonds.append((i, j, s))

    # --- C_3 ([111] 3-fold) permutation of the 8 vertices ---
    # find the spglib op whose rotation is the cyclic permutation (z,x,y)<-(x,y,z)
    C3R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    c3_op = None
    for R, t in zip(rots, trans):
        if np.array_equal(R, C3R):
            c3_op = (R, t)
            break
    assert c3_op is not None, "no [111] 3-fold found in P4_132 ops"
    R, t = c3_op
    c3_perm = np.zeros((8, 8))
    for i in range(8):
        q = (R @ verts[i] + t) % 1.0
        q = np.round(q, 6) % 1.0
        for k in range(8):
            if np.allclose(q, np.round(verts[k], 6) % 1.0, atol=1e-3):
                c3_perm[k, i] = 1.0     # P[k,i]=1: C3 maps vertex i -> vertex k
                break
        else:
            raise AssertionError(f"C3 image of vertex {i} not in orbit")

    # --- bipartite 2-colouring of the quotient (Q_3) ---
    qadj = {i: set() for i in range(8)}
    for (i, j, _c) in bonds:
        qadj[i].add(j)
    colors = [-1] * 8
    colors[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in qadj[u]:
            if colors[v] == -1:
                colors[v] = 1 - colors[u]
                stack.append(v)
            assert colors[v] != colors[u], "quotient not bipartite!"
    return verts, bonds, c3_perm, np.array(colors)


# Build once at import.
_VERTS, _BONDS, _C3_PERM, _BIP_COLORS = _build_srsz_bonds()


def _omega3():
    return np.exp(2j * np.pi / 3)


def _label_c3(c3_val):
    w = _omega3()
    if abs(c3_val - 1.0) < 0.3:
        return '1'
    if abs(c3_val - w) < 0.3:
        return 'w'
    if abs(c3_val - w ** 2) < 0.3:
        return 'w2'
    return '?'


# ============================================================================
# SrsZSubstrate — duck-types simulator.srs_substrate.SrsSubstrate
# ============================================================================

class SrsZSubstrate:
    """srs-z (the (10,3)-b net) as a drop-in substrate for the simulator.

    All structural counts are fixed (derived from the net). All spectral
    quantities are COMPUTED from srs-z's own Bloch operator; nothing is
    inherited from srs.

    Usage:
        from . import CountingKernel
        from .srsz_substrate import SrsZSubstrate
        K = CountingKernel(substrate=SrsZSubstrate())
    """

    # --- structural counts (the net, not parameters) ---
    K_STAR = 3            # coordination per vertex (same family as srs)
    N_ATOMS = 8           # |V| per primitive cell (Q_3 quotient)
    N_EDGES = 12          # |E| per primitive cell (undirected)
    N_DIRECTED = 24       # 2|E| — Hashimoto domain
    GIRTH = 10            # smallest cycle on the lattice
    D_SPATIAL = 3         # 3 spatial dimensions

    # high-symmetry k-points: srs-z is on a PRIMITIVE CUBIC lattice (P4_132),
    # so its BZ is a cube — Γ, X, M, R — plus the C_3-stabilised body-diagonal
    # point (1/4,1/4,1/4) (the closest analogue of srs's P-point: it lies on
    # the [111] axis where [H(k), C_3] = 0).
    K_POINTS = {
        'Gamma': (0.0, 0.0, 0.0),
        'X':     (0.5, 0.0, 0.0),
        'M':     (0.5, 0.5, 0.0),
        'R':     (0.5, 0.5, 0.5),       # BZ corner — carries the protected h (srs P-point analogue)
        'P':     (0.25, 0.25, 0.25),    # interior [111] point — C_3 acts but NO degeneracy here
    }

    # which K_POINTS entry plays the role srs's P-point plays: the BZ corner
    # with the C_3-protected, Ramanujan-saturating degenerate mode. For srs
    # that is its BCC corner (1/4,1/4,1/4); for srs-z it is the primitive-cubic
    # corner R = (1/2,1/2,1/2) (verified: λ=√3, mult 4; the interior (1/4)³
    # point has *no* degeneracy, unlike srs's).
    P_ANALOG = 'R'

    # bipartite (Witten-SUSY-QM) grading operator χ̃ = diag(±1) on the 8 atoms
    @cached_property
    def chi_tilde_quotient(self):
        """8x8 diagonal Z_2 grading from the Q_3 2-colouring (±1)."""
        return np.diag(1.0 - 2.0 * _BIP_COLORS)

    # --- closure rates: COMPUTED from srs-z's Ramanujan saddle, not inherited ---
    @cached_property
    def ramanujan_eigenvalue_at_P(self):
        """The C_3-protected Ramanujan-saturating Hashimoto eigenvalue, found
        by scanning ALL of srs-z's high-symmetry k-points (the analogue of
        srs's P-point is whichever point carries the protected degenerate
        |u|²=k*-1 mode — for srs-z this turns out to be the BZ corner R, not
        the interior point (1/4,1/4,1/4)).

        Returned with Im(u)>0, Re(u)>0 (srs analogue: h = (√3+i√5)/2). Raises
        if srs-z has NO protected Ramanujan eigenvalue at any k-point.
        """
        u, _info = self._ramanujan_saddle_info()
        return u

    @cached_property
    def ramanujan_structure(self):
        """Per-k-point report of the Ramanujan structure on srs-z.

        For each high-symmetry k-point: the adjacency spectrum, whether it
        carries a degenerate |u|²=k*-1 Hashimoto mode, that mode's
        multiplicity, and its C_3-isotypic content. Distinguishes srs-z from
        srs structurally (srs: protected mult-2 h at P; srs-z: see output).
        """
        return self._scan_ramanujan_all_k()

    @cached_property
    def closure_rate_amplitude(self):
        """ν_amp = |Im h|/|h|^2 for srs-z's saddle h (srs: √5/4 = 0.5590)."""
        h = self.ramanujan_eigenvalue_at_P
        return abs(h.imag) / (abs(h) ** 2)

    @cached_property
    def closure_rate_mass_squared(self):
        """ν_mass² = tan²(arg h) for srs-z's saddle h (srs: 5/3)."""
        h = self.ramanujan_eigenvalue_at_P
        return math.tan(math.atan2(h.imag, h.real)) ** 2

    @cached_property
    def closure_rate_edge_local(self):
        """ν_edge = 1 — Class-3 edge-local closure rate (k*-determined)."""
        return 1.0

    # --- derived spectral data ---
    @cached_property
    def adjacency_perron(self):
        return self.K_STAR          # 3 — 3-regular

    @cached_property
    def hashimoto_perron(self):
        return self.K_STAR - 1      # 2

    @cached_property
    def nb_survival_per_step(self):
        return Fraction(self.K_STAR - 1, self.K_STAR)   # 2/3

    # --- bond structure ---
    @cached_property
    def bonds(self):
        """List of directed bonds (i, j, (c1,c2,c3)); |bonds| = 24."""
        return [(int(i), int(j), tuple(int(x) for x in c)) for (i, j, c) in _BONDS]

    @cached_property
    def n_bonds_directed(self):
        return len(self.bonds)      # 24

    @cached_property
    def vertices_fractional(self):
        """The 8 Wyckoff-8c fractional positions (for the record / Cartesian k)."""
        return [tuple(float(x) for x in v) for v in _VERTS]

    @cached_property
    def c3_permutation(self):
        """8x8 permutation matrix of the [111] 3-fold on the vertices."""
        return _C3_PERM.copy()

    # --- Bloch operators (fractional k; only the bond translations matter) ---
    def _resolve_k(self, k_label_or_frac):
        if isinstance(k_label_or_frac, str):
            return self.K_POINTS[k_label_or_frac]
        return tuple(k_label_or_frac)

    def adjacency_at_k(self, k_label_or_frac):
        """8x8 Bloch adjacency A(k). A(k)[tgt, src] += exp(2πi k·cell)."""
        k = np.asarray(self._resolve_k(k_label_or_frac), float)
        H = np.zeros((self.N_ATOMS, self.N_ATOMS), dtype=complex)
        for src, tgt, cell in self.bonds:
            H[tgt, src] += np.exp(2j * np.pi * np.dot(k, cell))
        return H

    def adjacency_spectrum_at_k(self, k_label_or_frac):
        H = self.adjacency_at_k(k_label_or_frac)
        ev = la.eigvalsh(H)
        return np.sort(np.real(ev))

    def hashimoto_at_k(self, k_label_or_frac):
        """24x24 Bloch Hashimoto B(k) on the directed-edge space.

        Same gauge as simulator/srs_substrate.SrsSubstrate.hashimoto_at_k:
          B(k)[f, e] = [f.src = e.tgt] · [f ≠ rev(e)] · exp(-2πi k·e.cell).
        """
        k = np.asarray(self._resolve_k(k_label_or_frac), float)
        bonds = self.bonds
        n = self.n_bonds_directed
        B = np.zeros((n, n), dtype=complex)
        for e_idx, (e_src, e_tgt, e_cell) in enumerate(bonds):
            rev_cell = tuple(-c for c in e_cell)
            phase = np.exp(-1j * 2 * np.pi * np.dot(k, np.array(e_cell)))
            for f_idx, (f_src, f_tgt, f_cell) in enumerate(bonds):
                if f_src != e_tgt:
                    continue
                if f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell:
                    continue                  # non-backtracking
                B[f_idx, e_idx] += phase
        return B

    def hashimoto_spectrum_at_k(self, k_label_or_frac):
        return np.sort_complex(la.eigvals(self.hashimoto_at_k(k_label_or_frac)))

    # --- Cartesian-k Bloch (for dispersion Taylor coefficients) ---
    # srs-z is realised on a PRIMITIVE CUBIC cell (RCSR a≈0.8864); we use the
    # cubic-cell-as-unit-cube fractional coords directly as Cartesian (a=1).
    # The Wyckoff-8c free parameter x ≈ 0.6607 is RCSR's embedding value; it is
    # NOT a simple rational (barycentric placement gives x=5/8; RCSR uses a
    # numerically-optimised embedding), so srs-z's dispersion coefficients are
    # not expected to be K-rational — itself a structural fact, not an artefact.
    @cached_property
    def _bond_displacements(self):
        out = []
        verts = [np.asarray(v, float) for v in _VERTS]
        for src, tgt, cell in self.bonds:
            out.append(tuple(verts[tgt] + np.asarray(cell, float) - verts[src]))
        return out

    def adjacency_at_k_cartesian(self, k_cart):
        kx, ky, kz = float(k_cart[0]), float(k_cart[1]), float(k_cart[2])
        H = np.zeros((self.N_ATOMS, self.N_ATOMS), dtype=complex)
        for i, (src, tgt, _c) in enumerate(self.bonds):
            r = self._bond_displacements[i]
            H[tgt, src] += np.exp(1j * (kx * r[0] + ky * r[1] + kz * r[2]))
        return H

    def adjacency_at_k_cartesian_mp(self, k_cart, prec=200):
        import mpmath as mp
        mp.mp.prec = prec
        verts = [[mp.mpf(repr(c)) for c in v] for v in self.vertices_fractional]
        H = mp.matrix(self.N_ATOMS, self.N_ATOMS)
        for (src, tgt, cell) in self.bonds:
            r = [verts[tgt][d] + mp.mpf(cell[d]) - verts[src][d] for d in range(3)]
            arg = sum(k_cart[d] * r[d] for d in range(3))
            phase = mp.exp(mp.mpc(0, 1) * arg)
            H[tgt, src] = H[tgt, src] + phase
        return H

    # --- C_3 isotypic structure at the P-analogue (the protected corner) ---
    def c3_decompose_at_P(self, degen_tol=1e-7):
        """Simultaneously (block-)diagonalise A(P_analog) and the C_3
        permutation, where P_analog = 'R' for srs-z (the BZ corner with the
        protected degenerate Ramanujan mode). Returns (eigenvalues,
        c3_labels, ||[A,C_3]||).
        """
        k = self._resolve_k(self.P_ANALOG)
        H = self.adjacency_at_k(k)
        C3 = self.c3_permutation.astype(complex)
        comm = la.norm(H @ C3 - C3 @ H)
        evals, evecs = la.eigh(H)
        idx = np.argsort(np.real(evals))
        evals = np.real(evals[idx])
        evecs = evecs[:, idx]
        # group degenerate bands, diagonalise C3 within each
        c3_diag = np.zeros(self.N_ATOMS, dtype=complex)
        i = 0
        while i < self.N_ATOMS:
            grp = [i]
            while i + 1 < self.N_ATOMS and abs(evals[i + 1] - evals[i]) < degen_tol:
                i += 1
                grp.append(i)
            sub = evecs[:, grp]
            C3sub = sub.conj().T @ C3 @ sub
            if len(grp) == 1:
                c3_diag[grp[0]] = C3sub[0, 0]
            else:
                w, _ = la.eig(C3sub)
                for ig, b in enumerate(grp):
                    c3_diag[b] = w[ig]
            i += 1
        labels = [_label_c3(v) for v in c3_diag]
        return evals, labels, comm

    def c3_isotypic_decomposition_at_P(self):
        """C_3 multiplicities on V_Ram at the C_3 point (the srs analogue
        returns (4, 2, 2)). V_Ram = 2 × adjacency space per Stark-Terras.

        Returns (mu_trivial, mu_omega, mu_omega_bar) on the 2|V|=16-dim V_Ram.
        """
        _evals, labels, _comm = self.c3_decompose_at_P()
        return (2 * labels.count('1'), 2 * labels.count('w'), 2 * labels.count('w2'))

    # --- internal: locate srs-z's protected Ramanujan saddle (scan all k) ---
    def _adj_degeneracies_with_c3(self, k):
        """At k: return list of (lambda, multiplicity, [c3_labels]) over the
        adjacency spectrum (C_3 labels only meaningful when [A(k),C_3]=0)."""
        H = self.adjacency_at_k(k)
        C3 = self.c3_permutation.astype(complex)
        comm = la.norm(H @ C3 - C3 @ H)
        evals, evecs = la.eigh(H)
        idx = np.argsort(np.real(evals))
        evals = np.real(evals[idx]); evecs = evecs[:, idx]
        out = []
        i = 0
        while i < self.N_ATOMS:
            grp = [i]
            while i + 1 < self.N_ATOMS and abs(evals[i + 1] - evals[i]) < 1e-7:
                i += 1; grp.append(i)
            sub = evecs[:, grp]
            if comm < 1e-6:
                C3sub = sub.conj().T @ C3 @ sub
                w = la.eigvals(C3sub) if len(grp) > 1 else np.array([C3sub[0, 0]])
                labels = [_label_c3(v) for v in w]
            else:
                labels = ['?'] * len(grp)
            out.append((float(evals[grp[0]]), len(grp), labels, comm))
            i += 1
        return out

    def _scan_ramanujan_all_k(self):
        k1 = self.K_STAR - 1
        bound = 2.0 * math.sqrt(k1)
        report = {}
        for name, k in self.K_POINTS.items():
            degs = self._adj_degeneracies_with_c3(k)
            ram_modes = []
            for (lam, mult, labels, comm) in degs:
                if abs(lam) <= bound + 1e-9 and abs(abs(lam) - bound) > 1e-9:
                    disc = lam * lam - 4 * k1
                    u = (lam + np.sqrt(complex(disc))) / 2.0
                    if u.imag < 0: u = u.conjugate()
                    if u.real < 0: u = -u.conjugate()
                    ram_modes.append({'u': u, 'lambda': lam, 'mult': mult,
                                      'c3_labels': labels, 'c3_commutes': comm < 1e-6})
            report[name] = {'adj_spectrum': [round(d[0], 6) for d in degs],
                            'ramanujan_modes': ram_modes}
        return report

    def _ramanujan_saddle_info(self):
        """Scan all k-points; return the protected (degenerate) Ramanujan
        eigenvalue closest to srs's h = (√3+i√5)/2.  'Protected' = lies in a
        ≥2-fold degenerate adjacency eigenspace at a high-symmetry point."""
        h_srs = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
        cands = []
        for name, info in self._scan_ramanujan_all_k().items():
            for m in info['ramanujan_modes']:
                if m['mult'] >= 2:
                    cands.append((m['u'], name, m['lambda'], m['mult'], m['c3_labels']))
        if not cands:
            raise ValueError(
                "srs-z has NO protected (degenerate) Ramanujan-saturating "
                "eigenvalue at any high-symmetry k-point — the framework's "
                "spectral edifice (h, ν_amp, ν_mass², dark correction) has no "
                "srs-z analogue.")
        cands.sort(key=lambda c: (abs(c[0] - h_srs), -c[3]))
        u, name, lam, mult, labels = cands[0]
        return u, {'k_point': name, 'lambda_adj': lam, 'adj_multiplicity': mult,
                   'c3_labels': labels, 'all_protected': cands}


# module-level convenience
K_STAR_SRSZ = SrsZSubstrate.K_STAR
N_ATOMS_SRSZ = SrsZSubstrate.N_ATOMS
N_EDGES_SRSZ = SrsZSubstrate.N_EDGES
GIRTH_SRSZ = SrsZSubstrate.GIRTH
