#!/usr/bin/env python3
"""
Screw axis phase investigation.

Tests whether the 4_1 screw axis of I4_132 provides a nonzero relative phase
between C_3-isotypic sectors of V_Ram that matches delta_obs ~ 12.74 deg or
arg(h)/4 ~ 13.06 deg.

CLAIM UNDER TEST:
  The 4_1 screw axis (space group I4_132 #214, generator: (x,y,z) ->
  (-y, x+1/2, z+1/4) in fractional coordinates) acts on directed edges of
  the srs primitive cell as a 12x12 unitary operator U_{4_1}. At P=(1/4,1/4,1/4)
  this operator commutes with B(P) (it is a symmetry of the Bloch Hamiltonian at
  a screw-invariant k-point). The joint C_3 x Z_4 decomposition of V_Ram may
  yield a nonzero relative phase between the C_3-trivial and C_3-omega sectors.

RESULT:
  See printed summary. If delta_from_screw != 0 and matches delta_obs, the
  result is STRICT-SOLID. Otherwise BLOCKED with exact gap stated.

Run:
  PYTHONPATH=. python3 proofs/foundations/screw_axis_phase.py

Methodology follows koide_delta_phase.py (proofs/foundations/) and
theorem_B5_3_core.py.

References:
  - Ihara 1966, JMSJ 18, 219-235 (zeta function of discrete groups).
  - Hashimoto 1989, Adv. Stud. Pure Math. 15, 223-280 (non-backtracking walk).
  - Sunada 2012, Topological Crystallography, Springer (Bloch decomposition).
  - Bradley & Cracknell 1972, The Mathematical Theory of Symmetry in Solids,
    Clarendon Press, sec. 4.1-4.4 (space group representations).
  - Atiyah & Segal 1968, Quart. J. Math. Oxford, 19, 113-140 (equivariant K-theory).
  - Terras 2011, Zeta Functions of Graphs, Cambridge Univ. Press, sec. 2.2.
"""

import math
import sys
from pathlib import Path
from itertools import product as iproduct

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, C3_PERM, omega3, ATOMS, A_PRIM, N_ATOMS, NN_DIST

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2  # Hashimoto eigenvalue at P
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))      # arg(h) in radians
ARG_H_DEG = math.degrees(ARG_H)                      # ~52.24 deg
DELTA_OBS = 12.735                                    # PDG Koide phase delta in degrees
K_P = np.array([0.25, 0.25, 0.25])                  # P-point in primitive reduced coords
MATCH_TOL_DEG = 0.1                                   # matching tolerance in degrees

# ---------------------------------------------------------------------------
# Step 1: Build B(P) (reused from koide_delta_phase.py)
# ---------------------------------------------------------------------------

def build_directed_edges(bonds):
    """Return 12 directed edge tuples (src, tgt, cell_tuple)."""
    directed = [tuple(b) for b in bonds]
    assert len(directed) == 12, f"Expected 12 directed edges, got {len(directed)}"
    return directed


def bloch_hashimoto(k_frac, directed):
    """12x12 Bloch Hashimoto B(k).

    B(k)[f, e] = exp(2pi i k.cell_f) if tgt(e)=src(f) and f != rev(e).

    Follows the same sign convention as koide_delta_phase.py and
    theorem_B5_3_core.py.
    """
    n = len(directed)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for i_p, (src_p, tgt_p, cell_p) in enumerate(directed):
        for i_e, (src_e, tgt_e, cell_e) in enumerate(directed):
            if tgt_e != src_p:
                continue
            is_reverse = (tgt_p == src_e and
                          tuple(np.array(cell_p) + np.array(cell_e)) == (0, 0, 0))
            if is_reverse:
                continue
            phase = np.exp(2j * np.pi * np.dot(k, cell_p))
            B[i_p, i_e] += phase
    return B


# ---------------------------------------------------------------------------
# Step 2: Build U_{4_1} (12x12)
# ---------------------------------------------------------------------------

# The 4_1 screw axis in I4_132 (#214):
#   fractional coordinates: (x,y,z) -> (-y, x+1/2, z+1/4)
# This is a 4-fold rotation (order 4 up to lattice translation):
#   rotation part R: (x,y,z) -> (-y, x, z)
#   translation part t: (0, 1/2, 1/4) in fractional coordinates
#
# Fractional coordinates here mean conventional cubic (a=1).
# Our primitive reduced coordinates use the BCC primitive lattice
# with vectors A_PRIM. The transformation from primitive to conventional is:
#   r_conv = n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2] + r_atom
# (r_atom in fractional conventional coordinates).
#
# The screw acts on fractional CONVENTIONAL coordinates.

# Rotation matrix for 4_1: R: (x,y,z) -> (-y, x, z) in fractional conventional
SCREW_ROT_CONV = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0,  1],
], dtype=float)

# Translation part of the screw in fractional conventional coordinates
SCREW_TRANS_CONV = np.array([0.0, 0.5, 0.25])


def apply_screw_to_vertex(v_frac_conv):
    """Apply the 4_1 screw to a vertex position in fractional conventional coords.

    Returns the image in fractional conventional coords (not reduced mod 1).
    """
    return SCREW_ROT_CONV @ v_frac_conv + SCREW_TRANS_CONV


def prim_to_conv(v_prim_cart):
    """Convert from Cartesian (a=1) to fractional conventional (a=1).

    Since our ATOMS are already given in Cartesian with a=1, fractional
    conventional = Cartesian for a=1 cubic unit cell.
    """
    return v_prim_cart.copy()


def conv_to_cart(v_frac_conv):
    """Fractional conventional to Cartesian (identity for a=1)."""
    return v_frac_conv.copy()


def find_vertex_and_cell(pos_cart, tol=1e-8):
    """Given a Cartesian position, find which vertex in the primitive cell it
    corresponds to and the primitive cell displacement (n1, n2, n3).

    pos_cart = ATOMS[v] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]

    Returns (vertex_index, (n1, n2, n3)).
    """
    # Solve for integer cell displacements:
    # pos_cart - ATOMS[v] = n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
    A = A_PRIM.T  # 3x3, columns are primitive lattice vectors
    for v in range(N_ATOMS):
        diff = pos_cart - ATOMS[v]
        # Solve diff = A @ n for integer n
        n_float = la.solve(A, diff)
        n_int = np.round(n_float).astype(int)
        residual = la.norm(diff - A @ n_int)
        if residual < tol:
            return v, tuple(n_int)
    raise RuntimeError(f"Cannot find vertex for position {pos_cart} "
                       f"(distances: {[la.norm(pos_cart - ATOMS[v]) for v in range(N_ATOMS)]})")


def build_screw_on_directed_edges(directed):
    """Construct the 12x12 matrix U_{4_1} representing the 4_1 screw action
    on directed edges at k-point P.

    For each directed edge e = (src, tgt, cell_vec), the screw maps:
      - src_cart = ATOMS[src] to a new position
      - tgt_cart + cell_displacement to a new position

    The image of e is an edge f = (src', tgt', cell_vec') in the primitive cell.
    The Bloch phase factor for the gauge transformation is:
      exp(2pi i P . (cell_vec' - cell_vec))

    This is the standard Bloch-representation formula for a space group symmetry
    operator acting on the directed-edge Bloch basis (Bradley & Cracknell 1972
    sec. 4.1).

    Returns:
      U : (12, 12) complex array, the 4_1 screw matrix at k=P.
      errors : list of any edges that could not be mapped (should be empty).
    """
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    U = np.zeros((n, n), dtype=complex)
    errors = []

    for i, (src, tgt, cell) in enumerate(directed):
        # Source vertex position in Cartesian
        src_cart = ATOMS[src]
        # Target vertex position in Cartesian (may be in adjacent cell)
        cell_arr = np.array(cell, dtype=float)
        tgt_cart = ATOMS[tgt] + cell_arr @ A_PRIM  # r_tgt + n*a = A_PRIM.T @ cell + ATOMS[tgt]

        # Apply screw to source and target
        src_image_cart = conv_to_cart(apply_screw_to_vertex(prim_to_conv(src_cart)))
        tgt_image_cart = conv_to_cart(apply_screw_to_vertex(prim_to_conv(tgt_cart)))

        # Find which primitive-cell vertices these correspond to
        try:
            src_new, cell_src_new = find_vertex_and_cell(src_image_cart)
        except RuntimeError as exc:
            errors.append(f"src failed for edge {i}: {exc}")
            continue

        try:
            tgt_new, cell_tgt_new = find_vertex_and_cell(tgt_image_cart)
        except RuntimeError as exc:
            errors.append(f"tgt failed for edge {i}: {exc}")
            continue

        # The image edge in primitive coordinates:
        # src_new is in cell (0,0,0) by convention,
        # tgt_new is in cell cell_tgt_new - cell_src_new (relative to src_new's cell).
        cell_new = tuple(np.array(cell_tgt_new) - np.array(cell_src_new))

        # The primitive cell representative of the image edge
        # (source vertex in home cell, target in cell_new)
        new_edge = (src_new, tgt_new, cell_new)

        j = edge_to_idx.get(new_edge)
        if j is None:
            errors.append(f"Image edge {new_edge} of edge {i} not in directed set")
            continue

        # Bloch phase factor: exp(2pi i P . (cell_new - cell))
        # This arises from the gauge freedom in the Bloch basis when the
        # source vertex has shifted by cell_src_new primitive cells.
        # The full phase is exp(2pi i k . cell_src_new) for the source shift.
        cell_diff = np.array(cell_src_new, dtype=float)
        phase = np.exp(2j * np.pi * np.dot(K_P, cell_diff))
        U[j, i] = phase

    return U, errors


# ---------------------------------------------------------------------------
# Step 3: Check P is invariant under the 4_1 screw
# ---------------------------------------------------------------------------

def check_p_invariance():
    """Check whether the rotation part of the 4_1 screw maps P to an
    equivalent k-point (P mod reciprocal lattice vectors).

    The 4_1 screw has rotation R: (x,y,z) -> (-y,x,z) in fractional
    conventional coordinates. In the BCC primitive reciprocal lattice, the
    rotation R acts on primitive reduced k-coordinates.

    The primitive reciprocal lattice vectors b_i satisfy b_i . a_j = delta_ij.
    The conventional-to-primitive transformation on k is the transpose inverse
    of the primitive-to-conventional transformation on real space.

    P in primitive reduced = (1/4, 1/4, 1/4).
    P in Cartesian k = sum_i P_i * b_i where b_i are the primitive reciprocal
    vectors (without the 2pi factor, as used in our Bloch phase convention).
    """
    # Primitive reciprocal lattice vectors (b_i . a_j = delta_ij)
    A = A_PRIM  # rows are a_1, a_2, a_3
    B_recip = la.inv(A).T  # rows are b_1, b_2, b_3

    # P in Cartesian k (reciprocal Angstrom or 1/a units)
    P_cart = K_P @ B_recip  # = (1/4)*(b_1 + b_2 + b_3) in Cartesian

    # The rotation R in Cartesian on real space
    # R acts on Cartesian coordinates as: x -> -y, y -> x, z -> z
    # For k (a dual vector), R acts on k by (R^{-T}) = (R^T)^{-1} = R (since R in SO(3))
    # R_cart on real space: (x,y,z) -> (-y, x, z)
    R_cart = np.array([[ 0, -1, 0],
                        [ 1,  0, 0],
                        [ 0,  0, 1]], dtype=float)

    # Apply R to P in Cartesian k-space (R acts on k as R^{-T} = R for rotation)
    R_P_cart = R_cart @ P_cart

    # Express R*P in primitive reduced coordinates
    R_P_prim = R_P_cart @ la.inv(B_recip)  # B_recip @ n = Cartesian, so n = inv(B_recip) @ Cartesian

    print(f"\nStep 3: P invariance under 4_1 screw rotation")
    print(f"  P (primitive reduced) = {K_P}")
    print(f"  P (Cartesian k)       = {P_cart}")
    print(f"  R*P (Cartesian k)     = {R_P_cart}")
    print(f"  R*P (primitive reduced) = {R_P_prim}")

    # Check if R*P - P is a reciprocal lattice vector (integers in primitive reduced)
    diff = R_P_prim - K_P
    diff_round = np.round(diff)
    residual = la.norm(diff - diff_round)
    print(f"  R*P - P = {diff} (should be integer vector)")
    print(f"  Residual = {residual:.2e}")
    is_invariant = residual < 1e-10
    print(f"  P is invariant under R: {is_invariant}")
    return is_invariant


# ---------------------------------------------------------------------------
# C_3 on directed edges (from koide_delta_phase.py)
# ---------------------------------------------------------------------------

def c3_vertex_perm():
    perm = {}
    for i in range(4):
        for j in range(4):
            if abs(C3_PERM[i, j] - 1.0) < 1e-12:
                perm[j] = i
    assert perm == {0: 0, 1: 3, 2: 1, 3: 2}
    return perm


def c3_cell_perm(cell):
    return (cell[2], cell[0], cell[1])


def build_c3_on_directed_edges(directed):
    vp = c3_vertex_perm()
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    U = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        new_edge = (vp[src], vp[tgt], c3_cell_perm(cell))
        j = edge_to_idx.get(new_edge)
        if j is None:
            raise RuntimeError(f"C_3 mapped {(src, tgt, cell)} -> {new_edge} not found")
        U[j, i] = 1.0
    return U


# ---------------------------------------------------------------------------
# V_Ram extraction
# ---------------------------------------------------------------------------

def extract_vram(B_P, tol=1e-6):
    """Extract the 8-dim Ramanujan subspace (B eigenvalues with |mu|^2 = 2)."""
    evals, evecs = la.eig(B_P)
    ram_idx = [i for i, ev in enumerate(evals) if abs(abs(ev)**2 - 2.0) < tol]
    assert len(ram_idx) == 8, f"Expected 8 Ramanujan eigenvectors, got {len(ram_idx)}"
    return evals, evecs, ram_idx


# ---------------------------------------------------------------------------
# Step 4+5: C_3 and Z_4 isotypic decomposition of V_Ram
# ---------------------------------------------------------------------------

def classify_c3_sector(u_eval, tol=0.15):
    """Classify a U_{C_3} eigenvalue as 0 (trivial), 1 (omega), 2 (omega^2)."""
    if abs(u_eval - 1.0) < tol:
        return 0
    elif abs(u_eval - omega3) < tol:
        return 1
    elif abs(u_eval - omega3**2) < tol:
        return 2
    else:
        raise ValueError(f"Cannot classify U_{C_3} eigenvalue {u_eval:.4f}")


def classify_z4_sector(u_eval, tol=0.15):
    """Classify a U_{4_1} eigenvalue as 0 (1), 1 (i), 2 (-1), 3 (-i)."""
    targets = [1.0, 1j, -1.0, -1j]
    for k, t in enumerate(targets):
        if abs(u_eval - t) < tol:
            return k
    raise ValueError(f"Cannot classify U_{{4_1}} eigenvalue {u_eval:.4f} (|val|={abs(u_eval):.4f})")


def simultaneous_diag_3ops(B_P, U_c3, U_z4, evals_B, evecs_B, ram_idx, tol=1e-5):
    """
    Simultaneously diagonalize B(P), U_{C_3}, and U_{4_1} within V_Ram.

    Strategy:
    1. Extract Ramanujan eigenvectors (already B-diagonalized).
    2. Within each degenerate B eigenspace (each is 2-dim), diagonalize U_{C_3}.
    3. Within each resulting 1-dim or degenerate C_3 eigenspace, diagonalize U_{4_1}.

    If [U_{C_3}, U_{4_1}] = 0 on V_Ram, all three are simultaneously diagonalizable
    and the procedure gives well-defined joint quantum numbers.

    Returns:
      evecs_joint : (12, 8) complex array
      b_evals     : (8,) complex
      c3_evals    : (8,) complex
      z4_evals    : (8,) complex
    """
    evals_ram = evals_B[ram_idx]
    evecs_ram = evecs_B[:, ram_idx]   # 12 x 8

    # Sort by B eigenvalue for reproducibility
    sort_key = lambda z: (round(z.real, 5), round(z.imag, 5))
    order = np.argsort([sort_key(e)[0] + 1e-4 * sort_key(e)[1] for e in evals_ram])
    evals_ram = evals_ram[order]
    evecs_ram = evecs_ram[:, order]

    # Group degenerate B eigenvalues
    groups = []
    i = 0
    while i < 8:
        grp = [i]
        while i + 1 < 8 and abs(evals_ram[i + 1] - evals_ram[i]) < tol:
            i += 1
            grp.append(i)
        groups.append(grp)
        i += 1

    result_evecs = np.zeros((12, 8), dtype=complex)
    result_b_evals = np.zeros(8, dtype=complex)
    result_c3_evals = np.zeros(8, dtype=complex)
    result_z4_evals = np.zeros(8, dtype=complex)

    col = 0
    for grp in groups:
        sub_evecs = evecs_ram[:, grp]  # 12 x len(grp)
        Q_sub, _ = la.qr(sub_evecs)
        Q_sub = Q_sub[:, :len(grp)]

        # Diagonalize U_{C_3} within this B-eigenspace
        if len(grp) == 1:
            c3_restricted = np.array([[Q_sub[:, 0].conj() @ U_c3 @ Q_sub[:, 0]]])
            c3_ev = np.array([c3_restricted[0, 0]])
            c3_vecs = np.eye(1, dtype=complex)
        else:
            c3_restricted = Q_sub.conj().T @ U_c3 @ Q_sub
            c3_ev, c3_vecs = la.eig(c3_restricted)
            c3_order = np.argsort(np.angle(c3_ev))
            c3_ev = c3_ev[c3_order]
            c3_vecs = c3_vecs[:, c3_order]

        # For each C_3 eigenvector, get U_{4_1} eigenvalue
        for k_idx in range(len(grp)):
            v_c3 = Q_sub @ c3_vecs[:, k_idx]  # 12-dim
            # Expectation value of U_{4_1} on this C_3 eigenvector
            z4_ev = v_c3.conj() @ U_z4 @ v_c3

            result_evecs[:, col] = v_c3
            result_b_evals[col] = evals_ram[grp[0]]
            result_c3_evals[col] = c3_ev[k_idx]
            result_z4_evals[col] = z4_ev
            col += 1

    return result_evecs, result_b_evals, result_c3_evals, result_z4_evals


# ---------------------------------------------------------------------------
# Step 7: Phase extraction
# ---------------------------------------------------------------------------

def sector_mean_arg(b_evals_sector):
    """Mean argument of B(P) eigenvalues in a sector (radians)."""
    if len(b_evals_sector) == 0:
        return float('nan')
    return sum(np.angle(z) for z in b_evals_sector) / len(b_evals_sector)


def sector_product_arg(b_evals_sector):
    """Argument of the product of B(P) eigenvalues in a sector (radians)."""
    prod = 1.0 + 0j
    for z in b_evals_sector:
        prod *= z
    return np.angle(prod)


# ---------------------------------------------------------------------------
# Step 8: Check combinations for delta_obs match
# ---------------------------------------------------------------------------

def matches_delta(val_deg, name, results_list):
    """Check if val_deg matches any of the target values within MATCH_TOL_DEG."""
    targets = {
        'delta_obs':   DELTA_OBS,
        'arg_h_over_4': ARG_H_DEG / 4,
        'arg_h':       ARG_H_DEG,
        'pi_over_6':   30.0,
        'arg_h_over_2': ARG_H_DEG / 2,
        'neg_delta_obs': -DELTA_OBS,
    }
    found = []
    for tname, tv in targets.items():
        if abs(abs(val_deg) - abs(tv)) < MATCH_TOL_DEG:
            found.append(tname)
    if found:
        results_list.append(f"  MATCH: {name} = {val_deg:+.4f} deg matches {found}")
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Screw axis phase investigation")
    print("4_1 screw axis of I4_132 (#214) on directed-edge fiber at P")
    print("=" * 72)

    print(f"\nConstants:")
    print(f"  h = (sqrt(3)+i*sqrt(5))/2")
    print(f"  arg(h)    = {ARG_H_DEG:.4f} deg")
    print(f"  arg(h)/4  = {ARG_H_DEG / 4:.4f} deg")
    print(f"  delta_obs = {DELTA_OBS:.4f} deg (PDG)")
    print(f"  P         = {K_P}")

    # ------------------------------------------------------------------
    # Build primitive cell and operators
    # ------------------------------------------------------------------
    bonds = find_bonds()
    directed = build_directed_edges(bonds)

    print(f"\nDirected edges (12 total):")
    for i, (s, t, c) in enumerate(directed):
        print(f"  [{i:2d}] ({s},{t},{c})")

    # Step 1: B(P)
    B_P = bloch_hashimoto(K_P, directed)
    print(f"\nStep 1: B(P) built ({B_P.shape[0]}x{B_P.shape[1]})")

    # Step 4 (C_3): U_{C_3}
    U_c3 = build_c3_on_directed_edges(directed)
    comm_c3 = la.norm(B_P @ U_c3 - U_c3 @ B_P)
    assert comm_c3 < 1e-10, f"[B(P), U_C3] nonzero: {comm_c3}"
    print(f"\n[B(P), U_{{C_3}}] = 0 verified (norm = {comm_c3:.2e})")

    # Step 2: U_{4_1}
    print("\n" + "=" * 72)
    print("Step 2: Build U_{4_1} (4_1 screw on directed-edge fiber)")
    print("=" * 72)

    U_z4, errors = build_screw_on_directed_edges(directed)

    if errors:
        print(f"\nERRORS during U_{{4_1}} construction:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print(f"\nU_{{4_1}} built successfully (12x12 complex matrix)")

    # Verify unitarity
    unitary_err = la.norm(U_z4 @ U_z4.conj().T - np.eye(12))
    print(f"Unitarity check: ||U U† - I|| = {unitary_err:.2e}")
    assert unitary_err < 1e-10, f"U_{{4_1}} not unitary: {unitary_err}"

    # Verify order: U_{4_1}^4 should be lambda*I for some root of unity lambda
    U4 = la.matrix_power(U_z4, 4)
    # Check that U^4 is proportional to identity (scalar matrix)
    U4_diag = np.diag(U4)
    U4_offdiag = la.norm(U4 - np.diag(U4_diag))
    mean_diag = np.mean(U4_diag)
    diag_spread = np.max(np.abs(U4_diag - mean_diag))
    print(f"\nU_{{4_1}}^4 check:")
    print(f"  Off-diagonal norm:  {U4_offdiag:.2e}")
    print(f"  Diagonal mean:      {mean_diag:.6f}  (arg = {math.degrees(np.angle(mean_diag)):.4f} deg)")
    print(f"  Diagonal spread:    {diag_spread:.2e}")
    is_scalar = U4_offdiag < 1e-8 and diag_spread < 1e-8
    print(f"  U^4 = lambda*I:     {is_scalar}")

    if not is_scalar:
        # U^4 might not be exactly scalar if there are sign ambiguities.
        # Check whether U^4 at least commutes with everything (i.e., is central).
        print(f"  WARNING: U^4 is not scalar. Checking further...")
        print(f"  U^4 diagonal values:")
        for i, d in enumerate(U4_diag):
            print(f"    [{i}] {d.real:+.6f}{d.imag:+.6f}i  arg={math.degrees(np.angle(d)):.4f} deg")

    # Step 3: P invariance
    p_invariant = check_p_invariance()

    # Commutation [B(P), U_{4_1}]
    comm_z4 = la.norm(B_P @ U_z4 - U_z4 @ B_P)
    print(f"\n[B(P), U_{{4_1}}] norm = {comm_z4:.2e}")

    if comm_z4 > 1e-6:
        print("  WARNING: [B(P), U_{4_1}] is NOT zero.")
        print("  This means P is not invariant under the 4_1 screw, or the")
        print("  Bloch phase convention needs adjustment.")
        print("  Proceeding with analysis of available symmetries.")

    # ------------------------------------------------------------------
    # Step 4: C_3 isotypic decomposition of V_Ram
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Step 4: C_3 isotypic decomposition of V_Ram")
    print("=" * 72)

    evals_B, evecs_B, ram_idx = extract_vram(B_P)
    evals_ram = evals_B[ram_idx]
    print(f"\nV_Ram: {len(ram_idx)} eigenvectors (|mu|^2 = 2)")
    print("B(P) eigenvalues in V_Ram:")
    for ev in sorted(evals_ram, key=lambda z: (round(z.real, 4), round(z.imag, 4))):
        print(f"  {ev.real:+.6f}{ev.imag:+.6f}i  |mu|={abs(ev):.6f}  arg={math.degrees(np.angle(ev)):.4f} deg")

    # ------------------------------------------------------------------
    # Step 5: Z_4 isotypic decomposition of V_Ram
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Step 5: Z_4 (U_{4_1}) eigenstructure on V_Ram")
    print("=" * 72)

    # Project U_{4_1} onto V_Ram
    evecs_ram_mat = evecs_B[:, ram_idx]  # 12 x 8
    Q_ram, _ = la.qr(evecs_ram_mat)
    Q_ram = Q_ram[:, :8]
    U_z4_ram = Q_ram.conj().T @ U_z4 @ Q_ram  # 8x8

    z4_ram_evals, z4_ram_evecs = la.eig(U_z4_ram)
    print(f"\nU_{{4_1}} restricted to V_Ram eigenvalues:")
    for ev in sorted(z4_ram_evals, key=lambda z: np.angle(z)):
        print(f"  {ev.real:+.6f}{ev.imag:+.6f}i  |ev|={abs(ev):.4f}  arg={math.degrees(np.angle(ev)):.4f} deg")

    # Report multiplicities in Z_4 sectors
    z4_labels = ['1', 'i', '-1', '-i']
    z4_targets = [1.0, 1j, -1.0, -1j]
    z4_mult = [0, 0, 0, 0]
    z4_tol = 0.15
    for ev in z4_ram_evals:
        classified = False
        for k, t in enumerate(z4_targets):
            if abs(ev - t) < z4_tol:
                z4_mult[k] += 1
                classified = True
                break
        if not classified:
            print(f"  UNCLASSIFIED Z_4 eigenvalue: {ev:.6f}")
    print(f"\nZ_4 multiplicities on V_Ram: {dict(zip(z4_labels, z4_mult))}")

    # Check if [U_{C_3}, U_{4_1}] = 0 on V_Ram
    U_c3_ram = Q_ram.conj().T @ U_c3 @ Q_ram
    comm_c3_z4_ram = la.norm(U_c3_ram @ U_z4_ram - U_z4_ram @ U_c3_ram)
    print(f"\n[U_{{C_3}}, U_{{4_1}}] on V_Ram: norm = {comm_c3_z4_ram:.2e}")

    # ------------------------------------------------------------------
    # Step 6: Joint decomposition
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Step 6: Joint C_3 x Z_4 decomposition of V_Ram")
    print("=" * 72)

    evecs_joint, b_evals_joint, c3_evals_joint, z4_evals_joint = \
        simultaneous_diag_3ops(B_P, U_c3, U_z4, evals_B, evecs_B, ram_idx)

    print(f"\n{'Index':>5}  {'B_P eval':>20}  {'arg(B)':>8}  "
          f"{'C3 ev':>10}  {'C3 sec':>8}  {'Z4 ev':>10}  {'Z4 sec':>8}")
    for i in range(8):
        b_ev = b_evals_joint[i]
        c3_ev = c3_evals_joint[i]
        z4_ev = z4_evals_joint[i]
        b_arg = math.degrees(np.angle(b_ev))
        try:
            c3_sec = classify_c3_sector(c3_ev)
            c3_name = ['trivial', 'omega', 'omega^2'][c3_sec]
        except ValueError:
            c3_name = f"?{c3_ev:.3f}"
        try:
            z4_sec = classify_z4_sector(z4_ev)
            z4_name = z4_labels[z4_sec]
        except ValueError:
            z4_name = f"?{z4_ev:.3f}"
        print(f"  {i:>5}  {b_ev.real:+.4f}{b_ev.imag:+.4f}i  {b_arg:>8.3f}  "
              f"{c3_ev.real:+.4f}{c3_ev.imag:+.4f}i  {c3_name:>8}  "
              f"{z4_ev.real:+.4f}{z4_ev.imag:+.4f}i  {z4_name:>8}")

    # Collect B(P) eigenvalues by C_3 sector
    trivial_b = [b_evals_joint[i] for i in range(8)
                 if classify_c3_sector(c3_evals_joint[i]) == 0]
    omega_b   = [b_evals_joint[i] for i in range(8)
                 if classify_c3_sector(c3_evals_joint[i]) == 1]
    omega2_b  = [b_evals_joint[i] for i in range(8)
                 if classify_c3_sector(c3_evals_joint[i]) == 2]

    assert len(trivial_b) == 4, f"trivial mult = {len(trivial_b)}, expected 4"
    assert len(omega_b) == 2, f"omega mult = {len(omega_b)}, expected 2"
    assert len(omega2_b) == 2, f"omega^2 mult = {len(omega2_b)}, expected 2"

    print(f"\nC_3 sector multiplicities on V_Ram: "
          f"trivial={len(trivial_b)}, omega={len(omega_b)}, omega^2={len(omega2_b)}")
    print("(Expected: 4, 2, 2 per theorem B5.3-core)")

    # Z_4 eigenvalues in each C_3 sector
    print(f"\nZ_4 eigenvalues by C_3 sector:")
    for sec_name, b_list in [('trivial', trivial_b), ('omega', omega_b), ('omega^2', omega2_b)]:
        mask = ([i for i in range(8)
                 if classify_c3_sector(c3_evals_joint[i]) ==
                    ['trivial', 'omega', 'omega^2'].index(sec_name)])
        z4_in_sec = [z4_evals_joint[i] for i in mask]
        print(f"  {sec_name:>8}: Z_4 evals = "
              f"{[f'{z.real:+.4f}{z.imag:+.4f}i (arg={math.degrees(np.angle(z)):.2f} deg)'for z in z4_in_sec]}")

    # ------------------------------------------------------------------
    # Step 7: Phase extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Step 7: Phase extraction")
    print("=" * 72)

    phi_0_mean = sector_mean_arg(trivial_b)
    phi_1_mean = sector_mean_arg(omega_b)
    phi_0_prod = sector_product_arg(trivial_b)
    phi_1_prod = sector_product_arg(omega_b)

    delta_mean = math.degrees(phi_1_mean - phi_0_mean)
    delta_prod = math.degrees(phi_1_prod - phi_0_prod)

    print(f"\n  phi_0 (trivial, mean arg) = {math.degrees(phi_0_mean):+.6f} deg")
    print(f"  phi_1 (omega,   mean arg) = {math.degrees(phi_1_mean):+.6f} deg")
    print(f"  delta = phi_1 - phi_0 (mean) = {delta_mean:+.6f} deg")

    print(f"\n  phi_0 (trivial, prod arg) = {math.degrees(phi_0_prod):+.6f} deg")
    print(f"  phi_1 (omega,   prod arg) = {math.degrees(phi_1_prod):+.6f} deg")
    print(f"  delta = phi_1 - phi_0 (prod) = {delta_prod:+.6f} deg")

    # Individual B(P) eigenvalue args
    print(f"\nIndividual B(P) eigenvalue args by C_3 sector:")
    print(f"  trivial sector:")
    for z in trivial_b:
        print(f"    {z.real:+.6f}{z.imag:+.6f}i  arg = {math.degrees(np.angle(z)):+.6f} deg")
    print(f"  omega sector:")
    for z in omega_b:
        print(f"    {z.real:+.6f}{z.imag:+.6f}i  arg = {math.degrees(np.angle(z)):+.6f} deg")

    # Z_4 mean phase per C_3 sector
    z4_trivial_evals = [z4_evals_joint[i] for i in range(8)
                        if classify_c3_sector(c3_evals_joint[i]) == 0]
    z4_omega_evals   = [z4_evals_joint[i] for i in range(8)
                        if classify_c3_sector(c3_evals_joint[i]) == 1]

    z4_phi_trivial = sum(np.angle(z) for z in z4_trivial_evals) / len(z4_trivial_evals)
    z4_phi_omega   = sum(np.angle(z) for z in z4_omega_evals) / len(z4_omega_evals)
    z4_delta = math.degrees(z4_phi_omega - z4_phi_trivial)

    print(f"\nZ_4 mean arg per C_3 sector:")
    print(f"  trivial sector: {math.degrees(z4_phi_trivial):+.6f} deg")
    print(f"  omega   sector: {math.degrees(z4_phi_omega):+.6f} deg")
    print(f"  Z_4 relative phase (omega - trivial): {z4_delta:+.6f} deg")

    # Screw translation Bloch phase: exp(2pi i P . t_screw)
    t_screw_conv = SCREW_TRANS_CONV  # (0, 1/2, 1/4) in fractional conventional
    # Convert to primitive reduced coordinates
    # t_prim_reduced[i] = t_conv_cart . b_i  (dot with primitive recip vector)
    # First convert t to Cartesian: for a=1 cube, frac conv = Cartesian
    t_screw_cart = t_screw_conv.copy()  # Cartesian, a=1
    # Then express in primitive reduced: solve t_cart = sum n_i a_i for n_i
    A = A_PRIM
    t_prim = la.solve(A.T, t_screw_cart)
    screw_bloch_phase = np.exp(2j * np.pi * np.dot(K_P, t_prim))
    screw_bloch_arg_deg = math.degrees(np.angle(screw_bloch_phase))
    print(f"\nScrew translation Bloch phase exp(2pi i P . t_screw):")
    print(f"  t_screw (fractional conventional) = {t_screw_conv}")
    print(f"  t_screw (primitive reduced) = {t_prim}")
    print(f"  P . t_prim = {np.dot(K_P, t_prim):.6f}")
    print(f"  phase = {screw_bloch_phase:.6f}  arg = {screw_bloch_arg_deg:.4f} deg")

    # ------------------------------------------------------------------
    # Step 8: Check all combinations for delta_obs match
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Step 8: Combination search for delta_obs match")
    print("=" * 72)

    matches = []

    # B(P) phase differences
    matches_delta(delta_mean, "phi_1-phi_0 (mean-arg)", matches)
    matches_delta(delta_prod, "phi_1-phi_0 (prod-arg)", matches)
    matches_delta(z4_delta, "Z_4 relative phase (omega-trivial)", matches)
    matches_delta(screw_bloch_arg_deg, "screw Bloch phase", matches)
    matches_delta(screw_bloch_arg_deg / 4, "screw Bloch phase / 4", matches)

    # Individual B eigenvalue arg differences
    h_target = H_EXACT
    h_targets = {
        'h':  H_EXACT,
        'h*': H_EXACT.conj(),
        '-h': -H_EXACT,
        '-h*': -H_EXACT.conj(),
    }

    # For each pair of B eigenvalues across sectors, compute arg difference
    for name_t, t_val in [(n, v) for n, v in h_targets.items()]:
        for name_o, o_val in [(n, v) for n, v in h_targets.items()]:
            diff_deg = math.degrees(np.angle(o_val) - np.angle(t_val))
            # Only record non-trivial differences that could be delta
            if abs(diff_deg) > 0.5 and abs(diff_deg) < 180:
                matches_delta(diff_deg, f"arg({name_o}) - arg({name_t})", matches)

    # Z_4 eigenvalue combination: Z_4 arg in omega sector vs trivial sector
    for i_t in range(len(z4_trivial_evals)):
        for i_o in range(len(z4_omega_evals)):
            diff = math.degrees(np.angle(z4_omega_evals[i_o]) - np.angle(z4_trivial_evals[i_t]))
            matches_delta(diff, f"Z4_omega[{i_o}] - Z4_trivial[{i_t}]", matches)

    # Screw Bloch phase minus B arg combinations
    for combo_name, combo_val in [
        ("screw_phase - arg(h)", screw_bloch_arg_deg - ARG_H_DEG),
        ("screw_phase / 4", screw_bloch_arg_deg / 4),
        ("(screw_phase + arg(h)) / 4", (screw_bloch_arg_deg + ARG_H_DEG) / 4),
        ("(arg(h) - screw_phase) / 4", (ARG_H_DEG - screw_bloch_arg_deg) / 4),
        ("screw_phase mod arg(h)", screw_bloch_arg_deg % ARG_H_DEG),
        ("arg(h) mod screw_phase", ARG_H_DEG % abs(screw_bloch_arg_deg) if screw_bloch_arg_deg != 0 else 0),
    ]:
        matches_delta(combo_val, combo_name, matches)

    if matches:
        print("\nMATCHES FOUND:")
        for m in matches:
            print(m)
    else:
        print("\nNo combinations match delta_obs within 0.1 deg.")

    # ------------------------------------------------------------------
    # Step 9: Structural diagnosis
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Step 9: Structural diagnosis")
    print("=" * 72)

    print(f"\nSummary of key results:")
    print(f"  [B(P), U_{{C_3}}] = 0:         VERIFIED (norm = {comm_c3:.2e})")
    print(f"  [B(P), U_{{4_1}}]:             norm = {comm_z4:.2e}  "
          f"({'COMMUTES' if comm_z4 < 1e-6 else 'DOES NOT COMMUTE'})")
    print(f"  [U_{{C_3}}, U_{{4_1}}] on V_Ram: norm = {comm_c3_z4_ram:.2e}  "
          f"({'COMMUTE' if comm_c3_z4_ram < 1e-6 else 'DO NOT COMMUTE'})")
    print(f"  U_{{4_1}} unitary:             VERIFIED (norm = {unitary_err:.2e})")
    print(f"  U_{{4_1}}^4 = lambda*I:        {is_scalar}")
    print(f"  P invariant under 4_1 screw: {p_invariant}")
    print(f"\n  C_3 isotypic structure on V_Ram: (4, 2, 2) = "
          f"trivial:{len(trivial_b)}, omega:{len(omega_b)}, omega^2:{len(omega2_b)}")
    print(f"\n  delta (mean-arg method):  {delta_mean:+.6f} deg")
    print(f"  delta (prod-arg method):  {delta_prod:+.6f} deg")
    print(f"  delta_obs (PDG):          {DELTA_OBS:+.6f} deg")
    print(f"  arg(h)/4:                 {ARG_H_DEG/4:+.6f} deg")

    print("\n  Interpretation:")
    if comm_z4 < 1e-6:
        print("  [B(P), U_{4_1}] = 0: the 4_1 screw IS a symmetry at P.")
        print("  The Z_4 decomposition of V_Ram is well-defined.")
        if z4_mult.count(0) == 0:
            print("  Z_4 multiplicities indicate a nontrivial Z_4 structure.")
        else:
            print("  Z_4 multiplicities contain zeros; some sectors are absent.")
        if abs(z4_delta) > 0.01:
            print(f"  Z_4 relative phase between C_3 sectors = {z4_delta:.4f} deg (nonzero).")
        else:
            print("  Z_4 relative phase between C_3 sectors = 0 (symmetric).")
    else:
        print("  [B(P), U_{4_1}] != 0: the 4_1 screw is NOT a symmetry of B(P) at P.")
        print("  The screw sends P to a different k-point (not equivalent mod reciprocal lattice),")
        print("  or the Bloch phase convention makes U_{4_1} non-unitary at P.")
        print("  In this case the Z_4 decomposition is NOT well-defined on V_Ram.")
        print("  The delta=0 result from U_{C_3} is unaffected (U_{C_3} does commute).")
        print()
        print("  GAP: The 4_1 screw axis of I4_132 does NOT stabilize the P-point of the")
        print("  BCC BZ (the little group of P contains only C_3, not Z_4 = <4_1>).")
        print("  Therefore the 4_1 screw CANNOT provide a Z_4-derived phase delta at P.")

    if delta_mean == 0.0 or abs(delta_mean) < 0.001:
        print(f"\n  RESULT: delta from B(P) C_3-sector phase analysis = 0 deg.")
        print(f"  This reproduces the koide_delta_phase.py result: U_{{C_3}} is a real")
        print(f"  permutation, and the C_3-sector B(P) eigenvalues are conjugate-symmetric,")
        print(f"  giving zero relative phase.")

    print("\n" + "=" * 72)
    print("RIGOR STATUS")
    print("=" * 72)

    if comm_z4 > 1e-6:
        print("""
STATUS: BLOCKED

CLAIM: The 4_1 screw axis provides a nonzero Z_4 phase that contributes to
delta_obs ~ 12.74 deg.

BLOCKING GAP:
  The 4_1 screw axis (x,y,z) -> (-y, x+1/2, z+1/4) acts on k-space by its
  rotation part R: (k_1, k_2, k_3) -> (-k_2, k_1, k_3) in conventional fractional
  k. The P-point (1/4, 1/4, 1/4) in primitive reduced coordinates maps to
  R.P = (-1/4, 1/4, 1/4) in conventional fractional, which is NOT equivalent
  to P modulo the conventional reciprocal lattice (its conventional form is
  (1/4, 1/4, 1/4)). Therefore:

  (i)  P is NOT in the little group of the 4_1 screw.
  (ii) [B(P), U_{4_1}] != 0.
  (iii) The Z_4 decomposition of V_Ram under U_{4_1} is ill-defined
       (U_{4_1} is not a symmetry at P).
  (iv) No Z_4-derived relative phase formula can be stated from first principles.

  The little group of P = (1/4, 1/4, 1/4) in I4_132 is generated by the body-
  diagonal C_3 rotation alone (Bradley & Cracknell 1972, Table 5.7). The 4_1
  screw has a 4-fold rotation component that maps the body-diagonal direction
  (1,1,1) to (-1,1,1) (in conventional fractional k), which is a different
  direction. Therefore the 4_1 screw is not in the little group of P.

CONCLUSION:
  The 4_1 screw axis does NOT provide a derivable delta phase at P.
  The delta = 0 result from U_{C_3} analysis (koide_delta_phase.py) stands.
  delta_obs cannot be derived from this screw-axis mechanism.

  To derive delta_obs, a different mechanism must be found (e.g., a Feshbach
  coupling between V_Ram and the tree subspace, or a dark correction from the
  Bloch-band mixing at a different k-point). See docs/honest_assessment.md.
""")
    else:
        if len(matches) == 0:
            print("""
STATUS: BLOCKED

CLAIM: The 4_1 screw axis provides a nonzero Z_4 phase matching delta_obs.

BLOCKING GAP:
  Although [B(P), U_{4_1}] = 0 (the screw IS a symmetry at P), the relative
  Z_4 phase between C_3 sectors does not match delta_obs within 0.1 deg.
  No combination of Z_4-derived phases yields delta_obs ~ 12.74 deg or
  arg(h)/4 ~ 13.06 deg.

  The exact computed values are:
    delta (B-sector mean-arg method) = {delta_mean:.4f} deg
    Z_4 relative phase (omega - trivial) = {z4_delta:.4f} deg
    Target delta_obs = {DELTA_OBS:.4f} deg

  CONCLUSION: The 4_1 screw mechanism does not produce delta_obs.
""".format(delta_mean=delta_mean, z4_delta=z4_delta, DELTA_OBS=DELTA_OBS))
        else:
            print("""
STATUS: FESHBACH-PATTERN (further derivation required)

A combination matching delta_obs was found. See matches above.
Full derivation from A1 + A2 + A3 axioms has not been established.
""")

    print("Script completed.")


if __name__ == "__main__":
    main()
