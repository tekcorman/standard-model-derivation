#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_berry_phase_attempt.py — Approach Q' for arg(h) Path B''.

After the no-go theorem (`docs/theorems/theorem_beta_coefficient_unity.md` § F0_γ
attempt) ruled out k_P-local closure of c = 1 in β = c·sin(arg h)·α_EM,
this script tests the Berry-phase / topological route Q'.

Conceptual setup
----------------
The walker B(k) carries a doubly-degenerate Ramanujan eigenvalue h at k_P.
Locally at k_P, the chirality content (no-go theorem) vanishes by ω↔ω²
conjugate symmetry. But h(k) is a function over the BZ; at k near k_P,
the doubly-degenerate point generically SPLITS into two bands. The
splitting is a vortex / Berry monopole at k_P, protected by C_3 symmetry.

A photon's wave packet has finite spatial extent, so it samples Bloch
momenta near k_P (not just k_P exactly). When parallel-transported around
a small closed loop encircling k_P, the walker eigenstate accumulates a
Berry phase. This phase is the topological coefficient that converts
the local angle arg(h) into the photon polarization rotation.

The Wilson line / U(1) holonomy around a small C_3-invariant loop
around k_P should equal a quantized topological phase. For a single
Berry monopole at k_P with C_3-protected charge ±1, the loop integral
is exp(i · 2π · 1/N_C3) per encirclement, where N_C3 = 3 fixes the
discrete winding.

Hypothesis (Q')
---------------
The Berry phase γ_B around a small C_3-invariant loop encircling k_P,
restricted to the +h band, satisfies:
  γ_B = ±2π · n  where n is integer and pins c = 1 in
  β = c · sin(arg h) · α_EM.

If γ_B is independent of loop radius (topological, not dynamical), Q'
closes c = 1.

If γ_B varies with loop radius, the band crossing is "accidental" and
not topologically protected — c = 0 at theorem grade.

Implementation
--------------
1. Pick a small C_3-invariant triangular loop around k_P in the BZ:
   3 vertices at k_P + ε·(C_3-rotated unit vectors).
2. Discretize each edge into M sub-steps; total 3M points around the loop.
3. At each k_i, build B(k_i) and find its 2-dim "+h band" eigenspace.
   (Pick the 2 eigenstates closest to h(k_P) = (√3+i√5)/2.)
4. Parallel-transport: compute Wilson line as product of overlap matrices
   between successive k_i's.
5. Take det of full Wilson line → U(1) Berry phase γ_B.
6. Vary M (discretization) and ε (loop radius); check convergence.
7. Apply C_3 to the loop; verify Berry phase invariant under loop rotation.
8. Compare γ_B to predicted topological values.

Run with:
    PYTHONPATH=. python3 proofs/foundations/arg_h_path_b_q_prime_berry_phase_attempt.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
)
from srs_photon_c3_chainmap import K_P_RED
from srs_photon_chirality_coefficient import build_B_directed


TOL = 1e-10
H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2
ABS_H = math.sqrt(2.0)
SIN_ARG_H = math.sqrt(5.0 / 8.0)
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))
ARG_H_DEG = math.degrees(ARG_H)
PRINT_WIDTH = 78


def fmt_z(z, prec=6):
    return f"{z.real:+.{prec}f}{z.imag:+.{prec}f}j"


def find_h_band(B_at_k, h_target=H_EXACT, n_band=2):
    """Find the n_band eigenstates of B(k) closest to h_target.

    Returns the 12×n_band orthonormal basis spanning the band.
    """
    evs, evecs = la.eig(B_at_k)
    distances = np.abs(evs - h_target)
    idx = np.argsort(distances)[:n_band]
    band = evecs[:, idx]
    Q, _ = la.qr(band)
    return Q[:, :n_band], evs[idx]


def overlap_matrix(Q1, Q2):
    """Compute the n×n overlap matrix M[i,j] = ⟨ψ_1^i|ψ_2^j⟩."""
    return Q1.conj().T @ Q2


def wilson_line(bond_list, h_target=H_EXACT, n_band=2):
    """Compute Wilson line (cumulative overlap product) around closed loop.

    bond_list: list of B(k_i) at successive k-points around a closed loop,
               ending where it started.
    Returns:
       W: n_band×n_band Wilson loop matrix (full non-Abelian holonomy).
       phase_U1: U(1) Berry phase = arg(det(W)).
    """
    n = len(bond_list)
    bands = []
    for B_k in bond_list:
        Q, _ = find_h_band(B_k, h_target=h_target, n_band=n_band)
        bands.append(Q)
    # Wilson line: product of overlap matrices around loop
    W = np.eye(n_band, dtype=complex)
    for i in range(n - 1):
        M = overlap_matrix(bands[i], bands[i + 1])
        W = M @ W
    # Close loop: overlap from last to first
    M_close = overlap_matrix(bands[-1], bands[0])
    W = M_close @ W
    # Polar-decompose to extract U(1) phase: det(W) is the U(1) holonomy
    det_W = la.det(W)
    phase_U1 = np.angle(det_W)
    return W, phase_U1, det_W


def make_c3_invariant_triangle(k_center, eps, axis_perp=None):
    """Make a C_3-invariant triangle in the (1,1,1)-perpendicular plane.

    k_center: 3-vector, center of triangle (typically k_P).
    eps: distance from center to each vertex.
    axis_perp: optional alternative perpendicular axis (default uses (1,1,1)).
    Returns 3 vertex points that are C_3 rotations of each other.
    """
    # The C_3 axis at k_P is along (1,1,1). Vertices live in plane perp to this.
    # Pick first vertex along some perp direction, then C_3-rotate.
    if axis_perp is None:
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    else:
        axis = axis_perp / la.norm(axis_perp)

    # Pick a perpendicular vector to axis
    if abs(axis[0]) < 0.9:
        perp = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    perp = perp / la.norm(perp)

    # C_3 rotation about (1,1,1) axis: (x,y,z) -> (z,x,y)
    # In primitive Bloch coordinates k_red, the C_3 acts as cyclic permutation.
    v0 = eps * perp
    v1 = np.array([v0[2], v0[0], v0[1]])   # (z, x, y) = C_3 of v0
    v2 = np.array([v1[2], v1[0], v1[1]])   # C_3² of v0
    return [k_center + v0, k_center + v1, k_center + v2]


def discretize_path(vertices, M):
    """Discretize a closed polygonal path with M sub-steps per edge.

    vertices: list of vertices, last vertex same as first (closes the loop).
    M: number of sub-steps per edge.
    """
    n_verts = len(vertices) - 1   # number of edges = n_verts
    points = []
    for e in range(n_verts):
        v0, v1 = vertices[e], vertices[e + 1]
        for s in range(M):
            t = s / M
            points.append((1 - t) * v0 + t * v1)
    points.append(vertices[-1])   # close loop
    return points


def main():
    print("=" * PRINT_WIDTH)
    print("Q' — Berry phase of walker +h band around k_P")
    print("Tests c = 1 in β = c · sin(arg h) · α_EM via topological winding")
    print("=" * PRINT_WIDTH)

    # -----------------------------------------------------------------------
    # Apparatus.
    # -----------------------------------------------------------------------
    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    n_bonds = len(bonds)

    k_center = np.array(K_P_RED)   # (0.25, 0.25, 0.25)
    print(f"\n  k_P = {k_center}")
    print(f"  h = (√3 + i√5)/2 = {fmt_z(H_EXACT, 4)},  arg(h) = {ARG_H_DEG:.4f}°")
    print(f"  sin(arg h) = √(5/8) = {SIN_ARG_H:.6f}")

    # -----------------------------------------------------------------------
    # Step 1: Verify h-band at k_P is doubly degenerate.
    # -----------------------------------------------------------------------
    print(f"\nStep 1 — Verify h-band at k_P is doubly degenerate")
    print(f"-" * PRINT_WIDTH)
    B_at_kP = build_B_directed(bonds, k_center)
    evs_kP = la.eigvals(B_at_kP)
    near_h = sorted(evs_kP, key=lambda z: abs(z - H_EXACT))[:4]
    print(f"  4 eigenvalues closest to h:")
    for ev in near_h:
        print(f"    {fmt_z(ev, 6)}    distance = {abs(ev - H_EXACT):.2e}")

    # Hopefully the 2 closest are EXACTLY h (within numerical precision).
    deg_check = abs(near_h[1] - H_EXACT)
    print(f"  Doubly-degenerate check: |2nd-closest − h| = {deg_check:.2e}")

    # -----------------------------------------------------------------------
    # Step 2: Track h-band around small C_3-invariant triangle.
    # -----------------------------------------------------------------------
    print(f"\nStep 2 — Track +h band around C_3-invariant triangle, vary radius+steps")
    print(f"-" * PRINT_WIDTH)

    print(f"  {'eps':>10}  {'M':>4}  {'Berry phase (rad)':>18}  "
          f"{'(deg)':>10}  {'/π':>10}  {'/2π':>10}  {'/sin(arg h)':>14}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

    results = []
    for eps in [0.01, 0.005, 0.002, 0.001, 0.0005]:
        for M in [4, 8, 16, 32]:
            triangle_vs = make_c3_invariant_triangle(k_center, eps)
            triangle_closed = list(triangle_vs) + [triangle_vs[0]]
            path = discretize_path(triangle_closed, M)
            # Compute B(k) at each path point
            B_path = [build_B_directed(bonds, np.array(k)) for k in path]
            W, phase_U1, det_W = wilson_line(B_path, h_target=H_EXACT, n_band=2)
            phase_deg = math.degrees(phase_U1)
            ratio_pi = phase_U1 / math.pi
            ratio_2pi = phase_U1 / (2 * math.pi)
            ratio_sah = phase_U1 / SIN_ARG_H
            results.append((eps, M, phase_U1, abs(det_W)))
            print(f"  {eps:>10.4g}  {M:>4d}  {phase_U1:>+18.10f}  "
                  f"{phase_deg:>+10.4f}  {ratio_pi:>+10.6f}  "
                  f"{ratio_2pi:>+10.6f}  {ratio_sah:>+14.6f}")

    # -----------------------------------------------------------------------
    # Step 3: Convergence analysis — does γ_B → constant as eps→0, M→∞?
    # -----------------------------------------------------------------------
    print(f"\nStep 3 — Convergence: is γ_B independent of loop size? (topological?)")
    print(f"-" * PRINT_WIDTH)

    # Take the smallest eps and largest M; compare to others
    finest = next(r for r in results if r[0] == 0.0005 and r[1] == 32)
    eps_min, M_max, phase_finest, det_abs_finest = finest
    print(f"  Finest grid: eps={eps_min}, M={M_max} → γ_B = {phase_finest:+.10f} rad")
    print(f"  |det(W)| = {det_abs_finest:.10f} (should be ~1 if no leakage)")

    # Convergence comparison
    phase_largest = next(r for r in results if r[0] == 0.01 and r[1] == 4)[2]
    print(f"  Coarsest grid: eps=0.01, M=4 → γ_B = {phase_largest:+.10f}")
    print(f"  |Δ(phase)| coarse-to-fine: {abs(phase_finest - phase_largest):.4e}")

    # -----------------------------------------------------------------------
    # Step 4: C_3 symmetry test — rotate triangle and re-compute Berry phase.
    # -----------------------------------------------------------------------
    print(f"\nStep 4 — C_3 symmetry: rotate triangle, re-compute Berry phase")
    print(f"-" * PRINT_WIDTH)

    eps_test = 0.001
    M_test = 16
    # Original triangle (in k_z-axis-aligned perp plane)
    tri_orig = make_c3_invariant_triangle(k_center, eps_test)
    # Rotated triangle: shift k_perp axis
    tri_rot = make_c3_invariant_triangle(k_center, eps_test,
                                          axis_perp=np.array([1.0, 0.0, 0.0]))
    for label, tri in [('orig (1,1,1)-perp', tri_orig),
                       ('rotated (1,0,0)-perp', tri_rot)]:
        tri_closed = list(tri) + [tri[0]]
        path = discretize_path(tri_closed, M_test)
        B_path = [build_B_directed(bonds, np.array(k)) for k in path]
        W, phase_U1, det_W = wilson_line(B_path, h_target=H_EXACT, n_band=2)
        print(f"  Triangle {label}: γ_B = {phase_U1:+.10f}  "
              f"({math.degrees(phase_U1):+.4f}°)  "
              f"|det W| = {abs(det_W):.6f}")

    # -----------------------------------------------------------------------
    # Step 5: What does the framework predict for γ_B?
    # -----------------------------------------------------------------------
    print(f"\nStep 5 — Theoretical prediction targets")
    print(f"-" * PRINT_WIDTH)
    print(f"  Candidate values for the Berry phase γ_B (mod 2π):")
    candidates = [
        ("0",                       0.0),
        ("π",                       math.pi),
        ("2π/3",                    2 * math.pi / 3),
        ("π/3",                     math.pi / 3),
        ("arg(h) [≈ 52°]",          ARG_H),
        ("2·arg(h)",                2 * ARG_H),
        ("3·arg(h) [≈ 157°]",       3 * ARG_H),
        ("π − arg(h)",              math.pi - ARG_H),
        ("sin(arg h)·2π",           SIN_ARG_H * 2 * math.pi),
        ("2π·c=1·sin(arg h)",       2 * math.pi * SIN_ARG_H),
    ]
    for label, val in candidates:
        diff = abs(phase_finest - val)
        diff_mod = abs(((phase_finest - val + math.pi) % (2 * math.pi)) - math.pi)
        print(f"    γ_B vs {label:<28}  Δ = {diff:.4e}  Δ(mod 2π) = {diff_mod:.4e}")

    # -----------------------------------------------------------------------
    # Step 6: Verdict.
    # -----------------------------------------------------------------------
    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"VERDICT")
    print(f"=" * PRINT_WIDTH)

    # Test if Berry phase is "topological" (i.e., independent of eps and M)
    phases = [r[2] for r in results]
    phase_range = max(phases) - min(phases)
    is_topological = phase_range < 1e-3   # tolerance for numerical convergence

    print(f"\n  Berry phase range across all (eps, M) tested: {phase_range:.4e} rad")
    print(f"  Topological (γ_B independent of grid)? "
          f"{'YES' if is_topological else 'NO'}")

    if is_topological:
        print(f"\n  γ_B = {phase_finest:+.6f} rad = "
              f"{math.degrees(phase_finest):+.3f}°")
        print(f"\n  This is a quantized topological invariant — pins c via")
        print(f"  some structural relation. Match to known framework predictions:")
        # The hypothesis: γ_B = 2π·sin(arg h) for c = 1 case
        target = 2 * math.pi * SIN_ARG_H
        ratio = phase_finest / target if abs(target) > 0 else float('nan')
        print(f"    Target 2π·sin(arg h) = {target:+.6f}")
        print(f"    γ_B / (2π·sin(arg h)) = {ratio:+.6f}")
        if abs(ratio - 1.0) < 0.01:
            print(f"    ✓ Q' LANDS — γ_B = 2π·sin(arg h), c = 1 closure candidate")
        else:
            print(f"    γ_B ≠ 2π·sin(arg h) — Q' gives a different structural relation")
    else:
        print(f"\n  γ_B varies with (eps, M) → NOT topological")
        print(f"  Walker eigenmode bundle does NOT have a Berry monopole at k_P")
        print(f"  (or our discretization is too coarse; try smaller eps, larger M)")

    print(f"\n" + "=" * PRINT_WIDTH)
    print(f"OK: arg_h_path_b_q_prime_berry_phase_attempt completed without errors")


if __name__ == "__main__":
    main()
