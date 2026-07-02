#!/usr/bin/env python3
"""
β closure attempt — Route 2: photon-walker correspondence via π-lift.

Strategy
--------
The photon's L (ω-irrep at ω²=36) and R (ω²-irrep at ω²=36) modes live in
the 6-dim undirected-edge Hodge space at k_P.  Lift them to the 12-dim
directed-bond space (via the symmetric and/or antisymmetric π map) and
decompose into eigensectors of the Hashimoto walker B(P).

Hypothesis (canonical β reading): L is supported in the +h walker sector,
R is supported in the +h* walker sector.  If so:
- L picks up walker phase per step = arg(h).
- R picks up walker phase per step = arg(h*) = −arg(h).
- Polarization rotation per step = (arg L − arg R)/2 = arg(h).
- Chirality (parity-odd part of unit phasor) per step = sin(arg h).
- β = sin(arg h)·α_EM with c=1 — derived from the eigensector
  decomposition, no extra prefactor.

We test BOTH lifts (symmetric and antisymmetric) since chirality might
require the antisymmetric lift to survive (Im(B) is in the parity-odd
sector under bond orientation reversal).

What we look for
----------------
- Spectral overlap |⟨L|P_h|L⟩|²  where P_h projects onto +h walker eigenspace.
- Similarly for R, +h*, −h, −h*.
- If L is dominantly in +h (overlap close to 1), the canonical reading is
  validated.  If L splits between +h and +h* (e.g., 50-50), the chirality
  cancellation gives c=0 and the canonical reading is wrong.
- A clean c=1 derivation requires L cleanly in one walker eigensector with
  chirality +sin(arg h).
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
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_c3_chainmap import build_C3_edge, build_delta_1, K_P_RED
from srs_photon_chirality_coefficient import (
    build_pi_projector,           # symmetric (parity-even) lift
    build_C3_directed,
    build_B_directed,
)


def build_pi_antisym(bonds, edges, k_red):
    """Antisymmetric (parity-odd) lift π_anti : undirected → directed.
       π_anti[forward, e_k] = +1/√2
       π_anti[backward, e_k] = − e^{−2πi k·cell} / √2
    """
    n_bonds = len(bonds)
    n_edges = len(edges)
    pi = np.zeros((n_bonds, n_edges), dtype=complex)
    fwd_idx, bwd_idx = {}, {}
    for b_idx, (src, tgt, cell, _) in enumerate(bonds):
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
        pi[bwd_idx[e_idx], e_idx] = -inv_sqrt2 * bwd_phase
    return pi


def main():
    print("=" * 72)
    print("β closure — photon-walker correspondence via π-lift")
    print("=" * 72)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    k_red = K_P_RED

    d = incidence_matrix_primitive(k_red, edges, len(verts))
    d1 = build_d1(cycles, edge_lookup, k_red, len(edges))
    Delta_1 = build_delta_1(d, d1)
    B = build_B_directed(bonds, k_red)
    C3_e = build_C3_edge(edges, k_red)
    pi_sym = build_pi_projector(bonds, edges, k_red)
    pi_anti = build_pi_antisym(bonds, edges, k_red)

    print(f"\n--- 1. Photon L/R eigenvectors in the 6-dim undirected-edge space ---")
    eigs, vecs = la.eig(Delta_1)
    order = np.argsort(eigs.real)
    eigs, vecs = eigs[order], vecs[:, order]
    mask = np.abs(eigs.real - 36) < 1e-6
    photon_basis = vecs[:, mask]
    Q, _ = la.qr(photon_basis)

    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()
    C3_photon = Q.conj().T @ C3_e @ Q
    ev_C3, vec_C3 = la.eig(C3_photon)
    L_in_Q = vec_C3[:, np.argmin(np.abs(ev_C3 - omega))]
    R_in_Q = vec_C3[:, np.argmin(np.abs(ev_C3 - omega2))]
    L_in_Q /= la.norm(L_in_Q)
    R_in_Q /= la.norm(R_in_Q)
    L_undir = Q @ L_in_Q          # 6-dim
    R_undir = Q @ R_in_Q
    print(f"  |L_undir| = {la.norm(L_undir):.6f}, |R_undir| = {la.norm(R_undir):.6f}")
    print(f"  ⟨L|R⟩ = {np.vdot(L_undir, R_undir):+.2e}")

    print(f"\n--- 2. Walker B(P) eigendecomposition on directed bonds ---")
    Bevs, Bvecs = la.eig(B)
    h = complex(math.sqrt(3)/2, math.sqrt(5)/2)
    h_star = h.conjugate()
    print(f"  h = {h}, h* = {h_star}")
    targets = {"+h": h, "+h*": h_star, "-h": -h, "-h*": -h_star,
               "+1": 1+0j, "-1": -1+0j}
    proj = {}
    for label, target in targets.items():
        idx = [i for i, ev in enumerate(Bevs) if abs(ev - target) < 1e-6]
        if not idx:
            continue
        V = Bvecs[:, idx]
        # orthonormalize
        Q_w, _ = la.qr(V)
        # Build projector onto the +h (etc.) eigenspace
        P = Q_w @ Q_w.conj().T
        proj[label] = P
        # Sanity
        err = la.norm(B @ Q_w - target * Q_w) / la.norm(Q_w)
        print(f"  walker eigenspace {label}: dim {Q_w.shape[1]}, "
              f"verification err {err:.2e}")

    print(f"\n--- 3. Lift L,R photon modes to directed bonds via π_sym, π_anti ---")
    print(f"  Test how the lifted photon modes overlap with each walker eigensector.")

    for lift_name, pi_lift in [("π_sym (parity-even)", pi_sym),
                                ("π_anti (parity-odd)", pi_anti)]:
        L_dir = pi_lift @ L_undir
        R_dir = pi_lift @ R_undir
        L_dir /= la.norm(L_dir)
        R_dir /= la.norm(R_dir)
        print(f"\n  ---- Lift via {lift_name} ----")
        print(f"  Walker-sector overlaps |⟨γ|P_λ|γ⟩|:")
        print(f"    {'sector':>6}  {'|<L|P_λ|L>|':>15}  {'|<R|P_λ|R>|':>15}  "
              f"{'arg(λ)':>10}  sin(arg λ)")
        for label, P in proj.items():
            wL = np.real(np.vdot(L_dir, P @ L_dir))
            wR = np.real(np.vdot(R_dir, P @ R_dir))
            target = targets[label]
            arg = math.degrees(np.angle(target))
            sin_arg = math.sin(np.angle(target))
            print(f"    {label:>6}  {wL:>15.6f}  {wR:>15.6f}  "
                  f"{arg:>+9.2f}°  {sin_arg:+.6f}")

        # Compute the "chirality reading" = Σ_λ |⟨γ|P_λ|γ⟩| · sin(arg λ)
        chir_L = sum(np.real(np.vdot(L_dir, P @ L_dir)) * math.sin(np.angle(targets[lab]))
                     for lab, P in proj.items())
        chir_R = sum(np.real(np.vdot(R_dir, P @ R_dir)) * math.sin(np.angle(targets[lab]))
                     for lab, P in proj.items())
        sin_arg_h = math.sqrt(5/8)
        print(f"\n  Chirality reading = Σ_λ |overlap|·sin(arg λ):")
        print(f"    chir_L = {chir_L:+.6f}    sin(arg h)·(chir_L/sin) = {chir_L/sin_arg_h:+.6f}")
        print(f"    chir_R = {chir_R:+.6f}    sin(arg h)·(chir_R/sin) = {chir_R/sin_arg_h:+.6f}")
        print(f"    (chir_L − chir_R)/2 = {(chir_L - chir_R)/2:+.6f}")
        print(f"    expected if c=1 = ±sin(arg h) = ±{sin_arg_h:.6f}")
        print(f"    coefficient c = (chir_L − chir_R)/(2·sin(arg h)) = "
              f"{(chir_L - chir_R)/(2*sin_arg_h):+.6f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
