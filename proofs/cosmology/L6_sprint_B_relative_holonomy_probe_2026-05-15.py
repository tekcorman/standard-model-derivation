#!/usr/bin/env python3
"""
L6 Sprint B — relative-holonomy gate test (2026-05-15 EOD+5).

CONTEXT
-------
L6 Sprint A (`L6_sprint_A_bloch_decomposition_gate_2026-05-15.md`) FAILED:
the framework's photon-walker chirality correspondence is C_3-little-group
degeneracy-protected at high-symmetry k-points and dissolves in the
acoustic (generic small-k) regime where r_s lives.

Route (b) survival hinges on: even though c_1(photon Hodge bundle) = 0
(predictions/c1_photon_bundle.py — photon bundle topologically TRIVIAL,
proven), does the RELATIVE holonomy between the photon bundle and the
walker bundle carry non-trivial structure? Two bundles can each be
c_1-trivial while their mutual/relative winding is non-trivial — a
distinct invariant. If the photon-walker COUPLING carries topology that
neither bundle has alone, that is a symmetry-independent (label-free)
characterization of the coupling that survives the degeneracy lifting
Sprint A found.

GATE QUESTION
-------------
Build the "walker-dressed photon bundle":
    D(k) = orthonormalize( P_w(k) . pi . Psi_gamma(k) )
where:
    Psi_gamma(k) = 2-dim photon Hodge bundle = ker d^dag(k) (generic k)
    pi          = photon(undirected) -> walker(directed) lift
    P_w(k)      = projector onto the 2-dim walker eigenspace tracked
                  continuously from the K_P {+h, +h*} sector

c_1(bare photon) = 0 (known). Compute c_1(D) on BZ slices.

  Gate PASS:  c_1(D) != 0 (or dressed Wilson loop non-trivial) while
              c_1(bare) = 0  -> the coupling carries symmetry-independent
              topology; route (b) viable; Sprint C (extract dispersion
              from the relative connection) becomes attemptable.

  Gate FAIL:  c_1(D) = 0 too / dressed Wilson loop trivial  -> the
              photon-walker coupling has NO generic-k geometric content;
              route (b) dead; L6 wall confirmed at gate 2; cosmology
              cluster Scenario 3 final.

DESIGN
------
- Reuse srs_photon_berry FHS machinery (photon_eigenvectors, link_variable,
  plaquette Chern) — proven on the bare photon bundle (gives c_1 = 0).
- Walker B(k) via build_B_directed; track the 2 eigenvalues nearest
  {+h, +h*}_{K_P} continuously to define P_w(k).
- pi-lift: antisymmetric (parity-odd) PRIMARY — this is the lift the
  UNIQUE-THEOREM-GRADE beta cosmic birefringence uses (a parity-odd /
  chirality effect). Symmetric lift as cross-check.
- Compute c_1 of D(k) on (k_x,k_y) slices at several k_z, AND a small-loop
  Wilson loop near Gamma (the acoustic regime specifically).
"""

import math
import os
import sys

import numpy as np
from numpy import linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    canonical_edges_primitive,
    find_primitive_connectivity,
    incidence_matrix_primitive,
)
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_hodge import build_edge_lookup
from srs_photon_berry import photon_eigenvectors, link_variable
from srs_photon_chirality_coefficient import build_B_directed, build_pi_projector


def build_pi_antisym(bonds, edges, k_red):
    """Antisymmetric (parity-odd) pi-lift undirected->directed.
    Same as in L6_sprint_A_chirality_reading_check / srs_photon_walker_correspondence."""
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


def walker_subspace(B, targets, n_dim=2):
    """Return an orthonormal basis (12 x n_dim) of the walker eigenspace
    spanned by the n_dim eigenvectors whose eigenvalues are closest to the
    target list (continuous tracking of the K_P {+h,+h*} sector)."""
    eigs, vecs = la.eig(B)
    chosen = []
    used = set()
    for tgt in targets:
        dists = np.abs(eigs - tgt)
        # exclude already-used indices
        for u in used:
            dists[u] = np.inf
        j = int(np.argmin(dists))
        used.add(j)
        chosen.append(j)
    V = vecs[:, chosen]
    Q, _ = la.qr(V)
    return Q[:, :n_dim]


def dressed_photon_frame(k_red, edges, cycles, edge_lookup, bonds,
                         pi_kind, walker_targets, n_verts=4):
    """D(k) = orthonormalize( P_w(k) . pi . Psi_gamma(k) ), shape (12, 2)."""
    Psi_g = photon_eigenvectors(k_red, edges, cycles, edge_lookup,
                                n_verts=n_verts, n_photon=2)  # (6, 2)
    if pi_kind == "antisym":
        pi = build_pi_antisym(bonds, edges, k_red)
    else:
        pi = build_pi_projector(bonds, edges, k_red)
    lifted = pi @ Psi_g  # (12, 2)
    B = build_B_directed(bonds, k_red)
    W = walker_subspace(B, walker_targets, n_dim=2)  # (12, 2) orthonormal
    Pw = W @ W.conj().T  # (12,12) projector onto tracked walker eigenspace
    dressed = Pw @ lifted  # (12, 2)
    # Orthonormalize (QR); if rank-deficient, flag
    Q, R = la.qr(dressed)
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-9))
    return Q[:, :2], rank


def slice_chern_dressed(k_z, N, edges, cycles, edge_lookup, bonds,
                        pi_kind, walker_targets):
    """FHS first Chern number of the DRESSED photon bundle on a
    (k_x,k_y) slice at fixed k_z. Mirrors srs_photon_berry.slice_chern_number
    but uses the walker-dressed frame instead of the bare photon frame."""
    ks_x = np.linspace(-0.5, 0.5, N, endpoint=False)
    ks_y = np.linspace(-0.5, 0.5, N, endpoint=False)
    offset = 1.0 / (2 * N) if abs(k_z) < 1e-10 else 0.0
    ks_x = ks_x + offset
    ks_y = ks_y + offset

    psi_grid = np.empty((N, N, 12, 2), dtype=complex)
    min_rank = 2
    for i in range(N):
        for j in range(N):
            k = np.array([ks_x[i], ks_y[j], k_z])
            try:
                D, rank = dressed_photon_frame(
                    k, edges, cycles, edge_lookup, bonds,
                    pi_kind, walker_targets)
            except Exception:
                return None, None
            psi_grid[i, j] = D
            min_rank = min(min_rank, rank)

    Ux = np.empty((N, N, 2, 2), dtype=complex)
    Uy = np.empty((N, N, 2, 2), dtype=complex)
    for i in range(N):
        for j in range(N):
            ip = (i + 1) % N
            jp = (j + 1) % N
            Ux[i, j] = link_variable(psi_grid[i, j], psi_grid[ip, j])
            Uy[i, j] = link_variable(psi_grid[i, j], psi_grid[i, jp])

    chern_sum = 0.0
    for i in range(N):
        for j in range(N):
            ip = (i + 1) % N
            jp = (j + 1) % N
            F = Ux[i, j] @ Uy[ip, j] @ la.inv(Ux[i, jp]) @ la.inv(Uy[i, j])
            chern_sum += np.angle(la.det(F))
    c1 = chern_sum / (2 * np.pi)
    return c1, min_rank


def small_loop_wilson(center, radius, n_pts, edges, cycles, edge_lookup,
                      bonds, pi_kind, walker_targets):
    """Non-Abelian Wilson loop of bare vs dressed photon frame around a
    small square loop near `center` (acoustic regime probe).
    Returns (|det W_bare|, arg det W_bare, |det W_dressed|, arg det W_dressed)."""
    cx, cy, cz = center
    # Square loop in the (k_x, k_y) plane at fixed k_z = cz
    pts = []
    for s in np.linspace(0, 1, n_pts, endpoint=False):
        # parametrize a square perimeter
        if s < 0.25:
            t = s / 0.25
            pts.append((cx - radius + 2 * radius * t, cy - radius, cz))
        elif s < 0.5:
            t = (s - 0.25) / 0.25
            pts.append((cx + radius, cy - radius + 2 * radius * t, cz))
        elif s < 0.75:
            t = (s - 0.5) / 0.25
            pts.append((cx + radius - 2 * radius * t, cy + radius, cz))
        else:
            t = (s - 0.75) / 0.25
            pts.append((cx - radius, cy + radius - 2 * radius * t, cz))
    pts.append(pts[0])  # close loop

    W_bare = np.eye(2, dtype=complex)
    W_drs = np.eye(2, dtype=complex)
    for a in range(len(pts) - 1):
        k0 = np.array(pts[a])
        k1 = np.array(pts[a + 1])
        # bare photon frames
        pg0 = photon_eigenvectors(k0, edges, cycles, edge_lookup)
        pg1 = photon_eigenvectors(k1, edges, cycles, edge_lookup)
        W_bare = W_bare @ link_variable(pg0, pg1)
        # dressed frames
        d0, _ = dressed_photon_frame(k0, edges, cycles, edge_lookup, bonds,
                                     pi_kind, walker_targets)
        d1, _ = dressed_photon_frame(k1, edges, cycles, edge_lookup, bonds,
                                     pi_kind, walker_targets)
        W_drs = W_drs @ link_variable(d0, d1)

    return (abs(la.det(W_bare)), np.angle(la.det(W_bare)),
            abs(la.det(W_drs)), np.angle(la.det(W_drs)))


def main():
    print("=" * 76)
    print(" L6 Sprint B — relative-holonomy gate (coupled photon-walker)")
    print("=" * 76)
    print()

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    edge_lookup = build_edge_lookup(edges)
    print()

    h_KP = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
    walker_targets = [h_KP, h_KP.conjugate()]  # the K_P {+h, +h*} sector

    for pi_kind in ["antisym", "sym"]:
        print("=" * 76)
        print(f" pi-lift: {pi_kind.upper()}"
              f"  ({'parity-odd; the beta-birefringence lift' if pi_kind=='antisym' else 'parity-even cross-check'})")
        print("=" * 76)

        # --- (1) c_1 of bare photon (reference; expect ~0) vs dressed ---
        print()
        print(" (1) First Chern number on (k_x,k_y) slices")
        print(f"     {'k_z':>7}  {'c1_bare':>12}  {'c1_dressed':>12}  {'min_rank(D)':>12}")
        print(f"     {'-'*7}  {'-'*12}  {'-'*12}  {'-'*12}")
        from srs_photon_berry import slice_chern_number
        N = 14
        kz_vals = [0.0, 0.1, 0.2, 0.25, -0.15, 0.33]
        dressed_c1s = []
        for kz in kz_vals:
            c1_bare = slice_chern_number(kz, N, edges, cycles, edge_lookup)
            c1_drs, min_rank = slice_chern_dressed(
                kz, N, edges, cycles, edge_lookup, bonds, pi_kind, walker_targets)
            cb = f"{c1_bare:>12.5f}" if c1_bare is not None else f"{'None':>12}"
            cd = f"{c1_drs:>12.5f}" if c1_drs is not None else f"{'None':>12}"
            mr = f"{min_rank:>12}" if min_rank is not None else f"{'None':>12}"
            print(f"     {kz:>7.3f}  {cb}  {cd}  {mr}")
            if c1_drs is not None:
                dressed_c1s.append(c1_drs)

        # --- (2) small-loop Wilson loop near Gamma (acoustic regime) ---
        print()
        print(" (2) Small-loop Wilson loop near Gamma (acoustic regime)")
        print(f"     loop center offset from Gamma, shrinking radius:")
        print(f"     {'center':>22}  {'radius':>9}  {'|detW_bare|':>12}  "
              f"{'argW_bare':>10}  {'|detW_drs|':>11}  {'argW_drs':>10}")
        gamma_offsets = [
            ((0.05, 0.05, 0.05), 0.02),
            ((0.02, 0.02, 0.02), 0.008),
            ((0.01, 0.01, 0.01), 0.004),
            ((0.005, 0.005, 0.005), 0.002),
        ]
        drs_args = []
        for center, radius in gamma_offsets:
            try:
                ab, arb, ad, ard = small_loop_wilson(
                    center, radius, 40, edges, cycles, edge_lookup,
                    bonds, pi_kind, walker_targets)
                cstr = f"({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})"
                print(f"     {cstr:>22}  {radius:>9.4f}  {ab:>12.4f}  "
                      f"{arb:>+10.4f}  {ad:>11.4f}  {ard:>+10.4f}")
                drs_args.append(ard)
            except Exception as e:
                print(f"     center {center}: FAILED ({type(e).__name__}: {e})")

        # --- (3) verdict for this lift ---
        print()
        print(f" (3) {pi_kind.upper()} verdict:")
        if dressed_c1s:
            max_abs_c1 = max(abs(c) for c in dressed_c1s)
            # round to nearest integer to test topological non-triviality
            nearest_int = round(np.median([c for c in dressed_c1s]))
            integer_like = all(abs(c - round(c)) < 0.15 for c in dressed_c1s)
            nontrivial = max_abs_c1 > 0.3 and integer_like and nearest_int != 0
            print(f"     dressed c_1 values: "
                  f"{['%.4f' % c for c in dressed_c1s]}")
            print(f"     max|c_1(dressed)| = {max_abs_c1:.4f}; "
                  f"integer-like: {integer_like}; "
                  f"nearest int (median): {nearest_int}")
            if nontrivial:
                print(f"     -> dressed bundle carries NON-TRIVIAL Chern "
                      f"(c_1 ~ {nearest_int}) while bare photon c_1 = 0.")
                print(f"     -> {pi_kind.upper()} GATE INDICATION: PASS")
            else:
                print(f"     -> dressed bundle c_1 ~ 0 (trivial), same as "
                      f"bare photon.")
                print(f"     -> {pi_kind.upper()} GATE INDICATION: FAIL")
        else:
            print(f"     dressed Chern computation failed at all k_z "
                  f"(rank-deficiency / singular dressed frame).")
            print(f"     -> {pi_kind.upper()} GATE INDICATION: FAIL "
                  f"(no well-defined dressed bundle)")
        print()

    print("=" * 76)
    print(" SPRINT B GATE — overall reading")
    print("=" * 76)
    print()
    print(" The load-bearing lift is ANTISYM (the parity-odd lift the")
    print(" UNIQUE-THEOREM-GRADE beta cosmic birefringence uses). Gate PASS")
    print(" requires the ANTISYM dressed bundle to carry non-trivial Chern /")
    print(" Wilson-loop winding at generic k while bare photon c_1 = 0.")
    print(" SYM is a cross-check only.")
    print()
    print(" Interpretation:")
    print("   PASS  -> photon-walker coupling has symmetry-independent")
    print("            topological content; Sprint C (extract collective-mode")
    print("            dispersion from the relative connection) attemptable.")
    print("   FAIL  -> coupling has no generic-k geometric content; route (b)")
    print("            dead; L6 wall confirmed at gate 2; cosmology cluster")
    print("            {r_s, theta_*, sigma_8, n_s} Scenario 3 FINAL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
