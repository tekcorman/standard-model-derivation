#!/usr/bin/env python3
"""
srs-z = bipartite double cover of srs (graph-theoretic verification + prediction mining).

HYPOTHESIS (raised 2026-05-01 PM): srs-z's primitive-cell quotient (Q_3 cube graph)
is the bipartite double cover of srs's primitive-cell quotient (K_4). If true:
  - srs-z's state space is doubled (8 atoms vs 4) with bipartite pairing
  - Bipartite structure provides natural Z_2 grading — SUSY-shaped pairing operator
  - Each srs eigenvalue λ gives two srs-z eigenvalues ±λ (bipartite spectral mirror)
  - Framework's Bayesian observer on srs-z naturally derives "doubled" predictions

This probe:
  1. Generates srs's primitive K_4 quotient (4 atoms, 6 edges) from RCSR data via
     spglib (extracting primitive cell from I-centered I4_132 by halving).
  2. Constructs the abstract bipartite double cover of K_4: 8 vertices, 12 edges,
     bipartite, 3-regular.
  3. Generates srs-z's quotient (8 atoms, 12 edges) directly from RCSR.
  4. Verifies graph isomorphism: srs-z's quotient ≅ bipartite double of K_4.
  5. Computes Stark-Terras factorization on both and compares spectra.
  6. Mines what the framework's prediction machinery yields on srs-z:
       - Dark coefficient c (compare to srs's 5/12)
       - Sakharov chain length M = N_edges/cell (compare to srs's 6)
       - V_us = k²/(g·N_atoms) with srs-z parameters (compare to srs's 9/40)
       - h eigenvalue at C₃-stabilized k-point (compare to srs's (√3+i√5)/2 at k_P)
       - Multiplicity of the K-rational h (compare to srs's 2 → SUSY-flavored 4?)

NO srs-specific outputs imported as targets. We compute srs-z's intrinsic values
and identify their structural relationship to srs's via the bipartite-double-cover
algebra.
"""

import sys
import os
import numpy as np
import math
import spglib
from numpy.linalg import eigvalsh, eigvals
from itertools import permutations, combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import (
    parse_rcsr_3dall, get_space_group_ops, orbit_of, reconstruct_bonds,
    bloch_hashimoto, build_directed_edges, identify_irrational
)


# =============================================================================
# PRIMITIVE CELL EXTRACTION (for I-centered groups)
# =============================================================================

def to_primitive_I_centered(positions, tol=1e-6):
    """For I-centered crystal: halve orbit by removing body-centered duplicates.
    Two positions p, q are body-centered equivalent if q ≡ p + (1/2,1/2,1/2) (mod 1).
    Keep the lex-smallest representative of each pair.
    """
    n = len(positions)
    used = [False] * n
    primitive = []
    bcent = np.array([0.5, 0.5, 0.5])
    for i in range(n):
        if used[i]:
            continue
        p = positions[i]
        # Find body-centered partner index
        partner_idx = None
        for j in range(i + 1, n):
            if used[j]:
                continue
            diff = (positions[j] - p - bcent) % 1.0
            diff = np.where(diff > 0.5, diff - 1.0, diff)
            if np.linalg.norm(diff) < tol:
                partner_idx = j
                break
        # Keep position p (lex-smaller), mark partner as used
        primitive.append(p)
        used[i] = True
        if partner_idx is not None:
            used[partner_idx] = True
    return np.array(primitive)


# =============================================================================
# ABSTRACT BIPARTITE DOUBLE COVER
# =============================================================================

def bipartite_double_cover(adj_matrix):
    """Construct the bipartite double cover of a simple graph G.

    Input: n×n adjacency matrix A of G.
    Output: 2n×2n adjacency matrix of bipartite double cover B(G).
    Vertices labeled (v, 0) for v in [0,n) and (v, 1) for v in [n, 2n).
    Edges: ((u, 0), (v, 1)) and ((v, 0), (u, 1)) for each edge uv in G.
    """
    n = len(adj_matrix)
    A = np.zeros((2 * n, 2 * n), dtype=int)
    for u in range(n):
        for v in range(u + 1, n):
            if adj_matrix[u, v] > 0:
                # Edge (u, v) in G → edges (u_0, v_1) and (v_0, u_1) in B(G)
                A[u, n + v] += adj_matrix[u, v]
                A[n + v, u] += adj_matrix[u, v]
                A[v, n + u] += adj_matrix[u, v]
                A[n + u, v] += adj_matrix[u, v]
    return A


# =============================================================================
# GRAPH ISOMORPHISM CHECK (small graphs, brute force)
# =============================================================================

def graph_invariants(adj_matrix, tol=1e-7):
    """Compute graph invariants: spectrum + degree sequence + co-spectrum (line graph spectrum)."""
    n = len(adj_matrix)
    A = adj_matrix.astype(float)
    spec = sorted(np.real(eigvalsh(A)))
    spec = [round(x, 6) for x in spec]
    degrees = sorted(np.sum(A > 0, axis=1).tolist())
    return {
        'n_vertices': n,
        'n_edges': int(np.sum(A) // 2),
        'degree_sequence': degrees,
        'spectrum': spec,
    }


def check_iso_brute(A, B, max_n=10):
    """Brute-force isomorphism check between adjacency matrices A, B.
    Only for small graphs (n ≤ 10).
    Returns (is_iso, perm) — permutation maps A's vertices to B's.
    """
    n = len(A)
    if len(B) != n:
        return False, None
    if n > max_n:
        return None, None  # too large for brute force
    A_arr = np.array(A)
    B_arr = np.array(B)
    for perm in permutations(range(n)):
        P = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            P[i, j] = 1
        # Check if P A P^T == B
        if np.array_equal(P @ A_arr @ P.T, B_arr):
            return True, perm
    return False, None


# =============================================================================
# STARK-TERRAS HASHIMOTO SPECTRUM ON FINITE QUOTIENT
# =============================================================================

def stark_terras_spectrum(adj_matrix, k_coord, n_V=None, n_E=None):
    """Apply Stark-Terras factorization:
        det(uI − B) = (u² − 1)^(|E|−|V|) · ∏_λ (u² − λ u + (k−1))
    where the product is over the |V| adjacency eigenvalues λ.

    Returns dict with bipartite eigenvalues and oscillatory eigenvalues from each
    quadratic factor.
    """
    A = np.array(adj_matrix, dtype=float)
    n = len(A)
    if n_V is None: n_V = n
    if n_E is None: n_E = int(np.sum(A) // 2)
    eigvals_adj = sorted(np.real(eigvalsh(A)), reverse=True)
    bipartite = [(1.0, n_E - n_V), (-1.0, n_E - n_V)]
    oscillatory = []
    for lam in eigvals_adj:
        disc = lam * lam - 4 * (k_coord - 1)
        if disc >= 0:
            sd = math.sqrt(disc)
            u_plus = (lam + sd) / 2.0
            u_minus = (lam - sd) / 2.0
            oscillatory.append((u_plus, u_minus, 'real', lam))
        else:
            sd = math.sqrt(-disc)
            u_plus = complex(lam / 2.0, sd / 2.0)
            u_minus = complex(lam / 2.0, -sd / 2.0)
            oscillatory.append((u_plus, u_minus, 'complex', lam))
    return {
        'adj_eigenvalues': eigvals_adj,
        'bipartite_eigs': bipartite,
        'oscillatory_eigs': oscillatory,
        'n_V': n_V,
        'n_E': n_E,
        'k': k_coord,
    }


# =============================================================================
# FRAMEWORK PREDICTION FORMULAS APPLIED TO ANY (|V|, |E|, k, g)
# =============================================================================

def framework_predictions_on_substrate(n_V, n_E, k, g):
    """Compute the framework's prediction formulas on a generic 3-c substrate
    with given primitive cell parameters. These are formulas the framework
    derives on srs's K_4 quotient — applying them to other substrates is
    a structural exercise, not a claim that they GIVE the same physics.

    All values are in K = ℚ(√2, √3, √5) where applicable.
    """
    return {
        'V_us':       k * k / (g * n_V),                                     # = 9/40 on srs (k=3, g=10, |V|=4)
        'V_cb':       (k - 1)**8 / (g * n_V) if g > 0 else None,             # placeholder; framework V_cb is more involved
        'dark_c':     (2 * (n_E - n_V) + 1) / (2 * n_E),                     # = 5/12 on srs's K_4 (|V|=4, |E|=6); = 9/24 = 3/8 on Q_3
        'M_chain':    n_E,                                                    # Sakharov chain length = N_edges/cell; = 6 on srs
        'M_handshake': n_V * k // 2,                                          # alternative form
        'cycle_density': g / n_V,                                             # girth cycles per atom
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("srs-z = BIPARTITE DOUBLE COVER OF srs — verification + prediction mining")
    print("=" * 78)

    # ----- Step 1: get srs primitive quotient (K_4) -------------------------
    print("\n" + "-" * 78)
    print("STEP 1: Extract srs's primitive K_4 quotient")
    print("-" * 78)
    entries = parse_rcsr_3dall('/tmp/rcsr_3d_current.txt', ['srs', 'srs-z'])
    srs = entries['srs']
    srs_z = entries['srs-z']

    rotations, translations, _, _ = get_space_group_ops('I4(1)32')
    v_frac = np.array(srs['vertex_orbits'][0]['cartesian'])
    m_frac = np.array(srs['edge_orbits'][0]['cartesian'])
    atom_orbit_conv = orbit_of(v_frac, rotations, translations)
    midpoint_orbit_conv = orbit_of(m_frac, rotations, translations)
    print(f"  srs conventional cell: {len(atom_orbit_conv)} atoms, {len(midpoint_orbit_conv)} midpoints")

    atom_orbit_prim = to_primitive_I_centered(atom_orbit_conv)
    midpoint_orbit_prim = to_primitive_I_centered(midpoint_orbit_conv)
    print(f"  srs primitive cell:    {len(atom_orbit_prim)} atoms, {len(midpoint_orbit_prim)} midpoints")
    print(f"  (Expected: 4 atoms, 6 midpoints for K_4 quotient)")

    # Reconstruct primitive bonds
    bonds_srs_prim = reconstruct_bonds(atom_orbit_prim, midpoint_orbit_prim, tol=1e-3, max_shift=2)
    n_resolved = sum(1 for b in bonds_srs_prim if b is not None)
    print(f"  Bonds in primitive: {n_resolved}/{len(midpoint_orbit_prim)}")

    # Build adjacency
    n_v_srs = len(atom_orbit_prim)
    A_srs = np.zeros((n_v_srs, n_v_srs), dtype=int)
    for b in bonds_srs_prim:
        if b is None: continue
        i, j, _ = b
        A_srs[i, j] += 1
        if i != j:
            A_srs[j, i] += 1
    print(f"  srs primitive adjacency matrix:")
    print(A_srs)
    inv_srs = graph_invariants(A_srs)
    print(f"  Graph invariants: |V|={inv_srs['n_vertices']}, |E|={inv_srs['n_edges']}, "
          f"degree seq={inv_srs['degree_sequence']}, spectrum={inv_srs['spectrum']}")
    is_K4 = (inv_srs['n_vertices'] == 4 and inv_srs['n_edges'] == 6 and
             all(d == 3 for d in inv_srs['degree_sequence']) and
             inv_srs['spectrum'] == [-1.0, -1.0, -1.0, 3.0])
    print(f"  → Is K_4? {is_K4}")

    # ----- Step 2: Construct bipartite double cover of K_4 ------------------
    print("\n" + "-" * 78)
    print("STEP 2: Bipartite double cover of K_4 (abstract)")
    print("-" * 78)
    # K_4 = full graph on 4 vertices
    K4 = np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)
    print(f"  K_4 adjacency:\n{K4}")
    BD_K4 = bipartite_double_cover(K4)
    print(f"  Bipartite double cover B(K_4):\n{BD_K4}")
    inv_BD = graph_invariants(BD_K4)
    print(f"  Graph invariants: |V|={inv_BD['n_vertices']}, |E|={inv_BD['n_edges']}, "
          f"spectrum={inv_BD['spectrum']}")

    # ----- Step 3: Get srs-z primitive quotient (8 atoms, 12 edges) ---------
    print("\n" + "-" * 78)
    print("STEP 3: Extract srs-z's primitive quotient")
    print("-" * 78)
    rotations_z, translations_z, _, _ = get_space_group_ops('P4(1)32')
    v_frac_z = np.array(srs_z['vertex_orbits'][0]['cartesian'])
    m_frac_z = np.array(srs_z['edge_orbits'][0]['cartesian'])
    atom_orbit_z = orbit_of(v_frac_z, rotations_z, translations_z)
    midpoint_orbit_z = orbit_of(m_frac_z, rotations_z, translations_z)
    print(f"  srs-z primitive cell: {len(atom_orbit_z)} atoms, {len(midpoint_orbit_z)} midpoints")

    bonds_srs_z = reconstruct_bonds(atom_orbit_z, midpoint_orbit_z, tol=1e-3, max_shift=2)
    n_v_z = len(atom_orbit_z)
    A_z = np.zeros((n_v_z, n_v_z), dtype=int)
    for b in bonds_srs_z:
        if b is None: continue
        i, j, _ = b
        A_z[i, j] += 1
        if i != j:
            A_z[j, i] += 1
    print(f"  srs-z primitive adjacency matrix:")
    print(A_z)
    inv_z = graph_invariants(A_z)
    print(f"  Graph invariants: |V|={inv_z['n_vertices']}, |E|={inv_z['n_edges']}, "
          f"degree seq={inv_z['degree_sequence']}, spectrum={inv_z['spectrum']}")

    # ----- Step 4: Check graph isomorphism between BD(K_4) and srs-z's Q_3 --
    print("\n" + "-" * 78)
    print("STEP 4: Graph isomorphism check — BD(K_4) ?= srs-z's quotient")
    print("-" * 78)
    if inv_BD == inv_z:
        print(f"  Graph invariants MATCH ✓")
    else:
        print(f"  Graph invariants DIFFER:\n    BD(K_4): {inv_BD}\n    srs-z:   {inv_z}")
    is_iso, perm = check_iso_brute(BD_K4, A_z, max_n=10)
    if is_iso is None:
        print(f"  Graph too large for brute-force iso check.")
    elif is_iso:
        print(f"  Graphs ARE ISOMORPHIC (vertex permutation: {perm})")
    else:
        print(f"  Graphs NOT isomorphic (despite matching invariants — surprising)")

    # ----- Step 5: Stark-Terras spectra ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: Stark-Terras Hashimoto spectra")
    print("-" * 78)
    print("\n  srs (K_4 primitive, |V|=4, |E|=6, k=3):")
    st_srs = stark_terras_spectrum(A_srs, k_coord=3)
    print(f"    Adjacency eigenvalues: {st_srs['adj_eigenvalues']}")
    print(f"    Bipartite (u²−1)^{st_srs['n_E']-st_srs['n_V']}: u=±1 each mult {st_srs['n_E']-st_srs['n_V']}")
    print(f"    Oscillatory roots from (u² − λu + 2):")
    for u_plus, u_minus, kind, lam in st_srs['oscillatory_eigs']:
        if kind == 'complex':
            mod_sq = abs(u_plus)**2
            ram = "✓ RAMANUJAN" if abs(mod_sq - 2) < 1e-6 else f"|u|²={mod_sq}"
            re_id = identify_irrational(u_plus.real)
            im_id = identify_irrational(abs(u_plus.imag))
            print(f"      λ={lam:+.3f}: u = {u_plus.real:+.4f} ± {abs(u_plus.imag):.4f}i  "
                  f"({ram}; Re~{re_id}, Im~{im_id})")
        else:
            print(f"      λ={lam:+.3f}: u = {u_plus:.4f}, {u_minus:.4f}")

    print("\n  srs-z (Q_3 primitive, |V|=8, |E|=12, k=3):")
    st_z = stark_terras_spectrum(A_z, k_coord=3)
    print(f"    Adjacency eigenvalues: {st_z['adj_eigenvalues']}")
    print(f"    Bipartite (u²−1)^{st_z['n_E']-st_z['n_V']}: u=±1 each mult {st_z['n_E']-st_z['n_V']}")
    print(f"    Oscillatory roots from (u² − λu + 2):")
    for u_plus, u_minus, kind, lam in st_z['oscillatory_eigs']:
        if kind == 'complex':
            mod_sq = abs(u_plus)**2
            ram = "✓ RAMANUJAN" if abs(mod_sq - 2) < 1e-6 else f"|u|²={mod_sq}"
            re_id = identify_irrational(u_plus.real)
            im_id = identify_irrational(abs(u_plus.imag))
            print(f"      λ={lam:+.3f}: u = {u_plus.real:+.4f} ± {abs(u_plus.imag):.4f}i  "
                  f"({ram}; Re~{re_id}, Im~{im_id})")
        else:
            print(f"      λ={lam:+.3f}: u = {u_plus:.4f}, {u_minus:.4f}")

    # ----- Step 6: Framework predictions on srs vs srs-z --------------------
    print("\n" + "-" * 78)
    print("STEP 6: Framework prediction formulas — srs vs srs-z")
    print("-" * 78)
    pred_srs = framework_predictions_on_substrate(n_V=4, n_E=6, k=3, g=10)
    pred_z = framework_predictions_on_substrate(n_V=8, n_E=12, k=3, g=10)
    print(f"  {'Quantity':<25s} {'srs (K_4)':<20s} {'srs-z (Q_3)':<20s}  Ratio (srs-z/srs)")
    print(f"  {'-'*80}")
    for k in pred_srs:
        a, b = pred_srs[k], pred_z[k]
        if a is None or b is None:
            print(f"  {k:<25s} {'-':<20s} {'-':<20s}")
            continue
        ratio = b / a if a != 0 else float('inf')
        print(f"  {k:<25s} {str(a)[:18]:<20s} {str(b)[:18]:<20s}  {ratio:.4f}")

    # Honest interpretation
    print("\n  Interpretation:")
    print(f"    V_us(srs)   = 9/40 = {9/40:.4f}  matches PDG 0.22501 (the framework's prediction)")
    print(f"    V_us(srs-z) = 9/80 = {9/80:.4f}  what the framework formula gives ON srs-z's substrate")
    print(f"    Ratio = 1/2 — exactly the bipartite-double-cover doubling.")
    print(f"")
    print(f"    dark c(srs)   = 5/12 ≈ 0.4167")
    print(f"    dark c(srs-z) = 9/24 = 3/8 = 0.375")
    print(f"    Different value, both rational and in K = ℚ(√2,√3,√5).")
    print(f"")
    print(f"    M chain(srs)   = 6   → α₁^M = (2/3)^48 ≈ 3.5e-9")
    print(f"    M chain(srs-z) = 12  → α₁^M = (2/3)^96 ≈ 1.2e-17")
    print(f"    The chain length DOUBLES under bipartite double cover.")

    # ----- Step 7: Bloch h at C_3 saddle —————
    print("\n" + "-" * 78)
    print("STEP 7: Bloch h at C₃-stabilized k-points (mining the K-rational saddle)")
    print("-" * 78)
    # For srs (using primitive cell & primitive bonds): construct B(k) and probe at body-diagonal mid k=(1/4,1/4,1/4)
    arcs_srs = build_directed_edges([b for b in bonds_srs_prim if b is not None])
    arcs_z = build_directed_edges([b for b in bonds_srs_z if b is not None])

    print(f"\n  srs primitive: {len(arcs_srs)} directed arcs, B(k) is {len(arcs_srs)}×{len(arcs_srs)}")
    print(f"  srs-z primitive: {len(arcs_z)} directed arcs, B(k) is {len(arcs_z)}×{len(arcs_z)}")

    # Probe BZ at C_3-stabilized points (body-diagonal axis k=(t,t,t))
    print(f"\n  k-point sweep along (t, t, t) [body-diagonal C_3 axis], looking for K-rational h:")
    print(f"  {'t':>6s}   {'srs h max':>30s}  {'srs-z h max':>30s}")
    for t in [0.0, 0.125, 0.25, 0.375, 0.5]:
        k_frac = np.array([t, t, t])
        B_srs_k = bloch_hashimoto(arcs_srs, k_frac, n_v_srs)
        B_z_k = bloch_hashimoto(arcs_z, k_frac, n_v_z)
        eigs_srs = eigvals(B_srs_k)
        eigs_z = eigvals(B_z_k)
        # Find the eigenvalue with largest Im (the "characteristic" complex eigenvalue)
        eigs_srs_complex = [e for e in eigs_srs if abs(e.imag) > 1e-6]
        eigs_z_complex = [e for e in eigs_z if abs(e.imag) > 1e-6]
        if eigs_srs_complex:
            srs_max = max(eigs_srs_complex, key=lambda e: e.imag)
            srs_re_id = identify_irrational(srs_max.real) or f"{srs_max.real:.3f}"
            srs_im_id = identify_irrational(abs(srs_max.imag)) or f"{abs(srs_max.imag):.3f}"
            srs_str = f"({srs_re_id})+i·({srs_im_id})"
        else:
            srs_str = "no complex"
        if eigs_z_complex:
            z_max = max(eigs_z_complex, key=lambda e: e.imag)
            z_re_id = identify_irrational(z_max.real) or f"{z_max.real:.3f}"
            z_im_id = identify_irrational(abs(z_max.imag)) or f"{abs(z_max.imag):.3f}"
            z_str = f"({z_re_id})+i·({z_im_id})"
        else:
            z_str = "no complex"
        print(f"  {t:6.3f}   {srs_str:>30s}  {z_str:>30s}")

    # Probe specifically at srs's framework P-point (k_P = (1/4, 1/4, 1/4)) for both
    print(f"\n  At k_P = (1/4, 1/4, 1/4):")
    k_frac = np.array([0.25, 0.25, 0.25])
    B_srs_kP = bloch_hashimoto(arcs_srs, k_frac, n_v_srs)
    B_z_kP = bloch_hashimoto(arcs_z, k_frac, n_v_z)
    eigs_srs_kP = sorted(eigvals(B_srs_kP), key=lambda x: (round(x.real, 5), round(x.imag, 5)))
    eigs_z_kP = sorted(eigvals(B_z_kP), key=lambda x: (round(x.real, 5), round(x.imag, 5)))
    print(f"\n    srs B(k_P) eigenvalues:")
    for e in eigs_srs_kP:
        if abs(e.imag) > 1e-6:
            re_id = identify_irrational(e.real) or f"{e.real:.4f}"
            im_id = identify_irrational(abs(e.imag)) or f"{abs(e.imag):.4f}"
            mod_sq = abs(e)**2
            ram = " ✓ RAM" if abs(mod_sq - 2) < 1e-6 else f" |u|²={mod_sq:.3f}"
            print(f"      {e.real:+.4f} + {e.imag:+.4f}i   (Re~{re_id}, Im~{im_id}){ram}")

    print(f"\n    srs-z B(k_P) eigenvalues:")
    for e in eigs_z_kP:
        if abs(e.imag) > 1e-6:
            re_id = identify_irrational(e.real) or f"{e.real:.4f}"
            im_id = identify_irrational(abs(e.imag)) or f"{abs(e.imag):.4f}"
            mod_sq = abs(e)**2
            ram = " ✓ RAM" if abs(mod_sq - 2) < 1e-6 else f" |u|²={mod_sq:.3f}"
            print(f"      {e.real:+.4f} + {e.imag:+.4f}i   (Re~{re_id}, Im~{im_id}){ram}")

    # Count multiplicities of each unique eigenvalue
    from collections import Counter
    def round_eigs(eigs, prec=5):
        return Counter([(round(e.real, prec), round(e.imag, prec)) for e in eigs])

    cnt_srs = round_eigs(eigs_srs_kP)
    cnt_z = round_eigs(eigs_z_kP)
    print(f"\n  Multiplicity comparison at k_P:")
    print(f"    srs: {len(cnt_srs)} distinct eigenvalues (max mult: {max(cnt_srs.values())})")
    print(f"    srs-z: {len(cnt_z)} distinct eigenvalues (max mult: {max(cnt_z.values())})")


if __name__ == '__main__':
    main()
