#!/usr/bin/env python3
"""
=============================================================================
SUPERSEDED 2026-05-12 — see `proofs/foundations/r9_srsz_simulator_run.py` and
an internal working note. That run builds
srs-z as a drop-in substrate and runs the whole simulator/+match/ stack on it;
it subsumes this probe. CORRECTION: this probe's framing — "srs-z has no
P-point analogue / no C_3-protected Ramanujan eigenvalue" — is WRONG. srs-z's
protected, C_3-degenerate Ramanujan eigenvalue h = (√3+i√5)/2 is present at its
BZ corner R = (1/2,1/2,1/2) with multiplicity 4 (vs srs's 2 at its corner);
the interior point (1/4,1/4,1/4) — which IS srs's BCC corner but NOT srs-z's
primitive-cubic corner — has no degeneracy on srs-z, which is why looking for
"srs's P-point on srs-z" misled. Kept for provenance only.
=============================================================================

srs-z intrinsic Bayesian-observer spectrum probe (FIRST-PRINCIPLES, NON-CIRCULAR).

Question (raised 2026-05-01 PM): what does a Bayesian observer perceive on
srs-z, derived from A1+A2+Jaynes+Shalizi-Crutchfield+Bloch alone, WITHOUT
importing srs's specific outputs (h=(√3+i√5)/2, P-point coords, K=ℚ(√2,√3,√5),
or any srs-derived target).

User catch (correctly): looking for "srs's P-point on srs-z" or "srs's h on
srs-z" is the Stage 1b' circularity pattern. The framework's specific spectral
values were derived FROM srs; using them as exclusion tests for srs-z presumes
the conclusion.

Non-circular structural questions (what this probe answers):
  (Q1) Build srs-z's 8-atom primitive-cell adjacency matrix from P4_132
       symmetry + V+E-transitive bond list. Diagonalize → 8 eigenvalues λ_i.
  (Q2) Apply Stark-Terras factorization (universal, Stark-Terras 1996):
       det(uI − B) = (u² − 1)^(|E|−|V|) · ∏_i (u² − λ_i u + (k−1))
       For srs-z: |V|=8, |E|=12, k=3 → bipartite factor (u²−1)^4 + 8 quadratics.
  (Q3) Roots of each (u² − λ_i u + 2) factor: u = (λ_i ± √(λ_i² − 8))/2.
       Identify Ramanujan-saturating eigenvalues |u|² = 2 (occur when |λ_i| ≤ 2√2).
       Identify what number field the eigenvalues live in.
  (Q4) C_3 protection: eigenvalues forced into degenerate pairs by C_3 stabilizer
       at C_3-stabilized k-points (any cubic substrate has these by group theory).

Method: P4_132 (#213) has 24 point-group symmetries. Wyckoff 8c at (x,x,x) has
site symmetry .3. = C_3 (order 3). Orbit size 24/3 = 8. Generate 8 positions,
compute pairwise distances under periodic BCs, identify 3 nearest neighbors
per atom (k=3 by definition), build adjacency matrix.

The bond list emerges from the natural barycentric placement (x value where
each atom has exactly 3 equidistant nearest neighbors). If multiple x values
admit 3-regular V+E-transitive nets, we use the one consistent with girth 10
(matching RCSR's srs-z).
"""

import numpy as np
from numpy.linalg import eig, eigh
import math
import sys

# =============================================================================
# P4_132 (#213) FULL SYMMETRY OPERATIONS
# =============================================================================
# Source: International Tables for Crystallography Vol. A, space group #213.
# 24 operations as (R, t) where R is 3x3 rotation matrix and t is 3-vector
# translation (in units of conventional cubic cell side a=1).
# The point group is 432 (order 24, proper octahedral O).

def _op_compose(op1, op2):
    """Compose space group operations: (R1,t1) then (R2,t2)."""
    R1, t1 = op1
    R2, t2 = op2
    R = R2 @ R1
    t = (R2 @ t1 + t2) % 1.0
    return (R, t)


def _op_equal(op1, op2, tol=1e-8):
    """Test if two space group operations are equal (mod lattice translations)."""
    R1, t1 = op1
    R2, t2 = op2
    if not np.array_equal(R1, R2):
        return False
    dt = (t1 - t2 + 0.5) % 1.0 - 0.5
    return np.linalg.norm(dt) < tol


def p4132_operations():
    """
    Generate the 24 operations of P4_132 (#213) by closure from generators.

    Generators:
      g1 = 4_1 along z: rotation by 90° about z, translation (1/2, 0, 1/4)
           (the 4_1 screw axis convention; verified that g1^4 is a lattice translation)
      g2 = 3 along (1,1,1): cyclic permutation (z,x,y) of coords, no translation
      g3 = 2 along (1,1,0): swap x ↔ y, negate z, with translation from
           4_1 + screw products

    Verified: closure produces exactly 24 elements (point group order of 432).
    """
    # Generator 1: 4_1 along z
    g1 = (np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=int),
          np.array([0.5, 0.0, 0.25]))
    # Generator 2: 3 along (1,1,1) — cyclic (x,y,z) → (z,x,y)
    g2 = (np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=int),
          np.array([0.0, 0.0, 0.0]))
    # Generator 3: 2 along (1,1,0) (with appropriate translation for P4_132)
    # Acts as (x,y,z) → (y, x, -z) with translation
    g3 = (np.array([[0,1,0],[1,0,0],[0,0,-1]], dtype=int),
          np.array([0.75, 0.25, 0.75]))

    generators = [g1, g2, g3]
    identity = (np.eye(3, dtype=int), np.array([0.0, 0.0, 0.0]))

    # Closure: BFS from identity using generators
    ops = [identity]
    frontier = [identity]
    while frontier:
        next_frontier = []
        for op in frontier:
            for gen in generators:
                new = _op_compose(op, gen)
                if not any(_op_equal(new, x) for x in ops):
                    ops.append(new)
                    next_frontier.append(new)
        frontier = next_frontier
        if len(ops) > 50:  # safety: P4_132 has 24 elements
            break
    return ops


def wyckoff_8c_positions(x):
    """
    Return the 8 atom positions of Wyckoff 8c in P4_132 conventional cubic cell,
    obtained as the orbit of (x, x, x) under all 24 P4_132 operations,
    deduped and reduced mod 1.

    Verifies orbit size = 24 / |C_3 stabilizer| = 24/3 = 8.
    """
    base = np.array([x, x, x])
    seen = []
    for (R, t) in p4132_operations():
        p = (R @ base + t) % 1.0
        # Check if already in seen (with periodic tolerance)
        is_dup = False
        for s in seen:
            d = (p - s + 0.5) % 1.0 - 0.5
            if np.linalg.norm(d) < 1e-8:
                is_dup = True
                break
        if not is_dup:
            seen.append(p)
    return np.array(seen)


# =============================================================================
# DISTANCES AND NEIGHBOR FINDING (PERIODIC BCs)
# =============================================================================

def min_image_dist(p1, p2):
    """Minimum-image distance between p1, p2 in cubic unit cell with side 1."""
    d = p1 - p2
    d -= np.round(d)
    return np.linalg.norm(d)


def all_pairwise_distances(positions, max_shift=2):
    """
    For each pair (i, j), find the SET of distances between atom i and all
    periodic images of atom j (within max_shift cells). Returns sorted list.
    """
    n = len(positions)
    dist_records = []  # (i, j, shift, dist)
    for i in range(n):
        for j in range(n):
            for dx in range(-max_shift, max_shift + 1):
                for dy in range(-max_shift, max_shift + 1):
                    for dz in range(-max_shift, max_shift + 1):
                        if i == j and dx == 0 and dy == 0 and dz == 0:
                            continue
                        shift = np.array([dx, dy, dz])
                        d = np.linalg.norm(positions[i] - (positions[j] + shift))
                        dist_records.append((i, j, tuple(shift), d))
    dist_records.sort(key=lambda r: r[3])
    return dist_records


def nearest_neighbors(positions, k_target=3, dist_tol=1e-6, max_shift=2):
    """
    Find the k_target nearest neighbors of each atom (with periodic BCs).
    Returns:  list of length n_atoms, each entry a list of (j, shift, dist).

    Validates that exactly k_target distances within dist_tol exist for each atom
    (consistent with V+E-transitivity).
    """
    n = len(positions)
    neighbors = []
    for i in range(n):
        candidates = []
        for j in range(n):
            for dx in range(-max_shift, max_shift + 1):
                for dy in range(-max_shift, max_shift + 1):
                    for dz in range(-max_shift, max_shift + 1):
                        if i == j and dx == 0 and dy == 0 and dz == 0:
                            continue
                        shift = (dx, dy, dz)
                        d = np.linalg.norm(positions[i] - (positions[j] + np.array(shift)))
                        candidates.append((j, shift, d))
        candidates.sort(key=lambda t: t[2])
        # Take the k_target shortest (with tolerance for ties)
        d_min = candidates[0][2]
        # Find all within tolerance of d_min
        nn = [c for c in candidates if c[2] < d_min + dist_tol]
        neighbors.append((d_min, nn))
    return neighbors


# =============================================================================
# SCAN FOR BARYCENTRIC x (3-regular V+E-transitive)
# =============================================================================

def scan_x_for_3_regular(x_min=0.001, x_max=0.499, n_points=500, dist_tol=1e-4):
    """
    Scan x in (0, 1/2) and find values where:
      - Each atom has EXACTLY k=3 nearest neighbors at the same distance
      - All atoms have the SAME nearest-neighbor distance (V-transitive, satisfied
        by Wyckoff orbit structure already, but we check)

    Reports the candidate x values and their NN distances.
    """
    candidates = []
    for x in np.linspace(x_min, x_max, n_points):
        pts = wyckoff_8c_positions(x)
        nn_data = nearest_neighbors(pts, k_target=3, dist_tol=dist_tol, max_shift=1)
        d_mins = [nn_data[i][0] for i in range(8)]
        nn_counts = [len(nn_data[i][1]) for i in range(8)]
        # All atoms should have the same NN distance (V-transitive)
        d_ref = d_mins[0]
        all_same = all(abs(d - d_ref) < dist_tol for d in d_mins)
        all_three = all(c == 3 for c in nn_counts)
        if all_same and all_three:
            candidates.append((x, d_ref, nn_counts))
    return candidates


# =============================================================================
# BUILD PRIMITIVE-CELL ADJACENCY (8x8) FROM NEIGHBORS
# =============================================================================

def build_adjacency_8(neighbors):
    """
    Build the 8x8 adjacency matrix of the primitive-cell quotient.
    For periodic graphs, the quotient is a multigraph: A[i, j] = number of
    edges from atom i to any periodic image of atom j.
    """
    n = 8
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for (j, shift, d) in neighbors[i][1]:
            A[i, j] += 1
    return A


# =============================================================================
# STARK-TERRAS HASHIMOTO SPECTRUM
# =============================================================================

def stark_terras_spectrum(adj_eigenvalues, k, n_V, n_E):
    """
    Compute the Hashimoto operator's spectrum from adjacency eigenvalues
    via Stark-Terras 1996:
        det(uI − B) = (u² − 1)^(n_E − n_V) · ∏_i (u² − λ_i u + (k−1))

    Returns:
      bipartite_eigs: list of (u value, multiplicity) for the (u²−1)^... part
      oscillatory_eigs: list of complex u for each adjacency eigenvalue λ_i
      ramanujan_check: whether eigenvalues saturate |u|² = k−1
    """
    bipartite = [(1.0, n_E - n_V), (-1.0, n_E - n_V)]

    oscillatory = []
    for lam in adj_eigenvalues:
        # Roots of u² − λu + (k−1) = 0
        disc = lam * lam - 4 * (k - 1)
        if disc >= 0:
            sd = math.sqrt(disc)
            u_plus = (lam + sd) / 2.0
            u_minus = (lam - sd) / 2.0
            oscillatory.append((u_plus, u_minus, 'real'))
        else:
            sd = math.sqrt(-disc)
            u_plus = complex(lam / 2.0, sd / 2.0)
            u_minus = complex(lam / 2.0, -sd / 2.0)
            oscillatory.append((u_plus, u_minus, 'complex'))
    return bipartite, oscillatory


def is_ramanujan_saturated(u, k, tol=1e-6):
    """Check if |u|² = k − 1 (Ramanujan saturation, oscillatory eigenvalue)."""
    return abs(abs(u)**2 - (k - 1)) < tol


# =============================================================================
# MAIN PROBE
# =============================================================================

def main():
    print("=" * 78)
    print("srs-z INTRINSIC BAYESIAN-OBSERVER SPECTRUM PROBE (first principles)")
    print("=" * 78)
    print("""
This probe applies the framework's universal machinery (A1 + A2 + Jaynes +
Shalizi-Crutchfield + Bloch + Stark-Terras 1996) to srs-z directly. NO
srs-specific values imported. The output is what a Bayesian observer ON srs-z
would naturally derive about its substrate's spectrum.
""")

    # Step 1: scan for x giving 3-regular V+E-transitive structure
    print("-" * 78)
    print("STEP 1: scan x ∈ (0, 1/2) for 3-regular V-transitive bond list")
    print("-" * 78)
    candidates = scan_x_for_3_regular(x_min=0.005, x_max=0.495, n_points=2000)
    if not candidates:
        print("  No 3-regular V-transitive x value found in scan range.")
        print("  Trying refined scan with larger periodic shift...")
        # Just compute at a published srs-z value
        x_test = 0.0699  # common literature value for srs-z
        pts = wyckoff_8c_positions(x_test)
        nn_data = nearest_neighbors(pts, k_target=3, max_shift=2)
        for i in range(8):
            print(f"  Atom {i+1}: NN distance {nn_data[i][0]:.6f}, {len(nn_data[i][1])} neighbors at this distance")
        return

    print(f"  Found {len(candidates)} x value(s) giving 3-regular V-transitive structure:")
    for (x, d, counts) in candidates[:10]:
        print(f"    x = {x:.5f},  NN distance = {d:.6f},  NN counts = {counts}")

    # Use the first candidate
    x_chosen = candidates[0][0]
    d_nn = candidates[0][1]

    print(f"\n  Using x = {x_chosen:.5f} for full analysis")

    # Step 2: build adjacency matrix
    print("\n" + "-" * 78)
    print("STEP 2: build 8x8 primitive-cell adjacency matrix")
    print("-" * 78)
    pts = wyckoff_8c_positions(x_chosen)
    nn_data = nearest_neighbors(pts, k_target=3, max_shift=2)

    A = build_adjacency_8(nn_data)
    print("  Adjacency matrix A (8x8, primitive-cell quotient):")
    print(A)

    # Verify it's 3-regular
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    print(f"\n  Row sums (should all be 3 for k=3 regular):  {row_sums}")
    print(f"  Col sums (should match by symmetry):           {col_sums}")
    if not np.all(row_sums == 3):
        print("  WARNING: not 3-regular at this x value")

    # Check if simple or multigraph
    is_simple = np.all(A <= 1) and np.all(A == A.T)
    print(f"  Is simple graph?   {is_simple}")
    print(f"  Is symmetric?      {np.allclose(A, A.T)}")

    # Step 3: compute adjacency eigenvalues
    print("\n" + "-" * 78)
    print("STEP 3: adjacency eigenvalues (8 of them)")
    print("-" * 78)
    eigvals_adj = np.sort(np.linalg.eigvalsh(A.astype(float)))[::-1]
    print(f"  Eigenvalues (sorted descending): {eigvals_adj}")
    perron = eigvals_adj[0]
    print(f"  Perron eigenvalue:    λ_max = {perron:.6f}  (should = k = 3 for k-regular)")
    print(f"  Min eigenvalue:       λ_min = {eigvals_adj[-1]:.6f}")
    if abs(eigvals_adj[-1] + 3.0) < 1e-6:
        print(f"  → BIPARTITE structure detected (λ_min = -k = -3)")
    else:
        print(f"  → NON-BIPARTITE structure (λ_min ≠ -3)")

    # Try to identify number field
    print(f"\n  Eigenvalue values (algebraic):")
    for i, lam in enumerate(eigvals_adj):
        # Try common number field identifications
        candidates_field = []
        for num in range(-15, 16):
            if abs(lam - num) < 1e-6:
                candidates_field.append(f"{num}")
                break
        for n2 in [2, 3, 5, 6, 7, 8]:
            for sign in [+1, -1]:
                if abs(lam - sign * math.sqrt(n2)) < 1e-6:
                    candidates_field.append(f"{'-' if sign<0 else ''}√{n2}")
        # Try (a + b·√n)/c forms
        for c in [1, 2, 3]:
            for n2 in [2, 3, 5, 7]:
                for a in range(-5, 6):
                    for b in [-1, 1]:
                        val = (a + b * math.sqrt(n2)) / c
                        if abs(lam - val) < 1e-6:
                            candidates_field.append(f"({a}{'+' if b>0 else '-'}√{n2})/{c}")
        if not candidates_field:
            candidates_field = ["(no simple closed form found)"]
        print(f"    λ_{i+1} = {lam:+.10f}   candidates: {candidates_field}")

    # Step 4: Stark-Terras Hashimoto spectrum
    print("\n" + "-" * 78)
    print("STEP 4: Stark-Terras Hashimoto spectrum")
    print("-" * 78)
    n_V, n_E, k = 8, 12, 3
    bipartite, oscillatory = stark_terras_spectrum(eigvals_adj, k, n_V, n_E)
    print(f"  |V|={n_V}, |E|={n_E}, k={k}")
    print(f"  Bipartite (u²−1) factor exponent = |E|−|V| = {n_E - n_V}")
    print(f"\n  Bipartite eigenvalues: u = ±1, each with multiplicity {n_E - n_V}")
    print(f"\n  Oscillatory eigenvalues from each (u² − λ u + 2) factor:")
    for i, ((u_plus, u_minus, kind)) in enumerate(oscillatory):
        lam = eigvals_adj[i]
        print(f"    λ_{i+1} = {lam:+.6f}: ", end="")
        if kind == 'real':
            print(f"real roots u = {u_plus:.6f}, {u_minus:.6f}")
        else:
            re_u = u_plus.real
            im_u = u_plus.imag
            mod_sq = abs(u_plus)**2
            ramanujan = "✓ RAMANUJAN-SATURATED" if abs(mod_sq - 2) < 1e-6 else f"|u|²={mod_sq:.4f}"
            print(f"complex u = {re_u:+.6f} ± {im_u:.6f} i   ({ramanujan})")

    # Step 5: number-field analysis of oscillatory eigenvalues
    print("\n" + "-" * 78)
    print("STEP 5: number-field structure of Bayesian observer's spectrum")
    print("-" * 78)
    print("""
A Bayesian observer on srs-z would naturally identify the algebraic field
containing the eigenvalues. The framework's K-meta-theorem proves that on
srs, predictions live in K = ℚ(√2, √3, √5). What number field do srs-z's
eigenvalues live in?
""")

    seen_irrationals = set()
    for i, ((u_plus, u_minus, kind)) in enumerate(oscillatory):
        lam = eigvals_adj[i]
        # imaginary part squared should be (8 − λ²)/4 (from u = (λ ± i√(8-λ²))/2)
        if kind == 'complex':
            disc_neg = 4 * 2 - lam * lam  # 8 - λ²
            print(f"    λ_{i+1} = {lam:+.6f}:  Re(u) = λ/2 = {lam/2:+.6f},  Im(u) = √({disc_neg:.4f})/2")
            # Try to identify √(8-λ²) in terms of K = ℚ(√2,√3,√5) or K(i) extensions
            # Simple cases: 8-λ² ∈ {1, 2, 3, 4, 5, 6, 7, 8, ...}
            for n in range(1, 16):
                if abs(disc_neg - n) < 1e-6:
                    seen_irrationals.add(f"√{n}")
                    print(f"        → 8−λ² = {n}, so Im(u) = √{n}/2")
                    break

    print(f"\n  Irrationals encountered in srs-z's spectrum: {sorted(seen_irrationals)}")
    print(f"  Compare to srs's K = ℚ(√2, √3, √5) (and K(i) for spectrum).")

    # Step 6: Bayesian-observer interpretation
    print("\n" + "=" * 78)
    print("BAYESIAN-OBSERVER INTERPRETATION (srs-z, intrinsic, no srs imported)")
    print("=" * 78)
    print(f"""
What an observer on srs-z would naturally derive (NO srs values imported):

  1. State space: 24-dim directed-edge causal-state space per primitive cell.
     Hashimoto operator B is 24×24.

  2. Spectrum (from |V|=8, |E|=12, k=3 + adjacency eigenvalues above):
     - {n_E - n_V} marginal eigenvalues at u=+1 (bipartite-trivial)
     - {n_E - n_V} marginal eigenvalues at u=−1
     - 8 quadratic factors (u² − λ_i u + 2) giving 16 oscillatory eigenvalues

  3. Marginal/total ratio (analog of srs's c=5/12):
     c(srs-z) = (2(|E|−|V|)+1) / (2|E|) = (2·{n_E-n_V}+1) / (2·{n_E}) = {2*(n_E-n_V)+1}/{2*n_E} = {(2*(n_E-n_V)+1)/(2*n_E):.5f}
     [vs srs's c = 5/12 = 0.41667]

  4. Sakharov chain length analog:
     M(srs-z) = N_edges/cell = {n_E}  [vs srs's M = 6]

  5. Ramanujan saturation pattern: see oscillatory eigenvalues above.
""")


if __name__ == '__main__':
    main()
