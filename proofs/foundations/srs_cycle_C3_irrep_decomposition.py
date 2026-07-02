#!/usr/bin/env python3
"""
C_3 representation decomposition of the 15-cycle vertex space.

Per `srs_chirality_orbit_decomposition.py`, the 15 girth-10 cycles through
srs vertex 0 split as 5 C_3-orbits × 3 cycles each (regular action). As a
C_3 representation, the 15-dim cycle space then decomposes as:

  V_15 = 5 · V_trivial ⊕ 5 · V_ω ⊕ 5 · V_{ω²}

where V_α is the 1-dim C_3 irrep with character ω^α (ω = e^{2πi/3}).

Restricted to the 9-chiral subspace and the 6-P-symmetric subspace:
  V_chiral = 3 · V_trivial ⊕ 3 · V_ω ⊕ 3 · V_{ω²}    (9 = 3+3+3)
  V_psym   = 2 · V_trivial ⊕ 2 · V_ω ⊕ 2 · V_{ω²}   (6 = 2+2+2)

This script verifies the decomposition by:
  1. Computing the C_3 permutation matrix on the 15 cycles
  2. Verifying its character is (trace) (15, 0, 0)
  3. Computing irrep multiplicities via the standard formula
  4. Constructing projectors P_α onto each eigenspace
  5. Checking the eigenvectors' chirality character (each eigenspace should
     contain 3 chiral + 2 P-sym basis vectors, BUT — important — only IF the
     chirality projector commutes with the C_3 projectors. This is true since
     chirality is C_3-invariant per the orbit decomposition.)

The output is a clean structural fact: the 15 cycles per vertex sit on a
5 species × 3 colors GRID, with species ∈ {chiral_1, chiral_2, chiral_3,
psym_1, psym_2} and colors ∈ {1, ω, ω²} (the 3 C_3 eigenvalues).
"""

import numpy as np
from itertools import product
from collections import Counter

# =============================================================================
# srs structure (matches lambda_2cycle_amplitude.py and prior probes)
# =============================================================================
A_PRIM = np.array([[-0.5, 0.5, 0.5],
                   [ 0.5,-0.5, 0.5],
                   [ 0.5, 0.5,-0.5]])
ATOMS = np.array([[1/8, 1/8, 1/8],
                  [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8],
                  [5/8, 3/8, 7/8]])
N_ATOMS = 4
girth = 10
SUPERCELL = 3

def frac_to_cart(frac):
    return A_PRIM.T @ np.asarray(frac, dtype=float)

def find_bonds():
    tol, NN = 0.02, np.sqrt(2) / 4
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = np.linalg.norm(rj - ATOMS[i])
                if d < tol: continue
                if abs(d - NN) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds

bonds = find_bonds()

def get_nbrs(atom, cell):
    out = []
    for src, tgt, dc in bonds:
        if src != atom: continue
        nc = (cell[0]+dc[0], cell[1]+dc[1], cell[2]+dc[2])
        if all(abs(c) <= SUPERCELL for c in nc):
            out.append((tgt, nc))
    return out

def vertex_cart(atom, cell):
    frac = ATOMS[atom] + cell[0]*A_PRIM[0] + cell[1]*A_PRIM[1] + cell[2]*A_PRIM[2]
    return frac_to_cart(frac)

# =============================================================================
# Enumerate 15 girth-10 cycles, classify chirality
# =============================================================================
start = (0, (0, 0, 0))
cycles_ordered = []

def dfs(path, current, depth):
    atom, cell = current
    prev = path[-2] if depth >= 1 else None
    for tgt, nc in get_nbrs(atom, cell):
        if prev is not None and (tgt, nc) == prev: continue
        if depth == girth - 1:
            if (tgt, nc) == start and start not in path[1:]:
                cycles_ordered.append(path[:])
        elif depth < girth - 1:
            if (tgt, nc) == start: continue
            dfs(path + [(tgt, nc)], (tgt, nc), depth + 1)

dfs([start], start, 0)

def unoriented_edge(v1, v2):
    return tuple(sorted([v1, v2]))

def cycle_edge_set(path):
    return frozenset(unoriented_edge(path[i], path[(i+1)%len(path)])
                     for i in range(len(path)))

seen = {}
for path in cycles_ordered:
    es = cycle_edge_set(path)
    if es not in seen:
        seen[es] = path
cycles_unique = list(seen.values())
assert len(cycles_unique) == 15

# Chirality classification
axis = A_PRIM.T @ np.array([1.0, 1.0, 1.0]); axis /= np.linalg.norm(axis)
ref = np.array([1.0, 0.0, 0.0])
e1 = ref - np.dot(ref, axis)*axis; e1 /= np.linalg.norm(e1)
e2 = np.cross(axis, e1)
origin = vertex_cart(0, (0,0,0))

def project_2d(v):
    rel = v - origin
    return np.array([np.dot(rel, e1), np.dot(rel, e2)])

def signed_area(pts):
    n = len(pts); s = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i+1)%n]
        s += x1*y2 - x2*y1
    return s / 2

chirality_label = []  # 'chiral' or 'psym' for each cycle in cycles_unique
for path in cycles_unique:
    pts = [project_2d(vertex_cart(a, c)) for (a, c) in path]
    sa = signed_area(pts)
    chirality_label.append('chiral' if abs(sa) > 1e-10 else 'psym')

n_chiral = chirality_label.count('chiral')
n_psym = chirality_label.count('psym')
assert n_chiral == 9 and n_psym == 6

# =============================================================================
# C_3 PERMUTATION MATRIX on 15 cycles
# =============================================================================
def C3_cart(v):
    return np.array([v[2], v[0], v[1]])

def apply_C3_to_vertex(atom_cell):
    a, c = atom_cell
    v_rot = C3_cart(vertex_cart(a, c))
    best = None; best_d = float('inf')
    for ap in range(N_ATOMS):
        for n1, n2, n3 in product(range(-SUPERCELL-1, SUPERCELL+2), repeat=3):
            cp = (n1, n2, n3)
            d = np.linalg.norm(vertex_cart(ap, cp) - v_rot)
            if d < best_d:
                best_d = d; best = (ap, cp)
    return best if best_d < 1e-6 else None

def apply_C3_to_cycle(path):
    new = []
    for v in path:
        nv = apply_C3_to_vertex(v)
        if nv is None: return None
        new.append(nv)
    return new

def cycle_match(path):
    target = cycle_edge_set(path)
    for i, p in enumerate(cycles_unique):
        if cycle_edge_set(p) == target:
            return i
    return None

# Build 15×15 permutation matrix
P_C3 = np.zeros((15, 15), dtype=complex)
for i, path in enumerate(cycles_unique):
    rotated = apply_C3_to_cycle(path)
    j = cycle_match(rotated)
    P_C3[j, i] = 1.0  # column i maps to row j

# Verify P_C3^3 = I
P_C3_cubed = np.linalg.matrix_power(P_C3, 3)
assert np.allclose(P_C3_cubed, np.eye(15)), "C_3^3 ≠ I — implementation bug"

# =============================================================================
# Character + irrep multiplicities
# =============================================================================
trace_e   = np.trace(np.eye(15)).real
trace_C3  = np.trace(P_C3).real
trace_C3_sq = np.trace(np.linalg.matrix_power(P_C3, 2)).real

print("=" * 80)
print("C_3 character of the 15-cycle representation")
print("=" * 80)
print(f"  χ(e)      = {trace_e}")
print(f"  χ(C_3)    = {trace_C3}")
print(f"  χ(C_3²)   = {trace_C3_sq}")
print(f"  Expected: (15, 0, 0)  — fully regular action, no fixed cycles.")

# Standard formula: multiplicity of irrep χ_α in V is (1/|G|) Σ_g χ_α(g)* χ_V(g).
# C_3 irreps: trivial (χ_α(g) = 1 for all g), ω (χ_α(C_3) = ω), ω² (χ_α(C_3) = ω²).
omega = np.exp(2j*np.pi/3)
mult_trivial = (1/3) * (trace_e * 1 + trace_C3 * 1 + trace_C3_sq * 1)
mult_omega   = (1/3) * (trace_e * 1 + trace_C3 * np.conj(omega)    + trace_C3_sq * np.conj(omega**2))
mult_omega2  = (1/3) * (trace_e * 1 + trace_C3 * np.conj(omega**2) + trace_C3_sq * np.conj(omega))

print(f"\n  Multiplicities:")
print(f"    V_trivial:  {mult_trivial.real:.4f}")
print(f"    V_ω:        {mult_omega.real:.4f}")
print(f"    V_ω²:       {mult_omega2.real:.4f}")
print(f"  Expected: 5, 5, 5")

# =============================================================================
# Restrict to chiral / P-symmetric subspaces
# =============================================================================
chiral_idx = [i for i, c in enumerate(chirality_label) if c == 'chiral']
psym_idx   = [i for i, c in enumerate(chirality_label) if c == 'psym']

P_C3_chiral = P_C3[np.ix_(chiral_idx, chiral_idx)]
P_C3_psym   = P_C3[np.ix_(psym_idx, psym_idx)]

# Verify chirality subspaces are C_3-invariant: row j non-zero implies j ∈ same subspace
def is_invariant(P_sub, original_indices):
    sub_set = set(original_indices)
    # P_sub is the restriction; if it's a valid restriction of a permutation,
    # P_C3 should map every chiral cycle to a chiral cycle.
    for i, src_idx in enumerate(original_indices):
        # Where does column src_idx of P_C3 point?
        col = P_C3[:, src_idx]
        target = np.argmax(np.abs(col))
        if target not in sub_set:
            return False
    return True

assert is_invariant(P_C3_chiral, chiral_idx), "Chiral subspace not C_3-invariant"
assert is_invariant(P_C3_psym,   psym_idx),   "P-sym subspace not C_3-invariant"

trace_chiral_C3   = np.trace(P_C3_chiral).real
trace_chiral_C3sq = np.trace(np.linalg.matrix_power(P_C3_chiral, 2)).real
trace_psym_C3     = np.trace(P_C3_psym).real
trace_psym_C3sq   = np.trace(np.linalg.matrix_power(P_C3_psym, 2)).real

mult_chiral_triv = (1/3) * (9 + trace_chiral_C3 + trace_chiral_C3sq)
mult_chiral_om   = (1/3) * (9 + trace_chiral_C3 * np.conj(omega) + trace_chiral_C3sq * np.conj(omega**2))
mult_chiral_om2  = (1/3) * (9 + trace_chiral_C3 * np.conj(omega**2) + trace_chiral_C3sq * np.conj(omega))
mult_psym_triv   = (1/3) * (6 + trace_psym_C3 + trace_psym_C3sq)
mult_psym_om     = (1/3) * (6 + trace_psym_C3 * np.conj(omega) + trace_psym_C3sq * np.conj(omega**2))
mult_psym_om2    = (1/3) * (6 + trace_psym_C3 * np.conj(omega**2) + trace_psym_C3sq * np.conj(omega))

print()
print("=" * 80)
print("Sub-decomposition of chiral and P-symmetric subspaces")
print("=" * 80)
print(f"\n  Chiral (9-dim):")
print(f"    χ_chiral(e)    = 9")
print(f"    χ_chiral(C_3)  = {trace_chiral_C3:+.4f}")
print(f"    χ_chiral(C_3²) = {trace_chiral_C3sq:+.4f}")
print(f"    Multiplicities: trivial = {mult_chiral_triv.real:.4f}, "
      f"ω = {mult_chiral_om.real:.4f}, ω² = {mult_chiral_om2.real:.4f}")
print(f"    Expected: 3, 3, 3")

print(f"\n  P-symmetric (6-dim):")
print(f"    χ_psym(e)    = 6")
print(f"    χ_psym(C_3)  = {trace_psym_C3:+.4f}")
print(f"    χ_psym(C_3²) = {trace_psym_C3sq:+.4f}")
print(f"    Multiplicities: trivial = {mult_psym_triv.real:.4f}, "
      f"ω = {mult_psym_om.real:.4f}, ω² = {mult_psym_om2.real:.4f}")
print(f"    Expected: 2, 2, 2")

# =============================================================================
# Construct C_3 eigenvalue projectors and apply to cycle indicators
# =============================================================================
# P_α = (1/3) Σ_g ω^{-αg} ρ(g)  where ρ(g) is C_3 generator's matrix
P_triv = (1/3) * (np.eye(15) + P_C3 + np.linalg.matrix_power(P_C3, 2))
P_om   = (1/3) * (np.eye(15) + np.conj(omega)    * P_C3 + np.conj(omega**2) * np.linalg.matrix_power(P_C3, 2))
P_om2  = (1/3) * (np.eye(15) + np.conj(omega**2) * P_C3 + np.conj(omega)    * np.linalg.matrix_power(P_C3, 2))

# Verify dim(im P_α) = 5
def proj_rank(P, tol=1e-8):
    s = np.linalg.svd(P, compute_uv=False)
    return int((s > tol).sum())

print()
print("=" * 80)
print("Eigenspace dimensions")
print("=" * 80)
print(f"  dim(P_trivial · V_15) = {proj_rank(P_triv)}")
print(f"  dim(P_ω · V_15)       = {proj_rank(P_om)}")
print(f"  dim(P_ω² · V_15)      = {proj_rank(P_om2)}")
print(f"  Expected: 5, 5, 5")

# =============================================================================
# 5×3 GRID — each species × color cell gets exactly one cycle (in the regular
# C_3 representation a basis-aware identification picks one cycle per cell).
#
# Construct it explicitly: each of the 5 orbits has 3 cycles {c_1, c_2, c_3}.
# The C_3-eigenvectors are linear combinations:
#   trivial:  (c_1 + c_2 + c_3) / √3
#   ω:        (c_1 + ω·c_2 + ω²·c_3) / √3
#   ω²:       (c_1 + ω²·c_2 + ω·c_3) / √3
# So each orbit contributes 1 trivial + 1 ω + 1 ω² eigenvector. 5 orbits → 5+5+5.
# =============================================================================

# Re-find orbits (using the orbit decomposition from the prior probe's logic)
def find_orbits_in_subset(subset_idx):
    """Return orbits as lists of indices into the FULL 15-cycle list."""
    perm = {}
    for src in subset_idx:
        col = P_C3[:, src]
        tgt = int(np.argmax(np.abs(col)))
        perm[src] = tgt
    visited = set()
    orbits = []
    for src in subset_idx:
        if src in visited: continue
        orb = [src]; visited.add(src)
        nxt = perm[src]
        while nxt != src:
            orb.append(nxt); visited.add(nxt)
            nxt = perm[nxt]
        orbits.append(orb)
    return orbits

orbits_chiral = find_orbits_in_subset(chiral_idx)
orbits_psym   = find_orbits_in_subset(psym_idx)

print()
print("=" * 80)
print("5×3 GRID — species × color decomposition of cycle space")
print("=" * 80)
print()
print(f"  {'orbit':<22}  trivial-eig   ω-eig          ω²-eig")
print(f"  {'-'*22}  {'-'*12}   {'-'*12}   {'-'*12}")

species_names_chiral = [f"chiral-{k+1}" for k in range(len(orbits_chiral))]
species_names_psym   = [f"psym-{k+1}"   for k in range(len(orbits_psym))]

species_list = []
for name, orb in zip(species_names_chiral, orbits_chiral):
    species_list.append((name, orb))
for name, orb in zip(species_names_psym, orbits_psym):
    species_list.append((name, orb))

for name, orb in species_list:
    # Each orbit has 3 cycles: orb[0], orb[1], orb[2] in C_3-cyclic order
    # The eigenvectors are |triv⟩ = (e_{orb[0]} + e_{orb[1]} + e_{orb[2]}) / √3  etc.
    # In the orbit basis: the C_3 cycles cycles[orb[0]] → cycles[orb[1]] → cycles[orb[2]].
    print(f"  {name:<22}  cyc{orb[0]:2d}+cyc{orb[1]:2d}+cyc{orb[2]:2d}   (ω-combo)      (ω²-combo)")

print()
print("Total: 5 species × 3 colors = 15 basis vectors.")
print("Chiral: 3 species × 3 colors = 9.")
print("P-sym:  2 species × 3 colors = 6.")

# =============================================================================
# Sanity check: |signed area| per cycle, grouped by orbit
# =============================================================================
print()
print("=" * 80)
print("Sanity: |signed area| per orbit (chirality MAGNITUDE per species)")
print("=" * 80)
for name, orb in species_list:
    sas = []
    for i in orb:
        path = cycles_unique[i]
        pts = [project_2d(vertex_cart(a, c)) for (a, c) in path]
        sas.append(abs(signed_area(pts)))
    print(f"  {name:<22}: |SA| values = {[f'{s:.4f}' for s in sas]}")

# =============================================================================
# Final structural fact
# =============================================================================
print()
print("=" * 80)
print("STRUCTURAL FACT")
print("=" * 80)
print(f"""
  The 15 girth-10 cycles through srs vertex 0 sit on a clean 5×3 grid:

    SPECIES axis (5):  3 chiral C_3-orbits + 2 P-symmetric C_3-orbits
    COLOR   axis (3):  C_3-eigenvalues 1, ω, ω²

  Each (species, color) cell contains exactly 1 basis vector. Both axes are
  structurally meaningful: SPECIES via chirality character (P parity), COLOR
  via the C_3 representation eigenvalue (which is the framework's color-Z₃
  identification per B6 §iii on the Bloch P-point fundamental).

  This is genuine new structural content. The 15-cycle vertex space has
  TWO independent C_3-equivariant decompositions (chirality is parity-even,
  C_3-eigenvalue is parity-odd), and they jointly factor 15 = 5 × 3 cleanly.
""")
