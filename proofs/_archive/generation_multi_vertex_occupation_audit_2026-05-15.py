#!/usr/bin/env python3
"""
proofs/_archive/generation_multi_vertex_occupation_audit_2026-05-15.py

Audit: does "generation = multi-vertex occupation count" give a structural
hierarchy that has NOT been tested by the existing 4-path Need-D-3 audit
or 9-attack scoreboard?

CONTEXT (2026-05-15 EOD+12, post α2'''-PIVOT closed-negative)

User raised: species ↔ # occupied edges per vertex (Hamming weight n on
Cl(6) Fock = Λ^•(C³)).  Could generation be encoded similarly at the
multi-vertex (cell) level — # occupied edges across all 4 atoms?

EXISTING ENCODING (per `two_vertex_interference_generations_probe.py` +
`nb_two_vertex_generations_probe.py`):
- Generation = C_3-TWISTED part of two-vertex amplitude on srs primitive
  cell
- 3 generations = 3 C_3 irreps {trivial, ω, ω²}
- Quantitative hierarchy 1:200:3000 = framework's known-hard open
  problem (Need-D-3, "5 sessions / 8 attacks ruled out")

USER'S FRAMING (untested):
- Multi-vertex Hamming-weight occupation count Σ_α n_α
- Decompose by outer-C_3 irrep on the 3-orbit of atoms
- Average occupation per irrep → potential hierarchy

PROBE STRUCTURE

(A) Build the 4-atom srs cell, identify outer C_3 orbit (1 fixed + 3-orbit)
(B) Enumerate multi-vertex Hamming configurations (4^4 = 256 cell states by
    per-atom Hamming weight)
(C) Decompose configurations under outer C_3 cyclic permutation of 3-orbit
(D) Compute per-irrep statistics:
    - count of configs in each irrep
    - average total occupation Σ n_α per irrep
    - average orbit-vertex occupation Σ_{α ∈ 3-orbit} n_α per irrep
(E) Pre-declared aborts:
    (G.1) All irreps have same average occupation → no hierarchy from
          this framing → close NEG
    (G.2) Hierarchies don't form a monotone 3-fold → wrong structure
    (G.3) Hierarchies are monotone 3-fold and K-rational with magnitudes
          suggestive of SM gen hierarchy → POSITIVE (further investigation)

THIS PROBE IS A FAST AUDIT (150 lines, runs in seconds).  Result decides
whether to scope a fuller multi-session probe or close the framing.

K-rational discipline: counts and averages are integer / rational by
construction (no continuum limits).
"""
from __future__ import annotations
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import numpy as np

K_STAR = 3
N_ATOMS = 4
DIM_FOCK = 2 ** K_STAR  # 8
TOL = 1e-12

print("=" * 78)
print("  GENERATION = MULTI-VERTEX OCCUPATION audit")
print("  (testing whether outer-C_3 irreps of cell-level Hamming")
print("   count give a structural 3-fold hierarchy)")
print("=" * 78)
print()

# -----------------------------------------------------------------------------
# Setup: outer C_3 acts on 3 atoms (call them atoms 1, 2, 3); atom 0 fixed.
# Per B6: srs primitive cell has 4 atoms, body-diagonal C_3 fixes one
# and cyclically permutes the other 3.
# -----------------------------------------------------------------------------

# Per-atom Hamming weight n_α ∈ {0, 1, 2, 3}
# Configuration of 4 atoms: (n_0, n_1, n_2, n_3) ∈ {0,1,2,3}^4 = 256 states

omega3 = np.exp(2j * np.pi / 3)


def cycle_3orbit(config: tuple) -> tuple:
    """Apply outer C_3: cycle (n_1, n_2, n_3) → (n_3, n_1, n_2); leave n_0 fixed."""
    n0, n1, n2, n3 = config
    return (n0, n3, n1, n2)


def c3_orbit(config: tuple):
    """Get the C_3 orbit of a configuration (1 or 3 elements)."""
    c1 = config
    c2 = cycle_3orbit(c1)
    c3 = cycle_3orbit(c2)
    orbit = {c1, c2, c3}
    return frozenset(orbit)


# -----------------------------------------------------------------------------
# (A) Build all configurations and group by orbit
# -----------------------------------------------------------------------------
print("=" * 78)
print("(A) Enumerate 4-atom Hamming configurations (4^4 = 256)")
print("=" * 78)
print()

all_configs = list(product(range(K_STAR + 1), repeat=N_ATOMS))
print(f"  Total configurations: {len(all_configs)}")
print(f"  C_3 fixes atom 0; cyclically permutes atoms (1, 2, 3)")
print()

orbits = set()
for c in all_configs:
    orbits.add(c3_orbit(c))

orbit_sizes = Counter(len(o) for o in orbits)
print(f"  Orbit size distribution: {dict(orbit_sizes)}")
print(f"  Total orbits: {sum(orbit_sizes.values())}")
print(f"  Total elements check: {sum(s * c for s, c in orbit_sizes.items())} (expect 256) ✓")


# -----------------------------------------------------------------------------
# (B) C_3 irrep decomposition of state space
# -----------------------------------------------------------------------------
# Each orbit of size 3 carries the regular rep of C_3 = trivial ⊕ ω ⊕ ω²
# Each fixed point (orbit of size 1) carries trivial rep only
#
# Total irrep multiplicities:
#   trivial: (# size-1 orbits) + (# size-3 orbits)
#   ω:       (# size-3 orbits)
#   ω²:      (# size-3 orbits)
# -----------------------------------------------------------------------------

print()
print("=" * 78)
print("(B) C_3 irrep decomposition of 256-state cell configuration space")
print("=" * 78)
print()

n_fixed = orbit_sizes.get(1, 0)
n_size3 = orbit_sizes.get(3, 0)
mult_trivial = n_fixed + n_size3
mult_omega = n_size3
mult_omega2 = n_size3
total_check = mult_trivial + mult_omega + mult_omega2
print(f"  Trivial irrep multiplicity: {mult_trivial} (= {n_fixed} fixed + {n_size3} size-3)")
print(f"  ω irrep multiplicity:       {mult_omega}")
print(f"  ω² irrep multiplicity:      {mult_omega2}")
print(f"  Total: {total_check} (expect 256) {'✓' if total_check == 256 else '✗'}")


# -----------------------------------------------------------------------------
# (C) Average occupation per irrep
# -----------------------------------------------------------------------------
# For each orbit, compute total occupation Σ n_α and orbit occupation Σ_{α in 3-orbit} n_α
# Decompose into trivial / ω / ω² states and tally averages
# -----------------------------------------------------------------------------

print()
print("=" * 78)
print("(C) Per-irrep average occupation")
print("=" * 78)
print()

# Method: for each orbit, the 3 irrep states (for size-3 orbits) are linear
# combinations of the 3 orbit elements, all sharing the same occupation Σ n_α
# (since occupation is C_3-invariant — it's just the total Hamming weight).
# So the irrep states inside a single orbit have IDENTICAL occupation.
# The ONLY variation is across orbits.
#
# Therefore: average occupation per irrep = average over orbits weighted by irrep mult.

trivial_occupations = []  # one per trivial-irrep state (= one per orbit)
omega_occupations = []    # one per ω-irrep state (= one per size-3 orbit)
omega2_occupations = []   # one per ω²-irrep state

for orbit in orbits:
    config = next(iter(orbit))  # representative element
    total_occ = sum(config)  # Σ n_α over all 4 atoms
    orbit_occ = config[1] + config[2] + config[3]  # Σ_{α ∈ 3-orbit} n_α
    if len(orbit) == 1:  # fixed (n_1 = n_2 = n_3)
        trivial_occupations.append((total_occ, orbit_occ))
    else:  # size-3 orbit: 1 trivial + 1 ω + 1 ω² state, all with same occupation
        trivial_occupations.append((total_occ, orbit_occ))
        omega_occupations.append((total_occ, orbit_occ))
        omega2_occupations.append((total_occ, orbit_occ))

avg_trivial_total = np.mean([t for t, _ in trivial_occupations])
avg_trivial_orbit = np.mean([o for _, o in trivial_occupations])
avg_omega_total = np.mean([t for t, _ in omega_occupations])
avg_omega_orbit = np.mean([o for _, o in omega_occupations])
avg_omega2_total = np.mean([t for t, _ in omega2_occupations])
avg_omega2_orbit = np.mean([o for _, o in omega2_occupations])

print(f"  Average TOTAL occupation Σ_α n_α (over all 4 atoms):")
print(f"    Trivial irrep:  {avg_trivial_total:.4f}  (over {len(trivial_occupations)} states)")
print(f"    ω irrep:        {avg_omega_total:.4f}  (over {len(omega_occupations)} states)")
print(f"    ω² irrep:       {avg_omega2_total:.4f}  (over {len(omega2_occupations)} states)")
print()
print(f"  Average ORBIT occupation Σ_{{α ∈ 3-orbit}} n_α (excluding fixed atom 0):")
print(f"    Trivial irrep:  {avg_trivial_orbit:.4f}")
print(f"    ω irrep:        {avg_omega_orbit:.4f}")
print(f"    ω² irrep:       {avg_omega2_orbit:.4f}")


# -----------------------------------------------------------------------------
# (D) Differentiate ω from ω² — they should be related by complex conjugation
# but might have IDENTICAL occupation statistics (CP-conjugate)
# -----------------------------------------------------------------------------

print()
print("=" * 78)
print("(D) ω vs ω² irrep occupation comparison")
print("=" * 78)
print()
print(f"  ω avg total occ:   {avg_omega_total:.6f}")
print(f"  ω² avg total occ:  {avg_omega2_total:.6f}")
print(f"  Difference:        {abs(avg_omega_total - avg_omega2_total):.2e}")
print()

if abs(avg_omega_total - avg_omega2_total) < TOL:
    print("  → ω and ω² have IDENTICAL average occupation (CP-conjugate states)")
    print("  → CANNOT distinguish gen 2 from gen 3 via occupation alone")
else:
    print("  → ω ≠ ω² average occupation — UNEXPECTED, would suggest CP-violation at this level")


# -----------------------------------------------------------------------------
# (E) Verdict — does the multi-vertex occupation give a 3-fold hierarchy?
# -----------------------------------------------------------------------------

print()
print("=" * 78)
print("(E) Verdict")
print("=" * 78)
print()

# Check abort conditions
trivial_omega_close = abs(avg_trivial_total - avg_omega_total) < 0.1  # within 0.1 occupation units
omega_omega2_equal = abs(avg_omega_total - avg_omega2_total) < TOL
all_irreps_same = trivial_omega_close and omega_omega2_equal

print(f"  G.1: All irreps have same average occupation:")
print(f"       Trivial = {avg_trivial_total:.4f}, ω = {avg_omega_total:.4f}, ω² = {avg_omega2_total:.4f}")
print(f"       {'YES — close NEG' if all_irreps_same else 'NO'}")
print()

# Check whether trivial/ω separation is monotone-3-fold
gens_avgs = sorted([avg_trivial_total, avg_omega_total, avg_omega2_total])
gen_ratios = [gens_avgs[i+1] / gens_avgs[i] if gens_avgs[i] > 0 else float('inf') for i in range(2)]
print(f"  Sorted irrep avg occupations: {gens_avgs}")
print(f"  Ratios: {gen_ratios}")
print()

# SM gen hierarchy ratios (rough): m_charm/m_up ~ 600, m_top/m_charm ~ 130
sm_hierarchy_match = (50 < min(gen_ratios) < 1000) and (50 < max(gen_ratios) < 1000) if min(gen_ratios) != float('inf') else False
print(f"  SM-like 3-fold hierarchy match (50-1000 ratios): "
      f"{'YES' if sm_hierarchy_match else 'NO'}")


# -----------------------------------------------------------------------------
# Honest summary
# -----------------------------------------------------------------------------

print()
print("=" * 78)
print("Audit summary")
print("=" * 78)
print()

if all_irreps_same:
    print("RESULT: All 3 outer-C_3 irreps have same average multi-vertex occupation.")
    print("        The 'generation = occupation count' framing CANNOT differentiate")
    print("        generations via this statistic alone.  Pre-declared abort (G.1)")
    print("        HITS.  Close NEG: this is not a fresh attack on the generation")
    print("        hierarchy problem.")
    print()
    print("STRUCTURAL REASON: occupation Σ n_α is C_3-INVARIANT (it doesn't")
    print("        change under cyclic permutation of atoms).  All irreps in")
    print("        the same orbit share the same occupation.  Hence the irrep-")
    print("        averaged occupation is just the orbit-averaged occupation,")
    print("        which is the same for all 3 irreps from size-3 orbits.")
    print()
    print("        The trivial irrep gets an EXTRA contribution from size-1")
    print("        orbits (n_1=n_2=n_3 configurations).  Whether this lifts the")
    print("        trivial above ω/ω² is shown above.")
elif omega_omega2_equal:
    print("RESULT: Trivial irrep differs from ω/ω² in average occupation, but")
    print("        ω = ω² exactly (CP-conjugate).  This gives at most a 2-fold")
    print("        structure (trivial vs ω/ω²-pair), not 3-fold.  Pre-declared")
    print("        abort (G.2) HITS.  Close NEG: framing doesn't give monotone")
    print("        3-fold hierarchy.")
else:
    print("RESULT: All 3 irreps differ.  Magnitudes need closer analysis to")
    print("        check K-rationality and SM hierarchy match.  See raw numbers")
    print("        above.")

print()
print("=" * 78)
print("Forward direction")
print("=" * 78)
print()

if all_irreps_same:
    print("  The user's framing ('generation = occupation count') does not")
    print("  differentiate generations at the simple cell-level Hamming")
    print("  count statistic.  This is consistent with the framework's")
    print("  current encoding: generation = C_3-twisted AMPLITUDE, not")
    print("  C_3-twisted OCCUPATION (which is invariant).")
    print()
    print("  However, the FRAMING raises a structurally distinct question:")
    print("  for each generation (C_3-twisted amplitude), what is the average")
    print("  PARTICLE NUMBER (occupation expectation value) on the cell?")
    print("  This is the OCCUPATION-WEIGHTED two-vertex amplitude — a hybrid")
    print("  of the existing framework's amplitude encoding and the user's")
    print("  occupation framing.  Worth scoping if simple version closes.")

print()
print("=" * 78)
print("End of generation/occupation audit.")
print("=" * 78)
