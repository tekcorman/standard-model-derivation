#!/usr/bin/env python3
"""
R-2 closure: fixed-point involution residue → vacuum |0⟩.

Hypothesis (per an internal working note):
the substrate vacuum |0⟩ is the residue of a discarded p ≥ 3 fixed-point
involution alternative at Layer 0. Test: does the p=3 fixed-point automorphism
group structurally correspond to the |0⟩ stabilizer on the framework's
Hilbert space? If yes, |0⟩ acquires a parallel derivation chain via the
residue (TRACED). If no, R-2 REFUTED.

Test plan from scoping note:
  Step 1: identify the next-most-MDL-favoured fixed-point involution (p=3).
  Step 2: compute the fixed-point automorphism (centralizer) group.
  Step 3: compute the stabilizer of |0⟩ on the substrate Hilbert space.
  Step 4: check coincidence.
  Step 5: cross-check via perturbations.

OUTCOME: REFUTED.

  Step 2 yields ℤ/2 (centralizer of (1 2) in S_3 = {id, (1 2)}).
  Step 3 yields ALL of F_inv(E) (every left-translation L_g preserves the
         constant function |0⟩).
  Step 4: ℤ/2 ≠ F_inv(E). The structural correspondence FAILS.

Furthermore, the framework's |0⟩ has an existing complete derivation as the
Bloch-trivial eigenstate of the adjacency operator A with eigenvalue 3 on
srs (forward_construction_substrate_thermal_apparatus.md §3.1). This
derivation is not improved by R-2's residue mechanism, which neither
reproduces |0⟩'s actual stabilizer nor predicts a structurally-distinct
new state at testable detail.

R-2 fails Mode 2 of the refutation framework: the proposed observable (|0⟩
ontology) is already explained upstream, and the residue mechanism does not
match the existing structure.
"""

import numpy as np
from itertools import permutations

# ============================================================================
# Step 1: Identify the next-most-MDL-favoured fixed-point involution
# ============================================================================
#
# Fixed-point involutions on p elements are permutations σ on {0, ..., p-1}
# satisfying σ² = id with at least one fixed point. The MDL-cheapest p with
# a non-trivial fixed-point involution is p = 3 (encoding cost log₂(3) ≈
# 1.585 bits per state).

print("="*75)
print("Step 1: Next-most-MDL-favoured fixed-point involution")
print("="*75)

p_values_with_fixed_point_involution = []
for p in range(2, 8):
    # σ² = id with ≥ 1 fixed point
    found = False
    for perm in permutations(range(p)):
        sigma = perm
        sigma_sq = tuple(sigma[sigma[i]] for i in range(p))
        if sigma_sq != tuple(range(p)):
            continue  # not σ² = id
        fixed_points = [i for i in range(p) if sigma[i] == i]
        if len(fixed_points) > 0 and len(fixed_points) < p:  # non-identity, non-fully-trivial
            found = True
            break
    if found:
        p_values_with_fixed_point_involution.append((p, sigma, fixed_points))

print(f"\np values admitting non-trivial fixed-point involutions: {[p for p,_,_ in p_values_with_fixed_point_involution]}")
print(f"Smallest p ≥ 2 with such an involution: p = {p_values_with_fixed_point_involution[0][0]}")

p_test, sigma_test, fps_test = p_values_with_fixed_point_involution[0]
print(f"  σ on {p_test} elements: {sigma_test}")
print(f"  Fixed points: {fps_test}")
print(f"  Encoding cost: log₂({p_test}) = {np.log2(p_test):.4f} bits/state")
print(f"  vs framework's p=2: log₂(2) = 1.0 bit/state")
print(f"  Soft-gating margin: {np.log2(p_test) - 1:.4f} bits/state")

# For p = 3, the canonical non-trivial fixed-point involution is σ = (1 2)
# fixing 0. (The other two — (0 2) fixing 1, and (0 1) fixing 2 — are conjugate.)


# ============================================================================
# Step 2: Centralizer of σ = (1 2) in S_3
# ============================================================================

print("\n" + "="*75)
print("Step 2: Fixed-point automorphism group (centralizer in S_3)")
print("="*75)

def perm_compose(p, q):
    """Composition p∘q (apply q first, then p), as tuple."""
    return tuple(p[q[i]] for i in range(len(p)))

# All elements of S_3
S3 = list(permutations(range(3)))
sigma = (0, 2, 1)  # σ = (1 2) fixing 0

centralizer = []
for g in S3:
    # g σ g^{-1} = σ ?
    g_inv = tuple(g.index(i) for i in range(len(g)))
    conjugate = perm_compose(perm_compose(g, sigma), g_inv)
    if conjugate == sigma:
        centralizer.append(g)

print(f"\nσ = (1 2), fixing element 0")
print(f"Centralizer of σ in S_3: {centralizer}")
print(f"  Size: {len(centralizer)}")
print(f"  Structure: ℤ/2 = ⟨σ⟩ = ⟨(1 2)⟩")
print(f"  → This is the 'fixed-point automorphism group' R-2 predicts for |0⟩'s stabilizer.")

centralizer_size = len(centralizer)


# ============================================================================
# Step 3: Stabilizer of |0⟩ on the framework's substrate Hilbert space
# ============================================================================

print("\n" + "="*75)
print("Step 3: Stabilizer of substrate vacuum |0⟩")
print("="*75)

print("""
Per `forward_construction_substrate_thermal_apparatus.md` §3.1:
  |0⟩_substrate = eigenspace of adjacency operator A at eigenvalue 3
                = trivial Bloch eigenstate at all k
                = constant function on F_inv(E)
                = (in ℓ²(F_inv(E))) the symmetric superposition over all g ∈ F_inv(E)

The stabilizer of |0⟩ under the substrate's symmetry group:

  Symmetry: left-regular representation L_g acting as L_g|h⟩ = |gh⟩.
  Action on constant function c·1: L_g(c·1)(h) = c·1(g^{-1}h) = c (constant)
  preserved.

  Therefore L_g · |0⟩ = |0⟩ for ALL g ∈ F_inv(E).

  Stabilizer(|0⟩) = F_inv(E) (the full group).

For |E| = 6, F_inv(E) is the free product of 6 copies of ℤ/2 — an infinite
non-amenable discrete group. NOT isomorphic to ℤ/2 (the centralizer from
Step 2).
""")

# Numerical confirmation: build a small piece of L²(F_inv(E)) and verify
# the constant function is preserved by all generator translations.
print("Numerical confirmation (small Cayley graph):")

E = 6
# Enumerate F_inv(6) words up to length 2
words = [()]
for _ in range(2):
    new_words = []
    for w in words:
        for g in range(E):
            if not w or w[0] != g:  # involutivity reduction
                new_words.append((g,) + w)
    words = words + new_words
# Deduplicate
words = list(set(words))
n_words = len(words)
word_idx = {w: i for i, w in enumerate(words)}

# Constant function on this finite slice
psi_0 = np.ones(n_words) / np.sqrt(n_words)

# Apply each generator's left translation L_g (with reduction): check L_g psi_0 = psi_0?
all_preserve = True
for g in range(E):
    psi_translated = np.zeros(n_words)
    for w in words:
        # L_g(|w⟩) = |gw⟩ (with involutivity reduction if g·w starts with g)
        if w and w[0] == g:
            new_w = w[1:]  # cancellation
        else:
            new_w = (g,) + w
        if new_w in word_idx:
            psi_translated[word_idx[new_w]] = psi_0[word_idx[w]]
    # Compare (modulo boundary effects from finite slice)
    deviation = np.linalg.norm(psi_translated - psi_0)
    print(f"  L_{g} on constant function: |L_g·ψ - ψ| = {deviation:.4f} (boundary-affected)")

# (Boundary effects are real: in a finite-depth slice, generators move some
# words off the slice. The infinite-volume claim L_g·|0⟩ = |0⟩ holds in the
# full L²(F_inv(E)), not the finite truncation.)

print("""
The boundary-driven deviations above are numerical artifacts of truncating
to a finite Cayley graph slice. In the full L²(F_inv(E)), the symmetric
superposition is exactly preserved: ⟨w|L_g|0⟩ = ⟨g^{-1}w|0⟩ = c (constant)
for all w. The full stabilizer IS F_inv(E).
""")


# ============================================================================
# Step 4: Check coincidence
# ============================================================================

print("="*75)
print("Step 4: Structural correspondence check")
print("="*75)

print(f"""
Step 2 yields: centralizer ℤ/2 (size {centralizer_size}).
Step 3 yields: stabilizer F_inv(6) (infinite, non-amenable, ≠ ℤ/2).

ℤ/2 vs F_inv(6): different groups, different structure. F_inv(6) contains
many ℤ/2 subgroups (one per generator: ⟨e_i⟩ for i = 0, ..., 5), but no
canonical ONE that uniquely embeds the centralizer.

The structural correspondence test FAILS. R-2's predicted stabilizer (ℤ/2)
does NOT match the framework's |0⟩ stabilizer (F_inv(6)).
""")


# ============================================================================
# Closure
# ============================================================================

print("="*75)
print("R-2 CLOSURE — REFUTED")
print("="*75)

print("""
R-2 closes as REFUTED via Mode 2 (proposed observable already explained
upstream, AND the residue mechanism does not match it):

1. The framework's |0⟩ has an existing complete derivation as the Bloch-
   trivial eigenstate of the adjacency operator A with eigenvalue 3 on srs
   (forward_construction_substrate_thermal_apparatus.md §3.1; line 130).

2. |0⟩'s stabilizer under the substrate's symmetry group (left-regular
   representation of F_inv(E)) is the FULL F_inv(E) — the constant function
   is preserved by every left translation.

3. R-2's residue mechanism predicts a stabilizer of ℤ/2 (the centralizer
   of a p=3 fixed-point involution in S_3). Mismatch.

4. Therefore R-2's residue mechanism does not produce the framework's |0⟩.
   It would predict a DIFFERENT state (one whose stabilizer is ℤ/2 — a state
   symmetric under one specific edge involution but not under others).
   Such a state has no obvious physical interpretation in the framework and
   no observable correspondence.

5. The "partially trivial" exclusion of fixed-point involutions in
   p_toggle_derivation.md is structurally incompatible with the framework's
   substrate dynamics requirement: the fixed-point state would be a "dead"
   toggle target, never updated by its own toggle. The framework's substrate
   has all states equally toggle-active (uniform Markov chain stationary
   distribution per Stage 2c), which is consistent only with fixed-point-free
   involutions.

6. Net: the residue mechanism is structurally divergent from the framework's
   |0⟩, and the framework's |0⟩ is already fully derived. R-2 adds no
   testable content.

REGISTER STATE AFTER R-2 CLOSURE:

  R-1 (higher arity): OPEN, low priority
  R-2 (fixed-point → |0⟩): REFUTED (mode 2)
  R-3 (relations → cycles): REFUTED
  R-4 (d=4 → time): REFUTED
  R-5 (d≥5): REFUTED
  R-6 (ℍ → SU(2)_L): REFUTED
  R-7 (ths CKM): REFUTED
  R-8 (dia): REFUTED
  R-9 (full-MDL): RESTRICTED to chiral nets
  R-10 (finite-graph UV): OPEN, low priority
  R-11 (alphabet localization): OPEN, high priority (absorbs Cluster 1)
  R-12 (chirality): ACCOUNTED-FOR + STRUCTURAL FILTER

SEVEN REFUTED. R-11 is now the only high-priority OPEN, and it does not
predict new physics — it is a closure of Cluster 1.

The methodology is well-calibrated in the eliminative direction (7 REFUTED,
each with a precise mechanism: 6 via Mode 2 / observable-already-explained,
2 via Mode 1 / hard-gated, R-7 via R-12 chirality structural filter — Mode 1
plus); but has produced no positive (TRACED) closures. The residue register
appears to be primarily a *negative-discovery* tool — surfacing where soft-
gated alternatives are actually hard-gated, redundant with upstream, or
structurally divergent from the framework's existing observables.

This is itself a useful result: the framework's existing structure is
TIGHTLY CONSTRAINED. There are no "low-hanging" residue-derived observables.
Any additional physics from soft-gated alternatives must come from:
  (i) closure of R-11 (alphabet localization, the substrate-to-srs bridge);
  (ii) research-level NCG developments (Lorentzian signature scoping note);
  (iii) directions outside the current residue-register framing.
""")
