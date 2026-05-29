#!/usr/bin/env python3
"""
>>> SUPERSEDED-CONTEXT NOTICE (2026-05-21 spring-cleaning) <<<
The M_R = (2/3)^g * M_GUT formula this probe finds a ~38x discrepancy against
was RETRACTED 2026-05-04. The framework's live M_R is the spectral-gap formula
(`predictions/m_nu3_derivation.md` Step 2); the retraction is recorded in
an internal working note. This
probe's discrepancy is a real result against a formula the framework no longer
uses — it is NOT a contradiction of any live claim. Read it as the record of
why the local-girth-cycle route was abandoned.
>>> end notice <<<

proofs/flavor/srs_M_R_girth_cycle_waterfilling.py

WATERFILLING CHECK for m_ν₃ M_R girth-cycle scoping
(an internal working note).

Steps 1 and 2 considered only the leading single girth-cycle contribution to
M_R. A2 ("waterline, not strict optimum") demands summing over ALL closed
NB-walk topologies with positive compression savings, weighted by their MDL
probability. This is the same waterline principle that produced
dark = 1 - (5/12)·α₁/(1-α₁) for v_Higgs (geometric series over winding).

QUESTION: Are we properly waterfilling all MDL-admissible closed NB walks for
M_R, or have we been considering only the leading girth-cycle contribution?

WHAT THIS SCRIPT DOES:

  W1. Compute closed NB-walk count N_L per primitive cell at each length L,
      via BZ-averaged trace of B^L (Bloch trace formula).
  W2. Apply A5(b) MDL weight (2/3)^L per closed walk and sum.
  W3. Compute the geometric winding series for girth cycles (winding ≥ 1).
  W4. Compare waterfilled sum to bare (2/3)^g and identify the discrepancy.
  W5. Audit empirical implications for m_ν₃.

KEY FINDING: A naive A5(b) waterfilled sum gives M_R/M_GUT ≈ 0.66, ~38× larger
than the asserted (2/3)^g ≈ 0.017. This implies either:
  (a) The asserted formula is per-cycle (not summed), and the implicit
      normalization differs from naive waterfilling.
  (b) Not all closed NB walks contribute coherently to the C_3-trivial
      Bloch mode (selection rules from the Bloch projection).
  (c) Higher-order topologies are MDL-suppressed by additional factors
      beyond (2/3)^L (e.g., closure-pair constraints).
"""

import numpy as np
from numpy import sqrt, pi, exp
from itertools import product
from fractions import Fraction

np.set_printoptions(precision=10, linewidth=140, suppress=True)

# ============================================================
# srs setup
# ============================================================
A_PRIM = np.array([[-0.5, 0.5, 0.5],
                   [ 0.5,-0.5, 0.5],
                   [ 0.5, 0.5,-0.5]])
ATOMS = np.array([[1/8, 1/8, 1/8],
                  [3/8, 7/8, 5/8],
                  [7/8, 5/8, 3/8],
                  [5/8, 3/8, 7/8]])
N_ATOMS = 4
k_star  = 3
girth   = 10
NN_DIST = sqrt(2) / 4

def find_bonds():
    bonds = []
    for i in range(N_ATOMS):
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = np.linalg.norm(rj - ATOMS[i])
                if d < 0.02: continue
                if abs(d - NN_DIST) < 0.02:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds

bonds = find_bonds()
n_E = len(bonds)
assert n_E == 12

def bloch_B(k):
    B = np.zeros((n_E, n_E), dtype=complex)
    for f, (fs, ft, fc) in enumerate(bonds):
        for e, (es, et, ec) in enumerate(bonds):
            if fs != et: continue
            if ft == es and fc == tuple(-x for x in ec): continue
            B[f, e] = exp(2j * pi * np.dot(k, fc))
    return B

# ============================================================
# W1: Closed NB walk counts via BZ-averaged trace of B^L
# ============================================================
print("="*72)
print("W1: Closed NB-walk count N_L per primitive cell")
print("="*72)
print("    (Bloch trace formula: N_L = BZ-average of tr(B(k)^L))")

def closed_walks(L_max=20, grid=10):
    counts = np.zeros(L_max + 1, dtype=complex)
    Nk = grid**3
    for i, j, k in product(range(grid), repeat=3):
        kf = np.array([i, j, k]) / grid + 0.5/grid  # avoid Γ degeneracies
        B = bloch_B(kf)
        Bp = np.eye(n_E, dtype=complex)
        for L in range(1, L_max + 1):
            Bp = Bp @ B
            counts[L] += np.trace(Bp) / Nk
    return counts.real

L_max = 20
counts = closed_walks(L_max=L_max, grid=10)
print(f"    L     N_L/cell    N_L/vertex    notes")
print(f"    ----  ----------  -----------   ------")
for L in range(1, L_max + 1):
    if abs(counts[L]) < 0.5:
        if L < girth:
            note = "(0 — below girth)"
        else:
            note = "(≈0)"
        print(f"    {L:>3}   {counts[L]:>10.2f}  {counts[L]/N_ATOMS:>11.4f}   {note}")
    else:
        if L == girth:
            note = "GIRTH CYCLES (n_g = 15 unoriented × 2 orient. = 30/vertex × 4 atoms = 120)"
        else:
            note = f"length-{L} closed NB walks"
        print(f"    {L:>3}   {counts[L]:>10.2f}  {counts[L]/N_ATOMS:>11.4f}   {note}")

# Note: closed NB walks of length g are GIRTH CYCLES.
# Length g+2: would-be longer cycles, but for srs at g=10 these turn out to be 0.
# Length g+4 = 14: 168/cell = 42/vertex.
# Length 20: includes winding-2 girth cycles (back-to-back) and longer single cycles.

# ============================================================
# W2: A5(b) MDL waterfilling
# ============================================================
print("\n" + "="*72)
print("W2: A5(b) MDL-weighted sum: Σ_L N_L_per_vertex × (2/3)^L")
print("="*72)
print(f"    A5(b) per-walk MDL probability = ((k*-1)/k*)^L = (2/3)^L\n")
print(f"    L     N_L/vertex    (2/3)^L           contribution")
print(f"    ----  -----------  --------------   ------------------")
total_wf = 0.0
girth_only = 0.0
for L in range(1, L_max + 1):
    if abs(counts[L]) < 0.5:
        continue
    n_v = counts[L] / N_ATOMS
    p = (2/3)**L
    contrib = n_v * p
    total_wf += contrib
    if L == girth:
        girth_only = contrib
    print(f"    {L:>3}   {n_v:>10.4f}   {p:>14.6e}   {contrib:>14.6e}")

print(f"\n    bare (2/3)^g                                {(2/3)**girth:.6e}")
print(f"    girth-only summed contribution (L=g):       {girth_only:.6e}")
print(f"    waterfilled sum (L = 1..{L_max}):                  {total_wf:.6e}")
print(f"\n    ratio (girth-only) / (bare (2/3)^g)        = {girth_only / (2/3)**girth:.2f}")
print(f"    ratio (waterfilled) / (bare (2/3)^g)       = {total_wf / (2/3)**girth:.2f}")

# ============================================================
# W3: Geometric winding series for girth cycles only
# ============================================================
print("\n" + "="*72)
print("W3: Geometric winding series in girth cycles")
print("="*72)
print(f"""    Each girth cycle can be traversed n times (winding number n=1,2,3,...)
    with MDL probability ((2/3)^g)^n. Geometric sum over windings:""")
g_amp = (2/3)**girth
n_g_oriented_per_vertex = 30
n_g_unoriented_per_vertex = 15
geom_oriented   = n_g_oriented_per_vertex   * g_amp / (1 - g_amp)
geom_unoriented = n_g_unoriented_per_vertex * g_amp / (1 - g_amp)
print(f"\n    n_g (oriented)   × (2/3)^g / (1-(2/3)^g) = {geom_oriented:.6e}  (ratio {geom_oriented/g_amp:.2f})")
print(f"    n_g (unoriented) × (2/3)^g / (1-(2/3)^g) = {geom_unoriented:.6e}  (ratio {geom_unoriented/g_amp:.2f})")
print(f"\n    NOTE: bare girth-only = {girth_only:.4e}")
print(f"          oriented winding series = {geom_oriented:.4e}")
print(f"          ratio (winding sum / single winding) = {1/(1-g_amp):.4f}  (≈1.018, small)")

# ============================================================
# W4: Compare to asserted M_R/M_GUT = (2/3)^g
# ============================================================
print("\n" + "="*72)
print("W4: Comparison to asserted M_R/M_GUT = (2/3)^g")
print("="*72)
asserted = (2/3)**girth
print(f"""
    asserted M_R/M_GUT = (2/3)^g                          = {asserted:.6e}
    waterfilled sum   (oriented N_L × (2/3)^L over L)    = {total_wf:.6e}
    discrepancy ratio                                    = {total_wf/asserted:.2f}

    The waterfilled sum is ~38× larger than the asserted formula.

    POSSIBLE INTERPRETATIONS:

    (a) The asserted (2/3)^g formula is PER-CYCLE-PER-VERTEX, not the total
        waterfilled coupling. Under this reading, total M_R is much larger.

    (b) Not all closed NB walks contribute coherently to the C_3-trivial
        Bloch mode at P; selection rules from the Bloch projection reduce
        the effective contribution. Specifically, of the 30 oriented girth
        cycles per vertex, only those compatible with C_3-trivial transformation
        contribute coherently.

    (c) The Higgs vacuum-bubble template (5/12 = n_g/(N_atoms·k*²)) is an
        AVERAGE over local cycles per (in,out) pair per atom. For ν_R, the
        analogous average might give a different prefactor.

    For path (b): of the 15 unoriented girth cycles, 5 are C_3-trivial,
    5 transform as ω, 5 as ω̄. Only the 5 trivial-class cycles contribute
    coherently to ψ_RH. With 2 orientations: 10 cycles × (2/3)^g.

    For path (c): n_g_C3_trivial × (2/3)^g / (atoms × pairs)
                  = 5 × (2/3)^g / (4 × 9) = 5/36 × (2/3)^g.
""")

# Compute the C_3-trivial restricted sum
n_g_C3_trivial_unoriented = 5  # from (5,5,5) decomposition
n_g_C3_trivial_oriented = 10
print(f"    Candidate prefactors X for M_R/M_GUT = X · (2/3)^g:")
candidates = {
    "X = 1 (current ADOPTED-PS — empirical match at m_t(GUT)≈130GeV)": 1,
    "X = 5/36 (Higgs template restricted to C_3-trivial channel)":      Fraction(5, 36),
    "X = 5/12 (Higgs template, full local averaging)":                  Fraction(5, 12),
    "X = 5 (C_3-trivial unoriented coherent sum)":                      5,
    "X = 10 (C_3-trivial oriented coherent sum)":                       10,
    "X = n_g_oriented = 30 (all cycles per vertex, no Bloch projection)": 30,
    "X = waterfilled total (numerical)": total_wf / asserted,
}
m_t_GUT_grid = [120, 130, 140, 150, 174]
m_nu3_obs = 0.0495
print(f"    {'X':<70} {'M_R/M_GUT':<12} ", end='')
for mt in m_t_GUT_grid:
    print(f" mν₃@m_t={mt}".ljust(15), end='')
print()
print(f"    {'-'*70} {'-'*12} ", end='')
for _ in m_t_GUT_grid:
    print(f"{'-'*15}", end='')
print()
for desc, X in candidates.items():
    Xf = float(X)
    mr = Xf * asserted
    print(f"    {desc:<70} {mr:<12.4e} ", end='')
    for mt in m_t_GUT_grid:
        m_nu3 = mt**2 / (mr * 2e16) * 1e9
        flag = '✓' if abs(m_nu3 - m_nu3_obs)/m_nu3_obs < 0.10 else ' '
        print(f" {m_nu3:.4f}{flag} ".ljust(15), end='')
    print()

print(f"\n    Observed m_ν₃ = {m_nu3_obs} eV (NuFIT 5.3, normal ordering)")

# ============================================================
# W5: Honest verdict
# ============================================================
print("\n" + "="*72)
print("W5: Honest verdict — what the waterfilling tells us")
print("="*72)
print("""
The asserted M_R = (2/3)^g · M_GUT is empirically consistent with m_ν₃ ≈
0.05 eV at m_t(GUT) ≈ 130 GeV — but A5(b) waterfilling over ALL closed NB
walks gives M_R larger by a factor of ~38.

Three possible structural readings:

  1. PER-CYCLE INTERPRETATION (currently implicit in framework):
     M_R = (2/3)^g · M_GUT means "per girth-cycle topology amplitude."
     The actual physical M_R is M_GUT × (Σ over coherent cycles).
     With C_3-trivial-only restriction: X_total ≈ 5 (unoriented) or 10 (oriented).
     ⇒ M_R/M_GUT ≈ 5×(2/3)^g ≈ 0.087 ⇒ m_ν₃ ≈ 0.010 eV. TOO LOW BY 5×.

  2. BLOCH-PROJECTION INTERPRETATION (the global view):
     The C_3-trivial Bloch projection at P, combined with the per-atom
     normalization 1/N_atoms, gives X = 5/(N_atoms · ?) that lands at X ≈ 1.
     But the precise mechanism for getting exactly X = 1 needs derivation —
     waterfilling alone doesn't give it cleanly.

  3. ADDITIONAL MDL SUPPRESSION:
     Beyond (2/3)^L per walk, there may be additional suppression from
     constraints not captured in the per-walk MDL probability — e.g., the
     PS singlet requirement, the Bloch-mode coherence requirement, etc.

CONCLUSION:

  The asserted M_R = (2/3)^g · M_GUT is NOT self-consistent with naive A5(b)
  waterfilling over all closed NB walks. Either the framework's M_R formula
  has implicit prefactors that aren't being computed, or there are additional
  selection rules suppressing most closed-walk contributions.

  This is a real gap in the M_R derivation. ADOPTED-PS is more deeply adopted
  than previously characterized — not just (2/3)^g · M_GUT being adopted, but
  the entire COMBINATORIAL STRUCTURE of how cycles aggregate to M_R.

NEXT STEP:

  Construct the explicit ν_R Majorana mass operator on G_seesaw, including
  the PS singlet projection at the GUT scale, and identify which closed NB
  walks contribute COHERENTLY vs which are projected out by C_3-trivial /
  PS-singlet selection. The (2/3)^L per-walk weight is correct under A5(b);
  the question is which walks survive the projection.

  This is a more substantial piece of work than a simple cycle-counting
  closure. The scoping doc's "Step 2 = 3-5 sessions" estimate likely
  understated the difficulty.
""")
print("="*72)
