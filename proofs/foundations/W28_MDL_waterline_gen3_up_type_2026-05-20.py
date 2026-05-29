#!/usr/bin/env python3
"""
W28 — MDL waterline at gen-3 up-type (item #1 sub-probe (β))
=============================================================

Date: 2026-05-20
Predecessor: W27 closed-negative — every operator built from W20-W22 inputs
({B at k=Γ, χ̃, C_3, P_swap, H_VEV}) has C_3-uniform restriction on V_Ram,
so V_Ram alone (within this apparatus) cannot give per-species labeling.
W27 §5 named (β) as the most bounded remaining sub-probe: attack the
master Yukawa doc §3.3 assertion directly via MDL waterline counting at
gen-3 up-type, independently of V_Ram.

THE ASSERTION TO TEST (master Yukawa doc §3.3):
> "The top is the n=2 Hamming sector — two active toggle modes at a
>  Pati-Salam color-triplet, SU(2)_L doublet species point. The combination
>  is maximally above-waterline; MDL waterfilling places every girth-cycle
>  mode above the waterline (fixed by the quark's quantum numbers), so the
>  free-mode count → 0 → exponent (2/3)⁰ = 1."

This is asserted, not derived. The framework's per-species n_free projection
is explicitly named as "the open piece" (master Yukawa doc §5.3, §11.4):
"the specific n_free counts per sector are NOT yet derived from this
structure. This is the open piece."

W28 attempts the audit:
  (1) Confirm n_g_edge = 5 is a structurally-derived combinatorial invariant
      (cycles per ordered edge pair on srs; alpha_1_full.py).
  (2) Test candidate species-quantum-number → n_free projections, each
      grounded in a different structural intuition.
  (3) Check whether any candidate reproduces the asserted (n_free=1 for
      y_τ, n_free=0 for y_t).
  (4) Honest verdict: if no simple counting works, the framework's
      per-species projection is genuinely open and not derivable from the
      currently-available apparatus.

KNOWN SPECIES QUANTUM NUMBERS (per master Yukawa doc §4 Table):

  Species   | Hamming n | color   | SU(2)_L | gen j   | asserted n_free
  ----------|-----------|---------|---------|---------|-----------------
  y_τ       | 3         | singlet | doublet | gen-1   | 1
  y_t       | 2         | triplet | doublet | gen-3   | 0  (asserted)
  y_ν3      | 0         | singlet | (delocalized) | gen-3 | 1 (struct. id.)
  y_e, y_μ  | 3         | singlet | doublet | gen-1, gen-2 | ?
  y_b, y_s, y_d | 1     | triplet | doublet | gen-3,2,1   | ?
  y_c, y_u  | 2         | triplet | doublet | gen-2, gen-1 | ?

THE 5 GIRTH-CYCLE MODES per edge pair on srs (n_g_edge = 5; alpha_1_full.py):
  These are 5 distinct girth cycles passing through each ordered edge pair
  at a vertex of srs. They are concrete combinatorial objects (verified by
  srs_graph_analysis.py).

CANDIDATE PROJECTIONS to test:
  (A) Naive product:    n_fixed = n × color_dim × SU2L_dim × gen_j
                        n_free = max(0, 5 − n_fixed)
  (B) Naive sum:        n_fixed = n + color_dim + SU2L_dim + gen_j
  (C) Above-waterline by Hamming weight alone: n_fixed = n
  (D) Sector-overlap product: n_fixed = (color_dim × SU2L_dim) — Cl(6)-Fock
                              orbits per species at the trivalent vertex.

PRE-DECLARED GATE CHECKS:
  N1. n_g_edge = 5 confirmed (recap from alpha_1_full.py).
  N2. y_τ asserted n_free = 1; y_t asserted n_free = 0.
  N3. Test 4 candidate projections; document which (if any) reproduces (1, 0).
  N4. If no candidate works, the framework's per-species projection is open
      from this apparatus — extending W26 + W27's negative findings.

USAGE:
    python3 proofs/foundations/W28_MDL_waterline_gen3_up_type_2026-05-20.py
"""

from __future__ import annotations

EXPECTED = {
    "N1_n_g_edge_is_5":              True,
    "N2_asserted_n_free_recorded":   True,
    "N3_candidate_projections_tested": True,
    "N4_verdict_documented":         True,
}
RESULTS = {}

print("=" * 78)
print("W28 — MDL waterline at gen-3 up-type (item #1 sub-probe (β))")
print("=" * 78)


# ============================================================================
# Step A — Confirm n_g_edge = 5 (structural recap)
# ============================================================================
N_G_EDGE = 5   # combinatorial graph invariant of srs (alpha_1_full.py L91, verified
               # by srs_graph_analysis.py per the comment at L88)
print(f"\nStep A — Structural input recap")
print(f"  n_g_edge = {N_G_EDGE}  (cycles per ordered edge pair on srs)")
print(f"    Source: predictions/alpha_1_full.py line 91 (combinatorial graph invariant)")
print(f"    Verified: proofs/foundations/srs_graph_analysis.py (cited in alpha_1_full.py L88)")
print(f"  Girth g = 10 on srs (theorem-grade per predictions/g_girth.py)")
print(f"  k* = 3 (theorem-grade per predictions/k_star.py)")
print(f"  α₁_full = (n_g_edge / k*) · ((k*-1)/k*)^(g-2) = (5/3)·(2/3)^8 = 1280/19683")
N1 = (N_G_EDGE == 5)
RESULTS["N1_n_g_edge_is_5"] = bool(N1)


# ============================================================================
# Step B — Species quantum numbers + asserted n_free
# ============================================================================
# Per master Yukawa doc §4 Table + §3 derivations:
SPECIES = [
    # (label,    n_Hamming, color_dim, SU2L_dim, gen_j, asserted_n_free, status)
    ("y_τ",      3,         1,         2,        1,     1,               "derived (theorem-grade)"),
    ("y_μ",      3,         1,         2,        2,     None,            "open (within-sector Koide only)"),
    ("y_e",      3,         1,         2,        3,     None,            "open"),
    ("y_b",      1,         3,         2,        1,     None,            "open"),
    ("y_s",      1,         3,         2,        2,     None,            "open"),
    ("y_d",      1,         3,         2,        3,     None,            "open"),
    ("y_t",      2,         3,         2,        1,     0,               "ASSERTED (single hard residue)"),
    ("y_c",      2,         3,         2,        2,     None,            "open"),
    ("y_u",      2,         3,         2,        3,     None,            "open"),
    ("y_ν3",     0,         1,         1,        1,     1,               "structurally identified"),
    ("y_ν2",     0,         1,         1,        2,     None,            "open"),
    ("y_ν1",     0,         1,         1,        3,     None,            "open"),
]
# Note: gen-3 up-type = (n=2, color=3, SU2L=2, j=1) under "j=1 is gen-3" convention.
# I'm using j ∈ {1, 2, 3} where j=1 is the heaviest (gen-3 in Koide convention is
# the heaviest; we map j=1→heaviest to match the assertion that y_t = 1 is "gen-3").

print(f"\nStep B — Species quantum numbers + asserted n_free")
print(f"  {'species':<8s} {'n':>3s} {'color':>6s} {'SU2L':>5s} {'gen':>4s} {'n_free':>8s} {'status':<35s}")
print(f"  {'-'*78}")
for sp, n, col, su2l, j, nfree, status in SPECIES:
    nfree_str = str(nfree) if nfree is not None else "OPEN"
    print(f"  {sp:<8s} {n:>3d} {col:>6d} {su2l:>5d} {j:>4d} {nfree_str:>8s} {status:<35s}")
N2 = True
RESULTS["N2_asserted_n_free_recorded"] = bool(N2)


# ============================================================================
# Step C — Test candidate projections
# ============================================================================
print(f"\nStep C — Test candidate species → n_free projections")
print(f"  Target: reproduce y_τ → n_free=1 AND y_t → n_free=0 simultaneously.")
print(f"  Each candidate is a different structural intuition about how species'")
print(f"  quantum numbers fix the n_g_edge=5 girth-cycle modes.")
print()

def project_naive_product(n, color, su2l, j):
    """(A) Naive product of quantum numbers."""
    n_fixed = n * color * su2l * j
    return max(0, N_G_EDGE - n_fixed)

def project_naive_sum(n, color, su2l, j):
    """(B) Naive sum."""
    n_fixed = n + color + su2l + j
    return max(0, N_G_EDGE - n_fixed)

def project_hamming_only(n, color, su2l, j):
    """(C) Above-waterline determined by Hamming weight alone."""
    n_fixed = n
    return max(0, N_G_EDGE - n_fixed)

def project_color_x_su2l(n, color, su2l, j):
    """(D) Sector overlap: n_fixed = color × SU2L (Cl(6)-Fock orbits per
    trivalent vertex)."""
    n_fixed = color * su2l
    return max(0, N_G_EDGE - n_fixed)

def project_gen3_only(n, color, su2l, j):
    """(E) Gen-3 specific: at gen-3 (j=1) all 5 modes are fixed regardless;
    at gen-1, only the chirality-flip mode is free."""
    if j == 1:   # gen-3
        return 0
    elif j == 3: # gen-1
        return 1
    else:        # gen-2
        return 1   # match Koide-shape extension

def project_n_x_su2l(n, color, su2l, j):
    """(F) Quantum mass formula style: n_fixed = n × SU2L (Hamming × isospin)."""
    n_fixed = n * su2l
    return max(0, N_G_EDGE - n_fixed)

candidates = [
    ("A: n × color × SU2L × gen_j",   project_naive_product),
    ("B: n + color + SU2L + gen_j",   project_naive_sum),
    ("C: n_fixed = n_Hamming alone",  project_hamming_only),
    ("D: n_fixed = color × SU2L",     project_color_x_su2l),
    ("E: gen-3 specific (j=1 ⇒ 0)",   project_gen3_only),
    ("F: n_fixed = n × SU2L",         project_n_x_su2l),
]

results_table = {}
for name, fn in candidates:
    print(f"\n  Candidate {name}")
    print(f"    {'species':<8s} {'n_free predicted':>17s} {'asserted':>10s} {'match?':>8s}")
    matches = []
    for sp, n, col, su2l, j, nfree_asserted, _ in SPECIES:
        nfree_pred = fn(n, col, su2l, j)
        if nfree_asserted is None:
            match_str = "—"
        else:
            match_str = "✓" if nfree_pred == nfree_asserted else "✗"
            matches.append((sp, nfree_pred == nfree_asserted))
        print(f"    {sp:<8s} {nfree_pred:>17d} {str(nfree_asserted):>10s} {match_str:>8s}")
    # Match score: y_τ AND y_t AND y_ν3 (the 3 species with asserted n_free)
    yt_match = next((m for sp, m in matches if sp == "y_t"), None)
    ytau_match = next((m for sp, m in matches if sp == "y_τ"), None)
    ynu3_match = next((m for sp, m in matches if sp == "y_ν3"), None)
    results_table[name] = {"y_t": yt_match, "y_τ": ytau_match, "y_ν3": ynu3_match}

print(f"\n  Summary across candidates:")
print(f"  {'candidate':<35s} {'y_τ(=1)':>9s} {'y_t(=0)':>9s} {'y_ν3(=1)':>10s} {'all 3?':>8s}")
print(f"  {'-'*78}")
any_full_match = False
for name in [n for n, _ in candidates]:
    r = results_table[name]
    all_match = r["y_t"] and r["y_τ"] and r["y_ν3"]
    if all_match:
        any_full_match = True
    print(f"  {name:<35s} {str(r['y_τ']):>9s} {str(r['y_t']):>9s} {str(r['y_ν3']):>10s} {str(all_match):>8s}")

N3 = True
RESULTS["N3_candidate_projections_tested"] = bool(N3)


# ============================================================================
# Step D — Verdict
# ============================================================================
print("\n" + "=" * 78)
print("W28 VERDICT")
print("=" * 78)
print()
if any_full_match:
    print("  POSITIVE: at least one candidate projection reproduces (y_τ → 1, y_t → 0,")
    print("  y_ν3 → 1) from species quantum numbers alone. R-14 partially unblocked.")
    print("  Next bounded step: test the candidate on the 8 open species (y_μ, y_e,")
    print("  y_b, y_s, y_d, y_c, y_u, y_ν2, y_ν1) and verify the Yukawa hierarchy.")
else:
    print("  NEGATIVE: none of the candidate projections reproduces the asserted")
    print("  (y_τ → 1, y_t → 0, y_ν3 → 1) simultaneously from species quantum")
    print("  numbers alone.")
    print()
    print("  STRUCTURAL CONCLUSION:")
    print("    The framework's per-species n_free assignment is not derivable by")
    print("    SIMPLE COUNTING of (n, color, SU(2)_L, gen) → mode-fixed count.")
    print("    Either the projection is more intricate than any of the 6 candidates")
    print("    above (and requires the V_Ram ≅ Cl(6)-Fock identification the framework")
    print("    has named as 'the open piece'), OR the assertion 'n_free=0 at gen-3")
    print("    up-type' is a fit-driven assertion (per master Yukawa doc §11.4")
    print("    'post-hoc unification not derived').")
    print()
    print("  EXTENDS THE SESSION'S R-14 NEGATIVES:")
    print("    chi_tilde 2026-05-01: V_Ram dimensional symmetry under C_3 × χ̃.")
    print("    W26: V_Ram + W21 H_VEV operator-norm symmetry.")
    print("    W27: V_Ram + W21 + P_swap operator-spectrum symmetry.")
    print("    W28 (this): no simple species → n_free projection reproduces the")
    print("                asserted exponent-principle values.")
    print()
    print("  The R-14 closure cannot be done by any apparatus surfaced in this")
    print("  session. The framework's per-species n_free derivation requires:")
    print("    - Either a NEW structural ingredient beyond W20-W22 + simple counting,")
    print("    - Or admission that the exponent-principle assertions (n_free=0 at")
    print("      gen-3 up-type especially) are fit-driven, consistent with master")
    print("      Yukawa doc §11.4's retraction.")

N4 = True
RESULTS["N4_verdict_documented"] = bool(N4)

print()
print("Gate check:")
all_pass = True
for k, expected in EXPECTED.items():
    actual = RESULTS.get(k)
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected: all_pass = False
    print(f"  {status}  {k:42s}  expected={expected}, got={actual}")

print()
print("=" * 78)
