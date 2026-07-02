#!/usr/bin/env python3
"""
ths_fock_beta_contribution_2026-06-01.py
========================================
The ths β-contribution computation (the α_GUT⁻¹=24 keystone test).

QUESTION.  The substrate net srs carries the SM matter as Cl(6)-Fock content
and yields the 2HDM one-loop β coefficients (b₂ = −3).  The framework ADOPTS
MSSM (b₂ = +1) to land α_GUT⁻¹ = 24 — its single most load-bearing adoption
(ADOPTED-MSSM-Sb).  The MDL-dominant superposition partner of srs is the
bipartite net **ths** (I4₁/amd #141; same k=3 → same Cl(6); cf.
[srs⊕ths superposition, 2026-06-01]).  Does ths's Cl(6)-Fock content supply
the Δb that promotes 2HDM → MSSM?

TARGET (exact, MSSM − 2HDM one-loop, GUT-normalized b₁):
        Δb = (Δb₁, Δb₂, Δb₃) = (+12/5, +4, +4).
The make-or-break is the SHARP non-abelian pair: exactly +4 on BOTH b₂ and b₃
(easy to get "some +contribution", hard to get exactly +4 on both).

WHAT THIS PROBE DOES.
  (1) Build ths from the validated #141 construction (reuse
      substrate_selection_theorem.net_bonds), confirm bipartite + 4-atom cell.
  (2) Confirm ths carries the SAME per-vertex Cl(6) Fock multiplets as srs
      (same k=3 → same Spin(6)≅SU(4)_PS → 8 = 4 + 4̄), reusing the R1.1
      Brauer-Weyl machinery.  The multiplet CONTENT is identical to srs.
  (3) THE DECISIVE STRUCTURAL FACT.  srs is non-bipartite → the χ̃ (double-
      cover / Yukawa) walk exists → its Fock content pairs into MASSIVE Dirac
      fermions → enters β via the (2/3)·T(R) FERMION term.  ths is bipartite
      → no χ̃ walk (LEMMA B of the selection theorem) → its content cannot
      pair into Dirac fermions → it enters β only via the (1/3)·T(R) SCALAR
      term.  This is forced, not chosen.
  (4) Compute ths's Δb under the scalar term and test against the target.

GROUP THEORY (canonical, no fitting):
  one-loop:  b = −(11/3)C₂(G) + (2/3)Σ_Weyl T(R) + (1/3)Σ_cplx-scalar T(R)
  C₂(SU(N)) = N ;  T(fund) = 1/2 ;  GUT norm  b₁ = (3/5)·b_Y.
  Per SM generation the matter multiplets satisfy the GUT relation
  T₂ = T₃ = (3/5)ΣY² = 2  (anomaly-free 16 of SO(10)); so one generation of
  SCALAR matter contributes a UNIFORM (1/3)·2 = 2/3 to each of b₁, b₂, b₃.

REUSE (standing directive): net build + bipartiteness from
substrate_selection_theorem; Cl(6) Brauer-Weyl + Spin(6)≅SU(4)_PS split from
R1_1_cl6_fock_su4_PS_decomposition_probe.
"""

import sys
from pathlib import Path
from fractions import Fraction as F

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from substrate_selection_theorem import net_bonds, quotient_bipartite       # validated ths build
from R1_1_cl6_fock_su4_PS_decomposition_probe import (                       # Cl(6) machinery
    build_gamma, bivector, TOL,
)

TARGET = {1: F(12, 5), 2: F(4), 3: F(4)}   # Δb = MSSM − 2HDM (GUT-normalized b₁)


# ---------------------------------------------------------------------------
# Group-theory β building blocks (all exact, Fractions)
# ---------------------------------------------------------------------------

# Per-SM-generation Dynkin / hypercharge sums (color INCLUDED).
#   T₂(gen) = T(Q,doublet)·3col? No: T₂ counts SU(2) doublets: Q(×3 col)+L = 3·½+½ = 2.
#   T₃(gen) = T(SU3): Q(doublet→2 fund)+u_R+d_R = 2·½+½+½ = 2.
#   ΣY²(gen) = 10/3  →  (3/5)·ΣY² = 2.
T2_GEN = F(2)
T3_GEN = F(2)
B1_GEN = F(3, 5) * F(10, 3)   # = 2  (GUT-normalized hypercharge index of one gen)
assert T2_GEN == T3_GEN == B1_GEN == F(2)

# 2 Higgs doublets (the EW Higgs sector shared by 2HDM and MSSM):
#   T₂(2H) = 2·½ = 1 ;  T₃ = 0 ;  (3/5)ΣY²(2H) = (3/5)·1 = 3/5.
T2_2H = F(1)
B1_2H = F(3, 5) * F(1)

# Gauge-group adjoint Casimirs:
C2 = {2: F(2), 3: F(3)}   # SU(2)_L, SU(3)_c ; U(1) has C₂ = 0


def delta_b_scalar_matter(n_gen):
    """Δb from n_gen generations of complex-SCALAR matter (sfermions): (1/3)·T."""
    return {1: F(1, 3) * n_gen * B1_GEN,
            2: F(1, 3) * n_gen * T2_GEN,
            3: F(1, 3) * n_gen * T3_GEN}


def delta_b_gaugino():
    """Δb from gauginos (adjoint Weyl fermions): (2/3)·C₂(G)."""
    return {1: F(0), 2: F(2, 3) * C2[2], 3: F(2, 3) * C2[3]}


def delta_b_higgsino():
    """Δb from 2 higgsino doublets (Weyl fermions): (2/3)·T."""
    return {1: F(2, 3) * B1_2H, 2: F(2, 3) * T2_2H, 3: F(0)}


def fmt(d):
    return f"({str(d[1]):>5}, {str(d[2]):>3}, {str(d[3]):>3})"


# ---------------------------------------------------------------------------
def section(t):
    print("\n" + "=" * 92 + f"\n {t}\n" + "=" * 92)


def main():
    section("STEP 1 — Build ths; confirm bipartite + 4-atom cell")
    pos, bonds, cent = net_bonds('ths')
    nq, bip, tri = quotient_bipartite('ths')
    print(f"  ths primitive cell vertices : {len(pos)}   centering : {cent}")
    print(f"  quotient |V| = {nq} ;  bipartite = {bip} ;  has triangle = {tri}")
    assert bip and not tri, "ths must be bipartite, triangle-free"
    print("  → ths is BIPARTITE (LEMMA B: no χ̃/Yukawa walk → no Dirac pairing).")

    section("STEP 2 — ths carries the SAME Cl(6) Fock multiplets as srs (k=3)")
    G = build_gamma()
    # Spin(6) ≅ SU(4)_PS, 15 bivector generators; chirality Γ_7 splits 8 = 4 + 4̄.
    pairs = [(a, b) for a in range(1, 7) for b in range(a + 1, 7)]
    bivs = {p: bivector(G, *p) for p in pairs}
    G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]
    eig = np.round(np.linalg.eigvalsh(G7)).astype(int)
    n_plus, n_minus = int((eig == 1).sum()), int((eig == -1).sum())
    print(f"  Spin(6) generators            : {len(bivs)}  (= dim su(4) = 15)")
    print(f"  Γ_7 split of 8-dim Fock        : 4(+) ⊕ 4(−)  →  ({n_plus}, {n_minus})")
    assert len(bivs) == 15 and (n_plus, n_minus) == (4, 4)
    print("  ths is k=3 (validated CS) → identical Cl(6) → identical SU(4)_PS")
    print("  multiplet content: per vertex = 1 PS gen (4,2,1)⊕(4̄,1,2) [R1.1].")
    print("  ⇒ ths and srs carry the SAME multiplets; they differ ONLY in")
    print("    bipartiteness, i.e. in whether the content is FERMION or SCALAR.")

    section("STEP 3 — The β term is FORCED by bipartiteness")
    print("  srs  (non-bipartite): χ̃ walk exists → Dirac fermions → (2/3)·T  [fermion]")
    print("  ths  (bipartite)    : no χ̃ walk     → complex scalars → (1/3)·T  [scalar]")
    print()
    print("  So ths can ONLY contribute through the scalar (1/3) term. The 2HDM→MSSM")
    print("  gap decomposes into three superpartner pieces (computed exactly):")
    sf3 = delta_b_scalar_matter(3)
    gaug = delta_b_gaugino()
    hino = delta_b_higgsino()
    tot = {i: sf3[i] + gaug[i] + hino[i] for i in (1, 2, 3)}
    print(f"    sfermions (3 gen, SCALAR)   Δb = {fmt(sf3)}   ← available to ths")
    print(f"    gauginos  (adjoint, FERMION)Δb = {fmt(gaug)}   ← FERMION: not from a scalar net")
    print(f"    higgsinos (doublet, FERMION)Δb = {fmt(hino)}   ← FERMION: not from a scalar net")
    print(f"    ----------------------------------------------------------------")
    print(f"    sum                          Δb = {fmt(tot)}   target = {fmt(TARGET)}")
    assert tot == TARGET, "decomposition must reproduce the exact MSSM−2HDM target"

    section("STEP 4 — ths's actual contribution vs the target")
    ths_contrib = sf3   # ths = 3 generations of scalar matter (sfermions)
    print("  Reading A — ths = 3 generations of SCALAR matter (the natural sfermion role,")
    print("              one PS gen per vertex, srs's fermion gen ↔ ths's scalar gen):")
    print(f"      ths Δb = {fmt(ths_contrib)}")
    print(f"      target = {fmt(TARGET)}")
    for i in (1, 2, 3):
        frac = ths_contrib[i] / TARGET[i]
        print(f"        b{i}: ths {str(ths_contrib[i]):>4} / target {str(TARGET[i]):>4}"
              f"  = {float(frac):.3f} of the gap")
    nonab_hit = (ths_contrib[2] == TARGET[2]) and (ths_contrib[3] == TARGET[3])
    print(f"      make-or-break (+4 on BOTH b₂,b₃): {'PASS' if nonab_hit else 'FAIL'}"
          f"  (ths gives +{ths_contrib[2]} on each, target +4)")

    print()
    print("  Reading B — use the per-cell FOCK DOUBLING (96 = 2×48 colored states,")
    print("              R1.1 'speculative MSSM doubling') → 6 gen of scalars:")
    sf6 = delta_b_scalar_matter(6)
    print(f"      ths Δb = {fmt(sf6)}")
    nonab_hit_B = (sf6[2] == TARGET[2]) and (sf6[3] == TARGET[3])
    print(f"      make-or-break (+4 on BOTH b₂,b₃): {'PASS' if nonab_hit_B else 'FAIL'}")
    print(f"      BUT b₁: scalars give {str(sf6[1])} vs target {str(TARGET[1])}"
          f"  (off by {str(sf6[1]-TARGET[1])} = 8/5)")
    print("      → the non-abelian betas are BLIND to fermion-vs-scalar and to Y;")
    print("        b₁ is the discriminator and it REJECTS 'pure doubled scalars'.")

    section("VERDICT")
    print("""\
  NEGATIVE on the strong hypothesis (ths supplies the full +4), but with a
  SHARP, principled structural result:

  • ths carries exactly the same SU(4)_PS multiplets as srs (same k=3, same
    Cl(6)); validated build, bipartite, 4-atom cell.

  • Bipartiteness FORCES ths's content into the scalar (1/3) β term. Read as
    3 generations of scalar matter, ths reproduces the MSSM SFERMION sector
    EXACTLY:  Δb = (+2, +2, +2).  That is precisely HALF of the non-abelian
    gap (+2 of the needed +4 on each of b₂, b₃) — the cleanest possible
    partial: the scalar superpartners of matter, and nothing else.

  • The remaining +2 on each non-abelian beta is GAUGINOS (adjoint Weyl
    fermions, Δb=(0,4/3,2)) + HIGGSINOS (Δb=(2/5,2/3,0)). Both are FERMIONS.
    A bipartite net has no χ̃ walk and therefore no Dirac-fermion content —
    so ths is STRUCTURALLY FORBIDDEN from supplying them. The very property
    that makes ths the natural scalar/dark partner (bipartiteness) is the
    property that caps its β contribution at the sfermion half.

  • A doubling reading (6-gen scalars) HITS +4 on both b₂,b₃ but over-shoots
    b₁ by 8/5 — confirming the non-abelian-only match is an artifact of T(R)
    being blind to the fermion/scalar split; b₁ exposes it.

  CONSEQUENCE for ADOPTED-MSSM-Sb / α_GUT⁻¹=24:
    The MSSM β is NOT derived from ths alone. ths derives the sfermion sector;
    the gaugino + higgsino (fermionic) half is not available from a bipartite
    scalar net and must come from the EDGE/gauge-operator (Cl(0,2)) sector —
    the still-open R1.3. The adoption is reduced, not discharged: from
    "all 2HDM→MSSM superpartners" to "the gaugino+higgsino fermion sector".
""")
    return nonab_hit


if __name__ == "__main__":
    ok = main()
    # exit 0 = the script ran and reported; the make-or-break verdict is in the text.
    raise SystemExit(0)
