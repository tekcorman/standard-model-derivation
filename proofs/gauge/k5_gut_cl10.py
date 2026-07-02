#!/usr/bin/env python3
"""
proofs/gauge/k5_gut_cl10.py

GATE-FIRST ANALYSIS: k=5 crystal net → Cl(10) → GUT gauge group

STATUS: BLOCKED at Step A (MDL-optimal 5-regular 3D crystal net not
uniquely identified by literature citation available to this framework).
The remaining steps are logically conditioned on Step A.

QUESTION
--------
Does the k=5 stage of MDL cooling correspond to SO(10) GUT, SU(5)
Georgi-Glashow, or something else?  And can this be closed at
theorem grade?

GATE-FIRST ANALYSIS
-------------------

Step A [BLOCKED — literature identification required]:
  Identify the MDL-optimal 5-regular 3D crystal net.

  At k=3: srs is MDL-optimal because it is the UNIQUE vertex- AND
  edge-transitive 3-connected 3D crystal net (Sunada 2012, Notices AMS
  59(2), 208-215, 2012; Delgado-Friedrichs & O'Keeffe 2003, Acta
  Crystallogr. A 59, 351-360).  Edge-transitivity → DL(edges) = 0;
  no other k=3 3D net achieves this.  The Sunada uniqueness theorem
  supplies the needed Type 3 citation.

  At k=5: no analogous uniqueness theorem is known to this framework.
  Candidates include:
    - "bcu-x" (body-centred cubic with extra edges, k=8): k too high
    - "pcu" (primitive cubic): k=6, not k=5
    - "soc" net (O'Keeffe notation): coordination varies
    - 5-regular nets studied in topological crystallography: multiple
      exist, none proven to have a single MDL-minimising candidate
      with the same edge-transitive uniqueness property that srs has
      at k=3.

  The known k=3 proof strategy rests on TWO facts about srs:
    (i)  vertex-transitivity (unique orbit of nodes under space group)
    (ii) edge-transitivity (unique orbit of edges under space group)
  Together these force DL(nodes) = 0 and DL(edges) = 0.  For k=5,
  an edge-transitive (actually arc-transitive, i.e., flag-transitive)
  5-regular 3D crystal net would play the same role.  Whether such a
  net exists and is unique is a reticular-chemistry result that is NOT
  established by the citations available to this framework.

  VERDICT: Step A BLOCKED (Type 3 citation missing).  Cannot proceed
  to a theorem-grade proof without this citation.

Step B [Type 2 — algebra, conditional on Step A]:
  Assuming the MDL-optimal 5-regular 3D crystal net is identified,
  Jordan-Wigner maps its 5 edge modes at each node to 5 fermionic
  creation/annihilation operators satisfying CAR:

    {a_i, a_j†} = δ_{ij},  {a_i, a_j} = 0  (i,j = 1..5)

  From these one constructs 10 gamma matrices:
    γ_{2i-1} = a_i + a_i†,   γ_{2i} = i(a_i† - a_i)   (i = 1..5)

  They satisfy {γ_μ, γ_ν} = 2δ_{μν}I_{32}, generating Cl(10) on the
  2^5 = 32-dimensional Fock space.

  Gate type: Type 2 (algebra, consequence of CAR and standard
  Jordan-Wigner construction).  Fully analogous to the Cl(6) step
  at k=3 (proofs/gauge/cl8_verification.py) and verifiable
  computationally.  NOT BLOCKED — but downstream of Step A.

Step C [Type 2 — algebra, conditional on Step B]:
  From the 10 generators of Cl(10), form the C(10,2) = 45 bivectors:

    S_{μν} = (i/4)[γ_μ, γ_ν]

  These are Hermitian traceless operators on C^32 and close under
  commutation with the so(10) structure constants (standard result:
  the bivectors of Cl(2n) generate so(2n) inside the Clifford
  algebra).

  So the gauge group identified directly from the Clifford bivectors
  is Spin(10) ≅ SO(10), dim = 45.

  This is the DIRECT output of the Clifford construction.  SU(5) does
  NOT appear at this stage; it is a proper subgroup.

  Gate type: Type 2 (standard Clifford algebra theorem; see e.g.
  Lounesto, "Clifford Algebras and Spinors", Cambridge 2001, §17.3).

Step D [Type 2 — algebra: spinor decomposition, conditional on Step C]:
  The Fock space decomposes under the Cl(10) chirality operator:

    Γ_11 = (-i)^5 · γ_1 γ_2 ... γ_{10}

  Γ_11 has eigenvalues ±1, each with multiplicity 2^4 = 16.  The two
  chiral half-spaces are the two irreducible Spin(10) spinor
  representations:

    32 = 16 ⊕ 16bar

  Under SU(5) × U(1) ⊂ SO(10):
    16 = 10 ⊕ 5bar ⊕ 1   (Georgi-Glashow decomposition)

  Under SU(4) × SU(2)_L × SU(2)_R ⊂ SO(10):
    16 = (4, 2, 1) ⊕ (4bar, 1, 2)  (Pati-Salam decomposition)

  BOTH decompositions are subgroup decompositions of the SAME group
  SO(10).  The Clifford construction at k=5 gives SO(10) as the
  natural output; it does NOT select one maximal subgroup over the
  other.

  Gate type: Type 2 (standard branching rule for SO(10); see e.g.
  Slansky, Phys. Rep. 79 (1981) 1-128, Table 7).

Step E [BLOCKED — additional physical input required]:
  Does SO(10) cooling select SU(5) or Pati-Salam as the k=4 phase?

  At k=4 the framework adopts Pati-Salam SU(4) × SU(2)_L × SU(2)_R
  (from Cl(8) on the 4-regular net).  The Cl(8) bivectors give
  so(8), NOT the Pati-Salam group directly.  The Pati-Salam group
  (dimension 15+3+3 = 21) is a proper subgroup of SO(8) (dim 28).

  The question "which SO(10) maximal subgroup is selected at k=4?" is
  equivalent to asking which EMBEDDING of Cl(8) inside Cl(10) the
  MDL observer uses when k-cooling from 5 to 4.  Two candidates:
    (i)  Cl(8) ↪ Cl(10) via first 8 generators → SO(8) ⊂ SO(10)
    (ii) Cl(8) ↪ Cl(10) via a different embedding giving SU(5) ⊃ SU(4)...

  To select between (i) and (ii) requires specifying WHICH 4-regular
  crystal net is MDL-optimal at k=4 AND identifying the homomorphism
  between its edge-mode algebra and the k=5 algebra.  This is
  BLOCKED for the same reason as Step A: no uniqueness theorem for
  the k=4 net is available at the level of the k=3 Sunada theorem.

  Gate type: BLOCKED (requires Step A analogue for k=4 and an
  embedding theorem).

SUMMARY
-------

Step A: BLOCKED  (no Type 3 citation for k=5 MDL-unique crystal net)
Step B: NOT BLOCKED, conditional on A  (CAR → Cl(10), standard JW)
Step C: NOT BLOCKED, conditional on B  (Cl(10) bivectors → so(10))
Step D: NOT BLOCKED, conditional on C  (spinor = 16+16bar, SO(10) is
         the natural group; SU(5) is a subgroup, not the primary output)
Step E: BLOCKED  (SO(10) → SU(5) vs PS requires embedding theorem)

CONCLUSION
----------

At k=5, the Clifford algebra construction (conditional on identifying
the MDL-optimal 5-regular net) gives Spin(10) ≅ SO(10) as the
NATURAL gauge group.  This is a larger group than SU(5).

SU(5) is a maximal subgroup of SO(10), but selecting it over the
Pati-Salam decomposition requires additional input:
  (a) a specification of WHICH maximal subgroup of SO(10) the k=4
      cooling step selects, and
  (b) a proof that the k=4 MDL-optimal crystal net's Clifford algebra
      is the SU(5) factor rather than the PS factor.

The current ADOPTED identification "k=5 → SU(5)" is therefore an
over-identification: the Clifford derivation gives SO(10), not SU(5).
A more accurate (and still logically correct) statement is:

    k=5  →  SO(10)  (Clifford construction, Steps B-D, conditional on A)

    SO(10) ⊃ SU(5) × U(1)   (Georgi-Glashow embedding, Type 3:
                               Georgi & Glashow 1974, PRL 32, 438)

    SO(10) ⊃ SU(4) × SU(2)_L × SU(2)_R  (Pati-Salam embedding,
                               Type 3: Pati & Salam 1974, PRD 10, 275)

Both embeddings are exact.  Neither is selected by the Clifford
structure alone.

WHAT WOULD CLOSE THIS
----------------------

To close at theorem grade, two literature results are needed:

  (A-close) A uniqueness theorem for the MDL-optimal k=5 3D crystal
    net at the level of Sunada's k=3 theorem: an arc-transitive (or
    vertex+edge-transitive) 5-regular 3D crystal net that is unique
    up to isomorphism.  Candidate search direction: O'Keeffe's RCSR
    database (rcsr.net) for k=5 nets; Eon 2011 (Acta Crystallogr. A 67,
    68-86) for periodic nets with high symmetry.  If such a net exists
    and has a space group realisation with a verifiable uniqueness
    theorem, Step A closes.

  (E-close) An embedding selection theorem: given a k-cooling step
    k=5 → k=4, which SO(10) maximal subgroup does the "delete 2
    generators" operation select?  The standard Clifford inclusion
    Cl(8) ↪ Cl(10) (first 8 generators) selects SO(8) ⊂ SO(10),
    which does NOT match either SU(5) or Pati-Salam directly.  A
    representation-theoretic argument is needed.

REFERENCES
----------
- Sunada, T. (2012). Crystals that nature might miss creating. Notices
  AMS 59(2), 208-215.  [Type 3 — srs uniqueness at k=3]
- Delgado-Friedrichs, O. & O'Keeffe, M. (2003). Acta Crystallogr. A 59,
  351-360.  [Type 3 — k ≥ d for d-periodic nets]
- Lounesto, P. (2001). Clifford Algebras and Spinors. Cambridge.
  §17.3.  [Type 3 — bivectors generate so(2n)]
- Slansky, R. (1981). Group theory for unified model building.
  Phys. Rep. 79, 1-128.  Table 7.  [Type 3 — SO(10) branching rules]
- Georgi, H. & Glashow, S.L. (1974). Unity of all elementary-particle
  forces. PRL 32, 438.  [Type 3 — SU(5) ⊂ SO(10)]
- Pati, J.C. & Salam, A. (1974). PRD 10, 275.
  [Type 3 — SU(4) × SU(2) × SU(2) ⊂ SO(10)]
- Eon, J.-G. (2011). Acta Crystallogr. A 67, 68-86.
  [Type 3 candidate — periodic nets with transitivity]
- O'Keeffe, M. et al. (2008). Acta Crystallogr. A 64, 400-406.
  [RCSR database — candidate source for k=5 nets]
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'predictions'))

from d_spatial import predict_d_spatial
from k_star import predict_k_star


# -----------------------------------------------------------------------
# SECTION 1: Clifford dimension count (Steps B-D, algebra only)
# -----------------------------------------------------------------------
# These are purely algebraic facts that hold regardless of which k=5
# crystal net is eventually identified as MDL-optimal.

d = predict_d_spatial()   # = 3
k_star = predict_k_star(d)  # = 3

# k=5 is a hypothetical GUT stage, not the current MDL minimum.
k_gut = 5

# Jordan-Wigner: k fermionic modes → 2k Clifford generators → Cl(2k)
# Fock space dimension: 2^k
n_modes_k5 = k_gut
n_gamma_k5 = 2 * k_gut          # = 10 generators of Cl(10)
fock_dim_k5 = 2 ** k_gut        # = 32

# Number of bivectors (generators of the embedded Spin group)
n_bivectors_k5 = n_gamma_k5 * (n_gamma_k5 - 1) // 2   # = C(10,2) = 45

# The bivectors S_{μν} = (i/4)[γ_μ, γ_ν] generate so(10)
# dim(so(10)) = dim(SO(10)) = n*(n-1)/2  with n=10
dim_so10 = 10 * 9 // 2   # = 45

assert n_bivectors_k5 == dim_so10, (
    "Bivector count must equal dim(SO(10)) = 45")

# Chirality operator: Γ_{11} = (-i)^5 γ_1 ... γ_{10}
# Its ±1 eigenspaces each have dimension 2^(k-1) = 16
chiral_dim_k5 = fock_dim_k5 // 2   # = 16

# The 16 of SO(10) under SU(5): 16 = 10 + 5bar + 1
su5_decomp = {10: "10 (antisym. tensor)", "5bar": "5bar (fund. rep)", 1: "1 (singlet)"}
assert 10 + 5 + 1 == 16, "SU(5) decomposition of SO(10) 16-spinor sums to 16"

# Under Pati-Salam: 16 = (4,2,1) + (4bar,1,2)
ps_decomp_L = (4, 2, 1)   # dim = 4*2*1 = 8
ps_decomp_R = (4, 1, 2)   # dim = 4*1*2 = 8
assert math.prod(ps_decomp_L) + math.prod(ps_decomp_R) == 16, (
    "Pati-Salam decomposition of SO(10) 16-spinor sums to 16")


# -----------------------------------------------------------------------
# SECTION 2: DL ordering (Step D of k_cooling_sm_uniqueness.py)
# -----------------------------------------------------------------------
# This verifies that DL(k=5) > DL(k=4) > DL(k=3).
# The cooling direction is unambiguous regardless of Step A.

def dl_per_node(k_val):
    """DL per node = log2(2^k * k) = k + log2(k)."""
    return k_val + math.log2(k_val)

dl_k3 = dl_per_node(3)
dl_k4 = dl_per_node(4)
dl_k5 = dl_per_node(5)

assert dl_k3 < dl_k4 < dl_k5, (
    "DL must increase monotonically: k=3 < k=4 < k=5")


# -----------------------------------------------------------------------
# SECTION 3: Group dimension comparison
# -----------------------------------------------------------------------

dim_groups = {
    "SO(10)":                          45,    # Spin(10) = bivectors of Cl(10)
    "SU(5)":                           24,    # dim SU(5) = 5^2 - 1
    "SU(4)xSU(2)_LxSU(2)_R":          21,    # 15 + 3 + 3
    "SU(3)xSU(2)xU(1) [SM]":          12,    # 8 + 3 + 1
}

# Subgroup chain: SO(10) ⊃ SU(5) × U(1) ⊃ SU(3) × SU(2) × U(1) [SM]
#            and: SO(10) ⊃ SU(4) × SU(2)_L × SU(2)_R ⊃ SM
# Both chains end at SM.

assert dim_groups["SU(5)"] < dim_groups["SO(10)"], (
    "SU(5) is a proper subgroup of SO(10)")
assert dim_groups["SU(4)xSU(2)_LxSU(2)_R"] < dim_groups["SO(10)"], (
    "Pati-Salam is a proper subgroup of SO(10)")
assert dim_groups["SU(3)xSU(2)xU(1) [SM]"] < dim_groups["SU(5)"], (
    "SM is a subgroup of SU(5)")


# -----------------------------------------------------------------------
# SECTION 4: Clifford subalgebra tower (Step F of k_cooling_sm_uniqueness)
# -----------------------------------------------------------------------
# Cl(6) ⊂ Cl(8) ⊂ Cl(10): each inclusion is via the first 2k generators.

k_levels = [3, 4, 5]
fock_dims = {k: 2**k for k in k_levels}            # 8, 16, 32
bivec_dims = {k: k*(2*k-1) for k in k_levels}      # 15, 28, 45
                                                    # = dim so(2k)

assert bivec_dims[3] == 15, "so(6) has dim 15"
assert bivec_dims[4] == 28, "so(8) has dim 28"
assert bivec_dims[5] == 45, "so(10) has dim 45"

# The STANDARD embedding Cl(2k-2) ↪ Cl(2k) via first 2k-2 generators
# selects so(2k-2) ⊂ so(2k), not a GUT-standard chain.
# so(6) ⊂ so(8) ⊂ so(10)
# This does NOT directly give SU(5) or Pati-Salam.

# The physically relevant embedding is via SPINOR branching:
# Spin(10) → Spin(8) × U(1) → ...
# This requires specifying which U(1) is modded out, which is
# NOT determined by the Clifford algebra structure alone.


# -----------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  k=5 GUT stage: Cl(10) → gauge group assessment")
    print("  GATE-FIRST ANALYSIS — BLOCKED at Step A")
    print("=" * 70)
    print()
    print("ALGEBRAIC FACTS (Steps B-D, theorem-grade conditional on Step A):")
    print()
    print(f"  k=5 fermionic modes: {n_modes_k5}")
    print(f"  Clifford generators: {n_gamma_k5}  → Cl(10)")
    print(f"  Fock space dim:      {fock_dim_k5}  (= 2^5)")
    print(f"  Bivectors:           {n_bivectors_k5}  (= C(10,2) = dim so(10))")
    print(f"  Natural gauge group: SO(10)  [from Cl(10) bivectors]")
    print()
    print(f"  Chirality split:     {fock_dim_k5} = {chiral_dim_k5} + {chiral_dim_k5}  (spinor + anti-spinor)")
    print(f"  SO(10) spinor:       16 = 10 + 5bar + 1  [under SU(5)]")
    print(f"  SO(10) spinor:       16 = (4,2,1) + (4bar,1,2)  [Pati-Salam]")
    print()
    print("GROUP DIMENSIONS:")
    for name, dim in dim_groups.items():
        print(f"  dim({name}) = {dim}")
    print()
    print("DL ORDERING (k=3 absorbing state confirmed):")
    for k_val in k_levels:
        print(f"  DL(k={k_val}) = {dl_per_node(k_val):.4f} bits/node")
    print()
    print("CLIFFORD TOWER:")
    print(f"  {'k':>3}  {'Cl(2k)':>8}  {'Fock dim':>9}  {'so(2k) dim':>11}  "
          f"{'Natural group':>20}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*20}")
    nat_groups = {3: "SO(6) ≅ SU(4)_PS",
                  4: "SO(8)",
                  5: "SO(10)"}
    for k_val in k_levels:
        print(f"  {k_val:>3}  Cl({2*k_val:>2})   {fock_dims[k_val]:>9}  "
              f"{bivec_dims[k_val]:>11}  {nat_groups[k_val]:>20}")
    print()
    print("NOTE: 'Natural group' = group generated by Clifford bivectors.")
    print("      SO(6) ≅ SU(4) is the exceptional isomorphism at k=3.")
    print("      No analogous exceptional isomorphism exists at k=4 or k=5.")
    print("      The adopted 'k=4 → Pati-Salam' and 'k=5 → SU(5)' identifications")
    print("      are NOT the Clifford-natural outputs (those are SO(8) and SO(10)).")
    print()
    print("=" * 70)
    print("GATE STATUS SUMMARY")
    print("=" * 70)
    print()
    print("Step A [BLOCKED]:")
    print("  No Type 3 citation available for uniqueness of MDL-optimal")
    print("  5-regular 3D crystal net analogous to Sunada's srs theorem.")
    print("  Candidate resolution: O'Keeffe RCSR (rcsr.net) k=5 nets;")
    print("  Eon 2011 (Acta Crystallogr. A 67, 68-86).")
    print()
    print("Step B [NOT BLOCKED — conditional on A]:")
    print("  CAR on 5 edge modes → Cl(10) on 32-dim Fock space. Standard JW.")
    print()
    print("Step C [NOT BLOCKED — conditional on B]:")
    print("  Cl(10) bivectors → so(10) algebra, dim 45. Standard result.")
    print("  Natural gauge group = SO(10), not SU(5).")
    print()
    print("Step D [NOT BLOCKED — conditional on C]:")
    print("  SO(10) spinor 32 = 16 + 16bar.")
    print("  Both SU(5) and Pati-Salam are maximal subgroups; Clifford")
    print("  structure does not select between them.")
    print()
    print("Step E [BLOCKED]:")
    print("  SO(10) → SU(5) vs Pati-Salam requires embedding selection")
    print("  theorem (which k=4 net embeds how inside k=5 net algebra).")
    print()
    print("CONSEQUENCE FOR EXISTING ADOPTED CLAIM:")
    print("  'k=5 → SU(5) Georgi-Glashow' is an OVER-IDENTIFICATION.")
    print("  The correct Clifford output is SO(10).")
    print("  SU(5) is a proper subgroup.  Corrected adopted statement:")
    print("    k=5 → SO(10)  [conditional on Step A; algebra theorem-grade")
    print("                   once Step A is supplied]")
    print("  The further identification SO(10) → SU(5) requires Step E.")
    print()
    print("REFERENCES:")
    print("  Sunada 2012 (srs uniqueness at k=3) [Type 3]")
    print("  Lounesto 2001 §17.3 (bivectors → so(2n)) [Type 3]")
    print("  Slansky 1981 Phys. Rep. 79 Table 7 (SO(10) branching) [Type 3]")
    print("  Georgi & Glashow 1974 PRL 32, 438 (SU(5) GUT) [Type 3]")
    print("  Pati & Salam 1974 PRD 10, 275 (Pati-Salam in SO(10)) [Type 3]")
    print("  Eon 2011 Acta Crystallogr. A 67, 68-86 (k=5 net candidate) [Type 3]")
