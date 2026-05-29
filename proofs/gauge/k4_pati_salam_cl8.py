#!/usr/bin/env python3
"""
proofs/gauge/k4_pati_salam_cl8.py

INVESTIGATION: Does k=4 → Cl(8) → Pati-Salam?
CAN the G2-theorem approach be applied to the k=4 MDL crystal net?

VERDICT: BLOCKED.  The k=4 → Pati-Salam adoption in k_cooling_sm_uniqueness.py
is NOT closed at theorem grade.  This file documents every attempted step,
the gate status of each, and the precise locations of blockage.

SUMMARY OF FINDINGS
-------------------
(A) The MDL-optimal 4-regular crystal net is NOT established.
    Sunada (2012) identifies srs as the unique vertex-and-edge-transitive
    3-regular 3D net.  No analogous uniqueness theorem for k=4 is cited or
    established in this framework.  The candidate "dia" (diamond) net has
    2 atoms per primitive cell, not 8 = 2^4.

(B) Cl(8) bivectors give Spin(8) (dim=28), NOT SU(4)×SU(2)_L×SU(2)_R (dim=21).
    The Pati-Salam group PS fits INSIDE Spin(8) as a proper subgroup.
    No physical or MDL argument selects PS over the other maximal subgroups of
    Spin(8) (SO(7), Spin(7), G2×SU(2), etc.).

(C) The exceptional isomorphism used at k=3 — Spin(6) ≅ SU(4) — has NO
    analogue at k=4.  Spin(8) is NOT isomorphic to any product of classical
    groups.  It is a simple group with exceptional triality (Out(Spin(8)) = S_3),
    not SU(4)×SU(2)×SU(2).

(D) Even granting PS ⊂ Spin(8), there is no MDL argument identifying WHICH
    SU(4)×SU(2)×SU(2) subgroup of Spin(8) is selected.  The embedding is
    not unique.

(E) The adoption k=4 → PS in k_cooling_sm_uniqueness.py is based on the GUT
    hierarchy pattern (SU(5) → Pati-Salam → SM), not on a Cl(8) bivector
    derivation analogous to G2.  It remains ADOPTED and cannot be promoted
    to theorem without resolving the four blockers above.

This file is a NEGATIVE-RESULT SCOPING DOCUMENT.  It follows the gate-first
methodology of an internal note.

GATE CONVENTIONS (from docs/parameters/parameter_linter.md)
-------------------------------------------------
Type 1: Axiom (A1, A2, A3, A4, A5 from docs/framework/framework_axioms.md)
Type 2: Explicit algebra (symbolic/numerical, verifiable by CAS)
Type 3: Cited theorem with precise bibliographic reference
Type 4: Upstream closed predictions/ or proofs/ file

REFERENCES
----------
- predictions/k_star.py (Type 4: k*=3 for d=3)
- proofs/gauge/srs_so10_embedding.py (Type 4: G2 template at k=3)
- proofs/gauge/cl8_verification.py (Type 4: Cl(8) construction)
- proofs/cosmology/k_cooling_sm_uniqueness.py (Type 4: GUT hierarchy, k=4 ADOPTED)
- Sunada T (2012). "Crystals That Nature Might Miss Creating." Notices AMS 59(2). [Type 3]
- Delgado-Friedrichs O, O'Keeffe M (2003). Acta Crystallogr. A 59, 351-360. [Type 3]
- Baez JC (2002). The Octonions. Bull. AMS 39, 145-205. §4. [Type 3]
- Adams JF (1969). Lectures on Lie Groups. Benjamin/Cummings. [Type 3: Spin(8) structure]
- Clifford algebra classification: AMS table, e.g. Lawson-Michelsohn (1989) §I.5. [Type 3]
"""

# ============================================================
# STEP 1: WHAT IS THE MDL-OPTIMAL 4-REGULAR CRYSTAL NET?
# ============================================================
#
# GATE: BLOCKED (Type 3 citation missing; Type 2 analysis inconclusive)
#
# For k=3, d=3: the unique MDL-optimal crystal net is srs.
# Source: Sunada T (2012). Notices AMS 59(2), 208-215.
#   - srs is the UNIQUE vertex-transitive AND edge-transitive 3-regular 3D net.
#   - Vertex-transitivity: all nodes equivalent under space group.
#   - Edge-transitivity: all edges equivalent under space group.
#   - MDL argument: maximum space-group symmetry => minimum free parameters
#     => minimum DL (see proofs/foundations/dl_comparison.py for explicit DL accounting).
#
# For k=4, d=3: Candidate is "dia" (diamond net, RCSR symbol "dia").
#   - dia properties (Type 3: RCSR database, O'Keeffe et al. 2008;
#     Baburin et al. 2008, Acta Crystallogr. A 64, 45-54):
#     * 4-regular (tetrahedral coordination)
#     * Space group: Fd-3m (No. 227)
#     * Primitive cell: 2 atoms (Type 3: RCSR database)
#     * Vertex-transitive: yes (one Wyckoff position 8a)
#     * Edge-transitive: yes (all bonds equivalent)
#
#   - Is dia the UNIQUE vertex-and-edge-transitive 4-regular 3D net?
#     This is a mathematical question analogous to Sunada's uniqueness for srs.
#     STATUS: No such uniqueness theorem is cited in this framework.
#     The srs uniqueness (Sunada 2012) relied on:
#       (i)  Vertex-transitivity + edge-transitivity (the net is "maximally symmetric")
#       (ii) The Laves graph structure (I4_132, three-fold screw)
#     For k=4: multiple edge-transitive 4-regular 3D nets exist
#     (dia, acs, ...).  Whether dia is the unique MDL-minimum is not established.
#     [BLOCKER A: Missing Type 3 citation for uniqueness of dia at k=4.]
#
#   - DL comparison of dia:
#     * Space group Fd-3m has order 192 (double the order 96 of I4_132 for srs).
#     * BUT: dia has 2 atoms per primitive cell vs 4 for srs.
#     * The DL tradeoff is non-trivial; no explicit DL computation has been done
#       for dia vs other 4-regular nets in this framework.
#     [BLOCKER A continued: DL analysis needed but not performed.]
#
# CONCLUSION FOR STEP 1:
#   The MDL-optimal 4-regular crystal net is NOT established.
#   dia is a natural candidate but lacks the uniqueness certificate that srs has.
#   The G2 approach cannot proceed without a fixed net.

DIA_PRIMITIVE_CELL_ATOMS = 2    # dia: 2 atoms per primitive cell
DIA_IS_VERTEX_TRANSITIVE = True  # Type 3: RCSR
DIA_IS_EDGE_TRANSITIVE   = True  # Type 3: RCSR

# Note from explorations/srs_mass_scale_proof.py lines 829-845 (Type 4):
#   "k*=4: 2^(k*-1) = 8, diamond primitive cell = 2 atoms. NO MATCH."
#   "n = 2^(k*-1) is NOT a general relation."
# This is the first sign that the k=4 case is structurally different from k=3.


# ============================================================
# STEP 2: CL(8) CONSTRUCTION AT k=4
# ============================================================
#
# GATE: THEOREM-GRADE (Type 2 + Type 4)
#
# IF we grant the k=4 net, the local algebra follows by the same JW construction
# used at k=3.
#
# At each node of a 4-regular crystal net:
#   - 4 edges incident at the node => 4 edge qubits
#   - Jordan-Wigner (A4, closed in docs/theorems/theorem_car_local_jordan_wigner.md):
#     4 fermionic modes on a 16-dim Fock space
#   - Cl(8) generators: gamma_{2i-1} = a_i + a_i^dag,  gamma_{2i} = i(a_i^dag - a_i)
#     for i=1,2,3,4 (8 generators total)
#   - The Fock space has dimension 2^4 = 16
#   - Cl(8) acts irreducibly on the 16-dim space (Type 3: Clifford algebra classification,
#     Lawson-Michelsohn 1989 §I.5; Cl(8,0) ≅ M_16(R) as a ring)
#
# This step is TYPE 2 (explicit algebra, replicating cl8_verification.py construction)
# and TYPE 4 (Jordan-Wigner: theorem_car_local_jordan_wigner.md).
#
# GATE STATUS: PASS — Cl(8) construction is theorem-grade given k=4 and JW.

N_MODES_K4     = 4          # 4 edge modes at a k=4 node
FOCK_DIM_K4    = 2**4       # = 16
N_CL8_GENS     = 2 * 4      # = 8 generators of Cl(8)
N_CL8_BIVEC    = 8*7//2     # = 28  bivectors = dim(Spin(8)) = dim(so(8))

assert N_CL8_BIVEC == 28, "C(8,2) = 28"


# ============================================================
# STEP 3: SPIN(8) FROM CL(8) BIVECTORS
# ============================================================
#
# GATE: THEOREM-GRADE (Type 3)
#
# The Spin(8) generators are the Cl(8) bivectors:
#   S_{mu,nu} = (i/4)[gamma_mu, gamma_nu]   for 1 <= mu < nu <= 8
#
# There are C(8,2) = 28 bivectors => dim(Spin(8)) = 28.
# This is correct: Spin(8) ≅ SO(8), dim = 8*7/2 = 28.
#
# Properties (Type 3: standard Lie group theory, e.g. Adams 1969 Lectures on Lie Groups):
#   - Spin(8) is a rank-4 simple Lie group of type D_4.
#   - dim(Spin(8)) = 28.
#   - Spin(8) has three inequivalent 8-dim representations:
#       8_v (vector), 8_s (spinor-plus), 8_c (spinor-minus)
#     related by the TRIALITY automorphism.
#   - Out(Spin(8)) = S_3 (symmetric group on 3 elements, the triality group).
#     Type 3: Baez (2002) Bull. AMS 39, 145-205, §4.
#   - The Clifford algebra: Cl(8,0) ≅ M_{16}(R) (16x16 real matrices).
#     Type 3: Lawson-Michelsohn (1989) §I.5, Table 1.
#
# This step is THEOREM-GRADE.

DIM_SPIN8 = 28         # dim(Spin(8)) = dim(so(8)) = C(8,2)
RANK_SPIN8 = 4         # rank of Spin(8) = rank of D_4
N_SPIN8_REPS_DIM8 = 3  # three 8-dim reps (vector, spinor+, spinor-)


# ============================================================
# STEP 4: DOES SPIN(8) CONTAIN PATI-SALAM?
# ============================================================
#
# GATE: PASS (Type 3) — but this step does NOT establish that PS is selected.
#
# Pati-Salam group: G_PS = SU(4)_c × SU(2)_L × SU(2)_R
#   - dim(SU(4)) = 15
#   - dim(SU(2)) = 3
#   - dim(G_PS) = 15 + 3 + 3 = 21
#
# Is G_PS ⊂ Spin(8)?
#
# YES.  The standard embedding (Type 3: see e.g. Slansky 1981, Phys. Rep. 79, 1;
# Bertolini, Santachiara & Serone 2002; standard GUT model-building):
#
#   SU(4) ⊂ SO(6) ⊂ SO(8):
#     SO(6) ≅ SU(4) (exceptional isomorphism, dim=15) embeds in SO(8) via the
#     standard block-diagonal: diag(SO(6), I_2) ⊂ SO(8).
#     [The extra I_2 factor is a 2-dim trivial block.]
#
#   SU(2)_L × SU(2)_R ⊂ SO(4) ⊂ SO(8):
#     SO(4) ≅ SU(2) × SU(2) (exceptional isomorphism, dim=6) embeds in SO(8)
#     via diag(I_4, SO(4)) ⊂ SO(8).
#
#   Together: SU(4) × SU(2)_L × SU(2)_R ⊂ SO(6) × SO(4) ⊂ SO(8).
#   The dimensions check out: 15 + 6 = 21 ⊂ 28 (Pati-Salam is a PROPER subgroup).
#
# However:
#   dim(G_PS) = 21 < dim(Spin(8)) = 28.
#   The "extra" 28 - 21 = 7 generators are NOT in G_PS.
#   G_PS is a PROPER subgroup of Spin(8), not a quotient or isomorphic image.
#
# CRITICAL DIFFERENCE FROM k=3 CASE:
#   At k=3: Cl(6) bivectors give Spin(6) ≅ SU(4) (EXACT isomorphism, not embedding).
#            dim(Spin(6)) = 15 = dim(SU(4)). The exceptional isomorphism is exact.
#            This is why the k=3 derivation is theorem-grade: there are NO leftover
#            generators. Spin(6) IS SU(4)_PS, not merely a group containing SU(4)_PS.
#
#   At k=4: Cl(8) bivectors give Spin(8) (dim=28), which CONTAINS SU(4)×SU(2)×SU(2)
#            (dim=21) as a PROPER subgroup. There is no exceptional isomorphism making
#            Spin(8) equal to a Pati-Salam product. Spin(8) is a simple group (type D_4)
#            with no such product decomposition.

DIM_SU4   = 15   # dim(SU(4))
DIM_SU2   = 3    # dim(SU(2))
DIM_PS    = DIM_SU4 + DIM_SU2 + DIM_SU2   # = 21
DIM_SPIN8 = 28

LEFTOVER_GENS = DIM_SPIN8 - DIM_PS   # = 7 generators unaccounted for by PS

assert LEFTOVER_GENS == 7, f"Expected 7 leftover generators, got {LEFTOVER_GENS}"
assert DIM_PS < DIM_SPIN8, "G_PS is a proper subgroup of Spin(8)"

# There is NO exceptional isomorphism Spin(8) ≅ G_PS.
# Spin(8) is a simple group; G_PS is not simple (it has three non-trivial factors).
# Type 3: Adams (1969); Dynkin (1952) classification of maximal subgroups of D_4.

print(f"dim(G_PS = SU(4)×SU(2)×SU(2)) = {DIM_PS}")
print(f"dim(Spin(8)) = {DIM_SPIN8}")
print(f"Leftover generators in Spin(8) not in G_PS: {LEFTOVER_GENS}")
print(f"G_PS is a PROPER subgroup of Spin(8), not isomorphic to it.")


# ============================================================
# STEP 5: THE FOUR BLOCKERS FOR k=4 → PATI-SALAM
# ============================================================
#
# Even if we accept G_PS ⊂ Spin(8), there are four distinct gaps
# that prevent closing k=4 → PS at theorem grade.
#
# BLOCKER A: No uniqueness theorem for the k=4 MDL crystal net.
#   Status: OPEN
#   What is needed: A uniqueness result analogous to Sunada (2012) for k=4,
#   or an explicit DL comparison showing dia is the unique minimum.
#   Gate type needed: Type 3 (cited theorem) or Type 2 (explicit DL computation
#   across all k=4, d=3 edge-transitive nets with verified enumeration).
#
# BLOCKER B: Spin(8) ≇ G_PS.
#   At k=3, the key step is Spin(6) ≅ SU(4) (exceptional isomorphism).
#   There is NO analogous exceptional isomorphism at k=4.
#   Spin(8) is simple; SU(4)×SU(2)×SU(2) is not simple.
#   Gate type needed: There is no Type 3 theorem establishing Spin(8) ≅ G_PS;
#   such a theorem would be FALSE.
#   This blocker is IRRESOLVABLE — the algebra is what it is.
#
# BLOCKER C: No physical/MDL principle selects G_PS over other maximal subgroups.
#   Maximal subgroups of Spin(8) (Type 3: Dynkin 1952 Table 11; McKay-Patera 1981):
#     - SO(7) (dim 21)                    <= same dimension as G_PS!
#     - Sp(8) (dim 36, but contains Spin(8)? No — Sp(8) has larger dim)
#     Actually the maximal connected subgroups of SO(8) of dimension ≤ 28 include:
#     - SO(7) (dim 21)
#     - U(4) = SU(4) × U(1) (dim 16)
#     - SO(6) × SO(2) ≅ SU(4) × U(1) (dim 16)
#     - SO(5) × SO(3) ≅ Sp(4) × SU(2) (dim 13)
#     - SO(4) × SO(4) ≅ SU(2)^4 (dim 12)
#     - SU(4) × SU(2) × SU(2) = G_PS (dim 21, via SO(6)×SO(4) embedding)
#   Multiple 21-dimensional subgroups exist (SO(7) and G_PS both have dim 21).
#   No MDL argument from A1+A2+A3+A4 selects G_PS over SO(7).
#   Gate type needed: Type 1 (axiom) or Type 3 (theorem deriving PS from k=4 net
#   geometry). Neither is available.
#
# BLOCKER D: No geometry linking the k=4 crystal net to the specific PS embedding.
#   At k=3 (srs), the identification uses the GRAPH GEOMETRY:
#   - The srs chirality (I4_132) determines the SU(2)_L vs SU(2)_R distinction.
#   - The B-L generator = (2N-3)/3 arises from the FERMIONIC NUMBER operator,
#     which is directly the total edge-occupation N = n1+n2+n3.
#   - These identifications are forced by the specific srs geometry.
#   For k=4 (dia or other), no analogous geometric derivation has been attempted.
#   The dia space group (Fd-3m, No. 227) has different point symmetry (T_d)
#   from srs (I4_132); the Wyckoff-position structure is different.
#   Gate type needed: Type 2 (explicit algebraic derivation from dia geometry).

blockers = {
    "A": "MDL-optimal k=4 net not uniquely identified (no Type 3 uniqueness theorem).",
    "B": "Spin(8) ≇ G_PS; no exceptional isomorphism; irresolvable algebraic blocker.",
    "C": "No MDL/geometric principle selects G_PS over SO(7) or other dim-21 subgroups.",
    "D": "No geometric link between k=4 crystal net structure and PS embedding.",
}

for label, desc in blockers.items():
    print(f"  BLOCKER {label}: {desc}")


# ============================================================
# STEP 6: WHAT SPIN(8) DOES GIVE — TRIALITY AND GENERATIONS
# ============================================================
#
# GATE: PARTIALLY USEFUL (Type 3)
#
# Although Spin(8) ≇ G_PS, the Spin(8) structure has a different role in the
# framework: generating 3 generations via triality.
#
# Out(Spin(8)) = S_3 (Type 3: Baez 2002, §4).
# Triality permutes the three 8-dim representations of Spin(8):
#   8_v (vector), 8_s (left spinor), 8_c (right spinor).
# Each 8-dim representation contains one SM generation under suitable breaking.
# G_2 ⊂ Spin(8) acts identically on all three representations:
#   Spin(8) ⊃ G_2 × SU(2) (maximal subgroup, dim = 14 + 3 = 17)
#   Under G_2: 8 = 7 + 1 (adjoint + singlet) — the 7 is not yet SU(3).
#
# This connection between Spin(8) triality and 3 generations is referenced in:
#   - docs/theorems/theorem_41_screw_wigner.md §7 (Type 4)
#   - an internal sprint vs. external-research-note comparison (Type 4)
#
# HOWEVER: This triality argument does NOT close k=4 → PS.
# The triality structure is relevant for WHY n_gen = 3 (a different problem),
# not for WHY the k=4 gauge group is G_PS.
#
# The Cl(8) = Cl(6) ⊗ Cl(2) construction (proofs/gauge/cl8_verification.py, Type 4)
# builds Cl(8) via a TENSOR PRODUCT, not from 4 independent fermionic modes.
# In that construction:
#   - Cl(6): 3 Fock modes (k=3 srs edge qubits)
#   - Cl(2): orientation + causal direction (from G2 theorem, edge DOF)
# This is a DIFFERENT Cl(8) from "4 independent fermionic modes on a k=4 net."
# The two Cl(8) constructions may not be physically equivalent.

print("\n  Spin(8) triality gives 3 generations (n_gen=3 route)")
print("  But Spin(8) is NOT G_PS — different problem, different application.")
print("  Cl(8) = Cl(6) ⊗ Cl(2) [from srs + G2] ≠ Cl(8) [from 4 modes on k=4 net]")


# ============================================================
# STEP 7: THE ACTUAL STATUS OF k=4 → PATI-SALAM
# ============================================================
#
# The identification k=4 → G_PS in k_cooling_sm_uniqueness.py is an ADOPTION,
# not a theorem. It is labeled "ADOPTED (k=4 not MDL-derived)" in the code
# (k_cooling_sm_uniqueness.py line 168, Type 4).
#
# The adoption is based on the GUT hierarchy argument:
#   "The GUT cooling trajectory: k_max → k=5 (SU(5)) → k=4 (PS) → k=3 (SM)"
# This is a physical PATTERN (SU(5) ⊃ G_PS ⊃ G_SM), not a Clifford derivation.
#
# To close k=4 → G_PS at theorem grade analogously to k=3 → G_SM would require:
#
#   (i)   Uniqueness of k=4 MDL crystal net [BLOCKER A, missing Type 3]
#   (ii)  A group-theoretic selection of G_PS ⊂ Spin(8) [BLOCKER B+C, irresolvable]
#   (iii) A geometric link via the k=4 net's space group [BLOCKER D]
#
# ALTERNATIVE ROUTE (partially viable):
#
# There is one potential route that bypasses BLOCKER B:
#   Instead of asking "what group is Spin(8)?", ask:
#   "What is the group that commutes with the k=4 chirality operator?"
#
# At k=3 (srs), the chirality Gamma_7 splits the 8-dim Fock space as 4 + 4.
# The group acting on each 4-dim chiral sector is SU(4).
#
# At k=4 (hypothetical), the chirality Gamma_9 splits the 16-dim Fock space as 8 + 8.
# The group acting on each 8-dim chiral sector would be...
#
# Spin(8) restricted to one chiral half of the 16-dim spinor:
#   The 16-dim spinor of Cl(8) decomposes as 8_s + 8_c under chirality.
#   The subgroup of Spin(8) preserving 8_s is NOT a classical group.
#   8_s is the LEFT spinor representation, and the stabilizer of 8_s in Spin(8)
#   under triality is Spin(7) (dim=21).
#   Spin(7) ≅ SO(7) locally (they are the same Lie algebra).
#   SO(7) has dim 21 = dim(G_PS), but SO(7) ≇ G_PS.
#
# So the chiral-sector restriction gives Spin(7) ≅ SO(7), not G_PS.
# Another dim-21 subgroup, but not Pati-Salam.
#
# TYPE 3 source: Lawson-Michelsohn (1989) §I.5; Harvey (1990) Spinors and Calibrations.
#   "The stabilizer of a spinor in Spin(8) is isomorphic to G_2 (14-dim);
#    the stabilizer of a chiral (Weyl) spinor is isomorphic to Spin(7) (21-dim)."
#
# Therefore: even the chiral-sector restriction gives SO(7), not G_PS.

DIM_SO7  = 21
DIM_G2   = 14

assert DIM_SO7 == DIM_PS, "Coincidence: dim(SO(7)) = dim(G_PS) = 21"
# But SO(7) ≇ SU(4)×SU(2)×SU(2):
#   SO(7) is simple; G_PS is a product of three simple groups.
#   They have the same dimension but different Lie algebra structure.
# This is a coincidence, not an isomorphism.

print(f"\n  Chiral sector of Cl(8): Spin(7) (dim={DIM_SO7}), not G_PS (dim={DIM_PS})")
print(f"  dim(SO(7)) = dim(G_PS) = {DIM_PS} is a coincidence, NOT an isomorphism.")
print(f"  SO(7) is simple; G_PS = SU(4)×SU(2)×SU(2) is a product of three factors.")


# ============================================================
# STEP 8: GATE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("  GATE SUMMARY: k=4 → Cl(8) → Pati-Salam")
print("=" * 70)

gate_table = [
    ("Cl(8) from 4 modes via JW",      "Type 2+4", "PASS",    "Algebra; JW theorem (theorem_car_local_jordan_wigner.md)"),
    ("Spin(8) = Cl(8) bivectors",       "Type 2+3", "PASS",    "C(8,2)=28; SO(8), simple, rank 4, type D_4"),
    ("G_PS ⊂ Spin(8)",                  "Type 3",   "PASS",    "Standard GUT embedding; dim 21 < 28"),
    ("Uniqueness of k=4 crystal net",   "Type 3",   "BLOCKED", "No Sunada-type theorem for k=4"),
    ("Spin(8) ≅ G_PS",                  "Type 2",   "FALSE",   "Spin(8) is simple; G_PS is not. No exceptional isomorphism."),
    ("Selection of G_PS in Spin(8)",    "Type 1+2", "BLOCKED", "No MDL/geometric principle; SO(7) also dim=21"),
    ("Geometric link from k=4 net",     "Type 2",   "BLOCKED", "No derivation from dia/k=4 space group"),
    ("Chiral-sector restriction",       "Type 3",   "GIVES SO(7)", "Stabilizer of Weyl spinor is Spin(7)≅SO(7), not G_PS"),
]

for step, gate, status, note in gate_table:
    print(f"  {status:>12} | {gate:>10} | {step:<35} | {note}")

print("""
CONCLUSION:
  The k=4 → Pati-Salam identification CANNOT be closed at theorem grade
  by the G2-theorem approach.

  IRRESOLVABLE BLOCKER (BLOCKER B):
    Spin(8) is NOT isomorphic to G_PS.
    At k=3: Spin(6) ≅ SU(4)_PS (exceptional isomorphism, exact).
    At k=4: Spin(8) ⊋ G_PS (proper containment, no exceptional isomorphism).
    There is no equivalent of the "Spin(6)≅SU(4)" miracle at k=4.

  RESOLVABLE IN PRINCIPLE (BLOCKERS A, C, D):
    Require new uniqueness theorem, new MDL selection principle, and
    explicit geometric derivation. These are large open problems.

  STATUS OF k=4 → PS ADOPTION:
    Remains ADOPTED (GUT pattern argument, not Clifford derivation).
    Cannot be promoted to theorem without resolving all four blockers.

  WHAT Cl(8) FROM k=4 ACTUALLY GIVES:
    - Bivectors: Spin(8), dim=28, simple, type D_4.
    - Chiral sector: Spin(7)≅SO(7), dim=21 (NOT G_PS).
    - Triality: Out(Spin(8))=S_3, relevant for n_gen=3 (different problem).
    - No exceptional isomorphism mapping Spin(8) to G_PS.

  RECOMMENDATION:
    The k=4 → PS adoption should be retained as an adoption with explicit
    documentation of its status. Do NOT label it theorem-grade. The GUT
    hierarchy motivation is physically natural but algebraically ungrounded
    at the level of Clifford bivectors.
""")


# ============================================================
# STEP 9: THE CORRECT ROLE OF Cl(8) IN THIS FRAMEWORK
# ============================================================
#
# The Cl(8) that appears in the CLOSED framework is:
#   Cl(8) = Cl(6) ⊗ Cl(2)   [proofs/gauge/cl8_verification.py, Type 4]
#
# This is built from:
#   - Cl(6): 3 fermionic modes on the k=3 srs node (proven)
#   - Cl(2): edge orientation (f1) and causal direction (f2) (G2 theorem, proven)
#
# This Cl(8) acts on the 16-dim space = (8-dim Fock) ⊗ (2-dim edge qubit).
# The gauge content of THIS Cl(8) is:
#   - From Cl(6) sector: Spin(6) ≅ SU(4)_PS (Pati-Salam color) [proven, G2 chain]
#   - From Cl(2) sector: SU(2)_L (Higgs doublet) [proven, G2 theorem]
#   - Together: Cl(8) = Cl(6) ⊗ Cl(2) gives SU(4) × SU(2)_L at k=3.
#
# This is NOT the same as "k=4 gives G_PS".
# The Cl(8) = Cl(6) ⊗ Cl(2) construction lives entirely at k=3 and gives
# the PARTIAL Pati-Salam structure (SU(4) × SU(2)_L) at k=3, not at k=4.
#
# The SU(2)_R factor requires additional structure (right-handed chirality partner)
# from the causal-direction sector, which is being addressed in the Higgs VEV work.
#
# PUNCHLINE:
#   Pati-Salam structure (SU(4) × SU(2)_L × SU(2)_R) is NOT a k=4 phenomenon.
#   It is visible from within the k=3 framework via Cl(8) = Cl(6) ⊗ Cl(2):
#   - SU(4)_PS: from Cl(6) bivectors (Spin(6) ≅ SU(4))
#   - SU(2)_L:  from Cl(2) via G2 theorem
#   - SU(2)_R:  from the opposite chirality of the edge qubit (causal direction T-partner)
#
# This reframes the question: "k=4 → PS" may be WRONG AS STATED.
# The PS structure may already be present at k=3 via Cl(6) ⊕ Cl(2) = Cl(8),
# and the "k=4" adoption may be an artifact of the naive "k modes → Cl(2k)"
# identification without accounting for the proper Cl(2) origin.

print("\n  IMPORTANT REFRAMING:")
print("  Cl(8) = Cl(6) ⊗ Cl(2) [at k=3] already gives SU(4) × SU(2)_L [proven].")
print("  SU(2)_R comes from the T-partner sector in the same Cl(8).")
print("  The PS structure may be a k=3 phenomenon, not a k=4 phenomenon.")
print("  The 'k=4 → PS' adoption may be a category error.")

# ============================================================
# REFERENCES
# ============================================================
#
# [Sunada 2012]       Sunada T. "Crystals That Nature Might Miss Creating."
#                     Notices AMS 59(2), 208-215 (2012).
#                     [srs uniqueness at k=3; no k=4 analogue]
#
# [Delgado 2003]      Delgado-Friedrichs O, O'Keeffe M. "Identification of
#                     and symmetry computation for crystal nets."
#                     Acta Crystallogr. A 59, 351-360 (2003).
#                     [Crystal net enumeration; k ≥ d for d-dim nets]
#
# [RCSR]              O'Keeffe M et al. "The Reticular Chemistry Structure
#                     Resource (RCSR) Database of, and Symbols for, Crystal
#                     Nets." Acc. Chem. Res. 41, 1782-1789 (2008).
#                     [dia net: vertex-transitive, edge-transitive, 4-regular,
#                      2 atoms per primitive cell, space group Fd-3m]
#
# [Baez 2002]         Baez JC. "The Octonions." Bull. AMS 39, 145-205 (2002).
#                     math/0105155. §4.
#                     [Out(Spin(8)) = S_3; triality; G_2 ⊂ Spin(8)]
#
# [Adams 1969]        Adams JF. Lectures on Lie Groups. Benjamin/Cummings, 1969.
#                     [Spin(8) structure; D_4 root system; simple Lie groups]
#
# [Lawson 1989]       Lawson HB, Michelsohn ML. Spin Geometry. Princeton, 1989.
#                     §I.5, Table 1.
#                     [Clifford algebra classification; Cl(8,0) ≅ M_16(R);
#                      spinor representations; stabilizer of Weyl spinor = Spin(7)]
#
# [Harvey 1990]       Harvey FR. Spinors and Calibrations. Academic Press, 1990.
#                     [Spin(7) as stabilizer of a generic spinor in 8d]
#
# [Dynkin 1952]       Dynkin EB. "Maximal subgroups of the classical groups."
#                     Trudy MMO 1, 39-166 (1952). [AMS Transl. (2) 6, 245-378 (1957)]
#                     [Classification of maximal subgroups of SO(8);
#                      embedding SU(4)×SU(2)×SU(2) ⊂ SO(8)]
#
# [Slansky 1981]      Slansky R. "Group theory for unified model building."
#                     Phys. Rep. 79, 1-128 (1981).
#                     [GUT group embeddings; PS ⊂ SO(10); PS ⊂ Spin(8) embedding]
#
# [Pati 1974]         Pati JC, Salam A. "Lepton number as the fourth color."
#                     Phys. Rev. D 10, 275 (1974).
#                     [Original Pati-Salam group; SU(4)×SU(2)_L×SU(2)_R]
