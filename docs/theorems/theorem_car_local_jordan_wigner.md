# Local CAR from Jordan-Wigner — theorem (Stage 4 path iii)

**Date:** 2026-04-21 (Session 11).
**Status:** THEOREM — gate-passing under `../parameters/parameter_linter.md`. Every load-bearing step is Type 1 (axiom), Type 2 (explicit algebra), or Type 3 (cited theorem). The CAR proof is fully explicit 2×2 matrix arithmetic; Jordan-Wigner 1928 is credited for the string construction but the CAR relations are derived, not cited.
**Scope:** narrow. At each k*-valent node of srs, the k* toggle operators from A1, lifted to operators on the complex Hilbert space from A3, admit an explicit Jordan-Wigner construction whose output operators satisfy the canonical anticommutation relations of A4. A4 therefore need not be postulated as an axiom; it is a consequence of A1 + A3. The ordering used in the JW construction is a gauge (any of the k*! orderings gives isomorphic CAR structure); this is not an adoption.
**Scope limit:** this is LOCAL A4 at each node independently. Global CAR across distinct nodes of the srs lattice requires a consistent total ordering of all nodes, which is the B1 ordering workstream. Stage 4 path (iii) closes local A4; full A4 elimination requires B1.
**Distinguishes from:** an internal working note, which attempted A4 from A1+A2 and was BLOCKED. That attempt was blocked because A1+A2 produce a commutative Z/2-graded product algebra with no anticommutation sign. The present theorem uses A1+A3: A3 (CDP 2011) supplies the complex Hilbert space that makes σ^y and σ^± well-defined, which is the missing ingredient for the JW construction.

**Post-2026-05-08 axiom slate note.** A1 and A3-T (cited as inputs throughout this theorem) are now derived theorems of the new top-level slate: A1 from `theorem_toggle_from_self_containment.md` (under (A) self-containment + (B) finite observer + Shannon-Jaynes + (I) active reading); A3-T from `theorem_A3_complex_hilbert_from_multiway.md` (under the same top-level commitments). References to "A1 + A3" remain semantically valid; the JW construction and CAR derivation are unchanged. See `framework_axioms.md` §10 for the updated top-level summary.

---

## 1. Theorem statement

**Theorem (Local CAR from Jordan-Wigner).** Let v be any k*-valent node of the srs lattice, with k* = 3 directed edge modes e_1, e_2, e_3 incident to v. Under A1 + A3:

(i) The local Hilbert space at v is H_v = (C²)^⊗3, with the toggle operator for each edge mode acting as the bit-flip operator on the corresponding C² factor.

(ii) For any total ordering of the three edge modes, the Jordan-Wigner operators

    c_j = (σ^z_{e_1} ⊗ ··· ⊗ σ^z_{e_{j-1}}) ⊗ σ^-_{e_j} ⊗ I ⊗ ···

satisfy the canonical anticommutation relations:

    {c_j, c_j†} = I,    {c_i, c_j†} = 0  for i ≠ j,    {c_i, c_j} = 0  for i ≠ j.

(iii) The Majorana operators γ_{2j-1} = c_j + c_j† and γ_{2j} = i(c_j† − c_j) satisfy

    {γ_a, γ_b} = 2 δ_{ab} I    for a, b = 1, …, 2k* = 6.

(iv) The operators γ_1, …, γ_6 generate the Clifford algebra Cl(2k*) = Cl(6) on H_v, independently of which ordering was chosen.

**Corollary.** A4's assertion — that to each directed edge mode e is associated an operator γ_e satisfying {γ_e, γ_e'} = 2 δ_{ee'} I — holds at every node of srs as a theorem derivable from A1 + A3. A4 is not an independent axiom; it is a consequence.

---

## 2. Axioms and cited upstream

**Framework axioms:**
- **A1** (`../framework/framework_axioms.md` §2) — binary self-inverse toggle: T_e^2 = I for each directed edge e.
- **A3-T** (derived theorem; `theorem_A3_complex_hilbert_from_multiway.md`) — MDL canonicalization is a partial trace over an abstract purifying auxiliary; the visible-sector state space is a complex Hilbert space (CDP 2011 purification axiom). (Demoted from axiom A3 to derived theorem 2026-04-26.)

**Type 3 citation:**
- **Chiribella, D'Ariano, Perinotti** (2011). Informational derivation of quantum theory. *Phys. Rev. A* 84, 012311. Theorem 25 (§VIII): under A3's five CDP axioms, the state space is forced to be the density operators on a finite-dimensional **complex** Hilbert space. Load-bearing for the existence of σ^y (complex unit i in the 2×2 matrix algebra).

**Construction credit (not a load-bearing citation):**
- **Jordan, P., Wigner, E.** (1928). Über das Paulische Äquivalenzverbot. *Z. Phys.* 47, 631–651. Source of the string construction c_j = S_j σ^-_j. The CAR proof below is explicit algebra; JW 1928 is credited for the construction idea, not cited as a theorem we invoke.

**Distinguishes from:**

---

## 3. Setup (Type 1 + Type 3)

By A1, each directed edge mode e incident to node v carries a binary toggle operator T_e with T_e^2 = I.

By A3 + CDP 2011 Theorem 25, the local state space of the k* edge modes at node v is a finite-dimensional complex Hilbert space. Because each edge mode is an independent two-state system under A1 (distinct toggle generators — different edges, no shared dynamics at the toggle level), and because CDP axiom 4 (local distinguishability) forces the joint state space of independent systems to be their tensor product, the local Hilbert space at v is:

    H_v = (C²)^⊗k*

with orthonormal basis {|0_j⟩, |1_j⟩} at each factor j (|0_j⟩ = edge e_j off, |1_j⟩ = edge e_j on).

The toggle operator T_{e_j} flips edge e_j between |0_j⟩ and |1_j⟩, acting as identity on all other factors. In the {|0_j⟩, |1_j⟩} basis this is the 2×2 bit-flip matrix σ^x:

    T_{e_j} = σ^x_j ⊗ I_rest,    σ^x = [[0, 1], [1, 0]].

Since H_v is a complex Hilbert space (CDP 2011), the standard Pauli operators at each site are all well-defined 2×2 complex matrices:

    σ^z_j = [[1, 0], [0, -1]]  (number operator: +1 for |1⟩, -1 for |0⟩)
    σ^+_j = [[0, 1], [0, 0]]   (raising: |0⟩ → |1⟩)
    σ^-_j = [[0, 0], [1, 0]]   (lowering: |1⟩ → |0⟩)

These satisfy σ^± = (σ^x ± iσ^y)/2. The imaginary unit i is available because H_v is complex (A3 + CDP 2011 Theorem 25); this is the step that was unavailable under A1+A2 alone.

---

## 4. Key same-site algebra (Type 2)

The following identities follow by direct 2×2 matrix multiplication. No citation is required.

**F1.** {σ^+, σ^-} = σ^+ σ^- + σ^- σ^+ = I.

Proof: σ^+ σ^- = [[0,1],[0,0]][[0,0],[1,0]] = [[1,0],[0,0]].
       σ^- σ^+ = [[0,0],[1,0]][[0,1],[0,0]] = [[0,0],[0,1]].
       Sum = [[1,0],[0,1]] = I. □

**F2.** (σ^z)^2 = I.

Proof: [[1,0],[0,-1]]^2 = [[1,0],[0,1]] = I. □

**F3.** σ^z σ^- = −σ^-.

Proof: [[1,0],[0,-1]][[0,0],[1,0]] = [[0,0],[-1,0]] = −σ^-. □

**F4.** σ^- σ^z = +σ^-.

Proof: [[0,0],[1,0]][[1,0],[0,-1]] = [[0,0],[1,0]] = +σ^-. □

**F3 and F4 together:** σ^- and σ^z anticommute at the same site: {σ^-, σ^z} = 0. The sign flip between F3 and F4 is the entire mechanism of the Jordan-Wigner transformation.

**F5.** Operators at distinct sites commute: [σ_i^α, σ_j^β] = 0 for i ≠ j.

Proof: They act on different tensor factors of H_v. □

---

## 5. Jordan-Wigner construction (Type 2)

Choose any total ordering of the k* = 3 edge modes: e_1, e_2, e_3. Define the string operator at position j:

    S_j = σ^z_{e_1} ⊗ ··· ⊗ σ^z_{e_{j-1}} ⊗ I_{e_j} ⊗ ··· ⊗ I_{e_{k*}}

(empty product S_1 = I). Define the Jordan-Wigner annihilation operator at position j:

    c_j = S_j · σ^-_{e_j}

and its adjoint c_j† = S_j · σ^+_{e_j} (since S_j is Hermitian: S_j† = S_j, using (σ^z)† = σ^z).

Note: S_j^2 = I by F2, and S_j commutes with σ^±_{e_j} since they act on different factors (F5).

---

## 6. Proof of {c_j, c_j†} = I (Type 2)

    c_j c_j† = (S_j σ^-_j)(S_j σ^+_j) = S_j^2 σ^-_j σ^+_j = σ^-_j σ^+_j

    c_j† c_j = (S_j σ^+_j)(S_j σ^-_j) = S_j^2 σ^+_j σ^-_j = σ^+_j σ^-_j

    {c_j, c_j†} = σ^-_j σ^+_j + σ^+_j σ^-_j = {σ^-, σ^+}_j = I    (by F1). □

---

## 7. Proof of {c_i, c_j†} = 0 for i < j (Type 2)

Write S_j = S_i · σ^z_{e_i} · T_{ij}, where T_{ij} = ∏_{i < k < j} σ^z_{e_k} (empty product = I if j = i+1). T_{ij} acts on sites strictly between i and j; S_i acts on sites before i. All of S_i, σ^z_{e_i}, T_{ij}, σ^±_{e_j} act on different sites and commute with each other (F5), except that σ^z_{e_i} and σ^-_{e_i} share site i.

    c_i c_j† = (S_i σ^-_{e_i})(S_i σ^z_{e_i} T_{ij} σ^+_{e_j})
             = S_i^2 (σ^-_{e_i} σ^z_{e_i}) T_{ij} σ^+_{e_j}
             = +σ^-_{e_i} · T_{ij} · σ^+_{e_j}     [using F4: σ^- σ^z = +σ^-]

    c_j† c_i = (S_i σ^z_{e_i} T_{ij} σ^+_{e_j})(S_i σ^-_{e_i})
             = S_i^2 (σ^z_{e_i} σ^-_{e_i}) T_{ij} σ^+_{e_j}
             = −σ^-_{e_i} · T_{ij} · σ^+_{e_j}     [using F3: σ^z σ^- = −σ^-]

    {c_i, c_j†} = +σ^-_{e_i} T_{ij} σ^+_{e_j} − σ^-_{e_i} T_{ij} σ^+_{e_j} = 0. □

The sign flip between the two terms comes entirely from F3 vs F4 at site i. The JW string is constructed precisely to produce this cancellation.

---

## 8. Proof of {c_i, c_j} = 0 for i < j (Type 2)

Identical argument with σ^+_{e_j} replaced by σ^-_{e_j} throughout:

    c_i c_j = S_i^2 (σ^-_{e_i} σ^z_{e_i}) T_{ij} σ^-_{e_j} = +σ^-_{e_i} T_{ij} σ^-_{e_j}

    c_j c_i = S_i^2 (σ^z_{e_i} σ^-_{e_i}) T_{ij} σ^-_{e_j} = −σ^-_{e_i} T_{ij} σ^-_{e_j}

    {c_i, c_j} = 0. □

By taking adjoints, {c_i†, c_j†} = 0 follows immediately.

---

## 9. Majorana operators and Cl(6) (Type 2)

Define 2k* = 6 Majorana operators:

    γ_{2j-1} = c_j + c_j†,    γ_{2j} = i(c_j† − c_j),    j = 1, 2, 3.

Each γ_a is Hermitian (γ_a† = γ_a) by construction. From the CAR relations of §§6–8:

    {γ_a, γ_b} = 2 δ_{ab} I    for all a, b = 1, …, 6.

Verification for representative cases (Type 2):

- a = b = 2j−1: {γ_{2j-1}, γ_{2j-1}} = 2(c_j + c_j†)^2 evaluated: (c_j + c_j†)^2 = c_j^2 + {c_j, c_j†} + (c_j†)^2 = 0 + I + 0 = I, so {γ_{2j-1}, γ_{2j-1}} = 2I. ✓

- a = 2i−1, b = 2j−1 with i < j: {c_i + c_i†, c_j + c_j†} = {c_i, c_j} + {c_i, c_j†} + {c_i†, c_j} + {c_i†, c_j†} = 0 + 0 + 0 + 0 = 0. ✓

- Mixed (a = 2j−1, b = 2j): {γ_{2j-1}, γ_{2j}} = i{c_j + c_j†, c_j† − c_j} = i({c_j, c_j†} − {c_j, c_j} + {c_j†, c_j†} − {c_j†, c_j}) = i(I − 0 + 0 − I) = 0. ✓

The operators γ_1, …, γ_6 satisfy {γ_a, γ_b} = 2δ_{ab} I, which is the defining relation of a Clifford algebra on 6 generators — i.e., Cl(6). They act irreducibly on H_v = (C²)^⊗3 (dimension 8 = 2^3 = 2^{k*}), which is exactly the dimension of the unique irreducible representation of Cl(6) over C. The Clifford structure is independent of which ordering of the k* edge modes was chosen: all k*! = 6 orderings give unitarily equivalent sets of Majorana operators, hence isomorphic Cl(6) algebras.

### 9.1 Cl(6) chirality γ_7 — single-vertex vs walker-level realizations (added 2026-05-01 EOD)

The Cl(6) chirality element γ_7 := i · γ_1 γ_2 γ_3 γ_4 γ_5 γ_6 is Hermitian with γ_7² = I, and acts on H_v = (C²)^⊗3 as the fermion-number parity (−1)^F. Its restriction to the walker's F=1 single-fermion sector is therefore CONSTANT (= −1) at the per-vertex level — γ_7 alone does NOT provide a non-trivial Z_2 grading on the walker (per `proofs/foundations/srs_z_chi_layer5_cl6_relationship.py`).

However, the **half-bipartite product** γ_7^A := Π_{u ∈ A} γ_7_u over the A-side vertices of a bipartite primitive quotient gives a NON-TRIVIAL walker-level Z_2 grading on substrates whose primitive cell is bipartite. Specifically, for a walker basis vector |v⟩ with F_v = 1 and F_u = 0 for u ≠ v:

  γ_7^A |v⟩ = (+1)·(−1)^{|A|−1} |v⟩ if v ∈ A;  (−1)^{|A|} |v⟩ if v ∈ B.

For srs-z's bipartition |A| = |B| = 4 this evaluates to γ_7^A|_walker = −χ̃, where χ̃ is the bipartite chirality on directed arcs (`proofs/foundations/srs_z_gamma7_lift_recovers_chi.py`). Therefore the framework has ONE Cl(6) Z_2 chirality with TWO realizations:

- **Per-vertex on any substrate** — trivial on walker F=1 sector.
- **Walker-level on bipartite-primitive substrates** — via γ_7^A, equal to ±χ̃ where χ̃ is the bipartite chirality.

On non-bipartite substrates (srs's K_4, srs-c4/c8/c27, lou, okw — per `proofs/foundations/rcsr_candidate_sweep.py` 2026-05-01 post-EOD), γ_7^A has no canonical lift (no canonical "side A"); the walker-level Z_2 simply doesn't exist. On bipartite-primitive substrates ({srs-z, lov} per the same sweep), γ_7^A → ±χ̃ and the walker carries a Z_2 supercharge structure (anti-commutes with B(k) at all k; algebraically the structure of N=1 SUSY pairing — see `proofs/foundations/srs_z_pati_salam_chi_commutation.py` for PS × χ̃ commutation).

Forward-link: this construction is the foundation for the χ̃ unification + SUSY-pair phenomenology developed in an internal working note.

---

## 10. Parameter_linter gate summary

| Step | Claim | Gate type | Source |
|---|---|---|---|
| §3 H_v = (C²)^⊗3 | Local Hilbert space is tensor product of k* two-state systems | Type 1 + Type 3 | A1 (independent 2-state toggles) + A3 + CDP 2011 Theorem 25 (complex Hilbert space, local distinguishability → tensor product) |
| §3 T_{e_j} = σ^x_j ⊗ I_rest | Toggle acts as bit-flip on its factor | Type 1 + Type 2 | A1 (T_e flips edge e only) + matrix representation |
| §3 σ^y available | Complex unit i exists in the 2×2 algebra | Type 3 | CDP 2011 Theorem 25 (complex field forced) |
| §4 F1–F5 | Same-site Pauli algebra + cross-site commutativity | Type 2 | Explicit 2×2 matrix multiplication |
| §5 JW definition c_j = S_j σ^-_j | Construction | Type 2 | Definition |
| §6 {c_j, c_j†} = I | CAR at same mode | Type 2 | F1 + S_j^2 = I + §4 |
| §7 {c_i, c_j†} = 0 for i < j | CAR across modes | Type 2 | F3 vs F4 sign flip — the entire mechanism |
| §8 {c_i, c_j} = 0 for i < j | Anti-symmetric products vanish | Type 2 | Same as §7 |
| §9 Majorana {γ_a, γ_b} = 2δ_{ab} I | CAR in Majorana form | Type 2 | Definition + §§6–8 |
| §9 generates Cl(6) | Clifford algebra on H_v | Type 2 | Definition of Clifford algebra from generating set satisfying {γ_a, γ_b} = 2δ_{ab} I |
| §9 ordering-independence | All orderings give isomorphic Cl(6) | Type 2 | Clifford algebra is determined up to isomorphism by the quadratic form, independent of generator ordering |

**All steps gate-passing.** No Type 3 citation is required for the CAR proof itself — it is pure 2×2 matrix arithmetic. CDP 2011 Theorem 25 is load-bearing only for the complex Hilbert space structure (§3), which was already load-bearing in the framework under A3.

---

## 11. What this theorem closes

- **A4 is not an independent axiom.** The CAR operators γ_e required by A4 are explicitly constructible from A1 + A3 at each k*-valent node. A4 does not add new content beyond A1 + A3 at the local node level.
- **Local Cl(6) is derived.** The Clifford algebra Cl(6) = Cl(2k*) acts on H_v = (C²)^⊗3 as a consequence of A1 + A3, for any ordering of the k* = 3 edge modes.
- **The anticommutation sign is derived, not adopted.** The sign in {c_i, c_j†} = 0 for i ≠ j comes from the F3 vs F4 asymmetry (σ^z σ^- = −σ^- but σ^- σ^z = +σ^-), which is a fact about 2×2 complex matrices. It does not require A4.
- **The previous BLOCKED route is bypassed.** The MDL-Fock attempt failed because A1+A2 produce a commutative toggle algebra. A3 supplies the complex structure needed for σ^± and the JW string, resolving that blockage without the MDL formalization gap.
- **Stage 4 axiom-elimination arc advanced.** CAR at each node is a derived consequence of A1+A3. The axiom count can be reduced from five (A1–A5) to four (A1, A2, A3, A5) at the local level.

---

## 12. What this theorem does NOT close

- **Global CAR across distinct nodes.** The CAR relations §§6–8 hold at a single node independently. Ensuring that γ_e at node v and γ_e' at a different node v' also anticommute requires a consistent total ordering of all edge modes across the full srs lattice simultaneously — a global Jordan-Wigner string. This is the B1 ordering workstream (`../../predictions/theorem_B1_ordering_derivation.md`). Without B1, local CAR does not extend to global CAR.
- **Canonical identification of γ_e with physical fermions.** The k*! = 6 orderings at each node give 6 unitarily equivalent but concretely distinct sets of Majorana operators. Choosing which one corresponds to which Standard Model fermion generation requires either B1 (canonical spatial ordering) or A5 (physical identification). This theorem establishes existence, not canonical identification.
- **B3 (spinor-fermion identification).** B3 requires Cl(6) globally across the K4 quotient, with a specific association of generators to SM fermion representations. That requires global CAR + B1 + A5.
- **A3 is still load-bearing.** This theorem uses A3 essentially. Without the complex Hilbert space from A3 + CDP 2011, σ^y and σ^± are not defined and the JW construction does not work. This theorem eliminates A4 but not A3.

---

## 13. Honesty

**On the ordering gauge.** The JW construction requires choosing an ordering of the k* = 3 edge modes at each node. This is a gauge choice, not an adoption: all k*! orderings produce unitarily equivalent CAR structures and the same Clifford algebra Cl(6). The theorem statement is true for every ordering; no specific ordering is selected or privileged. This is consistent with an internal working note §3.3's finding that "ALL 6 choices give a valid JW transformation; the ambiguity is gauge-equivalent."

**On the distinction from the prior attempt.** The prior attempt targeted A1+A2 → A4 and was correctly BLOCKED: A1+A2 produce a commutative Z/2-graded product algebra, and the anticommutation sign is not derivable from commutativity alone. The present theorem targets A1+A3 → A4, which works because A3 (complex Hilbert space from CDP 2011) makes σ^y and σ^± available. The gap in the prior attempt (§4.2: "to introduce the JW sign... is a construction superimposed on the toggle algebra") is resolved by A3 supplying the complex structure. The construction is no longer "superimposed" — it is built from well-defined operators that A3 guarantees exist.

**On the JW citation.** Jordan & Wigner 1928 originated the string construction c_j = S_j σ^-_j. The proof that this construction gives CAR is explicit algebra (§§6–8), carried through in full above. We credit JW 1928 for the idea; we do not cite it as a theorem we appeal to.

**On A4's remaining role.** A4 as stated in `../framework/framework_axioms.md` §5 includes both the algebraic content (CAR operators exist) and the physical identification (these operators ARE the edge modes that appear in B3). This theorem closes the algebraic content. The physical identification is downstream of B3 and A5 and is not claimed here.

---

## 14. Axiom elimination roadmap update

| Stage | Target | Status after this theorem |
|---|---|---|
| Stage 1 | Branch measure μ | CLOSED (session 9) |
| Stage 2a | Edge-surprise thresholds | CLOSED (session 10) |
| Stage 2c | Energy functional + arrow of time | CLOSED (session 10) |
| Stage 3 | Lorentz invariance (causal sector) | CLOSED (session 10) |
| **Stage 4 path (iii)** | **Local CAR at each node** | **CLOSED (this theorem)** |
| Stage 4 full (A4 elimination) | Global CAR + B1 ordering | Needs B1 |
| Stage 5 (A3 elimination) | Observer MDL-residue → purification | Blocked on Stage 4 full |
| Stage 6 (A5 relocation) | Empirical → observer-specification | Blocked on Stage 5 |

---

## 15. References

### Framework axioms
- `../framework/framework_axioms.md` §2 (A1), §4 (A3).
- `../framework/framework_axioms.md` §5 (A4): canonical statement of what is being derived.

### Type 3 citation (load-bearing)
- **Chiribella, G., D'Ariano, G.M., Perinotti, P.** (2011). Informational derivation of quantum theory. *Phys. Rev. A* 84, 012311. Theorem 25 (§VIII): five operational axioms force density operators on a finite-dimensional complex Hilbert space. Load-bearing for §3's complex Hilbert space structure (complex field forced, hence σ^y and σ^± are well-defined).

### Construction credit (not a load-bearing citation)
- **Jordan, P., Wigner, E.** (1928). Über das Paulische Äquivalenzverbot. *Z. Phys.* 47, 631–651. Source of the string construction c_j = S_j σ^-_j. The CAR proof in §§6–8 is independent algebra.

### Related framework documents
- `theorem_lorentz_causal_sector.md` (Stage 3) — cross-edge independence (§4.3) supports the tensor product structure of §3 here.
- `../../predictions/theorem_B1_ordering_derivation.md` — B1.b: no MDL-canonical global edge ordering. The local ordering gauge in §5 is consistent with B1.b (all orderings are gauge-equivalent; Clifford structure is formulated invariantly via the CAR relation, per Lawson-Michelsohn 1989 Ch. 1 §1).

### Not load-bearing here
- Stage 2c / Stage 3 — referenced in §3 for cross-edge independence, but the CAR derivation works for any independent 2-state systems; Stage 2c and Stage 3 are not strictly necessary for the algebraic result.

---

## 16. Status

**THEOREM (rigor: closed under parameter_linter.md hard gate).** Every load-bearing step annotated. No fabricated citations; the CAR proof is explicit 2×2 matrix arithmetic. No adoptions: the JW ordering is a gauge (all choices give isomorphic Cl(6)), not a physical postulate.

**Axiom count update:** Under A1 + A2 + A3 + A5, A4 is a derived consequence at the local node level. The framework's effective axiom list for local node physics is A1, A2, A3, A5 (four axioms). Full global A4 (across nodes) still requires B1.
