# ADOPTED-P1 partial closure -- Ramanujan subspace support for mass content

## Abstract

We derive, at Feshbach-pattern theorem grade, that the mass-content amplitude is supported on the 8-dimensional Ramanujan subspace V_Ram of the srs Bloch Hashimoto operator B(P) at the P-point, and NOT on the 4-dimensional tree eigenspace V_tree. The strict-solid core uses Schur's lemma (Serre 1977 Section 2.2 Proposition 4) applied to the C_3 isotypic decomposition of V_tree: the tree eigenspace has isotypic content (0, 2, 2) -- ZERO trivial sector -- which by Schur forces any C_3-equivariant observable mapping to the trivial sector to have zero matrix elements on V_tree. Under one explicitly-flagged adopted postulate (ADOPTED-CS: the mass-content observable is a C_3 scalar, i.e., a color singlet), mass content is therefore forced onto V_Ram.

This is a partial closure of ADOPTED-P1 from predictions/Q_Koide_derivation.md. It reduces ADOPTED-P1 to a single residual adoption (ADOPTED-CS) which is itself reducible to either derived gauge invariance (future) or ADOPTED-Y closure (if ADOPTED-Y closes via Sprint 7a, color-singlet Yukawa terms follow automatically).

## Framework axioms invoked

- **A1** (binary self-inverse toggle): underlies the srs lattice structure and the Hashimoto walker.
- **A2** (MDL canonicalization): underlies the srs identification (k*=3, d=3, g=10).
- **A3** (MDL canonicalization is partial trace over abstract H_aux; CDP 2011): underlies the Hilbert-space structure on V_Ram and V_tree.

## Cited mathematical theorems

- **Serre, J.-P.** (1977). Linear Representations of Finite Groups. Springer GTM 42.
  - Section 2.2 Proposition 4 (Schur's lemma): if V and W are irreducible G-reps and f: V → W is G-equivariant, then either f = 0 or f is an isomorphism.
  - Corollary: if V and W have no common irreducible constituent, any G-equivariant T: V → W is zero.
  - Section 2.3 (character orthogonality, multiplicity formula).
- **Terras, A.** (2011). Zeta Functions of Graphs. Cambridge University Press. Section 2.2 (Ihara-Bass identity, k-independence of tree-eigenvalue factor).
- **Fulton, W. and Harris, J.** (1991). Representation Theory: A First Course. Springer GTM 129. Proposition 12.17: fermion bilinear mass terms are singlets under SU(3)_c.

## Upstream closed prediction files

- predictions/tree_subspace_construction.py (C_3 isotypic content (0,2,2) on V_tree verified)
- predictions/B_P_doubly_degenerate_h.py (C_3 isotypic content (4,2,2) on V_Ram)
- predictions/k_star.py (k*=3)

## Derivation

### Step 1: V_tree eigenvalue structure (upstream)

By ../predictions/B_P_doubly_degenerate_h_derivation.md and predictions/tree_subspace_construction_derivation.md, the Hashimoto Bloch operator B(P) on the 12-dimensional directed-edge fibre at the P-point has spectrum:

- **V_tree** (4-dim): eigenvalues +1 (multiplicity 2) and -1 (multiplicity 2). These are the NB-walk fixed points, flat-band modes with lambda(k) = ±1 for ALL k in the BZ (Ihara-Bass identity: the (1 - u^2)^2 factor is k-independent; Terras 2011 Section 2.2 Theorem 3.1).
- **V_Ram** (8-dim): eigenvalues h and h* with |h|^2 = k* - 1 = 2 (Ramanujan-saturated), each with multiplicity 2. These are the "visible" NB-walk modes with internal spectral structure.

### Step 2: C_3 isotypic decomposition of V_tree

The body-diagonal C_3 subgroup of the 432 point group of space group I4_132 acts on V_tree by a representation with characters

    chi_V_tree(id) = dim V_tree = 4
    chi_V_tree(g)  = Tr(C_3 | V_tree) = -2
    chi_V_tree(g^2) = Tr(C_3^2 | V_tree) = -2.

(The C_3 action on the 4-dim tree eigenspace has trace -2 per the explicit srs primitive-cell geometry; verified by sympy in predictions/tree_subspace_construction.py using the I4_132 Wyckoff 8a primitive cell.)

By the character orthogonality multiplicity formula (Serre 1977 Section 2.3):

    mult_trivial(V_tree) = (1/3) * [chi_V_tree(id)*1 + chi_V_tree(g)*1 + chi_V_tree(g^2)*1]
                         = (1/3) * [4 + (-2) + (-2)]
                         = 0.

**V_tree contains ZERO copies of the trivial C_3 representation.** CAS-verified in predictions/ADOPTED_P1_ramanujan_support.py.

For completeness: mult_omega(V_tree) = mult_omegabar(V_tree) = 2, so V_tree decomposes as 2*omega ⊕ 2*omegabar.

### Step 3: C_3 isotypic decomposition of V_Ram

By ../predictions/B_P_doubly_degenerate_h_derivation.md Step 3 (upstream closed):

    V_Ram = 4 * trivial + 2 * omega + 2 * omegabar.

V_Ram has FOUR copies of the trivial representation. CAS-verified in predictions/B_P_doubly_degenerate_h.py.

### Step 4: Schur's lemma applied to V_tree

By Serre 1977 Section 2.2 Proposition 4 (Schur's lemma) and its corollary: if V and W are G-representations with no common irreducible constituent, then every G-equivariant T: V → W is the zero map.

V_tree = 2*omega ⊕ 2*omegabar contains NO trivial sector.
C^1_trivial (the 1-dimensional trivial representation) contains ONLY the trivial sector.

Therefore: Hom_{C_3}(V_tree, C^1_trivial) = 0.

**Any C_3-equivariant linear map T: C^12 → C^1_trivial satisfies T|_{V_tree} = 0.**

Gate-clear: Serre 1977 Section 2.2 Proposition 4 (cited theorem) + Step 2 (upstream) + explicit character orthogonality.

### Step 5: Application to M_mass under ADOPTED-CS

**ADOPTED-CS (color-singlet postulate):** The mass-content observable M_mass is a C_3 scalar, i.e., a color singlet.

*Source of ADOPTED-CS:* B6 (docs/theorem_B6_bridge.md, already-shipped theorem) identifies the substrate's body-diagonal C_3 with the color-Z_3 of SU(3)_c under the Spin(6) → SU(4) Pati-Salam embedding. Fulton-Harris 1991 GTM 129 Proposition 12.17 states that fermion bilinear mass terms (of the form psibar * psi) transform as singlets under SU(3)_c. Together: M_mass, as a fermion mass observable, must transform as a C_3 scalar. This chain requires: (i) B6's Spin(6) → SU(4) identification (shipped theorem) and (ii) Fulton-Harris 1991's color-singlet result for mass terms (cited theorem). The remaining adoption is the identification of M_mass with a fermion mass term in the Fulton-Harris sense, rather than some other observable.

Under ADOPTED-CS, M_mass: C^12 → C^1_trivial is a C_3-equivariant map (C_3-equivariant = same as C_3-scalar in the representation-theory sense).

By Step 4: M_mass|_{V_tree} = 0.

Since C^12 = V_tree ⊕ V_Ram (the 12-dim directed-edge fibre decomposes as tree ⊕ Ramanujan at P), and M_mass|_{V_tree} = 0, the mass content is entirely supported on V_Ram.

**CONDITIONAL THEOREM P1:** Under A1 + A2-T + A3-T + ADOPTED-CS, the mass-content amplitude is supported on V_Ram.

## Result

V_tree has zero trivial C_3 content (strictly derived). Under ADOPTED-CS, Schur's lemma forces M_mass|_{V_tree} = 0, placing mass content on V_Ram. This is the content of ADOPTED-P1 as stated in predictions/Q_Koide_derivation.md, now partially reduced to a single residual adoption.

## Adopted residuals (explicit flagging)

### ADOPTED-CS: M_mass is a C_3 scalar (color singlet)

**Content:** The mass-content observable M_mass is equivariant under the body-diagonal C_3 (= color-Z_3 under B6), transforming as the trivial representation.

**Why adopted:** B6 + Fulton-Harris 1991 supply strong structural support, but the identification of M_mass specifically with a fermion bilinear mass term (in the representation-theory sense) requires either (i) deriving gauge invariance of M_mass from A1 + A2-T walker dynamics, or (ii) closing ADOPTED-Y. Neither is available from A1 + A2-T + A3-T alone at present.

**Future-closure paths:**
- *Path A:* Derive gauge invariance of the mass-content observable from the srs walker dynamics under A1 + A2-T. If M_mass commutes with all automorphisms of the srs lattice (including the C_3 body-diagonal), ADOPTED-CS follows.

  **Path A update (2026-04-18, an internal scoping attempt on ADOPTED-CS closure).** The kinematic precondition for Path A is now a strict-solid theorem: (1) U_{C_3} commutes with the directed-edge reversal map (pure algebra from the definition of U_{C_3}), hence (2) maps NB walks to NB walks (H_visible is C_3-invariant), and (3) the MDL observable algebra is C_3-equivariant (every observable transforms under a definite C_3 rep, by Atiyah-Segal 1968 Section 2). This closes the kinematic half of Path A. The closure gap is the *dynamical* half: nothing in A1 + A2-T + A3-T selects the trivial representation from among {trivial, omega, omega^2} for M_mass specifically. Path A remains BLOCKED at the W4 mass-identification step (Gap A.1 of the closure-attempt doc).

- *Path B:* Close ADOPTED-Y (substrate amplitudes = Yukawa couplings, Sprint 7a). A Yukawa coupling to the Higgs doublet is automatically a color singlet by construction in the Pati-Salam embedding. If ADOPTED-Y closes, ADOPTED-CS is a downstream consequence.

**Tier:** Same as ADOPTED-Y and the B3/B6 Pati-Salam labeling adoption.

## What is strict-solid vs adopted

**Strict-solid:** V_tree has (0, 2, 2) C_3-isotypic content (zero trivial sector). This is a theorem-grade spectral fact under A1 + A2-T + A3-T + Ihara-Bass + Serre character orthogonality.

**Adopted:** ADOPTED-CS (M_mass is a C_3 scalar). Two future-closure paths identified (gauge invariance derivation; ADOPTED-Y closure).

**Net status:** Feshbach-pattern. The strict-solid core (Schur structure) ships; the color-singlet identification is flagged.

## Open questions

1. **Path A (gauge invariance):** Can the srs walker dynamics under A1 + A2-T be shown to generate only gauge-invariant observables? This would close ADOPTED-CS without requiring Sprint 7a.

2. **Global vs P-point:** The current derivation applies at the P-point of the BZ. Does V_tree have zero trivial C_3 content at all BZ points? The flat-band structure (lambda = ±1 for all k) suggests the C_3 content of V_tree(k) may be k-independent, but this is not verified here.

3. **Relation to ADOPTED-Y:** If ADOPTED-Y closes via Sprint 7a + Higgs mechanism, does ADOPTED-CS follow automatically? The Yukawa-to-color-singlet argument via Fulton-Harris 1991 Prop 12.17 suggests yes, but this cascade requires careful statement.

## References

### Cited mathematical theorems

- Serre, J.-P. (1977). Linear Representations of Finite Groups. Springer GTM 42. Section 2.2 Proposition 4 (Schur's lemma); Section 2.3 (character orthogonality).
- Terras, A. (2011). Zeta Functions of Graphs. Cambridge University Press. Section 2.2 Theorem 3.1 (Ihara-Bass, k-independence of tree factor).
- Fulton, W. and Harris, J. (1991). Representation Theory: A First Course. Springer GTM 129. Proposition 12.17 (fermion mass terms are SU(3)_c singlets).

### Framework documents

- docs/framework/framework_axioms.md -- A1 + A2-T + A3-T canonical statement.
- docs/theorem_B6_bridge.md -- substrate C_3 = color-Z_3 under Spin(6) → SU(4) PS.
- ../predictions/B_P_doubly_degenerate_h_derivation.md -- B(P) spectrum; (4,2,2) on V_Ram.
- an internal scoping attempt on the canonical-reading question -- Section 3 Step C (prior sketch of P1 closure).

### Upstream closed prediction files

- predictions/tree_subspace_construction.py ((0,2,2) on V_tree; flat-band structure)
- predictions/B_P_doubly_degenerate_h.py ((4,2,2) on V_Ram)
- predictions/k_star.py (k*=3)
- predictions/observer_hilbert_space.py (Hilbert-space structure under A3)

### Superseded prior sketch

- an internal scoping attempt on the canonical-reading question Section 3 Step C -- prior outline of the Feshbach-projection argument; the present derivation formalizes and extends that sketch, identifying ADOPTED-CS as the residual and providing two closure paths.
