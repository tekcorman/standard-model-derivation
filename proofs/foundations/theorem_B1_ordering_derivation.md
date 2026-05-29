# Theorem B1 — No MDL-canonical ordering of K₄-quotient edges

**Audit anchor:** Row 15b of `docs/audits/registers/uniqueness_ledger.md` (B1 ordering OPEN globally; CAR/JW UNIQUE locally). Load-bearing for the A4 axiom-elimination workstream.

## Abstract

The primitive cell of the srs lattice (space group I4₁32, Wyckoff 8a) has 4
vertices and 6 undirected edges whose cell-quotient is the complete graph K₄.
A canonical ordering of these 6 edges would be required by any Clifford-algebra
programme that labels generators by explicit indices. We show that no such
ordering is forced by the framework axioms A1 (self-inverse toggle) and A2
(MDL): the automorphism group S₄ = Aut(K₄) partitions the 720 possible
orderings into 30 equally-costly orbits, the NB walker cannot visit each
quotient edge exactly once (Euler obstruction), and therefore no ordering is
MDL-preferred. Verdict: **B1.b** — the Clifford algebra on the 6-edge space
must be defined invariantly via the tensor-algebra quotient Cl(V, Q).

## Framework axioms invoked

- **A1** (self-inverse binary toggle): the srs NB walker is derived from A1;
  the K₄ quotient inherits its topology.
- **A2** (MDL, Grünwald 2007 §5.3): the two-part description-length criterion
  is invoked in Step 3 to attempt canonical ordering selection; it fails.

## Derivation

### Step 1 — S₄ orbit structure

**Claim.** S₄ = Aut(K₄) acts freely on the 720 orderings of K₄'s 6 edges,
partitioning them into exactly 30 orbits each of size 24.

By the orbit-stabilizer theorem (Dummit & Foote 2004 §4.1 Prop 2):

$$
|\text{orbit}| \cdot |\text{Stab}| = |S_4| = 24.
$$

The action of S₄ on 2-subsets of {0,1,2,3} is faithful (the kernel of
S₄ → Sym(E) is trivial; S₄ has trivial center for n ≥ 3, Dummit & Foote §4.3
Prop 11), so every stabilizer is trivial, every orbit has size 24, and the
number of orbits is 720/24 = 30. Computationally verified: all 30 orbits
enumerated with sizes uniformly 24.

### Step 2 — LG-Hamiltonian orderings

**Claim.** Exactly 240 of the 720 orderings are line-graph-Hamiltonian (LG-Ham):
consecutive edges share a K₄ vertex. These 240 orderings form exactly 10 of
the 30 S₄-orbits.

The line graph L(K₄) is the octahedron K_{2,2,2} (Whitney 1932; Harary 1969
§8). LG-Hamiltonicity is preserved by the S₄ action on L(K₄) (since S₄ acts
by graph automorphisms on L(K₄)), so LG-Ham orderings form whole orbits. Direct
enumeration gives 240 LG-Ham orderings = 10 complete orbits of size 24.

### Step 3 — MDL cost per orbit

The two-part MDL code for an ordering (Grünwald 2007 §5.3) is:

$$
DL = \underbrace{\log_2 N_{\text{orbits}}}_{\text{model}} +
     \underbrace{\log_2 |\text{orbit}|}_{\text{data}}.
$$

Because all 30 orbits have identical size 24, the data cost is the same for
every orbit. The model cost is also identical (all 30 models receive equal
uniform-code length). Therefore:

$$
DL_{\text{all}} = \log_2 30 + \log_2 24 = \log_2 720 \approx 9.492 \text{ bits}.
$$

Within the LG-Ham restricted class (10 orbits):

$$
DL_{\text{LG}} = \log_2 10 + \log_2 24 = \log_2 240 \approx 7.907 \text{ bits.}
$$

**No orbit is MDL-minimal** — the tie is structural, not numerical. A non-uniform
code would require an orbit ordering, which is itself an unforced choice
outside A1 + A2-T.

### Step 4 — Euler trail obstruction

**Claim.** K₄ has zero Eulerian trails. Therefore the NB walker on srs cannot
project to an ordering that uses each of the 6 K₄ edges exactly once.

**Proof.** A connected graph has an Eulerian trail iff it has exactly 0 or 2
vertices of odd degree (Euler 1736; Bondy & Murty 2008 Theorem 4.1). K₄ is
3-regular: all 4 vertices have odd degree 3. Hence zero Eulerian trails. This
closes the walker alternative canonicalisation route.

### Synthesis

No MDL-canonical ordering is forced (Steps 1–3). No walker-induced ordering
exists (Step 4). Theorem B1.b holds.

## Result

No MDL-canonical ordering of the K₄-quotient edges is forced by A1 + A2-T.
Downstream Clifford workstreams B2, B3, B4 must use the manifestly
S₆-equivariant tensor-algebra construction:

$$
\mathrm{Cl}(V, Q) := T(V) / \langle v \otimes v - Q(v) \cdot 1 : v \in V \rangle
$$

(Lawson & Michelsohn 1989 Ch. 1 §1 Eq. (1.1)).

## Comparison with experiment

N/A — foundational theorem. B1 constrains the Clifford-algebra formulation;
it makes no direct numerical prediction. Its consequence is that any physical
quantity extracted from Cl(V, Q) must be S₆-representation-theoretically
natural.

## Open questions

- Whether the 10 LG-Ham orbits carry further physical structure (e.g.
  chirality under I4₁32) that provides a refined but still non-unique
  partial ordering.
- Whether a third canonicalisation route (e.g. from a crystallographic
  branch-cut convention derived internally from A1 + A2-T) could break the tie;
  such a route would need to be derived, not assumed.

## References

- Bondy, J.A. & Murty, U.S.R. (2008). *Graph Theory.* Springer GTM 244.
  Theorem 4.1 (Eulerian trails).
- Dummit, D.S. & Foote, R.M. (2004). *Abstract Algebra* 3rd ed. Wiley.
  §2.2 Ex. 4, §4.1 Prop 2, §4.3 Prop 11.
- Euler, L. (1736). Solutio problematis ad geometriam situs pertinentis.
  *Commentarii Acad. Sci. Petropolitanae* 8, 128–140.
- Grünwald, P. (2007). *The Minimum Description Length Principle.* MIT Press.
  §5.3 (two-part code length).
- Harary, F. (1969). *Graph Theory.* Addison-Wesley. §8 (line graphs).
- Lawson, H.B. & Michelsohn, M.-L. (1989). *Spin Geometry.* Princeton Univ.
  Press. Ch. 1 §1 (tensor-algebra Clifford construction).
- Whitney, H. (1932). Congruent graphs and the connectivity of graphs.
  *Amer. J. Math.* 54, 150–168.
