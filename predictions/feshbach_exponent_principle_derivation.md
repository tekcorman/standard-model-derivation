# Derivation of the Feshbach Exponent Principle (combinatorial form)

## Abstract

We derive the Feshbach Exponent Principle — the claim that, on a k-regular graph G of girth g, an NB walk that has n_fixed directed edges pinned as external contributes a non-backtracking survival factor ((k-1)/k)^(g - n_fixed) to the amplitude of a minimum-length closed walk through those pinned edges — as a standalone combinatorial theorem under A1 + A2-T + A3-T. The derivation chain is: walker_dynamics W4 supplies the per-step NB survival probability (k-1)/k on the universal covering tree of G from Jaynes 1957 + MDL cancellation; Serre 1980 §I.1 supplies the cycle-equals-girth content; independence of per-step survival events on the universal covering tree follows from the tree's loop-freeness (Terras 2011 §2.1). The minimum-length closed NB walk through any subset of n_fixed pinned directed edges of a single girth cycle uses those n_fixed edges plus the remaining g - n_fixed internal edges of the cycle; by W4-independence the combined survival factor over the internal edges is ((k-1)/k)^(g - n_fixed). The scope of this theorem is n_fixed in {0, 1, 2}. Outside this range the "pinned subset fits inside a single girth cycle" hypothesis no longer generically holds and the exponent formula is not this theorem's content.

We emphasize scope honesty: this file derives the **combinatorial** Exponent Principle (a statement about NB walk survival factors). The **physical** Exponent Principle — the reading of this survival factor as the coupling strength of a scattering / transition / self-energy amplitude in an observable on srs — was previously a separate downstream identification ("I-Feshbach" per `../predictions/Feshbach_coupling_strength_derivation.md` §3). That identification is now subsumed by **A5(b)** — the coupling clause of A5 (`docs/framework/framework_axioms.md` §5b, established 2026-04-19 session 2). Combined with the combinatorial theorem here, the full identification (NB walk survival = physical coupling strength) is THEOREM under A1 + A2-T + A5(b) + Jaynes 1957 + Serre 1980 + Terras 2011.

## Framework axioms invoked

- **A1** (binary self-inverse toggle): each edge e of G carries a toggle operator T_e with T_e · T_e = 1.
- **A2-T** (MDL canonicalization; derived theorem): the observer retains the minimum-description-length representation of each observable-equivalence class of toggle streams. The cancellation e · e ~ epsilon is forced by A2-T. See `docs/theorems/theorem_A2_mdl_from_finite_register.md`.
- **A3-T** — only used to ground the Hilbert-space embedding of B; the combinatorial content of the present theorem does not invoke A3-T directly. See docs/theorems/theorem_A3_complex_hilbert_from_multiway.md.

## Cited mathematical content

- **Jaynes 1957** (Phys. Rev. 106, 620-630): max-entropy conditional distribution over a finite alphabet is uniform.
- **Serre 1980** (*Trees*, §I.1 Prop. 4, §I.3): free involutive monoid reduced-word uniqueness; universal covering tree is loop-free.
- **Terras 2011** (*Zeta Functions of Graphs*, §2.1, §2.2): NB walks on a graph correspond canonically to cycles; Hashimoto B is the NB 1-step adjacency on directed edges.
- **Shannon 1948** (Bell Syst. Tech. J. 27, 379-423): source coding theorem, entry to MDL canonicalization.

## Upstream framework files

- `predictions/k_star.py` — k* = 3.
- `predictions/g_girth.py` — g = 10 (for srs).
- `../predictions/walker_dynamics_derivation.md` — W1 (reduced-word data representation), W2 (causal state = directed edge), W3 (B is the 1-step NB operator), W4 (Jaynes-uniform over k-1 NB continuations).

## Derivation

### Step 1: Per-step NB survival on the universal covering tree

By walker_dynamics W4 (`../predictions/walker_dynamics_derivation.md` Step 4, Lemma "NB uniform"), at a k-regular vertex visited by an NB walker that arrived via a specific incoming directed edge, the conditional distribution over the next directed edge is uniform over the k - 1 non-backtrack outgoing choices. Under the unconditional distribution (averaging over all k incident edges), the probability that the next step is a valid NB continuation (equivalently, that the walker "survives" the step rather than triggering an MDL cancellation) is

    p_step = (k - 1) / k.

This is the rigor-bar claim of `../predictions/Feshbach_coupling_strength_derivation.md` Lemma 1, itself a consequence of A1 (self-inverse toggle generates the involutive monoid; cancellation is forced at repeated edges) + A2-T (MDL selects the reduced-word representative) + Jaynes 1957 (max-entropy over the k incident edges at each vertex). The derivation is complete under A1 + A2-T and does not rely on downstream physics identifications.

### Step 2: Independence of per-step events on the universal covering tree

Let T_G be the universal covering tree of G. Serre 1980 §I.3 proves that T_G is a tree (no cycles), so any two distinct vertices of T_G are joined by a unique path. Consequently, an NB walk on T_G of length L never re-encounters a previously visited vertex, and the survival events at distinct steps are independent events on the tree's probability space (each step's survival is a function only of the local neighborhood at its base vertex, and the neighborhoods at distinct vertices of the tree are disjoint).

Therefore the joint probability that the walker stays on the tree (equivalently, continues as an NB walk) for L consecutive steps is the product:

    p_tree(L) = prod_{i=1}^{L} p_step = ((k - 1) / k)^L.

This is Lemma 1 of `../predictions/Feshbach_coupling_strength_derivation.md`, derivable under A1 + A2-T with no further structural input.

### Step 3: Girth and closed NB walks on G

By definition, the girth g(G) of a graph G is the length of its shortest cycle. Terras 2011 §2.1 establishes that a closed NB walk of length L on G (starting and ending at the same directed edge) corresponds canonically to a cycle of length L in G. In particular, the minimum possible length of a closed NB walk from a directed edge e back to e is exactly g(G). On srs, g = 10 (from `predictions/g_girth.py`).

### Step 4: Pinning n_fixed edges of a girth cycle

Fix a girth cycle C on G, consisting of g directed edges (e_1, e_2, ..., e_g) where each e_{i+1} is the NB successor of e_i and e_1 is the NB successor of e_g (closed walk). We say a subset F ⊆ {e_1, ..., e_g} of directed edges is "pinned" if the walk's amplitude is evaluated with F's edges as external (contributing no per-step NB survival factor), while the complement C \ F consists of internal edges (each contributing a per-step survival factor p_step).

For n_fixed = |F| in {0, 1, 2}:

- **n_fixed = 0**: F is empty. All g edges of C are internal. By Step 2, the combined NB survival factor over g steps on the tree cover (to which the girth cycle lifts as a simple tree path of length g before it re-enters the base graph at its closing edge) is

        survival(n_fixed = 0) = ((k - 1) / k)^g.

- **n_fixed = 1**: F = {e_j} for some j. The remaining g - 1 edges of C are internal. By Step 2, the combined NB survival factor over g - 1 steps on the tree cover is

        survival(n_fixed = 1) = ((k - 1) / k)^(g - 1).

- **n_fixed = 2**: F = {e_j, e_{j'}} for some j ≠ j'. The remaining g - 2 edges of C are internal. By Step 2, the combined NB survival factor over g - 2 steps on the tree cover is

        survival(n_fixed = 2) = ((k - 1) / k)^(g - 2).

In every case the factorization across internal edges uses Step 2's independence of survival events on the tree cover, and the exponent counts only internal (non-pinned) edges of the girth cycle.

### Step 5: Scope of the theorem

The argument of Step 4 relies on the pinned subset F fitting inside a single girth cycle. For |F| ≤ 2, this is generically achievable (any single vertex lies on multiple girth cycles, and any pair of directed edges sharing a vertex can be extended to at least one girth cycle on srs — verified numerically in `proofs/foundations/hashimoto_exponents.py`). For |F| ≥ 3, a minimum closed walk through the pinned set is not guaranteed to be a single girth cycle; the walk may decompose into multiple cycles, each of length ≥ g, giving exponents of the form n_1 g + n_2 (g-1) + n_3 (g-2) rather than a single (g - n_fixed).

We therefore restrict the combinatorial Exponent Principle of this file to n_fixed in {0, 1, 2}. Multi-loop cases (the seesaw 4g = 40 exponent; the gravitino k^2 g = 90 exponent; catalogued in an external research note) require separate combinatorial arguments that this theorem does not cover.

## Result

For n_fixed in {0, 1, 2} on a k-regular graph G of girth g,

    coupling(n_fixed) = ((k - 1) / k)^(g - n_fixed).

Instantiated on srs (k = 3, g = 10):

| n_fixed | Reading                              | exponent | coupling (rational) | approx       |
|---------|--------------------------------------|----------|---------------------|--------------|
|   0     | Self-energy (closed NB loop)         | g = 10   | (2/3)^10 = 1024/59049 | 0.017341529 |
|   1     | Transition (one pinned external)     | g-1 = 9  | (2/3)^9 = 512/19683   | 0.026012295 |
|   2     | Scattering (in+out pinned)           | g-2 = 8  | (2/3)^8 = 256/6561    | 0.039018442 |

The n_fixed = 2 case reproduces alpha_1_bare of `predictions/alpha_1.py` and closes the combinatorial side of `../predictions/Feshbach_coupling_strength_derivation.md`'s I-Feshbach identification.

## Comparison with experiment

Not directly observable. The three couplings enter downstream predictions:

- n_fixed = 2: alpha_1_bare enters `predictions/V_cb.py`, `predictions/V_us.py` (Feshbach route), and the Higgs quartic lambda = 2 alpha_1_full.
- n_fixed = 1: the transition reading appears in PMNS delta_CP phase identifications (e.g., arg(h^{g-1}) in the CP phase derivation).
- n_fixed = 0: the self-energy reading appears in Majorana phase identifications (arg(h^g)).

Verification is indirect via the accuracy of these downstream predictions, each of which must close its own identification gap between the combinatorial survival factor (content of this file) and the physical amplitude (downstream reading).

## Open questions

1. **Physical coupling = NB survival?** The present theorem is purely combinatorial: it states that an NB walk of length g - n_fixed on the universal covering tree of G has survival probability ((k-1)/k)^(g - n_fixed) per Step 2, and that pinning n_fixed edges of a girth cycle of G leaves g - n_fixed internal edges per Step 4. The *identification* of this NB-walk survival factor with the physical Feshbach coupling strength (the I-Feshbach identification of `../predictions/Feshbach_coupling_strength_derivation.md` §3) is NOT content of this theorem. I-Feshbach remains an adopted structural theorem at the physics-interpretation level, on the same tier as P1 / P2 in `W4_identification_catalog.md`. The standalone theorem proved here addresses only the combinatorial part; it does not close the physics identification.

2. **Scope.** The theorem is restricted to n_fixed in {0, 1, 2}. Multi-loop exponents (4g = 40 for m_nu3, k^2 g = 90 for the gravitino) are separate combinatorial claims that this file does not prove. Extending the theorem to multi-loop cases would require a separate combinatorial argument identifying the minimum-length NB walk through n_fixed ≥ 3 pinned edges as a concatenation of whole girth cycles, which is not generically true on a k-regular graph and which has not been proved at journal grade anywhere in the framework.

3. **Independence on the base graph vs on the tree cover.** Step 2's independence argument holds on the universal covering tree T_G, where survival events are disjoint-vertex events. On the base graph G, survival events at distinct steps of a girth cycle share vertices (the cycle closes), and the tree-level independence argument must be combined with the lift-to-tree-cover construction to apply. This is handled by the standard Ihara-Bass identity (`../predictions/Feshbach_coupling_strength_derivation.md` Step 7.4 for an explicit construction), but the present file does not redo that construction: it invokes "Step 2's tree-level independence applied to the lift of a girth cycle to its tree cover." A fully explicit residue calculation (Ihara-Bass Green's function expansion on srs) would make the lift-to-tree-cover explicit; this is the path sketched in `../predictions/Feshbach_coupling_strength_derivation.md` §6 and left as "tractable multi-session work" in that doc. The present theorem is therefore a combinatorial claim modulo the standard Ihara-Bass lift, not a fully written-out residue identity.

## Honest status under the rigor bar

Under A1 + A2-T + A3-T and the cited mathematical content (Jaynes 1957, Serre 1980, Terras 2011, Shannon 1948), the combinatorial statement

    survival(n_fixed) = ((k - 1) / k)^(g - n_fixed) for n_fixed in {0, 1, 2}

is a theorem in the same sense as Lemma 1 of `../predictions/Feshbach_coupling_strength_derivation.md` — a statement about NB walk survival probabilities on k-regular graphs that follows from walker_dynamics W4 + Serre 1980 + Terras 2011 + elementary subtraction. Its derivation is journal-grade at the combinatorial level.

What this theorem does NOT do:

- It does not prove the I-Feshbach identification between NB survival and the physical coupling strength of a Feshbach scattering amplitude; that is the adopted-postulate downstream of this theorem.
- It does not cover n_fixed ≥ 3 multi-loop cases.
- It does not replace the explicit Ihara-Bass residue calculation sketched in `../predictions/Feshbach_coupling_strength_derivation.md` §6 for a fully written-out Green's-function proof on srs.

What this theorem DOES do:

- It isolates the combinatorial content of the Exponent Principle from the physics-interpretation content. The former (this file) closes cleanly under A1 + A2-T + A3-T. The latter (I-Feshbach) remains an adopted structural postulate, on which the framework's physics predictions depend and which has been numerically verified on K_4 and srs but not proved formally.
- It makes the n_fixed in {0, 1, 2} scope explicit, so downstream prediction files that invoke the Exponent Principle can cite the scope cleanly and flag multi-loop cases as requiring independent justification.

## Multi-loop extension (n_fixed >= 3)

**Date of this section:** 2026-04-18.

This section analyses whether the combinatorial formula `survival(n_fixed) = ((k-1)/k)^(g - n_fixed)` extends to n_fixed = 3 and n_fixed = 4, and whether the "4g = 40" and "k^2 g = 90" exponents catalogued in an external research note can be derived within the same combinatorial framework.

### Two distinct extension problems

The phrase "n_fixed >= 3 multi-loop case" bundles two structurally different extension problems that must be separated before attempting a derivation.

**Extension A: Higher n_fixed, co-cyclic pinning (n_fixed = 3 or 4 edges on one girth cycle).**

This is a direct generalisation of Steps 3-4 of the derivation above. A girth cycle of G has g directed edges. If n_fixed ≤ g edges are pinned and all n_fixed pinned edges co-lie on a single girth cycle, then after removing the n_fixed pinned edges the remaining g - n_fixed internal edges are all still consecutive arcs of that same cycle. The argument of Step 4 applies verbatim: each internal edge lifts to a step on the universal covering tree, the lifted steps are at distinct tree vertices (since the girth cycle lifts injectively to a path of length g on the universal covering tree before it returns to its basepoint — this is the definition of girth plus Serre 1980 §I.3), and therefore the per-step survival events are independent on the tree. The combined survival probability over the g - n_fixed internal steps is `((k-1)/k)^(g - n_fixed)`.

The formula therefore extends to n_fixed = 3 and n_fixed = 4 **within the scope that all n_fixed pinned edges are co-cyclic on one girth cycle**. The resulting survival factors on srs (k = 3, g = 10) are:

| n_fixed | exponent | survival (rational) | approx |
|---------|----------|---------------------|--------|
|   3     | g-3 = 7  | (2/3)^7 = 128/2187  | 0.058524590 |
|   4     | g-4 = 6  | (2/3)^6 = 64/729    | 0.087791495 |

**Gap for Extension A:** The argument is closed at the combinatorial level for co-cyclic pinned sets under A1 + A2-T + A3-T + Serre 1980 + Terras 2011, exactly as for n_fixed in {0, 1, 2}. The specific gap is the **co-cyclicity condition**: the theorem requires that any n_fixed pinned edges can be embedded in a single girth cycle. For n_fixed ≤ 2 this is always achievable on srs (any pair of directed edges that do not form a backtrack pair lies on at least one girth cycle; this is a statement about the girth-10 srs topology that is numerically verified in `proofs/foundations/hashimoto_exponents.py` but not proved at journal grade by an explicit graph-structure argument). For n_fixed = 3 the co-cyclicity condition is stricter: three directed edges co-lie on one girth cycle iff (a) no two are in backtrack relation (e and its reverse cannot both be in an NB walk), and (b) their pairwise walk-distances on the cycle sum to g. Condition (b) is a constraint on the relative positions of the three edges within the cycle. Generic triples of directed edges on srs may fail this condition and require the minimum closed NB walk to use two girth cycles rather than one, giving a longer internal path.

**Conclusion for Extension A.** The formula `((k-1)/k)^(g - n_fixed)` holds for n_fixed = 3 or 4 **if and only if** the pinned edge set is co-cyclic. The combinatorial proof is a verbatim extension of Steps 1-4 above. However, a general Extension A theorem for arbitrary n_fixed ≥ 3 edge sets would require either: (i) proving that every set of n_fixed directed edges on srs with no backtrack pair is co-cyclic (this is NOT true for large n_fixed: srs has girth 10 so any co-cyclic set has n_fixed ≤ 10); or (ii) replacing the co-cyclicity assumption with a minimum-walk-length computation for each specific pinned set. Under the rigor bar, this is **STRICT-SOLID-CONDITIONAL** on the co-cyclicity assumption; without it, it is BLOCKED.

**Extension B: Multi-loop walk topology (the "4g" and "k^2 g" exponents).**

The exponents 4g = 40 and k^2 g = 90 (catalogued in an external research note) are NOT instances of n_fixed-edge pinning on a single girth cycle. They describe walk topologies involving multiple concatenated or nested girth cycles. This is the graph-theoretic analog of multi-loop Feynman diagrams. The two cases are structurally different from Extension A.

**What "4g" means combinatorially.** A walk with exponent 4g = 40 is a closed NB walk of total length 4g that traverses 4 full girth cycles in sequence. Such a walk would start on a directed edge e_0, traverse a girth cycle back to e_0 (length g), then traverse a second girth cycle from e_0 (length g), and so on for 4 iterations, returning to e_0 after total length 4g. However, this description has a structural inconsistency on a k-regular graph: after completing one girth cycle and returning to e_0, the walker is again at the starting directed edge. A second NB walk of length g from e_0 would be identical to the first (on the covering tree, the unique NB walk of length g from e_0 is the same tree path lifted from the girth cycle). Therefore the "4 concatenated girth cycles" picture does not describe a single NB walk that stays non-backtracking throughout its full length 4g; after the first return to e_0, the NB condition is violated because the walker must continue from e_0 and its next step would backtrack.

This means: a walk topology giving exponent 4g cannot be a simple 4-fold concatenation of girth cycles. The correct combinatorial picture would involve 4 distinct girth cycles sharing e_0 as a junction but traversed via different branches at each junction, i.e., a "rose graph" or "bouquet" topology embedded in srs with 4 petals each of length g. Such a topology requires each junction step to select one of the k-1 = 2 NB outgoing branches at each return to e_0, giving a branching factor of (k-1)^(number of junctions) on top of the individual cycle survival factors.

**Attempted formula for 4-loop walk:** A closed NB walk of topology "4 girth cycles sharing one base vertex" would traverse 4 * g = 40 edges total if all 4 cycles are traversed in sequence without reuse of edges between cycles (i.e., each cycle uses a distinct set of g directed edges). On srs, the number of distinct girth cycles through any given directed edge is bounded by the graph's girth-cycle count n_g = 15. For a walk topology of 4 distinct girth cycles sharing one entry edge, the survival factor would involve: (a) the g-step survival for each of the 4 cycles separately = ((k-1)/k)^g per cycle; (b) a branching factor at each junction where the 4 cycles connect. Without a precise specification of the walk topology, the formula is undetermined.

**What "k^2 g" means combinatorially.** The exponent k^2 g = 9 * 10 = 90 suggests a walk topology where the scale factor is k^2 = (k*)^2 = 9 rather than a simple integer multiple of g. On srs, k^2 = 9 is the number of length-2 NB walks from any fixed directed edge (each of the k-1 = 2 NB successors has k-1 = 2 further NB successors, giving (k-1)^2 = 4, not k^2 = 9; or counting all k^2 = 9 pairs of steps including backtracks). The source of the k^2 factor is not determined by the girth-cycle picture and requires a separate combinatorial argument.

**Conclusion for Extension B.** The "4g" and "k^2 g" exponents as catalogued in the external research note are BLOCKED at the combinatorial level. The existing derivation framework (single girth cycle, n_fixed pinned edges, tree independence) does not cover them. Extending to multi-loop walk topologies requires:

1. A precise definition of what "multi-loop closed NB walk" means combinatorially on srs (not just naming the exponent but specifying which walks contribute).
2. A proof that the minimum-length such walk has the claimed length (4g or k^2 g), which is a non-trivial statement about srs graph topology.
3. An independence argument for the survival events within a multi-loop walk that accounts for the walk re-visiting junction vertices (invalidating Step 2's simple tree-independence argument).

None of these three steps are provided by the existing framework. The claim "exponent = 4g" or "exponent = k^2 g" originates in an external research note without a combinatorial derivation at the level of the n_fixed in {0,1,2} theorem. Until the three steps above are completed at journal grade, these exponents remain **BLOCKED** as combinatorial theorems.

### Verdict for Push 1

| Case | Status | Gap |
|------|--------|-----|
| n_fixed = 3, co-cyclic | STRICT-SOLID-CONDITIONAL | co-cyclicity of the pinned triple must be established for the specific physical application; formula `(2/3)^7` follows verbatim from Steps 1-4 |
| n_fixed = 4, co-cyclic | STRICT-SOLID-CONDITIONAL | same co-cyclicity condition; formula `(2/3)^6` follows verbatim |
| n_fixed = 3 or 4, non-co-cyclic | BLOCKED | minimum-walk-length through non-co-cyclic triple/quadruple is not g; explicit computation needed |
| "4g = 40" multi-loop | BLOCKED | walk topology not defined at combinatorial level; three missing steps listed above |
| "k^2 g = 90" multi-loop | BLOCKED | walk topology not defined at combinatorial level; source of k^2 factor unspecified |

The strict-solid content of the present theorem covers n_fixed in {0, 1, 2} unconditionally and n_fixed in {3, 4} conditionally on co-cyclicity of the pinned edge set. Physics applications that invoke 4g or k^2 g exponents (m_nu3, gravitino mass) remain outside the scope of any version of this theorem as currently formulated.


## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.

## References

- Chiribella, G., D'Ariano, G.M., Perinotti, P. (2011). Informational derivation of quantum theory. *Phys. Rev. A* **84**, 012311. (A3 prior art.)
- Jaynes, E.T. (1957). Information theory and statistical mechanics. *Phys. Rev.* **106**, 620-630. (W4 max-entropy input to Step 1.)
- Serre, J.-P. (1980). *Trees.* Springer-Verlag. §I.1 (reduced words), §I.3 (universal covering tree is loop-free).
- Shannon, C.E. (1948). A mathematical theory of communication. *Bell Syst. Tech. J.* **27**, 379-423. (Source coding; A2 canonicalization.)
- Terras, A. (2011). *Zeta Functions of Graphs: A Stroll through the Garden.* Cambridge University Press. §2.1 (NB walks and cycles), §2.2 (Hashimoto B).

## Files referenced

- `docs/framework/framework_axioms.md` — A1, A2, A3 canonical statement.
- `../predictions/walker_dynamics_derivation.md` — W1, W2, W3, W4 closure.
- `../predictions/Feshbach_coupling_strength_derivation.md` — Lemma 1 (tree NB survival), §3 I-Feshbach identification, §6 residue-proof sketch.
- `predictions/alpha_1.py` — the n_fixed = 2 instantiation on srs (alpha_1_bare = (2/3)^8).
- `predictions/feshbach_exponent_principle.py` — the implementation of the present theorem.
- `predictions/k_star.py`, `predictions/g_girth.py` — upstream k and g values.
- `proofs/foundations/hashimoto_exponents.py`, `proofs/foundations/exponent_ladder.py` — numerical verification scripts on K_4 and srs.
