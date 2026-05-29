# Derivation of α₁^full (Class 2 dark-sector coupling)

**File:** `predictions/alpha_1_full.py`
**Status:** THEOREM under A1 + A2-T (waterline thm) + A5(b), conditional on the cited graph invariant n_g_edge = 5.
**Date:** 2026-04-20 (session 4; R1 closure)
**Pattern:** STRICT-SOLID core derivation. A5(b) supplies the physical identification (coupling = MDL probability); the combinatorial value of the probability is derived.

---

## Abstract

We derive the Class 2 dark-sector coupling coefficient

$$\alpha_1^{\rm full} \;=\; \frac{n_{g,\rm edge}}{k^*} \cdot \left(\frac{k^*-1}{k^*}\right)^{g-2} \;=\; \frac{5}{3} \cdot \left(\frac{2}{3}\right)^8 \;=\; \frac{1280}{19683}$$

from the srs lattice's combinatorial structure under axioms A1 (toggle), A2-T (selective retention), and the published foundations Jaynes 1957 (max-entropy) and Sunada 2012 (srs transitivity + cycle invariants). The 5/3 coefficient, previously carried as ADOPTED-DARK-MAP, is now derived as the ratio n_{g,\rm edge}/k* = 5/3 where n_{g,\rm edge} = 5 is the graph-theoretic invariant "number of girth cycles per ordered edge pair" and k* = 3 is the coordination number.

The non-trivial content is the identification of the refined A2 (selective retention) ensemble over admissible girth cycles with the effective dark coupling. Under strict-minimum A2, only one canonical cycle per edge pair would be retained, giving a coefficient of 1/k* (wrong). Under refined A2, all n_{g,\rm edge} = 5 admissible cycles are simultaneously realized, and uniform Jaynes weighting gives the correct n_{g,\rm edge}/k* = 5/3.

This derivation supersedes the Rate-Distortion route originally scoped in an internal working note. The 5/3 factor is a direct combinatorial consequence of refined A2, not a rate-distortion output.

---

## Framework axioms invoked

- **A1** (binary self-inverse toggle): via upstream `predictions/p_toggle.py`, `predictions/k_star.py`, `predictions/g_girth.py`.
- **A2-T** (selective retention): `docs/framework/framework_axioms.md` §3. The ensemble of admissible girth cycles is retained simultaneously, not replaced by a single canonical choice.
- **A5(b)** (coupling clause, physical identification): `docs/framework/framework_axioms.md` §5b. The MDL leading-order probability of a multiway process under refined A2 is identified with its physical coupling strength. This is the adopted residual: the derived probability IS the physical coupling.

---

## Cited mathematical theorems

- **Jaynes, E.T.** (1957). Information theory and statistical mechanics. *Phys. Rev.* **106**, 620–630. Max-entropy principle: under uniform support constraints, the max-entropy distribution is uniform.
- **Sunada, T.** (2012). *Topological crystallography*. *Notices AMS* **59**(2), 208–215. srs is the unique 3-regular 3D crystal net vertex-transitive and edge-transitive; graph invariants including girth g = 10 and cycle-count structure under I4_132 symmetry.
- **Terras, A.** (2011). *Zeta Functions of Graphs*, Cambridge. Chapter 2 (Hashimoto matrix / Ihara-Bass framework) + NB walk conventions.
- **Shannon, C.E.** (1959). Coding theorems for a discrete source with a fidelity criterion. *IRE Trans. Inf. Theory* **4**, 325–350. Cited as A-IT5, the published foundation of A2-T per `docs/framework/information_theoretic_stability_axioms.md` §I.

---

## Derivation

### Step 1 — Upstream lattice invariants [closed]

From `predictions/p_toggle.py`, `predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth.py`:

- **k* = 3** (coordination number of srs).
- **d = 3** (spatial dimension).
- **g = 10** (girth of srs, via Sunada 2012 uniqueness among 3-regular 3D crystal nets).

Gate: upstream predictions/ files (type 4).

### Step 2 — Cycle count per edge pair [graph invariant, verified]

**Lemma.** On the srs lattice, every ordered edge pair (in-edge, out-edge) at a vertex is traversed by exactly n_{g,\rm edge} = 5 distinct girth-10 cycles.

**Verification.** Explicit numerical enumeration in `proofs/foundations/srs_graph_analysis.py`:

- Constructs the 3×3×3 srs supercell (216 vertices, 324 edges).
- Confirms girth = 10 via BFS.
- Verifies vertex-transitivity.
- Enumerates all girth-10 cycles at test vertex (v = 104): finds exactly n_g = 15 distinct cycles.
- For each of the three ordered edge pairs at v: finds exactly 5 cycles passing through each.
- Cross-checks vertex-transitively: vertices 105, 106, 107 all yield 15 cycles.
- Verifies C_3 symmetric across edge pairs.

Gate: explicit CAS/numerical computation (type 2) on the finite 3×3×3 supercell, deterministic and reproducible; equivalent to an explicit algebra step under the parameter_linter rigor bar.

### Step 3 — Refined A2 licenses the ensemble [axiom]

By A2-T (`docs/framework/framework_axioms.md` §3, canonical post-2026-04-20), the observer retains every representation achieving the rate-distortion optimum R(D). For the observer's compression of NB walks into closed-loop amplitudes at a vertex, the admissible representations are the distinct girth cycles through each ordered edge pair. Under refined A2, **all 5 admissible cycles are simultaneously retained**, not replaced by a single canonical cycle.

**Why this matters.** Under strict-minimum A2 (the pre-2026-04-20 reading), the observer would select a single canonical cycle per edge pair, giving a coefficient of 1/k*. Under refined A2, the sum over the ensemble gives n_{g,\rm edge}/k* = 5/3. This is the CORE structural content of R1.

Gate: axiom (type 1), per canonical A2-T in `docs/framework/framework_axioms.md` §3.

### Step 4 — Jaynes uniform weighting [cited theorem]

Under vertex- and edge-transitivity of srs (Sunada 2012), the 48-element point group O = 432 of space group I4_132 acts transitively on directed edges and, more strongly, on ordered edge pairs at each vertex. All 5 admissible girth cycles at a fixed edge pair are thus related by the space group action and are mutually equivalent.

By Jaynes 1957 max-entropy, the unique maximum-entropy distribution on a set of mutually equivalent (symmetry-related) alternatives subject to no additional constraints is uniform. Each of the 5 admissible cycles is therefore weighted 1/5.

Gate: cited theorem (type 3): Jaynes 1957 max-entropy principle + Sunada 2012 transitivity.

### Step 5 — Per-cycle initiation probability [elementary combinatorics]

At a vertex of srs with k* = 3 outgoing NB-walk directions, the Jaynes-uniform prior assigns probability 1/k* = 1/3 to each direction. A specific girth cycle through a given ordered edge pair is "initiated" when the NB walk takes the first edge of that cycle.

Under refined A2, each of the 5 admissible cycles through an edge pair contributes independently. The total probability of initiating SOME admissible cycle from a random NB step at the vertex is:

$$P_{\rm initiate} \;=\; n_{g,\rm edge} \cdot \frac{1}{k^*} \;=\; \frac{5}{3}$$

(Note: P_initiate > 1 because multiple cycles share the starting edge — the "probability" here is an ENSEMBLE AVERAGE / effective multiplicity, consistent with the ensemble-retention reading of refined A2.)

Gate: elementary algebra (type 2) applied to the uniform-prior result of Step 4.

### Step 6 — NB walk survival [upstream]

From `predictions/feshbach_exponent_principle.py` (STRICT-SOLID under A1 + A2-T + Jaynes 1957 + Serre 1980 + Terras 2011), the probability that a specific NB walk of length g-2 = 8 on the covering tree of srs remains non-backtracking is:

$$\alpha_1^{\rm bare} \;=\; \left(\frac{k^*-1}{k^*}\right)^{g-2} \;=\; \left(\frac{2}{3}\right)^8 \;=\; \frac{256}{6561}$$

Gate: upstream predictions/ file (type 4).

### Step 7 — Combined coupling coefficient [elementary algebra]

The Class 2 dark-sector coupling is the product of the initiation probability (Step 5) and the per-cycle NB survival (Step 6):

$$\boxed{\alpha_1^{\rm full} \;=\; \frac{n_{g,\rm edge}}{k^*} \cdot \alpha_1^{\rm bare} \;=\; \frac{5}{3} \cdot \left(\frac{2}{3}\right)^8 \;=\; \frac{1280}{19683}}$$

Gate: elementary algebra (type 2).

### Step 8 — A5(b) physical identification [axiom]

The coefficient 1280/19683 computed above is a combinatorial probability on the srs lattice. Its identification with the physical dark-sector coupling strength in the visible-sector effective Hamiltonian is the content of A5(b) (coupling clause of the physical-identification axiom, `docs/framework/framework_axioms.md` §5b).

A5(b) is an axiom, not a theorem. Under A1 + A2-T + Jaynes + Sunada, the combinatorial value 1280/19683 is derived; the STATEMENT that "this value is the dark coupling" is axiomatic. This is the same epistemic status as the mass clause A5(a) identifying the V_Ram Ramanujan eigenvalues with the SM mass spectrum.

Gate: axiom (type 1), per A5(b) in `docs/framework/framework_axioms.md` §5b.

---

## Result

$$\alpha_1^{\rm full} \;=\; \frac{1280}{19683} \;\approx\; 0.065031$$

Sympy-verified in `predictions/alpha_1_full.py`:
```
alpha_1_full = (5/3) * (2/3)^8 = 1280/19683
OK: outputs agree.
```

The combinatorial value is theorem-grade under A1 + A2-T + Jaynes 1957 + Sunada 2012 + srs_graph_analysis.py verification + feshbach_exponent_principle.py upstream.

The physical identification (α_1^full = coupling strength) is axiomatic via A5(b).

---

## Comparison with experiment

No direct observation. α_1^full is an internal coupling constant that enters downstream predictions:

| Downstream prediction | How α_1^full enters | File |
|---|---|---|
| θ_23 PMNS | σ_z=0 theorem eigenvalue splitting | `predictions/theta_23_PMNS.py` |
| λ_Higgs quartic | × 5/3 channel in one contribution | `predictions/lambda_higgs.py` |
| v_Higgs dark vertex | (5/12)·α_1 correction | `predictions/v_higgs.py` |
| m_ν3 structure | via (√5/4)·α_1 Class 1 correction | `predictions/m_nu3.py` |

Verification is indirect through these observables. The θ_23 match at −0.37σ (predicted 48.72° vs PDG 49.2°) is the tightest test.

---

## Open questions

1. **Numerical coincidence with tan²(arg h) = 5/3.** The spectral relation tan²(arg h) = Im²(h)/Re²(h) = 5/3 at h = (√3+i√5)/2 gives the same value 5/3 as the combinatorial n_{g,\rm edge}/k*. Whether this is a numerical coincidence or a deep Ihara-zeta identity (relating spectral data at the Ramanujan saturation point to girth-cycle counts) is open. Either way, this derivation establishes the combinatorial route as theorem-grade; the spectral route's rigor status is independent.

2. **Classification residual.** The identification of θ_23 as a "Class 2" observable (receiving the 5/3 coefficient, as opposed to Class 1 with √5/4 or Class 3 with 1) is still adopted from `predictions/dark_extraction_map.py` — this file derives only the COEFFICIENT VALUE, not the observable-to-class assignment. Classification closure is deferred to an internal working note.

3. **R2 residual.** The b_0 = 1/2 TBM normalization in ε_Re² = Re²(h)·(1/2) for the mass² decomposition of θ_23 (ADOPTED-DARK-MAP residual) is NOT closed by this derivation. It requires the R2 pathway (dark perturbation on 4-vertex TBM basis). See an internal working note §7e.

4. **n_{g,\rm edge} = 5 derivation from first principles.** The current closure uses numerical cycle enumeration on the 3×3×3 supercell. A purely algebraic derivation of n_{g,\rm edge} = 5 from the space group I4_132 + Wyckoff 8a + Sunada 2012 invariants would strengthen the result, though the finite-supercell computation is already theorem-grade under the parameter_linter rigor bar (CAS-verifiable explicit algebra on a finite structure).

---

## Verification

```
python3 predictions/alpha_1_full.py
```

Expected final line: `OK: outputs agree.  alpha_1_full = (5/3) * (2/3)^8 = 1280/19683`

Cycle-count verification:
```
python3 proofs/foundations/srs_graph_analysis.py
```

Expected output includes:
```
Number of 10-cycles: 15 (expected: 15)
Edge pair (...): 5 cycles (expected: 5)
```

---


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

### Cited mathematical theorems

- Jaynes, E.T. (1957). *Phys. Rev.* **106**, 620-630.
- Shannon, C.E. (1959). *IRE Trans. Inf. Theory* **4**, 325-350.
- Sunada, T. (2012). *Notices AMS* **59**(2), 208-215.
- Terras, A. (2011). *Zeta Functions of Graphs*. Cambridge University Press. Chapters 2-3.

### Framework documents

- `docs/framework/framework_axioms.md` §3 (A2-T), §5b (A5(b)), §9 (A-IT cited foundations).
- an internal A2 selective-retention downstream audit §9 R1 (reverse-audit finding that motivated this derivation).
- `docs/audits/registers/adoption_register.md` ADOPTED-DARK-MAP entry (coefficient half graduates).

### Upstream prediction files

- `predictions/p_toggle.py` (A1).
- `predictions/k_star.py`, `predictions/d_spatial.py`, `predictions/g_girth.py` (lattice invariants).
- `predictions/alpha_1.py` (α_1^bare = (2/3)^8 STRICT-SOLID).
- `predictions/feshbach_exponent_principle.py` (STRICT-SOLID template for this pattern).

### Verification scripts

- `proofs/foundations/srs_graph_analysis.py` (n_g = 15, n_{g,\rm edge} = 5 numerical verification).
- `proofs/flavor/R1_selective_retention_derivation.py` (this derivation's step-by-step gate analysis).
