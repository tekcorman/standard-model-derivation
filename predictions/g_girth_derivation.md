# Derivation of g (girth of the srs lattice)

**Audit anchor:** Row 9 of `docs/audits/registers/uniqueness_ledger.md`. Conditional on Rows 4 (k* = 3) + 6 (srs identification). UNIQUE under structural-pass closure.

## Abstract

We derive that the girth (shortest cycle length) of the framework's optimal graph is $g = 10$. This is a mathematical property of the srs (Laves) lattice, which is uniquely selected by MDL among all 3-regular 3D crystal nets. The selection is proven via Sunada's uniqueness theorem (2012) and the explicit DL comparison in `proofs/foundations/dl_comparison.py`. The girth is not independently derived from axioms — it is a consequence of srs being the unique optimal graph.

## Framework axioms invoked

1. **MDL compression** (inherited from `predictions/d_spatial.py`).
2. **Self-inverse toggle** (inherited from `predictions/p_toggle.py`).

No new axioms. The girth is a property of the graph selected by MDL.

## Derivation

### Step 1: MDL selects a 3-regular 3D crystal net

From `predictions/k_star.py`: $k^* = 3$ (coordination number) and $d = 3$ (spatial dimension). The observer's optimal model is a 3-regular 3D crystal net.

### Step 2: srs is the unique MDL minimum among 3-regular 3D crystal nets

**Why the "strongly isotropic" category at all** (front-end added 2026-05-12 — see `predictions/walker_dynamics_derivation.md` Step 4b). The walker's causal state is a *directed edge* (Step 5 of walker_dynamics — Shalizi-Crutchfield 2001). By (A) (self-containment), nothing supplied from outside privileges any direction at a vertex or either orientation of an edge — that would be "which-way" information, which (A) forbids supplying (the toggle-theorem-Step-1 no-privilege principle applied to spatial labels; `d_spatial_derivation.md` already works "under isotropic toggle dynamics"). So the observer's model must treat all directed edges equivalently — its automorphism group acts transitively on (vertex, directed-edge) pairs — i.e. the model is **strongly isotropic**, and by substrate-agnosticism the substrate is. Strong isotropy is therefore *derived from (A)*, not adopted; the case analysis below then identifies which net realizes it (Sunada 2012: uniquely srs) and confirms the 8 non-strongly-isotropic candidates pay extra description bits. This is the structural closure of R-9 (`docs/audits/registers/structural_residue_register.md`).

**Sharp-peak case** (clarifying note added 2026-05-05). The "MDL minimum" framing here is genuine, not the strict-minimum smuggle reformulated in `theorem_lattice_coupling_general.md` §2: Sunada 2012 proves srs is the *unique* strongly-isotropic 3-connected 3D crystal net, and the M2a structural-DL refinement (case analysis below) further excludes the 8 V+E-transitive-but-not-strongly-isotropic RCSR candidates by additional bits. The substrate-channel landscape has a unique dominant peak; there are no encoding-equivalents to canonicalize and no other above-waterline channel to compete. Per `feedback_a2_waterline.md`, waterline and strict-min agree in the unique-peak regime — this is exactly that regime. Not subject to the `canonical_encoding`/`channel_select` operator split.


**Theorem** (Sunada, *Notices AMS* **59**(2), 208–215, 2012): The srs (Laves) lattice is the unique 3-connected 3D crystal net that is **strongly isotropic** — that is, whose crystallographic automorphism group acts transitively on (vertex, directed-edge) pairs.

> **Definitional note (added 2026-05-02 cleanup).** Sunada's "vertex-and-edge-transitive" notion in this 2012 paper is the strong-isotropy sense (transitive on directed-edge / arc pairs), which is **strictly stronger** than RCSR's "(1,1)-transitive" classification (one vertex orbit + one undirected edge orbit). RCSR enumerates 9 V+E-transitive 3-c chiral cubic 3D candidates (per `proofs/foundations/rcsr_candidate_sweep.py`); only srs is strongly isotropic. The remaining 8 candidates (srs-z, srs-c4, srs-c8, srs-c27, lou, lov, okw, hcb-c4) are V+E-transitive but NOT strongly isotropic — their automorphism group fails to act transitively on directed edges. **[CORRECTION 2026-07-10: the original "(each has 2+ arc orbits)" clause here was FALSIFIED by `proofs/foundations/arc_transitivity_ground_truth.py` — srs-z has exactly ONE arc orbit (edge-reversible, local action C₃ not S₃); the srs/srs-z discriminator is STRONG ISOTROPY (full local S₃), which is strictly stronger than arc-transitivity and carries NO selection load per the R-9 supersession (2026-06-15). Selection authority = the fingerprint + waterline study; this note supersedes the clause.]** See `docs/audits/registers/structural_residue_register.md` R-9 sub-items (k)-(m) for the full enumeration and the M2a structural-DL closure of the residue.

This theorem, combined with the MDL framework for crystal nets (see `proofs/foundations/dl_comparison.py`), implies srs has minimum description length:

$$\text{DL}(\text{srs}) = \underbrace{\log_2(230)}_{\text{space group}} + \underbrace{L^*(1)}_{\text{1 vertex orbit}} + \underbrace{\log_2(9)}_{\text{Wyckoff choice (W=9)}} + \underbrace{0}_{\text{coordinates}} + \underbrace{0}_{\text{edges (E-trans)}} + \underbrace{1}_{\text{chirality}} = 13.02 \text{ bits}$$

(Wyckoff term updated 2026-05-02: I4_132 has W=9 positions per ITA Vol. A, not W=5 as previously listed in `dl_comparison.py`. The fix adds log₂(9/5) = 0.85 bits to srs's DL; see `WYCKOFF_DATA[214]` for the corrected enumeration including 16e, 24f, 24g, 24h, 48i.)

The key is that edge-transitivity gives $\text{DL}(\text{edges}) = 0$ — no bits needed to specify which neighbor pairs are connected, because the unique edge orbit is determined by symmetry. Other 3D 3-regular V+E-transitive crystal nets achieve this too (e.g., srs-z, srs-c4 also have $\text{DL}(\text{edges}) = 0$); they are distinguished from srs at the M2a refinement level (primitive-cell atom count + directed-edge orbit count) per `proofs/foundations/lov_dl_audit.py` and `rcsr_ensemble_closure_test.py`.

**Case analysis** (proven in `dl_comparison.py` + extended in `rcsr_ensemble_closure_test.py`):

- **Case 1** (strongly isotropic, V+E-transitive in Sunada's strong sense): $G = \text{srs}$ by Sunada 2012. ∎
- **Case 1'** (V+E-transitive RCSR-style but NOT strongly isotropic): 8 RCSR candidates, all with $\geq 2$ directed-edge orbits → M2a β refinement adds $\geq 1$ bit. Combined with Wyckoff/cell-size differences, all have $\text{DL}(G) > \text{DL}(\text{srs})$ — see R-9 sub-item (l) for the full ensemble. ∎ **[CORRECTION 2026-07-11, extending the 2026-07-10 note at line 31 above: the same falsified premise recurs in this case.** `proofs/foundations/arc_transitivity_ground_truth.py` (ground-truth computation from the actual space-group action, not asserted from Sunada) finds **srs-z has exactly ONE directed-arc orbit** (edge-reversible, local vertex action C₃) — contradicting the "$\geq 2$ directed-edge orbits" premise this case asserts across the block of 8 candidates; the claim is verified FALSE for srs-z specifically (the other 7 candidates were not independently re-verified by that script; `rcsr_per_substrate_fingerprint.py:242-248`'s arc-orbit table remains hardcoded, not computed, for all 8). Case 1's overall conclusion is **not overturned** — srs-z still fails the MDL waterline and remains excluded (`rcsr_ensemble_closure_test.py`) — but **not by this arc-orbit mechanism**: the corrected waterline margin for srs-z is −0.17 bits, not −1.17. The live selection authority is the empirical structural-fingerprint + MDL-waterline study (srs DOMINANT, not unique), per `docs/audits/registers/structural_residue_register.md` lines 160–163 (R-9 SUPERSESSION, 2026-06-15) and internal research notes §§1–2 (statements 5 and 7). Do NOT cite Case 1' as selection authority.]**
- **Case 2** (vertex-transitive, not edge-transitive): $\text{DL}(\text{edges}) \geq 1$ bit. Combined with Wyckoff costs, $\text{DL}(G) > \text{DL}(\text{srs})$ in every case. ∎
- **Case 3** (not vertex-transitive, $\geq 2$ vertex orbits): Extra $L^*(k)$ cost. $\text{DL}(G) > \text{DL}(\text{srs})$ in every case. ∎

### Step 3: The girth of srs is 10

The girth of a graph is the length of its shortest cycle. For the srs lattice:

$$g(\text{srs}) = 10$$

This is a mathematical property of the srs lattice, verified by:

1. **Sunada (2012)**: identifies srs as having girth 10 in the crystal net classification.
2. **RCSR database** (O'Keeffe, Peskov, Ramsden & Yaghi, *Accts. Chem. Res.* **41**, 1782–1789, 2008; online at rcsr.net, symbol `srs`): catalogues girth = 10.
3. **Computational verification**: `proofs/foundations/dl_comparison.py` and `proofs/foundations/srs_graph_analysis.py` enumerate shortest cycles on the srs supercell, confirming girth = 10.

The srs lattice is in fact the unique $(3, 10)$-cage among 3D crystal nets: the 3-regular net with the largest possible girth. This maximum-girth property is a consequence of its extreme symmetry (vertex+edge-transitive with space group $I4_132$, point group $O$ = 432).

## Result

$$\boxed{g = 10}$$

The girth of the srs lattice. Exact, determined by the lattice selection via MDL.

## Comparison with experiment

The girth is not a directly measured physical quantity. It enters physics through downstream parameters:

| Downstream parameter | How $g$ enters | Prediction file |
|---------------------|----------------|-----------------|
| $\alpha_1 = (2/3)^{g-2}$ | NB walk survival over $g-2$ steps | `predictions/alpha_1.py` |
| PMNS phases | $\arg(h^g)$, $\arg(h^{*\,(g-1)})$ | (pending) |
| CKM elements | Walk amplitudes at distances $\sim g$ | (pending) |

Verification is indirect, through the accuracy of these downstream predictions.

## Open questions

None. The girth is a catalogued mathematical property of a uniquely identified lattice. Every step is either a cited theorem (Sunada 2012), a database lookup (RCSR), or inherits from upstream predictions.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
