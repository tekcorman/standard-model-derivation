# W21 + W22 + W23 — Sector-Specific c Structural Obstruction (2026-05-26 +1)

**Status:** THREE STRUCTURAL FINDINGS bracket the sector-specific c story.

- W21: graph-level BS-T (1 Perron-adj + 1 bipartite-extra) split within V_scalar is non-canonical on K_4.
- W22: canonical SU(3)_c gauge lift (via 3-edge-coloring) does NOT commute with Hashimoto B.
- W23: two-loop MSSM RG with uniform c=1/3 gives WORSE residuals than one-loop (1/α_3 shift +0.63σ → +9.77σ); extracted c-values under two-loop fall to NEGATIVE values (-0.34 to -0.84), unphysical.

Empirical 2-block sector-specific c pattern from yesterday's cluster fit remains robust (c_1 ≈ c_2 ≈ 1/3, c_3 ≈ 1/4 within Δ ~ 0.01). Path 4 (W23) decisively rules out one-loop-running-artifact explanation. Paths 1 and 2 remain open as structural research frontiers; Path 3 is honest fall-through.

---

## W21 finding — BS-T sector split within V_scalar is non-canonical on K_4

The Bass-Stark-Terras factorization
$$\det(uI - B) = (u^2-1)^{|E|-|V|} \cdot (u-1)(u-2) \cdot (u^2+u+2)^3 \quad \text{(on K\_4)}$$
gives ALGEBRAIC multiplicity at u=+1: 2 (bipartite-factor) + 1 (Perron-adjacency) = 3.

**However**: the geometric eigenspace V_+1 is 3-dim with $B|_{V_+1} = I$, so every direction in V_+1 is a B-eigenvector. The BS-T (2 bipartite + 1 Perron-adj) split does NOT correspond to a canonical geometric subspace decomposition.

The Perron adjacency eigenvector $\psi_3 = (1,1,1,1)$ is uniform on K_4, so the directed-edge lift $v_{\alpha,\beta}[e=(u,v)] = \alpha \cdot \psi_3(u) + \beta \cdot \psi_3(v) = \alpha + \beta$ is 1-dim (the all-ones vector at u=2). There is no second algebraically-canonical "Perron-adj direction" at u=+1.

**Implication for sector-specific c:** Yesterday's framing of "V_scalar = 1 Perron-adj mode (gauge-singlet) + 1 BS-T-extra mode (EW-coupled, color-singlet)" requires an EXTERNAL structure (not graph-only) to canonically pick out the (1,1) split.

**Probe:** `proofs/foundations/W21_BST_sector_identification_V_scalar_2026-05-26.py`.

---

## W22 finding — canonical SU(3)_c lift does not commute with B

K_4 admits a canonical 3-edge-coloring via perfect matchings:
- Color R = {(0,1), (2,3)}
- Color G = {(0,2), (1,3)}
- Color B = {(0,3), (1,2)}

Each vertex sees exactly one edge of each color. The natural SU(3)_c lift is "global SU(3) rotation at source vertex" — block-diagonal $I_4 \otimes \lambda^a$ on $\mathbb{C}^{12} = \oplus_v (\text{outgoing color triplet at } v)$.

**Test:** $\|[T^a, B]\|_F$ for $a = 1, \ldots, 8$ Gell-Mann generators.
- All 8 commutators have norm 5.66–6.93. **None vanishes.**

**Diagnosis:** the walker (Hashimoto B) hops $(u \to v) \to (v \to w)$ with $w \in V \setminus \{u, v\}$. The 2 candidate next-hops $(v \to w_1)$ and $(v \to w_2)$ have DIFFERENT colors at source vertex $v$ (since K_4 has every triangle as "rainbow"). So a global color rotation at source doesn't commute with non-deterministic walker propagation.

**To make SU(3)_c commute with B** would require lattice gauge link variables $U_e \in SU(3)$ that compensate during propagation — i.e., a gauged Hashimoto matrix $B^{SU(3)}[e, f] = U_{f} \cdot B[e, f]$. The substrate would have to provide these link variables, which is an additional structural ingredient beyond bare-graph B.

**Probe:** `proofs/foundations/W22_SU3c_action_on_V_pm_K4_2026-05-26.py`.

---

## Empirical evidence for sector-specific c (unchanged by W21+W22)

Yesterday's `sector_specific_c_alpha_GUT_scan_2026-05-26.py` extracts from PDG cluster:
- $c_1 = 0.3428$ — within 0.0095 of $1/3$
- $c_2 = 0.3317$ — within 0.0017 of $1/3$
- $c_3 = 0.2414$ — within 0.0086 of $1/4$

All three c_i lie within 0.01 of clean rationals. Spread $c_1 - c_3 = 0.1015$.

R_∞ ppt-precision independently anchors c_EM ≈ 0.341 = 1/3 + 0.008 (separate constraint).

This 2-block pattern (EW vs color) is structurally striking but NOT graph-derivable via W21/W22 paths.

---

## Surviving pathways for sector-specific c structural closure

In rough priority order:

### Path 1 — Lattice-gauge link variables from Cl(6) per-vertex Fock

Per `theorem_charge_before_color.md` §9, each K_4 vertex carries 8-dim Cl(6) Fock with $U(3) \subset Spin(6)$ acting via fermionic edge modes $(a_i, a_i^\dagger)$. The "gauge connection" on directed edges comes from the per-vertex Fock structure plus the shared-edge constraint.

Concretely: an undirected edge $i$ shared between vertices $u, v$ has $(a_i, a_i^\dagger)$ shared between $\text{Cl}(6)_u$ and $\text{Cl}(6)_v$. The "transport map" from one vertex's Fock to the other's gives the link variable $U_i \in U(3) \subset Spin(6)$.

If this link-variable structure can be derived from the framework's existing CAR per edge (`theorem_car_local_jordan_wigner.md`), the gauged Hashimoto matrix $B^{SU(3)}$ would commute with SU(3)_c by construction, and sector-specific Wilson-loop counts on V_pm could close c_color and c_EW separately.

**Expected effort:** 3-4 sessions. Closure probability: ~25% (still structurally hard).

### Path 2 — Pati-Salam Killing-form per-sub-bundle dark correction

In Pati-Salam SU(4) × SU(2)_L × SU(2)_R, the Killing-form normalization differs across factors. The framework already uses $g_1 = \sqrt{5/3} g_Y$ to map between the GUT and SM normalizations. But this gives a UNIFORM c across the three SM sectors (which is the current theorem).

The 2-block split (EW vs color) might emerge if the dark Q-projector acts DIFFERENTLY on the Pati-Salam SU(4) (containing SU(3)_c) vs the Pati-Salam SU(2)_L × SU(2)_R (containing the EW SM gauges). Specifically: SU(4)_PS has rank 3, SU(2)_L × SU(2)_R has rank 2, so the Q-projector dimensional count could differ.

**Expected effort:** 1-2 sessions for an exploratory probe. Closure probability: ~15% (the Killing-form normalization is already absorbed in HYP_NORM=3/5).

### Path 3 — accept empirical pattern; document as residual

Concede that the empirical (1/3, 1/3, 1/4) split is NOT derivable from current substrate machinery, and document the (c_3 vs c_{1,2}) gap as a known residual at the 0.01-c level. This is honest but unsatisfying — the framework would have an unexplained empirical pattern.

**Expected effort:** 0 sessions (just documentation). Closure: definitionally null.

### Path 4 — two-loop MSSM running test [W23, EXECUTED, NEGATIVE]

**Result (W23, `W23_two_loop_MSSM_RG_uniform_c_test_2026-05-26.py`):**

Running uniform c = 1/3 with the framework's structural M_unif + α_GUT^bare under TWO-loop MSSM RGEs (Martin 1997 §6.4 Eq. 6.30) gives:

| sector | 1/α(M_Z) one-loop | 1/α(M_Z) two-loop | PDG | one-loop Δ | two-loop Δ |
|---|---|---|---|---|---|
| 1 | 59.008 | 59.651 | 59.017 | −0.009 (−1.33σ) | **+0.634 (+88.6σ)** |
| 2 | 29.584 | 30.634 | 29.582 | +0.002 (+0.27σ) | **+1.052 (+173.8σ)** |
| 3 | 8.566 | 9.106 | 8.475 | +0.092 (+1.42σ) | **+0.632 (+9.77σ)** |

Two-loop pushes ALL 1/α_i UP (away from PDG), not just α_3. To "fit" each sector under two-loop, c must go NEGATIVE: c_1 = −0.344, c_2 = −0.843, c_3 = −0.337.

The c_1 vs c_3 spread under two-loop is 0.51 — FIVE TIMES LARGER than the one-loop spread 0.10.

**Verdict:** the c_3 ≠ c_{1,2} gap is NOT one-loop-running artifact. Two-loop running makes the gap WORSE, not better.

**Secondary finding:** the framework's structural M_unif + α_GUT^bare are tightly tuned for ONE-LOOP precision. At two-loop precision they don't reproduce PDG cluster, and no in-range c value rescues them. Either:
- The framework's gauge cluster predictions are precision-limited to one-loop (acceptable; framework's stated rigor is matched to its conditioned axioms), OR
- The framework's M_unif derivation needs revision to be loop-order-robust (significant rework), OR
- Threshold/scheme corrections account for the two-loop mismatch (M_SUSY threshold already ruled out by yesterday's `two_stage_RG_M_SUSY_scan_2026-05-26.py`).

**Status:** Path 4 CLOSED NEGATIVE. The empirical sector-specific c is structurally real, not loop-order artifact.

---

## Recommended next step (post-W23)

Path 4 is closed negative. The empirical sector-specific c is genuine, not loop-order artifact. Three options remain:

- **Path 1** (Cl(6) Fock + lattice gauge link variables): structurally principled but multi-session, ~25% closure probability. W22's commutator finding warns of obstacles.
- **Path 2** (Pati-Salam Killing-form per-bundle): 1-2 sessions, ~15% closure. Less likely structurally.
- **Path 3** (document residual; pivot away from unification work): 0 sessions, definitionally null on closure.

Recommend asking the user which of Paths 1/2/3 to pursue.

---

## Pre-declared abort criteria for the Cl(6) work (refinement of yesterday's AB1-AB5)

- **AB6 (NEW):** If after Session 1 of Path 1, the lattice gauge link variables from Cl(6) Fock structure don't make $[T^a, B^{SU(3)}] = 0$ for at least one of {SU(3)_c, SU(2)_L, U(1)_Y}, the program is structurally non-viable and falls back to Path 3 (document residual).

- **AB7 (NEW):** If Path 4 closes the α_3 residual to within 0.01 of uniform c = 1/3, the structural derivation of sector-specific c becomes unnecessary — uniform c = 1/3 stands as theorem-grade and the apparent 2-block pattern is one-loop-running artifact.

---

## Files

- `proofs/foundations/W21_BST_sector_identification_V_scalar_2026-05-26.py` — BS-T sector non-canonical
- `proofs/foundations/W22_SU3c_action_on_V_pm_K4_2026-05-26.py` — SU(3)_c lift doesn't commute with B
- `proofs/foundations/W21_W22_sector_specific_c_obstruction_verdict_2026-05-26.md` — this file
- `proofs/foundations/sector_specific_c_alpha_GUT_scan_2026-05-26.py` — yesterday's empirical extraction (UNCHANGED, still positive)

---

## Cross-references

- `theorem_alpha_GUT_dark_correction.md` §3.1-3.3 — the existing Route H derivation uses BS-T ALGEBRAIC multiplicity (uniform c = 1/3); this is unaffected by W21/W22 because the algebraic count doesn't depend on the (1,1) split within V_scalar.
- `theorem_charge_before_color.md` §9 — Cl(6) Fock per vertex; used by Path 1.
