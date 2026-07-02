# δ_r — M_Z tree→pole oblique correction (substrate Δr-analog)

**Parameter:** δ_r ≡ (M_Z_tree − M_Z_pole)/M_Z_tree, the SM-tree-vs-pole
oblique radiative correction on M_Z.
**Canonical script:** `predictions/delta_r.py`
**Audit anchor:** Row P64 (M_Z). Companion of `predictions/delta_rho.py`
(Row P73, δρ) — δ_r is its sign-uniform sibling.
**Status:** THEOREM-GRADE-STRUCTURAL (the c_S Perron-residue piece is
theorem-grade; Clause 7 PASS; Clause 9 PASS — substrate analog, NOT the
SM Sirlin Δr import). Z/Perron channel of the unified-oblique theorem
(`docs/theorems/theorem_unified_oblique.md`).

## 1. Abstract

`predictions/M_Z.py` computes the SM **tree** relation
M_Z = √π·v·√(α₂+(3/5)α₁) ≡ g₂·v/(2cosθ_W) (ρ=1, no oblique). The
decomposition `proofs/foundations/M_Z_residual_is_tree_vs_pole_oblique_
2026-05-15.py` (commit 9501a65) proved — with **exact PDG inputs**
(g₂=0.652, sin²θ_W=0.23121, v=246.22) — that this tree relation
over-predicts the **pole** M_Z by +0.393%, *intrinsically*. That gap is
the tree-vs-pole oblique radiative correction (Δr / ρ-parameter family).
We derive its substrate analog as the Z/Perron eigen-channel of a
single Hashimoto-spectral resolvent (the UNIFIED-OBLIQUE THEOREM,
`docs/theorems/theorem_unified_oblique.md`),
δ_r = (1/12)·α₁_bare/(1−α₁_bare) ≈ +0.338%, which cuts the live M_Z
residual from +0.357% to +0.018% (20×). No fitting; the coefficient
c_S = 1/12 is **DERIVED** here as the gauge-singlet projection of the
B_NB(srs) Perron-eigenvalue residue, c_S = 1/(2|E|) — this replaces the
earlier retracted-Phase-A citation (the parameter_linter Checkpoint-1
provenance blocker).

## 2. Framework inputs invoked

- k* = 3, g = 10 (Rows P4/P9, theorem-grade structural).
- α₁_bare = ((k*−1)/k*)^(g−2) = (2/3)⁸ (Row P1, `predictions/alpha_1.py`).
- c_S = 1/12: **DERIVED** (§3 Step 3) as the gauge-singlet projection of
  the B_NB(srs) Perron-eigenvalue residue, c_S = 1/(2|E|) = 1/(N·k*).
- Master-doc Family-C universal counting template
  (`docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md`).
- Unified-oblique theorem (`docs/theorems/theorem_unified_oblique.md`;
  probe `proofs/foundations/unified_oblique_one_resolvent_2026-05-16.py`).

## 3. Derivation

**Step 1 — the residual is the tree-vs-pole oblique (decomposition).**
Decomposition Pt1 (`M_Z_residual_decomposition_diagnostic_2026-05-15.py`,
ffa89dc): ∂lnM_Z/∂lnM_unif ≈ −0.004 (M_unif-insensitive) and 2-loop-β
makes M_Z *worse* — so the residual is neither M_unif nor a running
issue. Pt2 (9501a65): with exact PDG inputs the SM tree relation itself
over-predicts the pole by +0.393%. ∴ the residual is the SM tree-vs-pole
**oblique** correction (Δr family).

**Step 2 — it is the sign-uniform sibling of δρ (Phase C).**
Phase C established ρ ≡ m_W²/(M_Z²cos²θ_W) = (1/2)·(Π_W/Π_Z), where the
Hashimoto operator B_NB(srs) is Ramanujan-saturated (|h_P|²=k*−1). The
W residue (h_P, phase) carries the custodial-breaking δρ (Row P73). The
**Z residue (Perron, real) is custodial-symmetric and cancels in the ρ
ratio** — which is why δρ never used it. But that same Z-Perron
sign-uniform self-energy residue IS the absolute-M_Z oblique shift: it
lowers M_Z (and m_W) together, cancelling only in the *ratio*. So:

$$\text{one Hashimoto object} \;\Rightarrow\;
  \begin{cases}\Pi_W\ (h_P\text{-phase}) \to \delta\rho & (\text{Row P73})\\
               \Pi_Z\ (\text{Perron-real}) \to \delta_r & (\text{this file})\end{cases}$$

**Step 3 — c_S = 1/12 DERIVED as the Perron-residue singlet projection.**
(`proofs/foundations/unified_oblique_one_resolvent_2026-05-16.py`,
Part 2; `docs/theorems/theorem_unified_oblique.md` §3.2. This replaces
the earlier "Phase-A two-routes, cited" provenance — the source probe
`family_E_phase_A_S_scale_gauge_2point_2026-05-15.py` is retracted at
its file head for stale base predictions, flagged by the
parameter_linter Checkpoint-1 triage as a Clause-1 provenance break.)

The Perron eigenvector of B_NB(srs) at Γ is the **uniform** directed-edge
vector: every directed edge has exactly k*−1 non-backtracking
continuations, so B_NB·**1** = (k*−1)·**1** (verified to machine
precision; srs is edge-regular so **1** is the left Perron eigenvector
too). The neutral-Z gauge vertex is the species-singlet channel; the
rank-1 Perron spectral projector P = |**1**⟩⟨**1**|/⟨**1**|**1**⟩
projected onto the unit singlet ŝ = **1**/√(2|E|) has weight **exactly**

$$c_S = \frac{\langle\hat s|P|\hat s\rangle}{2|E|} = \frac{1}{2|E|}
      = \frac{1}{12}.$$

The two historical readings are the SAME number by the **handshake
lemma** 2|E| = Σ_v deg(v) = N_atoms·k*:

- Route H (NB Hilbert-dim normalization): 1/(2|E|) = 1/12.
- Route C (cycle-counting): k*/(N_atoms·k*²) = 1/(N·k*) = 1/12.

Route H ≡ Route C is a *graph identity* (2|E|=N·k*=12), **not** a
numerical coincidence and **not** a fit — no v_Higgs target enters.

**Step 4 — apply the master-doc Family-C counting template.**
g_physical = g_bare·(1 − c·α₁_bare/(1−α₁_bare)) on the M_Z 2-point
(a mass observable):

$$\boxed{\;M_{Z,\rm pole} = M_{Z,\rm tree}\,(1-\delta_r),\qquad
   \delta_r = c_S\cdot\frac{\alpha_{1,\rm bare}}{1-\alpha_{1,\rm bare}}
            = \frac{1}{12}\cdot\frac{(2/3)^8}{1-(2/3)^8}\;}$$

## 4. Result

δ_r = (1/12)·(256/6561)/(1−256/6561) = (1/12)·(256/6305) ≈ **+0.33836%**.

Applied to the live tree M_Z = 91.5135 GeV:
M_Z_pole = 91.5135·(1−0.0033836) = **91.2039 GeV**.

## 5. Comparison with experiment

| | value |
|---|---|
| M_Z tree (live, ρ=1) | 91.5135 GeV (+0.3574% vs PDG) |
| δ_r | +0.33836% |
| **M_Z pole = tree·(1−δ_r)** | **91.2039 GeV (+0.0179%)** |
| PDG 2024 | 91.1876 ± 0.0021 GeV |

Relative residual cut **20×** (+0.357% → +0.018%). In σ_PDG it remains
≫1σ (M_Z is measured to 2.3 ppm) — the framework's intrinsic structural
precision floor, shared by the whole gauge cluster; reported honestly in
σ_PDG, no σ_theory.

Accuracy of the δ_r mechanism vs the tree→pole gap it must remove:
−5.3% relative vs the framework-input gap (+0.3384% vs +0.3573%),
−13.8% vs the exact-PDG gap (+0.393%) — the same δρ-comparable
structural grade (δρ was +4.58% rel).

## 6. Linter quality gate

- **Clause 7 (rigor):** PASS. c_S=1/12 **DERIVED** as the B_NB
  Perron-residue gauge-singlet projection 1/(2|E|) (Route H ≡ Route C
  by the handshake lemma 2|E|=N·k*; §3 Step 3) — the retracted-Phase-A
  provenance gap is closed. Counting Family-C template (Type-4);
  K-rational ∈ ℚ⊂K; O9-respecting; no fitting; no σ_theory.
- **Clause 9 (no bridge-attribution):** PASS. δ_r is the substrate
  Hashimoto-spectral analog (Phase-A/Phase-C mechanism), **NOT** the SM
  Sirlin Δr number — citing that continuum 2-loop QFT value is the
  bridge-attribution anti-pattern explicitly retracted (commit 4ce4d5c).
- **Grade:** *mathematically complete* (relies on the Family-C counting
  template at the Type-3 EW tier, same as the rest of the EW sector).
- **Clause 8:** relative residual on M_Z +0.357%→+0.018%; σ_PDG still
  ≫1 (intrinsic precision floor, not a missing mechanism).

## 7. No double-count

α_GUT carries its own dark correction (Row P40, c=1/k*) but that is a
**vertex-level** correction on the unification coupling at M_unif. δ_r
is a **propagator-level** correction on the M_Z 2-point. Different
sector (master-doc vertex-vs-propagator meta-classification) — no
double-count. `predictions/M_Z.py` applies no other 2-point oblique
(it is pure SM tree before δ_r).

## 8. Open questions

- The +5.3% gap between δ_r and the framework tree→pole gap is the same
  un-derived sub-leading-spectral residual flagged for δρ (beyond the
  leading h_P/Perron residue). Honest, named, not fitted.
- δ_r and δρ being two eigen-channels of one Hashimoto resolvent is
  now **written as a theorem** (`docs/theorems/theorem_unified_oblique.md`,
  2026-05-16): Π_Z→Perron→δ_r and Π_W→h_P→δρ from the one
  G_NB=(I−u·B_NB(srs))⁻¹. The remaining open item is narrower: the
  Perron-dominance-vs-h_P-subdominance argument *explains* which
  observable class takes the Family-C (resummed) vs Family-E (leading)
  form but is a **structural argument, not a from-resolvent computation**
  of the resummation. Upgrading that (expanding ⟨V|G_NB|V⟩ per channel
  and showing the Perron channel geometrically resums while the h_P
  channel terminates at leading order) would lift the form-selection
  from "consistent with the master-doc rule" to theorem-grade
  (theorem doc §6.1).
