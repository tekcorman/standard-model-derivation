# Yukawa exponent principle — master derivation (rigor-stratified)

**Date:** 2026-05-20.
**Status:** CONSOLIDATION-GRADE WITH MAJOR FINDING (§11) — the "exponent
principle" as stated is a **post-hoc unification**, NOT a derived master
mechanism. W8 probe (`proofs/foundations/W8_exponent_principle_consistency_2026-05-20.py`)
shows the formula `y_X = prefactor × (2/3)^(n_free·(g-2)) / k^(edge_sel)`
does NOT cover y_ν with consistent (prefactor, n_free, edge_sel) assignments:
the actual framework computation of y_ν uses spectral radius L_us = 2+√3,
not α₁; the "y_ν = α₁/k" docstring identification is 3 orders of magnitude
off the actual computed value. The Yukawa master theory as a single template
DOES NOT YET EXIST in the framework. What exists is 3 separately-motivated
derivations with shared structural motifs. Read §11 first for the final
verdict; §1–§10 below preserve the original framing for provenance.

**Why this document.** Three prior session-attempts to "master-theory the
Yukawas" via Koide-shape extension + per-process A2-T + sector-distinguishing
channel factors were misframed (deleted,. The framework's
actual master mechanism — the exponent principle stated in `srs_tan_beta.py`
PART 1 and applied at commit `66c8836` to derive y_t — is what should
underlie any consolidation. This document traces its dynamics through the
framework's standard layers and characterizes rigor honestly.

**Precision-honesty disclaimer.** The y_τ +0.13% and m_t +0.82% residuals
are STRUCTURALLY small but OBSERVATIONALLY large (m_τ pole precision ~10 ppm;
m_t pole precision ~0.17%). Both predictions are *falsified at observational
precision*. The framework's claim is theorem-grade at the leading STRUCTURAL
FORM, conditional on identified upstream open questions. Calling either
"theorem-grade at observational precision" is overclaim.

---

## 1. The exponent principle (the framework's actual master template)

Per `proofs/masses/srs_tan_beta.py` PART 1, verbatim:

$$y_X \;=\; (\text{prefactor}) \times (2/3)^{n_{\text{free}} \cdot (g-2)} \;/\; k^{\#\text{edge selections}}$$

where:
- **k = 3** (coordination number, theorem-grade per `predictions/k_star.py`)
- **g = 10** (girth of srs, theorem-grade per `predictions/g_girth.py`)
- **2/3 = (k−1)/k** (NB walker survival per step on srs)
- **n_free** = count of independent girth-cycle modes above MDL waterline that the species' quantum numbers leave *free* to be sampled
- **# edge selections** = number of fermion-line edge selections at the trivalent vertex (= number of independent A5(b) MDL probability factors)
- **prefactor** = universal cycle-amplitude pieces (the 5/3 chirality factor from tan²(arg h) at Bloch P-point; any structural multiplicities)

The mechanism's per-channel variability lives entirely in `(n_free, # edge selections, prefactor)`. The remaining gate is: from a species' quantum numbers, mechanically derive n_free.

---

## 2. Dynamics chain — substrate ↔ observer

The framework's standard two-sided reading places the Yukawa coupling on the
observer side (it appears in the observer's compressed Lagrangian as the
coefficient of ψ̄_L H ψ_R) and sources it from substrate-level multiway
dynamics. The chain has well-defined layers:

```
SUBSTRATE SIDE                                OBSERVER SIDE
─────────────                                ─────────────

multiway DAG  ──┐                                ┌──  y_X coupling in Lagrangian
(full A1 +      │                                │    (compared to m_X/v · √2 obs)
 branching      │                                │
 over all       │                                │
 toggle paths)  │                                │
                │                                │
substrate       │       A2-T waterline           │    A5(b) identification
states ─────────┴──→  (Csiszár I-projection: ───→┴──  (MDL prob = coupling
                       discard sub-waterline,         strength)
                       retain above)
                              │
                              ▼
                       (graph layer:
                        srs Hashimoto/
                        Bloch operators
                        capture the
                        compressed spectrum)
                              │
                              ▼
                       (compression
                        layer: F_inv(E)
                        → srs per
                        theorem_F_inv_E_
                        to_srs_compression.md)


DARK SECTOR                                  FAMILY D CORRECTION
───────────                                  ──────────────────

multiway content ──┐
discarded by A2-T  │  (Feshbach contour       multiplicative correction
projection         │   integral on the     ───→  to observer-side y_X
                   │   substrate complement)        (theorem-grade for τ
                   │                                via master doc §3(D))
substrate
non-perturbative  ─┘
content
```

**The Yukawa coupling y_X is the projection of substrate's multiway
chirality-flip retention onto the observer's compressed Lagrangian.**

### 2.1 What each layer contributes (rigor stratification)

| Layer | What it contributes to y_X | Rigor level |
|---|---|---|
| Toggle (A1) | binary self-inverse edge dynamics; defines the multiway state space | axiom (Type 1) |
| Multiway (A1 + branching) | full state-space of toggle sequences | derived (theorem-grade per multiway scoping) |
| Compression (A2-T waterline) | discards sub-waterline content; retains above-waterline | derived (theorem-grade via `theorem_A2_mdl_from_finite_register.md` + Csiszár I-projection) |
| Substrate → srs identification | substrate's compressed graph is srs (Sunada arc-transitive unique) | theorem-grade per `theorem_F_inv_E_to_srs_compression.md` (R-9 closure) |
| Graph (srs Hashimoto/Bloch) | NB walker survival = (2/3)^(g-2); chirality = tan²(arg h) at P-point | theorem-grade per `theorem_bloch_lift_mu.md` + Ramanujan saturation |
| A5(b) identification | MDL probability of above-waterline NB walk representation = physical coupling strength | **identification, NOT derivation** (Type 1 axiom; A3-T elimination open per Stage 5 of axiom roadmap) |
| Observer Lagrangian | y_X · ψ̄_L H ψ_R in observer's compressed model | observer-side construction (Type 4 via Peskin-Schroeder + Langacker) |
| Dark (Family D) | multiplicative correction g_phys = g_bare × (1 − c_g · α₁/(1−α₁)) | theorem-grade for y_τ (c=1 per master doc §3(D)) |

**The rigorous core**: NB walker survival × edge-selection MDL probability × Family D dark correction. All theorem-grade upstreams.

**The structural identification**: A5(b) connects substrate-side MDL probability to observer-side coupling. This is a framework axiom (not derived); it sets the IDENTIFICATION layer between substrate and observer. Stage 6 axiom roadmap targets relocating A5 to "empirical observer-specification" — open multi-session research.

**The open piece** (per commit 66c8836 itself): the explicit V_Ram ≅ Cl(6)-Fock identification that mechanically derives the n_free count per sector from quantum numbers. This is Need-D-3 / R-14.

---

## 3. The three derived channels (y_τ, y_ν, y_t) — stratified rigor

### 3.1 y_τ — gen-1 charged lepton, 2 edge selections, n_free = 1

**Formula** (per `theorem_ytau_corollary.md`):

$$y_\tau \;=\; \frac{\alpha_{1,\text{full}}}{k^2} \;=\; \frac{(5/3)(2/3)^8}{9} \;=\; \frac{1280}{177\,147} \;\approx\; 7.2256 \times 10^{-3}$$

**Stratification:**

| Component | Value | Rigor |
|---|---|---|
| NB walker survival (2/3)^(g−2) = (2/3)⁸ | 256/6561 | theorem-grade (Bloch decomposition at P-point) |
| Chirality factor 5/3 = tan²(arg h) at P-point with h = (√3+i√5)/2 | 5/3 | theorem-grade (Ramanujan saturation \|h\|² = k−1 = 2) |
| α₁_full = (5/3)·(2/3)⁸ | 1280/19683 | theorem-grade |
| 1/k² — two fermion edge selections (i_in, i_out) via A5(b) joint MDL | 1/9 | A5(b)-identification (axiom-grade) + bijection arithmetic |
| Higgs edge factor 1 — bijection at trivalent vertex | 1 | theorem-grade (W2 verdict 2026-05-19) |
| Cl(2) channel factor 1 — per-process A2-T reading | 1 | A2-T-identification (axiom-grade with explicit waterline argument) |
| Family D correction −(5/12)·α₁/(1−α₁) | −0.127% on y_τ | theorem-grade (master doc §3(D)) |

**Result vs observation:** y_τ_pred = 7.226e-3 vs y_τ_obs = m_τ_pole/v = 7.217e-3. **Residual: +0.13%, structurally small but ~10000σ at PDG precision of m_τ.** The residual is attributed in `theorem_ytau_corollary.md` §8 to "O(α_s, v_hierarchy) corrections expected at tree level" — but this attribution is hand-waved; the corrections themselves are not derived from substrate.

**Honest characterization:** y_τ = α₁_full/k² is theorem-grade at STRUCTURAL FORM. The +0.13% residual is the un-derived sub-leading dark/Feshbach content. **Falsified at observational precision; suggestive at structural-form precision.**

### 3.2 y_ν Dirac — delocalized state, 1 edge selection, n_free per `srs_neutrino_mass_scale.py`

**Formula** (per `srs_neutrino_mass_scale.py` L28-32):

> "Neutrinos are |000⟩ Fock states (delocalized, no edge structure). The coupling is GLOBAL, not edge-local like y_τ = α₁/k². For delocalized states: y_ν = α₁/k (one less edge resolution than edge-local y_τ = α₁/k²)."

**Stratification:** the 1/k vs 1/k² difference comes from delocalization removing one of the two edge-selection MDL probabilities. The remaining structural form follows y_τ's chain.

**Caveat (load-bearing for honesty):** `srs_neutrino_mass_scale.py` PART 3 actually computes `y_ν = (k−1)/k · √(L_us/k)` for the seesaw, NOT the bare `α₁/k`. The "α₁/k" form is cited in the file's docstring as the conceptual identification but the seesaw application uses a different L_us-weighted form. The exponent-principle reading per commit `66c8836` cites the docstring identification (y_ν = α₁/k). **The bare-form-vs-applied-form distinction within this channel is itself a sub-leading rigor question.**

The neutrino seesaw delivers m_ν3 ≈ 0.050 eV at +0.87% (per current m_ν3 derivation), with Majorana scale M_R = (2/3)^g · M_GUT.

**Honest characterization:** the y_ν exponent-principle ASSIGNMENT (1 edge selection vs 2) is structurally identified but the actual seesaw computation uses a more elaborate form. Channel is "structurally identified" rather than "theorem-grade-derived" in the exponent principle's strict reading.

### 3.3 y_t — gen-3 quark, 0 edge selections (asserted), n_free → 0 (asserted)

**Claim** (per commit 66c8836):

$$y_t(M_{\rm GUT}) = 1 \quad \Longrightarrow \quad m_t({\rm tree}) = v/\sqrt{2} \;=\; 174.104 \text{ GeV} \;\;(+0.82\%)$$

**Convention flag** (added 2026-05-20 per W25 convention audit, an internal working note). The `y_t = 1` quoted here is in **SM Peskin convention** (`m = y · v/√2`). In the framework's *operational* convention (`m = y · v`, used by `predictions/y_tau.py`, `predictions/m_tau.py`, and `theorem_ytau_corollary.md` §10), the equivalent value is `y_t_FW = 1/√2 ≈ 0.7071`. The numerical mass match `m_t = 174.104 GeV` at +0.82% is convention-invariant. Any future `predictions/m_top.py` should compute as `m_t = y_t_FW · v = 0.7071 · 246.22 = 174.104 GeV` to stay in the framework's operational convention. See §11.6.3 for the structural significance of this √2 convention switch (it is the concrete witness for the exponent principle's "post-hoc unification" retraction).

**The structural argument:** "the top is the n=2 Hamming sector — two active toggle modes at a Pati-Salam color-triplet, SU(2)_L doublet species point. The combination is maximally above-waterline; MDL waterfilling places every girth-cycle mode above the waterline (fixed by the quark's quantum numbers), so the free-mode count → 0 → exponent (2/3)⁰ = 1."

**Stratification:**

| Component | Value | Rigor |
|---|---|---|
| Tree-level form m_t = v/√2 if y_t = 1 | 174.10 GeV | tree-level Lagrangian arithmetic (theorem-grade) |
| y_t = 1 from "gen-3 limit n_free → 0" | 1 | **ASSERTION**, not derivation. The mechanical V_Ram ≅ Cl(6)-Fock identification is OPEN per the commit (= Need-D-3 / R-14). |
| Family D δy_t/y_t = −(5/6)α₁² | −0.127% | theorem-grade (same vertex topology as y_τ, master doc §3(D)) |
| Sub-leading α_s-propagated residual (M_unif threshold conditional) | +0.534% | conditional on M_unif threshold corrections (the same upstream g_1/g_2/g_3 cite) |
| Sub-sub-leading remainder | +0.157% | un-derived |

**Result vs observation:** m_t_pred (post-D) = 173.883 GeV vs m_t_pole = 172.69 ± 0.30 GeV. **Residual: +0.69% ≈ 4σ at PDG precision.** Decomposes coherently: Family D + α_s-propagated (M_unif conditional) + sub-leading.

**Honest characterization:** y_t = 1 is the framework's actual top-mass derivation, theorem-grade-conditional at STRUCTURAL FORM (the gen-3-limit assertion). **Observationally falsified at PDG precision** (+0.82% vs ~0.17% precision). The residual decomposition is itself a structural finding — it identifies M_unif threshold as the load-bearing open conditional shared with the gauge-coupling cluster.

---

## 4. The nine open channels — blocked on one identified gate

All other charged Yukawas (y_b, y_s, y_d, y_c, y_u, y_μ, y_e) and the two
light-neutrino Dirac Yukawas (y_ν2, y_ν1) are NOT derived via the exponent
principle. Per commit 66c8836's own characterization:

> "the explicit V_Ram ≅ Cl(6)-Fock identification that mechanically derives
> 'gen-3 → 0 free modes'; that is Need-D-3 / R-14."

**The single open gate** is: from a species' quantum numbers (Hamming weight
n, generation index j, color rep, chirality, U(1)_Y doublet partner),
mechanically derive n_free. Once derived, the exponent principle drops out
y_X for each channel.

| Channel | Quantum content | n_free | Status |
|---|---|---|---|
| y_e (gen-1 charged lepton, j=0) | n=3, color singlet | ? | open via V_Ram ≅ Cl(6)-Fock |
| y_μ (gen-2 charged lepton, j=2 in Koide convention) | n=3, color singlet | ? | open (or via Koide-shape from y_τ — within-sector closure) |
| y_τ (gen-3 charged lepton, j=0) | n=3, color singlet | 1 (gives (2/3)⁸) | derived per §3.1 |
| y_ν1 (gen-1 ν Dirac) | n=0, color singlet, delocalized | ? | open |
| y_ν2 (gen-2 ν Dirac) | n=0, color singlet, delocalized | ? | open |
| y_ν3 (gen-3 ν Dirac) | n=0, color singlet, delocalized | 1 (gives α₁/k) | per §3.2 (structurally identified) |
| y_d (gen-1 down quark) | n=1, color triplet | ? | open |
| y_s (gen-2 down quark) | n=1, color triplet | ? | open |
| y_b (gen-3 down quark) | n=1, color triplet | ? | open — possibly intermediate between y_τ and y_t |
| y_u (gen-1 up quark) | n=2, color triplet | ? | open |
| y_c (gen-2 up quark) | n=2, color triplet | ? | open |
| y_t (gen-3 up quark) | n=2, color triplet | 0 (gen-3 limit assertion) | per §3.3 (structurally identified) |

**Within-sector closure** via Koide shape from y_τ already exists for y_μ
and y_e (per `theorem_ytau_corollary.md` Corollary 2). The down-sector
within-sector shape and up-sector within-sector shape are governed by Row
P37's (ε²_up−2)/(ε²_down−2) = 14/5 theorem (`predictions/koide_quark_ratio_derivation.md`)
and the within-sector ε² values (R4-pinned for down at ≈[2.47, 2.68]; up
follows from Row P37 ratio). But **the inter-sector hierarchy** (m_t/m_b ≈
36, m_b/m_τ ≈ 2.7) comes from the exponent-principle's per-sector
n_free assignments, not from Koide shape.

---

## 5. Per-layer rigor characterization

### 5.1 Toggle layer (A1)

**Axiomatic.** A1 establishes binary self-inverse edge toggling.
Multiway branching is the full state space.

Rigor: Type 1 axiom.

### 5.2 Graph layer (srs Hashimoto / Bloch)

**Substrate-derived.** Per `theorem_F_inv_E_to_srs_compression.md`
(2026-05-05, theorem-grade-conditional on Sunada arc-transitivity), the
substrate's compressed form IS the srs Laves lattice — forced by
(A1+A2-T) → strong isotropy → Sunada 2012 uniqueness.

NB walker survival on srs: `α₁_bare = (k−1)^(g−2)/k^(g−2) = (2/3)^8` for k=3,
g=10. Theorem-grade per `predictions/alpha_1.py` + Bloch decomposition.

Chirality factor: 5/3 = tan²(arg h) at Bloch P-point with h = (√3+i√5)/2.
Theorem-grade per Ramanujan saturation |h|² = k−1 = 2 + algebraicity meta-
theorem.

**Rigor at this layer: theorem-grade for the structural form (2/3)^(g−2)
and the 5/3 chirality factor.**

### 5.3 Compression layer (A2-T waterline)

**Csiszár I-projection.** Per `theorem_A2_mdl_from_finite_register.md`,
A2-T waterline is the unique idempotent I-projection that maps substrate
states to retained-above-waterline configurations.

For Yukawa: the chirality-flip configuration is above waterline iff its
MDL bit-cost is below the cumulative compression budget at the species'
quantum-number constraint level. The waterline depends on which modes are
"fixed" by quantum numbers (this is the n_free count).

**Rigor at this layer: I-projection structure is theorem-grade; the
specific n_free counts per sector are NOT yet derived from this structure.**
This is the open piece.

### 5.4 Observer layer (A5(b) identification + Lagrangian)

**Identification, not derivation.** A5(b) is a Type 1 axiom: "MDL
probability of above-waterline NB walk representation = physical coupling
strength."

The observer-side Yukawa coupling y_X = (computed MDL probability above
waterline) is then a derived quantity GIVEN the A5(b) identification.
The Yukawa-vertex Lagrangian form ψ̄_L H ψ_R is Type 3 standard SM
(Peskin-Schroeder).

**Rigor at this layer: A5(b) is axiomatic; given it, the observer-side
y_X follows from substrate-side MDL probability mechanically. Stage 6 of
the axiom roadmap targets relocating A5 to "empirical observer-
specification" — open multi-session research.**

### 5.5 Dark layer (Family D correction)

**Theorem-grade for y_τ.** Per `theorem_substrate_feshbach_dark_corrections_master.md`,
Family D supplies a multiplicative correction `g_phys = g_bare × (1 −
c_g·α₁/(1−α₁))` to vertex couplings with leading DC absent. For y_τ,
c=1 (master doc §3(D)) gives the +0.13% match.

For other Yukawas: master doc says "Family D cannot help quarks alone —
the bottleneck is upstream of it, in R-14." Family D's per-sector c_g
coefficients for quarks are blocked at the same Need-D-3 / R-14 gate.

### 5.6 Multiway / substrate side

**Hosts the dynamics; provides the discarded content.** The full multiway
DAG includes both retained (observer-side) and discarded (dark-side)
content. The discarded content is what produces Family D corrections via
Feshbach contour integrals on the substrate complement.

**Rigor at this layer: A2-T as substrate Wilsonian RG flow is theorem-grade
per `forward_construction_substrate_renormalization.md`. The explicit
multiway DAG → observer projection chain is theorem-grade-conditional on
M1.B Galois tower (closed 2026-04-28 per `theorem_observer_substrate_iprojection_scoping.md`).**

---

## 6. The honest precision statement

**Theorem-grade STRUCTURAL FORMS (rigorous through substrate + A2-T + A5(b)):**
- `α₁_full = (5/3)(2/3)⁸` — fully derived from substrate dynamics
- `y_τ = α₁_full/k²` — structural form derived, conditional on A5(b)
- `y_t = 1` at gen-3 limit — structurally identified, conditional on V_Ram ≅ Cl(6)-Fock derivation of n_free → 0
- Family D leading correction for y_τ at c=1 — theorem-grade

**STRUCTURAL FORM + open conditional (residual is conditional on identified upstream):**
- y_τ +0.13% residual: conditional on sub-leading Feshbach analog (un-derived; open multi-session research)
- m_t +0.69% (post-Family-D) residual: conditional on M_unif threshold corrections (the same conditional that g_1/g_2/g_3 cite)

**Observationally falsified at PDG precision:**
- y_τ at +0.13% vs m_τ precision ~10 ppm: ~10000σ off
- m_t at +0.82% vs m_t pole precision ~0.17%: ~5σ off

**Open gate (single load-bearing structural question):**
- V_Ram ≅ Cl(6)-Fock identification → mechanical n_free derivation per sector → closes 9 of 12 remaining channels via the exponent principle
- = Need-D-3 / R-14 in the residue register

---

## 7. What rigor would require — per layer

If the goal is to upgrade from "theorem-grade at structural form, falsified
at observational precision" to "theorem-grade at observational precision,"
the work needed is layer-specific:

| Layer | What needs tightening | Bounded? |
|---|---|---|
| Toggle | A1's edge-toggle dynamics are axiomatic; rigorous as stated | already rigorous |
| Graph | Bloch decomposition at high-sym points is theorem-grade; α₁ structural form derived | already rigorous |
| Compression | I-projection structure theorem-grade; per-sector n_free derivation open | **OPEN** (V_Ram ≅ Cl(6)-Fock — Need-D-3) |
| Observer | A5(b) identification axiom-grade; Stage 6 relocation open | research-level |
| Substrate (multiway) | M1.B Galois tower closed; explicit projection theorem-grade-conditional | mostly rigorous |
| Dark | Family D theorem-grade for y_τ; per-sector for quarks blocked | **OPEN** (same R-14 gate) |

**The single load-bearing open piece is V_Ram ≅ Cl(6)-Fock → n_free per
sector.** Closing this:
- Unblocks 9 of 12 channels via the exponent principle
- Provides the sector-specific Family D c_g coefficients
- Reduces all per-channel residuals to the sub-leading Feshbach analog (the
  +0.13% on y_τ-class).

The sub-leading Feshbach analog (the +0.13% residual on y_τ and the
~+0.15% sub-leading on m_t) is a separate research-level open question
about substrate non-perturbative content.

---

## 8. What this consolidation does NOT do

- Does not produce any new Yukawa derivation
- Does not close any ledger row
- Does not change any predictions/*.py
- Does not "theorem-grade-promote" the existing 3 channels — they remain
  at the same precision they were (structural-form-theorem-grade with
  observationally-falsified residuals; conditional on identified open
  questions)
- Does not commit to the exponent principle being the unique mechanism;
  the Koide-shape extension tested in deleted W1/W1-up was falsified at
  value level but the exponent principle's gen-3-limit assertion remains
  itself conditional on V_Ram ≅ Cl(6)-Fock — they're attacking the same
  gate from different sides

What this consolidation DOES:
- Names the framework's actual master mechanism (the exponent principle)
- Stratifies its rigor honestly per layer
- Identifies the single load-bearing open gate (V_Ram ≅ Cl(6)-Fock → n_free)
- Surfaces the precision-honesty disclaimer (structural-form vs observational-
  precision)
- Provides the per-channel status across all 12 fermion species

---

## 9. Cross-references

**Framework infrastructure:**
- `proofs/masses/srs_tan_beta.py` PART 1 — exponent principle source
- `proofs/masses/srs_neutrino_mass_scale.py` — y_ν derivation chain
- `docs/theorems/theorem_ytau_corollary.md` — y_τ 4-factor template
- `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md` — dark master
- `docs/theorems/theorem_F_inv_E_to_srs_compression.md` — substrate → srs (R-9 closure)
- `docs/theorems/theorem_observer_substrate_iprojection_scoping.md` — M1.B Galois tower closure
- `docs/theorems/theorem_A2_mdl_from_finite_register.md` — A2-T waterline
- `docs/theorems/theorem_bloch_lift_mu.md` — Ramanujan saturation
- `docs/audits/registers/structural_residue_register.md` R-14 — Pati-Salam quark/lepton differentiation (the gate)
- Commit 66c8836 (2026-05-20) — y_t = 1 via gen-3 limit; the +0.69% residual decomposition
- Commit 775c39c (2026-05-20) — n=2 directed-walk seam; up≈down at static substrate confirmed (3 axes); persistence theorem m(n=2)/m(n=1) = 8/9

**Independent findings kept from prior session work:**

**Discipline:**

---

## 10. Status

**CONSOLIDATION-GRADE.** Honestly characterizes the framework's existing
Yukawa exponent principle, its rigor stratification per dynamics layer, and
its current precision status across 12 fermion species. Does not advance
beyond what the framework already has at the structural-form level (3
channels theorem-grade structurally; 9 channels blocked on V_Ram ≅ Cl(6)-Fock).

**The actual master theory's open content reduces to one named structural
question (V_Ram ≅ Cl(6)-Fock → n_free per sector) + one named precision
question (sub-leading Feshbach analog for the +0.13% / sub-leading m_t
residuals).** Both are research-level multi-session work; neither is
bounded by a single probe.

Anyone working forward from this document should:
1. NOT call the existing 3 derived channels "theorem-grade at observational
   precision" — they are theorem-grade at STRUCTURAL FORM, conditional on
   named open upstreams, observationally falsified at PDG precision.
2. Attack either: (a) the V_Ram ≅ Cl(6)-Fock identification (closes the 9
   channels' structural form); or (b) the sub-leading Feshbach analog
   (tightens the existing 3 to observational precision). These are
   independent research programs.
3. Honor the rigor levels per §5 — toggle/graph/compression are theorem-grade
   substrate-side; observer-side A5(b) is axiomatic; the bridge is the open
   piece.

---

## 11. W8 — the exponent-principle-as-master-formula DOES NOT HOLD (post-2026-05-20)

**Probe:** `proofs/foundations/W8_exponent_principle_consistency_2026-05-20.py`.

The user's "highest leverage; continue until decisions/conclusions" prompt
triggered a rigorous consistency test of the stated formula across the 3
derived channels. Result:

| Channel | Stated (n_free, edge_sel, prefactor) | Formula prediction | Actual framework value | Match |
|---|---|---|---|---|
| y_τ | (1, 2, 5/3) | 7.226e-3 | 7.226e-3 | PASS (by construction; formula derived from y_τ) |
| y_ν | (1, 1, 5/3) — docstring identification | 2.17e-2 | **7.44e-1** | **FAIL (+97% off; 3 orders of magnitude)** |
| y_t | (0, 0, **1.0**) | 1.0 | 1.0 | PASS only if prefactor ≠ 5/3 (no chirality factor) |

### 11.1 Why y_ν fails the formula

The framework's actual y_ν derivation (`srs_neutrino_mass_scale.py` PART 3,
the computation that lands in the seesaw producing m_ν3 ≈ 0.050 eV) is:

$$y_\nu = \frac{k-1}{k}\sqrt{\frac{L_{\rm us}}{k}} = \frac{2}{3}\sqrt{\frac{2+\sqrt{3}}{3}} \approx 0.7436$$

This uses **L_us = 2 + √3 (the srs Laplacian spectral radius)**, NOT α₁. The
file's docstring identifies the "edge-local-with-one-less-resolution"
reading as `y_ν = α₁/k ≈ 0.0217` and **explicitly annotates it as "3 orders
of magnitude too small."** Yet commit 66c8836's body cites the docstring
identification (α₁/k) as if it were the framework's actual y_ν. **It isn't.**

No (n_free, edge_sel) integer assignment with prefactor = 5/3 reproduces
y_ν ≈ 0.74 within the (2/3)^N · 3^−M lattice the formula generates. Closest
is (0, 1) at −25%; everything else is far further off. The formula's
generating set is structurally incapable of reaching the spectral L_us form.

### 11.2 Why y_t requires dropping the 5/3 prefactor

y_τ has prefactor = 5/3 (the tan²(arg h) chirality factor at the Bloch
P-point). y_t requires prefactor = 1 — no chirality factor. The stated
formula does not encode prefactor-dependence on n_free / edge_sel, so the
prefactor "turning off" at the gen-3 limit is an **EXTRA assumption beyond
the formula itself.** It is asserted in commit 66c8836's body ("the parity-
odd content... at gen-3 is fully expressed in the rest-energy via the
un-suppressed coupling y_t = 1, not parity-projected into a small
correction") but not derived.

### 11.3 What the framework actually has

Three separately-motivated derivations sharing structural motifs but not
unifying under a single (prefactor, n_free, edge_sel) template:

1. **y_τ** via the 4-factor template `α₁_full × 1/k² × c_H × c_Cl(2)` per
   `theorem_ytau_corollary.md`. Rigorous structural derivation at +0.13%.
2. **y_ν** via spectral seesaw with √(L_us/k) per `srs_neutrino_mass_scale.py`
   PART 3. **Structurally different object** (spectral radius, not NB walker
   amplitude). Lands m_ν3 at +0.87%.
3. **y_t = 1** via "gen-3 limit n_free → 0 + prefactor → 1" assertion in
   commit 66c8836. Asserted, not derived. Lands m_t at +0.82%.

The k-factor structure (1/k, 1/k², 1/k⁰) and the (2/3)-factor appearance
(from NB walker survival in y_τ; from (k-1)/k in y_ν; absent in y_t) ARE
shared motifs across the derivations. But they enter via different
structural objects (NB walker probability vs Laplacian spectral radius vs
gen-3-limit assertion). The "exponent principle" as a unified formula is a
**numerologically suggestive pattern, not a derived master mechanism**.

### 11.4 Implication for the Yukawa master theory program

**The Yukawa master theory as a single template DOES NOT YET EXIST in the
framework.** What exists is:

- y_τ: theorem-grade structural form (4-factor template), +0.13% observationally
- y_ν: derived structural form (spectral seesaw), +0.87% observationally
- y_t = 1: asserted structural form (gen-3 limit), +0.82% observationally

Each is theorem-grade *for its own derivation*, all conditional on identified
open upstreams (A5(b) identification + sub-leading Feshbach for y_τ;
M1.B-conditional + Bloch lift for y_ν seesaw; V_Ram ≅ Cl(6)-Fock for y_t's
"gen-3 limit" assertion).

A genuine master theory — one mechanism that derives all three with shared
inputs — requires one of:

**(α) Find the unified structural object.** A higher-level mechanism that
explains why y_τ's NB walker amplitude, y_ν's Laplacian spectral radius,
and y_t's gen-3 limit are all manifestations of the same underlying
substrate dynamics. Candidate: A2-T waterline retention applied per-sector
with explicit Cl(6)-Fock content + walker-vs-spectral distinguishing
mechanism. Research-level multi-session.

**(β) Derive the 9 remaining channels separately** via per-sector
structural arguments analogous to (1)-(3). Multi-channel research, but
each individually bounded.

**(γ) Honest acknowledgment that the "master theory" is aspirational.**
The framework has 3 derived Yukawa channels via structurally-distinct
mechanisms; presenting these as instances of one formula is overclaim.
This is the disposition this document now adopts.

### 11.5 What §1's "stated formula" should be read as

A **suggestive numerological pattern** that fits y_τ exactly (by
construction, since it was derived from y_τ) and asserts y_t = 1 (via
extra-assumption gen-3-limit prefactor-turnoff). It does NOT fit y_ν as
actually computed by the framework. As a *master formula* it is post-hoc
unification, not derivation.

The reader should not cite the exponent principle as "the framework's
Yukawa master mechanism" without this caveat. The framework's three
derived Yukawa channels are derived by three distinct mechanisms, not one.

### 11.5b Rigor-tier caveat (audit before citing `proofs/masses/*.py` formulas)

Methodological strengthening: the formula y_ν = (k−1)/k · √(L_us/k) used by
`srs_neutrino_mass_scale.py` PART 3 (the "actual framework computation" W8
compared against) is a `proofs/masses/*.py` PART-N commentary form, **not**
a derivation cited from a `docs/theorems/theorem_*.md` or
`predictions/<slug>_derivation.md` with explicit framework-axiom + gate-type
citations. Same caveat applies to the "exponent principle" formula stated
in `srs_tan_beta.py` PART 1: it lives as commentary in a proof script, with
no corresponding theorem doc.

**The framework's load-bearing derivations live in `docs/theorems/theorem_*.md`
+ `predictions/<slug>_derivation.md` with explicit axiom-citation, NOT in
raw `proofs/masses/*.py` scripts.** Before treating any per-channel formula
from a proof script as "the framework's actual mechanism," audit:

1. Does the file cite framework axioms + gate-types explicitly per
   `docs/parameters/parameter_linter.md`?
2. Is there a corresponding `theorem_*.md` or `<slug>_derivation.md`
   carrying the load-bearing derivation? (For y_τ: YES,
   `theorem_ytau_corollary.md`. For y_ν: the load-bearing derivation is
   the m_ν3 seesaw via global spectral gap, NOT the PART-3 form. For the
   "exponent principle": NO corresponding theorem doc exists.)
3. Does the formula's numerical agreement rely on smuggled adoptions or
   fits not framework-derived?

This DOES NOT change the W8 verdict at the structural level. It strengthens
it: only y_τ via `theorem_ytau_corollary.md` is verified rigorous-framework
Yukawa content. The W2 c_H = 1 bijection extension also stands as
rigorous-framework. y_t = 1 (commit 66c8836) and y_ν from `srs_neutrino_mass_scale.py`
PART-3 form should be framework-origin-audited before being cited as the
framework's actual derivations.

### 11.6 Decisions reached by this session

1. **The y_τ derivation stands** at theorem-grade structural form per
   `theorem_ytau_corollary.md` (the 4-factor template is rigorous; +0.13%
   residual is the un-derived sub-leading Feshbach analog).

2. **The y_ν derivation stands** at theorem-grade-conditional via
   spectral seesaw (`srs_neutrino_mass_scale.py` PART 3 + R = 228/7 +
   Bloch lift), conditional on M1.B Galois tower. The "α₁/k delocalized"
   identification in the docstring is a wrong-by-3-orders-of-magnitude
   simplification that should not be cited as the framework's y_ν.

3. **The y_t = 1 derivation is conditional-on-assertion** — the structural
   identification "gen-3 limit → n_free → 0 + prefactor → 1" is two
   separate assertions, neither derived from the framework's existing
   apparatus. Commit 66c8836 itself names V_Ram ≅ Cl(6)-Fock as the open
   piece that would mechanically derive n_free → 0; it does NOT name the
   prefactor → 1 piece, which is a separate assertion.

   **§11.6.3 sharpening (added 2026-05-20 per W25 convention audit,
   an internal working note).**
   The "prefactor → 1" assertion in commit 66c8836 is in SM Peskin
   convention (m = y · v/√2 with y_t = 1 giving m_t = 174.10 GeV). The
   y_τ derivation in `theorem_ytau_corollary.md` §10 uses the framework's
   *operational* convention (m = y · v with y_τ = 7.226e-3 giving m_τ =
   1.779 GeV). These conventions differ by exactly √2 (verified to machine
   precision in `proofs/foundations/W25_convention_audit_2026-05-20.py`).
   In a single consistent convention, the exponent-principle formula's
   prefactor for y_t at gen-3 up-type would need to be 1/√2 (in framework
   convention) or the y_τ prefactor would need to be 5√2/3 (in PT
   convention). Either way, the formula is missing a √2 in one channel
   that is currently absorbed by an unflagged convention switch between
   y_τ and y_t. This is a concrete structural witness for §11.4's claim
   that the exponent principle is "post-hoc unification, not derived
   master mechanism": the √2 is hidden in the convention choice rather
   than derived from substrate. Likely structural source: the SU(2)_L
   doublet ⟨h⁰⟩ = v/√2 identification interacts differently with the
   chirality factor 5/3 (active for y_τ) versus the gen-3-limit prefactor
   (active for y_t). Resolving this would either be part of, or downstream
   of, the V_Ram ≅ Cl(6)-Fock identification (R-14 / Need-D-3).

4. **The Yukawa master theory as a single template DOES NOT EXIST.** The
   framework has 3 derived channels via 3 distinct mechanisms. Presenting
   them as one formula is post-hoc unification.

5. **Forward path**: (β) is the most disciplined next move — derive
   additional individual Yukawa channels (e.g., y_b, y_μ) via per-sector
   structural arguments, rather than seeking the unified template. Each
   per-channel derivation can be rigor-stratified independently. Closing
   them progressively builds confidence that a master mechanism may
   exist; not finding common structural ground confirms (γ) honesty.

6. **The W1.2 + W2 independent findings remain valid** (scale convention
   load-bearing; c_H bijection extends to quark vertices) — these are
   per-channel structural extensions that hold regardless of whether the
   master theory exists.

7. **The user-asked "highest leverage" question is answered**: the
   highest-leverage move that can be completed in a session is W8 itself
   (this finding) — it prevents the framework from chasing a non-existent
   unified template and redirects attention to per-channel derivations
   or to the deeper substrate-side structural objects (NB walker amplitude
   vs spectral radius vs Cl(6)-Fock limit) that the 3 channels actually
   use.

---

## 12. Status (final)

**CONSOLIDATION-GRADE WITH MAJOR HONEST RETRACTION.** The original §1–§10
framing presented the exponent principle as the framework's actual Yukawa
master mechanism. §11 retracts this: the unified formula is post-hoc, not
derived; the framework has 3 separate derivations via distinct structural
mechanisms. The doc is preserved for provenance and stratified-rigor
characterization, but the headline claim ("exponent principle is the master
template") is withdrawn.

Working tree only (not committed). No ledger row moves. No DAG changes.
No `predictions/*.py` touched. The probe `W8_exponent_principle_consistency_2026-05-20.py`
remains as the falsification record.

The decision the session reached: **the Yukawa master theory in the sense
of "one formula for all channels" does not yet exist; the path forward is
per-channel derivation OR substrate-side unification of NB-walker /
spectral / Cl(6)-Fock-limit mechanisms, both of which are multi-session
research.** No further architecture-building until one of these paths
produces a new derived channel.

---

## 13. W9 — sharpening: the "3 separately motivated derivations" framing was an overcount

**Date: 2026-05-20 (same session, after user "do α unless β assists α, in
which case β first").** User-directed audit (a β-flavored step that assists
α by clarifying scope) revealed §11's "3 separately motivated derivations"
framing was itself an overcount. The framework's actual content reduces
to **ONE rigorous derivation + ONE named adoption**.

### 13.1 The audit finding

`predictions/m_nu3_derivation.md` Step 3 (2026-05-18 chain-audit honest
re-grading) explicitly states:

> "This step previously read 'm_D = v at leading order' — that is the
> adoption **y_ν = 1**, dressed as a derivation via the hand-wave 'the
> bilinear δ⁴ already captures the field content'. It is NOT derived.
> It is the **same undischarged up-sector Yukawa natural-scale anchor**
> that the master dark-correction doc calls **'the single hard residue'**
> and that forced the **m_top retraction (Row P38)**."

Cross-confirmed by the dark-correction master doc itself
(`docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md`
line 402): *"the up-sector y_t natural-scale anchor remains the single
hard residue."*

So:
- **y_t = 1** (asserted at gen-3 limit; commit 66c8836's "exponent
  principle" reframe) IS the framework's named single hard residue.
- **y_ν = 1** (used in the framework's load-bearing m_ν3 derivation,
  Step 3) IS the **same** adoption.
- Both are **one** undischarged adoption, not two separate channels.

The `(k-1)/k · √(L_us/k)` form in `srs_neutrino_mass_scale.py` PART 3
that W8 compared the exponent-principle formula against is NOT the
framework's load-bearing y_ν derivation — it's a PART-N commentary form
in a proof script with no corresponding `theorem_*.md` or
`<slug>_derivation.md` (per §11.5b rigor-tier criteria). The framework's
actual m_ν3 derivation uses y_ν = 1 (adopted), giving m_ν3 = 50.57 meV
at +0.87%.

### 13.2 Sharpened verdict — refines §11

**§11 said:** "the framework has 3 separately-motivated derivations
(y_τ via 4-factor; y_ν via spectral seesaw; y_t via gen-3 limit)
using structurally distinct objects; the exponent principle as a
unified formula is post-hoc."

**§13 sharpens:** the framework actually has **ONE rigorous derivation**
(y_τ via `theorem_ytau_corollary.md`) + **ONE named adoption**
(y_t = y_ν = 1 at gen-3-limit / Dirac-top-natural-scale-anchor =
"the single hard residue" per master dark-correction doc). The "two
parallel derivations of y_ν and y_t" framing collapses to one
undischarged adoption that the framework's own self-audit names.

The exponent principle's "gen-3 limit → n_free → 0 → y = 1" structural
story (commit 66c8836) is a **re-rationalization** of this same
adoption — it doesn't derive y_t = 1; it tells a story about WHY one
might expect y = 1 at the gen-3 limit. The story rests on the open
V_Ram ≅ Cl(6)-Fock identification (= Need-D-3 / R-14 in the residue
register).

### 13.3 Path (α) reduces to Need-D-3 / R-14

The substrate-side unification of "the three Yukawa derivations"
is not actually three problems — it's one: **derive the single hard
residue (y_t = y_ν = 1 at gen-3-limit / natural-scale anchor) from
substrate dynamics.** The "unification" is identifying that y_t and
y_ν are the same adoption.

Per R-14 register, this is what 9 prior attacks have failed at:
- R1 C₃ isotypic Yukawa (Λ¹ ≅ Λ² Hodge identical at k\*=3)
- Type 6c (3k\*-2)/k\* candidate (3 structural obstacles)
- V_{−1}-T_{B-L} symmetry-breaking (gives δ_CP not Yukawa hierarchy)
- Σ(h) charge-weighted lift (no per-sector signature)
- Bloch P-vs-N path-b (no new observables at N)
- Route 4 SU(2)_L pseudoreal (H, H̃ same SU(2)_L rep)
- Need-D-3 path-β preflight (5 operator-algebra structures in M_3(ℂ),
  all fail — need framework extension beyond M ⋊_α Z_3)
- sector_hamming_weight_yukawa (18 g_n forms, none match all 4 sectors)
- Plus the current session's W6 (state-counting) — retracted
- Plus the W1 + W1-up Koide-shape extensions — deleted as misframed

The framework's named open paths after these:
- **NA-4 Path B** — Wolfram-style multiway DAG carrying non-associative
  composition. Multi-sprint substrate redesign. Bounded prefix (NA-2')
  is done.
- **Non-linear coupling beyond M ⋊_α Z_3** — would require new framework
  axioms.

### 13.4 (α) verdict for this session

**Path (α) is structurally identical to the framework's existing
Need-D-3 / R-14 program.** I have no novel structural angle to add to
the 9-attack scoreboard in one session. The honest disposition:

1. The Yukawa master theory's open content = **the single hard residue =
   the up-sector y_t natural-scale anchor adoption** = Need-D-3 / R-14
   register's named gate.
2. The framework's existing 9 attacks ruled out exhaust the obvious
   structural approaches within the current substrate apparatus
   (operator-algebra within M ⋊_α Z_3, charge-weighted Yukawa forms,
   isotypic decompositions, screw-Wigner-D templates).
3. New attacks require either (a) NA-4 Path B substrate redesign
   (multi-sprint research; the bounded prefix NA-2' is already done),
   or (b) a genuinely novel structural ground not surfaced in the
   9 prior attacks or in this session.
4. **No novel angle is surfaced here.** The contribution of W9 is the
   *clarification* that the Yukawa master theory's open content = the
   single hard residue = ONE adoption, not three separate problems —
   refining §11's "3 separately motivated" framing to "1 derivation +
   1 named adoption."

### 13.5 What this means for the program

The Yukawa master theory is closed at:
- **y_τ + (y_μ, y_e via Koide-shape ratios)** rigorously
- **R_ν = 228/7 splitting ratio** rigorously
- **(ε²_up − 2)/(ε²_down − 2) = 14/5 ratio** rigorously (Row P37)
- The **structural Higgs edge factor c_H = 1** for all charged-fermion
  vertices rigorously (W2)

The Yukawa master theory is open at:
- **The single hard residue y_t = y_ν = 1 adoption** — this carries
  the up-sector natural-scale anchor and the neutrino seesaw scale.
  Closing this would close Row P38 (m_top) + the absolute m_ν3 scale
  + provide the n_free derivation for the exponent principle to become
  a genuine derived formula.
- **The 9 remaining individual Yukawa channels** (y_b, y_s, y_d, y_c,
  y_u, y_μ-direct, y_e-direct, y_ν1, y_ν2) — these inherit the same
  V_Ram ≅ Cl(6)-Fock / single-hard-residue blocker for absolute scale,
  modulo Koide-shape ratios within sectors.

The forward path is the framework's existing Need-D-3 / R-14 attack
program. Specifically, NA-4 Path B substrate-redesign is the named
multi-sprint research direction. Closing it = closing the single hard
residue = closing the Yukawa master theory.

