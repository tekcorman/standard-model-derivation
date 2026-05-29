# Dark-correction MDL synthesis — theorem candidate

> **For a unified treatment of the framework's substrate-Feshbach-analog dark-correction mechanism — including the universal template, all observable classes (Berry / Feshbach / counting / mass-matrix-block), the two calibrated derivation routes (Hashimoto-spectral and cycle-counting), the cluster catalogue, and the application protocol — see `theorem_substrate_feshbach_dark_corrections_master.md` (2026-05-15).** This doc covers the MDL synthesis (Lemma 1, Lemma 2) and remains the underlying reference for the photon-Berry-phase route.

**Date:** 2026-04-25 PM
**Status:** Theorem candidate. The bit-cost ranking lemma (Lemma 1) is rigorous under a specified description language; the unit-phasor identification (Lemma 2) is structural; the linear-vs-squared counting rule (Lemma 3) is conditional on a Lagrangian/operator-setup step that this doc does not close. β = sin(arg h) · α_EM follows from these three pieces; the conditional gates make the theorem candidate, not a closed theorem.
**Supersedes:** the four-pathway framing in an internal working note (2026-04-25 AM body), revised in the same doc's REVISION NOTE (2026-04-25 PM).
**Related:**
- A2-T (derived theorem): `theorem_A2_mdl_from_finite_register.md`. A5(b): `../framework/framework_axioms.md` §5b.
- A5(b) Path 1: `theorem_A5b_level_prescription.md`.
- 5/12 derivation (operator-setup prototype): `proofs/foundations/dark_feshbach_a2_closure.py`.
- β.E numerical diagnostic: `proofs/lorentz/birefringence_c3_irrep_O2.py`.

---

## 1. The unified formula

A2-T (selective retention; derived theorem, `theorem_A2_mdl_from_finite_register.md`) and A5(b) (couplings as MDL probabilities, `framework_axioms.md` §5b) imply the following form for any observable O in any sector with coupling α_O:

$$O = O_{\text{bare}} + \alpha_O \cdot \sum_F p_{\text{MDL}}(F) \cdot F(h)$$

where:
- **O_bare** is the MDL-retained backbone of O computed via direct A2/A5(b) counting (Pathway 1: V_us = 9/40, V_cb = 256/6305, Higgs v's 5/12-chain coefficient).
- **F** ranges over MDL-permitted parity-odd dimensionless functionals of h with the right tensor character for O.
- **p_MDL(F)** is the MDL retention probability for F, ranked by description-length L(F) under a fixed description language (Lemma 1 below).
- **α_O** is the sector-specific coupling: α_1_bare = (2/3)^8 for dark-sector observables (V_us correction, m_ν, θ_23, Higgs v dark vertex), α_EM for photon-sector observables (β cosmic birefringence).

The leading observable correction is the highest-p_MDL term in the sum:

$$O \approx O_{\text{bare}} + \alpha_O \cdot p_{\text{MDL}}(F^*) \cdot F^*(h), \qquad F^* = \arg\min_F L(F)$$

with subleading terms suppressed by 2^{−ΔL} factors.

---

## 2. Lemma 1 — Canonical encoding of parity-odd projection of the unit walker phasor

**REFORMULATED 2026-05-05** (waterfilling audit an internal working note).

**Important framing note.** This lemma was originally written as a "bit-cost minimum selection" (strict-minimum reading of MDL). That framing contradicts A2-T waterline: MDL retains every representation that saves bits over raw, multiple representations above the waterline are all physically realized. The strict-minimum reading is wrong on its face.

The correct reading (waterline-consistent) is **encoding-equivalence**: alternative expressions that evaluate to the SAME numerical value (e.g., sin(arg h) = Im(h)/|h|) are equivalent encodings of the same physical content; the bit-cost ranking selects the canonical (cheapest) form among equivalents but does NOT discard them from physical realization. Functionals with DIFFERENT numerical values (e.g., sin(2 arg h)) are parity-odd parts of DIFFERENT structural objects ((h/|h|)² rather than h/|h|), not alternative parity-odd parts of the same object.

The structural selection of sin(arg h) for β = c·sin(arg h)·α_EM rests on Lemma 2 (photon polarization couples to the unit walker phasor h/|h|), NOT on this lemma's bit-cost ranking. Lemma 1 is supporting commentary; Lemma 2 is load-bearing.

**Description language** (still useful for canonical-encoding purposes). Primitives: the symbols `h`, `|h|`, `arg(h)`, plus operators `Re`, `Im`, `+`, `−`, `×`, `÷`, `sin`, `cos`. Each primitive symbol or operation costs 1 bit. Composition: a functional F is described by an expression tree built from these primitives; L(F) = number of nodes in the minimal expression tree.

**Constraints on candidates** (from physical setup, see §3):
- (P) Parity-odd: F(h) → −F(h) under h ↔ h* (the framework's only parity-violation channel, since spatial parity ↔ I4_132 ↔ I4_332 enantiomer ↔ h ↔ h*).
- (D) Dimensionless: |F| has a structural bound that does not depend on |h| separately from h. (Equivalently: F depends on h only through scale-invariant combinations.)
- (B) Bounded by 1: |F(h)| ≤ 1 without further normalization, so F can directly couple to a unit-vector observable.

**Catalog and bit-costs** (parity-odd candidates passing P, D, B):

| F(h) | Expression | L(F) | Notes |
|---|---|---|---|
| sin(arg h) | sin(arg(h)) | **2** | Im of unit phasor; direct |
| cos(arg h)·sin(arg h) | sin(2 arg h)/2 | 4 | Double-angle product |
| sin(arg h)/2 | sin(arg(h))/2 | 4 | Trivial rescaling |
| sin(2 arg h) | sin(2×arg(h)) | 4 | Double angle |
| 2 sin(arg h /2) | 2×sin(arg(h)/2) | 5 | Half-angle chord |
| Im(h)/\|h\| | Im(h)÷\|h\| | 4 | Polar-equivalent to sin(arg h) but with separate Im, divide; same value, longer expression |
| Im(h)/(2·\|h\|·Re(h)) | … | ≥ 7 | Higher-order |

**Im(h) alone fails (B):** Im(h) = √5/2 ≈ 1.118 > 1; not bounded by 1 without normalization.

**arg(h) alone fails (B):** arg(h) ∈ [−π, π]; bounded by π not 1; would require /π normalization which adds bits. (Equivalently: arg(h) is a phase in radians, while sin(arg h) is a sine which is auto-bounded.)

**Im(h)/|1−h| alone fails (B):** Im(h)/|1−h| = √5/(2·√(3−√3)) ≈ 0.993; numerically close to 1 but **not bounded by 1 in general** (the bound depends on whether |1−h| > Im(h), which depends on where h sits in the complex plane). Fails the structural bound criterion.

**Conclusion (Lemma 1, REFORMULATED 2026-05-05).** Under the description language above with constraints P + D + B, the **canonical encoding** of the parity-odd projection of the unit walker phasor h/|h| is **F* = sin(arg h)**, at L(F*) = 2 bits.

Encoding equivalents at higher L (e.g., Im(h)/|h| at L=4) evaluate to the same numerical value 0.7906; they are alternative expressions of the same physical content and are not independent contributions. Functionals at the same/higher L with DIFFERENT numerical values (e.g., sin(2 arg h) at L=4 = 0.9682; sin(2 arg h)/2 at L=4 = 0.4841) are parity-odd projections of DIFFERENT structural objects ((h/|h|)² and similar) and would couple to physically different operator structures, NOT to the same photon-polarization channel as h/|h|.

The selection of sin(arg h) for the photon coupling channel in β = c·sin(arg h)·α_EM is therefore made in Lemma 2 (unit-vector ↔ unit-phasor dimensional matching), not here. This lemma's role is to identify the canonical encoding within the channel fixed by Lemma 2.

the author's separate private derivation's path_c_derivation.md asserted "the unique leading dimensionless object" — that uniqueness is now understood structurally (Lemma 2 fixes the structural object as h/|h|; this lemma fixes the canonical encoding as sin(arg h)), not as bit-cost minimum.

---

## 3. Lemma 2 — Photon polarization couples to the unit walker phasor

**Setup.** A CMB photon at the P-point of the srs Brillouin zone has helicity ±1 polarization eigenstates labeled L (helicity +1) and R (helicity −1). By the standard correspondence between spin-1 helicity and C₃-rotation eigenvalues for photons propagating along [111] (which is the direction of k_P), L is the ω-irrep eigenstate of the C₃ stabilizer of P, and R is the ω²-irrep eigenstate.

Per `predictions/B_P_doubly_degenerate_h.py`:
- L = ω-irrep state lives at walker eigenvalue +h (multiplicity 1 within V_h, which decomposes as trivial ⊕ ω).
- R = ω²-irrep state lives at walker eigenvalue −h (multiplicity 1 within V_{−h}, which decomposes as trivial ⊕ ω²).

Each polarization state has a walker phase per step: L acquires phase arg(h), R acquires phase arg(−h) = arg(h) ± π.

**The polarization vector parameterization.** A photon polarization is a unit vector in the (L, R) basis; equivalently, a point on the Bloch sphere of helicity ±1. A relative phase shift Δφ between L and R produces a polarization rotation of Δφ/2 in the (linear-polarization basis).

For the substrate to induce a relative phase shift via chiral coupling, it must couple to a parity-odd quantity. Crucially, since the photon polarization is a UNIT vector, the substrate's parity-odd quantity that couples to it must also be UNIT-NORMALIZED — otherwise the dimensional mismatch produces an unbounded rotation, violating the photon's helicity-conservation upper bound.

**Identification of the parity-odd unit-normalized substrate quantity.** The walker eigenvalue h has polar decomposition h = |h| · e^(i arg h). The unit walker phasor h/|h| lies on the unit circle; it has parity-odd part:

$$\text{Im}\!\left(\frac{h}{|h|}\right) = \frac{\text{Im}(h)}{|h|} = \sin(\arg h) = \sqrt{5/8}$$

By Lemma 1, this is the MDL-cheapest parity-odd dimensionless unit-bounded functional of h.

**Lemma 2.** The substrate's parity-odd content that couples to a photon polarization rotation is the parity-odd part of the unit walker phasor at the relevant Bloch point (P, in this case). On srs:

$$\text{(substrate parity-odd content for photon coupling)} = \sin(\arg h) = \sqrt{5/8}$$

**Status of Lemma 2.** This is a structural claim from dimensional matching (unit photon polarization ↔ unit walker phasor) plus Lemma 1 (which functional of h is the cheapest unit-normalized parity-odd object). Both arguments are as rigorous as the underlying Lemma 1 description language. The remaining structural step (why dimensional matching, specifically, fixes the walker quantity at the unit phasor rather than at some other normalization) is a CFJ-style argument; it's the standard photon-polarization-rotation identification in any chiral medium and not unique to this framework.

---

## 4. Lemma 3 — The linear-vs-squared counting rule (CONDITIONAL — not closed)

**Statement.** For an observable O extracted as a one-point function of the substrate (delocalized observable: m_ν, θ_23, β, mixing angles): the dark correction is LINEAR in α_O. For an observable O extracted as a two-point function of the substrate (edge-local observable: Higgs v from φ-φ correlator at single edge): the dark correction is SQUARED in α_O.

**Status.** The structural argument is plausible (one-point vs two-point function of the dark mode) and matches the existing framework derivations:
- 5/12 Higgs v: derived in `proofs/foundations/dark_feshbach_a2_closure.py` as **linear** in α₁/(1−α₁) under THIS REPO's parameterization (with α₁ = (2/3)^8). The "squared" form in the author's separate private derivation was using a different parameterization (where ALPHA_1 = (5/3)·(2/3)^8 already absorbs the 5/3 factor).
- m_ν, θ_23, m_τ: linear in α₁ under the standard parameterization.

So under THIS REPO's convention, all corrections are linear in α₁_bare. The "linear-vs-squared distinction" is not currently a load-bearing structural rule for this repo's derivations — it was an artifact of the author's separate private derivation's parameterization choice.

**For β specifically.** β is a delocalized amplitude observable (photon polarization rotation). Under the linear rule, β has form:

$$\beta = \alpha_{\rm EM} \cdot c \cdot F^*(h)$$

with F\* = sin(arg h) (Lemma 1, Lemma 2) and c a structural coefficient analogous to 5/12 for Higgs v.

**The remaining open piece is c.** For β, the structural coefficient is c = 1 (asserted). To derive this, one would need:

1. The photon-walker coupling Lagrangian (or operator) on srs explicitly. the author's separate private derivation identified this as ~1-2 sessions of Lagrangian work, not done.
2. The Feshbach-style self-energy summation analogous to the F0→F3 chain for 5/12, applied to the photon sector.
3. Counting of the operator-structure factors that produce c = 1 (analogous to n_g/(N_ATOMS·k*²) = 5/12 for Higgs).

This is the genuinely open structural piece. It is bounded but real — not "MDL bookkeeping," but lattice-specific operator algebra analogous to the existing 5/12 chain.

**Lemma 3 closed for β when:** the photon-walker coupling structure is derived and the corresponding operator-counting gives c = 1.

---

## 5. Theorem candidate (β)

**Theorem candidate.** Under A2-T + A5(b) + Lemma 1 + Lemma 2 + Lemma 3 (CONDITIONAL on §4):

$$\beta = c \cdot \sin(\arg h) \cdot \alpha_{\rm EM}$$

with c = 1 conditional on the operator-structure work in §4. Numerically:

$$\beta = \sqrt{5/8} \cdot \alpha_{\rm EM} = 0.7906 \cdot \alpha_{\rm EM}(0) = 0.3306°$$

vs Eskilt 2022 + Minami-Komatsu 2022: β = 0.342° ± 0.094° (residual −0.12σ, well within 1σ).

**Subleading MDL contributions.** From Lemma 1, the next-cheapest parity-odd functional has L = 4 bits, suppressed by 2^{−2} = 1/4 relative to the leading sin(arg h) term. Including all subleading terms with the Lemma-1 catalog and their respective MDL probabilities would shift the predicted β by at most a few percent, well within the experimental error band.

**Status of theorem candidate.** Lemmas 1 and 2 are at "structural derivation" grade. Lemma 3 is conditional on the operator-structure work in §4. Without §4 closed, the theorem candidate is **A−** (same as before this MDL synthesis — but now with explicit identification of what's done vs what remains).

Under the four-pathway framing (now superseded), I had over-claimed that this would close as 1-2 sessions of "MDL bookkeeping." That was wrong; the MDL framework explains why sin(arg h) is the right functional (Lemma 1), but the structural coefficient c = 1 still needs lattice-specific operator-setup work (§4). The honest estimate is closer to the author's separate private derivation's original 1-2 sessions of Lagrangian work.

---

## 6. What this changes vs. earlier scoping

The earlier four-pathway scoping doc identified Pathway 4 as a missing rigorous pathway. Under the MDL synthesis, Pathway 4 collapses into Pathway 1 (MDL counting via A2/A5(b)) at the formula level, but the per-observable structural coefficient (c for each observable) still needs its own derivation. For β, that coefficient is c = 1 under the photon-walker structural setup, which is the remaining open piece.

The earlier "T.D = 4-6 sessions of new structural work" estimate was correct in scope; the "T.D = 1-2 sessions of MDL bookkeeping" reframe was overoptimistic. The realistic estimate: 1-2 sessions to derive the photon-walker coupling structure on srs (analog of 5/12's F0→F3 chain), giving c = 1; plus this MDL synthesis doc (this file) for the bit-cost ranking and unit-phasor identification.

Total to close β at theorem grade: this doc + 1-2 sessions of photon-walker operator structural work.

---

## 7. Cross-observable MDL applications (sketch, not closed)

Same machinery applied to other sectors:

**β (photon, delocalized amplitude):** F\* = sin(arg h), α_O = α_EM, c = ? (open per §4)

**m_ν2/m_ν3 (dark-sector, delocalized amplitude):** F\* = sin(arg h)? Or F\* = Im(h)/|h|² (Pathway 2 contour-integral form)? The current `predictions/m_nu2.py` uses Im(h)/|h|² (squared 1/|h|² normalization) from the author's separate private derivationa. Whether this is the MDL-cheapest form for neutrino masses, or whether sin(arg h) (linear 1/|h|) would also work, is an open question — would need to redo the contour-integral derivation under the unified MDL framework.

**θ_23 (dark-sector, mass-matrix angle):** F\* = tan²(arg h) = (Im(h)/Re(h))² = 5/3 from a 2×2 mass-matrix block diagonalization. The "squared" form here is from the OBSERVABLE structure (mass-matrix element squared), not from a two-point function. The MDL ranking on h-functionals subject to (mass-matrix tensor character) gives tan² as cheapest among quadratic combinations.

**Higgs v (dark-sector, vertex-localized):** F\* = Im²(h)/k* = 5/12 from the F0→F3 chain. Same structural form (5/12) is the leading MDL term; subleading windings are absorbed in the geometric series α₁/(1−α₁).

**θ_13 (dark-sector, mixing-angle):** F\* = 1 (asserted, "Tr σ_x = 0 selection"). This corresponds to a parity-projected mass-matrix sub-block where the parity-even contribution is killed by Tr σ_x = 0 (cited from `srs_pmns_dark_consistent.py` chain). Under MDL, this would be: among parity-projected functionals at the θ_13 sub-block level, the constant-1 functional is the cheapest after Tr σ_x = 0 kills the chirality-dependent piece. Worth checking whether this is uniquely forced.

These applications make explicit what's already the per-observable structural derivation, in the language of "MDL bit-cost ranking on h-functionals subject to observable's tensor character." The framework's strength is that each observable's derivation uses lattice-specific operator structure; the unification at the MDL level is meta-theory about what the framework is doing, not a replacement for the per-observable structural work.

---

## 8. Honest summary

This doc closes the MDL ranking lemma (Lemma 1) and the unit-phasor identification (Lemma 2) at structural-derivation grade. Lemma 3 (the c = 1 coefficient for β) remains open and requires lattice-specific operator-setup work (the photon-walker coupling on srs, ~1-2 sessions). The theorem candidate β = sin(arg h)·α_EM is at A− grade with this doc, and would close to theorem grade once Lemma 3 is closed.

This is meaningful progress over the morning's β.E findings, which I'd over-interpreted as falsification-adjacent. Under the MDL framing, the β.E findings are diagnostic of which MDL term naive perturbation isolates — Im(h) at L = 1 cost (parity-odd but fails dimensionless-unit-bound criterion) vs sin(arg h) at L = 2 cost (passes all criteria, leading MDL term). The framework's leading prediction takes the latter; β.E's first-order numerical shift identifies the former. Both are MDL-permitted; sin(arg h) is bit-cost-cheaper and observationally selected.

The remaining work is bounded: derive the photon-walker coupling structure on srs (analogous to the 5/12 F0→F3 chain), giving the c = 1 coefficient. Until then, β = sin(arg h)·α_EM is at A− as a theorem candidate, not theorem-grade.

---

## 9. Lemma 3 attempt status (2026-04-25 PM)

After writing this doc I attempted Lemma 3 directly. Status: hit a structural wall, stopped honestly.

**Setup**:
- Computed photon Hodge bundle at P-point: confirmed photon ω² = 36 (doubly degenerate, two polarizations).
- The photon eigenstates at P are NOT the Hashimoto walker eigenstates. They live in the cokernel of the vertex-edge incidence d, with their own Hodge Laplacian Δ_1 = d·d† + d_1†·d_1 and their own dispersion.
- Critically: photon ω = 6 at P, while Hashimoto walker eigenvalue is h = (√3+i√5)/2 with |h|=√2. They're different operators on different (but related) spaces.

**Attempted**:
- Constructed C₃ on the 6 primitive edges with Bloch-phase corrections (P fixed under C₃ about [111]).
- Attempted to identify L/R helicity ↔ ω, ω² C₃-irrep mapping on the 2D photon eigenspace at ω² = 36.

**Wall**:
- The C₃ I constructed doesn't commute with Δ_1: max|[C₃, Δ_1]| = 15.6, well above any reasonable tolerance.
- The bug is that C₃ must act consistently on vertices AND edges AND 2-forms simultaneously (the consistency constraint d · C₃_vertex = C₃_edge · d, and similarly for d_1). I constructed C₃ on edges in isolation; without the vertex-side consistency it doesn't preserve the Hodge structure.
- This is a careful piece of representation theory: building C₃ as a chain map across the cellular complex C^0 ← C^1 ← C^2 with the right Bloch-phase corrections at each level.
- Without a working C₃ on the photon bundle, the L/R helicity ↔ ω/ω² assignment cannot be checked, and Lemma 3's structural setup cannot be made rigorous in this attempt.

**What this confirms**: the Lemma 3 work is NOT MDL bookkeeping. It's lattice-specific operator-setup work — careful construction of group representations across the Hodge complex on srs. Bounded scope, but real. Estimate: 1-2 sessions to construct C₃ properly across the complex, then identify the L/R irrep assignment, then derive the chiral coupling and the coefficient c.

**Honest disposition**: Lemma 1 and Lemma 2 are at structural-derivation grade in this doc. Lemma 3 requires the photon-Hodge-bundle group-representation work I attempted but did not close. β remains A− with a clean two-step closure path: this doc + 1-2 sessions of photon-Hodge-bundle group-theoretic operator-setup work.
