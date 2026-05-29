# Theorem — Color triplet → Γ concentration with γ_7 IB-root split (Yukawa master §4(C))

**Date:** 2026-05-21
**Status:** THEOREM-GRADE-CONDITIONAL on §4(D)'s walker-length L derivation. Algebraic argument + computational verification (W39 probe, 7/7 gates PASS). The conditional is exactly the same gate as the framework's existing y_t = 1 (commit 66c8836) and y_b ≈ Q^g (master synthesis §3) derivations.

**UPDATE 2026-05-21 — the §4(D) conditional is discharged.** The γ₇ IB-root split of §6 below (n=2 → h=1 saturation; n=1 → h=2 Perron) is now derived: `theorem_updown_split_conjugate_higgs_2026-05-21.md` proves the up-type (n=2) couples to the conjugate Higgs `H̃ = iσ₂H*`, which is even-grade and cannot flip handedness ⇒ `L=0` ⇒ the saturation root; the down-type (n=1) couples to the odd-grade Higgs `H` ⇒ flips handedness ⇒ `L=g` ⇒ the Perron root. §6's "W38 finding (probe-grade)" input is superseded by that derivation. This theorem is now THEOREM-GRADE-STRUCTURAL.

**Purpose.** Third of the four structural sub-theorems lifting §4 of `theorem_yukawa_master_theory_synthesis_2026-05-20.md` from sketch to theorem-grade. Closes the color-TRIPLET half of the master Yukawa synthesis (the color-SINGLET half is §4(B) + §4(B')), using the W38 γ_7 structural finding to fix the Ihara-Bass root choice within Γ trivial.

---

## 1. Statement

**Theorem (§4(C) — Color triplet → Γ concentration + γ_7 IB-root split).** Let X be a color-triplet fermion species — n ∈ {1, 2} in the Cl(6) Fock decomposition per `theorem_charge_before_color.md` §9, transforming as SU(3) fundamental 3 (n=1, d-quark family) or anti-fundamental 3̄ (n=2, ū_R / u-quark family). Then:

(a) The species' SU(3)-invariant projection has vertex-space content in V_triv at the **symmetric cycled-vertex axis** (e_1 + e_2 + e_3)/√3, orthogonal to the color-singlet's natural e_0 axis. The two axes span V_triv (2-dim).

(b) The species' Yukawa-vertex walker per-step amplitude h must be **real and positive** (real because the framework's y_t / y_b derivations use real h per `theorem_yukawa_exponent_principle_master.md` §3.3 + master synthesis §3; positive because the walker amplitude is identified with MDL probability via A5(b), which is non-negative).

(c) Among the C_3-stable Bloch points {Γ, H, P}, the unique site whose V_triv yields a real positive Ihara-Bass root is **Γ trivial λ_A = +3**. The two IB roots are h ∈ {1, 2}.

(d) The choice between h = 1 (saturation) and h = 2 (Perron walker) is fixed by **γ_7 = (−1)^n** per the W38 structural finding:
- n = 2 (ū_R, γ_7 = +1): h = 1, the saturation root (|h| = 1, walker amplitude unchanged per step).
- n = 1 (d_L, γ_7 = −1): h = 2, the Perron root (|h| = k* − 1, walker amplitude (h/k*)^L per L steps).

**Corollary 1 (y_t).** With h = 1 and walker length L = 0 (the gen-3-limit exponent-principle assertion per `theorem_yukawa_exponent_principle_master.md` §3.3), y_t_PT = h^0 = **1** in Peskin convention, giving m_t(tree) = v/√2 = 174.10 GeV (+0.82% vs PDG 172.69 GeV).

**Corollary 2 (y_b).** With h = 2 and walker length L = g = 10 (the Perron walker traversing the full girth cycle), y_b = (h/k*)^g = (2/3)^10 = **Q^g ≈ 0.01734** (+2.06% vs m_b/v ≈ 0.01698 at the m_b MS-bar scale, within the Family-D scale of the master synthesis §5).

**Corollary 3 (4-cell factorization of §3 selection table).** Combined with §4(B) (singlet w/ chir-5/3 → P) and §4(B') (singlet w/ chir-7 → Γ/H), the master synthesis §3 selection table factorizes cleanly across (color, γ_7):

| (color, γ_7) | Bloch site | h-class | Yukawa channel |
|---|---|---|---|
| (singlet, +1)  | Γ/H trivial λ=∓1 | chir 7        | ν (R_ν splitting + ν_amp)   §4(B') |
| (singlet, −1)  | P trivial         | chir 5/3      | y_τ                          §4(B)  |
| (triplet, +1)  | Γ trivial λ=+3    | h = 1 sat     | y_t                          §4(C)  |
| (triplet, −1)  | Γ trivial λ=+3    | h = 2 Perron  | y_b                          §4(C)  |

---

## 2. Setup and inputs

**Inherits from §4(A)** (`theorem_C3_block_decomposition_2026-05-21.md`):
- V_triv at Γ has A(Γ) eigenvalues {+3, −1}, with the eigenvectors (e_0 + e_1 + e_2 + e_3)/2 (fully symmetric, λ=+3) and (3e_0 − e_1 − e_2 − e_3)/(2√3) (orthogonal trivial, λ=−1).
- V_triv at H: {−3, +1}. V_triv at P: {+√3, −√3}.
- C_3-stable Bloch points: {Γ, H, P}.

**Inherits from §4(B)** (`theorem_color_singlet_P_concentration_2026-05-21.md`):
- Cl(6) color C_3 (cycling 3 edge modes a_i at v_0) ≡ §4(A) body-diagonal C_3 (cycling v_1 → v_3 → v_2) under the bijection edge_i ↔ v_i.
- Lifts to Fock space via U_σ on Cl(6); the n ∈ {1, 2} blocks transform as trivial + ω + ω².

**Inherits from W38 finding**:
- γ_7 = (−1)^F on Cl(6) Fock factorizes Bloch-chirality-class across (color, γ_7), 4/4 empirical correlation.
- For color triplet: γ_7 = +1 (n=2, ū_R) → "saturated" Ihara-Bass root; γ_7 = −1 (n=1, d_L) → "walker/Perron" root.

**Pre-existing framework derivations**:
- `srs_tan_beta.py` PART 1 + `theorem_yukawa_exponent_principle_master.md` §3.3: y_t = 1 at gen-3 limit; m_t = v/√2 = 174.10 GeV (+0.82%).
- `theorem_yukawa_master_theory_synthesis_2026-05-20.md` §3: y_b = Q^g = (2/3)^10 ≈ 0.01734 (+2.06%) via Γ Perron walker.

---

## 3. Proof of (a) — color triplet projects to the cycled-symmetric V_triv axis

A color triplet at v_0 is an n=1 or n=2 Fock state, transforming as SU(3) 3 (n=1) or 3̄ (n=2). The 3 basis states for n=1 are |100⟩, |010⟩, |001⟩, corresponding to occupation of edge 1, 2, 3 at v_0. Similarly the 3 basis states for n=2 are |110⟩, |101⟩, |011⟩ — occupation of all but one edge.

Under the bijection edge_i ↔ v_i (proved in §4(B) §3): the Fock state |100⟩ ↔ "occupied at the edge to v_1" ↔ vertex amplitude on e_1. Likewise for n=2: |011⟩ has zero amplitude on edge_1 ↔ amplitude concentrated at v_1 (vacated edge means the fermion is at v_0 with the other 2 edges occupied; the "color label" identifies which edge is vacated, hence which cycled vertex is the "color" of the state).

The SU(3) singlet (trivial) component of the SU(3) 3 representation is the symmetric combination

  |1_singlet⟩ = (|100⟩ + |010⟩ + |001⟩) / √3,

which under the bijection has vertex-space content (e_1 + e_2 + e_3)/√3. This is exactly the second basis vector of V_triv per §4(A) §4.

Verification (W39 Step C X2):
- P_triv · (e_1 + e_2 + e_3)/√3 = (e_1 + e_2 + e_3)/√3 (in V_triv).
- P_ω · (e_1 + e_2 + e_3)/√3 = 0.
- ⟨e_0, (e_1 + e_2 + e_3)/√3⟩ = 0 (orthogonal to color-singlet axis).  ∎

**Reading.** V_triv is 2-dimensional with two natural basis vectors:
- e_0: the C_3-fixed vertex. Color-singlet wavefunction's natural concentration site.
- (e_1 + e_2 + e_3)/√3: the cycled-symmetric combination. Color-triplet's SU(3)-singlet-projection site.

These two axes are orthogonal and span V_triv, naturally separating color singlet from color triplet within the C_3-trivial subspace.

---

## 4. Proof of (b) — real positive h requirement

The framework's y_t and y_b derivations (per `theorem_yukawa_exponent_principle_master.md` §3.3 + master synthesis §3) use a walker per-step amplitude h ∈ {1, 2} which is REAL.

The requirement comes from two structural constraints:

**Real h.** The species's substrate-side coupling is computed as a walker amplitude h^L (or (h/k*)^L) where L is the walker length. For y_t = 1 (a real number) and y_b ≈ Q^g (also real positive), the walker amplitude per step must be real. Complex h (chir 5/3 or chir 7) would produce oscillating contributions that don't match the framework's y_t = 1 and y_b ≈ Q^g identifications.

(More fundamentally: the color triplet, unlike the color singlet, does not source the framework's chirality factor — neither chir 5/3 (which is the U(1)_Y hypercharge normalization per α₁_full) nor chir 7 (the neutrino-amplitude chirality per ν_amp = √7/4). Color triplets use the real-h "amplitude/walker" branch.)

**Positive h.** A5(b) (`framework_axioms.md` §5b) identifies above-waterline NB-walk probability with physical coupling strength: y = P_MDL(walk). MDL probabilities are non-negative; hence the walker amplitude h must be positive (a negative-amplitude walker would give a negative probability).  ∎

---

## 5. Proof of (c) — Γ trivial λ=+3 is the unique selecting site

By §4(A) §8 corollary, V_triv eigenvalues across {Γ, H, P}:

| Bloch | V_triv eigenvalues | IB roots (h² − λh + 2 = 0) |
|---|---|---|
| Γ | {+3, −1} | {1, 2}; {(−1 ± i√7)/2} |
| H | {−3, +1} | {−1, −2}; {(1 ± i√7)/2} |
| P | {+√3, −√3} | complex (chir 5/3) only |

Applying the §4 (b) requirement of real positive h:

- **P** has no real h (all chir 5/3, complex). Eliminated.
- **Γ trivial λ=−1** has complex h chir 7 (used by ν per §4(B')). Eliminated for color triplet.
- **H trivial λ=+1** has complex h chir 7 (the H antipode of Γ λ=−1). Eliminated for color triplet.
- **H trivial λ=−3** has real h ∈ {−1, −2}, NEGATIVE. Eliminated by positivity.
- **Γ trivial λ=+3** has real h ∈ {+1, +2}, positive. SELECTED.

This is the unique C_3-stable Bloch site whose V_triv satisfies real positive h, so the color triplet's gen-3-anchor Yukawa walker concentrates there.  ∎

---

## 6. Proof of (d) — γ_7 IB-root split per W38

By §5, the color-triplet's gen-3-anchor walker uses an IB root of A(Γ)|_triv λ=+3, which are exactly h ∈ {1, 2} (Step F X5 verification: h² − 3h + 2 = (h − 1)(h − 2) = 0).

The choice between h = 1 and h = 2 within this single Bloch eigenvalue is fixed by the W38 finding: γ_7 = (−1)^F factorizes Bloch-chirality-class with 4/4 empirical correlation, specializing for color triplet to:

- γ_7 = +1 (n=2, ū_R / u-quark family) → "saturated" IB root = h = 1 (|h| = 1, walker amplitude unchanged per step).
- γ_7 = −1 (n=1, d_L / d-quark family) → "Perron" IB root = h = 2 (|h| = k* − 1, walker amplitude Q per step normalized).

**Inheritance status.** The W38 4/4 correlation is probe-grade (mechanism candidate: χ̃ bipartite chirality on srs-z directed arcs). §4(C) takes the W38 finding as STRUCTURAL INPUT; theorem-grade closure via χ̃ ↔ Class-A/B selection is a follow-up probe.  ∎

---

## 7. Proof of Corollaries 1 and 2 — y_t = 1 and y_b = Q^g

**Corollary 1 (y_t = 1).** Per §4(C) (a)–(d), the color triplet n=2 (γ_7 = +1) walker concentrates at Γ trivial λ=+3 with h = 1. The walker length L for y_t is L = 0 (the gen-3-limit assertion: maximal MDL waterline placement of free toggle modes forces n_free → 0; the structural derivation of L = 0 is §4(D)'s territory, here taken as input from `theorem_yukawa_exponent_principle_master.md` §3.3). The master synthesis §3 selection rule:

  y_X = chir(X) · Q^L(X) / k*^edge_sel(X)

with chir(y_t) = 1 (real h, no chirality factor), L(y_t) = 0 (saturation), edge_sel(y_t) = 0 (per exponent-principle §3.3 assertion):

  y_t_PT = 1 · (2/3)^0 / 3^0 = **1**

giving m_t(tree) = y_t · v / √2 = 246.22 / √2 = 174.104 GeV. Match to PDG m_t = 172.69 ± 0.30 GeV: **+0.82%** (≈ 5σ post-Family-D residual, the same M_unif-threshold-conditional gap as the framework's existing y_t derivation).

**Corollary 2 (y_b = Q^g).** Per §4(C) (a)–(d), the color triplet n=1 (γ_7 = −1) walker concentrates at Γ trivial λ=+3 with h = 2. The walker length L for y_b is L = g = 10 (full girth-cycle Perron walker, again §4(D)-pending; here taken as input from master synthesis §3). With chir(y_b) = 1, L(y_b) = g = 10, edge_sel(y_b) = 0:

  y_b = 1 · (2/3)^10 / 3^0 = **(2/3)^10 ≈ 0.01734**.

Match to m_b / v ≈ 0.01698 (at m_b MS-bar): **+2.06%** (within the Family-D scale of the master synthesis §5, with the analogous α_s-down-threshold decomposition still open per the master synthesis §7 item 1).  ∎

---

## 8. Computational verification

`proofs/foundations/W39_color_triplet_Gamma_concentration_2026-05-21.py` (7/7 gate checks PASS):

  X1. Color triplet n=1, n=2 Fock blocks decompose as trivial+ω+ω² (W36 inherit). PASS
  X2. SU(3)-invariant projects to (e_1+e_2+e_3)/√3 ∈ V_triv, orthogonal to e_0.   PASS
  X3. Real-h requirement rules out P (complex h chir 5/3 / chir 7).                PASS
  X4. Positivity rules out H trivial λ=−3 (h ∈ {−1, −2} negative).                 PASS
  X5. Ihara-Bass roots of λ=3 are exactly h ∈ {1, 2}.                              PASS
  X6. γ_7 = (−1)^n selects h=1 (n=2, y_t) vs h=2 (n=1, y_b) per W38.               PASS
  X7. Reproduces y_t_PT = 1 (+0.82%) and y_b = Q^g (+2.06%).                       PASS

---

## 9. What this theorem closes; what remains open

**Closes (theorem-grade-conditional on §4(D)).**
- The color-triplet half of the master synthesis §4 structural argument (point (3) of §4).
- The Bloch concentration site for y_t and y_b: Γ trivial λ=+3.
- The Ihara-Bass root choice within λ=+3: h = 1 vs h = 2 via γ_7 = (−1)^n.
- The 4-cell (color, γ_7) factorization of the §3 selection table.

**Does NOT close** (these are upstream / orthogonal):

- *Walker length L derivation.* §4(C) assumes L = 0 (y_t saturation) and L = g (y_b Perron walker) as inputs from the master synthesis §3 + the exponent-principle §3.3 (which itself is the framework's exponent-principle hypothesis, theorem-grade-conditional on V_Ram ≅ Cl(6)-Fock mechanical n_free derivation = Need-D-3 / R-14). The §4(D) sub-theorem is the rigorous derivation; until §4(D) closes, §4(C) is theorem-grade-conditional on these L values.

- *Mechanism behind γ_7 grading.* W38's 4/4 correlation is probe-grade; theorem-grade closure via χ̃ ↔ Class-A/B selection is a follow-up probe (W40+).

- *y_b residual decomposition.* §4(C) reproduces y_b ≈ Q^g at +2.06%; the analogous α_s-down-threshold + sub-leading decomposition (parallel to y_t's commit 66c8836) closing the +2% gap is a bounded ~1-session follow-up per master synthesis §7 item 1.

- *Light-generation color-triplet Yukawas* (y_c, y_u, y_s, y_d): from y_t and y_b via within-sector Koide rotations using ε²_up (Row P37 + R4-pinned band) and ε²_down. Bounded per-channel, per master synthesis §7 item 3.

---

## 10. Cross-references

**Builds on:**
- `theorem_C3_block_decomposition_2026-05-21.md` (§4(A)) — V_triv structure at C_3-stable {Γ, H, P}.
- `theorem_color_singlet_P_concentration_2026-05-21.md` (§4(B)) — Cl(6) color C_3 ≡ vertex C_3 algebraic identification, V_triv axis basis.
- `theorem_neutrino_chir7_concentration_2026-05-21.md` (§4(B')) — neutrino chir-7 sibling of singlet branch.
- `theorem_charge_before_color.md` §9 — Cl(6) Fock decomposition 1 ⊕ 3 ⊕ 3̄ ⊕ 1.
- `theorem_yukawa_exponent_principle_master.md` §3.3 — y_t = 1 framework derivation.

**Cited by:**
- `docs/theorems/theorem_yukawa_master_theory_synthesis_2026-05-20.md` §4 point (3).

**Successor pieces** (still SKETCH / open):
- §4(D) Hamming weight → walker length L via MDL waterline (the deepest piece).
- W40+ χ̃ ↔ Class-A/B mechanism probe (graduates W38 to theorem-grade).
