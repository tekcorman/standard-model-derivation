# Theorem — Color singlet → P-saddle concentration (Yukawa master §4(B))

**Date:** 2026-05-21
**Status:** THEOREM-GRADE. Algebraic argument + computational verification (W36 probe, 7/7 gates PASS).

**Purpose.** Second of four structural sub-theorems lifting §4 of `theorem_yukawa_master_theory_synthesis_2026-05-20.md` from sketch to theorem-grade. Combines `theorem_charge_before_color.md` §9 (Cl(6) Fock decomposition by Hamming weight) with `theorem_C3_block_decomposition_2026-05-21.md` (§4(A)) to force the spatial concentration of any chir-5/3-carrying lepton-class Yukawa coupling to the P-saddle Bloch point.

---

## 1. Statement

**Theorem (§4(B) — Color singlet → P-saddle).** Let X be a color-singlet fermion species of the framework's Cl(6) Fock identification (per `theorem_charge_before_color.md` §9): X is hosted by the Hamming-weight blocks n ∈ {0, 3} at a trivalent vertex. Assume X's Yukawa-coupling derivation requires a Bloch eigenmode with chirality `tan²(arg h) = 5/3` (the framework's chir-5/3 phase, used in `predictions/alpha_1_full.py`). Then X's substrate wavefunction must concentrate at the Bloch point P of the BCC primitive BZ.

Equivalently: **chir 5/3 is available to a color singlet ONLY at P** among the C_3-stable Bloch points {Γ, H, P}.

**Corollary.** The framework's existing y_τ derivation (`theorem_ytau_corollary.md`) factorizes as

  y_τ = α₁_full / k*² = (5/3)·(2/3)^8 / 9 = 1280 / 177147 ≈ 7.226 × 10⁻³ (+0.13% of m_τ/v),

and the (5/3) factor is the chirality of h at P. This theorem supplies the spatial-concentration justification: y_τ uses chir 5/3 ⇒ y_τ → P.

---

## 2. Setup

**Cl(6) Fock at a trivalent vertex** (`theorem_charge_before_color.md` §9). At each k*=3-valent vertex v of srs, the local Fock space is

  H_v = ℂ⁸ = |000⟩ ⊕ {|100⟩, |010⟩, |001⟩} ⊕ {|110⟩, |101⟩, |011⟩} ⊕ |111⟩,

decomposing under U(3) ⊂ Spin(6) as `1 ⊕ 3 ⊕ 3̄ ⊕ 1` indexed by Hamming weight n = b_1 + b_2 + b_3.

**Color-singlet subspace.** The SU(3) singlet states are the n ∈ {0, 3} blocks (the "extreme" Hamming weights). Per Furey 2018 §3 / Baez-Huerta 2010 §4, these correspond to ν_L (n=0) and e_L^+ (n=3) of one SM generation.

**Body-diagonal C_3.** R(x, y, z) = (z, x, y), the 120° rotation around (1,1,1)/√3, fixing v_0 and cycling v_1 → v_3 → v_2 → v_1 (per `theorem_C3_block_decomposition_2026-05-21.md` §3).

**Color C_3.** The cyclic subgroup C_3 ⊂ SU(3) permuting the 3 edge modes (a_1, a_2, a_3) at v_0 by σ: 1 → 3, 2 → 1, 3 → 2 — i.e., (a_1, a_2, a_3) ↦ (a_2, a_3, a_1). (Choice of cyclic direction matches the §4(A) vertex cycle under the natural bijection.)

---

## 3. Proof — the color C_3 IS the body-diagonal C_3

**Bijection edge_i ↔ v_i.** At v_0, the 3 incident edges go to the 3 cycled vertices: edge 1 to v_1, edge 2 to v_2, edge 3 to v_3. Under the body-diagonal R:

  v_1 → v_3 ⇒ edge 1 → edge 3
  v_2 → v_1 ⇒ edge 2 → edge 1
  v_3 → v_2 ⇒ edge 3 → edge 2

This is exactly the edge permutation σ given above. The two C_3 subgroups — the "color C_3" (cycling edge labels at v_0) and the "body-diagonal C_3" (cycling vertex labels in the primitive cell) — act on the same 3-element set with the same cyclic structure. They are the same group.

**Lift to the Fock space.** σ on edge labels lifts to a unitary U_σ on the Cl(6) Fock space via (a_i, a_i†) → (a_{σ(i)}, a_{σ(i)}†). On basis states, U_σ|b_1 b_2 b_3⟩ = |b_{σ⁻¹(1)} b_{σ⁻¹(2)} b_{σ⁻¹(3)}⟩ = |b_2 b_3 b_1⟩.

**W36 Step B verifies** U_σ³ = 1 on the Fock space. ∎

---

## 4. Proof — color singlets are C_3-trivial in the Fock space

By direct enumeration of fixed points under U_σ on the 4 Hamming-weight blocks (W36 Step C):

| Hamming weight n | dim | C_3 decomposition |
|---|---|---|
| n = 0 ( \|000⟩ ) | 1 | trivial (singleton fixed by σ) |
| n = 1 ( {\|100⟩, \|010⟩, \|001⟩} ) | 3 | trivial + ω + ω² (3-cycle permutation rep) |
| n = 2 ( {\|110⟩, \|101⟩, \|011⟩} ) | 3 | trivial + ω + ω² (3-cycle permutation rep) |
| n = 3 ( \|111⟩ ) | 1 | trivial (singleton fixed by σ) |

The character of U_σ on each block computed via tr(U_σ|_n) = # fixed states:
  n=0: tr(U_σ) = 1 (only |000⟩, fixed). ⇒ trivial-only.
  n=1: tr(U_σ) = 0 (3-cycle has no fixed states). Combined with multiplicity tr(U_σ²)= 0 and tr(U_σ⁰)=3 ⇒ inner products with χ_triv, χ_ω, χ_{ω²} all equal 1 ⇒ trivial + ω + ω².
  n=2: same as n=1 by the same arithmetic.
  n=3: tr(U_σ) = 1 (only |111⟩, fixed). ⇒ trivial-only.

So the SU(3)-singlet states (n ∈ {0, 3}) are precisely the C_3-trivial states of the Fock space.

**W36 Step C U3 verifies** decomp_n0 = decomp_n3 = (1·trivial, 0·ω, 0·ω²) numerically. ∎

---

## 5. Proof — color-singlet wavefunction lies in V_triv (vertex space)

A fermion wavefunction is a section of (vertex space ℂ⁴) ⊗ (local Fock ℂ⁸) ⊗ (other DOFs). The §4(A) body-diagonal R acts on the vertex space (cycling v_1 → v_3 → v_2) and is identified at §3 above with U_σ on the Fock space.

A color-singlet wavefunction is, by definition, SU(3)-invariant; in particular C_3-invariant under U_σ. Restricting to the (vertex space ⊗ Fock) sector, the wavefunction's Fock-space content is in trivial(Fock), and its vertex-space content must then be in V_triv (the C_3-trivial subspace of ℂ⁴): if Ψ = Σ_i Ψ_v(i) ⊗ ψ_Fock(i) is C_3-invariant under R ⊗ U_σ and ψ_Fock(i) is C_3-trivial, then R · Ψ_v(i) = Ψ_v(i) for all i, hence Ψ_v ∈ V_triv.

**V_triv is 2-dimensional** (§4(A) theorem (b)), with orthonormal basis {e_0, (e_1+e_2+e_3)/√3}. So a color-singlet wavefunction has its vertex-space content concentrated in this 2-d subspace.

**The fixed-vertex e_0 is in V_triv** (W36 Step G U7): P_triv · e_0 = e_0; P_ω · e_0 = P_{ω²} · e_0 = 0. The framework's identification "lepton lives at the C_3-fixed vertex" picks e_0 within V_triv. ∎

---

## 6. Proof — chir 5/3 is unique to V_triv at P

By §8 of `theorem_C3_block_decomposition_2026-05-21.md`, the trivial-block content of A(k) at each C_3-stable Bloch point is:

| Bloch point | A_triv(k) eigenvalues | h via Ihara–Bass | chirality tan²(arg h) |
|---|---|---|---|
| Γ | {+3, −1} | {1, 2} (from λ=3); (−1 ± i√7)/2 (from λ=−1) | 0; **7** |
| H | {−3, +1} | {−1, −2} (from λ=−3); (1 ± i√7)/2 (from λ=1) | 0; **7** |
| P | {+√3, −√3} | (±√3 ± i√5)/2 | **5/3** |

The chirality content of V_triv across the three C_3-stable sites is

  V_triv chiralities = {0 (Γ, H — real h), 5/3 (P), 7 (Γ, H — complex h from λ=∓1)}.

**Chirality 5/3 appears EXCLUSIVELY at P** within V_triv. Γ's trivial block has only real h and chir-7 complex h; H's trivial block has only real h (with opposite sign) and chir-7 complex h.

**Numerical verification** (W36 Step E U5): the chir-5/3 sites for V_triv are exactly {P}.  ∎

---

## 7. Proof — combining §§3–6 forces y_τ → P

Compose the implications:

1. y_τ is a lepton Yukawa coupling: lepton ≡ color singlet (per §2 setup, SM identification of n ∈ {0, 3}).
2. By §§3–5: color-singlet wavefunction Ψ has vertex content in V_triv at any C_3-stable Bloch point.
3. y_τ's existing derivation (`theorem_ytau_corollary.md` §4, `predictions/alpha_1_full.py`) employs the chirality factor tan²(arg h) = 5/3, which is the value at the P-saddle.
4. By §6: chir 5/3 is available in V_triv ONLY at P, not at Γ or H.
5. Therefore Ψ's Bloch concentration site must be P.

Consistency check (W36 Step F U6): y_τ_pred = (5/3)·(2/3)^8 / 9 = 1280/177147 ≈ 7.226 × 10⁻³, matching m_τ/v at +0.13%. ∎

---

## 8. What this theorem closes; what it does NOT close

**Closes.** The spatial-concentration content of §4 point (2) of the master Yukawa synthesis: "Color singlet sits in the trivial-C_3 rep at the fixed vertex … Color singlet naturally concentrates at P to host the y_τ-class lepton Yukawa." The "naturally" was a sketch; the present theorem turns it into a forced consequence of §4(A) + Cl(6) Fock decomposition + the chir-5/3 input.

**Does NOT close** (these are upstream / orthogonal):

- *Why y_τ has chirality 5/3 in the first place.* The chir-5/3 factor enters via `predictions/alpha_1_full.py`, which uses the Ramanujan P-saddle eigenvalue h = (√3 + i√5)/2 (Type-4 upstream A5(a) + Ihara–Bass). §4(B) takes this as input.

- *Why the neutrino (also color singlet, n=0) does NOT concentrate at P.* The neutrino's Yukawa derivation uses the Laplacian band-edge L_us = 2 + √3, not chir 5/3. It is "color singlet without chir-5/3" — the §4(B) hypothesis "X requires chir 5/3" fails, and the theorem doesn't apply. The neutrino's selection rule (y_ν3 → Laplacian band edge) belongs to a different branch of §4 — its concentration site is in the asymptotic spectral regime, not in the discrete C_3 isotypic blocks.

- *The within-sector Koide rotation for lighter generations* (y_μ, y_e from y_τ): handled by within-sector ε² + δ Koide structure, downstream of §4(B).

---

## 9. Computational verification

`proofs/foundations/W36_color_singlet_concentration_2026-05-21.py` (7/7 gate checks PASS):

  U1. Cl(6) Fock at v_0 decomposes as 1 ⊕ 3 ⊕ 3̄ ⊕ 1 by Hamming weight.   PASS
  U2. C_3 acts on n=1 and n=2 blocks as trivial + ω + ω².                  PASS
  U3. Color singlets (n ∈ {0, 3}) are C_3-trivial in Fock.                 PASS
  U4. Cl(6) color C_3 ≡ §4(A) body-diagonal C_3 (algebraic identification). PASS
  U5. Chirality 5/3 unique to V_triv at P (inherits §4(A) §8 corollary).    PASS
  U6. y_τ_pred = α₁_full/k*² = 7.226e-3 matches m_τ/v at +0.13%.            PASS
  U7. e_0 (C_3-fixed vertex basis vector) ∈ V_triv.                        PASS

---

## 10. Cross-references

**Builds on:**
- `theorem_C3_block_decomposition_2026-05-21.md` (§4(A)) — supplies the C_3-stable Bloch points and the trivial-block chirality inventory.
- `theorem_charge_before_color.md` §9 — supplies the Cl(6) Fock decomposition by Hamming weight + the SU(3) ⊂ U(3) ⊂ Spin(6) embedding.
- `theorem_ytau_corollary.md` — supplies the y_τ derivation using the P-saddle chir-5/3 factor via α₁_full.
- `predictions/alpha_1_full.py` — defines α₁_full = (5/3)(2/3)^8 with (5/3) = tan²(arg h) at P.
- `theorem_car_local_jordan_wigner.md` — supplies the Cl(6) Fock structure at each k*=3-valent vertex.

**Cited by:**
- `docs/theorems/theorem_yukawa_master_theory_synthesis_2026-05-20.md` §4(2).

**Successor theorems** (the rest of §4's structural sketch):
- §4(C) color triplet → Γ (planned next).
- §4(D) Hamming weight → walker length L (planned, multi-session).
