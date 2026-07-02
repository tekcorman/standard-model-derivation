# Theorem — Color singlet without chir-5/3 → chir-7 at Γ/H → neutrino (Yukawa master §4(B'))

**Date:** 2026-05-21
**Status:** THEOREM-GRADE. Algebraic argument + computational verification (W37 probe, 7/7 gates PASS) + reproduction of two pre-existing framework predictions (R_ν = 228/7 and ν_amp = √7/4).

**Purpose.** Sibling theorem to `theorem_color_singlet_P_concentration_2026-05-21.md` (§4(B)). Together they exhaust the master Yukawa synthesis's "color singlet" branch:

- **§4(B)**:  color singlet with chir 5/3 input → V_triv at P → y_τ (lepton).
- **§4(B')** (this doc): color singlet without chir 5/3, with chir 7 input → V_triv at Γ (λ=−1) or H (λ=+1) → neutrino sector (R_ν splitting + ν amplitude).

Identifies the substrate concentration site for the neutrino's within-sector structural content, separate from the gen-3 mass-scale anchor (which uses the asymptotic Laplacian band edge per the master synthesis §3).

---

## 1. Statement

**Theorem (§4(B') — chir-7 → neutrino concentration).** Let X be a color-singlet fermion species (n ∈ {0, 3} in the Cl(6) Fock decomposition per `theorem_charge_before_color.md` §9). Assume X's Yukawa-vertex structural content requires a Bloch eigenmode with chirality `tan²(arg h) = 7` (the framework's chir-7 phase, used in `predictions/R_nu_splitting.py` and `proofs/foundations/n_point_mass_predictions_2026-05-11.py`). Then X's substrate wavefunction must concentrate at the V_triv subspace of A(Γ) at eigenvalue λ_A = −1, or equivalently at the V_triv subspace of A(H) at eigenvalue λ_A = +1.

**Corollary 1 (R_ν).** The neutrino mass-splitting ratio Δm²₃₁/Δm²₂₁ = 228/7 ≈ 32.57 closes via the K_4 Ihara phase φ = arctan(√7) (the K_4 quotient is exactly A(Γ); `theorem_C3_block_decomposition_2026-05-21.md` §7) + Chebyshev distance n = 5 selected by q³ = 5q − 2 at q = k* − 1 = 2 + the Gaussian integer identity (1 + i√7)⁵ = 176 − 16i√7. Match to NuFIT 6.0 (Δm²₃₁/Δm²₂₁ = 33.83 ± 0.92): **1.4σ**.

**Corollary 2 (ν amplitude).** The Class-1 amplitude ν_amp = |Im(h)|/|h|² evaluates to √7/4 at both chir-7 sites h_Γ = (−1+i√7)/2 and h_H = (+1+i√7)/2.

**Corollary 3 (color-singlet branching).** The framework's color singlet has access to *multiple* chirality contents in V_triv (chir 5/3 at P; chir 7 at Γ λ=−1, H λ=+1; real h ∈ {1, 2, −1, −2} at Γ/H λ=±3). The species' Yukawa-vertex input chirality determines which of these the wavefunction concentrates at:

| Color-singlet sub-branch | V_triv ∩ site | Yukawa-vertex content |
|---|---|---|
| chir 5/3 input | P-saddle | y_τ (§4(B)) |
| chir 7 input | Γ λ=−1 OR H λ=+1 | neutrino R_ν, ν_amp (§4(B'), this theorem) |
| asymptotic spectral | Laplacian band edge | y_ν3 gen-3 mass scale (separate mechanism) |
| real h ∈ {1, 2} | Γ λ=+3 | NOT color singlet — y_t color triplet |
| real h ∈ {−1, −2} | H λ=−3 | NOT color singlet — y_b-like color triplet at H antipode |

---

## 2. Setup and inputs

**Inherits from §4(A)** (`theorem_C3_block_decomposition_2026-05-21.md`):
- V_triv at Γ has A(Γ) eigenvalues {3, −1}.
- V_triv at H has A(H) eigenvalues {−3, +1}.
- V_triv at P has A(P) eigenvalues {+√3, −√3}.
- C_3-stable Bloch points are exactly {Γ, H, P}.

**Inherits from §4(B)** (`theorem_color_singlet_P_concentration_2026-05-21.md`):
- Color singlets (n ∈ {0, 3} Cl(6) Fock states) are C_3-trivial in the Fock space (§4(B) §4).
- A color-singlet wavefunction has its vertex-space content in V_triv (§4(B) §5).
- The Cl(6) color C_3 IS the §4(A) body-diagonal C_3 (§4(B) §3).

**Pre-existing framework derivations**:
- `predictions/R_nu_splitting.py` — derives R_ν = 228/7 from the K_4 Ihara phase φ = arctan(√7) + Chebyshev n = 5. THEOREM-grade form documented in `docs/parameters/R_theorem.md`; conditional on Rows 16, 17, 18 of the uniqueness ledger (Cl(6;ℂ), Pati-Salam, C³_obs structures).
- `proofs/foundations/n_point_mass_predictions_2026-05-11.py` — identifies ν_amp = |Im(h)|/|h|² = √7/4 at h_Γ = (−1+i√7)/2 and h_H = (+1+i√7)/2.
- `proofs/wave_engine/dark_5_12_spectral.py` — identifies the 6-dim oscillatory subspace at Γ as the chir-7 modes (multiplicity-3 λ_A = −1 eigenvalue × 2 Ihara-Bass roots).

---

## 3. Proof — chir-7 lives in V_triv at Γ (λ=−1) and H (λ=+1)

By the §4(A) explicit block spectra (theorem (e)):
- A(Γ) V_triv has eigenvalues {3, −1}.
- A(H) V_triv has eigenvalues {−3, +1}.

Apply Ihara–Bass h² − λ·h + (k*−1) = 0 with k* = 3 on the V_triv eigenvalues:

- λ = −1: h² + h + 2 = 0 ⟹ h = (−1 ± i√7)/2. **chir tan²(arg h) = 7**, |h|² = 2 (Ramanujan).
- λ = +1: h² − h + 2 = 0 ⟹ h = (+1 ± i√7)/2. **chir tan²(arg h) = 7**, |h|² = 2.

The chir-7 eigenvalues live in V_triv at Γ (via λ = −1) and H (via λ = +1). Computational verification: W37 Step B V1 PASS. ∎

---

## 4. Proof — the Ihara phase identity 7 = 4(k* − 1) − 1

For the K_4 graph (the complete graph on 4 vertices, which is exactly A(Γ) for the srs primitive cell by `theorem_C3_block_decomposition_2026-05-21.md` §7), the Ihara phase satisfies

  φ = arctan(√(4(k* − 1) − 1)).

At k* = 3: 4·2 − 1 = **7**, so φ = arctan(√7). The number 7 in chir-7 is the same 7 in the K_4 Ihara phase — these are not independent.

W37 Step C V2 + Step D V3 PASS. ∎

---

## 5. Proof of Corollary 1 — R_ν = 228/7

Direct re-derivation following `predictions/R_nu_splitting.py`:

1. **Chebyshev selection.** The cubic q³ = 5q − 2 has the unique positive integer root q = 2 (factoring as (q − 2)(q² + 2q − 1) = 0 with the second factor having irrational roots). At q = k* − 1 = 2, the Chebyshev-U expansion of the K_4 Green's function selects distance n = 5.

2. **Gaussian integer identity.** Expand (1 + i√7)⁵ via the binomial theorem in the ring ℤ[i√7]. The real and imaginary parts are integer multiples of √7-conjugate norms:

   (1 + i√7)⁵ = 1 + 5i√7 + 10(i√7)² + 10(i√7)³ + 5(i√7)⁴ + (i√7)⁵
              = 1 + 5i√7 − 70 − 70i√7 + 245 + 49i√7
              = (1 − 70 + 245) + (5 − 70 + 49)i√7
              = **176 − 16i√7**.

3. **sin²(5φ) evaluation.** sin(5φ) = Im((1 + i√7)⁵) / |1 + i√7|⁵ = (−16√7) / 8^(5/2) = −16√7 / (32√2 · 2√2) = √7/(8√2). [|1 + i√7|² = 8.] Squaring: sin²(5φ) = 7 / 128.

4. **R_ν.** R_ν = 2/sin²(5φ) − 4 = 256/7 − 4 = (256 − 28)/7 = **228/7 ≈ 32.5714**.

Match: NuFIT 6.0 (Sep 2024, normal ordering) gives Δm²₃₁/Δm²₂₁ = 33.83 ± 0.92 — 1.4σ from 228/7. W37 Step E V4 PASS. ∎

---

## 6. Proof of Corollary 2 — ν_amp = √7/4

The Class-1 amplitude is the standard substrate quantity |Im(h)|/|h|² used in the framework's CKM-element derivations (V_us, V_ub, V_cb per `n_point_mass_predictions_2026-05-11.py`). Evaluating at the two chir-7 sites:

- h_Γ = (−1 + i√7)/2: |Im(h_Γ)| = √7/2, |h_Γ|² = (1 + 7)/4 = 2.
  ⟹ ν_amp(h_Γ) = (√7/2) / 2 = **√7/4**.

- h_H = (+1 + i√7)/2: |Im(h_H)| = √7/2, |h_H|² = 2.
  ⟹ ν_amp(h_H) = √7/4. (Identical: Γ and H are antipodal partners.)

For contrast, the framework's other Ramanujan saddles give:
- h_P = (√3 + i√5)/2: ν_amp(h_P) = √5/4 (used in V_us, V_ub, V_cb).
- h_N = (√5 + i√3)/2: ν_amp(h_N) = √3/4 (N-saddle, different sector).

W37 Step F V5 PASS. ∎

---

## 7. Proof of Corollary 3 — color-singlet branching map

By §4(B) §5, a color-singlet wavefunction Ψ has Ψ_vertex ∈ V_triv at any C_3-stable Bloch point. By §4(A) §8 corollary, V_triv at each C_3-stable site has the following chirality content:

| Bloch | V_triv eigenvalues | Ihara–Bass h | Chirality |
|---|---|---|---|
| Γ | {+3, −1} | {1, 2}; (−1±i√7)/2 | 0 (real); **7** |
| H | {−3, +1} | {−1, −2}; (1±i√7)/2 | 0 (real); **7** |
| P | {+√3, −√3} | (±√3±i√5)/2 | **5/3** |

So the chir-content of V_triv across C_3-stable sites is {real h ∈ {±1, ±2}, chir 5/3 at P, chir 7 at Γ/H λ=∓1}. The color singlet's concentration site is determined by which chirality content its Yukawa-vertex input requires:

- *chir 5/3* → only P trivial supplies it ⇒ y_τ → P (§4(B)).
- *chir 7* → Γ (λ=−1 trivial) or H (λ=+1 trivial) supplies it ⇒ neutrino → Γ or H (§4(B'), this theorem).
- *asymptotic spectral* (Laplacian L_us) → none of the discrete V_triv blocks; uses the framework's spectral seesaw at the band edge ⇒ y_ν3 gen-3 mass-scale → Laplacian (separate mechanism per master synthesis §3).
- *real h* alone → would be a color-singlet candidate at Γ λ=+3 (h=1 saturation) or H λ=−3 (h=−1, −2), but the framework's color-singlet species don't use the real-h saturation mode for their Yukawa derivations (the real-h modes are used by quarks, which are color triplet, not singlet — §4(C) territory).

The neutrino's "three concentration sites" picture (gen-3 mass scale at Laplacian, splitting at chir-7, amplitude at chir-7) is consistent with this branching: y_ν3 takes the spectral branch for its absolute scale; the within-sector splitting and amplitude take the chir-7 branch for their structural content. ∎

---

## 8. Computational verification

`proofs/foundations/W37_chir7_neutrino_concentration_2026-05-21.py` (7/7 gate checks PASS):

  V1. Chir-7 eigenvalues (−1+i√7)/2, (1+i√7)/2 in V_triv at Γ/H.            PASS
  V2. Ihara phase identity 7 = 4(k*−1) − 1.                                  PASS
  V3. K_4 Ihara phase arctan(√7) matches chir-7 argument.                    PASS
  V4. R_ν = 228/7 ≈ 32.57 from chir-7 (NuFIT match 1.4σ).                   PASS
  V5. ν_amp = √7/4 at both h_Γ and h_H chir-7 sites.                         PASS
  V6. Chir-7 accessible in V_triv (color singlet) AND V_ω, V_ω² (color triplet) at Γ, H. PASS
  V7. 6-dim visible oscillatory Hashimoto subspace at Γ = chir-7 home (from dark_5_12_spectral.py). PASS

---

## 9. What this theorem closes; what it does NOT close

**Closes.**
- The "color singlet without chir-5/3" branch of the master synthesis §3 selection rule, identifying the chir-7 modes at Γ/H trivial blocks as the structural home for the neutrino's within-sector content (R_ν splitting + Class-1 ν amplitude).
- The structural grounding for `predictions/R_nu_splitting.py`'s 228/7 derivation: the K_4 Ihara phase φ = arctan(√7) is the chir-7 argument at h_H, and the K_4 graph is A(Γ) of the srs primitive cell.
- The §4(B)/§4(B') siblings complete the color-singlet half of the master synthesis §4 structural argument.

**Does NOT close** (these remain open):

- *Why the neutrino uses chir-7 specifically (rather than some other chirality).* Currently the framework treats this as empirically grounded: R_ν matches observation at 1.4σ; ν_amp matches in `V_us`, `V_cb` derivations via the same Class-1 amplitude machinery. A theorem-grade derivation of "neutrino's Yukawa-vertex input chirality = 7" from the framework's MDL / waterline / Bloch concentration structure is downstream of §4(B') and would close the residual upstream question.

- *The relationship between the three Ramanujan chiralities {3/5, 5/3, 7} and the SM gauge-coupling normalizations.* chir 5/3 ↔ g_1 = √(5/3) · g_Y is the framework's hypercharge normalization. Whether 3/5 (N-saddle) and 7 (Γ/H chir-7) correspond to other gauge structures or SO(10) breaking patterns is open.

- *Whether the Cl(6) chirality element γ_7 := i·γ_1...γ_6 (the "7th gamma" of Cl(6), Hermitian with γ_7² = I, acting as fermion parity (−1)^F per `theorem_car_local_jordan_wigner.md` §9.1) is structurally linked to the chir-7 = tan²(arg h) = 7 of this theorem, or is a labeling coincidence.* The two "7"s could be deeply related (fermion parity ↔ neutrino chirality) or independent (Cl(6) chirality is a Z_2 grading; chir-7 is a Bloch phase). A focused probe could settle this.

---

## 10. Cross-references

**Builds on:**
- `theorem_C3_block_decomposition_2026-05-21.md` (§4(A)) — chir-7 inventory in V_triv at Γ/H.
- `theorem_color_singlet_P_concentration_2026-05-21.md` (§4(B)) — color-singlet wavefunction ⊂ V_triv setup.
- `theorem_charge_before_color.md` §9 — Cl(6) Fock decomposition.

**Verifies (computationally) and structurally explains:**
- `predictions/R_nu_splitting.py` — R_ν = 228/7 via K_4 Ihara phase + Chebyshev (now grounded in the chir-7 ↔ V_triv at Γ/H concentration of this theorem).
- `proofs/foundations/n_point_mass_predictions_2026-05-11.py` — ν_amp = √7/4 at h_H, h_Γ.
- `proofs/wave_engine/dark_5_12_spectral.py` — 6-dim visible oscillatory subspace at Γ.

**Cited by:**
- `docs/theorems/theorem_yukawa_master_theory_synthesis_2026-05-20.md` §3 selection table (chir-7 branch row to be added) and §4 structural argument (new point (2b) following §4(B)).

**Successor theorems** (the rest of §4's structural sketch):
- §4(C) color triplet → Γ (planned next): the parallel branch for color triplets at h = 1, h = 2 real-h saturation/Perron walks (y_t, y_b).
- §4(D) Hamming weight → walker length L (planned, multi-session).

**Open probes seeded by this theorem:**
- γ_7 ↔ chir-7 link probe (1 session, bounded). Worth running.
- "Why neutrino chirality = 7" upstream derivation (multi-session, research-grade).
