# Theorem — the body-centering mirror: antiperiod, zeta factorization, sectors, towers

**Date:** 2026-06-11
**Status:** THEOREM-GRADE (T1–T5 numerically established at machine precision by
four backbone probes, verify.py entries; T1, T4, T5 admit short closed-form
proofs sketched below). Interpretive wording panel-corrected (2026-06-11
ultracode adjudication, verdict PARTIAL — see Honest Scope).
**Probes:** `proofs/foundations/zeta_factorization_srs_srsz_2026-06-10.py`,
`zeta_channel_dictionary_probe_2026-06-10.py`,
`phase1_3_s1_mirror_is_bodycentering_2026-06-11.py`,
`phase1_3_zeta_sectors_parity_mirrorgirth_2026-06-11.py`,
`phase1_3_s3_winding_towers_2026-06-11.py`,
`phase1_3_neutrino_mirror_saddles_2026-06-11.py`.

---

## Setup

srs primitive cell (proofs/common.py): 4 atoms, 12 directed bonds with
cell-offsets c ∈ ℤ³ in the BCC primitive basis A_PRIM; Bloch adjacency A(k)
(4×4) and Bloch–Hashimoto B(k) (12×12, non-backtracking) with phases
e^{2πi k·c}. Mirror shift **Δ = (½, ½, −½)** (primitive-fractional) = the
image of the odd cubic reciprocal shift; dually, the body-centering
translation t = ½(1,1,1)_cubic = (1,1,1)_prim.

## T1 — Matrix antiperiod

**A(k+Δ) = −A(k) and B(k+Δ) = −B(k), exactly, for all k.**

*Proof.* Every srs bond cell-offset has odd coordinate sum (verified
exhaustively: all 12 bonds), so Δ·c ≡ ½ (mod 1) and e^{2πi(k+Δ)·c} =
−e^{2πik·c} for every bond. ∎

Consequence: spec A(k+Δ) = −spec A(k) (the spectral antiperiod), and the
Δ-shifted Hashimoto fiber equals the sign-voltage-twisted fiber:
det(I − uB(k+Δ)) = det(I − uB_sgn(k)).

*Caveat (panel):* T1 is generic-in-k; the mechanism is BCC reciprocal
arithmetic + the parity of srs's bond offsets, not deep srs-specific
structure. srs-specific content enters through the spectra.

## T2 — Cover identification

**Crystallographic srs-z (P4₁32, 8 atoms/cubic cell) = srs folded onto cubic
translations = the bipartite double cover of srs, with deck transformation =
the body-centering translation.** Per cubic momentum κ:
spec A_srsz(κ) = spec A_srs(k₁) ∪ spec A_srs(k₁+Δ), k₁ = A_PRIM·κ, exactly;
likewise at the Hashimoto-determinant level.

## T3 — Zeta factorization (Stark–Terras, geometric character)

**ζ_srs-z(u) = ζ_srs(u) · L(u, sgn)**, with the sign character realized
geometrically as body-center winding parity. Per fiber:
det(I − uB_srsz(κ)) = det(I − uB(k₁)) · det(I − uB(k₁+Δ)),
and L(u, sgn; k)⁻¹ = (1−u²)² det(I + A(k)u + 2u²) = the Δ-shifted fiber.
Bass identity per fiber: det(I − uB(k)) = (1−u²)² det(I − A(k)u + 2u²).
"Mirror holonomy" (grade-blind mass classification) = parity of body-center
windings of the closed walk.

## T4 — Parity theorem and sector girths

For every closed NB walk of length L closing by translation R (primitive
coords): **(−1)^L = (−1)^{ΣR}** — walk-length parity equals body-center-coset
parity. (*Proof:* immediate from T1's odd-offset property: each step flips
the coset parity. ∎)

Translation-resolved cycle counts N_L(R) = ∫_BZ Tr[B(k)^L] e^{−2πik·R} dk
(exact on uniform grids; trig-polynomial integrand):
- **Bulk sector (R = 0):** girth 10, N₁₀ = 120; V_us = k\*³/N₁₀ = 9/40 is a
  zeta-coefficient functional.
- **Mirror (odd) sector:** girth **3 = k\***; 3 NB triangles per ⟨111⟩
  direction (8 directions) — the srs helix axes.
- **Screw sector (cubic-axis R):** girth 4; 4 cycles per axis direction (6).

## T5 — Winding towers (forced symmetry content)

**N(3n, n·(1,1,1)) = 3 and N(4n, n·(1,1,0)) = 4 for all n ≥ 1** (verified
n = 1..4; forced by the C₃ [111] and 4₁ ⟨100⟩ screw orbits — the minimal
n-wound cycles along an axis are exactly the 3 (resp. 4) phase-offset
helices; panel Lens D extremal argument). The u^{8m} geometric series
(V_cb winding series) numerically coincides with even windings of the 4₁
tower — see Honest Scope for why this remains a CANDIDATE address only.

## Saddle orbit map (exact)

Under k ↦ k+Δ: **Γ ↔ H** (exchanged; H = Γ+Δ exactly, so the H fiber is the
L(u,sgn) fiber at Γ); **P ↦ −P** (mod ℤ³; mirror acts at P as complex
conjugation; self-conjugate); **N ↦ N\*** (self-conjugate; spec A(N) =
±{1, √5}). Holonomy inventory at P: exactly 4 modes with λ¹⁰ = 1 (the
fiber-generic Ihara–Bass (1−u²)^{|E|−|V|} content — 4 at every generic k)
and 8 Ramanujan modes with arg(λ¹⁰) = ±162.3876°; at Γ/H the Ramanujan
modes (arg ±110.705°/±69.295°) give arg(λ¹⁰) = ±27.05°.

## Honest scope (panel verdict, 2026-06-11)

What this theorem does **not** establish:

1. **No over-determination of the walker dictionary.** The Γ/H antipodality
   A(H) = −A(Γ) was documented 2026-05-21 and underlies the 2026-05-27
   ν-saddle assignment; T1–T3 *explain* that fact (one fact, two readings).
   The explanation is the contribution; agreement carries no evidential weight.
2. **The neutrino reading is a dictionary-conditional CANDIDATE.** Conditional
   on the 2026-05-27 walker dictionary (A5-mass freedom priced 15.3 bits),
   (Γ,H) is the unique mirror-exchanged fermionic saddle pair and P is
   self-conjugate — consilient with a Majorana reading; NOT forced.
3. **Anchor sector-addresses are unforced selections** (~3–6 bits, itemized in
   the Phase 1.3 spec): L=8 carries three nonzero sector classes with two
   competing in-repo homes; u¹⁰ ↔ girth is tautological.
4. **The Majorana-phase holonomy identification remains ADOPTED-NU-MAJ-PHASE**
   (undischarged). Open seam: the mass chain reads the phase at P (162.39°)
   while the dictionary's ν_R saddle (H) gives 27.05°.

## Dependencies / downstream

Upstream: proofs/common.py cell conventions; Bass (1992) / Stark–Terras
(1996/2000) graph zeta and L-function theory; Phase 1.1–1.3 probes.
Downstream consumers: Phase 1 zeta channel dictionary
(`docs/scoping/zeta_channel_dictionary_2026-06-10.md`), Phase 1.3 bet spec,
self-MDL ledger (2026-06-11 amendment).
