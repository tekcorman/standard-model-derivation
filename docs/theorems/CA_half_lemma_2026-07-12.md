# The C_A = ½ lemma — the particle-hole symmetry of the vacuum covariance and its odd-dimension corollary

**Date:** 2026-07-12 (Push 3, W3a — "cheap standalone lemma" opener).
**Status:** **LEMMA-GRADE, PROVEN + MACHINE-CHECKED.** Pure linear algebra on the already-built object (`derivation_topdown/state/the_net.py`'s `complex_structure_J6` / `vacuum_covariance` / `region_data`); no new construction, no physical number, no region-geometry input. Closes an item that was previously only an observed 4-orbit-exact fact inside the FOCK-0d region machinery (`_three_edge_region_orbits` / `field_side_flow_generator`), never derived from first principles.
**Verification:** `../../proofs/foundations/CA_half_lemma_check_2026-07-12.py` — **31/31 PASS**. Standalone, seconds-fast, self-reporting. Imports `the_net.py` read-only; does **not** modify it (accretion law: wiring this lemma back into `the_net.py`'s docstrings/API is left to a later architect-directed station).
**Corrects:** the mechanism carried on record as "odd-dimension Schmidt pairing" (internal research notes line 153, internal research notes W3(a)). That name is not wrong as a description of the *effect*, but it named the wrong *cause*. The actual mechanism is a **particle–hole / conjugation symmetry of the vacuum covariance C itself** (§1 below); the exact-½ eigenvalue on odd regions is a **corollary** of that symmetry (§2), not a freestanding "Schmidt pairing" argument on the region's dimension alone.

---

## 1. The pairing symmetry (the real content)

**Setup (verbatim from `the_net.py`).** `J6 = complex_structure_J6()` is the A4-covariant complex structure on the 6-edge cell space: **real**, **antisymmetric** (`J6^T = -J6`), `J6² = -I` (machine-checked to `1e-15`, §0 of the check script — this was already asserted by `the_net.py`'s own docstring at line 71 and is reconfirmed here, not re-derived). The vacuum covariance is `C = (I + i·J6)/2` (`vacuum_covariance`, sign=+1), an exact rank-3 projector on the 6-dim space.

**Step 1 — C is Hermitian, and conj(C) = I − C exactly (global identity).**
Since `J6` is real, `J6^† = J6^T = -J6`, so
```
C^† = ((I + i J6)/2)^† = (I - i J6^T)/2 = (I + i J6)/2 = C
```
`C` is Hermitian. Separately, because `J6` is real (conjugation only touches the `i`):
```
conj(C) = (I - i J6)/2 = I - (I + i J6)/2 = I - C.
```
Both facts follow **purely from J6 being real and antisymmetric** — no other property of `J6` (its A4-covariance, its specific numerical entries) is used. Machine-checked to `2e-16` (script §0).

**Step 2 — restriction to a region commutes with conjugation, for ANY region A.**
A "region" is a coordinate subset `A ⊆ {0,...,5}` of edges; `C_A := C[A,A]` (principal submatrix). Taking a principal submatrix is an entrywise/index selection, which trivially commutes with entrywise complex conjugation for *any* index set — no reality, geometry, or orbit structure of `A` is needed for this step. Hence
```
conj(C_A) = conj(C)_A = (I - C)_A = I_A - C_A          for EVERY region A.
```
This is exactly the identity `the_net.py` already uses at line 3488 under the name "the log-of-inverse identity" (`Cm|_A = I_A − C_A`, where `Cm = vacuum_covariance(sign=-1) = conj(C)`) as a consistency check on the bit-reversal `σ: J → −J`. **This lemma identifies that identity as the base mechanism**, rather than a side-observation about the bit.

**Step 3 — the spectral pairing.** `C_A` is a principal submatrix of a Hermitian matrix, hence Hermitian, hence has a real spectrum. Conjugating a matrix with real spectrum leaves the spectrum unchanged as a multiset (`spec(conj(M)) = conj(spec(M)) = spec(M)` when `spec(M) ⊂ ℝ`). Combined with Step 2 (`conj(C_A) = I_A - C_A`, whose spectrum is `{1 − λ : λ ∈ spec(C_A)}`):

**Theorem (pairing symmetry).** For every region A (any size, any subset of edges),
$$\operatorname{spec}(C_A) \;=\; \{\, 1-\lambda : \lambda \in \operatorname{spec}(C_A) \,\} \quad\text{as a multiset.}$$

Machine-checked to `< 1e-9` on all 4 three-edge A4-orbit representatives **and** 18 random regions of dimension 2–5 (script §(a)); every single one holds the identity to double-precision round-off (`residual` printed as `0.000e+00` or `~1e-16` throughout).

This is the standard **particle–hole symmetry** of a free-fermion Gaussian covariance built from a real orthogonal complex structure — it is a property of the *state* `C` alone, not of region geometry, and it holds for every region, not just odd or 3-edge ones.

---

## 2. Corollary (odd dimension) — the forced modular zero mode

Group the (real) spectrum of `C_A` into pairs `(λ, 1−λ)` under the Step-3 symmetry. A pair with `λ ≠ 1−λ` (i.e. `λ ≠ ½`) contributes an **even** count (2) to `dim A`; a self-paired eigenvalue `λ = ½` contributes an **odd** count (1, itself — it cannot pair with a distinct partner). Since
$$\dim A = 2\cdot(\text{\# non-self-paired pairs}) + (\text{\# eigenvalues at } \tfrac12),$$
**`dim A` odd forces an odd number (hence ≥ 1) of exact eigenvalues at ½.**

Machine-checked (script §(b)): every odd-dimension region tested (3 and 5 edges, 4 orbits + random draws) carries an exact `λ = ½` eigenvalue to `<1e-6`; on even-dimension regions (2 and 4 edges) the ½-eigenvalue is **not forced** — it appears incidentally on only 3/9 tested even regions, with no structural argument requiring it. This confirms the corollary is genuinely an odd/even-dimension effect, not an artifact of the check tolerance.

By `entanglement_hamiltonian`'s convention (`h_A = log((I−C_A)C_A⁻¹)`, `the_net.py:337`), `λ = ½` gives modular energy `ε = log((1−½)/½) = log 1 = 0` — an exact **modular zero mode**, forced on every odd region, independent of which odd region or which orbit.

---

## 3. Corollary (3-edge regions) — {λ, 1−λ, ½}, one energy magnitude ±ε

For `dim A = 3` the odd corollary forces exactly one eigenvalue at ½ (generically — see the degenerate-case check below) and the remaining two eigenvalues form the single leftover pair `(λ, 1−λ)`. The full occupation spectrum is therefore
$$\{\lambda,\ 1-\lambda,\ \tfrac12\}, \qquad \varepsilon = \log\!\frac{1-\lambda}{\lambda},$$
i.e. exactly **one** modular-energy magnitude `ε`, realized as the pair `{−ε, 0, +ε}` in the sorted single-particle spectrum.

**Machine-checked on all 4 A4-orbit representatives of 3-edge regions** (`_three_edge_region_orbits`, script §(c)):

| orbit | representative | is triangle | orbit size | ζ (occupations) | ε |
|---|---|---|---|---|---|
| 0 | (0,1,2) | no | 4 | {0.0669873, 0.5, 0.9330127} | 2.633916 |
| 1 | (0,1,3) | **yes** | 4 | {0.0669873, 0.5, 0.9330127} | 2.633916 |
| 2 | (0,1,4) | no | 6 | {0.1464466, 0.5, 0.8535534} | 1.762747 |
| 3 | (0,1,5) | no | 6 | {0.1464466, 0.5, 0.8535534} | 1.762747 |

(λ = (2−√3)/4 for orbits 0–1, λ = (2−√2)/4 for orbits 2–3 — closed forms consistent with the printed 10-digit values; not separately re-derived here, out of scope.) All four orbits: exactly one ζ at ½ (residual `<1e-6`), the other two sum to 1 (residual `0`–`2e-16`), exactly one nonzero |ε| magnitude with the two nonzero entries summing to `0` (the ± pairing), and exactly one exact zero mode in ε.

**Degenerate case explicitly checked and found NOT to occur:** the fully-degenerate collapse `spec(C_A) = {½,½,½}` (all three eigenvalues pinned, which the pairing symmetry alone does not forbid) is checked for on all 4 orbits and does not occur — every orbit lands on the generic odd case (one zero mode + one genuine ±ε pair), not the maximally-degenerate one.

---

## 4. Scope statement

**What the lemma says.** The pairing symmetry `spec(C_A) = {1−λ}` (§1) is a **state-structure fact** about the vacuum covariance `C = (I+iJ6)/2` alone — it follows from `J6` being real and antisymmetric, and holds for **every** region A of every size, with **no region-geometry input** (no A4-orbit structure, no triangle/non-triangle distinction, no adjacency). The odd-dimension zero mode (§2) and the 3-edge `{λ,1−λ,½}` single-magnitude structure (§3) are corollaries that use only `dim A`, not which edges A contains — the 4 orbits differ in their **λ value** (a geometric/orbit-dependent number) but not in the **qualitative** {zero mode + one ±ε pair} structure, which is forced for every 3-edge region without exception.

**What it does not say.** It says nothing about *why* a given region has a particular λ (that is orbit/geometry-dependent and outside this lemma), nothing about regions with more than one independent Fock mode of freedom beyond the pair count, and nothing about any physical magnitude — no M_Z, no ppm, no coupling constant appears anywhere in the proof or the check.

**Connection 1 — the W2 obstruction chain (clock-incommensurability).** The single-magnitude-per-odd-region structure (§3) is exactly the fact used, orbit-by-orbit, inside `field_side_flow_generator`'s bit-reversal consistency check (`the_net.py:3486-3510`, which already relies on `Cm|_A = I_A − C_A` as a verified-not-derived identity). This lemma supplies the missing derivation of that identity (§1–2) and confirms it is not an accident of the 4 sampled orbits but a forced consequence of `J6`'s reality/antisymmetry — hardening the base of the region-clock machinery that W2's obstruction argument builds on.

**Connection 2 — L0b's rank-starvation and ML-1d's rank/profile read.** L0b records that "a 3-edge region's flow is rank-starved: one magnitude ±ε plus the forced ½ zero mode" (internal research notes:26`). This lemma is the first-principles reason: the zero mode at 3-edge scale is **structurally forced by odd dimension** (§2), not a coincidence of the specific triangle/non-triangle orbits sampled — so it cannot be designed away by choosing a different 3-edge region. For ML-1d's "rank/profile growth read" (does the positive-energy dimension of `h_A`'s spectrum grow with region size, the TICK-REDUCES discriminator, `ML1d_derived_horizon_prereg_2026-07-12.md:73-75`), this lemma is the reason that **every odd-size region's positive-spectrum dimension count must first subtract the forced zero mode(s)** before comparing to surface size — an odd region of size `2k+1` has, by §2, at least one exact-½ eigenvalue that contributes nothing to modular energy; failing to exclude it would systematically undercount the "boost-like enrichment" signal ML-1d is testing for. (Even-size regions carry no such forced subtraction — §2's corollary is dimension-parity-specific, confirmed empirically absent-by-default on even regions in §2's check.)

**Not claimed:** any statement about even-dimension regions' spectra beyond the general pairing symmetry (§1, which does hold for them too, just without a forced fixed point); any claim about regions larger than 3 edges' *modular energy content* beyond "the pairing symmetry holds" — the "single magnitude" content of §3 is specific to `dim A = 3` (2 non-fixed eigenvalues = exactly 1 pair); a 5-edge region would generically show a forced zero mode (§2) plus **two** independent ε-pairs, not one, and this is not checked or claimed here (out of scope for W3a; a natural extension for the region-ladder work ML-1d is already doing).

---

## Files

- Object (read-only, unmodified): `derivation_topdown/state/the_net.py` — `complex_structure_J6`, `vacuum_covariance`, `region_data`, `entanglement_hamiltonian`, `_three_edge_region_orbits`.
- Check (new, standalone): `proofs/foundations/CA_half_lemma_check_2026-07-12.py` — 31/31 PASS.
- This write-up: `docs/theorems/CA_half_lemma_2026-07-12.md`.
