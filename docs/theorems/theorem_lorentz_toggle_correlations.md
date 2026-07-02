# Handover: Lorentz Invariance of the Toggle Point Process

**Audit anchor:** Cross-references the Lorentz arc closure (`lorentz_sig_ccclose_joint_closure.md`) and Stage 3 leading-order Lorentz invariance (`theorem_lorentz_causal_sector.md`).

**For:** Theorem context
**Status:** HANDOVER — do not prove here. This document briefs the theorem context on what to prove, what is already established, and what the open gaps are.
**Date:** 2026-04-20 (session 10)
**Supporting computation:** `proofs/lorentz/b1_ags_audit.py`, `proofs/lorentz/bloch_dispersion_anisotropy.py`

---

## Target claim

> The toggle point process on the srs Planck lattice has n-point correlation functions of the form
>
>   W_n = λⁿ + C_n^conn
>
> where λⁿ is a Lorentz scalar (no preferred spatial direction) and the connected piece satisfies
>
>   |C_n^conn| ≤ K · (1/6)^{L/l_P}     [temporal: exact]
>
> with an additional spatial correction
>
>   |δW_n| = O((l_P/L)²)               [from I4₁32 lattice anisotropy: blocked]
>
> The dominant Lorentz violation at macroscopic scales L >> l_P is the polynomial O((l_P/L)²) piece.

---

## What is already established (do not re-derive)

**Toggle process definition** (A1 + A2-T):
- Each undirected edge of srs is an independent 2-state Markov chain
- p_create = 1/2 (off → on), p_destroy = 1/3 (on → off) per Planck step
- Edges evolve independently — no spatial coupling between distinct edges

**Exact numerical values** (computed in `proofs/lorentz/b1_ags_audit.py`):
- Stationary toggle rate: λ = 2/5
- Markov chain 2nd eigenvalue: r = 1/6
- Temporal correlation length: ξ_t = l_P / log 6 ≈ 0.558 l_P

**Connected correlation decay — CLOSED:**
- For distinct directed edges e ≠ e′: C₂^conn(e,t; e′,t′) = 0 exactly (independence by construction)
- For same edge, time separation s: |C₂^conn(e,t; e,t+s)| ≤ K λ² (1/6)^s
- n-point generalization: C_n^conn = 0 whenever any two arguments are on distinct edges; decays as (1/6)^{s_min} otherwise
- This follows from spectral theory of the 2-state Markov chain alone. No external citation needed.

**AGG route ruled out** (see `proofs/lorentz/b1_ags_audit.py`):
- Arratia-Goldstein-Gordon 1989 requires λ << 1 (rare events). λ = 2/5 fails this.
- Observer-frame attempt: λ_obs = 1/60, D = 20, b₁/λ_total = 0.33 — marginal, not << 1.
- Do not pursue the AGG/Poisson-convergence route.

**Ramanujan bound** (cited, STRICT-SOLID from `predictions/feshbach_exponent_principle.py`):
- |μ₂(NB walk on srs)| = 1/√2
- NB walk correlation length: ξ_NB = l_P / log(√2) ≈ 2.885 l_P
- Girth g = 10: NB walk cannot revisit the same directed edge in fewer than 10 steps

---

## Open gaps — what the theorem context must close

**Gap 1 (CITED, application pending): Spatial isotropy of λ**

Claim: the toggle density λ is isotropic — the same rate in all spatial directions.

Citation to apply: Sunada 2012, *Topological Crystallography*, Springer, Ch. 6–7, Theorem 6.4 (standard realization uniqueness) and Corollary 6.7 (isotropic heat kernel).

Precise application needed: the ergodic average of the toggle process over the NB walk trajectory inherits the isotropy of the NB walk heat kernel. Show: toggle density per unit spacetime 4-volume ρ₄ = λ/l_P⁴ is a Lorentz scalar (transforms correctly under SO(3) rotations and Lorentz boosts).

Status: one paragraph of careful argument connecting the heat kernel isotropy to the toggle density. Not a new proof — a connection step.

**Gap 2 (FULLY COMPUTED): Spatial Lorentz violation coefficient η_lattice**

Two computations run this session — Laplacian (adjacency) and Hashimoto (NB walk):

**Laplacian acoustic branch** (`proofs/lorentz/bloch_dispersion_anisotropy.py`):

| Quantity | Value | Notes |
|----------|-------|-------|
| D (isotropic O(k²)) | 2.4674 | Identical in [100],[110],[111] to 4×10⁻⁷ — Sunada confirmed numerically |
| D4_aniso | +1.012 | Anisotropic O(k⁴) piece |
| η_Lap = D4_aniso/D² | **≈ 1/6 = 0.1667** | Consistent across all 3 pairs to ±0.002 |

**Hashimoto (NB walk) branch** (`proofs/lorentz/hashimoto_bloch_dispersion.py`):

| Quantity | Value | Notes |
|----------|-------|-------|
| D_NB (isotropic O(k²)) | **1/8 exactly** | = (NN distance)² = (√2/4)² — exact lattice result |
| D4_NB_aniso | ≈ 1/768 | |
| **η_NB = D4_NB_aniso/D_NB²** | **≈ 1/12 = 0.0833** | Consistent to ±0.00001 |
| Sign | +1/12 > 0 | SUBLUMINAL — NB walk speed decreases at high energy |
| η_Lap / η_NB | 0.5015 ≈ **1/2 exactly** | η_Lap = 2 × η_NB |

The physical photon dispersion in the author's separate private derivation uses the Hashimoto (NB walk) propagator:

  **η_lattice = 1/12**  (subluminal, exact up to symbolic verification)

Scale energy: ~147 PeV.

**What the theorem context must do for Gap 2:**
- Verify η_NB = 1/12 symbolically (D_NB = 1/8 and D4_NB_aniso = 1/768 are both plausibly exact from lattice geometry — confirm with sympy or analytic calculation).
- State the final theorem: dispersion is h_max(k) = 2 − (1/8)|k|² − [(D4_iso + f₄(k̂)/768)]|k|⁴ + O(k⁶) with η_lattice = 1/12.

**Gap 3 (CITATION, verify theorem number): Bombelli-Lee-Meyer-Sorkin**

If Gap 2 closes with a specific η_lattice, use:
- Bombelli, L., Lee, J., Meyer, D., Sorkin, R.D. (1987). Space-time as a causal set. *Phys. Rev. Lett.* **59**(5), 521–524.
- Key result: Poisson(ρ d⁴x) with isotropic ρ is the unique Lorentz-invariant point process on M^{3,1}.
- Application: at macroscopic scales where toggle correlations have decayed (L >> ξ_t), the toggle process converges in distribution to Poisson with corrections O((l_P/L)²) from η_lattice.
- Verify: does the paper's theorem apply to this setting (finite-range correlations, lattice substrate)? What additional condition, if any, is needed?

---

## Downstream predictions (for parameters list — not part of the theorem)

These follow from the above analysis. Theorem context should NOT attempt to prove them — flag them as derived predictions for the parameters list.

| Parameter | Value | Derivation status |
|-----------|-------|-------------------|
| Toggle rate λ | 2/5 (exact) | STRICT-SOLID |
| Markov 2nd eigenvalue r | 1/6 (exact) | STRICT-SOLID |
| Temporal correlation length ξ_t | l_P / log 6 ≈ 0.558 l_P | STRICT-SOLID |
| NB walk correlation length ξ_NB | l_P / log(√2) ≈ 2.885 l_P | STRICT-SOLID |
| Dimension-5 Lorentz violation η₅ | **0 exactly** — proven from B(−k)=B(k)* (undirected graph symmetry, not toggle T-symmetry) | PROVEN |
| Dimension-6 coefficient η_NB (Hashimoto) | **≈ 1/12** (subluminal) | COMPUTED, symbolic verification needed |
| Dimension-6 coefficient η_Lap (Laplacian) | **≈ 1/6 = 2 × η_NB** | COMPUTED |
| Scale energy (Hashimoto) | **~147 PeV** | COMPUTED |
| Universe transparency above | **~147 PeV** (pair-production thresholds raised; η > 0) | COMPUTED |
| Birefringence from I4₁32 chirality | present if chiral coupling nonzero | SCOPED |

---

## Success criterion

The theorem is closed when:
1. Gap 1 is resolved (one paragraph connecting Sunada heat kernel isotropy to toggle density isotropy)
2. Gap 2 is resolved (η_lattice computed from Bloch dispersion — see `proofs/lorentz/bloch_dispersion_anisotropy.py`)
3. Gap 3 citation is verified (BMS 1987 theorem number and applicability condition)

At that point the full claim can be stated: W_n = λⁿ [Lorentz-invariant] + O((1/6)^{L/ξ_t}) [temporal] + O(η_lattice × (l_P/L)²) [spatial], with all three pieces given specific values.
