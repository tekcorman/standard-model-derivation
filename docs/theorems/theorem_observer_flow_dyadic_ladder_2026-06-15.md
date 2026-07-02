# Theorem (structural): the observer-side flow is a dyadic ladder on the Hubble spine

**Date:** 2026-06-15
**Grade.** Mixed, stated per-clause. (S1) the spine is theorem-grade (upstream). (S2) the
dyadic monomial structure is a solid re-description of already-derived relations, with the
honest caveat that the dyadic-ness of the purely-cosmological rungs is generic FRW physics.
(S3) the observer read-map and the −1/4 floor are measured on the count-walk model; the
"no observable below −1/4 from a single read" is a falsifiable structural prediction.
(C) one scale input N_hub is irreducible; the sub-spine rungs are era-dependent.

**Probes (all promoted + gated, exit 0).**
`proofs/foundations/vev_exponent_observer_recurrence_2026-06-14.py` (the read-map and −1/4);
`proofs/foundations/observer_flow_dyadic_ladder_2026-06-15.py` (clause S2, the dyadic ladder —
deterministic structural gates LG1–LG4);
`proofs/foundations/vev_read_floor_single_halving_2026-06-15.py` (clause S3, the single-read
floor — gates G1–G4); `proofs/foundations/era_handoff_thermal_rung_reconcile_2026-06-15.py`
(clause C, the thermal-rung era-handoff — gates EG1–EG4). Exploratory originals in
`proofs/_scratch/flow_*_2026-06-15.py`, `era_handoff_*_2026-06-15.py`.
**Companions.** `project_native_rg_equivalent_resolution_flow_2026-06-14` (the cascade =
power-law resolution flow, no β-function); `project_vev_observer_read_decomposition_2026-06-14`
(the VEV = one observer-read); `predictions/v_higgs_derivation.md` (Step 4, criticality-free).

---

## Statement

On the static substrate the flow bet is **void** (the srs Dirac is gapped, |E|_min = 0.59, so
there is no scale-invariant regime and no running). The **dynamics is observer-side**: the
N-flow ∂_N along the event-count direction τ = log N. This flow is **not a coupling-flow**
(no non-trivial β-function — the static substrate forbids feedback); it is a **pure power-law
resolution flow**, `d ln X / d ln N = p_X` with constant exponents. The theorem characterizes
its structure:

> **The observer-side flow is a single-generator dyadic ladder.** Every dimensional cascade
> observable X is a monomial in one spine, the Hubble rate **H ~ N^{−1}**:
> $$X \;=\; M_{\rm Pl}^{\,a_X}\; H^{\,q_X}, \qquad q_X \in \{\,\pm 2^{j} : j \in \mathbb{Z}\,\}\;\;(\text{dyadic}).$$
> The exponents are dyadic because **every observable↔spine relation is a square, a
> square-root, or a reciprocal** of a derived physical law. The **bottom physical rung is the
> observer's order-parameter read**, the Higgs VEV **v ~ N^{−1/4}**, which is the **floor of
> the single-read sector**: a single observer-read produces exactly one halving below the
> counting law, and no single-read observable scales below −1/4.

The generator is the dilation `N d/dN = d/dτ`, whose eigenvalue on X ~ N^{p} is p; the
content of the theorem is that the {p_X} are **one spine raised to dyadic powers**, not a set
of independent exponents.

---

## (S1) The spine — theorem-grade (upstream)

`H · N · t_P = 1` with coefficient **exactly 1** (cascade D1+D2+D3; de Sitter vs power-law
fixed by srs being a Ramanujan expander). Hence **H ~ N^{−1}** is the single scaling spine,
and there is an N-flow at all. *This is the first framework-specific input — why a flow
exists.* Eigenvalue of the dilation on H is −1.

## (S2) The dyadic monomial structure — solid re-description

Each cascade observable is H raised to a dyadic power, by a **derived** relation:

| observable | relation (derived, in-repo) | spine power q | N-exponent p = −q | type |
|---|---|---:|---:|---|
| Λ | Λ = 3H² (Friedmann, vacuum) | 2 | −2 | square |
| H | the spine (S1) | 1 | −1 | — |
| t (age) | t = 1/H | −1 | +1 | reciprocal |
| T_rad | H ~ T²/M_Pl (radiation Friedmann) | 1/2 | −1/2 | √ (era-limited) |
| m_ν | m_ν ~ (y v)²/M_R (seesaw) | 1/2 | −1/2 | square in v |
| G_F | G_F = 1/(√2 v²) | −1/2 | +1/2 | reciprocal-square |
| **v** | v ~ (H M_Pl³)^{1/4} (observer read, S3) | **1/4** | **−1/4** | **observer halving** |

The powers {2, 1, 1/2, 1/4} are **consecutive powers of 2**. **Why dyadic:** Friedmann
H² ∝ ρ is quadratic, the seesaw is quadratic, G_F ∝ v⁻² is reciprocal-quadratic — squares and
square-roots halve and double exponents, so a ladder built from them is dyadic.

**Honest caveat.** The dyadic-ness of the purely-cosmological rungs {Λ, H, t, T} is **generic
FRW physics**, true of any Friedmann cosmology — *not* a framework discovery. It must not be
sold as "the framework predicts powers of 2." The framework-specific content is exactly two
items: the spine (S1) and the v rung (S3).

## (S3) The observer read-map and the −1/4 floor — measured on the count-walk model

The observer reads the M-edge substrate by a graph-blind count-walk (the lean = #on-edges;
`real_multiway_lean`: P(up|k) = (M−k)/M). The relevant lemma, **measured**:

> **Read-map lemma.** A single one-pass read of a length-L scalar order parameter has
> effective sample size `N_eff = √L` (the diffusive local time / one-pass recurrence of the
> 1-D lean). Measured exponent 0.51 across L = 64…4096
> (`vev_read_floor_single_halving`, gate G1); 0.46–0.50 in the recurrence probe.

Consequences (the counting law `spread = N_eff^{−1/2}` is the CLT, fixed):
- **Full read** (N_eff = M): spread ~ M^{−1/2} — the naive counting exponent.
- **One-pass read** (N_eff = √M): spread ~ (√M)^{−1/2} = **M^{−1/4}** — the VEV exponent.

So **v ~ N^{−1/4} is one observer halving below the −1/2 counting law**, with no criticality
invoked (this dissolves the circular "MDL forces μ²=0" Step 4 of `v_higgs_derivation.md`).

**The floor.** A deeper rung at −1/8 would require `N_eff = M^{1/4}`, i.e. a *second*
recurrence = the recurrence-of-the-recurrence = a **read-of-a-read** (a second observer reading
the first's return-process). That is a depth-2 *meta-observation*, not the framework's "one
observer-read of one walk." Hence:

> **Floor corollary.** A single observer-read yields **exactly one halving**; **−1/4 is the
> floor of the single-read sector**. *Falsifiable structural prediction:* no physical observable
> scales as N^{−1/8} or below from a single read. (Repo check: nothing populates sub-(−1/4);
> v = −1/4 is the smallest-magnitude negative rung.)

**Honest limit on (S3).** The read-map L→√L (one halving) and the −1/2, −1/4 rungs are cleanly
measured. The deeper dyadic tower (−1/8, −1/16, … by iterating the √-map) is an **analytic
consequence** of the verified read-map but is **not numerically demonstrated**: the second
nesting acts on length-√M ≤ 256 sequences, which lack the dynamic range for a clean second
square-root (depth-2 measured −0.075, a finite-size artifact, not −1/8). The physical claim —
single read ⇒ one halving ⇒ −1/4 floor — does not depend on reaching the deeper rungs.

---

## The keystone: v bridges particle and cosmology

The cosmological sub-spine ladder bottoms at −1/2 (T, m_ν). The observer's single read supplies
**one further halving**, the −1/4 rung, which is the *only* exponent not handed over by quadratic
cosmology. That rung **bolts the particle VEV onto the cosmological spine**:
$$v \sim (H\,M_{\rm Pl}^3)^{1/4} \quad\Longleftrightarrow\quad 1/H_0 \sim N^{+1},$$
so the lab observable v (no cosmology in it) and the cosmic clock 1/H₀ fix the **same** scale
N — agreeing to **1.06×**. This is the framework's derivation of the Higgs–Hubble coincidence
v ∝ (H₀ M_Pl³)^{1/4}, and it is the **one genuine over-determination** of the single scale.

## (C) Conditional / open

- **N_hub** — the absolute scale (the universe's age ≈ 2²⁰² Planck ticks) — is the framework's
  **one irreducible dimensional input**, not derivable from the substrate (a cosmological
  Cauchy datum, like "what time is it"); the v↔1/H₀ over-determination confirms there is
  exactly one such freedom. Deriving it from structure is a category error.
- **Era-stratification (reconciled).** The sub-spine thermal rung T ~ N^{−1/2} holds in the
  *radiation* era (a ∝ t^{1/2} ∝ N^{1/2}); in the matter era a ∝ N^{2/3} so T ~ N^{−2/3}, and in
  the Λ era steeper still — **one rung per era**, slope set by a(t). The ladder below the spine is
  therefore era-stratified, not a single eternal spectrum. The known ~50× (in N) thermal "outlier"
  — inverting the radiation rung with *today's* Λ-era CMB T — is **exactly the radiation→matter→Λ
  handoff**, fully accounted (`proofs/foundations/era_handoff_thermal_rung_reconcile_2026-06-15.py`,
  gates EG1–EG4): the rung holds BBN→matter-radiation equality (EG1); today it overshoots 6.5× in
  T = 42× in N (EG2); that overshoot equals the actual scale-factor growth (1+z_eq = 3403) divided
  by the radiation N^{1/2} extrapolation (520) (EG3); and read at a *radiation-era* anchor the rung
  recovers the right N (EG4). So the thermal row is not a framework miss — it is the spine
  (H·N·t_P = 1) plus T ∝ 1/a evaluated across the standard era transitions. *This is a consistency
  reconciliation; it imports the standard era structure (z_eq, the per-era a(t) laws).*

---

## One line

The observer-side flow ∂_N is one dilation generator acting as **dyadic monomials on the
Hubble spine H ~ N^{−1}**; every cascade observable is H to a power of 2, set by the quadratic
physical laws, and the Higgs VEV **v ~ N^{−1/4}** is the **single-read floor** — exactly one
observer halving below the counting law — which bolts the particle sector onto the cosmological
spine over a single contingent scale N_hub.
