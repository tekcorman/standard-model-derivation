# Incomplete-equations to-do — the top-down completeness audit

## The law (why this file exists)
Every quantity in this framework is a **forced read of the complete spectrum of the one object**
(`D = B(srs ⊗ srs-z) ⊗ ∂_N`). Masses, couplings, mixings are **not targets to fit** — they *are* the spectrum.

**If a value does not fall out of the spectrum top-down, the DEFINING EQUATION OF THE OBJECT is incomplete
at that point.** The task is *never* to fit the residual or pattern-match a constant to a target (that is
numerology, and it is forbidden). The task is to (1) **locate the incompleteness**, (2) **add it here**, and
(3) **chase the math up to the complete top-level equation** — exactly as every closed result in this repo
was obtained. Think in **spectra and forced objects**, never in "mass" or "target parameter."

The question for any quantity is: *what is the equation whose solution it is, and is that equation complete?*

## Open items (where the equation is not yet complete top-down)

### 1. The charged-lepton mass spectrum (the per-rep / "70 ppm" residual) — LOCALIZED 2026-06-30 to ∂_N
The spectral read was traced top-down (`the_run.py:read_masses`). The result: **the residual cannot come from
the current object's static spectrum; it is forced to live in ∂_N (the run operator), which is not built.**
The chase, with every dead-end ruled out by computation (so they are never re-walked):

- **Baseline confirmed:** `read_masses` = circulant √m_j = |c₀ + c₁ωʲ + c₁* ω⁻ʲ| with moduli frozen at the Γ
  Perron/shell values {2,√2,√2} (= Born (4,2,2)) and phase δ from `read_phases`. Reproduces **m_e/m_τ −70.3 ppm,
  m_μ/m_τ −60.5 ppm** exactly. So the −70 ppm IS the truncation of this read.
- **RULED OUT — "move the moduli with the run":** using the actual run-position eigenvalues gives garbage
  (+10⁹ ppm) and breaks J-reality (the run-position eigenvalues carry the intrinsic ~110° shell phase, which is
  NOT the generation phase δ). The modulus/phase separation is essential; the residual is not a modulus-move.
- **RULED OUT — "dress each winding by the framework factor (1−α₁/h)":** overshoots by ~10⁴× (gives −10⁵ ppm).
  That factor is **α₁-level (few-%) and species-common** — it cancels in within-species ratios exactly as
  `m_e.py` states. The residual is **α₁³-scale (59 ppm)**, a different order.
- **KEY STRUCTURAL FACT:** at Γ the four modes of each C₃ isotype are **orthogonal eigenvectors — decoupled**
  ({dominant, shell(s), ±1}; verified). The read keeps only the dominant pole. Because the modes are decoupled,
  the dominant-pole read is **EXACT at Γ** — there is no static self-energy to add. ⟹ the α₁³ residual is **not
  a static-spectrum effect at all.**
- **LOCALIZED:** the only way the dropped subdominant modes re-enter the kept dominant amplitude is through the
  **run coupling them** — ⟨subdominant|dB/ds|dominant⟩. Computed: this is **nonzero and winding-asymmetric**
  (isotype-1 mixing 2.21 vs isotype-2 1.43 — the e/μ asymmetry's structural home). So **the 70 ppm is the
  ∂_N (run) non-adiabatic coupling of the dropped subdominant modes into the dominant pole**, winding-resolved.
- **The open equation, stated exactly:** complete the read = build ∂_N as the concrete run operator carrying
  this inter-mode coupling, then m_j = full spectrum of the dominant pole **dressed by the ∂_N-mixed subdominant
  modes**. The 70 ppm (magnitude AND e/μ chirality) must then *fall out*, forced — or the incompleteness moves
  up. **Same frontier as `build_D4` / the ∂_N-completion thread; the 70 ppm is now its sharpest concrete probe.**
- Status: **LOCATED, not yet derived.** No fit. The number is a forced consequence of an un-built operator,
  not a free parameter and not a mystery.
- **MASSIVE-MODE / CASCADE ROUTE RULED OUT (2026-06-30):** the ∂_N *massive modes* are the **cosmic cascade**
  (dyadic ladder on H~N^{−1}: Λ,m_ν,v…), all **dimensional** and N-dependent. The −70 ppm is a **dimensionless
  ratio** → on the **N-independent disconnected axis** (the cascade theorem flows only dimensional rungs;
  N_hub.py:31-32). Orthogonal — the cascade cannot reach the generation ratio. ⟹ fourth independent ruling: the
  −70 ppm is the structurally-isolated **C₃-screw run-Dirac subleading** (not scale, not joint cover, not Higgs,
  not cascade).
- **DEGENERATE-PT RULED OUT (2026-06-30):** computed the shell-doublet degenerate PT — over-applies to **75,000+
  ppm** (1st-order rate = π). Joins the others.
- **EXHAUSTIVE VERDICT (2026-06-30):** the −70 ppm is the **O(α₁³)=(2/3)²⁴ girth-window SURVIVAL/Dyson diagram**
  — a DARK object, NOT a spectral/run object. Every run/spectral technique over-applies because the run is
  O(1)-scale and the survival is 24 powers of (2/3) below it; they are different scales, not different
  approximations. **NINE routes now ruled out, each with a reason:** transport (×3), band/modulus curvature,
  resolvent/cycle, joint cover_B, enantiomer twist, scale/N_hub, cosmic cascade, degenerate-PT. **Grade:
  conjecture-grade — the 1/μ_rep MDL water-filling ceiling, the SAME ceiling as Q=2/3 and c_F.** Open miss,
  mechanism IDENTIFIED (the α₁³ diagram), encoding at the grade-ceiling. The only lift is the continuum-D₄
  spectral action (research-level, unbuilt, gates no value). **This is the honest floor of the operator-route
  search — NOT a relabel: the miss stays OPEN; what's exhausted is the spectral/run *route* to it.**
- **CONTINUUM-D₄ CONE ROUTE EXPLORED (2026-06-30, next session) — does NOT force the allocation; MDL ceiling
  STANDS** (`proofs/foundations/lepton_70ppm_continuum_D4_cone_2026-06-30.py`;
  `research_frontier_dN_alpha1cubed` probe 3). The decisive structural finding (resolves probe 2's sign failure):
  the framework's dark is **MULTIPLICATIVE** (resolvent Σ=α₁/h, the m_b/m_t object) → per-isotype `2α₁³Re(h)/μ` →
  shells NEGATIVE (Re h=−½), wrong sign. The **ADDITIVE** spectral-action structure (`a₂=Tr D²`; `c²→μ+α₁³`,
  α₁³ isotype-blind scalar) → `+α₁³/(2μ)` → the **+1/μ_rep allocation with CORRECT sign falls out**. BUT adopting
  additive for leptons *because* it gives the right sign would be a FIT and **contradicts the working
  multiplicative heavy-quark dark** (which uses the real Perron h=2). ⟹ the continuum cone does **not**
  operator-force the 1/μ_rep; **MDL ceiling stands; −70 ppm OPEN.** Frontier SHARPENED to one question: *why would
  the lepton generation-allocation dark be additive when the heavy-quark single-channel dark is multiplicative?*
  **All known routes (self-energy/resolvent/transport/band/curvature/cover/enantiomer/scale/cascade/degenerate-PT/
  Berry/continuum-D₄) now explored — the −70 ppm is at the operator-route floor, conjecture-grade MDL.**
- **BERRY-HOLONOMY ROUTE RULED OUT (2026-06-30, next session) — the 10th route** (`build_dN` Step 5,
  `proofs/foundations/lepton_70ppm_berry_holonomy_2026-06-30.py`): the closed-loop Berry holonomy of
  D₄=∂_s+B(s·AXIS) was the doc's last *untried* spectral/geometric probe. **Built it. Falsified.** Genuine operator
  period along the screw is **√3** (B(√3·AXIS)=B(0) to 1e-16), not √7 (that's the eigenvalue-phase period — doc
  conflated them); the true closed loop is s∈[0,√3]. Abelian closed-loop Berry phase per winding = EXACTLY
  **{−π,0,0}** — purely **topological** (Z₂; Perron winding flips sign, shells get 0), carries NO continuous
  ~60 ppm. Open-path geometric phase to s_lep = O(0.1–1 rad) ≈ 1e4–1e5 ppm (~1e4× over); non-abelian holonomy
  collapses (det W→0) at the Perron→shell crossings. **The closed loop DOES cancel the over-application — but to
  exactly 0/−π (topological), NOT ~60 ppm.** ⟹ the −70 ppm is **NOT a band-geometric/Berry effect**; the
  spectral/run/geometric route is now FULLY exhausted. **Only the continuum-D₄ Dirac-cone spectral action remains**
  (research-level, unbuilt). **Miss stays OPEN; the Berry ROUTE is ruled out.** (Byproduct: the Perron-winding Z₂
  holonomy −π is clean new screw-loop topology.)
- **∂_N BUILD ATTEMPTED 2026-06-30 (an internal working note):** ∂_N's *leading* operator is now FORCED —
  φ=2π/√7 = d(arg h)/ds|₀ and the Ihara–Bass-pinned moduli fall out of B(s·AXIS), no insertion. But the −70 ppm
  is **provably not a first-order spectral read**: six forced constructions (non-adiabatic transport ×3, band
  curvature, modulus curvature, resolvent trace, dressed/resolvent dominant) **all over-apply at O(α₁)~10⁴ ppm
  or give wrong moduli**, because the run's true s-dependence is violent (Perron→shell, mode crossings) and the
  leading read correctly freezes it. **The residual is O(α₁³) — a 2-loop Dyson diagram, not a spectral
  correction.** ⟹ refined incompleteness: *no operator-forced derivation of the α₁³ winding-resolved Dyson
  diagram exists*; W1's `2α₁³/μ_rep` matches the magnitude at **conjecture-grade** (forced pieces: 16-bubble,
  first-girth-return, Λ•(ℂ³)=(4,2,2); structural piece: the 1/μ_rep encoding — same grade-ceiling as the α₁²
  `c_F`). Remaining routes: closed-loop Berry holonomy over s∈[0,√7]; full continuum D₄ cone. Both research-level.
- **JOINT-OBJECT DIAGNOSTIC (2026-06-30) — wrong-object hypothesis RULED OUT (`build_dN` Step 4):** the live
  mass read uses single-srs B, but the framework's `cover_B` (srs⊗srs-z) = B⊗σ_x gives **identical** moduli
  (4,2,2) — we ARE reading the right object. A principled enantiomer twist differs at the run but **breaks
  J-reality** (complex masses) — not valid; tuning twists = fitting, not done. ⟹ the −70 ppm is a
  **clean-extraction wall on the CORRECT object**: leading masses forced/exact, the subleading is a **genuine
  OPEN miss whose mechanism is UNIDENTIFIED** (self-energy, transport, band/modulus curvature, resolvent, cycle,
  joint cover, enantiomer — ALL ruled out with numbers). NOT an artifact, NOT grade-only. **Scale/Higgs-side
  search (N_hub→v) is the next untried place — see item 2.**

### 2. The Higgs/scale sector vs the electron mass — SEARCHED 2026-06-30, scale route RULED OUT
Hypothesis (user): a small Higgs-side correction, shared via N_hub→v, brings in the electron mass and makes
N_hub consistent. **Searched (v_higgs.py, m_H.py, lambda_higgs.py, m_e.py); result: DISFAVORED for the dominant
−70 ppm, with numbers.**
- Higgs sector is fully dark-corrected and **closes**: m_H = √(2λ)·v, λ carries its own Family-D (−4α₁², from 4
  Higgs legs) → m_H **−0.05σ**. The one asymmetry: the v_higgs Family-D (−α₁²) is **absorbed into N_hub** (not
  applied) via the G_F round-trip — but that touches the scale/cosmology.
- **The framework's own `m_e.py` decomposes the residual:** −70 ppm **Koide RATIO** (scale-independent, ~84%) +
  −13 ppm **m_τ absolute scale** (Higgs/N_hub-touchable). **VERIFIED:** the ratio (f_min/f_max)² is *invariant*
  under a +1000 ppm v/N_hub shift ⟹ a Higgs/v/N_hub correction **CANCELS in it and cannot supply the −70 ppm**.
- ⟹ the −70 ppm is the per-rep **δ-gap** (δ for m_e/m_τ-exact = 0.2222208 vs 2/9), scale-independent,
  lepton-Yukawa-side — confirming item 1. The Higgs/scale route is **ruled out** alongside the joint-object route.
- The ONLY Higgs/scale-touchable part is the **−13 ppm m_τ absolute scale** (16%), the v←N_hub←G_F circular
  calibration (Gap G1) — small, separate from the −70 ppm, and mostly y_τ.
- **Doc-lag corrected:** the memory line "m_e 70 ppm = the N_hub over-determination residual" was WRONG (it's the
  scale-independent ratio; the over-determination is the −13 ppm). Fixed in
  `memory/N_hub-overconstrained-higgs-vs-electron-2026-06-29`.

### 3. N_hub calibration omits a derived correction → H_0 is flattered (over-determination, 2026-06-30)
The N_hub value is pinned by the G_F round-trip using **only the Class-C (5/12) dark correction on v**.
`N_hub.py` (lines 120-129) explicitly instructs: *"if a higher-order Feshbach analog on v is later derived
(above and beyond 5/12), N_hub should be recomputed."* The **Family-D (−α₁²) on v IS that analog** — derived
theorem-grade in `v_higgs.py` — and N_hub was **never recomputed** with it (it is "absorbed").
- **Consistent treatment (apply the derived Family-D):** N_hub −0.61% (×(1−α₁²)^V) ⟹ H_0 ∝ 1/N_hub shifts +0.61%.
  - CMB/substrate side: **68.18 (+1.56σ) → 68.60 (+2.39σ)** vs Planck 67.4 — **WORSE.**
  - observer side: 72.72 (−0.30σ) → 73.17 (+0.12σ) vs SH0ES 73.04 — slightly better; but Planck dominates.
- **Finding (AM):** thought the reported H_0 +1.56σ was flattered by omitting a derived Family-D (consistent
  value +2.39σ). **CORRECTED PM by the full audit (an internal working note):** the Family-D
  **does not belong on v** (condensate ≠ legged scattering vertex; not actually applied; its "absorption check"
  is a by-design tautology). ⟹ omitting it is **correct**, **H_0 = +1.56σ stands** (the +2.39σ assumed a
  correction that shouldn't be applied). The real error is the *opposite*: `v_higgs.py`/`N_hub.py` **over-claim a
  v Family-D that shouldn't exist.** (Genuine dispute: the framework asserts a v "1H+0F vertex"; category
  grounds say no → +1.56σ.)
- **BIGGER finding (the audit's bottom line): v/N_hub is an ASSEMBLED, CALIBRATED form — NOT a forced top-down
  read.** v = v_obs to 0σ **by construction** (N_hub inverted from G_F ≡ from v_obs). Effectively ONE free input
  (N_hub, Gap-G1, band-B) + M_P unit + ~4–5 non-forced modeling choices. The "UNIQUE/THEOREM-GRADE/forced"
  labels overclaim; v's 0σ is a calibration artifact, not a prediction. Defensible core: 1/√2 overlap, −1/4
  finite-size exponent, 5/12 count, one adopted N_hub. **Corrections owed are listed in the honest-grade doc.**
- Does NOT touch the −70 ppm electron ratio (scale-independent, item 2).

### 4. The dark-correction SIGN — DOWN is derived (rate framing); the standalone formal lemma is the open piece (2026-07-01)
**Status: sign is DERIVED DOWN (foundation-conditional); the CAS-closeable lemma is the open equation.** The dark
self-energy Σ=α₁/h is magnitude-forced (Re/−Im to 1e-12) and the sign is **DOWN** via the framework's foundational,
user-confirmed **mass = dynamical recurrence RATE** definition: the rate/velocity reading (cycle-takers waste steps
→ delayed) gives mass×(1−α₁/h) = DOWN, reproducing the framework's value (`theorem_dark_self_energy_unified §3`).
- **Already forced:** the vertex-dark sign (y_τ, c_F) is rigorously DOWN (Peskin −1, a separate closed loop, §2.5).
- **The open equation:** a *standalone* CAS-checkable lemma that "mass=recurrence-rate ⇒ reading (2) [rate→DOWN]
  over (1) [amplitude→no-change] and (3) [return-amplitude→UP]" does NOT yet close (`§3 correction 2026-06-29`).
  The mass=rate *definition* selects DOWN; a from-nothing formal derivation of that selection is un-built.
- ⚠ The sign itself is NOT undetermined — it is DOWN. Do not relabel this as "empirical/open sign"; what is owed
  is only the formalization of a settled result.

### 5. The gauge-β FORMULA (ζ_{D₄}(0)) — the β VALUES fall out; the FORMULA's native origin is the open piece (2026-07-01)
**Status: β VALUES DERIVED (all three, verified); the β FORMULA is the open equation.** `read_gauge_running`
(`derivation_topdown/bridge/the_run.py`) computes the 2HDM β natively (Dynkin sums off the Cl(6)-Fock Hamming-weight
content) and adds the COMPUTED 4D time-completion `(1/3)T_f+(2/3)T_H+(2/3)C₂(G)`; **b₁,b₂,b₃ ALL reproduce
{33/5, 1, −3} exactly** (verified by running — the MSSM-lit values are now a comparison-only cross-check; the
hardcoded "target" was removed 2026-07-01). The +4 completion is DERIVED, not injected.
- **The open equation:** the one-loop β FORMULA itself (the −11/3, ⅔, ⅓ Dynkin structure) is still standard-QFT
  typed (Layer-2). Its native top-down form is **ζ_{D₄}(0)** — the spectral zeta of D₄ = B⊗∂_N (the 4D Dirac-cone
  completion, KO-dim 2→6). RESEARCH-LEVEL (continuum Dirac-cone + KO-6 doubling; lattice heat-kernel = dead end).
  GATES NO VALUE (the β values are in hand) — it is the GRADE/derivation frontier, the "sharpest open extension."
- **Within-repo caution to resolve:** `O_native_beta_eliminate_mssm_adoption:32` argues only the +2 scalar-half has
  a clean substrate home, so "the full +4 is FORCED (not merely reproduces the values)" is not yet unanimous —
  closing ζ_{D₄}(0) is what settles whether the completion is forced.

### 6. The substrate-selection discriminator — srs is DOMINANT among waterline survivors, not uniquely forced (ruled 2026-07-01)
**Status: the selection equation is incomplete top-down.** Per the R-9 SUPERSESSION
(`docs/audits/registers/structural_residue_register.md`, 2026-06-15, probe-backed ×4) and the 2026-07-01
ruling accepting it: the operative substrate closure is the RCSR structural-fingerprint study — srs is
**DOMINANT in an MDL-waterline superposition of survivors {srs, srs-c8, lou, lov}**, discriminated by PDG
observables. The "(A) → arc-transitive → Sunada → srs unique" chain is retained as provenance only
(`arc_transitivity_ground_truth.py`: srs-z IS arc-transitive, so arc-transitivity does not hard-gate it;
strong isotropy is a true Sunada-certified property of srs but was shown 4 independent ways not to carry
the selection load).
- **The open equation, stated exactly:** what structural functional of the one object, computable WITHOUT
  data, separates srs from {srs-c8, lou, lov}? The register's srs-z study shows all 14 prediction
  differences trace to the single fact of cell-doubling (extensive quantities; the intensive spectrum is
  bit-identical) — so the candidate discriminator is extensive/topological (cell size, cover degree), and
  its MDL cost must be *forced*, not asserted. Until it is derived, the substrate selection step consumes
  data, and the claim is scoped accordingly.
- **Honest scoping (not a weakening of the physics):** a survivor superposition is the NATIVE A2 reading —
  the waterline retains every representation that saves bits; observation discriminates. The same
  selective-retention logic used for chirality and triality applies to the substrate itself. But per the
  top-down law the *discrimination* must eventually fall out of the object, not the data — that is this
  open equation.
- **No predicted value changes** (all live values are srs reads). This is the claim-level honesty of the
  selection step; front-door language updated accordingly (A1 pass, 2026-07-01).

## ✅ RESOLVED 2026-06-30 — M_Z via the BZ-integrated Z-current vacuum polarization: 0.810 does NOT fall out; M_Z is a forced oblique residual
**The planned attack was carried out. The `0.810` does NOT fall out of the BZ integral (`R = 0.2046`, not 0.810).
M_Z is confirmed to be a FORCED substrate-vs-SM oblique difference — a real ~4%-relative residual — exactly the
honest prior. Complete honest result, not a failure, not a fit.** Deliverable:
`proofs/foundations/M_Z_BZ_integrated_vacuum_polarization_2026-06-30.py`; theorem:
`docs/theorems/theorem_M_Z_BZ_vacuum_polarization_2026-06-30.md`.

**What was built (forced, basis-free).** On `directed_edges()`'s own ordering ([B(0),P]=0 verified), the C₃ dart
permutation `P` and winding operator `W=(P−P²)/(i√3)` (eigs {0,±1}; Tr W²=8). Reproduced the Γ split exactly:
Perron Σw²=0, shell √2 **Σw²=4** (chiral half=2), |λ|=1 Σw²=4. Then the genuine BZ integral
`<Σw²·F>_BZ = ∫_BZ Σ_{Im λ>0} |⟨l|W|r⟩|²·Im(λ)/|λ|²` (chirality = the Im λ>0 hemisphere; at Γ the two
hemispheres cancel, h:+2·√7/4, h̄:−2·√7/4). Ratio to the Γ template `[Σw²·F]_Γ=2·√7/4`.

**Result: `R = 0.2046`** (converged ngrid 12→44; basis-free `½Σ|F|` cross-check identical; ENTIRELY shell-band,
|λ|=1 gives 0.000). The Γ "bracket" was an **artifact of evaluating F at its BZ maximum (Γ)** — the genuine
BZ-integrated shell is ~5× smaller and does **not** bracket:

| oblique | value | M_Z |
|---|---|---|
| δ_r (Perron singlet, **LIVE**) | 0.3384% | **+8.1σ** under |
| + chiral shell **@Γ** (artifact, F at BZ max) | 0.3614% | −1.9σ over |
| + **BZ-integrated** shell (R·0.0230% = +0.0047%) | **0.3431%** | **+6.1σ** (NOT closed) |
| SM tree→pole target | 0.3570% | substrate UNDER-predicts by 3.9% rel |

⇒ the substrate's Q-current vacuum polarization, integrated honestly over the Brillouin zone, predicts the EW
oblique to **~4% relative** (0.343% vs SM 0.357%) → M_Z **+6σ**. The framework's **intrinsic precision floor on
the oblique**. The live single-term δ_r (+8.1σ) stands; the forced next term (BZ shell) improves to +6.1σ but does
not close. **`0.810` is NOT forced by T₃−s²Q** — it was the ratio of the BZ maximum to the BZ average, a
coincidence (we did NOT pattern-match it; we built the integral and it came out 0.205). Robustness: the full
two-propagator bubble (interband, genuine field-theory correlator) gives R≈0.57 but does **not** converge
(exceptional-point ill-conditioning of the non-normal B) and is still **not** 0.810 — no natural definition reaches
it. **LESSON BANKED:** a k-point template at a high-symmetry point over-estimates a BZ integral (Γ = spectral
extremum); an apparent Perron-vs-Γ-shell "bracket" can be an evaluation artifact — integrate before bracketing.

**M_Z is now CLOSED as a research question:** it is the framework's last σ-lever and it bottoms out as a genuine
~few-% substrate-vs-SM oblique residual, as the honest prior predicted. No closure exists without un-forcing the
substrate spectrum.

## Standing audit task
Walk the repo for **every** place a value is fit, pattern-matched, calibrated, or graded
"structural-conditional" rather than a **forced read of the object's spectrum**. Each such place is an
incomplete equation → list it here, with the precise statement of *which equation* and *where it stops being
forced*. (**STARTED 2026-06-30** — first verified instances logged below; audit remains open.)

### Logged fit-instances (δ=2/9 / generation-splitting; verified 2026-06-30 by two background audits + spot-check)
The recurring soft spot: **δ=2/9 (the generation splitting) reverse-engineered to the observed value and dressed
as derived.** The VALUE 2/9 is forced three ways (Q(1−Q), Wigner-HM, φ·s); but the *splitting itself* is adopted
(the substrate's leading construction is phase-degenerate, δ≡0) and the −70 ppm subleading is OPEN. Instances
where a file presents a *fit* as a *derivation*:
- **`proofs/foundations/delta_dynamical.py:1176,1477`** — hardcodes `target = 2/9`, scans ~10 measures against it
  (most printed as failing: "≠2/9", "that's 1/3 not 2/9"…), selects "harmonic mean = 2/9", prints
  **"VERDICT: the dynamical derivation is SUCCESSFUL."** Target-driven; the "success" is selection-against-target,
  not a forced read. (The HM route gives the VALUE, but the 1559-line search framing is reverse-engineering.)
- **`proofs/_scratch/O_generation_phase_is_born_invariant_not_phase_2026-06-17.py:23-25`** — `assert
  abs(pat[s]/obs[s]-1) < 0.01` against **observed** δ `{0.22227, 0.1102, 0.0744}`, then prints "δ is a forced A4
  Born-invariant." A fit-to-data asserted as an invariant (the −1% tolerance is the tell).
- **`proofs/_scratch/O_generation_angle_from_fiber_eigenvalue_2026-06-18.py`** — self-labeled "VERIFIED (reproduces
  all 3 sector δ) but **NOT DERIVED**"; needs δ "ADOPTED." Honest tag, but lives as a candidate-derivation.
- **`proofs/foundations/V_Ram_Cl6_iso_all_yukawas_2026-05-26.py:153`** — `delta = 2/9` hardcoded into the LIVE
  Yukawa generator (repo flags it "stale artifact to RETIRE"; persists). The down/up δ {1/9, 2/27} are likewise
  empirical extractions (`BR4_cyclic_toeplitz_koide_reframe` G4/G5 = "EMPIRICAL EXTRACTION, NOT a derivation").
- **Route A vs Route B (the generation derivation is two inconsistent objects):** `derive_generation_spectrum.py`
  (the "forced" ∂_N spectrum, ε=2, **not wired into any prediction**) derives a one-parameter SHAPE that cannot be
  set to the leptons; the LIVE masses adopt empirical Koide (ε=√2, δ=2/9). Scope note added to the file
  (2026-06-30). Reconciling them (or retiring Route A's "FORCED MASS" framing) is the open structural item.
- **`predictions/delta_Koide.py`** — graded δ=Q(1−Q)=2/9 as a "[THEOREM] identity of the Koide parametrisation";
  **corrected 2026-06-30** (it is NOT a parametric identity — δ and Q are independent; CAS verifies only the
  arithmetic; value forced 3 ways, splitting adopted, subleading OPEN). No value change.

## Retraction log (so the same mistake is not re-made)
- **2026-06-30:** the lepton e/μ chirality was claimed "forced" as `δ/k* = 2/27` (ratio 29/25). It is a FIT —
  the operator gives κ_e/κ_μ = 4.15, not 1.16. Retracted. Lesson: a number matching the data is *not* a
  derivation; only the object's spectrum producing it is. This file exists because that lesson kept being lost.
