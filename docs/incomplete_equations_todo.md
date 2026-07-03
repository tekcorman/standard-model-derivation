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
- **THE DICHOTOMY DISSOLVED 2026-07-02 (Ω session 1, `proofs/foundations/OMEGA_T2_a2_dichotomy_2026-07-02.py`,
  ALL PASS) — the winding-side route is CLOSED WITH NUMBERS and the correction's home is LOCALIZED.** The 06-30
  additive-vs-multiplicative dichotomy was posed on the toy map m_t ≈ c_t. Through the framework's ACTUAL
  C₃-Fourier read (baseline reproduces −70.3/−60.5 ppm exactly), with pre-registered sign+magnitude kills and
  no rescans: **additive-on-winding-weights gives −362 ppm (sign wrong, ×5 over); the stability-admissible
  multiplicative gives −2170 ppm (sign wrong, ×31); complex/phase variants −4179/−2012 ppm** — the electron's
  near-cancellation is a ~50× LEVER on every pre-Fourier quantity (∂ln(m_e/m_τ)/∂lnε = −48.7, ∂/∂δ = −51.2,
  exact). The 06-30 "additive +1/μ_rep falls out with the correct sign" was the toy-map artifact. ⟹ **the
  correction attaches to the generation (post-Fourier/C₃-isotype) label, where the lever is exactly 1 — the W1
  water-filling shape κ_j ~ 2α₁³/μ_rep(j)** (τ-row bookkeeping still conjecture-grade: 0.42–0.49× with τ-shift,
  0.84–0.98× without). A blind a₂ scalar on D_F² does NOT force the allocation (absolute shifts over-apply
  ×~500; uniform multiplicative cancels in ratios) — **the 1/μ_rep allocation is genuinely extra information =
  the water-filling theorem; the MDL ceiling STANDS; the −70 ppm is OPEN.** Do not re-walk the winding side.
- **STATION 3 EXECUTED 2026-07-02 (Ω session 2, `proofs/foundations/OMEGA_S2_Q3_isotype_allocation_2026-07-02.py`,
  ALL PASS) — Q3 ANSWERED: NO OPERATOR WITH 1/μ_j DIAGONALS CAN BE THE RESOLUTION; the W1 water-filling
  conjecture is REFUTED as the −70 ppm's closure; the miss STAYS OPEN with its sharpest-ever target.**
  (1) **The correlation decomposition [the honest σ-structure of the demand]:** both ratio rows carry m_τ's
  ±67.5 ppm (soft, corr ≈ +1; ±0.12 MeV ≥ the whole α₁³ = 59.35 ppm budget); the m_τ-FREE combination is the
  hard direction: **δ(m_e/m_μ) = +9.83 ± 0.022 ppm demanded (452σ_exp)**. (2) **The conjugation theorem
  (exact):** the object is real/rational ⟹ conjugation intertwines ω ↔ ω̄ ⟹ μ_ω = μ_ω̄ in EVERY C₃-graded
  sector ⟹ every isotype-multiplicity correction is CHIRALITY-BLIND. (3) **The class kill:** all 6 assignments
  × both τ-rows give m_e/m_μ differentials ∈ {0, ±29.7} ppm — ≥452σ from the demand; **the τ-row question is
  MOOT** (it only ever moved the soft rows); W1's "0.85×/0.98× match" lived entirely inside the m_τ soft noise.
  Kickoff candidates (i) real-walk-class 2nd-order PT and (ii) real resolvent residues die by the same theorem.
  (4) **The sharpened localization (confirming §1's ORIGINAL ∂_N-subleading localization):** the hard core is
  ONE CHIRAL NUMBER — the run phase's next-order completion **ε = δ_eff − 2/9 = −1.7515e-7 ± 3.9e-10 rad
  (0.22%-pinned)** (exact levers: ∂ln(m_e/m_μ)/∂δ = −56.14); one chiral number satisfies the ENTIRE demand
  vector (demo, not a closure: hard row → 0, soft rows → −0.91σ each). Surviving shape class: the chiral/
  δ-dressed sector only — (S1) the ∂_N next-order phase ε, or (S2) mass-dependent dressings g(m_j) (chiral
  through the leading δ); shape coefficients PRE-POISONED (ε*/α₁³ = −0.0029, ε*/α₁⁴ = −0.076 — recorded
  non-matches, NO adoption). (5) **MDL-ceiling framing REVISED:** the ceiling argument applied to the soft
  (experimentally-unpinned) common shift; the hard content is a PHASE, not an allocation. The soft direction
  stays unpinned until m_τ improves ~6× (±0.12 → ±0.02 MeV). **The −70 ppm is OPEN; what closed is the
  allocation DETOUR (by theorem), and the target is now a sub-percent-pinned single number for the ∂_N
  frontier.**
- **∂_N-CHIRAL STATION A EXECUTED 2026-07-02 (`proofs/foundations/DN_CHIRAL_A_route_reaudit_2026-07-02.py`,
  ALL PASS; kickoff pre-registered & committed BEFORE the run: `docs/scoping/DN_CHIRAL_kickoff_2026-07-02.md`,
  7eadd72) — the CHEAP-ROUTE SPACE IS CLOSED against the corrected target.** Classification: R1
  (conjugation-symmetric routes) = exact ZERO in the pinned m_e/m_μ direction by the station-3 theorem (never
  candidates for the hard core — their 06-30 kills concerned the soft direction); R2 (topological chiral: Z₂
  Berry −π, Chern ∓2) = quantization no-go. R3 (dynamical chiral), computed blind then compared once: the
  shell-phase V4 shape α₁³√7/4 = 3.93e-5 (×224 over); the tracked run-phase antisymmetric deviation −0.235 rad
  (×1.3e6 over; identification recorded-dead — and the probe recorded WHY at operator level: BOTH IB branches
  (h, h̄, equal modulus) coexist within EACH winding block, and the J-breaking symmetric part is O(0.2 rad));
  the 2nd-order non-adiabatic differential scale 0.300 rad (×1.7e6 over). Machinery validated: |d(arg h)/ds| =
  2π/√7 to 0.06% tracked on the FULL B(s·AXIS) (the Γ winding projectors do NOT commute with B off Γ — the
  screw needs its Bloch cocycle; recorded). **Poison did real work: 2α₁⁵ = 1.809e-7 sits +3.3% from |ε| and is
  EXCLUDED at 15σ_ε by the pinning itself.** ⟹ **ε is dynamical-chiral-RESUMMED content of the complete ∂_N —
  the suppression from O(0.1 rad) violence to 1.75e-7 IS the resummation.** ARCHITECTURAL BOTTOM LINE: all
  three walk-down residues gate on ONE construction — the run-side/time-leg fluctuation dynamics beyond the
  matching point — with three PINNED read-outs waiting: (1) ε = −1.7515e-7 ± 3.9e-10 rad (−70 ppm hard core);
  (2) the Zff̄ pole-vertex deficit −0.437% ± 0.092% (Γ_Z/M_Z, §7); (3) the graded time-leg a₄ = (2/3)C₂ +
  (2/3)T_H (the gauge row, §5). **The miss stays OPEN; the detours are closed; the next move is the
  construction, not another route.**
- **C3 EXECUTED 2026-07-02 (∂_N construction program, `proofs/foundations/DN_C3_resummed_chiral_phase_2026-07-02.py`,
  ALL PASS; pre-registration committed BEFORE the run, 1472589) — the pre-registered KILL fires: ε is NOT
  free-loop-gas-dressable at ANY forced evaluation level.** Blind candidates: total-gas tick-cumulant shift
  (clock-free δ̄-anchor) ×4.4e10 over; winding-mode cumulant (chiral via the complex shell occupation) ×2.1e6
  over; the all-orders one-body dressing (the most-resummed object a FREE ensemble owns — the tracked
  Green's-function phase) ×4.9e3 over. Recorded observation (no value claim): over-application falls ~3
  orders per resummation level; the free gas bottoms out ~5e3 over — nothing left to resum. Poisons held
  (the cumulant inversion N_eff = 102.19 ± 0.11 vs g² = 100: 19σ, EXCLUDED as exact — pre-declared before
  computing; 2α₁⁵ stays 15σ-excluded). NO adoption (pre-registered rule). **LOCALIZATION SHARPENED: ε
  requires the INTERACTING run — the coupling between the loop ensemble (C0) and the CAR/matter sector —
  which is EXACTLY C1's named edge (the walk↔Fock dictionary at theorem grade). That one edge now carries
  BOTH the gauge-row grade AND the −70 ppm number-mover.** R-ε stays OPEN; the −70 ppm stays OPEN.
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
- **THE SHARPEST FORM YET (2026-07-02 EOD, after the loop-program E-arc — full chain in §7's LOOP entries and
  `docs/scoping/state_of_the_theory_and_strategy_2026-07-02_EOD.md`):** the open equation is now
  **ε = the chiral phase of the DERIVED interacting propagator G_int(u) = ⟨0|(I−uW)⁻¹|0⟩ (E2a: forced, zero
  constants, pairing C = I + iJ), PROJECTED THROUGH THE READ'S OWN CHANNEL WEIGHTS** — δ is the phase of the
  ω-isotype amplitude c₁ of the generation triple (read_masses' C₃-Fourier), whose derived home is E1b's
  odd-half triplet channel (triple→d-slot/Λ¹). What is PROVEN: the interacting ensemble's chiral channel
  EXISTS and flips with the layer bit (E2a; the free ensemble provably has none — the Q3 conjugation theorem
  as control). What is EXCLUDED: the bare dart-channel phase functional (E2b's pre-registered kill, ×2.8e5 —
  it drags the intrinsic shell phase and free-class violence; off-Γ the IB branch pair is not
  conjugate-paired, trap #4). **NEXT: E2c = derive the read-projection functional (the dressed c₁ amplitude);
  then E2d = the blind number.** Target unchanged (Q3): ε = −1.7515e-7 ± 3.9e-10 rad.
- **E2c EXECUTED — THE STATE-BLOCK CLASS IS DEAD; THE LOCALIZATION MOVES UP TO THE WINDING WELD (2026-07-02/03
  sitting, `proofs/foundations/LOOP_E2c_read_projection_2026-07-02.py` ALL PASS 24/24; pre-reg ed410f9 +
  pre-probe amendment (auto-sync ffc0394, disclosed) BEFORE the probe; verify 65/65).** The "project G_int
  through the read's channel weights" program CANNOT be completed on state-blocks — three theorems/measurements:
  (1) **THE BIT-PARITY THEOREM:** the −J frame = the conjugate frame at Γ + conjugation flips the dart winding
  ⟹ for EVERY Fock-state-block winding-compressed rate functional of (I−uW)⁻¹ the mass read's only
  FIRST-order invariant (the δ/phase-difference direction, lever −56.14) is BIT-EVEN (measured ≤ 8e-10 on the
  vacuum block AND the E1b Λ¹ triple-slot block); **E2a's chiral iJ channel feeds ONLY the χ/phase-sum
  direction (bit-odd, flips exactly) — which moves masses at SECOND order only (lever < 1e-6, theorem)** —
  the chirality sits in the invariant the mass read cannot see. (2) **u⁰ VIOLENCE:** the paired-step content
  is STRUCTURAL (M₂ = B² + iK₂, ‖K₂‖/‖B²‖ = 1.05; E1's dictionary has coupling strength 1, no constant to be
  small) ⟹ the extracted channel dressing is u-independent at leading order and O(φ)-large at every u
  (−0.75/−0.76/−0.82 φ at u = 0.05/0.11/0.23) — and the shipped leading read's own 70-ppm agreement therefore
  EXCLUDES the class (were the read such a functional, leading masses would be O(1) wrong). (3) **THE
  WINDING-CATEGORY MISMATCH (new structure):** the interacting ensemble has NO dart-winding grading at any
  computed Fock block ([G_int(Γ), P₃] = u²·1.00 exactly = the iK₂ mixing); the coupled system's true screw is
  **P₃ ⊗ U_π, U_π = the UNIQUE pin lift of the UNSIGNED edge permutation ([W, P₃⊗U_π] = 5e-16), SPINORIAL
  (U_π³ = −I, order 6 = ℤ₆ = the double cover of the C₃ deck action) and VACUUM-MOVING (|⟨0|U_π|0⟩| = ½)** —
  spinor windings do not restrict to Fock blocks. Also banked: the pre-registered E2c carrier B_eff =
  (I−G_int⁻¹)/u died by arithmetic BEFORE the probe (⟨0|W^L|0⟩ = 0 for odd L ⟹ the coupled ensemble is
  PAIRED-STEP ONLY, no free part; ‖B_eff−B‖ → ‖B‖) — disclosed pre-probe, reproduced in-probe; the E2b-era
  "int−free onsets at O(u²)" re-lock phrase was a misreading. The read itself is now IDENTIFIED at theorem
  grade (shipped read ≡ [Γ winding-block moduli (2,√2,√2)] × [Γ-normalized increments ±φs] → C₃-Fourier at
  1e-14; δ first-order, χ/κ/θ_seam second-order; θ_seam EXACT-drops from the δ-invariant to all orders).
  **THE OPEN EQUATION'S NEW HOME (the C0-pattern incompleteness): the READ ↔ ENSEMBLE WINDING WELD — derive
  the bridge between the read's vector-C₃ winding label (the mass circulant's ω-isotype) and the coupled
  system's spinor-ℤ₆ winding label (P₃⊗U_π). Until the weld exists, no functional of the E2a ensemble has
  free image = the leading read + interacting image = its dressing.** NO E2d was run (nothing to evaluate —
  the class died before any number; no blind stage was opened). The E-arc stop-rule fired ⟹ the R-ε research
  front PAUSES (cleanup arc next per the standing strategy); the −70 ppm stays OPEN at target
  ε = −1.7515e-7 ± 3.9e-10 rad; do not re-walk state-block projections of G_int.

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
- **STATION 2 EXECUTED 2026-07-02 (Ω session 2, `proofs/foundations/OMEGA_S2_Q2_internal_a4_gauge_row_2026-07-02.py`,
  ALL PASS) — Q2 ANSWERED (computed, not inherited); THE β-FORMULA LAYER CLOSES; the gauge row localizes to the
  time-leg complex.** (1) **The (−11/3, 2/3, 1/3) row structure is DERIVED** from the heat kernel's two universal
  Seeley–DeWitt coefficients a₄ ⊃ (1/12)trΩ² + (1/2)trE² with the magnetic-moment endomorphism E = −2F·S —
  validated on exact spectra (torus θ-normalization Poisson-exact; Landau trace tB/sinh(tB): the t² coefficient
  IS (1/12)trΩ² = −B²/6); per helicity pair b = −(−1)^{2s}[(2s_z)² − 1/3]: {+1/3 complex scalar, +2/3 Weyl,
  −11/3 vector+ghost} with ONE unit normalization and TWO forced outcomes; component-level ghost bookkeeping
  agrees (+2/3 − 4 − 1/3 = −11/3); b_2HDM re-assembles {21/5, −3, −7} with no per-row tuning (matter-row
  regression: the Weyl row IS the 06-25 cone result's content). **Seeley–DeWitt replaces "one-loop QFT" as the
  declared Type-3 import — the β FORMULA now lives in the same spectral-action layer as ζ_{D₄}(0); the Layer-2
  tag upgrades accordingly** (wording user-gated). (2) **The graded theorem (exact):** opposite-statistics
  pairing cancels the orbital −1/3's pairwise; only paramagnetic content survives: vector pair = −3, chiral/
  Higgs pair = +T ⟹ b_graded = −3C₂ + T_f + T_H, and the completion's add ≡ the shadow rows exactly
  ((1/3)T_f sfermion + (2/3)T_H higgsino + (2/3)C₂ gaugino). (3) **The object-side pairing (the Q2 decision):
  D₃ IS the supercharge** — every nonzero mode's even/odd components are isospectral D₃²-pairs with trivially
  commuting internal charges ⟹ the multiplet reading is DERIVED for all massive/cone content, conditional on
  ONE named identification (form-parity ↔ statistics = the KO 2→6 step). **The FLATS are D₃-UNPAIRED** (parity-
  definite zero modes — the same fact as the index/β separation) ⟹ the spatial complex supplies NO shadow for
  the gauge sector ⟹ **the remaining open equation, sharpened: build the TIME-LEG (γ_t∂_N) fluctuation complex
  for the flat/Higgs sector; its graded a₄ must supply (2/3)C₂ + (2/3)T_H.** β values unchanged; nothing shipped.
- **C0+C1 EXECUTED 2026-07-02 (∂_N construction program; `DN_C0_run_measure_2026-07-02.py` +
  `DN_C1_timeleg_graded_a4_2026-07-02.py`, both ALL PASS; hypothesis pre-registered in C0's committed probe
  BEFORE C1 ran) — THE TIME-LEG COMPLEX IS BUILT; the shadow rows are DERIVED-CONDITIONAL.** C0: the run
  direction's fluctuation measure is FORCED = the object's own loop ensemble (free energy ln ζ(u) =
  −Tr ln(I−uB), Ihara–Bass verified per fiber; propagator = the Q1 fugacity-phase resolvent; subcritical at
  α₁; occupations Bose-form with entropic energies, COMPLEX on the shell ⟹ signed/interference on modes,
  positive on paths); the matter sector's CAR-KMS(β=1) is independently forced; **the Bass exponent
  |E|−|V| = b₁−1 = 2 = the flat count — the gauge sector's fluctuation determinant is the (1−u²)² prefactor.**
  C1: **the graded pairing is the TICK-LATTICE MATSUBARA DOUBLING** — (1−u²e^{2iω})^{b₁−1} =
  (1−ue^{iω})^{b₁−1}(1−ue^{i(ω+π)})^{b₁−1} exactly (periodic + antiperiodic sectors per mode, identical
  internal content); the antiperiodic sector is FERMIONIC by A4/CAR through the walk↔Fock dictionary (tick
  parity = Fock parity; the even sector's u²-quanta = fermion bilinears; parity period = p_toggle = 2). ONE
  rule (antiperiodic partner, statistics flipped, spin |s−½| by A2-minimal selection, station-2 row
  dictionary) reproduces **all three completion rows (sfermion 1/3·T_f, higgsino 2/3·T_H, gaugino 2/3·C₂)
  with the matter row as the no-tuning control**; per group add = {12/5, 4, 4}, b_2HDM + add = {33/5, 1, −3}.
  **ζ_{D₄}(0) status now: β FORMULA derived (Seeley–DeWitt, station 2) + completion CONTENT derived-conditional
  (here). The remaining research edge, stated exactly: theorem-grade the two named framework-class steps —
  (i) the walk↔Fock dictionary (A5-class), (ii) the |s−½| minimal-content selection (A2-class).** Shadows are
  loop content, NOT sparticles (standing note). No value moved.
- **SHARPENED 2026-07-02 (Ω session 1, `proofs/foundations/OMEGA_T1_zeta_D4_gauge_row_2026-07-02.py`, ALL PASS;
  post-Q0 — see §7 for the Q0 correction):** mechanism 1 (the D₄ heat kernel) is VALIDATED: the factorization
  Tr e^{−tD₄²} = (4πt)^{−1/2}·Tr_band is exact; the band trace's cone sector obeys the Albanese dictionary
  (A·t^{−3/2}, v = 1/2, V_alb = 4, verified on a 40³ grid); and **the index/β separation is a per-fiber
  IDENTITY**: Str e^{−tD₃²}(k) = χ(K₄) = −2 for all k, t (γ_t = (−1)^F, D₃ = the supercharge; the H¹ flats are
  the index density — "flat band → index not β" is now exact). **What the completion IS:** b_4d ≡ −3C₂(G) +
  T_f + T_H (the N=1 index/holomorphic form), exactly, all three groups — the "+4 shadows" are the D₄
  complex's own grading partners, and what ζ_{D₄}(0) must produce is −3C₂ + ΣT, not the raw −11/3 list.
  **Localization (the open equation, sharpened):** the band sector CANNOT carry the gauge row — the band-side
  gauge fields (H¹ flats) generate the deck U(1)³, abelian by construction (C₂ ≡ 0); the non-abelian charges
  live in the Cl(6)-Fock internal space ⟹ **the gauge row = the a₄ of the internal (D_F) fluctuation sector
  against the D₄ cone, un-built.** Grade frontier still open; no value gated.

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

### 7. Widths/lifetimes — the frequency-RESOLVED self-energy Σ_X(ω) with thresholds is un-built (F4, located 2026-07-02)
**Status: LOCATED by the pre-registered over-application audit
(`proofs/foundations/F4_width_math_verification_2026-07-02.py`, 20/20; scoping doc
`docs/scoping/F4_session1_real_math_2026-07-02.md`).** A width is −2·Im of a pole of an
energy-resolved resolvent fed by OPEN channels only. The framework's Im structure at the matching
point (Σ(h)=α₁/h, √5/4, √7/4, the 1/√2 step) is **transport/dephasing content** (δρ's verified
use), NOT a particle width: measured Γ/m spans >61 decades (e→t) while every matching-point read
is one constant (Γ_e=0 kill-test; μ over-applied ×1.5e16; gauge bosons ×1.6 = the only right-order
regime).
- **The open equation, stated exactly:** the map X → (channel, pole frequency ω_X in the band
  variable, open final-state set) and the girth-window embedding Σ_X(ω) evaluated THERE — the
  Feshbach theorem gives one constant Σ(h); the width needs the FUNCTION, with thresholds =
  the framework's own dressed masses (Γ_e = 0 by channel-emptiness exactly; top closed at the Z
  by its own m_t read). The E-resolution already exists in the object (Im g_cavity = π·DOS on-cut,
  exact; Bloch F = Im λ/|λ|² over the BZ); what is un-built is Σ at a specified ω off the
  matching point.
- **In-reach sub-question ANSWERED 2026-07-02 (same day, S2a) — the band route is the KILL branch**
  (`proofs/foundations/F4_cone_spectral_function_2026-07-02.py`, ALL PASS; results in the session
  doc §5-RESULT): the substrate cones (adjacency Γ/R and Hodge-Dirac Γ) are ~~chirally warped
  spin-1-like multifolds (non-metric under the cubic little group ⇒ un-isotropizable), C =
  2.76×(1/12π) in mean-v units — NOT the Dirac value~~ **[CORRECTED same day by Q0, next bullet:
  the velocities (v₁₀₀=1/√2, v₁₁₀=1/2, v₁₁₁=1/√3, k·p-verified) are right, but the "non-metric/
  chirally-warped/non-universal-C" interpretation was a coordinate artifact]**; the direct
  pair-creation channel is **q²-DARK**, all low-ω weight flowing through the two exact H¹ flat
  bands (the gauge sector) — these parts stand. ⟹ **the 1/(12π) phase space is NOT band-geometric;
  it is Clifford-kinematic** — still true, in the sharpened LOCKING sense of the Q0 bullet. The
  Σ_X(ω) equation needs: Clifford-trace vertex kinematics + band-side thresholds/content.
  Converges on the SAME continuum-D₄/Clifford keystone as the −70 ppm.
- **Q0 ANSWERED 2026-07-02 (Ω session 1, `proofs/foundations/OMEGA_Q0_albanese_isotropy_2026-07-02.py`,
  ALL PASS; session doc `docs/scoping/OMEGA_session1_Q0_and_targets_2026-07-02.md`): (a) — ISOTROPY
  RESTORATION DERIVED.** The S2a "non-metric" verdict applied O(3) in homology Bloch coordinates;
  the actual little-group action is GL(3,Z) preserving the H₁ cycle Gram (all 24 automorphisms
  computed; invariant form unique by irreducibility). Exact results: both cones are perfect metric
  cones at leading order (sympy char-poly identities); **Q_adj⁻¹ = Gram_H₁ = 3I+C exactly (no free
  scalar)** = the Kotani–Sunada standard-realization/Albanese metric read off the object's own
  H¹/gauge sector; in Albanese momentum **v_adj = 1 and v_Hodge = 1/2 exactly**; the substrate
  cone constant is the **universal isotropic spin-1 value 1/(6π)** (pipeline re-run, +0.15%); the
  S2a anomaly 0.0733 = 2⟨v⟩·(1/6π) postdicted to 0.11% including the two-object coincidence
  (Q_h = Q_a/4). Bands are even in q at every order, so nothing about the cone was "chiral"
  (chirality lives in eigenvectors: adjacency Γ-triple Chern = (−2,0,+2), R = (+2,0,−2) conjugate,
  Hodge pair REAL/Chern-0 — `OMEGA_T4_clifford_12pi_2026-07-02.py`). **a₂/a₄ now mean:** cone-sector
  Seeley–DeWitt coefficients w.r.t. the Albanese volume (V_alb = 4/cell, v = 1/2 explicit); the
  flats are a separate 1D index sector (Str ≡ −2 = χ per fiber, exact).
- **The 1/(12π) layer moved 2026-07-02 (Ω session 1, `OMEGA_T4_clifford_12pi_2026-07-02.py`, ALL
  PASS):** the per-Weyl unit 1/(24π) and the per-Dirac **(v²+a²)/(12π) are DERIVED exactly** from
  the Clifford trace + the Q0 metric (symbolic phase-space integral + calibrated-pipeline
  cross-check at −0.00%). The band cannot supply them — **the LOCKING VIOLATION:** the multifold's
  three "Weyl counts" are pairwise unequal (timelike 4, spacelike 1, topological 2); only a Lorentz
  (Clifford) channel locks all three to its content. Named residual (not absorbed): that the
  physical EW current is the spinor current γ^μ(v−aγ⁵) — P3 FORM derived, PS-embedding split =
  the identified Type-3-conditional step (Clause 10c upgrade argued, prediction-file wording
  user-gated).
- **T-ID2 SITTING 1 EXECUTED 2026-07-02 (`proofs/foundations/TID2_A_split_and_J_2026-07-02.py`, ALL PASS;
  kickoff pre-committed 5d46928) — THE SPACETIME/INTERNAL SPLIT THEOREM CHAIN LANDS.** The Cl(6) generator
  space = the edge space R⁶; its decomposition H¹ ⊕ B¹ is **UNIQUE** (the two inequivalent 3-dim S₄ irreps;
  Hom_{S₄} = 0 — no invariant alternative exists); spacetime = H¹ (Q0/Albanese). Cl(3)_{H¹} exact; equal
  chiralities; commutant M₂ ⊕ M₂. **THE J-THEOREMS: no S₄-invariant complex structure exists; Hom_{A₄} = 1 ⟹
  THE A₄-canonical J (unique up to ±) ⟹ the CAR/Fock quantization FORCES S₄ → A₄ — the framework's ℂ[A₄] is
  the stabilizer of the complex structure quantization requires; every odd permutation flips J → −J ⟹ the ±J
  pair IS the enantiomer pair (srs ↔ srs-z; the joint object = both quantizations).** The canonical modes
  satisfy the CAR exactly, form an A₄ TRIPLET, and reproduce the Hamming/species grading — **read_species'
  Fock structure now derives from THE canonical J, not a pairing convention.** ~~Recorded for sitting 2: N̂'s
  commutant fraction 0.8125~~ **CORRECTED SITTING 2 (`TID2_B_current_form_2026-07-02.py`, ALL PASS): the
  fraction is 3/4 EXACTLY** (exact identity N̂ = 3/2 + D̂/2; the recorded 0.8125 was a vec-convention bug,
  predicted in the pre-registration and verified). **SITTING-2 OUTCOME: the commutant = even-Cl(3)_{B¹} ⊗
  {1, ω₃} exactly = an internal su(2) PER CHIRALITY (all-doublet, Casimir ≡ 3/4, ω₃ 4+4 — four internal
  labels = four species slots per spatial spinor); the dipole remainder D̂ is A₄-invariant and purely mixed =
  the HIGGS-DIRECTION candidate (CLEANROOM §5, flagged). K4 FIRED: the 'charges-internal' half of the
  candidate DIES — the parity (−1)^N anticommutes with the spatial Clifford (split-odd ω₆-class) ⟹ Q̂ has
  ZERO internal component and the Hamming ladder's internal shadow is trivial; the species ladder (1+3+3+1)
  and the split labels (2⊗2 per chirality) are TWISTED by exactly the parity sector. The split-uniqueness and
  J-theorems STAND. **SITTING 3 EXECUTED
  (`TID2_C_lorentzian_assembly_2026-07-02.py`, ALL PASS): THE LORENTZIAN ASSEMBLY LANDS — Cl(3,1) exact with
  γ⁰ = the internal B¹-VOLUME (the (−,+,+,+) signature EMERGES from the split; γ⁰² = −1, nothing inserted);
  γ⁵ = −ω₆ = the existing cl6_chirality (P3's grading IS the assembled 4D chirality — consistency lock); γ⁰
  is A₄-invariant and flips under every odd permutation (the time orientation rides the enantiomer choice);
  the Cl(3,1) commutant = EXACTLY ONE su(2) (chirality-preserving) ⟹ the site-local Fock space = (4-comp
  DIRAC SPINOR) ⊗ (su(2) DOUBLET) — one Dirac doublet per site; and the sitting-2 obstruction's exact
  identity: the Fock parity (−1)^N = −i·ω₆ ∝ γ⁵ — the species (−1)ⁿ factors are CHIRALITY/axial factors; the
  U(1)/color content is deck/winding-sector (read_gauge's own home).** **SITTING 4 EXECUTED
  (`TID2_D_chirality_bit_2026-07-02.py`, ALL PASS) — T-ID2's PLANNED CORE COMPLETE (4/4 sittings, every
  pre-registration git-witnessed): the mirror is T-LIKE (odd permutations: det R|_{H¹} = +1 — SPACE
  orientation PRESERVED; det R|_{B¹} = −1 — time gamma and 4D chirality FLIP); THE ONE-BIT THEOREM: one Z₂
  datum (the enantiomer) coherently carries {quantization sign J, time orientation γ⁰, chirality γ⁵, dart
  handedness e₁e₂} ⟹ a CHIRAL coupling needs no import — L-vs-R IS the srs-vs-srs-z choice; the read's
  T₃-pattern operator (−1)^N/2 = (i/2)γ⁵ EXACTLY (the weak-isospin READ is the chirality grading — the
  sitting-2 twist fully named); the SM-form T̂₃ = (iK₃)P_L (both factors previously derived) has spectrum
  {±½ ×2, 0 ×4} = the one-generation doublet pattern; P_L ↔ P_R rides the bit.** **T-ID1 SITTING 1 EXECUTED
  (`TID1_A_coupling_rule_2026-07-02.py`, ALL PASS; kickoff pre-committed 37b3310): THE COUPLING RULE EXISTS —
  one function rule(channel, disc-class, projection, order) reproduces ALL EIGHT worked dark-sector instances
  exactly from forced inputs (the case law is ONE LAW; the Ihara–Bass discriminant = the coherence criterion:
  disc > 0 ⟹ resummed windings, disc ≤ 0 ⟹ leading-only component-wise-real per S2b; pole dressings rank 1/2
  by the L-rule; order-2 leg counts; c_v and ½-EW flags KEPT). R2 first computation, OUTCOME (i): the mirror
  DISTINGUISHES the factors — the deck U(1) charge is C-conjugated (axis-preserving mirror: W → −W exact),
  the su(2) is SELF-CONJUGATE (bivector Λ², det ≡ +1, exact inner automorphism) ⟹ CANDIDATE per-factor rule:
  charge-flippable ⟹ vector-like; self-conjugate ⟹ chiral (P_L on exactly the su(2)); NAMED TENSION: SM
  hypercharge chirality via the Pati–Salam decomposition (Y = T₃R + B−L mix) — sitting 2 must DERIVE the rule
  and resolve it. R3 (the rate clause) stated with S2b/S6/Q1/C2 pedigree; the loop program's entry form is
  fixed (c-weighted CAR-KMS EW loop).** **T-ID1 SITTING 2 EXECUTED
  (`TID1_B_per_factor_rule_2026-07-02.py`, ALL PASS; pre-reg 1e4edfa): the SECOND su(2) = Cl(0,2)'s
  quaternion factor (the dart qubit) — the PATI–SALAM PAIR su(2)_{B¹} × su(2)_{02} survives the Lorentzian
  assembly (16-dim commutant, commuting pair verified); the dart swap is inner and flips ω₀₂, bit-locked with
  χ ⟹ the LR-MIRROR (joint object LR-symmetric, each enantiomer chiral — PS realized by the mirror pair);
  the HYPERCHARGE TENSION RESOLVES by exact rational arithmetic on the read's own table: **B−L =
  (−1)ⁿ(2n−k*)/k*** (a clean Fock read), **B−L = 2Q − (−1)ⁿ** (the charge's vector/chirality split = the
  split T-ID2 s2 measured on Q̂), **Y_L = (B−L)/2 and Y_R = T₃ᴿ + (B−L)/2 on all 8 states** — hypercharge
  chiral only through T₃ᴿ; the one-unit principle at A2-class with a 4/4 consequence table.** **T-ID1 SITTING 3 EXECUTED
  (`TID1_C_vertex_selector_2026-07-02.py`, ALL PASS; pre-reg 509fc36) — T-ID1 COMPLETE AT MAXIMAL GRADE.**
  The real structure built explicitly (C₈ = γ¹γ³γ⁵∘conj fixes the gammas, C₈² = −1, flips χ; C₂ = σy∘conj
  quaternionic — Cl(0,2) ≅ ℍ confirmed at the real-structure level; su(2)_L generators C-real). The survival
  table's kill fired UPWARD into an **IMPOSSIBILITY THEOREM: the real structure is CHIRALITY-BLIND by
  construction (identical vector/axial antisymmetry columns, cellwise) ⟹ the vertex-level chirality selection
  is NECESSARILY the layer/enantiomer bit = THE ARROW OF TIME (T-ID2 s4's one-bit theorem) ⟹ the SM's
  L-selection adds ZERO description length — it is the already-counted arrow.** Deriving L-vs-R further would
  contradict the joint object's mirror symmetry. Bonus recorded: the temporal/spatial survival split is
  factor-typed (charge-type = densities; su(2)-type = spatial currents; site-local — the cover propagates).
  **T-ID1 SCOREBOARD: R1 (one law, 8/8) · R2 (classification + PS pair + hypercharge on the read's table +
  the bit = the arrow + the impossibility closure) · R3 (rate clause) — THE LOOP PROGRAM'S PROJECTIONS ARE
  FULLY SPECIFIED.** Front-door interpretations user-gated.
- **The ω-resolved VERTEX class established 2026-07-02 (Ω session 1,
  `OMEGA_T3_width_vertex_omega_class_2026-07-02.py`, ALL PASS) — the first width-side object with
  the demanded SIGN.** S6 pins the winding amplitudes z-flat (topological uⁿ, Z_res = 1); the
  forced DURATION (a winding = g ticks) gives the probe-frequency response W(ω) = Σuⁿe^{ingω}.
  **Sign lemma (sympy, exact): W(0) − Re W(θ) = u(1+u)(1−cosθ)/[(1−u)(1−2ucosθ+u²)] ≥ 0** — the
  class can ONLY reduce the pole vertex (DOWN = the demand's direction; the residue class was
  UP-only: the algebra is now bracketed). Pattern-type matches S4 by construction (width-only,
  Γ_W/Γ_Z-cancelling, pole positions untouched, Γ_e = 0 preserved); magnitude range covers the
  demand (max 2u/(1−u²) = 7.8% raw, 0.65% under c_S — c_S re-use stays POISONED). **Value NOT
  claimed (±21% rule): the remaining §7 core = (i) the pole-phase map X → θ_X (one dimensionless
  phase per particle), (ii) the forced vertex projection.** θ-inversions recorded as poisoned
  comparisons only (1.85/1.16/0.44 rad). Γ_Z/M_Z stays OPEN (+4.8σ).
- **C2 EXECUTED 2026-07-02 (∂_N construction program, `proofs/foundations/DN_C2_vertex_loop_class_2026-07-02.py`,
  ALL PASS; pre-registration committed BEFORE the run, 2188fbe) — R-V's CLASS SELECTED: the CAR-KMS matter
  loop.** The q²-dark admixture lead RETIRED (pair-darkness re-verified: |M|² ratio 4.00, 6.8e-6 at q = 0.03;
  the admixture fraction has no forced nonzero home — 0 under the P3 identification, (E/E_sub)²-suppressed
  otherwise; its sign argument was right, its magnitude structurally empty). **The demand in the loop's own
  natural unit (α₂/4π = 0.2690%, fresh from the g₂ leaf): +0.89 ± 0.33 (G_F^v-form) / −1.62 ± 0.34 (α-form) —
  the FIRST O(1)-coefficient candidate class in the entire F4→Ω→∂_N chain** (all others orders-off or excluded
  by pedigree/sign/theorem; the S3 frozen accounting independently attributed the residual to exactly this
  layer). All four S4/falsification surfaces hold (common part cancels in Γ_W/Γ_Z; differential ~0.08% ≪ ±2%;
  pole positions untouched; Γ_e = 0). **REDUCTION: conditional on the P3/PS current identification the loop's
  content is standard EW ⟹ R-V = SM-REPRODUCTION-CONDITIONAL (the 1/(12π) grade family); the from-scratch
  coefficient requires the interacting sector coupling.** ⟹ **PROGRAM-PHASE CLOSE: with C1 (R-G derived-
  conditional) and C3 (R-ε → interacting run), ALL remaining content hangs on ONE keystone — theorem-grade the
  identification layer (the walk↔Fock dictionary / P3-PS current split, the framework's single A5-class
  seam).** Γ_Z/M_Z stays +4.8σ OPEN as shipped. Named-not-acted user-gated option: a NEW registered assembly
  variant importing the EW radiative layer as declared Type-3 (like 48π/1.409) would close Γ_Z/M_Z numerically
  at SM-reproduction grade — a registration decision, not a derivation.
- **LOOP V1 EXECUTED 2026-07-02 (`proofs/foundations/LOOP_V1_car_kms_calibration_2026-07-02.py`, ALL PASS
  32/32; pre-registration committed BEFORE the probe, a5287f4; verify 65/65) — the R-V loop machinery is
  CALIBRATED and the EVALUATION RULE IS DERIVED.** Calibration at/beyond the S2a standard: the Veltman doublet
  Δρ ≡ (N_c g²/(64π²m_W²))F(m₁², m₂²) SYMBOLICALLY (per-log-atom residues exactly 0), Ward/custodial/decoupling
  and Q_u/s²/μ²-independence exact, optical-vs-symbolic Im at 3e-13%, dispersion rebuild ~1e-12%, sub-threshold
  absorptive parts exactly zero (the Γ_e = 0 structural fact), **the massless lock Im Π_T = s(v²+a²)/(12π) at
  1e-14 — the T4 Clifford unit as the machinery's own optical theorem** (1/(48π) = (1/12π)×(g/2c)²-norm).
  **RED: the KMS loop family has exactly TWO parameter-free evaluations (β→0 dead / β→∞ vacuum); interior β =
  a forbidden continuous input (CLEANROOM §7 + §6 III₁); a derived clock is Q1-excluded; the ARROW (the
  already-counted one bit) selects the VACUUM loop ⟹ the EW radiative layer = standard EW one-loop with
  framework inputs — C2's reduction with the evaluation rule now forced; thermality enters as statistics only
  (C1's parity doubling). NEW conditionals: none.** No framework number touched; the target appears nowhere in
  the probe. Γ_Z/M_Z stays +4.8σ OPEN. **NEXT: V2 (fresh session) — pre-registration must freeze the scheme
  (framework couplings in their validated MS-bar-analog roles), pre-decide the known α_s×x_t two-loop-Δρ import
  question (band-relevant), and gate all four surfaces; single marked comparison.**
- **LOOP V2 EXECUTED 2026-07-02 (`proofs/foundations/LOOP_V2_rv_blind_evaluation_2026-07-02.py`, ALL PASS
  12/12; pre-registration committed BEFORE the probe, d37a679, incl. the frozen scheme + tier rule) —
  R-V LANDS: the EW radiative layer on the α-form golden rule = −0.4864% = −1.81 loop units vs the
  pre-registered demand −0.437% ± 0.092% = −1.62 ± 0.34 (pull −0.54, LANDING tier).** Γ_Z/M_Z closes
  **+4.76σ → −0.55σ BY DERIVATION** — equal to the SM's own −0.53σ residual (SM-REPRODUCTION grade doing
  exactly what it says). Method: the certified PDG-2024 worked example (Table 10.6; sums pass at 0.03 MeV;
  the α-form W channel reproduces Γ(W→eν) = 226.29 ± 0.04 MeV at +0.010%) extracted against the SHIPPED
  α-form tree at the PDG MS̄ point; applied at framework leaves with all input-drift sensitivities bounded
  (|ΔS| < 0.012 loop units, 30× under band; scheme legitimacy = the repo's own scheme convention §7: the
  RG-endpoint couplings are MS̄-at-M_Z by declaration). Surfaces: Γ_W/Γ_Z −0.06σ → +0.14σ sub-σ HOLDS (the
  kickoff's "differential ≲0.1%" size-estimate MISSED, actual +0.41% — the κ̂/b-vertex content has no W
  analog; disclosed, not relabeled); poles untouched; Γ_e = 0. Disclosed pre-reg calibration miss: the
  blanket ±2% per-channel gate fires on the b-row (−2.45% = exactly its named content ρ_t −1.25% + κ̂_b
  −0.18% + b-mass −0.41% + common −0.63%; certified independently by the Eq.-10.55 structure check,
  residual −0.41% in-window). **GRADE: SM-REPRODUCTION-CONDITIONAL (C2's reduction + V1's derived vacuum-
  loop evaluation rule; standing conditional = the P3/PS identification). NO VALUE SHIPPED: the +4.8σ
  header STANDS until the user gates the registration step (Clause 10; scope includes the Γ_W/Γ_Z
  companion header). NEXT: R-ε — the γ⁵-graded sector of the INTERACTING run; no worked example exists;
  genuinely from-scratch (C3 killed all free-gas evaluations).**
- **REGISTRATION EXECUTED same day (USER GATE): Γ_Z/M_Z = 0.027350 (−0.55σ, Clause 8c PASS) and
  Γ_W/Γ_Z = 0.83802 (+0.14σ) now SHIPPED with the derived layer** via the new single-source leaf
  `predictions/ew_width_layer.py` (Clause 9b bridge tag explicit; [external] certified PDG-2024 worked
  example in-file; anti-drift welds; 10b tripwire asserts the pre-layer deficit's presence). Value lock
  re-frozen deliberately (103 → 104; the designed FAIL fired on exactly the intended 2 drifts + 1 new);
  DAG 114/0; verify ALL-PASS; MDL ledger margin unchanged (+168.0 — widths are not parameter rows).
  **THE GRADE CEILING IS THE REMAINING §7 CONTENT: the native O(1) coefficient (the interacting sector
  coupling / walk↔Fock dictionary at theorem grade) — the row can never pass bridge-conditional until it
  lands. The M_Z pole oblique (+6σ-class) is NOT touched (rates only, R3 clause) and remains the pole-side
  open item. R-ε (−70 ppm hard core) remains the loop program's open number-mover.**
- **LOOP E1 EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E1_walk_fock_dictionary_2026-07-02.py`, ALL PASS
  16/16; pre-reg witnessed e82ee62 BEFORE the probe) — the walk↔Fock dictionary's OPERATOR LAYER IS DERIVED;
  the A5 seam narrows to ONE weld.** (D1) The Ihara–Bass identity derived in-probe over the dart-reversal
  involution: det(I−uB) = [Π_edges(1−u)(1+u) = the dart-qubit swap eigen-split, T14-welded (swap flips ω₀₂)]
  × [site-cavity determinant with the u²(D−1) backtrack self-energy = cavity_gf's structure; its quadratic's
  two roots = the IB branch pair]; exact at Γ + generic k, real + complex fugacity; all blocks S4-covariant.
  The pair sector's u-quanta are SINGLE pair-mode excitations ⟹ **tick parity = Fock parity is now a THEOREM
  on the pair/flat sector — C1's conditional (i) upgrades there (wording user-gated; R-G grade improves).**
  (D2, the sharpest result) **the step lift is FORCED OUTRIGHT: of 16 A₄-equivariant parity-odd edge-covariant
  families, the vacuum-block discriminator leaves ZERO freedom — X_a = γ_a unique ⟹ ε cannot live in any
  operator deformation; it lives in the STATE coupling of the two forced measures** (C0's walk ensemble ×
  CAR-KMS) across the site↔species seam = the one named remaining conditional. Naive matter-Fock transfer
  candidates measured (Fock traces select ℤ₂-even/cycle classes, not NB): non-matches recorded. **E2 gate:
  force the seam OR show its freedom doesn't touch the mirror-odd sector — pre-register exactly one. The
  −70 ppm stays OPEN.**
- **LOOP E1b EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E1b_seam_parity_2026-07-02.py`, ALL PASS 18/18;
  pre-reg e185e0e) — (b-PASS): THE E2 GATE OPENS.** In the DERIVED frame (the Fock functor Γ(V) = the CAR
  structure's own lift, vacuum-canonical by construction; δ = det V_g ≡ 1, mode rep ⊂ SU(3)): vertex ℂ⁴ ≅
  F_even ≅ F_odd ≅ trivial⊕triplet ⟹ the seam ambient = 4 Schur-forced channels with the derived table
  (even: Perron→vacuum, triple→u-slot; odd: Perron→e-slot, triple→d-slot; all images Hamming-pure). The
  statistics theorem (E1+C1) kills the even half ⟹ **admissible seams = odd-half isometries with ONE
  physical relative phase θ_seam** (recorded bonus: = the (c₀, c₁e^{iδ}) relative-phase class of the mass
  read — question (a)'s remaining content). **The mirror maps the entire admissible set out of itself
  (clean half-exchange) ⟹ ZERO in-layer mirror-odd seam freedom — the mirror on admissibility = the pure
  layer swap = the already-counted bit ⟹ ε's odd channel factors through DERIVED structure only; E2 (blind
  ε) is WELL-POSED with the seam quarantined.** Sitting disclosures: E1 ERRATUM (conj-nullspace bug in
  lift_U; E1 re-run corrected — the D2 zero-freedom verdict REPRODUCES; conclusions stand) + two new trap
  entries propagated to the spine §1 (conj(Vh) nullspaces; phase-incoherent lifts vs the derived Γ(V)
  frame). The −70 ppm stays OPEN; NEXT = E2 (fresh sitting, own pre-reg: T10-bit odd projection frozen,
  lepton-slice point, resummation protocol vs C3's ladder, 4 surfaces, single marked comparison).
- **LOOP E2b EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E2b_blind_epsilon_2026-07-02.py`, run exactly as
  pre-registered 361da9f, banked AS-RUN with its FAILs visible) — THE PRE-REGISTERED TIER-KILL FIRES:
  ε_raw(as-registered) = −4.97e-2 rad = ×2.8e5 over; R-ε stays OPEN; no adoption, no relabeling.**
  Post-mortem (the kill's information): (1) the frozen functional was MIS-FRAMED (my design error, exposed
  by the probe's own control): the difference-form ½(arg g_h − arg g_h̄) equals the intrinsic SHELL PHASE on
  conjugate pairs — the exact non-δ trap the pre-reg cited; the conjugation-ODD invariant is the phase-SUM /
  modulus-DIFFERENCE form (E2a's own A = t₁ − conj(t₂) structure); (2) the line component (+8.9e-4) sits at
  C3-c's killed scale (bare-channel free-class violence); (3) off Γ the IB branch pair is not
  conjugate-paired (trap #4) ⟹ no functional patch suffices. **RE-LOCALIZATION (pre-named, now confirmed
  twice): the READ-PROJECTION LAYER — δ is the phase of the ω-isotype amplitude of the GENERATION TRIPLE
  (E1b's odd-half triplet channel, triple→d-slot/Λ¹); the dressing of δ = the phase of the DRESSED
  triplet-channel amplitude, not the bare dart-channel expectation. The −70 ppm's open equation is now: the
  interacting triplet-channel amplitude's chiral phase (E2a's forced G_int projected through the READ's own
  channel weights).** The interacting chiral channel itself remains a THEOREM (E2a). NEXT: E2c = derive the
  read-projection functional (fresh sitting, own pre-reg); only then a further blind evaluation.
- **LOOP E2a EXECUTED 2026-07-02 (`proofs/foundations/LOOP_E2a_interacting_form_2026-07-02.py`, ALL PASS
  12/12; pre-reg committed before the probe) — THE INTERACTING FORM IS FORCED; THE CHIRAL CHANNEL IS OPEN.**
  The vacuum pairing on the derived Fock structure is **C = I + iJ EXACTLY** (Wick/Pfaffian certified
  in-probe); the interacting walk propagator G_int(u) = ⟨0|(I−uW)⁻¹|0⟩ with W = ΣB_{d'd}γ_{e(d')}⊗E_{d'd}
  (all pieces forced: B, the E1-rigid step lift, the V1 vacuum, the canonical J; γ→1 reduction = the free
  ensemble exactly; = the Wick-weighted path sum order-by-order; odd u-orders vanish = the u²/bilinear
  grading, C1-consistent). **SELECTION RULES: the mirror flips the iJ part exactly; at Γ the free ensemble
  reproduces Q3's conjugation theorem (μ_ω = μ_ω̄ at 1e-16, the control) while the INTERACTING ensemble
  carries a NONZERO ω-vs-ω̄ asymmetry flipping with the layer bit (A(+J) = −conj(A(−J))) — the chiral
  channel the −70 ppm requires EXISTS, is FORCED, and evades the conjugation theorem exactly through the
  iJ pairing.** C3's free-gas over-application ladder explained structurally (the free ensemble had NO
  chiral channel; the free candidates borrowed violence from the wrong sector). K4a did not fire (no
  un-forced choice anywhere). **E2b (the blind ε number) FROZEN in the E2a banner: the winding-chiral phase
  of G_int along the screw line to the lepton slice with the Bloch cocycle (trap #5) frozen in E2b's own
  pre-reg; resummation = the resolvent; 4 surfaces; single marked comparison; tier rule; C3 ladder as
  reference. The −70 ppm stays OPEN until E2b lands or kills.**
- **Q1 DECIDED 2026-07-02 (Ω session 2 station 1, `proofs/foundations/OMEGA_S2_Q1_which_clock_2026-07-02.py`,
  ALL PASS — the pre-registered S1-KILL): the ω-vertex VALUE claim is FALSIFIED; the winding layer
  is excluded TWO-SIDEDLY.** The ω-extension is the fugacity phase (forced, operator-level: one
  tick per NB step). The Z channel in the tick frame is REAL, SUBCRITICAL (2u = 0.078 < 1 — the
  same fact as the arrow) and OVERDAMPED: spectrum max at ω = 0, no real-frequency resonance, the
  only pole purely imaginary (−i·2.55) ⟹ **the channel has no frequency to hand the winding
  interferometer.** Every III₁-admissible phase candidate is trivial (0, 2π) or Γ/M-sized (max
  0.17% raw — out of band); an absolute clock gives QUADRATIC triviality (deficit ∝ θ², sympy —
  even a 100× hierarchy is 40× below band; the framework's ladder ⟹ zero — and this same fact
  PROTECTS all shipped matching-point pole reads); the gap continuation is UNFORCED (+iκ diverges
  4.7e9; the physical retarded depth ≈ 0; −κ full-kill = a by-hand choice, its c_S pairing 0.3384%
  band-edge-outside anyway). **With S6: z-side (residue UP-only, amplitudes waterline-flat) +
  ω-side (zero response at EW poles) ⟹ the −0.437% ± 0.092% is NOT winding-layer content in ANY
  slot.** The T3 sign lemma survives as structure (the two lemmas bracket and exclude the layer).
  S1a (current-projected winding content) SUPERSEDED — no value-slot remains to project onto.
  **The §7 core moves UP: the pole-vertex deficit lives in the INTERNAL (Cl(6)/Clifford) EW-loop
  vertex layer — the framework-native ρ_f/s̄²_eff analogs, genuinely UN-BUILT** (only formally-
  signed existing class: per-leg Family-D c_F u², ~11× too small — S6). Leading sign-correct
  successor candidate, NAMED not built: the q²-DARK band-side admixture of the physical vertex
  (timelike darkness can only REMOVE pole weight ⟹ DOWN forced; magnitude = the vertex's
  band-orbital admixture fraction, requiring the P3/PS-embedding current identification of
  `OMEGA_T4`). Γ_Z/M_Z stays OPEN (+4.8σ); nothing shipped, no falsification surface touched.
- **Lemma PROVED 2026-07-02 (S2b, CAS: `proofs/foundations/F4_S2b_width_ratio_dark_lemma_2026-07-02.py`):**
  (L1) a REAL multiplicative dressing leaves Γ/M invariant EXACTLY and cancels identically in
  Γ_W/Γ_Z when common — and the gauge sector's matching-point dark reads the exactly-real Perron
  channel ⇒ the known dark sector cannot touch widths; (L2) a complex dressing would shift Γ/M by
  (2/(1−Σ_r))(1+(Γ/2M)²)Σ_i = 0.0444 for every shell fermion; (L3) that is over-applied ×1.6e16
  (μ) and contradicts Γ_e = 0 ⇒ EXCLUDED ⇒ **the dark map's component-wise REAL usage is FORCED
  BY STABILITY, not a convention** — the fermion pole stays real at matching-point order; widths
  live in Σ_X(ω) only.
- τ_μ is NOT a target: G_F is calibrated FROM τ_μ (`predictions/G_F.py`) — only the 192π³ rate
  structure could ever be honest content.
- **S3 EXECUTED 2026-07-02 — first width observables shipped as class-(b) assemblies**
  (`predictions/Gamma_W_over_Gamma_Z.py` −0.06σ PASS; `predictions/Gamma_Z_over_M_Z.py`
  **+0.44% = +4.8σ_exp OPEN residual**; registered per parameter_linter.md, value lock 103 PASS).
- **S4 EXECUTED 2026-07-02 — the "EW radiative layer" DECOMPOSED against the framework's own
  oblique set** (`proofs/foundations/F4_S4_width_oblique_decomposition_2026-07-02.py`, ALL PASS;
  diagnostic only, no closure, residual stays OPEN). Findings, each computed not asserted:
  1. **The width assembly is parametrization-consistent** — α-form ≡ G_F^tree-form×(1+δρ) is an
     exact tree identity of the framework's own quantities ⟹ the +0.44% is invariant content.
  2. ~~The G_F TRIANGLE GAP: +0.410%, "wired into nothing" (7b)~~ — **RETIRED same day (S5,
     chase-the-math-up): it is an EXACT IDENTITY of the existing oblique pair.**
     `proofs/foundations/F4_S5_GF_triangle_identity_2026-07-02.py` (ALL PASS): symbolically,
     within the framework's own chain (M_Z = √π·v·√(α₂+α_Y)(1−δ_r), m_W = M_Z·c·√(1+δρ),
     c² = α₂/(α₂+α_Y), g₂² = 4πα₂, G_F^v = 1/(√2v²)):
     **G_F^v/G_F^tree = (1−δ_r)²(1+δρ)** — the "+0.410% gap" = δρ − 2δ_r + O(δ²) = +0.408%,
     DERIVED content (live slack +0.0028%, located: the g₂-leaf vs M_Z-iteration α₂ rounding).
     The S4 framing "wired into nothing" was WRONG — it was fully determined; the identity had
     not been noticed. Corollary: α-form width ≡ G_F^v-form/(1−δ_r)² exactly.
  3. **Sub-equation (7a) ATTACKED AND LOCALIZED (S6, same day) — the residue route is
     SIGN-EXCLUDED; 7a merges into §7-proper as its sharpest numerical target.**
     `proofs/foundations/F4_S6_width_residue_no_go_2026-07-02.py` (ALL PASS):
     - **Sign lemma (sympy, exact):** for the Z-channel dressed pole with per-winding profile
       φ(z) = u(z₀/z)^a, Z_res − 1 = a·c_S·u/[(1−u)² − a·c_S·u] ≥ 0 for ALL a ≥ 0 (the sign
       class fixed by the PROVEN shell z-structure Σ(z) = α₁/z, decreasing). The residue can
       only dress the width UP: {a=0: 0, a=1: +0.353%, a=g: +3.65%}; the demand is DOWN
       (−0.437% ± 0.092%). **Residue route excluded regardless of coefficient** — decisive
       precisely because the demand band is ±21% in coefficient units (1/12, 1/9, 1/8 would
       all "pass" a magnitude test; only the sign/class argument is honest).
     - **Waterline theorem-let:** the framework's own reading (windings = A2-T topological
       classes, the axioms' 2026-04-21 NOTE — explicitly NOT a dynamical resummation) forces
       the profile a = 0 ⟹ **Z_res = 1 exactly: the framework predicts NO oblique-residue
       dressing of the width.**
     - **Taxonomy sweep (argument, not fit):** singlet c_S re-use = the mass-shift projection,
       no derivation for coupling re-use, POISONED (would "pass" at 1.1σ — the trap); vertex
       Family-D O(u²) 8× too small; channel c=1 9× too large; democratic 5/12 4× too large;
       custodial δρ wrong sign and 2.5× too large; S3 omissions ≤0.05% each; combining = fit.
     ⟹ **the −0.437% width normalization is genuine ω-resolved VERTEX content — the Σ_X(ω)
     equation of this §7, now carrying its sharpest target: the Zff̄ effective vertex at the
     pole must come out 0.437% ± 0.092% below the α-form tree assembly (and simultaneously fit
     the S4 pattern across M_Z/m_W/Γ_W/Γ_Z).** Mass-side reads (δ_r, δρ) are pole-POSITION
     content; the width's normalization is not — the matching-point program for widths is
     complete and honestly closed. Γ_Z/M_Z stays OPEN (+4.8σ).
  4. **VERDICT (V2): MULTI-COMPONENT.** The residual vector {M_Z +0.018%, m_W +0.040%,
     Γ_Z/M_Z +0.438%, Γ_W/Γ_Z −0.120%} is NOT collinear with a common ρ̄ direction (single-scalar
     test fails on 3 of 4 rows). Any candidate derivation of the layer must reproduce the
     PATTERN (three distinct directions: width-ρ̄, Δr̄/triangle, δ_r-completion), not one number
     — a much sharper falsification surface. Γ_W/Γ_Z is layer-insensitive (stays sub-σ), as
     shipped.
  5. **☠ PRE-POISONED NUMEROLOGY (declared upon computation, UNUSABLE without forced
     derivations):** demand −0.401×δρ ≈ −2/5·δρ (0.25% apart); the 0.599 ≈ 3/5 complement;
     the 7a candidate-dressing list (item 3). **POISON RESOLVED BY DERIVATION (S5): the
     "+0.410% ≈ (3/8)·δρ" proximity was accidental — the true object is the IDENTITY
     δρ − 2δ_r (item 2). Case study: the poison discipline worked — the coincidence was
     quarantined instead of adopted, and the real algebra arrived one session later.**

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
