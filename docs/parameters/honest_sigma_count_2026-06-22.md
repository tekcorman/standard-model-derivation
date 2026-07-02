> → For the clean restart state, see an internal working note. (This doc is layered with mid-session drift; read the top "UPDATED σ TABLE" + the 2026-06-23 follow-on as current. Lines 3–20 are superseded by the balancing update directly below them.)

# Honest closed-count with genuine σ (Phase 0, 2026-06-22)

> **★★★ NATIVE CORE vs NON-NATIVE OVERLAY (2026-06-22 — re-marking after the dark-corrections nativeness audit,
> an internal working note).** The count below mixes two very different things and
> must be read with this split:
> - **NATIVE CORE (parameter-free, real):** the *bare* reads of the one object (G_NB on srs⊕srs-z, run by ∂_N)
>   predict the SM couplings/masses/mixings to **~1–2%** (a few % at worst) with **zero free parameters** —
>   plus the gauge sector (sin²θ_W=3/8, α_GUT=1/24), the gauge-oblique resolvent (δ_r/δρ = native channels of
>   (I−uB)⁻¹), the species map, the arrow of time. THIS is the defensible achievement.
> - **NON-NATIVE OVERLAY:** the dark corrections (Feshbach Family A/B/C/D/E) that take those ~1–2% bare reads to
>   "≤1σ" are NOT native — the MDL-projection Schur self-energy is provably zero (`H_multiway_construction.py`,
>   B_VD=0), the actual correction is a contour integral with a *chosen* prescription, and the coefficients are
>   tautological/fitted/circular: **5/12 for v is COSMETIC** (cancels identically under the N_hub back-solve —
>   v matches v_GF by G_F **calibration**, not by 5/12; verified c_v∈{5/12,0,1} all give v=246.2197); **c_F is
>   an admitted Clause-6c smuggle**; **β c=1 NOT CLOSED**.
> - **⟹ HONEST RE-MARK:** predictions that reach ≤1σ *only via* the dark overlay (v [=calibration], m_H←λ,
>   m_τ←y_τ, m_ν, V_us, …) are **"native to ~1–2% + non-native correction,"** NOT "derived to ≤1σ." The honest
>   *native* ≤1σ count is smaller than 39; the honest *native* achievement is the **parameter-free ~1–2%** level.
>   The σ_PDG table below is unchanged as a record of the LIVE DAG output, but every dark-corrected row carries
>   this caveat. Granular per-row ledger re-mark = follow-on.
> - **★ BALANCING UPDATE (same day): the dark COEFFICIENTS are native after all — the "non-native" verdict
>   over-retreated.** A constructive derivation found the unifying read **Σ = α₁/h** (resolvent at the channel
>   eigenvalue × girth amplitude; a read of G_NB): one global prescription gives the lepton/ν coefficients
>   EXACTLY, and at the Perron eigenvalue h_P=2 it **CLOSES the two open heavy quarks** — m_b −α₁/2=−1.95%→
>   **+0.23σ** (from +2.99σ), m_t −α₁/4=−0.98%→**−1.0σ** (from +4.71σ), forced coefficient, no tuning. So the
>   dark *read* is native + unified; only the *self-energy mechanism label* is wrong (MDL-Schur=0). **5/12 for v
>   stays cosmetic-for-v (a separate counting read; calibration).** **★ POWER RULE NOW FORCED:** power(L)=2 iff
>   L=0 else 1 (tensor rank from propagation; independently reproduces the framework's own R1/R2 trigger). So
>   **m_b → +0.23σ and m_t → −1.0σ close as a fully-forced statement** (ready to apply to the live DAG; held
>   pending mechanism-narrative review). **M_Z honest negative:** Σ doesn't close its +7.76σ (mixed sub-leading
>   oblique + g_2-running at the 2.3-ppm floor → reclassify precision-floor); **m_W only ~45% M_Z-inherited
>   (→ +1.31σ).** Full writeup: an internal working note.

## ★ UPDATED σ TABLE after the 2026-06-22/23 derivation work (the 12 open gaps)

The dark correction is now the **native first-girth-return of the resolvent, Σ=α₁/h** (mechanism settled,
an internal working note). Applying it (derivation-closed; live-DAG application
deferred for linter discipline):

| param | was | now | status |
|---|---|---|---|
| **m_t** | +4.71σ | **−1.0σ** | **CLOSED** (forced first-return, L=0 → ×(1−α₁/4)) |
| **m_b** | +2.99σ | **+0.23σ** | **CLOSED** (forced first-return, Perron → ×(1−α₁/2)) |
| **M_Z** | +7.76σ | 0.018% | **PRECISION-FLOOR** (structural to 0.018% vs 2.3-ppm PDG; mixed sub-oblique + g_2-run) |
| m_W | +2.39σ | **+1.31σ** | improved (~45% M_Z-inherited); own custodial residual (likely floor) |
| m_ν3 | +2.18σ | +2.18σ | OPEN — ν formula (retire y_ν=1; pin √R step) + scale ← **main genuine target** |
| m_ν2 | +1.87σ | +1.87σ | OPEN — ν formula |
| Ω_DM | +1.66σ | +1.66σ | OPEN — adopted z_eff (NOT a framework defect) |
| V_tb | +1.62σ | +1.62σ | OPEN — V_cb incl/excl data tension ~3.3σ (observational) |
| Ω_b | −1.53σ | −1.53σ | OPEN — adopted z_eff |
| V_ts | −1.50σ | −1.50σ | OPEN — V_cb data tension (observational) |
| g_2 | −2.52σ | −2.52σ | OPEN — gauge-running precision (likely floor) |
| α_EM | +1.01σ | +1.01σ | OPEN — gauge-running precision |

**Net: 12 → 2 CLOSED (m_b, m_t) + 1 precision-floor (M_Z) + 9 open (all ≤2.5σ).** Of the 9, **4 are not
framework defects** (V_ts/V_tb observational; Ω_DM/Ω_b adopted-scale). The biggest σ-failers (m_t +4.71, M_Z
+7.76, m_b +2.99) are all resolved.

**★ 2026-06-23 follow-on derivations (neutrino + precision-floor reassessment):**
- **Neutrinos m_ν2/m_ν3 — formula is COMPLETE/forced, NOT an open formula gap.** y_ν=1 is the *forced unit*
  (top-Yukawa-at-GUT under Pati-Salam; the persistence-law cannot output the unit it presupposes) — retires
  from "adopted" to "forced." R=228/7 forced (K₄ topological invariant, no correction by theorem). Residuals
  fully attributed, neither derivable: **m_ν3 → +1.58σ** (the one adopted scale N_hub, m_τ-anchored), **m_ν2 →
  +1.18σ** (R=228/7 sitting −1.1σ below measured R=33.55±0.89 — a sub-2σ data-tension on a forced invariant).
- **m_e, m_μ — "precision-floor" was MISLABELED; the residual is the per-C₃-rep generation-phase derivation.**
  Any channel-common first-return (leading or sub-leading) **cancels EXACTLY in the mass ratio, all orders
  (proven symbolic)**; the −70 ppm residual is generation-dependent (e/μ asymmetry, the ω/ω̄ phase / V_Ram-iso δ
  frontier) — a NAMED open derivation, not a measurement floor. (This is why the ~22 α-power probes failed.)
- **M_Z, g_2, α_EM — NOT floors; they are TARGETS (user directive 2026-06-23: no floors accepted).** The
  leading first-return ladder is exhausted in δ_r, so the residual is *new structure to derive*: M_Z's +0.054%
  is the **sub-leading SM oblique carried by resolvent channels OUTSIDE the uniform-Perron singlet** (δ_r is the
  Perron-singlet eigenvector projection; the other eigenvector projections of G_NB are the target), and the
  −0.035% is the **gauge running** to derive from the intrinsic ∂_N (NOT the falsified SM 2-loop). g_2/α_EM are
  the same native-running target; m_W follows M_Z.

**⟹ Re-targeted follow-on (no floors, no adoptions) — see an internal working note:**
T1 = generation phase δ — **✅ DONE (settled 2026-06-21, NOT open): δ is DERIVED = the forced directed phase of
the chiral ∂_N run** (0:±2π√(3/7); "the chirality of the running, NOT a parameter" — `generation-hierarchy-forced-recurrence`).
The δ=2/9 hardcode (`V_Ram_Cl6_iso_all_yukawas:153`) is the OLD *worthless* fit, SUPERSEDED → retire it. Only
residual = the observer's slice (the one free axis, same as scale+arrow); m_e/m_μ −70 ppm lives there. (My 2T
dive only re-confirmed the phase is dynamical, not static — an internal working note.)
**The lead derivation target is now T2** = the M_Z
non-Perron oblique channels + the native gauge running (M_Z, g_2, α_EM, m_W). T3 = the adopted scales (derive
z_eff for Ω_DM/Ω_b; derive-or-justify N_hub for m_ν3/H_0/t_0/Λ). T4 = observational tensions (V_ts/V_tb V_cb
data; m_ν2 R-vs-data) — verify, track. **Derivation work this session DONE; these are the live targets.**

---

Re-tally of the **live DAG** (`run_predictions.py`, 110 files, `_validate_dag.py` clean) in the
**σ_PDG-only** convention — the genuine experimental error, never widened by a framework systematic.

## The σ convention (settled)

`parameter_linter.md` is internally contradictory: Clause 8a prescribes `σ_combined = √(σ_obs²+σ_theory²)`;
Clause 8c and the bridge-check §2c say "σ_PDG only, do not widen." **The live code resolves it: σ_PDG only**
(`run_predictions.py:303`; every file uses experimental σ; no file computes σ_theory). This is the honest
convention and the one used here. (Doc fix owed: delete/deprecate the Clause-8a σ_combined language so the
written rule matches the live rule.)

## The honest split (live table, σ_PDG)

| category | n | meaning |
|---|---|---|
| **Numerically closed (≤1σ_PDG)** | **39** | a measured observable, matched within experimental error |
| **σ_PDG open gap (>1σ)** | **12** | structurally derived but >1σ_PDG and **NOT closed** — the previously-cited "floors" are falsified/retracted (below) |
| **Precision-floor** | **2** | m_e, m_μ: relative ~0.008%, but σ_PDG is sub-ppb — unreachable by any integer ratio |
| **Category-B (coasting)** | **8** | framework-vs-ΛCDM; the σ vs CMB-PDG is a *predicted* difference (Clause-8 special) |
| **Calibration anchor** | **1** | G_F — trivially matched (it calibrates N_hub); not a prediction |
| **Structural / definitional / derived-no-obs** | **~19** | forced + exact (k*=3, |V|=4, N_gen=3, θ_QCD=0, Q_Koide=2/3) or unmeasured (PMNS Majorana phases, R_ν) |
| **(framework-internal, not SM targets)** | ~25 | α₁, h_walker, ξ_t, N_hub, b_i, … — the DAG's internal quantities |

## The 12 open gaps — derived values that miss by >1σ_PDG (channel-characterized; no *clean* closure yet)

**Read this correctly: all 12 are DERIVED and their values are LIVE IN THE DAG** — that is why each has a
number and a σ. "Open gap" means only: *the pulled-in derived value misses the precise measurement by >1σ_PDG,
and we have no further K-rational / non-tuned correction for the residual.* It is a residual on a derived value,
**not** a missing derivation, and **not** "unexplained" — the channel/mechanism of each residual IS
characterized (below). What is absent is a *disciplined closure*; closing several would require tuning the free
scale, a non-K-rational constant (c=π²), or physical sparticles — i.e. the cheating the discipline forbids.

**ON TARGET — do not re-report these as "misses" (the derived mechanism WORKS):** the substrate resolvent
G=(I−uB)⁻¹ already reproduces the EW obliques K-rationally and puts the sector on target. **M_Z:** the leading
δ_r (Perron, c_S=1/12) is K-rational/theorem-grade and cuts the tree residual **20× → 0.018%**; the +7.76σ is
that sub-percent residual measured against a 0.0023% error bar (sub-leading c=π², right sign, K-rationality
pending). **m_W:** custodial δρ (h_P) is modulus-locked → leading read exact, ρ-test **PASSES +0.76σ**; the
absolute +2.39σ is M_Z-inherited. **ν:** structure fully forced (rank-2, m_ν1=0, seesaw, R_ν=228/7,
modulus-locked); relative residuals **m_ν3 +0.87%, m_ν2 +2.37%**, the rest formula-completion (y_ν=1, the √R
ratio step). The σ-labels measure remaining *precision*, not a missing derivation.

| rows | residual | mechanism found (channel) | why no clean closure |
|---|---|---|---|
| m_t (+4.71σ) | anchor y_t=1 **forced-exact** (saturation channel, all orders) | the +0.82% lives in the v/√2→pole-mass **bridge** (SM-side scheme), not the mass mechanism |
| m_b (+2.99σ) | Perron channel; run forces a **right-sign** negative s²-correction | magnitude needs the scale `s` fixed independently + coefficient c=π² is **non-K-rational** (closing now = tuning/forbidden) |
| M_Z (+7.76σ) | leading δ_r **pulled in** (cut residual 20×: 0.357%→0.018%) | remaining piece is the sub-leading Perron c=π² (non-K-rational). [2-loop FALSIFIED; Sirlin Δr RETRACTED] |
| m_W (+2.39σ) | custodial δρ **passes clean** (+0.76σ, K-rational, pulled in) | the absolute +2.39σ is **inherited from M_Z** |
| α_EM (+1.01σ), g_2 (−2.52σ) | gauge running from the forced boundary (3/8, 1/24) | EW-oblique Δr closure retracted (non-K-rational); α_EM just over the line, g_2 the largest running residual |
| m_ν2 (+1.87σ), m_ν3 (+2.18σ) | structure **forced** (rank-2, m_ν1=0, seesaw, R_ν=228/7, modulus-locked channel) | per-parameter **formula** incomplete (adopted y_ν=1; the m_ν2=m_ν3/√R step carries ~1.5% extra); scale is spent on v |
| V_ts (−1.50σ), V_tb (+1.62σ) | unitarity-derived from matched elements | sit *within* the observed V_cb excl/incl tension (data self-disagrees ~3.3σ) — largely **observational**, not a closure |
| Ω_DM (+1.66σ), Ω_b (−1.53σ) | derived conditional on the **adopted** z_eff | the P22-partition residual (~0.7%) is not independently derived |

**The directive (user, 2026-06-22):** do not categorize a gap unless we can prove the categorization or
have the answer. The path is to **use the new top-down math** (the master equation / recurrence-under-running
/ persistence-motion) to understand what each gap actually *is* — not to paste a falsified floor on it.

## New-math diagnosis of the m_t/m_b gaps (2026-06-22, sealed in-box + firewall)

Recon finding: the live m_t/m_b chain applies **no running** — the residual is 100% in the static Yukawa
**anchor** (m_t=(v/√2)·1; m_b=v·(2/3)¹⁰). So it is an anchor question, not a running one (the falsified
2-loop is the *gauge* sector, a different gap-class).

Sealed in-box result (recurrence-under-running, posed with NO targets): the static power ((k−1)/k)^L is the
**Perron channel's** leading term, and the run corrects the channel-types **differently** —
- **Saturation (L=0, the m_t channel):** persistence = 1 **exactly, all orders** — the run adds nothing.
  ⟹ y_t=1 is *forced-exact*; the +0.82% is **NOT in the mass mechanism**. It is in the (v/√2)→pole bridge
  (scheme/conversion or genuine miss) — *not* closeable by touching the anchor.
- **Perron (L=g, the m_b channel):** the run forces a real **negative** s²-order modulus correction
  `P=(2/3)^L·[1−(4π²L/(k−1))s²+…]` — **right sign** to lower the +2.1%-high m_b. Candidate closure;
  magnitude set by the run-position `s` (scale) — to be fixed **independently, not tuned**.
- **Shell (complex, the lepton channel):** modulus **exactly locked** (Ihara–Bass rigidity), pure phase
  (φ=2π/√7). Consistent with the leptons being the precision-matched sector.

DERIVED characterization (not a falsified floor): the gap is **channel-structured** — top-anchor exact,
bottom right-sign-correctable, leptons locked.

## New-math diagnosis of the M_Z/m_W gaps (2026-06-22, sealed + firewall)

Recon: the leading oblique is ALREADY K-rationally closed by the substrate resolvent G=(I−uB)⁻¹ — δ_r
(Perron, c_S=1/(2|E|)=1/12) cuts the M_Z tree residual **20×** (0.357%→0.018%); δρ (complex/h_P, custodial)
gives a clean ρ-test **+0.76σ PASS**. m_W's +2.39σ absolute is **M_Z-inherited**; its own custodial passes.
So this is NOT a falsified-floor gap — the retracted thing was the SM Sirlin Δr (continuum); the framework
replaced it K-rationally.

Sealed in-box result (resolvent read g(h;u)=1/(1−uh), sub-leading under the run, NO targets, srs girth g=10):
- **Complex band-edge channel** (δρ/m_W-custodial): modulus **rigidity-locked** (|h|²=k−1 ∀s) → leading read
  **EXACT**, phase-only (φ=2π/√7). ⟹ *derives* why m_W custodial passes clean: leading δρ is complete.
- **Real Perron channel** (δ_r/M_Z): root modulus **moves** → forced **negative O(s²)** correction
  δg_P ≈ −2c·u·s²/(1−2u)², c=π². **Right sign** to lower the +0.018%-high M_Z.

## Unified channel-picture (both clusters, firewall)

| channel | run behavior | who | gap status |
|---|---|---|---|
| saturation (L=0, \|h\|=1) | exact, all orders | m_t | anchor exact → gap is bridge-side |
| Perron (real, modulus moves) | forced **negative** s² | m_b, M_Z(δ_r) | high-by-a-bit → right-sign correction |
| band-edge / shell (complex, locked) | leading **exact**, phase-only | leptons, m_W(δρ), **ν** | modulus precision-matched/clean; **ν mass-formula incomplete (scale is spent on v)** |

**Open checks (no goal-seeking, K-rational discipline):** (i) fix `s` **independently** (the scale), then test
the Perron magnitude — resolves m_b AND M_Z at once (shared mechanism, one s); NOT tuned. (ii) the sub-leading
coefficient c=π² (Brillouin-zone band-top Hessian) is **not K-rational** — adjudicate under Clause 6/9
(Bloch-gradient-type admissible vs continuum-π forbidden). If forbidden, the new math *characterizes* these
residuals (mechanism + sign) but does NOT give a K-rational theorem-grade closure. The leading δ_r/δρ ARE
K-rational/theorem-grade. (iii) m_t's bridge/scheme gap is separate.

## New-math diagnosis of the m_ν gaps (2026-06-22, sealed + firewall)

Recon: m_ν3 = v²/M_R (Type-I seesaw, rank-2); M_R = δ⁴·M_Pl/(2·k*·N_atoms) (K-rational); v ∝ N_hub^(−1/4)
⟹ m_ν3 ∝ 1/√N_hub. R_ν=228/7 (theorem-grade, anchor-free, +1.4σ). Load-bearing adoption: y_ν=1.

Sealed in-box result (self-conjugate sector, NO targets, srs g=10): the neutrino sits on the **complex
Ramanujan shell** |h|²=k−1=2 — the *same* modulus-LOCKED band-edge channel as leptons/m_W-custodial.
- **m_ν1=0 (rank-2) is FORCED:** of 3 windings, the trivial one has girth-holonomy h^g=+1 (inert) →
  2 dynamical + 1 inert. The massless lightest neutrino is structural.
- **Modulus LOCKED**, run acts by phase only (rate 2π/√7) → no channel-correction.
- **Seesaw suppression ((small)²/(large)) AND the internal ratio are FORCED** (clean algebraic from the
  holonomy phase), amplitude-independent.
- **Overall amplitude/scale is the FREE UNIT-INPUT** (III₁ scale-freeness — the one thing the object cannot
  supply; the gap-strength m / time unit).

FIREWALL (CORRECTED 2026-06-22 per user — the "it's the scale" framing is REJECTED): the one free unit is
**pinned ONCE by v/G_F** (v matches −0.0001%); m_ν3 ∝ N^(−1/2) vs v ∝ N^(−1/4) **CONFLICT** — the +1.74%
unit-shift that would close m_ν3 breaks v by −0.43% (and still misses m_ν2). A unit-consistency scan over ALL
unit-dependent parameters confirms **no single unit fits all** (N^(−1/4) EW sector consistent; N^(−1/2) ν and
N^(−1) H_0 conflict). So with the v-pinned unit, **m_ν3 (+0.87%) and m_ν2 (+2.37%) are GENUINE
FORMULA-INCOMPLETE residuals** — in the seesaw/M_R, the adopted **y_ν=1**, and the m_ν2=m_ν3/√R step (m_ν2's
+2.37% is NOT just m_ν3's +0.87% propagated → an extra ~1.5% lives in the ratio step). The *structure*
(rank-2, m_ν1=0, seesaw form, R_ν, modulus-locked channel) is still FORCED; what is NOT closed is the
per-parameter FORMULA, and the scale is **not available** to absorb it. Field-ext flag: phase 2π/√7 ∈ ℚ(√7),
outside K=ℚ(√2,√3,√5).

## Phase 0 close-out — V_ts/V_tb and Ω_DM/Ω_b (characterization, no sealed needed)

- **V_ts (−1.50σ), V_tb (+1.62σ):** unitarity-derived (third-row CKM) from the well-matched elements; sit
  within / inherit the observed **V_cb excl/incl tension (~3.3σ data self-disagreement)** — largely an
  OBSERVATIONAL tension, not a framework defect. Caveat: V_tb is precisely measured, so +1.62σ is borderline
  (inherits the V_ts/third-row unitarity).
- **Ω_DM (+1.66σ), Ω_b (−1.53σ):** explicitly CONDITIONAL on the **adopted z_eff** (an N_hub-class cosmology
  adoption); residual is the P22-partition ~0.7%. Same free-scale class as the neutrino — ties to N_hub.

## The one free unit (the math object's scale) — the discipline (2026-06-22, user)

Frame the scale as the **math object**, not the number N. The substrate has exactly **one free unit** — the
III₁ scale = the observer's run-coordinate. It is **fixed ONCE**, and the data pins it via v/G_F (v matches
−0.0001%). **The unit is then SPENT.** It is NOT a per-parameter dial: a unit shift is global, hitting every
dimensionful prediction by its own power (v: N^(−1/4), m_ν: N^(−1/2), H_0: N^(−1), Λ: N^(−2), t_0: N^(+1)).

**Rule (rejecting the cherry-pick):** no gap may be attributed to "the scale" unless the SAME unit fixes ALL
unit-dependent parameters simultaneously. The unit-consistency scan shows it does NOT — so:
- the **N^(−1/4) EW sector** (v, m_H, m_τ, m_W, M_Z) is unit-consistent; its residuals are channel/formula
  (Perron oblique, m_t bridge), not scale;
- the **neutrinos (N^(−1/2)) and H_0 (N^(−1))** carry **genuine formula-incomplete residuals** the unit
  cannot absorb.

**Completeness criterion (the goal):** the theory is complete when the ONE unit, fixed once, makes EVERY
dimensionful prediction fit *simultaneously* — i.e. when every per-parameter FORMULA is complete. The open
work is **completing formulas**, never re-tuning the scale.

## Follow-on closure tests — to complete the theory later (DO NOT LOSE)

The gap-*understanding* is complete and channel-structured. To *close* the gaps later, the disciplined tests:
1. **Perron cluster (m_b + M_Z, shared mechanism):** fix the run-position `s` **independently** (the scale),
   then test whether the forced negative correction −2c·u·s²/(1−2u)² (c=π²) lands +2.1% (m_b) and +0.018%
   (M_Z) — ONE `s` resolves both. Never tune `s` to the masses.
   **★ SEALED RESULT 2026-06-22 (`derivation_topdown/bridge/perron_curvature_run_scratch.py`):** the FORM is
   exact — Perron band top **h_P=2 at Γ**, band-top curvature **H=4π² (exact)**, read expansion
   a=−2π²u/(2u−1), b=π⁴u(130u−29)/[18(2u−1)²]. **Two terms suffice for small `s` (perturbative window s≲0.13);
   the residuals sit at s≈0.01 (deep in it), so YES at the data's scale — but at the screw fraction s=1/3 the
   displacement is non-perturbative (must be summed).** **HONEST CATCH: the bare object forces the direction
   (C₃ screw) and the shape (H,Q,a,b) but NOT the magnitude `s` — `s` is the free type-III₁ scale (= the one
   unit already spent on v).** So the object gives the correction's form+sign, not its size. NOT yet closed:
   needs the v-pinned unit to supply ONE `s` giving BOTH m_b and M_Z without per-parameter tuning (the
   cross-pollination test — same scale-consistency question as ν/H_0).
   **★ FIREWALL CONSISTENCY CHECK RAN 2026-06-22 (`derivation_topdown/bridge/firewall_s_mb_MZ_consistency_scratch.py`): CONSISTENT, NOT CLOSED.** Backing `s` out of each residual independently (NO tuning): m_b (+2.15%) → **s=0.0104**; M_Z (+0.018% remaining, sub-leading on the already-applied δ_r) → **s=0.0146**; **ratio 0.71 (agree to factor 1.5)**, same sign, both deep in the perturbative window (≪0.13, quartic negligible). A real non-trivial pass — a free `s` could have implied wildly different / oppositely-signed values and did not. **BUT not a closure:** `s` is backed OUT of the data (not derived); the ~40% spread means one `s` does NOT cancel both to precision (order+sign consistency, not a precise joint fit); interpretation-dependent (sub-leading-on-δ_r; replace-δ_r disagrees 6×). The magnitude `s`≈0.012 is undetermined by {D,srs,MDL}+v (geometric candidates 1/3, 0.355 overshoot ~1000×). **The only principled source for `s` is a GAP EQUATION / interaction self-consistency (cleanroom i02 dimensional transmutation, s~e^(−1/gN₀) naturally small) — the real frontier, NOT tuning.**
   **★ GAP-EQUATION ROUTE RAN 2026-06-22 — CLEAN STRUCTURAL NEGATIVE (`derivation_topdown/bridge/gap_fixes_s_scratch.py`).** The transmutation m~W·e^(−1/gN₀) is a band-CENTRE order parameter (finite DOS → exponential); **`s` is a band-TOP crystal-momentum coordinate where the DOS VANISHES** (Perron top = a 3D band edge, DOS~√(ε_top−ε), exponent 1/2 confirmed at 40³ sampling). At the top the pairing kernel is FINITE (0.385) → the gap equation has only a THRESHOLD solution m~(g−g_c)^power, **NO exponential transmutation**. So `s` is structurally NOT the transmutation ratio; forced identifications give s≈0.25–0.72 (off the band) and depend on the regularization-ambiguous N₀. **⟹ ALL THREE routes to fix `s` are now ruled out: geometry (candidates 1/3,0.355 overshoot 1000×), v (doesn't supply), gap-equation (structural). `s` is an IRREDUCIBLE free type-III₁ scale (plausibly the observer's run-coordinate, like the time-axis). The Perron cluster m_b/M_Z/m_W does NOT close — mechanism derived + data-consistent (one s≈0.012 to 1.5×), magnitude undetermined by {D,srs,MDL}+v.**
   **★★ MODULAR-TIME ROUTE RAN 2026-06-22 (`derivation_topdown/bridge/modular_time_fixes_s_scratch.py`) — `s` IS slaved to time, but the cosmic value is NEGLIGIBLE (decisive).** The modular structure FORCES **s(τ)=C_s·e^(−τ/2)**, τ=log N, C_s=√3/(2π)≈0.276 — exponent ½ forced (3 routes: MDL concentration, modular cooling at the 3D edge, III₁ energy–time conjugacy). So `s` is NOT an independent free scale — it collapses to τ's origin (the "when"); another confirmation of "time is the one free axis." **BUT at the cosmic modular time τ=log N_hub≈140: s≈C_s·e^(−70)≈1e−31 ~ 1/√N_hub → δh~1/N_hub~1e−61, NEGLIGIBLE.** (The s≈0.012 the data wanted = τ≈6, an early-universe time, not now.) **⟹ DECISIVE: the intrinsic ∂_N run-correction is ~0 at cosmic time — it does NOT explain the m_b/M_Z residuals; the "s≈0.012 consistency" was the FORM matching a value the framework doesn't produce. The leading band-top read IS the framework's heavy-mass answer; the residuals are the OBSERVABLE-BRIDGE (structural→pole/MS̄ scheme conversion) — the standard SM running between scales, NOT the intrinsic displacement.** Corrects the "Perron forced-negative correction could close m_b" framing (real in form, negligible in magnitude). NEXT for m_b/M_Z: the scheme/pole bridge.
   **⚠ RETRACTED 2026-06-22 (the claim below that the bridge is a NATIVE Feshbach self-energy is WRONG — see an internal working note).** Critical read: the framework's OWN native computation (`proofs/foundations/H_multiway_construction.py`) shows the MDL-projection Schur self-energy is **identically ZERO** (MDL is absorbing, B_VD=0). The actual "Feshbach" correction is a contour integral with a *chosen* contour + assumed density (non-native); the families A/B/C/D/E are an accreting taxonomy; the least-native coefficient c_F is an admitted Clause-6c smuggle. m_b/m_t carry no dark correction and the would-be one is ~10× too small. **So the mass residuals have NO native closure, and dark-corrected predictions (v, λ, y_τ, m_ν, V_us) rest on a non-native overlay (strongest=5/12 for v, possibly real; weakest=c_F).** The only native kernel is the unified-oblique RESOLVENT (δ_r/δρ = channels of (I−uB)⁻¹), gauge-sector only. Original (now-retracted) claim follows:
   **★ THE OBSERVABLE-BRIDGE IS FRAMEWORK-NATIVE — the FESHBACH SELF-ENERGY (`docs/framework/framework_scheme_convention.md`).** Not SM MS̄/pole machinery: the bare srs combinatorial coupling is dressed by the walker excursioning into the MDL-discarded substrate via one girth cycle (amplitude α₁=(2/3)⁸≈4%); C = C_bare + Σ_Feshbach = physical value. "Tree-level in the framework's sense" (what looks like 1-loop matching = a substrate-vs-rendering geometric fact). Machinery = the dark-correction contour integral (Ramanujan circle), **theorem-grade & applied for v (5/12 → −0.0001%), V_us, m_ν2, m_ν3.** **The residuals on λ(+0.6%), y_τ(+0.13%), m_b, m_t, M_Z are — in the framework's OWN framing — UN-DERIVED FESHBACH ANALOGS** (same machinery, not yet computed for these observables; sizes fit: α₁≈4% → ~1–2% after coefficients). So m_b/M_Z/m_t closure = compute their Feshbach analogs, NOT import SM loop machinery. **CAVEAT (quarks): the convention says "bare+Feshbach=pole-equiv," but m_b≈4.27 sits near MS̄(4.18) not pole(~4.78) — quark reads are MS̄-like; pinning which scheme the quark reads land in = the Need-D-3 per-Weyl-spinor dictionary. Leptons/Higgs sit clean; quarks carry this extra scheme ambiguity.**
2. **π² K-rationality adjudication — RESOLVED 2026-06-22:** the Perron band-top Hessian **H=4π² is genuine
   Bloch-band-curvature**, a rational multiple of the forced C₃-screw phase rate c=2π/√3 (H=3c²=4π²); π enters
   *only* through lattice BZ periodicity. So it is **admissible (lattice Bloch-gradient), NOT an imported
   continuum transcendental** — the c=π² blocker is dissolved. (The ν √7∈ℚ(√7) field question is separate, still open.)
3. **Complete the ν + H_0 FORMULAS (NOT the scale):** the unit is spent on v (N^(−1/4)); m_ν3/m_ν2 (N^(−1/2),
   +0.87%/+2.37%) and H_0 (N^(−1), +1.16%) carry formula-incomplete residuals the unit cannot absorb. Complete
   the seesaw/M_R + retire **y_ν=1** + pin the **m_ν2=m_ν3/√R** ratio step (the extra ~1.5%); for H_0 decide
   genuine-coasting-tension vs missing term. The test of success: every dimensionful prediction fits the ONE
   v-pinned unit *simultaneously*. Frame as the math object's structural completeness — never as tuning N.
4. **m_t bridge/scheme:** anchor proven exact; the +0.82% is the framework-value-vs-pole relation — resolve on
   the SM-bridge side, NOT the anchor.
5. **m_ν2 sub-leading:** pin the extra ~1.5σ over √R (observable convention Δm² vs √Δm², or a sub-leading ratio).
6. **y_ν=1 adoption:** the ν Dirac Yukawa is adopted (load-bearing for the scale) — derive or characterize.
7. **Field-extension flag:** ν phase 2π/√7 ∈ ℚ(√7), outside K=ℚ(√2,√3,√5) — the chir-7 sector may force K⊇√7.

## Top-down synthesis (2026-06-22): the gaps are the framework's intrinsic RUNNING

Zooming out across the whole residual table, the misses are not scattered — they are **one mechanism**. The
framework computes **leading-order reads of G_NB at its natural reference scale** (the boundary/structure);
observables live at other scales; **the residual is uniformly the flow between scales = the framework's own
∂_N running, currently truncated at leading order.** This is the framework's INTRINSIC running (the resolvent's
run-position `s`-dependence), **NOT** the SM 2-loop RG (that import was falsified — it moved the wrong way).

| residual | the running it is |
|---|---|
| M_Z, m_b (m_W via M_Z) | **CORRECTED 2026-06-22:** the *intrinsic* ∂_N run-displacement is negligible at cosmic time (s~e^(−τ/2)~1/√N_hub~1e−31). So the residual is the **observable-bridge** (structural→pole/MS̄ scheme conversion), NOT the intrinsic displacement. (Curvature 4π² & sign were right; magnitude ~0.) |
| m_t | v/√2→pole = QCD running framework-scale → pole (observable-bridge) |
| m_e, m_μ | the 10⁻⁴ = scale-dependence of the Koide read (observable-bridge / scheme; precisely calibrated) |
| g_2, α_EM | gauge-running residuals (observable-bridge) |

**Refinement (2026-06-22):** "the gaps are the running" = the **observable-bridge** running (structural value → measured-scheme conversion: pole/MS̄ masses, scale-dependence), a known SM calculation the framework hasn't applied — **NOT** the framework's *intrinsic* ∂_N modular displacement, which is exponentially negligible at the cosmic modular time (s=C_s·e^(−τ/2), τ=log N_hub≈140 ⟹ s≈1e−31). The heavy quarks sit essentially exactly at the band-top today.

**Precision-floor REFRAME (user, 2026-06-22):** a ppb measurement is a *sharper target*, not a reason to
discard it. m_e/m_μ match to 0.008% but are measured far tighter ⟹ **the mass formula is incomplete at the
10⁻⁴ level, and the leptons measure that incompleteness more precisely than anything else.** They are the
best-calibrated NLO residual, NOT "floored away." (The kernel of truth — an exact-rational leading formula
cannot match sub-ppb — does not license discarding the residual; it pins the missing NLO running.)

**Two riders that are NOT this running:** (a) **neutrinos** add a field-extension gate — phase 2π/√7 ∈ ℚ(√7)
(IB discriminant 8−1=7), and Perron c=π² ∉ K=ℚ(√2,√3,√5): is K too small (needs √7), and is band-curvature π²
admissible (Bloch-geometric) vs forbidden (continuum)? (b) **V_ts/V_tb** = the V_cb data's own ~3.3σ
self-disagreement (observational); **Ω_DM/Ω_b** = conditional on the adopted z_eff. Neither is framework
incompleteness.

**Most productive next move:** derive the sub-leading ∂_N run-position correction, **fix `s` from geometry (not
tuned)**, test on the cluster — m_b + M_Z share one mechanism + one `s` ⟹ **3 gaps from one derived quantity**;
then check the same running gives the lepton 10⁻⁴ and the m_t pole-bridge (mass sector closes as a BLOCK). Run
the √7/π² field-extension adjudication in parallel (it gates the Perron closure AND the ν sector).

## The correction to the prior count

The previously-reported "**96 ✅ closed**" used ✅ for *structural* closure (Clauses 1–7 pass), which
**conflated genuine numerical matches with ~12 σ-failers** (m_b, m_t, m_ν2, m_ν3 were ✅ despite >1σ_PDG; M_Z,
m_W already flagged 🟡). The honest semantics split ✅ into:
- **NUM-CLOSED** — derivation rigorous AND ≤1σ_PDG (39).
- **STRUCT-CLOSED** — forced/exact, no direct σ to test (~19).
- **OPEN-GAP** — derivation rigorous, >1σ_PDG, **no established closure** (the conventional floors are falsified/retracted) (12).
- **PRECISION-FLOOR** — relative-matched, σ_PDG unreachable (2: m_e, m_μ).
- **CATEGORY-B** — framework-vs-ΛCDM coasting; tested against the framework-side observation set (8).

## Honest headline

Of the SM-relevant shipping predictions:
- **39 match within 1σ_PDG** (the genuine experimental hits — leptons except the precision-floored pair,
  all CKM except the two unitarity-tension entries, the PMNS angles, η_B, β-birefringence, the light quarks, …).
- **12 are open gaps** — all DERIVED and live in the DAG, but the derived value misses by >1σ_PDG with no *clean* (K-rational, non-tuned) closure for the residual. **The channel/mechanism of each is characterized** (saturation/Perron/band-edge/running/observational/adopted-z_eff — see the table above); what is absent is a disciplined closure. The conventional candidates are dead (2-loop moves m_t/m_b the *wrong* way; the EW-oblique Δr is retracted, non-K-rational), and closing several would require tuning the scale, a non-K-rational c=π², or physical sparticles. So: understood residuals on derived values, **not** unexplained misses.
- **2 are precision-floored** (m_e, m_μ: ~0.008% relative — excellent — but the measurement is sub-ppb, so σ_PDG is unreachable by construction).
- **8 are framework-vs-ΛCDM** (coasting cosmology — the Hubble-tension split is a *prediction*, not a failure).
- **~19 are forced structural/exact** (k*, |V|, N_gen, θ_QCD, Q_Koide, the unmeasured neutrino phases).

This is the count to take public: not "96 closed," but **"39 within-1σ_PDG matches + ~19 forced exact, with
12 understood-floor misses and 2 precision-floored — all stated in honest σ_PDG, none widened."**
