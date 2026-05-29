# Derivation of $A_{\rm hemispherical}$ — UNIQUE-THEOREM-GRADE

**Status (post-2026-05-07 PM):** UNIQUE — THEOREM-GRADE. The composition rule $A = \varepsilon_{\rm toggle} \cdot \langle (\hat{e} \cdot \hat{z})^2 \rangle$ identifies the cosmological preferred-axis amplitude with $\varepsilon_{\rm toggle}$. The persistence of $\alpha = \varepsilon_{\rm toggle}$ from N=1 IC to N_hub observer epoch was a named adoption (ADOPTED-COSMOLOGICAL-IC-AMPLITUDE, introduced 2026-05-07 AM) and is now derived at theorem grade via `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md` (the observer-MDL persistence chain composing A1 → P1' theorem + A2-T waterline + Bridge 1 + DL accounting probe). The closure operates under the framework's observer-MDL primary posture (post-2026-05-02 axiom slate {A1} alone): cosmological observables are functionals of the observer's compressed cosmological model. The identification of $1/15$ with the CMB sky observable retains an OTHER-SMUGGLE qualitative flag (cosmological-observable identification, separate from the structural-value side; not affected by the 2026-05-07 PM closure).
**Result:** $A_{\rm hemispherical} = 1/15 \approx 0.0667$.
**Closure date:** Pre-existing closure; paired derivation md added 2026-04-29; structural status revised 2026-05-07 AM (named adoption introduced) and 2026-05-07 PM (adoption graduated to theorem grade via observer-MDL persistence chain).

---

## Abstract

The CMB hemispherical power asymmetry $A$ is a Planck 2018 anomaly: temperature-fluctuation power on one hemisphere of the sky is observed to exceed that on the other by $A = 0.07 \pm 0.02$ at low multipoles ($\ell_{\max} = 64$). The Standard Model + $\Lambda$CDM has no derivation of $A$ — it appears as an unexplained large-scale asymmetry, sometimes invoked as evidence for "anomalous" inflation or for cosmic-defect models.

The framework derives $A = 1/15$ as the product of two structural ingredients: a **Bayesian-toggle posterior asymmetry** $\varepsilon_{\rm toggle} = 1/5$ (probability axioms, no physics imported) and a **srs cubic-moment geometric factor** $\langle (\hat{e} \cdot \hat{z})^2 \rangle = 1/k^* = 1/3$ (theorem-grade per `predictions/srs_cubic_moment.py` at $n = 1$). The product $A = (1/5)(1/3) = 1/15$ matches Planck at $+0.17\sigma$.

Both ingredients are theorem-grade. The non-trivial content is the Bayesian derivation of $\varepsilon = 1/5$ as a *unique* posterior-asymmetry value forced by Beta-conjugate updating, plus the Class D / cubic-moment composition.

---

## Framework axioms invoked

- **A1** (binary toggle): toggle events on directed edges form Bernoulli-like trials with binary outcomes; Bayesian inference applies.
- Row 4 of `docs/audits/registers/uniqueness_ledger.md`: $k^* = 3$.
- Row 6 of `docs/audits/registers/uniqueness_ledger.md`: srs lattice with $I4_132$ space group → cubic moment $\langle (\hat{e} \cdot \hat{z})^2 \rangle = 1/k^*$ at $n = 1$.
- Cited result (Type 3): Gelman et al., *Bayesian Data Analysis* Ch. 2 — Beta-conjugate prior/posterior arithmetic.

---

## Derivation

### Step 1 — Bayesian setup [Type 1 + Type 3]

Toggle events on each directed edge are binary outcomes (toggled vs not). Under uniform prior Beta(1, 1) on the per-edge toggle probability $p$, after a single confirmation event (a toggle is observed), the posterior is Beta(2, 1) (Gelman et al. *BDA* Ch. 2 conjugate update).

Two probabilities follow:
- **Creation probability** (next toggle event happens): $P_{\rm create} = \int_0^1 p \cdot \text{Beta}(p; 1, 1)\, dp = 1/2$ (uniform-prior expected value of $p$).
- **Disruption probability** (next event is opposite to the established trend): $P_{\rm disrupt} = \int_0^1 (1-p) \cdot \text{Beta}(p; 2, 1)\, dp = 1/3$ (post-confirmation expected value of $1 - p$).

### Step 2 — Posterior asymmetry $\varepsilon_{\rm toggle} = 1/5$ [Type 1 + upstream Type 4]

The asymmetry between creation and disruption is the unique scalar invariant of the (creation, disruption) pair under linear normalization to $[-1, 1]$:

$$\varepsilon_{\rm toggle} = \frac{P_{\rm create} - P_{\rm disrupt}}{P_{\rm create} + P_{\rm disrupt}} = \frac{1/2 - 1/3}{1/2 + 1/3} = \frac{1/6}{5/6} = \frac{1}{5}.$$

This is probability-axioms arithmetic on top of the two theorem-grade upstream pieces $P_{\rm fresh} = 1/2$ (`predictions/S_fresh.py`) and $P_{\rm persist} = 1/3$ (`predictions/S_disconfirm.py`); no physics is imported and no geometric or observable-channel factor enters. Equivalently, the Bayesian-posterior ratio for "creation" given the (create, disrupt) outcome alphabet is:

$$p_{\rm creation} = \frac{P_{\rm create}}{P_{\rm create} + P_{\rm disrupt}} = \frac{1/2}{5/6} = \frac{3}{5}, \qquad \varepsilon = 2 p_{\rm creation} - 1 = \frac{1}{5}.$$

The standalone substrate-primitives derivation is captured in `proofs/foundations/epsilon_toggle_substrate_derivation.py` (CAS-exact via `Fraction` and `sympy`; cross-imports `S_fresh.py` and `S_disconfirm.py` to verify $2^{-S_{\rm fresh}} = 1/2$ and $2^{-S_{\rm disconfirm}} = 1/3$ at machine precision). The same $\varepsilon_{\rm toggle} = 1/5$ propagates to two other observable channels with different geometric factors:

- $|\varepsilon_{\rm CP}| = \varepsilon_{\rm toggle} = 1/5$ (Row P28; no geometric factor).
- Cascade D2-extended rate-gap $H_{\rm obs}/H_{\rm sub} = 1 + \varepsilon_{\rm toggle}/k^* = 16/15$ (Rows P19/P20; same chiral-cubic geometric factor as A_hemis but composed additively on the cascade rate, conditional on the inheritance coefficient $c = 1$ in $\alpha = c \cdot \varepsilon_{\rm toggle}$ — Route 4 open per `proofs/cosmology/cascade_step5_amplitude_via_A_dilution.py`). The $c = 1$ conditional applies only to the cascade D2-ext rate-gap; A_hemis and $\varepsilon_{\rm CP}$ apply $\varepsilon_{\rm toggle}$ directly without that step.

### Step 3 — Geometric factor from srs cubic moment [Type 4]

The CMB hemispherical asymmetry projects onto the "preferred axis" $\hat{z}$ via the cubic moment of the srs edge orientation:

$$\langle (\hat{e} \cdot \hat{z})^2 \rangle = \frac{1}{k^*}.$$

Per `predictions/srs_cubic_moment.py` (theorem-grade, Row P53 of the parameter ledger): for $n = 1$ (linear cubic moment), the average over the $I4_132$ orbit of edge directions $\hat{e}$ gives $1/k^*$. With $k^* = 3$, the geometric factor is $1/3$.

This is a structural property of srs's $C_3$ symmetry on directed edges; no fit, no observable input.

### Step 4 — Composition [Type 1]

The hemispherical asymmetry is the product of the Bayesian asymmetry (toggle creation/disruption imbalance) and the geometric projection onto $\hat{z}$:

$$A_{\rm hemispherical} = \varepsilon_{\rm toggle} \cdot \langle (\hat{e} \cdot \hat{z})^2 \rangle = \frac{1}{5} \cdot \frac{1}{k^*} = \frac{1}{5 k^*}.$$

With $k^* = 3$: $A = 1/15$.

---

## Result

$$\boxed{A_{\rm hemispherical} = \frac{1}{5 k^*} = \frac{1}{15} \approx 0.0667.}$$

---

## Comparison with experiment

- Planck 2018 VII (A&A 641, A7), $\ell_{\max} = 64$: $A = 0.07 \pm 0.02$.
- Framework prediction: $A = 1/15 \approx 0.0667$.
- Deviation: $+0.17\sigma$.

The prediction has zero theoretical uncertainty for the structural value $1/15$. The $0.17\sigma$ residual is dominated by the Planck 2018 measurement uncertainty.

---

## Open questions

### 1. Identification of the structural value with the CMB sky observable

The framework derives $1/15$ as a structural amplitude — a probability-times-geometric-factor on the substrate. The identification of this amplitude with the *Planck 2018 power-asymmetry estimator* (a specific likelihood maximization over CMB temperature maps with a chosen multipole cut) is a step external to the framework's derivation chain. This step uses standard cosmological-observable theory (statistical isotropy + dipole modulation models per Planck collaboration analysis) and is not derived from A1+A2+A3+A4.

In the framework's internal strict-gating audit (referenced in the .py file note as §4.1), this identification is flagged as an "OTHER-SMUGGLE" step — the structural value is theorem-grade, but the connection to the *specific* Planck observable inherits its rigor from cosmological ML theory rather than from framework axioms. This is the standard situation for any cosmological observable; the flag is a transparency note, not a defect.

### 2. Class A spectral cross-route — k=3 numerical coincidence

Per `docs/theorems/theorem_class_A_audit.md` (2026-04-28 audit), $A_{\rm hemispherical}$ also admits a Class A spectral identification via the Hashimoto Γ-spectrum. The Class A formula and the Class D Bayesian formula above agree at $k^* = 3$ but diverge for other $k$. This is therefore a **k=3 numerical coincidence**, not an algebraic cross-class unification. The Class D Bayesian derivation (this document) is the *primary* route; the Class A spectral consistency does not provide independent corroboration beyond the agreement-at-$k=3$ that follows from Row 4 anyway.

This is a transparency note: claiming the spectral cross-route as independent confirmation would double-count Row 4. The primary derivation stands.

### 3. Same Bayesian asymmetry as $\varepsilon_{\rm CP}$

The Bayesian-toggle asymmetry $\varepsilon_{\rm toggle} = 1/5$ is the *same* quantity as the per-process baryon-CP asymmetry $\varepsilon_{\rm CP}$ (Row P28). The two observables — CMB hemispherical power and baryon-CP asymmetry — share a single structural origin (the Beta(1,1)→Beta(2,1) Bayesian update), differing only in the geometric factor that converts to the relevant observable. This is a non-trivial cross-prediction connection: agreement of both with experiment is partial confirmation of the Bayesian-toggle setup.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-2026-05-07 PM):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure only. The previous additional conditional on ADOPTED-COSMOLOGICAL-IC-AMPLITUDE is REMOVED — the adoption graduated to derived theorem 2026-05-07 PM via `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md` (the observer-MDL persistence chain composes A1 → P1' theorem + A2-T waterline + Bridge 1 + DL accounting probe). The composition rule's persistence is now theorem-grade.
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL. (ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED 2026-05-07 PM — no longer a deferred conditional.)
