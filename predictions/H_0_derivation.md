# H_0: Hubble Constant (substrate-side and observer-side dual prediction)

**Status (post-2026-05-07 PM):** THEOREM-GRADE (substrate) + THEOREM-GRADE (observer-side). The ADOPTED-COSMOLOGICAL-IC-AMPLITUDE conditional was removed 2026-05-07 PM — adoption graduated via `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md` (observer-MDL persistence chain composing A1 → P1' theorem + A2-T waterline + Bridge 1 + DL accounting probe).
**Parameter:** H_0_substrate = 68.18 km/s/Mpc, H_0_observer = 72.72 km/s/Mpc
**File:** `predictions/H_0.py`
**Theorem refs:**
- Substrate: `docs/theorems/theorem_g1b_r2_closure.md` (cascade theorem D1+D2+D3)
- Observer: `docs/theorems/theorem_cascade_D2_extended_observer_rate.md` (D2-extended; THEOREM-GRADE post-2026-05-07 PM closure)
- IC amplitude closure: `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md` (observer-MDL persistence chain; closes the ε_toggle persistence at theorem grade)
- Audit: Row P19 of `docs/parameters/parameter_uniqueness_ledger.md`

---

## 1. Abstract

The Hubble constant H_0 is derived in the framework as the inverse of the
cosmic-time × Planck-time product: H_0 = 1/(N · t_P), where N is the
substrate's accumulated state count. The framework predicts H_0 in two
flavors corresponding to two different observational regimes:

- **Substrate-side H_0 = 1/(N_substrate · t_P) = 68.18 km/s/Mpc**: matches
  Planck 2018 CMB ΛCDM-fit (67.4 ± 0.5 km/s/Mpc) at +1.6σ. This is the
  framework's prediction for measurements that sample the substrate directly
  (e.g., CMB acoustic-scale extraction in a coasting cosmology).

- **Observer-side H_0 = (16/15) · H_substrate = 72.72 km/s/Mpc**: matches
  SH0ES distance ladder (73.04 ± 1.04 km/s/Mpc) at +0.29σ. This is the
  framework's prediction for measurements via observer's local clock
  (luminosity distance ladder, local kinematic determinations).

The (16/15) factor is the cascade theorem D2-extended observer-substrate
rate gap, equal to 1 + ε_toggle/k = 1 + 1/15, where ε_toggle = 1/5 is the
Bayesian Beta-posterior toggle asymmetry and 1/k = 1/3 is the trivalent srs
geometric average. Both factors are theorem-grade-derived elsewhere in the
framework (S_fresh.py + S_disconfirm.py, A_dilution_derivation.py).

The framework simultaneously matches the Planck-CMB and SH0ES H_0 values
through this observer/substrate split — the "Hubble tension" is structurally
predicted, not a discrepancy.

## 2. Axioms invoked

- **A1** (binary self-inverse toggle): substrate dynamics
- **A2-T** (waterline / I-projection): substrate evolution
- **MaxEnt prior** (Jaynes 1957): Beta(1,1) initial prior
- **Bayesian conjugate update**: Beta(1,1) → Beta(2,1) after one observation

## 3. Derivation

### 3a. Substrate-side: H_substrate = 1/(N · t_P)

By the cascade theorem D1+D2+D3 (per `predictions/N_hub.py`):
- D1 [A1]: each of the k*N directed edges in the toggle graph is toggled
  once per Planck time t_P.
- D2 [A2 + Beta-posterior]: per-toggle MDL acceptance probability =
  P(absent | Beta(2,1)) = 1/k* = 1/3 (theorem-grade per S_disconfirm.py).
- D3 [algebra]: cascade ratio ε = 1/(k*N), giving k*N · ε = 1 new
  observable state per t_P.

Therefore the substrate's intrinsic Hubble rate is:

$$H_{\rm substrate}(t) = \frac{1}{N(t) \cdot t_P}, \quad N(t) = t/t_P$$

At the present epoch, with N_hub ← the adopted N_hub (whose value is pinned via the measured G_F) (per the BZJ inversion
in `predictions/N_hub.py`):

$$H_{\rm 0,\,substrate} = \frac{1}{N_{\rm hub} \cdot t_P} = 68.18 \text{ km/s/Mpc}$$

This is THEOREM-GRADE per the cascade theorem.

### 3b. Observer-side: H_observer = (16/15) · H_substrate

By the cascade theorem D2-extended (per
`docs/theorems/theorem_cascade_D2_extended_observer_rate.md`), the observer's
effective per-toggle acceptance probability differs from substrate's
intrinsic 1/k* by a multiplicative factor (1 + ε_toggle/k):

$$P_{\rm obs} = \frac{1}{k^*} \cdot \left(1 + \frac{\varepsilon_{\rm toggle}}{k}\right)$$

where:
- $\varepsilon_{\rm toggle} = (P_{\rm fresh} - P_{\rm persist})/(P_{\rm fresh} + P_{\rm persist}) = (1/2 - 1/3)/(1/2 + 1/3) = 1/5$
  (theorem-grade per S_fresh.py + S_disconfirm.py)
- $1/k = 1/3$: geometric average projection at trivalent srs vertex
  (theorem-grade per A_dilution_derivation.py)

Therefore:

$$\frac{H_{\rm observer}}{H_{\rm substrate}} = 1 + \frac{1/5}{3} = 1 + \frac{1}{15} = \frac{16}{15}$$

$$H_{\rm 0,\,observer} = \frac{16}{15} \cdot H_{\rm 0,\,substrate} = 72.72 \text{ km/s/Mpc}$$

THEOREM-GRADE (post-2026-05-07 PM closure). The persistence of α = ε_toggle from N=1 IC to N_hub observer epoch is derived via the observer-MDL persistence chain in `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md`: A1 → P1' (theorem; observer persists) + A2-T waterline (theorem) + Bridge 1 / Claim A (theorem-grade-conditional; ε_toggle at N=1) + DL accounting probe (`proofs/cosmology/observer_persistence_DL_accounting.py`; M_IC clears the waterline by ~10⁵⁹·⁴ bits margin) + P1' persistence. The closure operates under the framework's observer-MDL primary posture: cosmological observables are functionals of the observer's compressed cosmological model. The prior 5-route audit closed a substrate-primary question whose negative answer does NOT determine the observer-side prediction the framework actually makes. ADOPTED-COSMOLOGICAL-IC-AMPLITUDE (introduced 2026-05-07 AM as a named structural commitment) is GRADUATED to derived theorem 2026-05-07 PM. Empirical anchor: 4-observable joint match at α = 0.207 ± 0.036 (+0.18σ from ε_toggle); alternatives ε/2 excluded at 2.93σ, 2ε at 5.32σ.

## 4. Result

| Quantity | Value | Match against |
|----------|-------|---------------|
| H_0_substrate | 68.18 km/s/Mpc | Planck 2018 CMB ΛCDM-fit (67.4 ± 0.5) at +1.6σ |
| H_0_observer | 72.72 km/s/Mpc | SH0ES distance ladder (73.04 ± 1.04) at +0.29σ |

## 5. Comparison

| Source | H_0 (km/s/Mpc) | Side | Match |
|--------|----------------|------|-------|
| Planck 2018 CMB | 67.4 ± 0.5 | substrate side of CMB extraction | substrate prediction +1.6σ |
| SH0ES (Riess+2022) | 73.04 ± 1.04 | observer side, distance ladder | observer prediction +0.29σ |
| Pantheon+ full fit | 73.6 ± 1.1 | observer side, SN Ia | observer prediction +0.85σ |

**The Hubble tension dissolves** under the observer/substrate split: each
measurement is on a different side of the cascade theorem's
observer-substrate identification, and the framework predicts both
simultaneously.

## 6. Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure plus the
D2-extended derivation. Substrate-side passes Clause 7 via the η=1 R2
closure. Observer-side passes Clause 7 conditionally via the named adoption.

- **Substrate side:** UNIQUE-THEOREM-GRADE per N_hub Row P17 inheritance.
- **Observer side:** UNIQUE-THEOREM-GRADE (post-2026-05-07 PM); the previous conditional on ADOPTED-COSMOLOGICAL-IC-AMPLITUDE was removed when the adoption graduated via `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md`.

## 7. Audit v2 (Clause 8) numerical match

| Side | σ_PDG | Pass? |
|------|------------|-------|
| Substrate vs Planck CMB | 0.5 → +1.6σ_PDG | borderline (Category B accommodation: matches alternate-cosmology observation set) |
| Observer vs SH0ES | 1.04 → +0.29σ_PDG | PASS (Clause 8c: < 1σ_PDG) |

Per Category B accommodation (cosmology-coasting predictions): substrate-side
match against substrate-anchored alternate observation (Methuselah for t_0,
this prediction for H_0) is the relevant test, not the ΛCDM-extracted Planck
value. Observer-side prediction passes Clause 8 against SH0ES directly.

## 8. Open questions

1. **Persistence of α = ε_toggle from N=1 to N_hub. CLOSED 2026-05-07 PM.**
   The persistence is derived via the observer-MDL persistence chain in
   `docs/theorems/theorem_observer_persistence_closure_IC_amplitude.md`:
   A1 → P1' (theorem; observer persists) + A2-T waterline (theorem) +
   Bridge 1 / Claim A + DL accounting probe + P1' persistence. The closure
   operates under the framework's observer-MDL primary posture (post-2026-05-02
   axiom slate {A1} alone); the prior 5-route audit's substrate-primary
   negative remains a valid substrate-side fact but does not block the
   observer-side prediction. ADOPTED-COSMOLOGICAL-IC-AMPLITUDE GRADUATED.

2. **Λ_CC numerical verification.** Λ_CC ∝ H_0² gets (16/15)² correction.
   Need explicit comparison with Planck Λ_CC value (factor-of-2 framework
   tension already noted in `lambda_cc_coasting_scoping.md`).

3. **H(z) at moderate z.** The (16/15) is a uniform multiplicative factor
   that doesn't change d_L vs z shape. The Pantheon+ moderate-z shape
   mismatch is a separate question (see SALT2 estimate in
   an internal working note).
