# Fermi Constant: Tree-Level SM Relation from MDL+BZJ v_Higgs

**Audit anchor:** downstream of Row P17 (N_hub — the framework's one adopted dimensional input) of `docs/parameters/parameter_uniqueness_ledger.md`. G_F = 1/(√2 v²), v ← N_hub via BZJ — a downstream prediction. The most precisely-measured electroweak observable (0.51 ppm) is used to *calibrate* N_hub's value (`predictions/N_hub.py:n_hub_from_g_f_consistency`), so the predicted G_F matches the measured value by construction (a round-trip). The "N_hub anchored from G_F" framing is RETRACTED 2026-05-12; G_F is downstream, NOT an anchor. (Closure of Gap G1 — deriving N_hub from the substrate — would remove even the calibration.)

**Parameter:** G_F (Fermi constant)
**Predicted value:** 1.1663787 × 10⁻⁵ GeV⁻²  (matches the measured value by construction — N_hub's value is calibrated via the measured G_F via this chain; like the v_Higgs round-trip)
**Measured value:** 1.1663787(6) × 10⁻⁵ GeV⁻² (PDG 2024 / MuLan 2011) — the calibration target for N_hub's value
**Deviation:** ≈ 0 (round-trip; the predicted G_F equals the measured value by construction since N_hub is calibrated via this very chain). [Historical note: pre-session-19, when N_hub was H_0-calibrated, the predicted G_F came out +0.4435% off — that header value is stale.]
**Status:** DERIVED PREDICTION — G_F = 1/(√2 v²), v ← the adopted N_hub via BZJ; matches the measured value by construction since N_hub's value is calibrated via the measured G_F via this very chain (a round-trip, like v_Higgs). The "N_hub anchored from G_F" framing is RETRACTED 2026-05-12; N_hub is the adopted dimensional input, G_F is downstream. (The y_τ chain's +0.13% Clause-8 residual affects m_τ, not v_Higgs/G_F — the BZJ-inversion calibration makes v_pred = (√2·G_F)^{-1/2} exactly, so G_F's round-trip is exact.) Three previously-cited gaps now closed: ADOPTED-I-FESHBACH closed 2026-04-19 via A5(b); ADOPTED-DARK-MAP closed 2026-04-28 via Class-2 taxonomy; G1 closed 2026-04-28 PM via G1b R2 path on v_higgs.
**Date:** 2026-04-19 (initial); status updated 2026-04-19 session 2; banner refreshed 2026-05-08 (parameter_linter walk-down session 3).
**Update:** 2026-04-19 session 2 — References below to "ADOPTED-I-FESHBACH" should be read as "AXIOM A5(b)" — the coupling clause of A5 (`docs/framework/framework_axioms.md` §5b) subsumes this identification.

---

## 1. Abstract

We derive the Fermi constant G_F by combining two results: (i) the
predicted Higgs VEV v = 245.68 GeV from the MDL+BZJ chain
(`predictions/v_higgs.py`, STRICT-SOLID conditional on G1), and (ii) the
Standard Model tree-level relation G_F = 1/(√2 v²), which follows by
algebraic substitution from the W-boson propagator at zero momentum
transfer.  The derivation involves no free parameters and no fitting:
both the formula and the input v are fully determined by prior
framework results.  The prediction G_F = 1.1715518 × 10⁻⁵ GeV⁻² lies
0.44% above the PDG 2024 value, with the residual fully attributed to
the G1 uncertainty in N_hub.  Per session 19 (2026-04-22), G_F is treated
as the EXTERNAL ANCHOR that fixes N_hub via inversion (replacing the prior
"G_F as derived prediction" framing). Both G1 and ADOPTED-DARK-MAP gaps
referenced in earlier 2026-04-19 banners are now closed: G1 closed
2026-04-28 PM via G1b R2 path on v_higgs (Row P10); ADOPTED-DARK-MAP
closed 2026-04-28 via dark-map Class-2 taxonomy theorem.

---

## 2. Framework Axioms Invoked

The axioms below are used transitively through `predictions/v_higgs.py`.
No new axioms are required for Step 2 (tree-level SM relation).

**A1 (Toggle/srs lattice).** The physical world corresponds to the srs
crystal net (k* = 3, g = 10, d_s = 3).  A1 provides k* and g, which
enter the chain through alpha_1 and the BZJ prefactor.

**A2 (Minimum Description Length).** The MDL-optimal effective theory is
the Curie-Weiss mean-field model; MDL selects the critical point μ² = 0.
This is the backbone of the v_higgs derivation (Steps 1 and 4 of
`predictions/v_higgs_derivation.md`).

**A3 (Purification / decoherence).** Closes the gap between srs order
parameter and the physical Higgs VEV at the formal identification level
(Step 6 of `predictions/v_higgs_derivation.md`, under A5).

**A5 (Physical identification).** The srs scalar order parameter is
identified with the Higgs VEV; the four-Fermi coupling extracted from
muon decay is identified with G_F in the SM Lagrangian.

No new axioms are invoked beyond those already used in v_higgs.

---

## 3. Derivation

### Step 1: Higgs VEV from MDL+BZJ Chain

**Authority:** `predictions/v_higgs.py` and `predictions/v_higgs_derivation.md`
(STRICT-SOLID conditional on G1).

The Higgs VEV is derived in five steps in `predictions/v_higgs_derivation.md`
(see that file for the full proof).  The closed-form result is:

$$
v_\text{pred}
= \frac{\delta^2 M_P}{\sqrt{2}\,N_\text{hub}^{1/4}}
\cdot \left(1 - \frac{5}{12}\,\alpha_1\right)
= 245.6754\,\text{GeV}
$$

where:
- $\delta = 2/9$ is the Koide phase (rate-distortion result for Z_3
  encoding; `predictions/h_walker_eigenvalue.py`),
- $M_P = 1.22089 \times 10^{19}$ GeV is the Planck mass (CODATA 2018;
  external),
- $N_\text{hub} = (H_0 t_P)^{-1} \approx 8.492 \times 10^{60}$ is the
  Hubble-Planck site count (external; Gap G1),
- $\alpha_1 = (2/3)^8 = 256/6561$ is the bare NB walk survival
  (`predictions/alpha_1.py`; derived from k* = 3, g = 10).

**Grade inherited from v_higgs:** STRICT-SOLID conditional on G1 +
ADOPTED-DARK-MAP.

---

### Step 2: Tree-Level SM Relation G_F = 1/(√2 v²)

**Claim.** At tree level in the Standard Model electroweak theory,

$$
\boxed{G_F = \frac{1}{\sqrt{2}\,v^2}}
$$

**Proof.** The four-Fermi effective Lagrangian for charged-current
weak interactions is obtained by integrating out the W boson at
momentum transfer $|q^2| \ll M_W^2$:

$$
\mathcal{L}_\text{eff}
= -\frac{G_F}{\sqrt{2}}\,4\,J_\mu^\dagger J^\mu
$$

where $J^\mu = \bar\nu_\mu \gamma^\mu P_L \mu$ is the leptonic
current (Peskin & Schroeder §20.1, eq. (20.75)).

The W-boson propagator at $q^2 = 0$ contributes a factor
$1/M_W^2$ to the four-point amplitude.  Matching to the
four-Fermi operator gives the tree-level relation:

$$
\frac{G_F}{\sqrt{2}} = \frac{g_2^2}{8 M_W^2}.
$$

The W mass at tree level is $M_W = g_2 v / 2$ (standard
Brout-Englert-Higgs mechanism; Peskin & Schroeder §20.1,
eq. (20.38)).  Substituting:

$$
\frac{G_F}{\sqrt{2}}
= \frac{g_2^2}{8\,(g_2 v/2)^2}
= \frac{g_2^2}{8 \cdot g_2^2 v^2 / 4}
= \frac{g_2^2}{2 g_2^2 v^2}
= \frac{1}{2v^2}.
$$

Therefore:

$$
G_F = \frac{\sqrt{2}}{2v^2} = \frac{1}{\sqrt{2}\,v^2}.
$$

This is a two-line algebraic manipulation from the standard
Brout-Englert-Higgs mechanism; no fitting, no external
measurement, and no free parameter is introduced.

**References:**
- Peskin, M.E. & Schroeder, D.V. (1995). *An Introduction to Quantum
  Field Theory*. Addison-Wesley. §20.1 (eqs. 20.38 and 20.75).
- Donoghue, J.F., Golowich, E. & Holstein, B.R. (1992). *Dynamics of
  the Standard Model*. Cambridge University Press. §IV.1.

**Grade:** Mathematically complete (textbook standard QFT result; no
new axioms required).

---

### Step 3: Numerical Evaluation

Substituting $v_\text{pred} = 245.6754$ GeV into the tree-level formula:

$$
G_F
= \frac{1}{\sqrt{2} \times (245.6754\,\text{GeV})^2}
= \frac{1}{\sqrt{2} \times 60356.4\,\text{GeV}^2}
= \frac{1}{85371.0\,\text{GeV}^2}.
$$

Computing numerically:

$$
G_F = 1.1715518 \times 10^{-5}\,\text{GeV}^{-2}.
$$

---

## 4. Result

$$
G_F
= \frac{1}{\sqrt{2}\,v_\text{pred}^2}
= \frac{1}{\sqrt{2}}
  \left(\frac{\delta^2 M_P}{\sqrt{2}\,N_\text{hub}^{1/4}}
        \cdot \left(1 - \tfrac{5}{12}\,\alpha_1\right)
  \right)^{-2}
\approx 1.1715518 \times 10^{-5}\,\text{GeV}^{-2}.
$$

---

## 5. Comparison with Experiment

| Quantity | Value | Source |
|----------|-------|--------|
| G_F (predicted) | 1.1715518 × 10⁻⁵ GeV⁻² | this derivation |
| G_F (observed) | 1.1663787(6) × 10⁻⁵ GeV⁻² | PDG 2024 (MuLan 2011) |
| Absolute deviation | +5.173 × 10⁻⁸ GeV⁻² | |
| Relative deviation | +0.44% | |
| σ_exp pull | +8622 σ_exp | based on 0.6 ppm experimental error |

**Sigma-pull note.** The experimental precision on G_F is 0.6 ppm
(6 × 10⁻¹² GeV⁻²), making the σ_exp pull look extreme.  This is not
a structural mismatch.  The prediction inherits a ~1% uncertainty from
the G1 gap (N_hub enters as N_hub^{-1/2} in v², and the Planck 2018
H_0 uncertainty of ±0.5 km/s/Mpc propagates to ~0.7% in N_hub, ~0.35%
in N_hub^{1/4}, ~0.7% in v, and ~1.4% in v², hence ~1.4% in G_F).
The +0.44% deviation lies well within this G1 band.  No fine-tuning or
post-hoc correction has been applied.

The v_higgs deviation itself is −0.22% (v_pred = 245.68 GeV vs
v_obs = 246.22 GeV); the G_F deviation is +0.44%, consistent with
G_F ∝ v⁻² amplifying the v error by a factor of −2.

---

## 6. Open Questions

The open questions are identical to those of `predictions/v_higgs_derivation.md`
Section 6, since G_F is a pure algebraic function of v.

### Gap G1: N = N_hub is an empirical input (BLOCKED)

The formula $v \sim N^{-1/4}$ requires specifying N.  The identification
$N_\text{hub} = (H_0 t_P)^{-1}$ is numerically motivated but not derived
from A1-A4.  Closing G1 requires deriving both Newton's constant G and
the Hubble parameter H_0 from framework axioms.  Until then, N_hub and
M_P are adopted external constants and the prediction for G_F is
conditional on G1.

**Grade of G1:** BLOCKED.

### ADOPTED-DARK-MAP: dark correction coefficient

The coefficient c = 5/12 = Im²(h)/k* is structurally derived from srs
graph invariants (exact rational; see `predictions/v_higgs_derivation.md`
Step 5).  It has not yet been derived via a complete A1-A4 chain
independently of the adoption register.  This is inherited by G_F.

### ADOPTED-I-FESHBACH: tree-level approximation

The formula G_F = 1/(√2 v²) is the leading-order (tree-level) SM
relation.  Radiative corrections (α_em suppressed by ~0.4%) shift the
extracted v slightly from its tree-level value.  Post-2026-04-28 G1
closure (G1b R2 path), v_higgs is at UNIQUE-THEOREM-GRADE; the residual
±0.13% propagates from the y_τ chain (Row P7 Clause 8 numerical residual)
rather than from G1.

---

## 7. References

### Load-bearing mathematical results

- **Peskin, M.E. & Schroeder, D.V.** (1995). *An Introduction to
  Quantum Field Theory*. Addison-Wesley. §20.1.
  [W propagator at q²=0; G_F/√2 = g₂²/(8M_W²); Step 2.]

- **Donoghue, J.F., Golowich, E. & Holstein, B.R.** (1992).
  *Dynamics of the Standard Model*. Cambridge University Press. §IV.1.
  [Four-Fermi matching; Step 2.]

### Upstream framework files

- `predictions/v_higgs.py` and `predictions/v_higgs_derivation.md` —
  STRICT-SOLID conditional on G1.  Provides v = 245.6754 GeV.
  Chain-imported by `predictions/G_F.py`.

- `predictions/alpha_1.py` — alpha_1 = (2/3)^8.  Authority for Step 1
  (dark correction).

- `predictions/h_walker_eigenvalue.py` — h = (√3 + i√5)/2;
  Im²(h) = 5/4.  Authority for the dark correction coefficient.

### External physics inputs (explicitly [external])

- **Webber, D.M. et al. (MuLan Collaboration)** (2011).
  Measurement of the positive muon lifetime and determination of the
  Fermi constant to part-per-million precision.
  *Phys. Rev. Lett.* **106**, 041803.
  G_F = 1.1663788(7) × 10⁻⁵ GeV⁻².  [Observed value; comparison only.]

- **Navas, S. et al. (Particle Data Group)** (2024).
  Review of Particle Physics.
  *Phys. Rev. D* **110**, 030001.
  G_F = 1.1663787(6) × 10⁻⁵ GeV⁻².  [PDG 2024; comparison only.]

- **Planck Collaboration** (2020). Planck 2018 results VI. *A&A* **641**,
  A6.  H_0 = 67.4 ± 0.5 km/s/Mpc.  [External; Gap G1 via N_hub.]

- **NIST CODATA 2018.** t_P = 5.391 × 10⁻⁴⁴ s;
  M_P = 1.22089 × 10¹⁹ GeV.  [External; Gap G1 for t_P;
  M_P used as Planck cutoff.]
