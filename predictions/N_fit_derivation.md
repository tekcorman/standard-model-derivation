# Derivation of N_fit — weighted least-squares toggle graph node count

**Audit anchor:** Alternative anchoring for Row P17 (N_hub) of `docs/parameters/parameter_uniqueness_ledger.md`. Aggregates multiple observables to estimate N; superseded by the G_F-calibration of N_hub's value (Session 19, 700× precision improvement; N_hub is the adopted input). Retained as historical / cross-check.

## Abstract

The toggle graph node count $N$ is the single scale parameter that enters
the BZJ formula for the Higgs VEV:

$$
v \;=\; \frac{\delta^2 M_P}{\sqrt{2}\, N^{1/4}} \times \left(1 - \tfrac{5}{12}\alpha_1\right).
$$

This file performs a weighted least-squares combination of every independent
observable that constrains $N$ through this formula.  The result is a
best-fit $N_{\rm fit}$ with propagated uncertainty, a residuals table, and
the implied predicted values of $H_0$, $v$, $G_F$, and $m_H$ at the
best-fit node count.

**Grade:** identification (combines adopted $N$ estimates from multiple
observables; not a new prediction from first principles).

**Upstream chain imports:** `predictions/N_hub.py`, `predictions/alpha_1.py`,
`predictions/h_walker_eigenvalue.py`, `predictions/lambda_higgs.py`,
`predictions/v_higgs.py`, `predictions/G_F.py`, `predictions/m_H.py`.

---

## Framework axioms invoked

The BZJ formula inherits the full upstream chain:

- **(A1)** Binary self-inverse toggle — `predictions/p_toggle.py`.
- **(A2)** MDL compression — `predictions/d_spatial.py`, `predictions/k_star.py`.
- Upstream theorems: $k^* = 3$, $d = 3$, srs, $g = 10$, $\alpha_1 = (2/3)^8$,
  $\delta = 2/9$, $\lambda = 2560/19683$.
- **ADOPTED-DARK-MAP**: dark correction coefficient $5/12 = \operatorname{Im}^2(h)/k^*$
  from `dark_correction_theorem_2026-04-14.md` §4c.5b.
- **ADOPTED-I-FESHBACH**: $\alpha_1$ as physical Feshbach coupling.
- **Gap G1**: $N = N_{\rm hub}$ requires deriving $H_0$ (and $G$) from
  A1–A4; same wall as Newton's $G$ and $\Lambda_{\rm CC}$.

---

## THEOREM STATEMENT

Given the BZJ inversion

$$
N_i \;=\; \left(\frac{\delta^2 M_P\, \mathrm{dark}}{\sqrt{2}\; v_i}\right)^4,
\qquad \mathrm{dark} = 1 - \tfrac{5}{12}\,\frac{\alpha_1}{1-\alpha_1},
$$

and error propagation $\sigma_{N_i} = 4\, N_i\, (\sigma_{v_i}/v_i)$,
the weighted least-squares estimate is

$$
N_{\rm fit} = \frac{\sum_i w_i N_i}{\sum_i w_i},
\qquad w_i = \sigma_{N_i}^{-2},
\qquad \sigma_{\rm fit} = \left(\sum_i w_i\right)^{-1/2},
\qquad \chi^2 = \sum_i w_i (N_i - N_{\rm fit})^2.
$$

---

## INPUTS

| Symbol | Value | Status | Source |
|--------|-------|--------|--------|
| $\delta$ | $2/9$ | derived | `predictions/h_walker_eigenvalue.py` |
| $\alpha_1$ | $(2/3)^8$ | derived | `predictions/alpha_1.py` |
| $M_P$ | $1.22089\times10^{19}$ GeV | external | CODATA 2018; Gap G1 |
| $t_P$ | $5.391247\times10^{-44}$ s | external | CODATA 2018 |
| $\lambda$ | $2560/19683$ | derived (UNIQUE-THEOREM-GRADE 2026-04-29) | `predictions/lambda_higgs.py` |
| $G_F$ | $1.1663787\times10^{-5}$ GeV$^{-2}$ | measured (calibrates N_hub's value) | PDG 2024 / MuLan 2011 |
| $\sigma_{G_F}$ | $6\times10^{-12}$ GeV$^{-2}$ | measured (calibrates N_hub's value) | MuLan 2011 (0.51 ppm) |
| $H_0^{\rm CMB}$ | $67.4$ km/s/Mpc | external | Planck 2018, arXiv:1807.06209 |
| $\sigma_{H_0}^{\rm CMB}$ | $0.5$ km/s/Mpc | external | Planck 2018 |
| $H_0^{\rm ladder}$ | $73.0$ km/s/Mpc | external | Riess et al. 2022, arXiv:2112.04510 |
| $\sigma_{H_0}^{\rm ladder}$ | $1.0$ km/s/Mpc | external | Riess et al. 2022 |
| $m_H$ | $125.20$ GeV | external | PDG 2025 (ATLAS + CMS Run-2 average) |
| $\sigma_{m_H}$ | $0.11$ GeV | external | PDG 2025 |

---

## IMPLEMENTATION

### Step 1. BZJ inversion for each observable

The formula $v = \delta^2 M_P\, \mathrm{dark} / (\sqrt{2}\, N^{1/4})$ is
inverted:

$$
N = \left( \frac{\delta^2 M_P\, \mathrm{dark}}{\sqrt{2}\, v} \right)^4.
$$

Logarithmic error propagation: since $N \propto v^{-4}$,

$$
\frac{\sigma_N}{N} = 4\, \frac{\sigma_v}{v}.
$$

Each observable is converted to an effective VEV $v_i$ as follows.

**Observable 1 — $G_F$ (MuLan/PDG 2024, 0.51 ppm).**
The Standard Model tree-level relation $G_F = 1/(\sqrt{2}\, v^2)$ gives

$$
v = \frac{1}{\sqrt{\sqrt{2}\, G_F}}, \qquad
\frac{\sigma_v}{v} = \frac{1}{2}\frac{\sigma_{G_F}}{G_F}.
$$

The $G_F$ uncertainty is 0.51 ppm, so $\sigma_v/v \approx 0.26$ ppm and
$\sigma_N/N = 4 \times 0.26\;\mathrm{ppm} \approx 1.03$ ppm.

*Reference:* Webber et al. (MuLan), Phys. Rev. Lett. **106**, 041803 (2011);
PDG 2024, Navas et al., Phys. Rev. D **110**, 030001 (2024).

**Observable 2 — $H_0$ (Planck 2018 CMB).**
$N = 1/(H_0\, t_P)$ directly from the Hubble-Planck identification
(`predictions/N_hub.py`), so $v$ is not an intermediate here; the BZJ
formula is inverted from $v = \delta^2 M_P\, \mathrm{dark} / (\sqrt{2}\, N^{1/4})$.
Since $N \propto H_0^{-1}$:

$$
\frac{\sigma_N}{N} = \frac{\sigma_{H_0}}{H_0} = \frac{0.5}{67.4} \approx 0.742\%.
$$

*Reference:* Planck Collaboration 2018, arXiv:1807.06209, Table 2.

**Observable 3 — $H_0$ (distance ladder).**
Same formula as Observable 2, but with the Riess et al. value:

$$
\frac{\sigma_N}{N} = \frac{1.0}{73.0} \approx 1.37\%.
$$

This row is included explicitly to make the $H_0$ tension visible in the
residuals table.

*Reference:* Riess et al. 2022, arXiv:2112.04510 (SH0ES collaboration).

**Observable 4 — $m_H$ (LHC, $\lambda$-contaminated).**
The tree-level relation $m_H = \sqrt{2\lambda}\, v$ gives

$$
v = \frac{m_H}{\sqrt{2\lambda}}, \qquad
\frac{\sigma_v}{v} = \frac{\sigma_{m_H}}{m_H}.
$$

Experimental uncertainty only: $\sigma_N/N = 4\,\sigma_{m_H}/m_H \approx 0.35\%$.
The $\lambda = 2560/19683$ adoption (ADOPTED-I-FESHBACH + ADOPTED-DARK-MAP)
carries an additional systematic not captured by this row's $\sigma_N$.

*Reference:* PDG 2025 Higgs boson review; ATLAS-CONF-2023-037; CMS Run-2.

**Note on $v_{\rm direct}$ (246.22 GeV, PDG).**
The PDG value $v = 246.22 \pm 0.12$ GeV is derived from $G_F$ via
$v = 1/\sqrt{\sqrt{2}\, G_F}$; it is not independent and is not included
as a separate row.

---

### Step 2. Weighted least-squares combination

With weights $w_i = 1/\sigma_{N_i}^2$:

$$
N_{\rm fit} = \frac{\sum_i w_i N_i}{\sum_i w_i},\quad
\sigma_{\rm fit} = \left(\sum_i w_i\right)^{-1/2},\quad
\chi^2 = \sum_i w_i(N_i - N_{\rm fit})^2.
$$

Because $G_F$ is measured to 0.51 ppm while $H_0$ and $m_H$ are measured to
0.74%–1.4%, the $G_F$ weight dominates by a factor of roughly
$(0.0074 / 0.0000051)^2 \approx 2\times10^6$ over the CMB $H_0$ row.
The fit is therefore entirely dominated by the $G_F$ constraint.

---

### Step 3. Implied predictions at $N_{\rm fit}$

Given $N_{\rm fit}$, the implied values are:

$$
H_0^{\rm pred} = \frac{1}{N_{\rm fit}\, t_P},\qquad
v_{\rm pred} = \frac{\delta^2 M_P\, \mathrm{dark}}{\sqrt{2}\, N_{\rm fit}^{1/4}},\qquad
G_F^{\rm pred} = \frac{1}{\sqrt{2}\, v_{\rm pred}^2},\qquad
m_H^{\rm pred} = \sqrt{2\lambda}\, v_{\rm pred}.
$$

These are self-consistency checks: $G_F^{\rm pred}$ trivially matches the
input $G_F$ (since the fit is dominated by that constraint), while
$H_0^{\rm pred}$ and $m_H^{\rm pred}$ reflect what the adopted $N$ (value calibrated via the measured $G_F$)
implies for those observables.

---

## Numerical results

Computed values (see `predictions/N_fit.py`):

| Observable | $N_i$ (×$10^{60}$) | $\sigma_N/N$ | Weight fraction |
|------------|-------------------|--------------|-----------------|
| $G_F$ (MuLan) | 8.4175 | 1.03 ppm | ~100% |
| $H_0$ (Planck CMB) | 8.4918 | 0.742% | ~0% |
| $H_0$ (distance ladder) | 7.8404 | 1.37% | ~0% |
| $m_H$ (LHC) | 8.5194 | 0.351% | ~0% |

**Best-fit result:**

$$
\boxed{N_{\rm fit} = (8.4175 \pm 0.0000087) \times 10^{60}}
$$

The uncertainty is $\sigma_N/N \approx 1.03$ ppm, propagated from
$\sigma_{G_F}/G_F$ via $\sigma_N/N = 2\,\sigma_{G_F}/G_F$.

**Chi-squared:** $\chi^2 = 41.9$ on 3 d.o.f., reflecting genuine tension
among the inputs.

**Residuals:**

| Observable | Pull $(\sigma)$ | Interpretation |
|------------|-----------------|----------------|
| $G_F$ | $-0.001\,\sigma$ | trivially consistent (dominant weight) |
| $H_0$ (CMB) | $+1.2\,\sigma$ | 1.2σ high relative to the G_F-calibrated $N$ |
| $H_0$ (dist. lad.) | $-5.4\,\sigma$ | $H_0$ tension visible here |
| $m_H$ | $+3.4\,\sigma$ | framework $\lambda$ predicts $m_H$ 3.4σ high |

The $m_H$ pull of $+3.4\,\sigma$ matches the level of discrepancy already
visible in `predictions/m_H.py` ($+0.91\,\sigma$ experimental, but the
dominant uncertainty is G1, not the experimental $\sigma_{m_H}$).  The
$3.4\,\sigma$ here uses $\sigma_{N_{m_H}}$ which is derived from the
experimental $\sigma_{m_H}$ only; a full systematic including $\lambda$
adoption would widen the band and reduce this pull.

**Implied values at $N_{\rm fit}$:**

| Quantity | Predicted | Observed | Comment |
|----------|-----------|----------|---------|
| $H_0^{\rm pred}$ | 68.0 km/s/Mpc | $67.4 \pm 0.5$ | 1.2σ high (CMB-consistent) |
| $v_{\rm pred}$ | 246.22 GeV | $246.22 \pm 0.12$ | trivially consistent |
| $G_F^{\rm pred}$ | $1.1664\times10^{-5}$ GeV$^{-2}$ | $1.1664\times10^{-5}$ | matches by construction — N_hub's value is calibrated via the measured G_F |
| $m_H^{\rm pred}$ | 125.58 GeV | $125.20 \pm 0.11$ | 3.4σ high (λ-sys) |

---

## Discussion

### Why $G_F$ dominates

The MuLan experiment measured $G_F$ to 0.51 ppm — roughly $10^4$ times
more precise than the Planck CMB $H_0$ (0.74%) and $3\times10^3$ times
more precise than the $m_H$ determination (0.35%).  In a weighted fit the
$G_F$ row carries essentially all the weight.  The other rows serve as
consistency cross-checks, not as constraints.

### The $H_0$ tension

The CMB $H_0 = 67.4 \pm 0.5$ km/s/Mpc and distance-ladder
$H_0 = 73.0 \pm 1.0$ km/s/Mpc imply

$$
N_{\rm CMB} = 8.492\times10^{60}, \quad N_{\rm ladder} = 7.840\times10^{60}.
$$

Relative to the G_F-calibrated $N = 8.418\times10^{60}$, the CMB
value is $+1.2\,\sigma_{N_{\rm CMB}}$ away (consistent) while the distance
ladder value is $-5.4\,\sigma_{N_{\rm ladder}}$ away (the $H_0$ tension in
node-count language).  The framework itself has no current mechanism to
prefer one over the other; the tension is an open cosmological problem.

### The $m_H$ residual and the $\lambda$ systematic

The $m_H$ row implies $N_{m_H} = 8.519\times10^{60}$, which is $+3.4\,\sigma$
above $N_{\rm fit}$.  Equivalently, the framework's prediction
$m_H = \sqrt{2\lambda}\, v_{\rm pred} = 125.58$ GeV is $+0.38$ GeV
above the PDG central value of $125.20$ GeV ($+3.4\,\sigma_{\rm exp}$
using $\sigma_{m_H} = 0.11$ GeV).  This discrepancy absorbs the
ADOPTED-I-FESHBACH and ADOPTED-DARK-MAP uncertainties in $\lambda$; a
shift of $\delta\lambda/\lambda \approx +0.3\%$ in the Higgs quartic
coupling would remove the tension.

### Status: identification, not prediction

$N_{\rm fit}$ is a weighted average of four empirical estimates of the same
adopted scale anchor $N_{\rm hub}$.  It does **not** constitute a
first-principles derivation of $N$ from A1–A4.  The correct classification
is **identification** (analogous to $G_{\rm Newton}$ as a scale anchor),
pending the closure of Gap G1 (derivation of $H_0$ from A1–A4 via
$\Lambda_{\rm CC}$).

---

## References

- Webber, D. M. et al. (MuLan Collaboration) (2011). *Phys. Rev. Lett.*
  **106**, 041803.  $G_F$ at 0.6 ppm.
- PDG 2024: Navas et al., *Phys. Rev. D* **110**, 030001 (2024).
- Planck Collaboration (2018). *Astron. Astrophys.* **641**, A6 (2020);
  arXiv:1807.06209.  CMB $H_0 = 67.4 \pm 0.5$ km/s/Mpc.
- Riess, A. G. et al. (SH0ES) (2022). *Astrophys. J.* **934**, L7;
  arXiv:2112.04510.  Distance-ladder $H_0 = 73.04 \pm 1.04$ km/s/Mpc.
- PDG 2025 Higgs review; ATLAS-CONF-2023-037; CMS Run-2 combined.
- `predictions/N_hub.py` — $N_{\rm hub}$ adoption.
- `predictions/v_higgs.py` — BZJ formula and dark correction.
- `predictions/G_F.py` — $G_F$ prediction chain.
- `predictions/m_H.py` — $m_H$ prediction chain.
- `predictions/lambda_higgs.py` — $\lambda = 2560/19683$ derivation.
- `docs/honest_assessment.md` — Gap G1 and status framework.
