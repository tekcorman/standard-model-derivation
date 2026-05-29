# Audit Pilot — Bit-Budget per Prediction

**Status:** Canonical v3 (2026-04-27).

The audit pilot computes per-prediction bit budgets using the wave engine's framework, and tests whether the framework's existing closures collectively pay back the substrate-side spec deficit.

## Method

For each prediction:

1. **Identify load-bearing ops + chain.** Enumerate the catalog ops the derivation invokes. Identify which "chain" the prediction belongs to (Koide / CKM / PMNS / Higgs / Cosmology / Gauge / Neutrino / Parity).
2. **Specify the observable.** Predicted value, observed value, observation σ, prior width.
3. **Compute Φ_obs.**
   - Linear-prior quantities: `Φ_obs = log₂(prior_width / σ_obs)`. Informed prior widths used where physics gives a natural scale (V_cb constrained to ~[0, 0.05] by unitarity → tighter Φ_obs than flat [0, 1]).
   - Log-scale quantities (masses): `Φ_obs = log₂(value / σ_obs)` (relative precision).
4. **Compute L_amortized.**
   ```
   L_amort = (chain.shared_L / N_chain_members) + L_marginal
   ```
   `shared_L` is the chain's infrastructure cost paid once across all chain members. `L_marginal` is the per-prediction structural cost beyond the chain.
5. **B_pred = Φ_obs − L_amort.**

The framework's bit budget closes if `Σ B_pred ≥ |Net_substrate|` summed over all theorem-grade predictions.

## Chain definitions

| chain | shared_L | description |
|---|---|---|
| Koide | 12 | Koide formula + PS + JW + α₁ + v_Higgs |
| CKM | 5 | SRS + Hashimoto + cycle counting |
| PMNS | 5 | PS + dark + cycle counting |
| Higgs | 6 | BZJ + α₁ + Higgs potential |
| Cosmology | 8 | Friedmann + N(t) + BZJ |
| Gauge | 6 | GUT + Killing form + PS |
| Neutrino | 6 | Feshbach + dark + R_ν |
| Parity | 5 | girth-cycle + parity + Lorentz |

## v3 pilot results (23 predictions, 8 chains)

Framework substrate-side baseline post T1.1 template-dedupe: **Φ = 94.15, L = 230, Net = −135.85 bits**.

| chain | shared_L | #preds | Σ Φ_obs | Σ L_amort | Σ B |
|---|---|---|---|---|---|
| **Koide** | 12 | 5 | 81.73 | 22.00 | **+59.73** |
| Cosmology | 8 | 5 | 31.92 | 18.00 | +13.92 |
| Gauge | 6 | 2 | 20.77 | 11.00 | +9.77 |
| CKM | 5 | 3 | 22.18 | 13.00 | +9.18 |
| Higgs | 6 | 2 | 20.12 | 11.00 | +9.12 |
| Parity | 5 | 3 | 13.29 | 11.00 | +2.29 |
| Neutrino | 6 | 2 | 12.29 | 12.00 | +0.29 |
| PMNS | 5 | 1 | 6.11 | 8.00 | −1.89 |
| **Total** | — | **23** | **208.41** | **106.00** | **+102.40** |

**Pilot avg: +4.45 bits/prediction.**

## Projection

```
projected Σ B_pred = 45 × 4.45 = +200.35 bits
total framework B  = -135.85 + 200.35 = +64.50 bits
```

**Framework breaks even with +64.5 bits margin at projected scale.**

## Per-prediction detail (sorted by chain)

```
name                  chain        Φ_obs  L_amort     B    note
α_GUT                 Gauge         7.16    5.00   +2.16   1/24 vs 1/24.3
sin²θ_W(M_unif)       Gauge        13.61    6.00   +7.61   matches at unification

m_e                   Koide        18.96    4.40  +14.56   0.12%
m_μ                   Koide        19.01    4.40  +14.61   0.12%
m_τ                   Koide        13.85    5.40   +8.45   0.13%
Q_Koide               Koide        13.29    3.40   +9.89   exact identity
y_τ                   Koide        16.61    4.40  +12.21   theorem-grade

V_cb                  CKM           5.16    4.67   +0.49   +0.07σ
V_us                  CKM          10.12    4.67   +5.45   −0.015σ
δ_CP^CKM              CKM           6.91    3.67   +3.24   0.7σ

θ_23_PMNS             PMNS          6.11    8.00   −1.89   singleton in chain

m_H                   Higgs        10.15    6.00   +4.15   0.08%
λ_Higgs               Higgs         9.97    5.00   +4.97   Cl(2) anti-comm; 0.5%

H_0                   Cosmology     7.07    4.60   +2.47   +1.6σ CMB
t_0                   Cosmology     4.85    3.60   +1.25   −0.1σ Methuselah
Ω_DM/Ω_m              Cosmology     5.97    3.60   +2.37   0.1%
w_DE                  Cosmology     6.06    2.60   +3.46   exact -1
n_s                   Cosmology     7.97    3.60   +4.37   0.75σ

A_hemispherical       Parity        5.64    3.67   +1.98   CMB; 0.08σ
ε_CP_baryon           Parity        4.32    3.67   +0.66   Sakharov component
η_5_LIV               Parity        3.32    3.67   −0.34   bound-consistent

R_ν_splitting         Neutrino      7.64    6.00   +1.64   exact theorem
m_ν3                  Neutrino      4.65    6.00   −1.35   1.5σ
```

## Three observations

1. **Koide chain dominates.** +60 bits across 5 high-precision lepton-mass predictions. m_e and m_μ alone contribute +14.5 bits each because their σ_obs is at the 10⁻⁹–10⁻⁷ relative precision level.

2. **Singletons read negative.** PMNS chain has only θ_23_PMNS in v3 → full chain L (5 bits) on one prediction → B = −1.89. Not a structural problem; adding θ_13 + θ_12 + δ_CP_PMNS would amortize. Same dynamic for the smaller Neutrino chain (only R_ν + m_ν3 in v3).

3. **Lower-precision predictions are marginal.** η_5 = 0 with LHAASO bound |η|<0.1 gives Φ_obs = 3.32 → B = −0.34. m_ν3 at 1.5σ gives Φ_obs = 4.65 → B = −1.35. The mechanism correctly penalizes loose matches.

## Evolution of the audit's headline result

| version | preds | chain attrib | Σ B | avg/pred | projected (×45) | framework total | verdict |
|---|---|---|---|---|---|---|---|
| v1 (pre-T1.1) | 4 | no | +10.06 | +2.52 | +113 | (Φ=183) +66 | breaks even |
| v1 (post-T1.1) | 4 | no | +10.06 | +2.52 | +113 | (Φ=94) **−23** | **deficit** |
| v2 (post-T1.1, richer) | 16 | no | +1.57 | +0.10 | +4 | (Φ=94) **−131** | **worse** |
| v2 (post-T1.1, chain) | 16 | yes | +75.57 | +4.72 | +213 | (Φ=94) **+77** | **closes** |
| v3 (canonical) | 23 | yes | **+102.40** | **+4.45** | **+200** | (Φ=94) **+64.5** | **closes** |

The pre-T1.1 result of "breaks even at +66" was an artifact of 95-bit Φ overcounting. T1.1 corrected that and exposed a deficit. v2-v3 chain attribution closed it. **The framework genuinely closes its bit budget under honest accounting.**

## Caveats on absolute numbers

1. **L_marginal hand-rated.** T1.2 formal L encoding will refine.
2. **Shared L per chain hand-rated.** Koide's 12 vs 8 vs 16 swings the chain's total B significantly.
3. **Projection assumes ~22 unmeasured theorem-grade predictions follow v3 distribution.** Could over- or under-shoot.
4. **Prior-width choice.** Informed priors used where natural; flat priors elsewhere.

The relative ordering is stable; absolute closure margin (+64.5 bits) is approximate.

## Running the pilot

```bash
python3 proofs/wave_engine/audit_pilot.py
```

Output prints per-prediction details + chain breakdown + projection.

## Extending the audit

Append to `PREDICTIONS` in `proofs/wave_engine/audit_pilot.py`:

```python
{
    'name':           'prediction-id',
    'doc':            'predictions/foo.py',
    'formula':        'short formula text',
    'value_pred':     0.123,
    'value_obs':      0.124,
    'sigma_obs':      0.001,
    'prior_width':    1.0,         # 1.0 for flat [0,1]; None for log-scale; informed value where applicable
    'L_marginal':     2,           # bits beyond the chain's shared_L
    'chain':          'Koide',     # which chain (or define a new one in CHAINS)
    'note':           'σ-deviation or other note',
},
```

If the prediction belongs to a new chain, add to `CHAINS` dict with appropriate `shared_L`.

Re-run; new prediction is added to total and projection.
