# Theorem: Unified spectral dark structure of the substrate

**Status:** Theorem-grade. Synthesizes four independently-derived dark/visible coefficients of the framework into a single spectral picture on the substrate's Hashimoto operator B(k) at Γ.

**Written:** 2026-04-28.

## Statement (revised post-audit 2026-04-28)

**Four** framework constants — q_NB, α_1_bare, α_1_full (= V_cb), and c (= 5/12) — are spectral observables of the substrate's Hashimoto operator B at Γ that **algebraically unify** with their primary non-spectral derivations: the spectral and non-spectral routes give the **same formula in (k, |V|, |E|, g)**. These are theorem-grade structural over-determinations.

**Two** additional framework constants — ε_CP and A_hemispherical — have spectral observables that **coincidentally agree** with their primary (Bayesian, Class D) derivations *only at k = 3*. For k ≠ 3, the spectral and Bayesian formulas diverge. These are Class A in a taxonomic sense but the spectral identification is a numerical coincidence specific to the framework's k* = 3, not an algebraic unification.

See `theorem_class_A_audit.md` for the audit details.

For srs (Wyckoff 8a, |V|=4, |E|=6, k*=3, girth g=10):

### Algebraically unified (theorem-grade structural over-determination)

| coefficient | spectral identification | value | algebraic unity? |
|---|---|---|---|
| q_NB | λ_max(B) / λ_max(A) = (k−1)/k | 2/3 | ✓ (Markov + spectral give same formula) |
| α_1_bare | q_NB^(g−2) | 256/6561 | ✓ (cumulative q_NB) |
| α_1_full = V_cb | q_NB^(g−2)/(1−q_NB^(g−2)) | 256/6305 | ✓ (geometric series in q_NB) |
| c (dark Feshbach) | (2(\|E\|−\|V\|)+1)/(2\|E\|) = dim(marginal sector)/dim(B) | 5/12 | ✓ (cycle 15/36 = same formula in k, V via n_g identity) |

### Coincidentally agreeing at k = 3 (NOT algebraic unification)

| coefficient | spectral form | Bayesian form (primary) | agree at? |
|---|---|---|---|
| ε_CP | 1/(2k−1) | (k−2)/(k+2) | k = 3 only |
| A_hemispherical | inherits ε_CP/k* | inherits ε_CP/k* | inherits ε_CP's caveat |

Each coefficient has a *separate* derivation in the framework via different (cycle-counting, Bayesian-toggle, MDL, Markov) routes. The spectral identifications give a *unified* derivation: all four reduce to spectral observables of (A, B) at Γ.

## Background: substrate Hashimoto and adjacency at Γ

The substrate is the chiral 3-coordination crystal net `srs` (space group I4_132, Wyckoff 8a positions). At the Γ point of the Brillouin zone:

- **Adjacency A**: 4×4 matrix on the K_4 primitive-cell quotient. Spectrum: σ(A) = {+3, −1, −1, −1}.
- **Hashimoto B**: 12×12 matrix on the directed-edge space. Built per Stark-Terras 1996.

By the Stark-Terras factorization:
$$\det(uI - B) = (u^2 - 1)^{|E|-|V|} \cdot \prod_{\lambda \in \sigma(A)} (u^2 - \lambda u + (k_*-1))$$

For srs:
$$\det(uI - B) = (u^2 - 1)^2 \cdot (u^2 - 3u + 2) \cdot (u^2 + u + 2)^3$$

This factors B's 12-dim spectrum into three sectors:

- **Perron sector** (1-dim): u = +2 from (u−2). Visible — carries dynamical growth.
- **Oscillatory sector** (6-dim): complex pairs (−1 ± i√7)/2 from (u² + u + 2)³. Visible — carries oscillation.
- **Marginal sector** (5-dim): u = ±1 from bipartite factor + u = 1 from λ=+3 factor. Dark — no net dynamics.

The Perron-Frobenius theorem gives λ_max(A) = k* = 3 (adjacency) and λ_max(B) = k*−1 = 2 (Hashimoto NB-walk top eigenvalue).

## Spectral derivations

### 1. q_NB = 2/3 = Perron ratio

The per-step non-backtracking survival probability is the ratio of Perron eigenvalues:
$$q_{\rm NB} = \frac{\lambda_{\max}(B)}{\lambda_{\max}(A)} = \frac{k_*-1}{k_*} = \frac{2}{3}$$

**Interpretation:** Over many steps, all-walks count grows as λ_A^N = 3^N; NB walks grow as λ_B^N = 2^N. The ratio q_NB = 2/3 is the per-step "discount" of NB-walking versus all-walking. This is the substrate's intrinsic information-decay rate for NB observables.

**Existing derivation (structural ledger Row 23):** Markov memorylessness of the NB walker on a uniform branch measure (Row 12), combined with k* = 3 from Row 4. The spectral identification is consistent with — and equivalent to — the Markov derivation.

### 2. α_1_bare = (2/3)^8 = cumulative q_NB

The NB walk survival probability over a full girth-cycle window:
$$\alpha_{1,\rm bare} = q_{\rm NB}^{g-2} = \left(\frac{\lambda_{\max}(B)}{\lambda_{\max}(A)}\right)^{g-2} = \left(\frac{2}{3}\right)^8 = \frac{256}{6561}$$

For srs, girth g = 10, so the exponent is g − 2 = 8 (the number of NB-walk steps to traverse a girth cycle and exclude the starting/closing edges).

**Interpretation:** α_1_bare is the visible-sector survival probability over a girth-cycle window. It propagates into:
- α_1_full = 256/6305 (after A2-T waterline correction)
- V_cb = α_1_full = 256/6305
- y_τ = α_1_full / k*² = 1280/177147

All these "geometric" (2/3)^L appearances are spectral consequences of the Perron ratio.

### 3. c = 5/12 = Q-projector dim fraction

The dark Feshbach amplitude is the dimensional fraction of the marginal sector (Q-space):
$$c = \frac{\dim(\text{marginal sector})}{\dim(B)} = \frac{2(|E|-|V|) + 1}{2|E|} = \frac{5}{12}$$

The marginal sector consists of the 5 eigenvalues with |λ|=1: 4 from the bipartite factor (u²−1)² + 1 from the Perron-A image (u=1 root of u²−3u+2). This is the natural Q-space in Feshbach projection: modes that neither grow (Perron) nor oscillate (complex pairs) — they carry no net dynamical information and must be projected out to isolate the visible sector.

**Existing derivation (parameter ledger Row P5):** cycle-counting, n_g/(N_atoms·k*²) = 15/36. The spectral derivation (per `theorem_dark_5_12_spectral.md`) gives the same value via a separate decomposition; both routes are over-determined.

### 4. ε_CP = 1/5 = Perron asymmetry

The baryon CP asymmetry per process is the spectral asymmetry of the Perron eigenvalues:
$$\varepsilon_{\rm CP} = \frac{\lambda_{\max}(A) - \lambda_{\max}(B)}{\lambda_{\max}(A) + \lambda_{\max}(B)} = \frac{k_* - (k_*-1)}{k_* + (k_*-1)} = \frac{1}{2k_*-1} = \frac{1}{5}$$

**Interpretation:** ε_CP measures the asymmetry between "all-walks rate" (λ_A) and "NB-walks rate" (λ_B). It's the relative spectral gap between the substrate's adjacency and Hashimoto Perron eigenvalues. For k* = 3 this gives 1/5; for k* → ∞ it goes to 0 (the gap closes asymptotically).

**Existing derivation (parameter ledger Row P28):** Bayesian-toggle posterior, ε_CP = (1/2 − 1/3)/(1/2 + 1/3) = 1/5. The spectral derivation (per `proofs/wave_engine/dark_eps_cp_spectral.py`) gives the same number from Perron eigenvalues.

**Bonus identity for srs:** since 2k* − 1 = 5 = 2(|E|−|V|) + 1 = marginal sector dim, we additionally have ε_CP = 1/dim(marginal). This equality only holds because srs's specific (|V|=4, k=3) satisfies |V| = 2(k−1)/(k−2). For other cells, the two identifications would diverge.

## Joint substrate selectivity

The framework's substrate (|V|=4, k=3, |E|=6) is *uniquely identified* by requiring the four dark coefficients to take their observed values simultaneously:

- q_NB = 2/3 forces k = 3 (via Perron ratio (k−1)/k = 2/3)
- ε_CP = 1/5 forces k = 3 (via the PRIMARY Bayesian form (k−2)/(k+2) = 1/5 ⟹ k = 3, a linear inversion; the spectral 1/(2k−1) = 1/5 pins the same k but is the k = 3-coincidence route per the Class-A audit caveat above — the selectivity claim rests on the primary form)
- α_1_bare = (2/3)^8 forces (k = 3, girth = 10)
- c = 5/12 forces (|V| = 4, |E| = 6, k = 3) (via spectral decomposition uniqueness)

All four constraints jointly satisfy at exactly (|V|=4, k=3, |E|=6) — the srs primitive cell. Any deviation produces a different value at one or more of the four coefficients.

This is over-determination: the framework's substrate is constrained by four independent observations of dark physics, all converging on the same configuration.

## Connection to non-spectral dark coefficients

Two other dark coefficients of the framework are NOT spectral:

- **Ω_DM/Ω_m = 0.8488** (cosmological dark matter fraction, Row P22): derived from Poisson(2k*) tail above the visible-toggle cutoff k* = 3. *Statistical*, not spectral. Lives at the random-graph layer, not the operator layer.
- **5/12 vs 0.8488** are independent: 5/12 / 0.8488 ≈ 0.491 (off from clean 1/2 by 1.7%), and they cannot have a clean rational ratio since 0.8488 is irrational (involves e^(−6)).

The framework's dark structure thus has at least two layers:
1. **Spectral dark** (q_NB, α_1, c, ε_CP): live on Hashimoto B(Γ) and adjacency A(Γ).
2. **Statistical dark** (Ω_DM/Ω_m): lives on random-graph degree distributions.

Both layers contribute to the framework's dark-sector predictions; conflating them is a category error. Each requires its own derivation route.

## Summary table

| coefficient | name | spectral formula | k=3 value | rational? |
|---|---|---|---|---|
| q_NB | NB walk survival | λ_max(B) / λ_max(A) | 2/3 | yes |
| α_1_bare | NB cumulative survival | (λ_max(B)/λ_max(A))^(g−2) | 256/6561 | yes |
| c | dark Feshbach amplitude | (2(\|E\|−\|V\|)+1) / (2\|E\|) | 5/12 | yes |
| ε_CP | baryon CP asymmetry | (λ_A−λ_B) / (λ_A+λ_B) | 1/5 | yes |
| Ω_DM/Ω_m | cosmological dark fraction | NOT spectral; Poisson(2k*) tail | ≈ 0.8488 | **no** (e^(−6)) |

## Ihara-map value/gradient merger at the Perron (added 2026-04-28; reframed post-audit)

The Ihara map u(λ) = (λ + √(λ² − 4(k−1)))/2 connects adjacency eigenvalues to Hashimoto eigenvalues. At the Perron eigenvalue λ = k:
- **u(k) = k − 1** (Hashimoto Perron — Class A value)
- **u'(k) = k/(2√((k−2)²)) + 1/2 = (k−1)/(k−2)** for k > 2 (Class B link factor)

Setting u(k) = u'(k) gives the equation k − 1 = (k−1)/(k−2), with solutions **k = 1 (trivial) and k = 3 (non-trivial)**.

For srs's k* = 3: u(3) = u'(3) = 2. The value AND gradient of the Ihara map coincide at the Perron — a non-generic spectral configuration. For any other coordination (k = 4: u(k) = 3, u'(k) = 3/2; k = 5: 4, 4/3; etc.), Class A "value" relations and Class B "gradient" relations have distinct numerical content.

**Observation, not unification.** k = 3 is the *unique* coordination > 2 where the Ihara map's value-gradient merger at the Perron collapses Class A and Class B onto a single substrate constant (k − 1 = 2). Row 4's Brown 1986 information-bound argument independently selects k* = 3.

**Honest framing post-audit (2026-04-28).** Two structurally independent arguments — Brown 1986 fixed-degree information bound (Row 4) and Ihara value/gradient merger at the Perron (this section) — both pick k = 3. The two use disjoint mathematical machinery (information-theoretic minimum-redundancy on crystal-net periodicity vs Hashimoto zeta function geometry on the cross-walker map). Whether their convergence on k = 3 reflects a deeper structural reason or is a non-trivial numerical coincidence is **open**: we have no derivation linking Fisher rank to Hashimoto zeta function geometry that would force the agreement. The Ihara observation is recorded as a *consistent* finding, not a proof of cross-validation. See `theorem_class_A_audit.md` for the audit detail and `../audits/registers/uniqueness_ledger.md` Row 4 for the row-level statement.

## Implications

1. **Dark physics has unified spectral substrate identity** at the operator layer. Four of the framework's principal dark coefficients reduce to spectral observables of one operator family (A, B at Γ). This is structurally tighter than treating each coefficient as an independent prediction.

2. **The Hashimoto operator's spectrum encodes the framework's dark dynamics.** The split into Perron (visible-growing), oscillatory (visible-oscillatory), and marginal (dark) sectors corresponds to the framework's dark/visible distinction. This is a *first-principles* identification of the dark sector — not a phenomenological postulate.

3. **Joint coefficient selectivity over-determines the substrate.** Removing any one structural row would shift at least one dark coefficient. The framework's (|V|=4, k=3, |E|=6) is constrained from multiple independent angles; consistency of all four spectral identifications is a non-trivial cross-check.

4. **Two dark layers, not one.** The framework needs both spectral and statistical mechanisms to produce all observed dark phenomena. The 5/12 (Feshbach) and 0.8488 (cosmological) coefficients are independent observables that together specify the dark sector.

## References and verification

Numerical verification scripts (all confirm exact spectral identifications):
- `proofs/wave_engine/dark_5_12_spectral.py` — c = 5/12 via Stark-Terras factorization.
- `proofs/wave_engine/dark_alpha1_spectral.py` — q_NB = 2/3 and α_1 = (2/3)^8 via Perron ratio.
- `proofs/wave_engine/dark_eps_cp_spectral.py` — ε_CP = 1/5 via Perron asymmetry.
- `proofs/wave_engine/dark_omega_spectral_attempts.py` — confirms 0.8488 has NO spectral identification.
- `proofs/wave_engine/dark_statistical_cell_dependence.py` — joint cell-selectivity at (|V|=4, k=3).

Theorem documents:
- `theorem_dark_5_12_spectral.md` — spectral derivation of 5/12.
- `theorem_dark_correction_mdl.md` — existing cycle-counting derivation of 5/12 (cross-checks).

Parameter ledger entries:
- Row P1, P2 — α_1_bare, α_1_full (q_NB^(g−2) and corrections).
- Row P5 — c = 5/12 (dual spectral + cycle derivation).
- Row P22 — Ω_DM/Ω_m (statistical, not spectral).
- Row P27, P28 — A_hemispherical = 1/15, ε_CP = 1/5.

Structural ledger entries:
- Row 7 — |E| = 6 (forces marginal sector dim).
- Row 16 — |V| = 4 (forces K_4 primitive cell).
- Row 23 — q_NB = 2/3 (Perron ratio identification, updated 2026-04-28).

External references:
- Stark, H.M. and Terras, A. 1996/2000/2007 — Hashimoto operator and Ihara zeta function for k-regular graphs.
- Sunada, T. 2012, *Topological Crystallography* §4.3 — srs cycle structure.
- Levin, Peres, Wilmer 2009, *Markov Chains and Mixing Times* §1.14 — geometric q^L Markov scaling.
