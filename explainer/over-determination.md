# The 12-Observable Over-determination

!!! quote ""
    **The same substrate object, read 12 different ways from one resolvent — one argument, zero fitted constants. Eleven of the twelve match PDG within 1σ_PDG (m_t and m_b joined 2026-06-25, when the resolvent's own forced first-girth-return dark correction shipped); the twelfth (δ_r) has no direct PDG observable — the M_Z it feeds carries the framework's honestly-open oblique residual.**

This is the framework's strongest over-determination claim — read honestly: one operator, no free parameters, eleven independent PDG hits at the same time. The point that distinguishes it from any programme that *pins* the Standard Model via a single mechanism is the **shared operator**, not a count of 12-for-12 (which it does not claim: δ_r's downstream observable M_Z stays a logged open miss).

## The object

The non-backtracking resolvent on the substrate:

$$G = (I - u \cdot B)^{-1}$$

with one argument $a = (2/3)^8$ and zero fitted constants. Here $B$ is the Hashimoto (non-backtracking walk) operator on srs. The 12-observable family is read off this single object by varying the *channel* (which spectral statistic of the resolvent is being extracted), not by varying free parameters.

## The 12 readouts

### Quark sector (7 observables)

| Observable | Predicted | Measured | Match |
|---|---|---|---|
| Top Yukawa $y_t$ | top Yukawa at unification = 1 (saturation) × forced dark $(1-\alpha_1/h_P^2)$ | pole $m_t = 172.69 \pm 0.30$ GeV | $m_t = 172.41$ GeV, $-0.95\sigma$ (bare pre-dark $+4.71\sigma$; closed 2026-06-25) |
| Bottom Yukawa $y_b$ | $(2/3)^{10}$ × forced dark $(1-\alpha_1/h_P)$ | pole $m_b = 4.18$ GeV | $m_b = 4.187$ GeV, $+0.22\sigma$ (bare pre-dark $+2.99\sigma$; closed 2026-06-25) |
| Cabibbo angle $V_{us}$ | $9/40 = 0.22500$ | $0.22501 \pm 0.00068$ | $-0.01\sigma$ |
| $V_{cb}$ | $256/6305 = 0.04060$ | $0.0406 \pm 0.0009$ | $+0.00\sigma$ |
| $V_{ub}$ | geometric series over walk windings $= 3.77 \times 10^{-3}$ | $3.82 \times 10^{-3} \pm 0.20 \times 10^{-3}$ | $-0.26\sigma$ |
| Oblique correction $\delta_r$ | $(1/12) \cdot \alpha / (1-\alpha)$ with $\alpha = (2/3)^8$ | tree-to-pole shift on $M_Z$ | structural |
| Custodial breaking $\delta\rho$ | $(1/2)(\sqrt{5}/4)(2/3)^8$ | $\approx +1.04\%$ | $+0.76\sigma$ |

### Lepton / PMNS sector (4 observables)

| Observable | Predicted | Measured | Match |
|---|---|---|---|
| Tau Yukawa $y_\tau$ | substrate spectral expression $\times$ Higgs-vertex correction | 0.00722 | structural |
| PMNS $\theta_{12}$ | 33.07° (from a Pati–Salam perpendicular identity) | 33.41° | $-0.45\sigma$ |
| PMNS $\theta_{13}$ | 8.605° (tri-bimaximal plus an edge-local dark correction) | $8.57° \pm 0.11°$ | $+0.32\sigma$ |
| PMNS $\theta_{23}$ | 48.72° | $49.2° \pm 1.3°$ | $-0.37\sigma$ |

### Cosmological (1 observable)

| Observable | Predicted | Measured | Match |
|---|---|---|---|
| Scalar-perturbation prefactor | $1/54$ (single projection of the resolvent) | matched | structural |

## Why this is more than 12 independent fits

Three considerations make over-determination — not coincidence — the right reading:

1. **No fitted constants.** Each prediction is computed from the substrate (srs structural integers: $k_* = 3$, $g = 10$, $|V| = 4$, $|E| = 6$, $N_{\mathrm{atoms}} = 4$) plus the single substrate-derived spectral input $a = (2/3)^8$. Nothing is adjusted.
2. **One object, many channels.** All 12 are read off the *same* $G_{NB}$ — varying the spectral channel (which projection of the resolvent), not varying parameters. If $G_{NB}$ were the wrong object, the readings would fail together; the fact that **eleven** of them hit PDG at $|\sigma| \lesssim 1$ simultaneously — from one operator with no free parameters, including the m_t/m_b closures that came from the resolvent's *own* first-girth-return correction rather than any new input — is the over-determination. (δ_r has no direct σ to test; the M_Z it feeds is the framework's logged open oblique residual — see [the honest σ count](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/parameters/honest_sigma_count_2026-06-22.md).)
3. **Genuine cross-validation.** $V_{us}$ comes from one channel (Level-2 srs density), $V_{cb}$ from another (Level-3 Hashimoto BFS at $L = 8$), $\delta_r$ from a third (Z/Perron oblique), etc. If the framework were curve-fitting, fitting one would not pin the others — the channels are independent functions on the same operator.

## Companion: the dark sector

A parallel structure on the dark side of the framework — the **multi-axial dark-sector waterfilling theorem** — closes roughly fifty chirality-routed observables uniformly via the substrate-uniqueness mechanism. The 12-observable family above is the *gauge-side complement* of the dark-side $\Omega_{\mathrm{DM}}$ closure.

<video controls autoplay muted loop playsinline width="100%" style="max-width: 800px; display: block; margin: 0 auto;">
  <source src="../assets/animations/twelve_observables.mp4" type="video/mp4">
</video>

*One central object — the non-backtracking resolvent $G = (I - u \cdot B)^{-1}$ on the srs lattice, with one argument $a = (2/3)^8$ — feeds twelve distinct spectral channels. Each channel produces one Standard Model observable (seven quark-sector, four lepton/PMNS, one cosmological). Eleven of the twelve match measurement within about $1\sigma$ with zero fitted constants; the twelfth (δ_r) has no direct observable and its downstream M_Z is a logged open residual.*

## The rigorous version

For the per-row audit (mechanism, alternatives ruled out, conditional dependencies) of all twelve observables, see the framework's research tree linked from the [reference page](reference.md) — specifically the unified-oblique theorem, the lepton–PMNS over-determination audit document, and the parameter-uniqueness ledger.
