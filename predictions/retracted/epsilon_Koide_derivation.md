# ε_Koide — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. See predictions/Q_Koide_v2.py for the post-A3 Born-rule re-derivation of the color-sector identity under the three-axiom framework (A1+A2+A3). Canonical axiom statement: docs/framework_axioms.md.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation's Step 6 matches `√m_j = √μ_triv + 2√μ_ω · cos(2π j / 3)` to the Koide parametric form `√m_j = √M (1 + ε cos(2π j / 3))`, with the index `j ∈ {0, 1, 2}` read as a charged-lepton generation index. B6 (`docs/theorem_B6_bridge.md`) proves the srs body-diagonal C_3 is the color-Z_3 of SU(3)_c via Spin(6)≅SU(4)→PS embedding, not a generation label. Under B6, the three C_3 irreps label color components within ONE Pati-Salam family, so the coherent sum aggregates across colors, not across generations; the resulting `ε` is a ratio of color-sector multiplicities, not a charged-lepton amplitude parameter.

**Re-derivation target**: Sprint 11 workstream B7.4 (Koide parameters under the C³_gen mass operator; see `docs/master_plan.md` §Sprint 11). The framework has a separate structural path to three generations via the observer's minimum viable Hilbert space dimension C³_gen (MDL + Gleason's theorem).

**What survives as math**: the color-sector multiplicity identity

$$\varepsilon_{\text{color-sector}}^{2} \;=\; \frac{4\mu_{\omega}}{\mu_{\text{triv}}} \;=\; \frac{4 \cdot 2}{4} \;=\; 2, \qquad \varepsilon_{\text{color-sector}} \;=\; \sqrt{2}$$

is a **standalone mathematical lemma** about the srs Ramanujan (4, 2, 2) color-isotypic multiplicity structure at the P-point. The lemma is label-agnostic and remains valid under B6; only the identification of `ε` with the observed charged-lepton Koide amplitude is retracted.

## Specific failing step

Step 6 matches the P2-aggregated amplitude `√m_j = √μ_triv + 2√μ_ω · cos(2π j / 3)` (which comes from Step 5 of `Q_Koide_derivation.md`, its own failing step) to the Koide parametric form and extracts `ε² = 4 μ_ω / μ_triv`. Under the retracted reading, `j` was a charged-lepton generation index; under B6, `j` labels color components within one PS family. The parametric match is arithmetically correct as a statement about color-sector spectral data; it does not derive the empirical ε of the charged-lepton Koide fit.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (PDG 2024) | Status |
|---|---|---|---|
| ε_Koide | √2 = 1.41421 | 1.414209 ± 0.000011 | not explanatory under current framework |

The numerical match between the color-sector multiplicity-ratio √2 and the observed charged-lepton Koide amplitude is an empirical coincidence under the retracted reading. Whether it is re-derivable under the C³_gen framework is the Sprint 11 B7.4 open question.

## Preserved original derivation (for reference; superseded)

---

# Derivation of the Koide amplitude ε (SUPERSEDED, retained for reference)

## Abstract

We derive $\varepsilon = \sqrt{2}$, the amplitude of the charged-lepton Koide mass parametrisation $\sqrt{m_j} = \sqrt{M}\,(1 + \varepsilon\cos(2\pi j/3))$, as a consequence of the $C_3$ irreducible-representation multiplicity structure of the Ramanujan subspace of the srs Bloch non-backtracking walk operator $B(\mathbf{k})$ at the $P$-point. Steps 1–5 are identical to `predictions/Q_Koide_derivation.md`: upstream $k^*=3$ + srs = I4_132, the `theorem_walker_dynamics` closure of W1–W3, the $\pm\sqrt{3}$-eigenspace $C_3$ decomposition from `theorem_BP_doubly_degenerate_h`, and the Ihara–Bass lift giving 8-dim Ramanujan multiplicities $(\mu_{\text{triv}}, \mu_\omega, \mu_{\omega^{2}}) = (4, 2, 2)$, plus adopted postulates P1 (Ramanujan selection) and P2 ($\sqrt{\mu}$ coherent aggregation) from the W4 catalog. Step 6 extracts $\varepsilon$ by matching the aggregated amplitude to the Koide parametric form: $\varepsilon^{2} = 4\mu_\omega/\mu_{\text{triv}} = 2 = 2(k^{*}-2)$.

## Framework axioms invoked

Identical to `predictions/Q_Koide_derivation.md`:

- **(A1)** Binary self-inverse toggle, **(A2)** MDL compression.
- **Theorem** `docs/theorem_walker_dynamics.md` — walker dynamics are NB walks; $B$ = Hashimoto.
- **Theorem** `docs/theorem_BP_doubly_degenerate_h.md` — $P$-point spectrum, $C_3$-decomposition of $\pm\sqrt{3}$-eigenspaces.
- **Postulates P1, P2** — `docs/W4_identification_catalog.md` §3, adopted structural content.

## Derivation

### Steps 1–5. Upstream and multiplicity structure — identical to Q_Koide

These steps are proven verbatim in `predictions/Q_Koide_derivation.md` §Derivation Steps 1–5. They establish:

- $k^{*} = 3$, $d = 3$; srs = I4_132 with Wyckoff 8a, $x = 1/8$.
- $B$ is the Hashimoto NB transition operator; L-step amplitudes are $B^{L}$ matrix elements.
- At the $P$-point, the $+\sqrt{3}$ and $-\sqrt{3}$ A-eigenspaces each decompose under $C_3$ as (trivial $\oplus$ charged), with the charged piece being $\omega$ and $\omega^{2}$ respectively.
- Via Ihara–Bass, the 8-dim Ramanujan subspace $\{h, h^{*}, -h, -h^{*}\}$ of $B(P)$ has $C_3$ multiplicities
$$\mu_{\text{triv}} = 4, \qquad \mu_\omega = 2, \qquad \mu_{\omega^{2}} = 2.$$

**Under B6, these are color-isotypic counts for one PS generation, not per-generation counts.**

### Step 6. Match the aggregated amplitude to the Koide parametric form [FAILING STEP under B6]

By postulate P2, the generation-$j$ mass amplitude is
$$\sqrt{m_j} \;=\; \sqrt{\mu_{\text{triv}}} \;+\; \sqrt{\mu_\omega}\,\omega^{j} \;+\; \sqrt{\mu_{\omega^{2}}}\,\omega^{-j}, \qquad \omega = e^{2\pi i / 3}.$$

Since $\mu_\omega = \mu_{\omega^{2}}$ (required for a real mass spectrum; the ${\pm\sqrt{3}}$ A-eigenspaces are paired under complex conjugation by theorem_BP Step 3), this simplifies to
$$\sqrt{m_j} \;=\; \sqrt{\mu_{\text{triv}}} \;+\; 2\sqrt{\mu_\omega}\,\cos\!\left(\tfrac{2\pi j}{3}\right).$$

Substituting $(\mu_{\text{triv}}, \mu_\omega) = (4, 2)$:
$$\sqrt{m_j} \;=\; 2 \;+\; 2\sqrt{2}\,\cos\!\left(\tfrac{2\pi j}{3}\right).$$

**The Koide parametric form** is (Koide, *Phys. Lett. B* **120**, 161–165, 1983)
$$\sqrt{m_j} \;=\; \sqrt{M}\,\bigl(1 + \varepsilon\cos(2\pi j/3 + \delta_{\text{phase}})\bigr).$$

Matching the two expressions term-by-term forces $\delta_{\text{phase}} = 0$ and
$$\sqrt{M} \;=\; \sqrt{\mu_{\text{triv}}}, \qquad \sqrt{M}\cdot\varepsilon \;=\; 2\sqrt{\mu_\omega},$$
so that
$$\varepsilon^{2} \;=\; \frac{4\,\mu_\omega}{\mu_{\text{triv}}} \;=\; \frac{4\cdot 2}{4} \;=\; 2, \qquad \varepsilon \;=\; \sqrt{2}.$$

**This is the step that B6 retires**: the index `j` is treated here as a charged-lepton generation index running over {e, μ, τ}. Under B6, `j` indexes color components of one PS family. The arithmetic remains a valid statement about color-sector multiplicity ratios; its identification with the observed charged-lepton Koide amplitude is retracted.

### Consistency with the compact form $\varepsilon^{2} = 2(k^{*}-2)$

The older version of this file derived $\varepsilon^{2} = 2(k^{*}-2)$ by solving $Q = (2 + \varepsilon^{2})/(2k^{*}) = (k^{*}-1)/k^{*}$. That derivation assumes the Koide parametrisation of the masses as an input. The rigorous route in Step 6 *derives* the parametrisation (with $\sqrt{M} = 2$, $\varepsilon = \sqrt{2}$, $\delta_{\text{phase}} = 0$) from the multiplicity data, and the value $\varepsilon^{2} = 2$ emerges numerically from the specific srs multiplicities $(4,2,2)$. The numerical identity $2 = 2(k^{*}-2)$ at $k^{*}=3$ is a matching between two different forms of the same answer; it is not an independent combinatorial result.

## Result (color-sector arithmetic lemma only; generation identification retracted under B6)

$$\varepsilon_{\text{color-sector lemma}} \;=\; \sqrt{2} \;\approx\; 1.41421.$$

## References

- Bass, H. (1992). The Ihara–Selberg zeta function of a tree lattice. *Int. J. Math.* **3**, 717–797.
- Ihara, Y. (1966). On discrete subgroups of the two-by-two projective linear group over $p$-adic fields. *J. Math. Soc. Japan* **18**, 219–235.
- Koide, Y. (1983). A fermion-boson composite model of quarks and leptons. *Phys. Lett. B* **120**, 161–165.
- Sunada, T. (2012). Lecture on topological crystallography. *Notices AMS* **59**(2), 208–215.
- Terras, A. (2011). *Zeta Functions of Graphs: A Stroll through the Garden.* Cambridge University Press. §2.2 (Hashimoto matrix), §2.3 (Ihara–Bass identity).
- `docs/theorem_B6_bridge.md` — B6 bridge theorem identifying the srs C_3 as color-Z_3 (retraction source).
