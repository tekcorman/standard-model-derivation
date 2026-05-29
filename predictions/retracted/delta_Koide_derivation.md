# δ_Koide — STATUS: BLOCKED under B6 (2026-04-17)

**NOTE (post-A3, 2026-04-18):** Historical pre-A3 two-axiom derivation, retained as-is. See predictions/Q_Koide_v2.py for the post-A3 Born-rule re-derivation of the upstream color-sector Q identity under the three-axiom framework (A1+A2+A3). Canonical axiom statement: docs/framework_axioms.md.

## Status

**BLOCKED under Theorem B6 retraction.** This derivation is a one-line algebraic corollary of `Q_Koide_derivation.md`: given the Koide-parametrisation identity `δ = Q(1 − Q)` and the upstream closure `Q = 2/3`, one computes `δ = 2/9`. The upstream `Q_Koide` closure itself is retracted under B6, because its Step 5 identifies the C_3 irrep index `j` with a charged-lepton generation index, whereas B6 (`docs/theorem_B6_bridge.md`) proves the srs body-diagonal C_3 is the color-Z_3 of SU(3)_c via Spin(6)≅SU(4)→PS embedding, not a generation label. With the upstream generation identification retracted, the δ = Q(1 − Q) chain no longer derives the charged-lepton Koide phase.

**Re-derivation target**: Sprint 11 workstream B7.4 (Koide parameters under the C³_gen mass operator; see `docs/master_plan.md` §Sprint 11). The framework has a separate structural path to three generations via the observer's minimum viable Hilbert space dimension C³_gen (MDL + Gleason's theorem).

**What survives as math**: the algebraic identity `δ = Q(1 − Q)` is a purely algebraic consequence of the Koide parametrisation (Bernoulli second-moment identity) and remains valid for any Q; in particular, combining it with the color-sector arithmetic lemma `Q_{color-sector} = 2/3` (from the retracted `Q_Koide_derivation.md`) gives `δ_{color-sector} = 2/9` as a color-sector arithmetic lemma. This is label-agnostic; only the identification with the observed charged-lepton Koide phase is retracted.

## Specific failing step

Step 1 inherits `Q = 2/3` from `Q_Koide_derivation.md`, whose Step 5 makes the C_3-irrep-index-equals-generation-index identification. The failing step therefore sits one level upstream (`Q_Koide` Step 5: "P2 Fourier formula with j = generation index"). The present file's Step 2 (δ = Q(1 − Q) algebra) is label-free and survives as a mathematical identity; Step 3's substitution `δ = (2/3)(1/3) = 2/9` becomes a color-sector arithmetic statement rather than a charged-lepton Koide-phase derivation.

## Empirical comparison (flagged as coincidence, not derivation)

| Quantity | Derived (under retracted reading) | Observed (PDG 2024) | Status |
|---|---|---|---|
| δ_Koide | 2/9 = 0.222222… | 0.2222227 ± 0.0000009 | not explanatory under current framework |

The numerical match between the color-sector arithmetic lemma `δ = Q(1 − Q) = 2/9` and the observed charged-lepton Koide phase is an empirical coincidence under the retracted reading. Whether it is re-derivable under the C³_gen framework is the Sprint 11 B7.4 open question.

## Preserved original derivation (for reference; superseded)

---

# Derivation of the Koide phase parameter δ (SUPERSEDED, retained for reference)

## Abstract

We derive

$$\delta \;:=\; Q_{\text{Koide}}\,\bigl(1 - Q_{\text{Koide}}\bigr) \;=\; \tfrac{2}{9}$$

as an algebraic corollary of the closed result $Q_{\text{Koide}} = (k^{*}-1)/k^{*} = 2/3$ proven in `predictions/Q_Koide_derivation.md`. The Koide phase is, by the standard Koide parametrisation, the Bernoulli second moment $Q(1-Q)$; once $Q$ is fixed by the upstream derivation, $\delta$ is a single algebraic step away, and no additional structural postulate beyond those of $Q_{\text{Koide}}$ is required. The alternative "harmonic mean of squared Wigner $d^{1}$ survival amplitudes" route that was adopted in earlier versions of this file is shown in an appendix to be a numerical coincidence at $k^{*} = 3$ — the HM identity $\text{HM}(P_+, P_0, P_-) = (k-1)/k^{2}$ holds only when the cubic $k^{3} - 2k^{2} + k - 12 = 0$ is satisfied, which has $k = 3$ as its unique real root. It is therefore not a general derivation and is not used as the primary chain.

## Framework axioms invoked

Inherited via `predictions/Q_Koide_derivation.md`; no new axioms or postulates introduced here.

- **(A1)** Binary self-inverse toggle (`predictions/p_toggle.py`).
- **(A2)** MDL compression (`predictions/d_spatial.py`, `predictions/k_star.py`).
- **Theorem** `docs/theorem_walker_dynamics.md` — walker dynamics on srs are non-backtracking walks; $B$ is the Hashimoto matrix on directed edges; $L$-step transition amplitudes are matrix elements of $B^{L}$.
- **Theorem** `docs/theorem_BP_doubly_degenerate_h.md` — $P$-point Bloch spectrum; $\pm\sqrt{3}$ A-eigenspaces decompose under $C_3$ as $(\text{trivial}\oplus\omega)$ and $(\text{trivial}\oplus\omega^{2})$.
- **Postulate P1** (`docs/W4_identification_catalog.md` §3) — Ramanujan selection.
- **Postulate P2** (same §3) — $\sqrt{\textrm{multiplicity}}$ coherent aggregation.
- **Upstream closed result** `predictions/Q_Koide_derivation.md` — $Q_{\text{Koide}} = (k^{*}-1)/k^{*} = 2/3$ exactly (its own closure is through Steps 1–7 of that file, which together with P1 and P2 consume only the axioms and theorems above). **[Upstream file now BLOCKED under B6.]**

## Derivation

### Step 1. Upstream: $Q_{\text{Koide}} = (k^{*}-1)/k^{*}$ [INHERITED UPSTREAM BLOCK]

By `predictions/Q_Koide_derivation.md` Steps 1–7, under postulates P1 and P2 the Koide ratio at $k^{*} = 3$ on srs is
$$Q \;=\; \frac{\sum_{j=0}^{k^{*}-1} m_j}{\Bigl(\sum_{j=0}^{k^{*}-1}\sqrt{m_j}\Bigr)^{2}} \;=\; \frac{k^{*}-1}{k^{*}} \;=\; \frac{2}{3}.$$

This upstream derivation's Step 5 interprets the C_3 irrep index `j` as a generation index; B6 retires this interpretation.

### Step 2. The Koide phase is the Bernoulli second moment $Q(1-Q)$

The standard Koide parametrisation writes the three generation mass amplitudes as
$$\sqrt{m_j} \;=\; \sqrt{M}\,\bigl(1 + \varepsilon\cos(2\pi j/k^{*} + \delta_{\text{phase}})\bigr), \qquad j = 0, 1, \dots, k^{*}-1.$$

(Koide 1983, *Phys. Lett. B* **120**, 161–165; see also `predictions/epsilon_Koide_derivation.md` for the amplitude.) The three Koide invariants $(Q, \varepsilon, \delta)$ extracted from the spectrum $\{m_j\}$ are related by the identities

$$Q \;=\; \frac{\Sigma\,m}{(\Sigma\sqrt{m})^{2}}, \qquad \varepsilon^{2} \;=\; 2(k^{*}Q - 1), \qquad \delta \;=\; Q\,(1 - Q).$$

These relations are purely algebraic consequences of evaluating $\Sigma\sqrt{m}$, $\Sigma m$, and $\Sigma m^{2}$ under the Koide parametric form above and eliminating $M, \varepsilon, \delta_{\text{phase}}$.

### Step 3. Substitute $Q = (k^{*}-1)/k^{*}$

$$\delta \;=\; Q\,(1 - Q) \;=\; \frac{k^{*}-1}{k^{*}}\cdot\frac{1}{k^{*}} \;=\; \frac{k^{*}-1}{k^{*\,2}}.$$

At $k^{*} = 3$:
$$\delta \;=\; \frac{2}{9} \;=\; 0.2222\overline{2}.$$

## Result (color-sector arithmetic lemma only; generation identification retracted under B6)

$$\delta_{\text{color-sector lemma}} \;=\; \tfrac{2}{9} \;=\; 0.2222\overline{2}.$$

## Appendix A. The Wigner $d^{1}$ harmonic-mean route is a $k=3$ coincidence

Earlier versions of this file took $\delta$ to be the harmonic mean
$$\text{HM}(P_+, P_0, P_-) \;=\; \frac{3}{1/P_+ + 1/P_0 + 1/P_-}$$
of the squared diagonal elements of the Wigner small-$d$ matrix at "angle $\beta$ with $\cos\beta = 1/k^{*}$," with
$$P_\pm \;=\; \Bigl(\tfrac{1 + \cos\beta}{2}\Bigr)^{2}, \qquad P_0 \;=\; \cos^{2}\beta.$$

(Sakurai, *Modern Quantum Mechanics*, 3rd ed., §3.8, eq. 3.8.33.) This route is not a general derivation: the identity $\text{HM}(P_+, P_0, P_-) = (k-1)/k^{2}$ fails off $k = 3$.

**Claim.** $\text{HM}(P_+, P_0, P_-) = (k-1)/k^{2}$ iff $k^{3} - 2k^{2} + k - 12 = 0$, i.e. iff $(k-3)(k^{2} + k + 4) = 0$; the only real root is $k = 3$.

**Proof.** Writing $c = 1/k$,
$$P_\pm = \Bigl(\tfrac{1+c}{2}\Bigr)^{2}, \quad P_0 = c^{2}, \quad \text{HM} = \frac{3}{2\Bigl(\tfrac{2}{1+c}\Bigr)^{2} + \tfrac{1}{c^{2}}} = \frac{3\,c^{2}(1+c)^{2}}{8\,c^{2} + (1+c)^{2}}.$$

Substituting $c = 1/k$ and multiplying numerator and denominator by $k^{4}$,
$$\text{HM}(k) \;=\; \frac{3\,(k+1)^{2}}{k^{2}\bigl(8 + (k+1)^{2}\bigr)} \;=\; \frac{3\,(k+1)^{2}}{k^{2}\,(k^{2} + 2k + 9)}.$$

Setting $\text{HM}(k) = (k-1)/k^{2}$ and clearing denominators gives
$$3(k+1)^{2} \;=\; (k-1)(k^{2} + 2k + 9),$$
i.e.
$$3k^{2} + 6k + 3 \;=\; k^{3} + k^{2} + 7k - 9,$$
i.e.
$$k^{3} - 2k^{2} + k - 12 \;=\; 0.$$

Factoring, $k^{3} - 2k^{2} + k - 12 = (k - 3)(k^{2} + k + 4)$. The quadratic factor has discriminant $1 - 16 = -15 < 0$, so its roots are complex; the only real root of the cubic is $k = 3$. $\square$

## References

- Hardy, G.H., Littlewood, J.E. & Pólya, G. (1952). *Inequalities,* 2nd ed. Cambridge University Press. Ch. 2 (power means and means of the first kind).
- Koide, Y. (1983). A fermion–boson composite model of quarks and leptons. *Phys. Lett. B* **120**, 161–165.
- Sakurai, J.J. & Napolitano, J. (2020). *Modern Quantum Mechanics*, 3rd ed. Cambridge University Press. §3.8 eq. 3.8.33 (Wigner small-$d$ matrix for $j = 1$; cited only in Appendix A).
- Sunada, T. (2012). Lecture on topological crystallography. *Notices AMS* **59**(2), 208–215.
- Terras, A. (2011). *Zeta Functions of Graphs: A Stroll through the Garden.* Cambridge University Press. §2.2 (Hashimoto matrix), §2.3 (Ihara–Bass identity).
- Cross-references: `predictions/Q_Koide_derivation.md` (BLOCKED under B6), `predictions/epsilon_Koide_derivation.md` (BLOCKED under B6), `docs/theorem_B6_bridge.md`.
