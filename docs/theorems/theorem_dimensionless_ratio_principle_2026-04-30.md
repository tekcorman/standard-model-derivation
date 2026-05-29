# Framework meta-principle: dimensionless-ratio prediction level

**Date:** 2026-04-30 EOD final.
**Status:** Meta-principle codified from G_sub closure findings. Companion to
the algebraicity meta-theorem (`theorem_lattice_coupling_general.md`)
and audit v2 protocol.
**Triggered by:** G_sub closure 2026-04-30 EOD revealing that the Step 3
path (a) "$\omega_{\rm obs}$ near pole" was a unit-mixing artifact —
the framework's actual prediction was the dimensionless ratio
$M_{\rm substrate}/M_{\rm Pl} = \sqrt{\pi}/8$ (substrate below Planck mass),
not a single number.

**Corrected 2026-04-30 EOD final:** the substrate-Planck mass ratio is
$M_{\rm substrate}/M_{\rm Pl} = \sqrt{\pi}/8 \approx 0.222$ (substrate
*below* Planck mass; substrate length *above* Planck length by factor
$8/\sqrt{\pi} \approx 4.51$). The earlier text in this doc and the Drude
closure doc had the ratio inverted (an algebra error: π and 16·N_atoms
were swapped across the path-(b) equation). See `theorem_g_sub_drude_closure_2026-04-30.md`
for the corrected derivation.

## 1. Statement of the principle

For any observable $X$ with non-trivial mass dimension, the framework's
**natural prediction level** is:

  *(running structure or invariant form) + (dimensionless scale ratio)*

**not**

  *(a single dimensionless number)*.

Specifically: if $X$ has mass dimension $[M^d]$, then
$X \cdot M_{\rm ref}^{-d}$ is a dimensionless quantity whose value depends
on the reference scale $M_{\rm ref}$. The framework predicts:

1. The *form* of $X$ (running structure, functional dependence on
   substrate operators, K[π] structural coefficient).
2. The *dimensionless ratio* $X(M_1) / X(M_2) \cdot (M_2/M_1)^d$ between
   any two reference scales.
3. The substrate-natural scale ratio $M_1 / M_{\rm substrate}$ for
   **one** reference scale, anchoring the dimensional content to the
   substrate.

The framework does **not** predict $X$ as a single dimensionless number
in some chosen unit system, because that "number" includes a *unit
choice* (definitional) plus a *physical content* (predicted) — and
conflating them is the failure mode that audit v2 was designed to catch.

## 2. The G_sub worked example

$G_{\rm Newton}$ has mass dimension $[M^{-2}]$. The framework's natural
prediction level:

1. **Running structure (form)**: $1/(16\pi G(\omega)) = 4/\pi^2 - 1/(36\omega^2)$
   from Drude/Kubo on the substrate's Bloch operator. Theorem-grade
   per `theorem_g_sub_drude_closure_2026-04-30.md` Step 1+2 +
   audit v2 PASS.

2. **UV asymptote in lattice units**: $G_{\rm UV} \cdot M_{\rm substrate}^2 = \pi/64$.
   Dimensionless prediction.

3. **Substrate-Planck scale ratio**: $M_{\rm substrate}/M_{\rm Pl} = \sqrt{\pi}/8 \approx 0.222$
   (equivalently $M_{\rm Pl}/M_{\rm substrate} = 8/\sqrt{\pi}$).
   Theorem-grade dimensionless ratio (path b) per Drude closure doc.

Together these three ingredients give:

  $G_{\rm UV}$ in Planck units $= G_{\rm UV} \cdot M_{\rm Pl}^2 = (\pi/64) \cdot (M_{\rm Pl}/M_{\rm substrate})^{2} = (\pi/64) \cdot (64/\pi) = 1$.

This **matches the observed $G_N \cdot M_{\rm Pl}^2 = 1$ exactly** — but
the "1" here is **definitional** (Planck units are defined to make
$G_N \cdot M_{\rm Pl}^2 = 1$), not a separate framework prediction.

The framework's *actual* prediction is the dimensionless ratio
$8/\sqrt{\pi}$ between substrate and Planck scales, plus the running
structure. The "value of $G_N = 1$ in Planck units" emerges as a
consequence, not as a separately-derivable number.

### Why this matters

The Hashimoto-Sakharov candidate $729\sqrt{3}/(128\pi^2)$ at 0.05% match
was attempting to predict $G_N = 1$ as a separate dimensionless number.
It FAILED audit v2 because the framework doesn't actually need this —
$G_N = 1$ is definitional given the substrate-Planck ratio.

The failure mode: **chasing a numerical match for a quantity that's
partly definitional**. Audit v2 catches this by requiring mechanism gating;
when no mechanism gates the chosen value, the apparent match reveals
itself as numerology rather than prediction.

## 3. Pattern recognition: when does this apply?

The dimensionless-ratio principle applies when an observable has the
following features:

1. **Non-trivial mass dimension** (or any dimension involving a unit
   convention — length, time, energy, etc.).
2. **Defined relative to a reference scale** (e.g., "in Planck units",
   "at the Z mass", "in lattice units") whose choice is a definition,
   not a prediction.
3. **Framework predicts the running structure** (the functional
   dependence on substrate operators, BZ integration, etc.).

Examples where this applies:

| Observable | Mass dim | Reference scale | Framework predicts |
|---|---|---|---|
| $G_{\rm Newton}$ | $[M^{-2}]$ | Planck mass | running form + $M_{\rm substrate}/M_{\rm Pl} = \sqrt\pi/8$ |
| $\Lambda_{\rm CC}$ (cosmological constant) | $[M^4]$ | Planck mass | running form (?) + scale ratio |
| $v_{\rm Higgs}$ | $[M^1]$ | electroweak scale | running form + ratio to substrate scale |
| $M_{\rm Pl}$ itself | $[M^1]$ | substrate scale | $M_{\rm substrate}/M_{\rm Pl}$ ratio (theorem-grade) |
| $H_0, t_0$ (Hubble, age) | $[M^1, M^{-1}]$ | cosmological scale | $N$-cascade × $t_{\rm Pl}$ |

Examples where this does NOT apply (genuinely dimensionless predictions):

| Observable | Comment |
|---|---|
| $V_{\rm us}, V_{\rm cb}, V_{\rm ub}$ | mixing angles — dimensionless |
| Mass ratios $m_e/m_\mu$, etc. | dimensionless |
| $\sin^2\theta_W$, $\alpha_{\rm EM}$ | dimensionless couplings |
| $\eta_B$, $\Omega_{\rm DM}/\Omega_m$ | dimensionless |
| Mixing-matrix phases | dimensionless |

For dimensionless observables, the framework's natural prediction IS a
single number (in K[π]). For dimensional observables, the natural
prediction is the *ratio* + *running form*, with the absolute number
in any specific unit system having a definitional component.

## 4. Implication for audit v2

When auditing a dimensional observable (mass, length, time, etc.):

1. **Identify the unit system**. The "value" of the observable in any
   unit system has a definitional piece (the unit choice itself) and a
   predicted piece (the dimensionless ratio).
2. **Audit the dimensionless ratio**, not the dimensional value.
3. **Don't chase numerical matches** for the dimensional value if the
   match could be tautological under a definitional unit choice.

The G_sub case is the canonical worked example: matching $G_N = 1$ in
Planck units is partly tautological (Planck units are *defined* to make
this 1), so chasing structural derivations of "1" as a single number
produces numerology. The right audit target is $M_{\rm substrate}/M_{\rm Pl}$.

## 5. Implication for the algebraicity meta-theorem

The algebraicity meta-theorem (`theorem_lattice_coupling_general.md`)
says framework predictions for Class A/B/C/E observables are in
$K = \mathbb{Q}(\sqrt{2}, \sqrt{3}, \sqrt{5})$ extended by $\pi$ as needed.
This applies cleanly to **dimensionless** observables.

For **dimensional** observables, the algebraicity claim applies to:
- The running form's coefficients (each in $K[\pi]$).
- The substrate-natural scale ratio (in $K[\sqrt{K[\pi]}]$ or similar).

It does NOT apply to the dimensional value in a specific unit system,
because the unit choice is not a framework operation.

## 6. Implication for predictions/

A `predictions/X.py` for a dimensional observable should output the
framework's natural prediction:
- The dimensionless ratio.
- The running form (if applicable).
- The dimensional value computed *only after* the substrate-natural
  scale ratio is identified, with the unit conversion explicitly tracked.

This avoids the unit-mixing failure mode that produced the
"$\omega_{\rm obs}$ near pole" phantom in the G_sub Step 3 path (a)
attempts.

## 7. Worked checks: which framework predictions need updating?

Scan of `predictions/` and parameter ledger for dimensional observables
that might benefit from path-(b) reframing:

| Prediction | Status | Action needed |
|---|---|---|
| G_sub | NOW theorem-grade via path (b) reframing | done 2026-04-30 |
| Λ_CC (P24) | external-anchored via Row 25 | check whether path (b) applies |
| v_Higgs (P10) | currently anchored to G_F | check; v_Higgs/M_substrate ratio? |
| m_τ family (P11) | mass ratios theorem-grade; absolute scale anchored | already in dimensionless-ratio form |
| H_0 (P19) | t_0 = N · t_P (cascade) | already in dimensionless-ratio form |
| t_0 (P20) | cascade theorem | already in dimensionless-ratio form |
| N_hub (P17) | dimensionless | not affected |

For each "check needed" row above, the question is: does the framework's
prediction of the dimensional value implicitly chase a definitional
target (like $G_N = 1$)? If so, reframe via dimensionless-ratio principle.

## 8. Cross-references

- `theorem_g_sub_drude_closure_2026-04-30.md` — worked example.
  for the dimensionless-ratio prediction level.
  the single-number candidate.
  — phantom finding from unit confusion.
- `theorem_lattice_coupling_general.md` — algebraicity meta-theorem
  (applies to dimensionless predictions).

## 9. Status

**Meta-principle codified.** Applies to all dimensional observables in
the framework. The G_sub closure is the canonical worked example;
parameter ledger rows P24, P10, P11, P19, P20 should be reviewed against
this principle in a follow-up sweep.
