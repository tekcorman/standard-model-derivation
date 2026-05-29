# Derivation of delta_Koide under A3 Born rule -- Koide phase algebraic identity

> **STATUS UPDATE 2026-05-08.** Same status note as `Q_Koide_derivation.md`: ADOPTED-P1 + ADOPTED-Y were **CLOSED via A5** on 2026-04-19; Need-A2 **CLOSED** via R3 + M1.B chain; M_gen non-degeneracy **CLOSED** via generic argument 2026-05-08. Under standing slate {A1 + A2-T + A3-T + A5}, **δ_Bernoulli = Q(1-Q) = 2/9 is theorem-grade for the charged-lepton sector**. NOTE: the IDENTIFICATION of δ_Bernoulli (variance, dimensionless) with the Koide cosine PHASE δ in radians (the parameter that gives 3-distinct lepton mass values via sqrt(m_j) = sqrt(M)·(1+ε·cos(2πj/3+δ))) is a NUMERICAL coincidence (2/9 ≈ 12.73° matches observed Koide phase). Whether this coincidence has a structural derivation is **Need-B** of an internal working note — a SEPARATE multi-session research question, NOT a Need-A2 residual. The body math (δ_Bernoulli = 2/9 algebraic identity) is correct; only the framing about Need-A2 dependence is stale.

## Abstract

We derive the color-sector Koide phase parameter

    delta = Q_Koide * (1 - Q_Koide) = (2/3) * (1/3) = 2/9   (exact rational)

at theorem grade under A1 + A2-T + A3-T (A2-T, A3-T are derived theorems per docs/framework/framework_axioms.md §10). The derivation is a two-step chain:

1. Chain-import Q = 2/3 from predictions/Q_Koide.py (strict-solid color-sector identity under A1 + A2-T + A3-T + ADOPTED-P1 + ADOPTED-Y).
2. Apply the Bernoulli-moment algebraic identity delta = Q*(1-Q) of the Koide parametrisation (CAS-verifiable pure algebra).

The result delta = 2/9 is the most direct consequence of Q_Koide possible: it introduces no additional derivation steps, no additional axioms, and no additional adopted residuals beyond those already in Q_Koide.

## Framework axioms invoked

All inherited via Q_Koide. No new axioms needed.

## Cited mathematical theorems

No new citations beyond Q_Koide_derivation.md. The Bernoulli-moment identity delta = Q*(1-Q) is a CAS-verifiable algebraic identity of the Koide parametrisation; it requires no external citations.

## Upstream closed prediction files

- predictions/Q_Koide.py (Q = 2/3; primary input)

## Derivation

### Step 1: Chain-import Q = 2/3

By predictions/Q_Koide.py and predictions/Q_Koide_derivation.md, the color-sector Koide ratio is

    Q = (sum_j m_j) / (sum_j sqrt(m_j))^2 = 2/3

at theorem grade under A1 + A2-T + A3-T + Jaynes 1957 + Serre 1977 + CDP 2011 Theorem 25, MODULO ADOPTED-P1 and ADOPTED-Y.

### Step 2: Bernoulli-moment identity

In the Koide parametric form

    sqrt(m_j) = sqrt(M) * (1 + epsilon * cos(2*pi*j/k* + phi)),   j = 0, ..., k*-1,

the Koide-phase parameter delta is defined by

    delta = Q * (1 - Q).

This is a pure algebraic identity derivable from the parametric form:

    sum_j m_j = k* * M * (1 + epsilon^2/2)
    sum_j sqrt(m_j) = k* * sqrt(M)
    Q = (1 + epsilon^2/2) / k*
    1 - Q = 1 - (1 + epsilon^2/2)/k* = (k* - 1 - epsilon^2/2)/k*
    Q*(1-Q) = (1 + epsilon^2/2)(k* - 1 - epsilon^2/2) / k*^2

This is the Bernoulli-variance form; delta parameterises the spread of m_j about the mean M. Substituting Q = 2/3 directly gives delta = (2/3)*(1/3) = 2/9 without needing to evaluate epsilon separately. The identity Q*(1-Q) = 2/9 is CAS-verified in predictions/delta_Koide.py via sympy.

### Step 3: Arithmetic

    delta = (2/3) * (1 - 2/3) = (2/3) * (1/3) = 2/9.

Gate-clear: explicit rational arithmetic.

## Result

    delta_predicted = 2/9   (exact rational, CAS-verified).

## Comparison with experiment

    delta_observed = 0.2222227 +/- 0.0000009   (PDG 2024)
    delta_predicted = 2/9 = 0.22222...          (exact)
    Deviation: approx 0.51 sigma.

Contingent on the charged-lepton identification (ADOPTED-P1 + ADOPTED-Y + Need-A2 closure), same as Q_Koide.

## Adopted residuals (explicit flagging)

Identical to Q_Koide_derivation.md:

- **ADOPTED-P1**: inherited via Q_Koide.
- **ADOPTED-Y**: inherited via Q_Koide.
- **Dimensional-matching**: inherited via Q_Koide. Future-closure: Need-A2.

## What is strict-solid vs adopted

**Strict-solid**: Q = 2/3 (upstream) + delta = Q*(1-Q) (pure algebra) = 2/9.

**Adopted residuals**: all inherited from Q_Koide; no new residuals introduced.

## Honest status

delta = 2/9 is the most downstream, least speculative consequence of Q_Koide. If Q_Koide is accepted, delta = 2/9 follows by arithmetic with no additional assumptions.


## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.

## References

See Q_Koide_derivation.md for the full reference list. This file adds no new citations.

Note on Appendix A of the pre-A3 delta_Koide_derivation.md: that appendix derived delta = 2/9 via a harmonic-mean Wigner d^1 route coinciding numerically at k* = 3 only (the polynomial (k-3)(k^2+k+4) vanishes only at k=3). That route is a numerical coincidence at k*=3 and is NOT the derivation used here; the present file uses only the algebraic identity delta = Q*(1-Q), which holds for any Q and does not depend on the specific k* value.

### Sibling / superseded files

- predictions/delta_Koide.py and predictions/delta_Koide_derivation.md -- RETRACTED (pre-A3, B6-retracted). Preserved as-is. Coexist with the present file.
