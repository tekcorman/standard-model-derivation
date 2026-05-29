# Derivation of epsilon_Koide under A3 Born rule -- color-sector amplitude identity

> **STATUS UPDATE 2026-05-08.** Same status note as `Q_Koide_derivation.md`: ADOPTED-P1 + ADOPTED-Y were **CLOSED via A5** on 2026-04-19 (adoption register lines 42, 70); Need-A2 generation-Z₃ existence **CLOSED** via R3 + M1.B chain (April 2026, rediscovered 2026-05-08); M_gen non-degeneracy **CLOSED** via generic argument 2026-05-08. Under standing slate {A1 + A2-T + A3-T + A5}, **ε² = 2 is theorem-grade for the charged-lepton sector**, not just the color-sector amplitude identity. The "modulo P1+Y" framing in older sections is stale.

## Abstract

We derive the color-sector Koide amplitude parameter

    epsilon^2 = 4 * mu_omega / mu_trivial = 2   (exact)

at theorem grade under A1 + A2-T + A3-T (A2-T, A3-T are derived theorems per docs/framework/framework_axioms.md §10). The derivation follows the same chain as predictions/Q_Koide_derivation.md: the C_3 multiplicities (4, 2, 2) of the 8-dim Ramanujan subspace of B(P) on srs determine, via Jaynes max-entropy + A2 (ADOPTED-P1) and the A3-derived Born rule (CDP 2011 Theorem 25, ADOPTED-Y), the substrate amplitudes in the Koide parametric form

    sqrt(m_j) = sqrt(M) * (1 + epsilon * cos(2*pi*j/3)),   j = 0, 1, 2.

Matching to the C_3 Fourier-transformed amplitudes amp_j = sqrt(mu_trivial) + 2*sqrt(mu_omega)*cos(2*pi*j/3) from Steps 3-4 of Q_Koide gives sqrt(M) = sqrt(mu_trivial) = 2 and epsilon = 2*sqrt(mu_omega)/sqrt(mu_trivial), so epsilon^2 = 4*mu_omega/mu_trivial = 4*2/4 = 2 exactly.

The identification of this color-sector identity with the charged-lepton Koide amplitude parameter requires the same two adopted structural postulates as Q_Koide: ADOPTED-P1 (Ramanujan-subspace support) and ADOPTED-Y (substrate amplitudes = Yukawa-coupling amplitudes).

## Framework axioms invoked

- **A1** (binary self-inverse toggle): same as Q_Koide_derivation.md.
- **A2** (MDL canonicalization): same.
- **A3** (MDL canonicalization is partial trace over abstract H_aux; CDP 2011): same.

## Cited mathematical theorems

- **Chiribella, D'Ariano, Perinotti** (2011). Phys. Rev. A 84, 012311. Theorem 25: Born rule.
- **Jaynes, E.T.** (1957). Phys. Rev. 106, 620-630. Max-entropy under support constraint.
- **Serre, J.-P.** (1977). Linear Representations of Finite Groups. Springer GTM 42. Section 2.3 (C_3 Fourier transform).

## Upstream closed prediction files

- predictions/k_star.py (k* = 3)
- predictions/d_spatial.py (d = 3)
- predictions/g_girth.py (g = 10)
- predictions/B_P_doubly_degenerate_h.py ((4, 2, 2) C_3 multiplicities)
- predictions/observer_hilbert_space.py (Born rule from A3 + CDP 2011)
- predictions/Q_Koide.py (Q = 2/3; provides Bernoulli cross-check)

## Derivation

### Steps 1-4 (upstream, identical to Q_Koide_derivation.md)

See Q_Koide_derivation.md Steps 1-4 verbatim. The output of Step 4 is the three generation-indexed substrate amplitudes

    amp_j = sqrt(mu_trivial) + sqrt(mu_omega)*omega^j + sqrt(mu_omega_bar)*omega^{-j}
          = 2 + sqrt(2)*omega^j + sqrt(2)*omega^{-j}
          = 2 + 2*sqrt(2)*cos(2*pi*j/3).

### Step 5: Koide-form matching

The Koide parametric form is

    amp_j = sqrt(M) * (1 + epsilon * cos(2*pi*j/3))       (*)

Matching the constant term: sqrt(M) = 2, so M = 4 = mu_trivial.

Matching the cosine amplitude: sqrt(M) * epsilon = 2*sqrt(2) = 2*sqrt(mu_omega).
Solving: epsilon = 2*sqrt(mu_omega) / sqrt(mu_trivial) = 2*sqrt(2)/2 = sqrt(2).

Hence:

    epsilon^2 = 4 * mu_omega / mu_trivial = 4 * 2 / 4 = 2.

Gate-clear: explicit algebra. CAS-verified in predictions/epsilon_Koide.py.

### Step 6: Algebraic cross-check via Bernoulli-moment identity

The Bernoulli-moment identity of the Koide parametrisation gives epsilon^2 = 6*Q - 2. Substituting Q = 2/3 (from Q_Koide):

    epsilon^2 = 6*(2/3) - 2 = 4 - 2 = 2.  ✓

Gate-clear: algebraic identity (CAS-verified in delta_Koide.py at generic (M, epsilon) using sympy), chain-importing from Q_Koide.

### Step 7: Born rule (ADOPTED-Y)

By the same ADOPTED-Y as Q_Koide (substrate amplitudes = Yukawa amplitudes), the Born-rule squared moduli m_j = |amp_j|^2 are the mass-like quantities. Under this adoption, epsilon^2 = 2 is the color-sector amplitude ratio entering the observed charged-lepton Koide parametrisation.

## Result

    epsilon_predicted = sqrt(2),   epsilon^2 = 2   (exact rational, CAS-verified).

## Comparison with experiment

    epsilon_observed = sqrt(2) approx 1.414209 +/- 0.000011   (PDG 2024)
    epsilon_predicted = sqrt(2) = 1.41421356...                (exact)
    Deviation: approx 0.43 sigma.

As with Q_Koide, this comparison is contingent on the charged-lepton identification requiring ADOPTED-P1 + ADOPTED-Y + Need-A2 closure.

## Adopted residuals (explicit flagging)

Identical to Q_Koide_derivation.md:

- **ADOPTED-P1**: substrate amplitudes supported on V_Ram. Future-closure: Feshbach projection (an internal scoping attempt on the canonical-reading question, Section 3 Step C; see also predictions/ADOPTED_P1_ramanujan_support_derivation.md for partial progress under ADOPTED-CS).
- **ADOPTED-Y**: substrate amplitudes = Yukawa-coupling amplitudes. Future-closure: Sprint 7a v_Higgs + Higgs mechanism.
- **Dimensional-matching**: Fourier index j identified with generation label. Future-closure: Need-A2.

## What is strict-solid vs adopted

**Strict-solid**: (4, 2, 2) multiplicities (upstream) + Jaynes on V_Ram (given P1) + Koide-form matching algebra + epsilon^2 = 2 as a closed-form rational identity.

**Adopted residuals**: ADOPTED-P1, ADOPTED-Y, dimensional-matching (same three as Q_Koide).

## Honest status

epsilon^2 = 2 is a direct algebraic consequence of the same (4, 2, 2) isotypic content that gives Q = 2/3. The derivation adds no new axioms, no new adopted residuals, and no new structural input beyond Q_Koide. It is strictly a downstream consequence of the same calculation.


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

### Sibling / superseded files

- predictions/epsilon_Koide.py and predictions/epsilon_Koide_derivation.md -- RETRACTED (pre-A3, B6-retracted). Preserved as-is. Coexist with the present file.
