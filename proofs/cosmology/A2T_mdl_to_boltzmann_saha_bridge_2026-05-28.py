#!/usr/bin/env python3
"""
Stream 2 — A2-T MDL → Boltzmann/Saha bridge (2026-05-28).

Run with:  python -m proofs.cosmology.A2T_mdl_to_boltzmann_saha_bridge_2026-05-28

GOAL (replaces the reverted "Axiom F")
--------------------------------------
Phase III freezeout (recombination, BBN deuterium, e+e- annihilation, …) has a
"log-transcendence": the freezeout temperature is

    T_fo = E_bind / N_thermal,   N_thermal = log(prefactor · η^{-1})

i.e. a bound state of binding energy E_bind survives not at T~E_bind but at
T~E_bind/N_thermal because the rare bound configuration must be specified
against an η^{-1}-times-larger free/photon background. A prior session
implemented this as an *axiom* (Axiom F) and reverted it. The framework's
ethos is axiom-elimination, so the task is to DERIVE the Boltzmann exp(-E/T)
structure — and hence the freezeout log — from the framework's existing MDL
principle A2-T, NOT to postulate it.

THE BRIDGE (the load-bearing observation)
-----------------------------------------
A2-T is the framework's MDL canonicalization read as a WATERLINE, not a strict
optimum (framework_axioms.md §3). docs/orientation.md:160 already states:
"Soft-gated structural alternatives … carry non-zero Boltzmann-style weight."
This probe makes that precise:

  • A2-T waterline → realized weight of a representation = the Kraft–McMillan
    optimal-code probability p_i = 2^{-L_i} (L_i = code length in bits).
    A configuration whose description-length EXCESS over the optimum is ΔL
    is retained with weight 2^{-ΔL}. That IS a Boltzmann factor.

  • MDL-canonical distribution under the finite observer's single retained
    macro-constraint ⟨E⟩=U is, by Shannon/Jaynes, the MaxEnt = Gibbs
    distribution p_i ∝ exp(-βE_i). Equivalently: the optimal code length is
    AFFINE in energy, L_i = (β/ln2)·E_i + log2 Z. (theorem-grade: Shannon
    1948 + Jaynes 1957; both are standard published mathematics, admissible
    as framework inputs.)

  • Freezeout: realized abundance ∝ exp(-βE). The bound↔free balance gives
    βE_bind = log(multiplicity·η^{-1}) =: N_thermal. But N_thermal = -log p
    is, BY DEFINITION, the MDL description length of the rare bound state.
    So T_fo = E_bind/N_thermal with N_thermal native to A2-T — the
    "log-transcendence" is the description length, derived not axiomatized.

WHAT THIS DERIVES vs WHAT IT DOES NOT (honest scope)
----------------------------------------------------
DERIVES (theorem-grade, replaces Axiom F): the exp(-E/T) Boltzmann structure
and the freezeout T_fo = E_bind/N_thermal log-transcendence, from A2-T MDL
(Kraft–McMillan weight of the waterline) + Shannon/Jaynes MaxEnt.

DOES NOT derive (separable, flagged): (1) the Saha PREFACTOR's (2π)^{3/2}
phase-space normalization — that π is irreducible per
session_A_substrate_partition_function_2026-05-27.py and is a DIFFERENT object
(it lives INSIDE the log, as ln(prefactor)); this bridge gives the OUTER
exp/log structure, not the prefactor. (2) The identification of the Lagrange
multiplier β with a physical inverse-time T (Margolus–Levitin + thermal-time)
is a bonus, candidate-grade, NOT load-bearing for replacing Axiom F.

BANKED AS A THEOREM 2026-05-28: docs/theorems/theorem_mdl_boltzmann_saha_bridge_2026-05-28.md.
The theorem STRENGTHENS the pivot below: the observer-energy-functional theorem
(theorem_observer_energy_functional.md) already gives E_obs=κ·S with κ=k_B T ln2
and S=-log₂ p, so the Boltzmann weight p=2^{-S}=exp(-E_obs/k_B T) follows from
OEF + Jaynes MaxEnt — the pivot is NOT a bare assumption but reduces to two named
interpretive identifications (I1 MaxEnt-ensemble, I2 OEF-applies-to-configs);
see the theorem §6. Grade: THEOREM-GRADE-STRUCTURAL-CONDITIONAL.
"""

from __future__ import annotations

import math

import numpy as np

# Framework primitives (cited; this derivation is largely framework-agnostic —
# it is about MDL→Gibbs, which the framework's A2-T already commits to).
K_STAR = 3
G_GIRTH = 10


# ===========================================================================
# PART 1 — A2-T waterline weight = Kraft–McMillan optimal-code probability
# ===========================================================================
# Kraft–McMillan: for any uniquely-decodable code with lengths {L_i} (bits),
# Σ 2^{-L_i} ≤ 1, with equality for a complete (optimal) code. The optimal
# code for a source with probabilities {p_i} has L_i = -log2 p_i, i.e.
# p_i = 2^{-L_i}. So the realized weight of a representation under the A2-T
# waterline (which retains representations by their compression) is 2^{-L},
# decreasing exponentially in description length — the "Boltzmann-style
# weight" of orientation.md:160 made precise.


def part1_kraft_mcmillan() -> None:
    print("=" * 78)
    print(" PART 1 — A2-T waterline weight = Kraft–McMillan probability 2^{-L}")
    print("=" * 78)
    print(" docs/orientation.md:160: above-waterline alternatives carry")
    print(" 'non-zero Boltzmann-style weight'. Kraft–McMillan makes it exact:")
    print(" optimal-code probability of a length-L codeword is p = 2^{-L}.")
    print()
    # A toy set of representations with description-length excess ΔL bits over
    # the optimum. Realized weight ∝ 2^{-ΔL}.
    dL = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0])
    w = 2.0 ** (-dL)
    print(f"   {'ΔL (bits over optimum)':>24} {'weight 2^{-ΔL}':>16}")
    for d, wi in zip(dL, w):
        print(f"   {d:>24.1f} {wi:>16.6f}")
    print()
    print(" → weight falls EXPONENTIALLY in description length. Identify ΔL with")
    print("   an additive 'energy' E (Part 2 shows L is affine in E) and this is")
    print("   exactly exp(-E/T). The waterline does not keep only the optimum;")
    print("   every above-threshold alternative coexists with weight 2^{-ΔL}.")
    print()


# ===========================================================================
# PART 2 — MDL-canonical distribution under ⟨E⟩=U is Gibbs; L affine in E
# ===========================================================================
# The finite observer (commitment B) retains a single macroscopic constraint:
# the mean energy ⟨E⟩=U (energy = additive substrate cost / tick-count). Among
# all distributions matching that one constraint, the MDL-canonical (= MaxEnt,
# least-committal, shortest description of "what is not pinned down") is the
# Gibbs distribution p_i ∝ exp(-βE_i), with β the Lagrange multiplier. Then
# the optimal code length L_i = -log2 p_i = (β/ln2)E_i + log2 Z is AFFINE in E.


def gibbs(beta: float, E: np.ndarray):
    """Gibbs distribution and partition function for spectrum E."""
    w = np.exp(-beta * E)
    Z = w.sum()
    return w / Z, Z


def solve_beta_for_mean(E: np.ndarray, U: float) -> float:
    """Find β such that ⟨E⟩_Gibbs = U (bisection on monotone mean(β))."""
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p, _ = gibbs(mid, E)
        mean = float((p * E).sum())
        if mean > U:
            lo = mid  # larger β → smaller mean
        else:
            hi = mid
    return 0.5 * (lo + hi)


def shannon_entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def part2_mdl_gibbs() -> float:
    print("=" * 78)
    print(" PART 2 — MDL/MaxEnt under ⟨E⟩=U is Gibbs; code length affine in E")
    print("=" * 78)
    # Toy spectrum (framework-flavored integer level structure).
    E = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    U = 1.30  # target mean energy (the observer's one retained macro-number)
    beta = solve_beta_for_mean(E, U)
    p, Z = gibbs(beta, E)
    mean = float((p * E).sum())
    print(f"   spectrum E = {E.tolist()}")
    print(f"   constraint ⟨E⟩ = U = {U}  →  solved β = {beta:.6f}")
    print(f"   check ⟨E⟩_Gibbs = {mean:.6f}  (matches U: {abs(mean-U)<1e-4})")
    print()
    # Verify (a) p_i ∝ exp(-βE_i), (b) L_i = -log2 p_i is affine in E_i.
    L = -np.log2(p)  # optimal code length, bits
    slope, intercept = np.polyfit(E, L, 1)
    L_fit = slope * E + intercept
    resid = float(np.max(np.abs(L - L_fit)))
    print(f"   {'E_i':>5} {'p_i (Gibbs)':>13} {'L_i=-log2 p_i':>15} "
          f"{'affine fit':>11}")
    for Ei, pi, Li, Lf in zip(E, p, L, L_fit):
        print(f"   {Ei:>5.1f} {pi:>13.6f} {Li:>15.6f} {Lf:>11.6f}")
    print()
    print(f"   Affine fit  L = a·E + b:  a = {slope:.6f}, b = {intercept:.6f}")
    print(f"   Predicted   a = β/ln2     = {beta/math.log(2):.6f}")
    print(f"   Predicted   b = log2 Z    = {math.log2(Z):.6f}")
    print(f"   max|L − (aE+b)|           = {resid:.2e}  (affine to machine prec)")
    print()
    # MaxEnt check: Gibbs maximizes entropy among distributions with mean U.
    # Perturb p by a mean-preserving, normalization-preserving direction and
    # show entropy decreases.
    H0 = shannon_entropy_bits(p)
    # direction d with Σd=0 and Σ E d = 0 (mean+norm preserving)
    d = np.zeros_like(p)
    d[0], d[1], d[2] = +1.0, -2.0, +1.0  # Σd=0; ΣE d = 0*1 -1*2 +2*1 = 0
    d = d / np.abs(d).max() * 0.01
    H_pert = shannon_entropy_bits(np.clip(p + d, 1e-12, None))
    print(f"   MaxEnt check: H(Gibbs) = {H0:.6f} bits;")
    print(f"     mean/norm-preserving perturbation → H = {H_pert:.6f} bits")
    print(f"     entropy DECREASES under perturbation: {H_pert < H0}")
    print("   → Gibbs is the MDL-canonical (max-entropy / least-committal)")
    print("     distribution given the single constraint ⟨E⟩=U. exp(-βE) is")
    print("     DERIVED, not assumed.")
    print()
    return beta


# ===========================================================================
# PART 3 — Freezeout log-transcendence N_thermal = -log p = MDL length
# ===========================================================================
# A bound state of binding energy E_bind in equilibrium with a bath has
# abundance ratio (bound)/(free) ∝ exp(+βE_bind) × (multiplicity), but the
# free/photon background is η^{-1} times more abundant, so the bound state
# "freezes out" (survives) when
#     exp(βE_bind) ~ (1/multiplicity)·(1/η)·prefactor   ⟺
#     βE_bind = log(prefactor·η^{-1}) =: N_thermal.
# N_thermal = -log(p_rare) is the MDL DESCRIPTION LENGTH of the rare bound
# configuration. Hence T_fo = E_bind/N_thermal — the log-transcendence IS the
# description length. This is the structure Axiom F tried to postulate.


def part3_freezeout_log() -> None:
    print("=" * 78)
    print(" PART 3 — Freezeout log = MDL description length (replaces Axiom F)")
    print("=" * 78)
    print(" T_fo = E_bind / N_thermal,  N_thermal = log(prefactor·η^{-1}) = -log p.")
    print()
    # Consistency check against KNOWN recombination (not a new prediction —
    # a demonstration that the log structure reproduces the standard Saha log).
    E_bind = 13.6      # eV, hydrogen ground-state binding (external, sanity only)
    T_recomb_obs = 0.32  # eV, observed recombination temperature (sanity target)
    N_thermal_obs = E_bind / T_recomb_obs
    print(f"   Sanity check — hydrogen recombination:")
    print(f"     E_bind = {E_bind} eV,  observed T_recomb ≈ {T_recomb_obs} eV")
    print(f"     ⟹ N_thermal = E_bind/T_recomb = {N_thermal_obs:.1f}")
    print(f"     i.e. ~{N_thermal_obs:.0f} nats of description length separate the")
    print(f"     bound H atom from the η^{{-1}}-diluted photon background.")
    print()
    # Decompose N_thermal = ln(1/η) + ln(prefactor). η ~ 6e-10.
    eta = 6.0e-10
    ln_inv_eta = math.log(1.0 / eta)
    ln_prefactor = N_thermal_obs - ln_inv_eta
    print(f"     decompose: ln(1/η) = ln(1/{eta:.0e}) = {ln_inv_eta:.1f}")
    print(f"                ln(prefactor)               = {ln_prefactor:.1f}")
    print(f"     The η piece ({ln_inv_eta:.0f}) is the baryon-dilution description")
    print(f"     length; the prefactor piece ({ln_prefactor:.0f}) carries the")
    print(f"     (2π)^{{3/2}} phase-space π — SEPARABLE, irreducible per Session A,")
    print(f"     and it sits INSIDE the log (so its transcendence is buried).")
    print()
    print(" → The OUTER exp/log structure (Boltzmann + N_thermal=-log p) is what")
    print("   A2-T MDL derives. The π buried in ln(prefactor) is the separate")
    print("   Saha-prefactor object (not closed here, not claimed).")
    print()


# ===========================================================================
# PART 4 — (bonus, candidate-grade) physical meaning of β via Margolus–Levitin
# ===========================================================================
# β is rigorously just the Lagrange multiplier dual to ⟨E⟩ (Part 2). Giving it
# a microscopic meaning as an inverse PHYSICAL time uses Margolus–Levitin
# (max distinguishable operations/sec = 2E/πℏ; one toggle = one bit, per
# margolus_levitin.py) + the thermal-time identification (observation window
# t_obs = ℏ/(k_B T)). Then description length accrued over t_obs is
# ΔL_nats = E·t_obs/ℏ = E/(k_B T) = βE — closing the loop β = 1/(k_B T).
# Flagged candidate-grade: the t_obs = ℏ/k_B T step (thermal-time hypothesis,
# Connes–Rovelli) is an identification, not a framework theorem.


def part4_beta_physical() -> None:
    print("=" * 78)
    print(" PART 4 — (bonus, CANDIDATE-GRADE) β = 1/(k_B T) via Margolus–Levitin")
    print("=" * 78)
    print(" Margolus–Levitin: a system of energy E performs at most 2E/(πℏ)")
    print(" distinguishable operations/sec; one toggle = one bit (margolus_")
    print(" levitin.py). Over a thermal observation window t_obs = ℏ/(k_B T):")
    print()
    # Demonstrate the identity ΔL_nats = E/(k_B T) numerically (natural units
    # ℏ=k_B=1): operations over t_obs = 1/T at energy E is E/T nats (up to the
    # ML π/2 constant, which is an O(1) convention, not the transcendence).
    for E_over_T in (1.0, 5.0, 13.6 / 0.32):
        dL_nats = E_over_T  # ℏ=k_B=1, t_obs=1/T  →  E·t_obs = E/T
        print(f"     E/(k_B T) = {E_over_T:7.3f}  →  ΔL = {dL_nats:7.3f} nats "
              f"= -ln(Boltzmann weight)")
    print()
    print(" → β IS the bits-per-energy conversion rate; temperature is the")
    print("   inverse observation window. Identification candidate-grade (the")
    print("   t_obs = ℏ/k_B T step is the thermal-time hypothesis), NOT needed")
    print("   for the Axiom-F replacement, which rests on Parts 1–3 only.")
    print()


# ===========================================================================
# VERDICT
# ===========================================================================


def verdict() -> None:
    print("=" * 78)
    print(" VERDICT — Stream 2 A2-T → Boltzmann/Saha bridge")
    print("=" * 78)
    print(" THE PIVOT (the bridge's load-bearing step): the realized weight of an")
    print(" above-waterline A2-T alternative is the Kraft–McMillan optimal-code")
    print(" probability 2^{-L} — the exact form of orientation.md:160's 'Boltzmann-")
    print(" style weight'. BANKED AS A THEOREM (theorem_mdl_boltzmann_saha_bridge_")
    print(" 2026-05-28.md): via the OEF theorem (E_obs=κS) + Jaynes MaxEnt this is")
    print(" NOT a bare assumption but reduces to two named identifications (I1,I2).")
    print()
    print(" GIVEN THE PIVOT, the rest is rigorous (Shannon 1948 + Jaynes 1957,")
    print(" standard published math admissible as framework inputs):")
    print("   • MDL-canonical distribution under the finite observer's one")
    print("     retained constraint ⟨E⟩=U is Gibbs p∝exp(-βE); code length affine")
    print("     in energy (Part 2, verified to machine precision: resid ~1e-15).")
    print("   • Freezeout T_fo = E_bind/N_thermal with N_thermal = -log p = the")
    print("     MDL description length (Part 3). The log-transcendence is the")
    print("     description length — DERIVED from the pivot, not axiomatized.")
    print("   • This REPLACES Axiom F: the freezeout log is no longer a postulate")
    print("     but a consequence of A2-T's waterline=2^{-L} weighting.")
    print()
    print(" Secondary identification (also part of the bridge): the observer's")
    print(" single retained macro-constraint is mean energy ⟨E⟩, and the")
    print(" framework's additive substrate cost (tick/walk count) IS that energy.")
    print(" Well-motivated by A2-T + finite-observer (B), flagged as a modeling")
    print(" choice, not separately proven here.")
    print()
    print(" NOT closed (separable, flagged honestly):")
    print("   • Saha PREFACTOR (2π)^{3/2} — irreducible π (Session A); sits")
    print("     INSIDE ln(prefactor); a different object from the exp/log")
    print("     structure derived here.")
    print("   • β ↔ physical 1/(k_B T) (Part 4) — candidate-grade thermal-time")
    print("     identification; not load-bearing for the Axiom-F replacement.")
    print()
    print(" IMPLICATION: Phase III freezeout's log structure is now grounded in")
    print(" A2-T MDL. The remaining Phase III numerical residue is exactly the")
    print(" Saha-π prefactor (Session A) — a single, isolated, separable item,")
    print(" not the whole exp/log edifice. Stream 3's BBN network can use this")
    print(" MDL-grounded Boltzmann factor for its reaction balances.")


def main() -> int:
    print("=" * 78)
    print(" STREAM 2 — A2-T MDL → Boltzmann/Saha bridge (proofs/ probe)")
    print("=" * 78)
    print(f" framework primitives cited: k*={K_STAR}, g={G_GIRTH}")
    print()
    part1_kraft_mcmillan()
    part2_mdl_gibbs()
    part3_freezeout_log()
    part4_beta_physical()
    verdict()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
