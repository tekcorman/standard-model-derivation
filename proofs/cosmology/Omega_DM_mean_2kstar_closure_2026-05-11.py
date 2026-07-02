#!/usr/bin/env python3
"""
Mean = 2k* closure verification — Ω_DM/Ω_m chain.

Verifies that the per-vertex event count mean of 2k* (the load-bearing input
to the Poisson(2k*) tail giving Ω_DM/Ω_m = 1 - 61·e⁻⁶) follows from
predictions-folder primitives via the substrate observation cycle
(return-to-original-state Markov cycle).

Primitives (all theorem-grade in predictions/):
  - k*          from predictions/k_star.py
  - p_create    from predictions/S_fresh_derivation.md (Beta(1,1) predictive)
  - p_destroy   from predictions/lambda_toggle_rate_derivation.md (Beta(2,1) predictive)
  - λ_toggle    from predictions/lambda_toggle_rate_derivation.md (Markov stationary)

Chain (per predictions/Omega_DM_over_Omega_m_derivation.md Step 2):
  T_cycle      = 1/p_create + 1/p_destroy
  λ_toggle     = 2·p_create·p_destroy / (p_create + p_destroy)
  events/edge/cycle  = λ_toggle · T_cycle
  events/vertex/cycle = k* · (λ_toggle · T_cycle)  ⇒ should equal 2k*

Then Poisson(2k*) tail at k = k* gives Ω_DM/Ω_m.
"""

from fractions import Fraction
import math
import sys

# --- Theorem-grade primitives (exact rationals) -----------------------

k_star = 3                              # predictions/k_star.py
p_create = Fraction(1, 2)               # predictions/S_fresh_derivation.md
p_destroy = Fraction(1, 3)              # predictions/lambda_toggle_rate_derivation.md

# --- Derived quantities (closed-form, exact rationals) ---------------

T_cycle = Fraction(1, 1) / p_create + Fraction(1, 1) / p_destroy
lambda_toggle = (2 * p_create * p_destroy) / (p_create + p_destroy)
events_per_edge_per_cycle = lambda_toggle * T_cycle
events_per_vertex_per_cycle = k_star * events_per_edge_per_cycle

mean_target = 2 * k_star  # what we want to verify

# --- Verification gates ----------------------------------------------

print("Substrate primitives (theorem-grade in predictions/):")
print(f"  k*         = {k_star}")
print(f"  p_create   = {p_create}")
print(f"  p_destroy  = {p_destroy}")
print()
print("Derived (exact arithmetic):")
print(f"  T_cycle              = 1/p_create + 1/p_destroy = {T_cycle} Planck steps")
print(f"  λ_toggle             = 2 p_c p_d / (p_c + p_d)  = {lambda_toggle} events/edge/step")
print(f"  events/edge/cycle    = λ_toggle · T_cycle       = {events_per_edge_per_cycle}")
print(f"  events/vertex/cycle  = k* · (above)             = {events_per_vertex_per_cycle}")
print()
print(f"Target mean = 2k* = {mean_target}")

assert T_cycle == 5, f"T_cycle expected 5, got {T_cycle}"
assert lambda_toggle == Fraction(2, 5), f"λ_toggle expected 2/5, got {lambda_toggle}"
assert events_per_edge_per_cycle == 2, f"events/edge/cycle expected 2, got {events_per_edge_per_cycle}"
assert events_per_vertex_per_cycle == mean_target, (
    f"events/vertex/cycle = {events_per_vertex_per_cycle}, expected 2k* = {mean_target}"
)
print()
print(f"✓ Mean = 2k* verified exactly from substrate primitives.")

# --- Downstream: Poisson(2k*) tail -----------------------------------

mu = float(mean_target)
visible_cdf = sum(mu**j * math.exp(-mu) / math.factorial(j) for j in range(k_star + 1))
dark_fraction = 1.0 - visible_cdf

# Closed-form check: visible = 61·e⁻⁶ for k* = 3
visible_closed_form = 61 * math.exp(-6)
assert abs(visible_cdf - visible_closed_form) < 1e-15, (
    f"visible_cdf = {visible_cdf}, closed-form = {visible_closed_form}"
)

print()
print(f"Poisson({mean_target}) CDF at k = k* = {k_star}:")
print(f"  visible fraction  = {visible_cdf:.6f}  (closed form: 61·e⁻⁶ = {visible_closed_form:.6f})")
print(f"  dark fraction     = {dark_fraction:.6f}  (= 1 - 61·e⁻⁶)")
print()
print(f"Planck 2018: Ω_DM/Ω_m = 0.842 ± 0.016")
sigma_pull = (dark_fraction - 0.842) / 0.016
print(f"Framework:   Ω_DM/Ω_m = {dark_fraction:.4f}  → pull = {sigma_pull:+.2f}σ")
print()
print(f"✓ Closure verified end-to-end.")
