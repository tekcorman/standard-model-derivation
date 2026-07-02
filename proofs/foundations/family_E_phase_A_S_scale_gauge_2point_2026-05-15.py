#!/usr/bin/env python3
"""
proofs/foundations/family_E_phase_A_S_scale_gauge_2point_2026-05-15.py

*** RETRACTED 2026-05-15 EOD+15 — STALE BASE PREDICTIONS ***
This probe used M_Z_pred=91.97, m_W_pred=80.69 (read from M_Z.py DOCSTRING
HEADER — pre-α_GUT-DC stale values).  Live predictions are M_Z=91.5135,
m_W=80.2373 with OPPOSITE-SIGN residuals.  The "c_S=1/12 two-routes
convergent / magnitude in neighborhood" finding does NOT hold against
correct inputs (c_S target with live base ≈ 0.024, not 0.10/0.15).
See `family_E_phase_AB_CORRECTED_base_predictions_2026-05-15.py` (commit
c66bc02).  Caught by parameter_linter Checkpoint 1.  Preserved for record.

Phase A of the Family C + Family E joint derivation for M_Z/m_W cluster
(per `an internal working note
scoping_2026-05-15.md`).

GOAL: Derive S_scale (Family C sign-uniform piece) for the gauge-boson
2-point function via Route H (Hashimoto-spectral) and Route C (cycle-
counting), calibrated against v_Higgs c_v = 5/12.

EMPIRICAL TARGET:
  M_Z residual: +0.86% (predicted-too-high)
  m_W residual: +0.40% (predicted-too-high)
  Common shift S_scale ≈ -0.40% (negative correction to bring predicted down)
  Asymmetric piece R_asymmetric ≈ -0.46% extra on M_Z (Phase B target)

For S_scale ≈ -0.40%: c_S × α₁/(1-α₁) = 0.0040
  → c_S = 0.0040 / 0.0406 ≈ 0.099 ≈ 0.10

K-rational neighborhood of 0.10:
  1/k*² = 1/9 ≈ 0.111
  5/48 ≈ 0.104
  1/12 ≈ 0.083
  1/(N_atoms·k*) = 1/12 ≈ 0.083

ROUTE H (Hashimoto-spectral) — calibrated form:
For v_Higgs: c_v = (2(|E|-|V|) + 1) / (2|E|) = 5/12 on srs
- Numerator: marginal-mode (|λ|=1) dim of Hashimoto B operator
- Denominator: total NB Hilbert dim = 2|E|

For gauge-boson 2-point, the "relevant sector" candidates:
  (RH.1) Cycle-space dim H_1 = |E|-|V|+1 = 3 (one per crystal axis)
         → c = 3/12 = 1/4 = 0.25
  (RH.2) U(1)_Y subspace dim = 1 (single Cartan direction)
         → c = 1/12 ≈ 0.083
  (RH.3) Marginal mode minus cycle space = (2(|E|-|V|)+1) - (|E|-|V|+1)
         = (5 - 3) = 2 (perpendicular to cycles, at |λ|=1)
         → c = 2/12 = 1/6 ≈ 0.167
  (RH.4) Per-direction dim 1 / total per-cell dim = 1/(2|E|) = 1/12

ROUTE C (cycle-counting) — calibrated form:
For v_Higgs: c_v = n_g / (N_atoms · k*²) = 15/36 = 5/12
- Numerator: girth-cycle count per vertex = 15
- Denominator: per-cell counting normalization

For gauge-boson 2-point candidates:
  (RC.1) Cycle-axis dim / (N_atoms · k*) = 3 / 12 = 1/4
  (RC.2) 1 / (N_atoms · k*) = 1/12 ≈ 0.083
  (RC.3) k* / (N_atoms · k*²) = 1/(N_atoms · k*) = 1/12
  (RC.4) (2(|E|-|V|)+1) / (4 · N_atoms · k*²) = 5/(4·36) = 5/144 ≈ 0.035

PRE-DECLARED ABORT:
(CS.1) No Route H or Route C form gives c_S in K-rational neighborhood of 0.10
       AND calibrates to v_Higgs 5/12 → close NEG
(CS.2) Form derives but magnitude off > 50% from -0.40% target → close NEG
(CS.3) Form derives, calibrates, AND matches within sub-percent → PHASE A POSITIVE
"""
from __future__ import annotations
from fractions import Fraction
import numpy as np

# ---------------------------------------------------------------------------
# Framework constants
# ---------------------------------------------------------------------------
k_star = 3
g = 10
N_ATOMS = 4
N_EDGES = 6  # bonds per primitive cell
N_DIRECTED_EDGES = 2 * N_EDGES  # = 12
H1_DIM = N_EDGES - N_ATOMS + 1  # = 3 (cycle space dim per cell)
MARGINAL_DIM = 2 * (N_EDGES - N_ATOMS) + 1  # = 5 (Stark-Terras)

alpha_1_bare = Fraction(k_star - 1, k_star) ** (g - 2)  # = 256/6561
alpha_1_full = Fraction(5, 3) * alpha_1_bare  # = 1280/19683
n_g = 15  # girth-cycle count per vertex (theorem-grade)

# Family C template factor
alpha_factor = float(alpha_1_bare) / (1.0 - float(alpha_1_bare))

print("=" * 78)
print("  Phase A — Family C S_scale derivation for gauge-boson 2-point")
print("=" * 78)
print()
print(f"  Framework constants: k*={k_star}, g={g}, N_atoms={N_ATOMS}, |E|={N_EDGES}")
print(f"  α₁_bare = (2/3)^8 = {float(alpha_1_bare):.6f}")
print(f"  α₁_bare / (1 - α₁_bare) = {alpha_factor:.6f}")
print(f"  H_1 (cycle space dim) = {H1_DIM}")
print(f"  Marginal-mode dim (Stark-Terras) = {MARGINAL_DIM}")
print(f"  Total NB Hilbert dim per cell = {N_DIRECTED_EDGES}")
print()


# ---------------------------------------------------------------------------
# Calibration: v_Higgs c_v = 5/12 must hold on both routes
# ---------------------------------------------------------------------------
print("=" * 78)
print("Calibration check: v_Higgs c_v = 5/12")
print("=" * 78)
print()

c_v_route_H = Fraction(MARGINAL_DIM, N_DIRECTED_EDGES)
c_v_route_C = Fraction(n_g, N_ATOMS * k_star ** 2)
print(f"  Route H: c_v = (2(|E|-|V|)+1) / (2|E|) = {MARGINAL_DIM}/{N_DIRECTED_EDGES} = {c_v_route_H}")
print(f"  Route C: c_v = n_g / (N_atoms · k*²)   = {n_g}/{N_ATOMS * k_star ** 2} = {c_v_route_C}")
print(f"  Both routes converge: {c_v_route_H == c_v_route_C} (= 5/12) ✓")
print()


# ---------------------------------------------------------------------------
# Empirical target
# ---------------------------------------------------------------------------
M_Z_PDG = 91.1876
m_W_PDG = 80.3692

# Predicted values (from existing M_Z.py, m_W.py)
M_Z_pred = 91.97
m_W_pred = 80.69

# Residuals (predicted-observed)/observed
delta_M_Z = (M_Z_pred - M_Z_PDG) / M_Z_PDG
delta_m_W = (m_W_pred - m_W_PDG) / m_W_PDG

S_scale_target = -min(delta_M_Z, delta_m_W)  # common shift = smaller of two (m_W)
R_asymmetric_target = delta_M_Z - delta_m_W  # extra on M_Z over m_W
target_c_S = abs(S_scale_target) / alpha_factor

print("=" * 78)
print("Empirical decomposition")
print("=" * 78)
print()
print(f"  M_Z prediction error: +{delta_M_Z * 100:.4f}% (predicted-too-high)")
print(f"  m_W prediction error: +{delta_m_W * 100:.4f}% (predicted-too-high)")
print(f"  Common shift S_scale (negative correction):  {S_scale_target * 100:+.4f}%")
print(f"  Asymmetric extra on M_Z:                     {R_asymmetric_target * 100:+.4f}%")
print()
print(f"  Target c_S = |S_scale| / (α₁/(1-α₁)) = {target_c_S:.6f}")
print(f"  K-rational neighborhood: 1/k*² = {1/9:.6f}, 5/48 = {5/48:.6f}, "
      f"1/12 = {1/12:.6f}")
print()


# ---------------------------------------------------------------------------
# Route H candidates
# ---------------------------------------------------------------------------
print("=" * 78)
print("Route H (Hashimoto-spectral) candidates for c_S on gauge-boson 2-point")
print("=" * 78)
print()

route_H_candidates = {
    "RH.1: H_1 (cycle space) / 2|E|":
        Fraction(H1_DIM, N_DIRECTED_EDGES),
    "RH.2: 1 (U(1)_Y direction) / 2|E|":
        Fraction(1, N_DIRECTED_EDGES),
    "RH.3: (Marginal - H_1) / 2|E|":
        Fraction(MARGINAL_DIM - H1_DIM, N_DIRECTED_EDGES),
    "RH.4: 2 (W^± pair) / 2|E|":
        Fraction(2, N_DIRECTED_EDGES),
    "RH.5: H_1 / Marginal":
        Fraction(H1_DIM, MARGINAL_DIM),
    "RH.6: 1 / Marginal":
        Fraction(1, MARGINAL_DIM),
    "RH.7: 2 / Marginal (vector-doubling)":
        Fraction(2, MARGINAL_DIM),
}

print(f"{'Form':<48} {'c_S':<10} {'c_S × α₁/(1-α₁)':<18} {'Match?'}")
print("-" * 96)
for label, c in route_H_candidates.items():
    pct = float(c) * alpha_factor * 100
    rel_err = abs(float(c) - target_c_S) / target_c_S
    match = "✓ within 5%" if rel_err < 0.05 else (
        "(within 20%)" if rel_err < 0.20 else f"({rel_err * 100:.0f}% off)")
    print(f"  {label:<46} {str(c):<10} {pct:>+10.4f}%  {match}")
print()


# ---------------------------------------------------------------------------
# Route C candidates
# ---------------------------------------------------------------------------
print("=" * 78)
print("Route C (cycle-counting) candidates for c_S on gauge-boson 2-point")
print("=" * 78)
print()

route_C_candidates = {
    "RC.1: 1 / (N_atoms · k*²)":
        Fraction(1, N_ATOMS * k_star ** 2),
    "RC.2: H_1 / (N_atoms · k*²)":
        Fraction(H1_DIM, N_ATOMS * k_star ** 2),
    "RC.3: k* / (N_atoms · k*²) = 1/(N_atoms · k*)":
        Fraction(k_star, N_ATOMS * k_star ** 2),
    "RC.4: n_g / (4 · N_atoms · k*²)":
        Fraction(n_g, 4 * N_ATOMS * k_star ** 2),
    "RC.5: (k*-1) / (N_atoms · k*²)":
        Fraction(k_star - 1, N_ATOMS * k_star ** 2),
    "RC.6: 1 / k*² (no atom factor)":
        Fraction(1, k_star ** 2),
    "RC.7: n_g / (k* · N_atoms · k*²)":
        Fraction(n_g, k_star * N_ATOMS * k_star ** 2),
}

print(f"{'Form':<55} {'c_S':<10} {'c_S × α₁/(1-α₁)':<18} {'Match?'}")
print("-" * 96)
for label, c in route_C_candidates.items():
    pct = float(c) * alpha_factor * 100
    rel_err = abs(float(c) - target_c_S) / target_c_S
    match = "✓ within 5%" if rel_err < 0.05 else (
        "(within 20%)" if rel_err < 0.20 else f"({rel_err * 100:.0f}% off)")
    print(f"  {label:<53} {str(c):<10} {pct:>+10.4f}%  {match}")
print()


# ---------------------------------------------------------------------------
# Convergence check: do any Route H + Route C candidates match?
# ---------------------------------------------------------------------------
print("=" * 78)
print("Two-routes convergence check")
print("=" * 78)
print()

convergent_pairs = []
for label_H, c_H in route_H_candidates.items():
    for label_C, c_C in route_C_candidates.items():
        if c_H == c_C:
            convergent_pairs.append((label_H, label_C, c_H))

if convergent_pairs:
    print(f"  Convergent (Route H == Route C):")
    for label_H, label_C, c in convergent_pairs:
        pct = float(c) * alpha_factor * 100
        rel_err = abs(float(c) - target_c_S) / target_c_S
        print(f"    c = {c}: {label_H} == {label_C}")
        print(f"      shift = {pct:+.4f}%  (target {S_scale_target*100:+.4f}%, rel err {rel_err*100:.1f}%)")
else:
    print(f"  No exact Route H == Route C convergence among candidates above.")
print()


# ---------------------------------------------------------------------------
# Best magnitude match (for descriptive purposes only — NOT closure)
# ---------------------------------------------------------------------------
print("=" * 78)
print("Magnitude ranking (DESCRIPTIVE ONLY)")
print("=" * 78)
print()

all_candidates = list(route_H_candidates.items()) + list(route_C_candidates.items())
ranked = sorted(all_candidates, key=lambda x: abs(float(x[1]) - target_c_S))
print(f"  Top 5 closest to target c_S = {target_c_S:.4f}:")
for label, c in ranked[:5]:
    pct = float(c) * alpha_factor * 100
    rel_err = abs(float(c) - target_c_S) / target_c_S
    print(f"    c = {c} = {float(c):.6f}  →  shift {pct:+.4f}%  ({rel_err*100:.1f}% off target)  {label}")
print()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("=" * 78)
print("Phase A verdict")
print("=" * 78)
print()

best_label, best_c = ranked[0]
best_pct = float(best_c) * alpha_factor * 100
best_err = abs(float(best_c) - target_c_S) / target_c_S
best_in_convergent = any(c == best_c for _, _, c in convergent_pairs)

print(f"  Best magnitude match: c_S = {best_c} from {best_label}")
print(f"    Shift: {best_pct:+.4f}% vs target {S_scale_target*100:+.4f}% "
      f"({best_err*100:.1f}% relative error)")
print(f"    Two-route convergence: {'YES' if best_in_convergent else 'NO'}")
print()

print(f"  Calibration discipline (master doc §8 rule 2):")
print(f"    Both routes for v_Higgs converge at c_v = 5/12 ✓ (built into RH.* and RC.* numerators)")
print(f"    Question: is the gauge-boson 2-point form a STRUCTURAL extension of v_Higgs form,")
print(f"    or numerology fishing in K-rational space?")
print()

print(f"  Pre-declared aborts:")
print(f"    (CS.1) No K-rational form derived AND calibrated → close NEG")
print(f"    (CS.2) Magnitude off > 50% → close NEG")
print(f"    (CS.3) Form derived, calibrates, matches sub-percent → PHASE A POS")
print()

# Verdict logic
in_neighborhood = best_err < 0.20
sub_percent = best_err < 0.05
if in_neighborhood and best_in_convergent:
    if sub_percent:
        print(f"  → (CS.3) PHASE A POSITIVE: {best_label} converges across routes,")
        print(f"           magnitude within 5% of target. Proceed to Phase B.")
    else:
        print(f"  → CONDITIONAL: {best_label} converges but {best_err*100:.0f}% off magnitude.")
        print(f"           Phase A NOT closed; needs deeper structural derivation.")
elif in_neighborhood:
    print(f"  → (CS.2) Magnitude in neighborhood (within 20%) but no Route H = Route C convergence.")
    print(f"           Single-route candidates exist but two-routes discipline NOT met.")
    print(f"           Phase A INCONCLUSIVE without Route convergence.")
else:
    print(f"  → (CS.1) Best candidate {best_err*100:.0f}% off; no structural derivation forced.")
    print(f"           Phase A → close NEG.")

print()


# ---------------------------------------------------------------------------
# Alternative decomposition: half-sum / half-difference
# ---------------------------------------------------------------------------
print("=" * 78)
print("Alternative decomposition (half-sum / half-difference)")
print("=" * 78)
print()

delta_Z = M_Z_PDG / M_Z_pred - 1  # multiplicative shift needed for M_Z
delta_W = m_W_PDG / m_W_pred - 1
S_alt = (delta_Z + delta_W) / 2  # common (sign-uniform)
R_alt = (delta_W - delta_Z) / 2  # asymmetric (m_W vs M_Z opposite)

print(f"  Multiplicative shifts needed (observed/predicted - 1):")
print(f"    δ_Z = {delta_Z * 100:+.4f}%  (M_Z predicted too high)")
print(f"    δ_W = {delta_W * 100:+.4f}%  (m_W predicted too high)")
print()
print(f"  Half-sum decomposition:")
print(f"    S = (δ_Z + δ_W)/2 = {S_alt * 100:+.4f}%  (sign-uniform, both M_Z and m_W shift down)")
print(f"    R = (δ_W - δ_Z)/2 = {R_alt * 100:+.4f}%  (asymmetric: m_W relative to M_Z)")
print()
print(f"  Δρ predicted from R: 4·R = {4 * R_alt * 100:+.4f}% (vs empirical +1.04%)")
print()
print(f"  Required c values for this decomposition:")
target_c_S_alt = abs(S_alt) / alpha_factor
target_c_E_alt = R_alt / alpha_factor
print(f"    c_S target = {target_c_S_alt:.6f}  (Family C, sign-uniform)")
print(f"    c_E target = {target_c_E_alt:.6f}  (Family E, asymmetric)")
print()

# Check c_E = 1/18 = c_S × (k*-1)/k* hypothesis
c_E_natural = Fraction(1, 18)
c_E_predicted = float(c_E_natural)
err_c_E = abs(c_E_predicted - target_c_E_alt) / target_c_E_alt
print(f"  HYPOTHESIS: c_E = c_S × (k*-1)/k* = (1/12) × (2/3) = 1/18 = {float(c_E_natural):.6f}")
print(f"    Actual target c_E = {target_c_E_alt:.6f}  ({err_c_E * 100:.2f}% off)")
print()

# Combined fit check with c_S = 1/12 + various c_E
print(f"  Combined fit check: c_S = 1/12 (Phase A convergent) + various c_E:")
print(f"  {'c_E':>10}  {'M_Z shift':>12}  {'m_W shift':>12}  {'δρ_pred':>10}  {'M_Z resid':>12}  {'m_W resid':>12}")
print("-" * 100)
c_S_fixed = float(Fraction(1, 12))
S_shift = -c_S_fixed * alpha_factor  # negative correction
for c_E_label, c_E_val in [("0", 0), ("1/18", float(Fraction(1, 18))),
                             ("1/9", float(Fraction(1, 9))), ("5/48", float(Fraction(5, 48)))]:
    R_shift = c_E_val * alpha_factor  # m_W gets +R_shift, M_Z gets -R_shift
    delta_M_Z_total = S_shift - R_shift
    delta_m_W_total = S_shift + R_shift
    # corrected predictions:
    M_Z_corr = M_Z_pred * (1 + delta_M_Z_total)
    m_W_corr = m_W_pred * (1 + delta_m_W_total)
    M_Z_resid = (M_Z_corr - M_Z_PDG) / M_Z_PDG * 100
    m_W_resid = (m_W_corr - m_W_PDG) / m_W_PDG * 100
    rho_pred_change = 2 * (delta_m_W_total - delta_M_Z_total) * 100  # ≈ 4 R
    print(f"  {c_E_label:>10}  {delta_M_Z_total*100:>+11.4f}%  {delta_m_W_total*100:>+11.4f}%  "
          f"{rho_pred_change:>+9.4f}%  {M_Z_resid:>+11.4f}%  {m_W_resid:>+11.4f}%")
print()


# ---------------------------------------------------------------------------
# Required c_S given c_E hypothesis is correct
# ---------------------------------------------------------------------------
print("=" * 78)
print("Forward path: if c_E hypothesis = 1/18 is right, what c_S closes the cluster?")
print("=" * 78)
print()
# Need: M_Z_pred × (1 + S - R) = M_Z_obs and m_W_pred × (1 + S + R) = m_W_obs
# With R = c_E × α/(1-α) = 1/18 × 256/6305 = 256/113490
# Solve for S given R
R_hyp = float(c_E_natural) * alpha_factor
S_needed_for_M_Z = delta_Z + R_hyp
S_needed_for_m_W = delta_W - R_hyp
print(f"  With R = c_E × α₁/(1-α₁) = {R_hyp * 100:.4f}% (c_E = 1/18):")
print(f"    S needed to close M_Z: {S_needed_for_M_Z * 100:+.4f}% → c_S = {-S_needed_for_M_Z/alpha_factor:.6f}")
print(f"    S needed to close m_W: {S_needed_for_m_W * 100:+.4f}% → c_S = {-S_needed_for_m_W/alpha_factor:.6f}")
print(f"    Average c_S needed: {(-S_needed_for_M_Z/alpha_factor + -S_needed_for_m_W/alpha_factor)/2:.6f}")
print()
print(f"  Required c_S for joint fit (with c_E=1/18): ≈ 0.16 (= 8/48 = 4/24 ≈ 1/6)")
print(f"  Phase A convergent c_S = 1/12 = 0.083  → only ~50% of required")
print(f"  → Family C derivation needs ADDITIONAL FACTOR of ~2 beyond the 1/5-reduction-from-v_Higgs form")
print()
print("=" * 78)
print("End of Phase A probe.")
print("=" * 78)
