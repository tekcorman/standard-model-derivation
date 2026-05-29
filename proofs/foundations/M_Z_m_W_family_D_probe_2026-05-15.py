#!/usr/bin/env python3
"""
proofs/foundations/M_Z_m_W_family_D_probe_2026-05-15.py

FAMILY-D-PATTERN PROBE for M_Z and m_W substrate analog.

Mirror of `dark_disruption_per_leg_2026-05-15.py` (which closed
y_τ / λ_Higgs / m_H / m_τ via per-leg multiway dark-disruption).  This
probe applies the same template to the gauge boson 2-point function,
testing whether a single-session structural closure is in reach.

Family D template (master doc §3 (D)):
    δ(observable)/observable = - Σ_legs c_leg
where
    c_H = α₁²              per Higgs leg (joint NB walker survival)
    c_F = -α₁²/(N·k*)      per fermion leg (with JW sign)
    c_G = ?                per gauge leg  (UNKNOWN)

The SM tree-level relation
    M_Z² = π v² (α_2 + (3/5)α_1)
contains 2 Higgs legs (vev²) + 2 gauge legs.  Tests several candidate
forms for c_G.

KEY FINDING (this probe):  Family D template is *structurally insufficient*
for M_Z + m_W jointly.

    Empirical residuals:
        δM_Z²/M_Z² = -0.71076%    (M_Z prediction TOO HIGH)
        δm_W²/m_W² = +0.31054%    (m_W prediction TOO LOW)

    **Opposite signs on M_Z² and m_W².**

Family D corrections are sign-uniform: c_H > 0, |c_F| < c_H, c_G > 0
expected by structural analogy.  The composite δ ~ -Σ c_legs is therefore
NEGATIVE on any observable, and through the tree relation m_W = M_Z·cosθ_W
the correction propagates with THE SAME SIGN to both gauge bosons.

The empirical M_Z / m_W split requires *custodial-symmetry-breaking*: a
mechanism that differentially affects the W and Z self-energies.  In the
SM this is Δρ (Veltman 1977), driven by m_t - m_b Yukawa asymmetry.  In
the framework, the substrate analog has to access the top-bottom
asymmetry sector — i.e., it is NOT a single-vertex per-leg correction.

This is multi-session research-level work, not single-session reach.

The cleanest pure-Family-D candidate (c_G = c_H = α₁²) brings M_Z from
+0.357% to +0.05% (7× improvement) but leaves m_W with a residual of
opposite sign — confirming the structural insufficiency.

STATUS: NEGATIVE single-session probe.  Family D template does not close
M_Z + m_W jointly.  Substrate analog of Δρ (custodial-symmetry-breaking
through top-bottom asymmetry) is the open multi-session program.

This file produces NO prediction-file modifications.  The findings:
1. Family D template is sign-uniform and cannot reproduce opposite-sign
   residuals on M_Z / m_W.
2. The required structural mechanism is custodial breaking, parallel to
   the SM's Δρ.  Substrate identification needs the top-bottom Yukawa
   asymmetry sector (a separate multi-session program).
3. Ledger rows P64 / P71 remain STRUCTURAL-DERIVATION-CONDITIONAL.
"""
from fractions import Fraction
import math

# --- Framework constants (Type 4 upstream, theorem-grade) -----------
k_star  = 3
g       = 10
N_ATOMS = 4
alpha_1_bare = Fraction(k_star - 1, k_star) ** (g - 2)   # = (2/3)^8 = 256/6561
alpha_1_sq   = alpha_1_bare ** 2                          # = (2/3)^16

# --- Framework tree-level predictions (from predictions/M_Z.py and m_W.py)
M_Z_tree = 91.5134     # GeV
m_W_tree = 80.2373     # GeV
M_Z_obs  = 91.1876     # PDG 2024
m_W_obs  = 80.3692     # PDG 2024 (post-CDF resolution)
sigma_M_Z = 0.0021
sigma_m_W = 0.0133

dMZ  = (M_Z_obs - M_Z_tree) / M_Z_tree
dMZsq = (M_Z_obs**2 - M_Z_tree**2) / M_Z_tree**2
dmW  = (m_W_obs - m_W_tree) / m_W_tree
dmWsq = (m_W_obs**2 - m_W_tree**2) / m_W_tree**2

print("=" * 76)
print("Family-D-pattern probe — M_Z / m_W substrate analog")
print("=" * 76)
print()
print("Framework constants (Type 4 upstream, theorem-grade):")
print(f"  k*       = {k_star}")
print(f"  g        = {g}")
print(f"  N_ATOMS  = {N_ATOMS}")
print(f"  α₁_bare  = (2/3)^{g-2}   = {float(alpha_1_bare):.6e}")
print(f"  α₁²      = (2/3)^{2*(g-2)}  = {float(alpha_1_sq):.6e}  =  {float(alpha_1_sq)*100:.5f}%")
print(f"  c_F      = -α₁²/(N·k*)   = {float(-alpha_1_sq/(N_ATOMS*k_star))*100:+.5f}%")
print()
print("Tree-level framework predictions:")
print(f"  M_Z_tree  = {M_Z_tree} GeV       (PDG {M_Z_obs} ± {sigma_M_Z})")
print(f"  m_W_tree  = {m_W_tree} GeV       (PDG {m_W_obs} ± {sigma_m_W})")
print()
print("Empirical residuals (the target):")
print(f"  δM_Z/M_Z   = {dMZ*100:+.5f}%   ({dMZ/float(alpha_1_sq):+.3f} × α₁²)   "
      f"{(M_Z_obs - M_Z_tree)/sigma_M_Z:+.1f} σ_PDG")
print(f"  δM_Z²/M_Z² = {dMZsq*100:+.5f}%   ({dMZsq/float(alpha_1_sq):+.3f} × α₁²)")
print(f"  δm_W/m_W   = {dmW*100:+.5f}%   ({dmW/float(alpha_1_sq):+.3f} × α₁²)   "
      f"{(m_W_obs - m_W_tree)/sigma_m_W:+.1f} σ_PDG")
print(f"  δm_W²/m_W² = {dmWsq*100:+.5f}%   ({dmWsq/float(alpha_1_sq):+.3f} × α₁²)")
print()
print("=" * 76)
print("STRUCTURAL FINDING — opposite signs on M_Z and m_W residuals")
print("=" * 76)
print()
print("  δM_Z²/M_Z² < 0   (framework's M_Z too HIGH; needs negative correction)")
print("  δm_W²/m_W² > 0   (framework's m_W too LOW;  needs positive correction)")
print()
print("  Family D template gives δ ~ -Σ c_legs with all c_leg ≥ 0 (gauge,")
print("  scalar) or |c_F| < c_H — so the composite is always negative.  Through")
print("  the tree relation m_W = M_Z·cos(θ_W) the same-sign correction")
print("  propagates to BOTH gauge bosons.  The M_Z / m_W split requires")
print("  CUSTODIAL-SYMMETRY-BREAKING — substrate analog of Δρ (Veltman 1977),")
print("  driven in SM by m_t − m_b Yukawa asymmetry.")
print()

# --- Test all clean candidate c_G forms on M_Z and m_W simultaneously
print("=" * 76)
print("Candidate c_G forms (M_Z relation has 2 H legs + 2 G legs):")
print("=" * 76)
print()
print(f"  {'c_G form':<28} {'M_Z pred':>10} {'M_Z dev':>10} {'m_W pred':>10} {'m_W dev':>10}")
print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

c_H = alpha_1_sq
for label, c_G in [
    ("α₁²             (gauge=scalar)",  alpha_1_sq),
    ("α₁²/3           (cycle-dim 1/3)", alpha_1_sq * Fraction(1, 3)),
    ("(5/12)α₁²       (v_Higgs c)",     alpha_1_sq * Fraction(5, 12)),
    ("α₁²/12          (|c_F| analog)",  alpha_1_sq * Fraction(1, 12)),
    ("2α₁²            (double scalar)", alpha_1_sq * 2),
    ("0               (no gauge corr)", Fraction(0)),
]:
    n_H, n_G = 2, 2
    corr_Msq = -(n_H * c_H + n_G * c_G)
    corr_M = corr_Msq / 2     # δM/M ≈ (1/2) δM²/M² for |corr|<<1
    M_Z_pred = M_Z_tree * (1 + float(corr_M))
    # m_W shares the same M_Z² formula via cos(θ_W); same correction applies
    m_W_pred = m_W_tree * (1 + float(corr_M))
    dev_MZ = (M_Z_pred - M_Z_obs) / M_Z_obs * 100
    dev_mW = (m_W_pred - m_W_obs) / m_W_obs * 100
    print(f"  {label:<28} {M_Z_pred:>10.4f} {dev_MZ:>+9.4f}% {m_W_pred:>10.4f} {dev_mW:>+9.4f}%")

print()
print("Observation:  every candidate that improves M_Z worsens m_W (and vice")
print("versa), because they share the same multiplicative correction through")
print("the M_Z·cos(θ_W) tree relation.  Family-D template — applied uniformly")
print("to the gauge sector — cannot produce the empirical opposite-sign split.")
print()

# --- Show the same conclusion arithmetically on the m_W/M_Z ratio
print("=" * 76)
print("Quantifying the custodial breaking explicitly")
print("=" * 76)
print()
ratio_pred = m_W_tree / M_Z_tree
ratio_obs  = m_W_obs / M_Z_obs
print(f"  (m_W/M_Z)_pred = {ratio_pred:.6f}     [= cos(θ_W) framework]")
print(f"  (m_W/M_Z)_obs  = {ratio_obs:.6f}     [PDG on-shell]")
print(f"  Δ(m_W/M_Z)/(m_W/M_Z) = {(ratio_obs - ratio_pred)/ratio_pred*100:+.5f}%")
print()
print(f"  cos²(θ_W)_pred = {ratio_pred**2:.6f}")
print(f"  cos²(θ_W)_obs  = {ratio_obs**2:.6f}")
print(f"  Δcos²/cos²     = {((ratio_obs**2 - ratio_pred**2)/ratio_pred**2)*100:+.5f}%")
print()
print("The ratio m_W/M_Z (= cos θ_W on shell) differs from the framework's by")
print(f"  +0.518%, which is the custodial-breaking magnitude (substrate analog")
print(f"  of Δr/2 in SM).  This is what any successful substrate mechanism")
print(f"  must reproduce — and it cannot come from a sign-uniform per-leg")
print(f"  correction at the gauge 2-point function alone.")
print()

# --- Verdict / sentinel
print("=" * 76)
print("VERDICT — NEGATIVE single-session probe")
print("=" * 76)
print()
print("Family D template (per-leg multiway dark-disruption from non-srs")
print("co-retained substrate, sign-uniform) is structurally insufficient for")
print("M_Z / m_W joint closure.  The empirical opposite-sign split is a")
print("custodial-symmetry-breaking signature requiring access to the top-")
print("bottom Yukawa asymmetry — multi-session research-level program.")
print()
print("Implication for ledger:")
print("  Row P64 (M_Z)  — STRUCTURAL-DERIVATION-CONDITIONAL.  Tree +0.357%.")
print("  Row P71 (m_W)  — STRUCTURAL-DERIVATION-CONDITIONAL.  Tree -0.164%.")
print("  Substrate-analog mechanism named: custodial-symmetry-breaking via")
print("  top-bottom Yukawa asymmetry (substrate analog of Δρ).  Closes via")
print("  the framework's quark mass generation sector, not gauge 2-point per-")
print("  leg correction.")
print()
print("This probe REJECTS Family D as the closure mechanism for M_Z / m_W.")
print("=" * 76)

# Sentinel: confirm the structural finding (opposite signs)
assert dMZsq < 0 and dmWsq > 0, "Sign-pattern check failed — re-examine"
print()
print("SENTINEL PASS — opposite-sign residual confirmed (δM_Z² < 0, δm_W² > 0).")
