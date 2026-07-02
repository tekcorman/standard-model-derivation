"""
proofs/foundations/substrate_rg_beta_function.py

F7 — explicit substrate β-function for the framework's gauge coupling α_1.

This is the explicit calculation the F7 doc
`docs/forward_constructions/forward_construction_substrate_renormalization.md`
flags as "pending (~2-3 sessions)" subtask (a)-(b)-(c)-(d).

PRIOR THEOREM-GRADE STRUCTURE (from F7 doc):
- A2-T = I-projection (Csiszár 1975) ⇒ Wilsonian RG semigroup via tower
  property (Csiszár-Matuš 2003).
- A2-T waterline = IR-attractive fixed point (MDL monotonicity = Lyapunov
  via KL divergence).
- α_1 = 256/6305 = (2/3)^8 / (1 - (2/3)^8) is the renormalized coupling
  at the IR fixed point — geometric-series sum over girth-cycle windings.

THIS SESSION'S DELIVERABLE:
- Identify the running α_1(Λ) explicitly via the winding-cutoff scale
  identification (§1).
- Compute β_1 = dα_1/d log Λ (§2).
- Verify β_1 vanishes at the IR fixed point and is IR-attractive (§3).
- Compute the leading-order anomalous dimension γ at the fixed point
  (§4).
- HONEST comparison to the framework's c = 5/12 dark Feshbach factor
  (§5).

Honest scope: §§1-4 are theorem-grade-conditional under the structural
identifications already made in the F7 doc. §5 is a numerical comparison
— either the coefficient matches 5/12 (theorem-grade match) or it does
not (open question, structural reason to investigate).
"""

import sys
import math
from pathlib import Path
from fractions import Fraction

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine import CountingKernel
from match import alpha_1_bare, alpha_1_full, dark_feshbach_c


class TestStats:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, name, condition, msg=""):
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        else:
            print(f"  ✗ {name}: {msg}")
            self.failed.append((name, msg))

    def check_close(self, name, predicted, expected, atol=1e-12):
        ok = abs(float(predicted) - float(expected)) < atol
        if ok:
            print(f"  ✓ {name}: {predicted}")
            self.passed += 1
        else:
            print(f"  ✗ {name}: predicted {predicted}, expected {expected}")
            self.failed.append((name, f"{predicted} vs {expected}"))

    def summary(self):
        total = self.passed + len(self.failed)
        print(f"\n  RESULT: {self.passed}/{total} passed")
        if self.failed:
            print("  FAILURES:")
            for nm, m in self.failed:
                print(f"    - {nm}: {m}")
        return len(self.failed) == 0


# ============================================================================
# §1 — Running α_1(Λ) via winding-cutoff scale identification
# ============================================================================
#
# The framework's α_1_geom = 256/6305 is the geometric-series sum over all
# girth-cycle winding numbers n ≥ 1:
#
#     α_1_geom = Σ_{n=1}^∞ (α_1_bare)^n = α_1_bare / (1 - α_1_bare)
#
# where α_1_bare = (2/3)^8 = 256/6561 is the single-winding NB-walk
# survival (Feshbach Exponent Principle, n_fixed=2).
#
# WILSONIAN SCALE IDENTIFICATION: a winding-n NB walk on the girth cycle
# probes scales of order (1/N) of the IR; equivalently, a UV cutoff at
# winding-cutoff N_max retains windings n ≤ N_max and discards n > N_max.
#
# Define the dimensionless RG scale via:
#
#     Λ ≡ α_1_bare^{N_max}    ∈ (0, α_1_bare]
#
# with Λ → 0 as N_max → ∞ (IR limit) and Λ = α_1_bare at N_max = 1
# (UV / single-winding cutoff). Then the running coupling is:
#
#     α_1(N_max) = Σ_{n=1}^{N_max} (α_1_bare)^n
#                = α_1_bare · (1 - α_1_bare^{N_max}) / (1 - α_1_bare)
#                = α_1* · (1 - α_1_bare^{N_max})
#
# where α_1* = α_1_bare / (1 - α_1_bare) is the IR fixed point. Substituting
# Λ = α_1_bare^{N_max}:
#
#     α_1(Λ) = α_1* · (1 - Λ)
#
# At Λ = 0: α_1(0) = α_1* (IR fixed point). At Λ = α_1_bare: α_1(α_1_bare)
# = α_1*·(1 - α_1_bare) = α_1_bare (UV / single-winding theory).


def alpha_1_running(Lambda, kernel=None):
    """Substrate running coupling α_1(Λ).

    Args:
        Lambda: dimensionless RG scale, Λ ∈ (0, α_1_bare].
        kernel: optional CountingKernel.

    Returns:
        α_1(Λ) = α_1* · (1 - Λ), where α_1* = α_1_bare/(1 - α_1_bare).
    """
    kernel = kernel or CountingKernel()
    a_bare = float(alpha_1_bare(kernel))     # (2/3)^8
    a_star = a_bare / (1 - a_bare)            # 256/6305 = IR fixed point
    return a_star * (1 - Lambda)


# ============================================================================
# §2 — β_1 = dα_1/d log Λ (closed form)
# ============================================================================
#
# From α_1(Λ) = α_1* (1 - Λ):
#
#     dα_1/dΛ = -α_1*
#     dα_1/d log Λ = Λ · dα_1/dΛ = -α_1* · Λ
#
# Substituting Λ = 1 - α_1(Λ)/α_1*:
#
#     β_1(α_1) = -α_1* · (1 - α_1/α_1*) = α_1 - α_1*


def beta_1(alpha_1, kernel=None):
    """β_1(α_1) = α_1 - α_1* — the substrate β-function.

    Args:
        alpha_1: current value of the running coupling.
        kernel: optional CountingKernel.

    Returns:
        β_1 = dα_1/d log Λ at coupling α_1.
    """
    kernel = kernel or CountingKernel()
    a_bare = float(alpha_1_bare(kernel))
    a_star = a_bare / (1 - a_bare)
    return alpha_1 - a_star


# ============================================================================
# §3 — IR-attractive fixed point: β_1(α_1*) = 0; ∂β_1/∂α_1 |_{α_1*} = 1
# ============================================================================
#
# At α_1 = α_1*: β_1 = 0 (fixed point).
# ∂β_1/∂α_1 = 1 > 0 ⇒ for α_1 > α_1*, β_1 > 0 (drives further from FP)
#                    ⇒ for α_1 < α_1*, β_1 < 0
#
# IR-ATTRACTIVENESS: as Λ → 0 (IR), the running α_1(Λ) = α_1*(1 - Λ) → α_1*
# from below; the IR limit is approached monotonically. The β-function
# slope ∂β_1/∂α_1 = +1 means deviation δα_1 = α_1 - α_1* satisfies
# dδα_1/d log Λ = δα_1, so δα_1(Λ) ∝ Λ → 0 as Λ → 0. Linear IR-attraction.


def fixed_point_alpha_1(kernel=None):
    """α_1* = α_1_bare / (1 - α_1_bare) — IR fixed point of substrate RG."""
    kernel = kernel or CountingKernel()
    a_bare = float(alpha_1_bare(kernel))
    return a_bare / (1 - a_bare)


# ============================================================================
# §4 — Anomalous dimension γ at the fixed point
# ============================================================================
#
# Linearize β_1 around α_1*:
#
#     β_1(α_1* + δα_1) = δα_1 + O(δα_1²)
#
# So the anomalous dimension at the fixed point is:
#
#     γ ≡ ∂β_1/∂α_1 |_{α_1*} = +1
#
# This is the substrate's structural anomalous dimension for α_1 at IR.
# CANONICAL SCALING: the leading deviation flows like δα_1(Λ) ~ Λ^γ = Λ^1.
# A γ = 1 anomalous dimension corresponds to a "marginal" operator at the
# fixed point in the standard QFT sense (one-loop running gives δα ~ Λ).


def anomalous_dimension(kernel=None):
    """γ = ∂β_1/∂α_1 |_{α_1*} = +1 — substrate anomalous dimension at FP."""
    return 1.0


# ============================================================================
# §5 — Honest comparison: leading-order coefficient vs c = 5/12 dark factor
# ============================================================================
#
# The framework's dark Feshbach c = 5/12 = (2(|E|-|V|)+1)/(2|E|) is the
# fraction of Hashimoto modes in the "marginal" cycle space (5 of 12).
#
# Question: does the substrate β-function's leading-order coefficient
# match c = 5/12?
#
# β_1(α_1) = α_1 - α_1*
# α_1* = α_1_bare / (1 - α_1_bare) ≈ 0.04060 (vs c = 0.4167)
#
# The numerical values do NOT match: α_1* ≠ c (off by factor ~10). This is
# the EXPECTED structural answer:
#
# - α_1* is a fixed-point coupling in the gauge channel (NB-walk geometric
#   series at girth = 10).
# - c = 5/12 is a per-mode counting in the Hashimoto operator's marginal
#   cycle space (independent of girth — purely combinatorial |E|, |V|).
#
# These are distinct framework objects from distinct counting families —
# § "dark vs gauge" decomposition. The F7 doc's subtask (c) "Verify that
# the resulting one-loop beta function reproduces the framework's 5/12
# dark correction at leading order" is FALSIFIED at face value: the
# leading-order RG coefficient of α_1 around its IR fixed point is α_1*
# itself, not 5/12.
#
# What COULD be the connection? The dark factor c = 5/12 multiplies certain
# DARK-SECTOR predictions (η_lattice = 1/12 = c × 1/5; β_birefringence ∝ c
# in some chains). The β-function for the dark sector would presumably
# involve c. This session derives only the gauge-sector β_1; a dark-sector
# β_dark (β-function for the marginal cycle modes) is a separate calculation.
#
# HONEST CONCLUSION: this session closes §§1-4 (gauge β_1 explicit, IR fixed
# point verified, γ = 1 anomalous dimension). The 5/12 connection (subtask
# (c) of the F7 doc) is NOT closed by this calculation; the F7 doc's claim
# that "5/12 = leading-order RG correction" needs revision or a different
# matching.


def comparison_alpha_star_vs_dark_c(kernel=None):
    """Return the comparison values: α_1* vs c = 5/12."""
    a_star = fixed_point_alpha_1(kernel)
    c = float(dark_feshbach_c(kernel))
    return {'alpha_1_star': a_star, 'c_dark': c, 'ratio': a_star / c}


# ============================================================================
# Tests
# ============================================================================

def test_running(stats):
    print("\n[§1] Running α_1(Λ) via winding cutoff")
    k = CountingKernel()
    a_bare = float(alpha_1_bare(k))
    a_star = a_bare / (1 - a_bare)

    # IR limit (Λ = 0): α_1 → α_1*
    a_IR = alpha_1_running(0.0, k)
    stats.check_close("α_1(Λ=0) = α_1*", a_IR, a_star, atol=1e-12)
    stats.check_close("α_1* = 256/6305", a_star, 256 / 6305, atol=1e-12)

    # UV limit (Λ = α_1_bare): α_1 → α_1_bare
    a_UV = alpha_1_running(a_bare, k)
    stats.check_close("α_1(Λ=α_bare) = α_1_bare", a_UV, a_bare, atol=1e-12)


def test_beta_function(stats):
    print("\n[§2-3] β_1 = α_1 - α_1*; vanishes at IR fixed point")
    k = CountingKernel()
    a_star = fixed_point_alpha_1(k)

    # β_1 vanishes at fixed point
    stats.check_close("β_1(α_1*) = 0", beta_1(a_star, k), 0.0, atol=1e-15)

    # β_1 negative below FP, positive above
    a_below = a_star - 1e-3
    a_above = a_star + 1e-3
    stats.check("β_1(α_1 < α_1*) < 0 (RG flow toward FP)",
                beta_1(a_below, k) < 0)
    stats.check("β_1(α_1 > α_1*) > 0",
                beta_1(a_above, k) > 0)


def test_IR_attractive_flow(stats):
    print("\n[§3] IR-attractive: δα_1(Λ) ∝ Λ as Λ → 0")
    k = CountingKernel()
    a_star = fixed_point_alpha_1(k)
    # Sample at decreasing Λ; deviation should scale linearly with Λ
    deviations = []
    for Lambda in [0.01, 0.001, 0.0001]:
        delta = alpha_1_running(Lambda, k) - a_star
        deviations.append((Lambda, delta))
    # δα_1 / Λ should be approximately constant (= -α_1*)
    ratios = [d / L for L, d in deviations]
    stats.check_close("δα_1/Λ ≈ -α_1* (linear scaling)",
                      ratios[0], -a_star, atol=1e-12)
    # Spread should be at floating-point noise level (≪ 1e-10 in absolute terms)
    stats.check("ratios approximately equal across scales",
                max(ratios) - min(ratios) < 1e-10)


def test_anomalous_dimension(stats):
    print("\n[§4] Anomalous dimension γ = ∂β/∂α |_{α*} = 1")
    k = CountingKernel()
    a_star = fixed_point_alpha_1(k)
    # Numerical derivative
    h = 1e-8
    gamma_num = (beta_1(a_star + h, k) - beta_1(a_star - h, k)) / (2 * h)
    stats.check_close("γ = 1 (canonical / marginal scaling)",
                      gamma_num, 1.0, atol=1e-6)


def test_dark_factor_comparison(stats):
    print("\n[§5] HONEST: α_1* vs c = 5/12 dark factor")
    cmp = comparison_alpha_star_vs_dark_c()
    print(f"    α_1* = {cmp['alpha_1_star']:.6f}  ≈ 256/6305")
    print(f"    c    = {cmp['c_dark']:.6f}  = 5/12")
    print(f"    α_1* / c = {cmp['ratio']:.4f}")
    # Verify they are NOT equal (this is the honest result)
    stats.check("α_1* ≠ c (different framework objects)",
                abs(cmp['alpha_1_star'] - cmp['c_dark']) > 0.1)
    stats.check("ratio α_1*/c ≈ 0.0975 (not a clean rational)",
                abs(cmp['ratio'] - 0.0975) < 0.001)
    print()
    print("    HONEST FINDING: the substrate β_1's leading-order coefficient is")
    print("    α_1* (the IR fixed point itself), not c = 5/12. The F7 doc's")
    print("    subtask (c) — 'verify β_1 reproduces 5/12 at leading order' —")
    print("    does NOT close at face value via the gauge-sector running. The")
    print("    5/12 factor lives in a different counting family (dark Feshbach")
    print("    marginal cycle modes), and would correspond to a separate")
    print("    β_dark for the marginal cycle sector.")


def main():
    print("=" * 78)
    print("F7 — Substrate β-function for α_1 (IR fixed-point closure)")
    print("=" * 78)
    print("\nPRIOR THEOREM-GRADE STRUCTURE (F7 doc):")
    print("  A2-T = I-projection (Csiszár 1975) ⇒ Wilsonian RG semigroup")
    print("  A2-T waterline = IR-attractive fixed point (MDL monotonicity)")
    print("  α_1* = 256/6305 = renormalized coupling at IR fixed point")
    print("\nTHIS SESSION:")
    print("  §§1-4: explicit α_1(Λ), β_1, fixed-point verification, γ = 1")
    print("  §5:    honest comparison to c = 5/12 dark factor")

    stats = TestStats()
    test_running(stats)
    test_beta_function(stats)
    test_IR_attractive_flow(stats)
    test_anomalous_dimension(stats)
    test_dark_factor_comparison(stats)

    print("\n" + "=" * 78)
    success = stats.summary()
    if success:
        print("\nALL TESTS PASS — F7 §§1-4 closed.")
        print()
        print("Net session deliverable:")
        print("  α_1(Λ) = α_1*·(1 - Λ) running coupling [explicit]")
        print("  β_1(α_1) = α_1 - α_1*                  [explicit]")
        print("  β_1(α_1*) = 0; γ = ∂β/∂α |_{α*} = 1   [verified]")
        print("  Linear IR-attraction: δα_1 ∝ Λ → 0 as Λ → 0")
        print()
        print("Open (research-level, NOT closed this session):")
        print("  - 5/12 dark factor as gauge β_1 leading coefficient: FALSIFIED")
        print("    via direct calculation; dark factor lives in separate counting.")
        print("  - β_dark for marginal cycle sector (multi-session research).")
        print("  - Continuum-limit lift to standard Wilsonian RG (5+ sessions).")
        print()
        print("This session closes §§4.2(a)-(b) of the F7 doc")
        print("(`forward_construction_substrate_renormalization.md`); §4.2(c)")
        print("(5/12 reproduction) is FALSIFIED as written and needs reframing.")
    else:
        print("\nSome tests FAILED — F7 calculation needs review.")
        sys.exit(1)
    print("=" * 78)


if __name__ == "__main__":
    main()
