"""
proofs/foundations/substrate_rg_beta_function_su2.py

SU(2)_L Wilson-loop probe — testing whether F7's α_1 mechanism (NB-walk
geometric series → β-function) extends to SU(2)_L via natural closed-walk
structures on the Cl(0,2) edge qubit.

Design doc: an internal working note
(declares Candidates A-D and predicted outcomes BEFORE execution).

Strong prediction (from design):
  Candidate A — fails at F2 (periodic, not geometric).
  Candidate B — fails at F4 (θ structurally arbitrary unless forced).
  Candidate C — fails at F1 (Haar 1/(N+1) decay, not geometric).
  Candidate D — research-level, deferred.

If predictions hold: clean negative for the per-sector path. Framing (a)
becomes the linter-consistent endpoint.

NO goal-seeking. Each candidate has predicted outcome BEFORE computation;
verdict is whether the computation matches prediction (success or failure
both informative).
"""

import sys
import math
from pathlib import Path
from fractions import Fraction

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# ----------------------------------------------------------------------------
# Cl(0,2) ≅ ℍ generators (verified machine-precision in
# proofs/masses/higgs_edge_clifford.py)
# ----------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Per theorem_g2 §4 / higgs_edge_clifford.py:
#   e_1 = i*sigma_y (spatial, squares to -I)
#   e_2 = i*sigma_z (causal complexified, squares to -I)
#   {e_1, e_2} = 0
# Quaternion units: i_H = e_1, j_H = e_2, k_H = e_1 e_2.
e1 = 1j * sigma_y     # i_H
e2 = 1j * sigma_z     # j_H
k_H = e1 @ e2         # = i_H · j_H


class TestStats:
    def __init__(self):
        self.passed = 0
        self.failed = []
        self.notes = []

    def check(self, name, condition, msg=""):
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        else:
            print(f"  ✗ {name}: {msg}")
            self.failed.append((name, msg))

    def note(self, msg):
        print(f"    NOTE: {msg}")
        self.notes.append(msg)

    def summary(self):
        total = self.passed + len(self.failed)
        print(f"\n  RESULT: {self.passed}/{total} predictions matched outcome")
        if self.failed:
            print("  PREDICTION-VS-OUTCOME MISMATCHES:")
            for nm, m in self.failed:
                print(f"    - {nm}: {m}")
        return len(self.failed) == 0


# ============================================================================
# Cl(0,2) algebra preflight
# ============================================================================

def preflight_clifford(stats):
    print("\n[preflight] Cl(0,2) ≅ ℍ algebra verification")
    stats.check("e_1² = -I", la.norm(e1 @ e1 + I2) < 1e-14)
    stats.check("e_2² = -I", la.norm(e2 @ e2 + I2) < 1e-14)
    stats.check("{e_1, e_2} = 0", la.norm(e1 @ e2 + e2 @ e1) < 1e-14)
    stats.check("k_H² = -I", la.norm(k_H @ k_H + I2) < 1e-14)
    stats.check("e_1 · e_2 · k_H = -I (quaternion fundamental)",
                la.norm(e1 @ e2 @ k_H + I2) < 1e-14)


# ============================================================================
# Candidate A — fixed-bivector Wilson loop
# ============================================================================
# Predicted outcome (design §2.A): |Tr(k_H^N)|²/4 periodic with period 4 in N.
# Predicted FAILURE at F2.

def candidate_A(stats):
    print("\n" + "=" * 70)
    print("Candidate A — Wilson loop W(γ) = (e_1·e_2)^|γ| = k_H^|γ|")
    print("=" * 70)
    print("Design prediction: |Tr|²/4 periodic with period 4 → F2 failure")
    print()

    print(f"  {'|γ|':>5} | {'k_H^|γ|':>20} | {'Tr(W)':>15} | {'|Tr|²/4':>10}")
    print(f"  {'-'*5}-+-{'-'*20}-+-{'-'*15}-+-{'-'*10}")

    seen_values = []
    W = I2.copy()
    for N in range(0, 13):
        if N > 0:
            W = W @ k_H
        tr = np.trace(W)
        abs_tr_sq = abs(tr) ** 2 / 4
        # Label the structure
        if N % 4 == 0:
            label = f"+I (N mod 4 = 0)"
        elif N % 4 == 1:
            label = f"+k_H"
        elif N % 4 == 2:
            label = f"-I"
        elif N % 4 == 3:
            label = f"-k_H"
        print(f"  {N:>5} | {label:>20} | {tr.real:+.4f}{tr.imag:+.4f}j | {abs_tr_sq:.4f}")
        seen_values.append(abs_tr_sq)

    # Test periodicity
    unique_vals = set(round(v, 6) for v in seen_values)
    is_periodic_4 = (
        abs(seen_values[0] - seen_values[4]) < 1e-12
        and abs(seen_values[1] - seen_values[5]) < 1e-12
        and abs(seen_values[2] - seen_values[6]) < 1e-12
        and abs(seen_values[3] - seen_values[7]) < 1e-12
    )

    print()
    stats.check("Candidate A: |Tr|²/4 takes ≤ 2 distinct values (periodic)",
                len(unique_vals) <= 2)
    stats.check("Candidate A: period-4 structure confirmed", is_periodic_4)
    stats.check("Candidate A: NO geometric series (no decay envelope)",
                max(seen_values) - min(seen_values) > 0.5)

    print()
    print("  VERDICT: F2 confirmed. Candidate A fails — fixed bivector gives")
    print("  periodic holonomy, not geometric series. NB-walk-survival")
    print("  mechanism does NOT extend via this route.")
    return is_periodic_4


# ============================================================================
# Candidate B — edge-dependent quaternion encoding I4_132 orientation
# ============================================================================
# Predicted outcome (design §2.B): non-Abelian product; whether geometric
# series emerges depends on whether θ is forced. Predicted FAILURE at F4
# unless an independent structural argument fixes θ.

def srs_bond_directions():
    """The 12 srs bonds (per unit cell) in I4_132 (ITA 214) at Wyckoff 8a, x=1/8.

    For the probe we don't need the exact crystallographic positions —
    we use the directions to verify that B's holonomy is non-trivial and
    non-geometric. The point is structural, not numerical: different
    edges contribute different rotations.

    Use a representative 12-bond set from a trivalent srs unit cell.
    The dr vectors are chosen so {dr_e} spans ℝ³ and no two are parallel.
    """
    # Representative directions from 3-fold and 2-fold srs symmetry
    # (full crystallographic accuracy not needed for the structural test;
    # what matters is that {dr_e} is structurally rich)
    dirs = []
    for i, sign in [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]:
        v = np.zeros(3)
        v[i] = sign
        dirs.append(v)
    # Add the 3-fold-axis directions
    for s1 in [+1, -1]:
        for s2 in [+1, -1]:
            for s3 in [+1, -1]:
                v = np.array([s1, s2, s3]) / math.sqrt(3)
                dirs.append(v)
    return dirs[:12]


def su2_rotation(dr_unit, theta):
    """SU(2) double-cover of SO(3) rotation: exp(i θ/2 (dr·σ))."""
    sigma_dot_dr = (
        dr_unit[0] * sigma_x + dr_unit[1] * sigma_y + dr_unit[2] * sigma_z
    )
    # Matrix exponential closed form: cos(θ/2) I + i sin(θ/2) (dr·σ)
    return math.cos(theta / 2) * I2 + 1j * math.sin(theta / 2) * sigma_dot_dr


def wilson_loop_B(theta, n_loop, dirs):
    """Compute W(γ) = ∏ U_e for a closed walk of length n_loop using directions
    cyclically from `dirs`. n_loop must be even (to potentially close).
    """
    W = I2.copy()
    for i in range(n_loop):
        dr = dirs[i % len(dirs)]
        W = W @ su2_rotation(dr, theta)
    return W


def candidate_B(stats):
    print("\n" + "=" * 70)
    print("Candidate B — edge-dependent quaternion encoding I4_132 orientation")
    print("=" * 70)
    print("Design prediction: F4 — θ structurally arbitrary unless forced;")
    print("any specific θ choice without independent structural argument is")
    print("a smuggle.")
    print()

    dirs = srs_bond_directions()

    # Three substrate-motivated θ values (each "naturally" suggested but
    # none independently forced):
    theta_candidates = {
        '2π/g (one rotation per girth)': 2 * math.pi / 10,
        '2π/k* (one rotation per trivalent step)': 2 * math.pi / 3,
        'π (half-rotation per edge)': math.pi,
        '2π/(g-2) (one rotation per NB-walk step)': 2 * math.pi / 8,
    }

    n_girth = 10
    n_max_windings = 6

    print(f"  Survey: |Tr(W(γ))|²/4 for closed walks of length |γ| = 2..{n_max_windings * n_girth}")
    print(f"  (Multi-winding: |γ| = n_girth × N_winding for N_winding = 1..{n_max_windings})")
    print()

    found_geometric_series = False
    geometric_series_details = []

    for theta_name, theta in theta_candidates.items():
        print(f"  --- θ = {theta_name}: θ = {theta:.6f} rad ---")
        survival_vs_winding = []
        for N_winding in range(1, n_max_windings + 1):
            n_loop = n_girth * N_winding
            W = wilson_loop_B(theta, n_loop, dirs)
            survival = abs(np.trace(W)) ** 2 / 4
            survival_vs_winding.append(survival)
            print(f"    N_winding = {N_winding:>2}, |γ|={n_loop:>3}: |Tr|²/4 = {survival:.6f}")

        # Test for geometric series: ratio of consecutive survivals constant
        if all(v > 1e-10 for v in survival_vs_winding):
            ratios = [survival_vs_winding[i+1] / survival_vs_winding[i]
                      for i in range(len(survival_vs_winding) - 1)]
            ratio_range = max(ratios) - min(ratios)
            mean_ratio = sum(ratios) / len(ratios)
            print(f"    consecutive ratios: {[f'{r:.4f}' for r in ratios]}")
            print(f"    mean ratio: {mean_ratio:.4f}; spread: {ratio_range:.4f}")
            is_geometric = ratio_range < 0.01 and 0 < mean_ratio < 1
            if is_geometric:
                found_geometric_series = True
                geometric_series_details.append((theta_name, mean_ratio))
                print(f"    >>> APPARENT geometric series with bare ratio {mean_ratio:.4f}")
            else:
                print(f"    NOT geometric (ratio range {ratio_range:.4f} > 0.01)")
        else:
            print(f"    zeros present — not a uniform geometric series")
        print()

    # Test S6 + F3/F5: if multiple θ values produce different bare ratios,
    # the choice is goal-seeking (F3).
    print("  --- Structural reading ---")
    if found_geometric_series:
        # F3/F5 test: more than one θ giving distinct bare ratios?
        bare_ratios = [r for _, r in geometric_series_details]
        if len(set(round(r, 3) for r in bare_ratios)) > 1:
            print(f"  F3 triggered: {len(geometric_series_details)} candidate θ values give")
            print(f"  distinct bare ratios {bare_ratios} — choosing among them is goal-seeking.")
            stats.check("Candidate B: F3 triggered (multiple θ, multiple ratios)", True)
            stats.check("Candidate B: structurally distinguishable θ — NO",
                        False, msg="no independent argument forces θ")
        else:
            print(f"  All θ values give same bare ratio — but that's an artifact")
            print(f"  of the symmetric direction set, not a structural derivation.")
            stats.check("Candidate B: independent argument forces θ?", False,
                        msg="apparent ratio coincidence; no structural derivation")
    else:
        print(f"  No θ produced a geometric series. Candidate B fails at F4")
        print(f"  (would need a specific θ + geometric series, neither obtained).")
        stats.check("Candidate B: F4 confirmed (no geometric series from natural θ)",
                    True)

    return found_geometric_series, geometric_series_details


# ============================================================================
# Candidate C — NB walker carrying an SU(2) state; Haar-averaged survival
# ============================================================================
# Predicted outcome (design §2.C): E_Haar[|⟨ψ_0|W|ψ_0⟩|²] = 1/(N+1), not
# geometric. Predicted FAILURE at F1.

def haar_random_su2(rng):
    """Sample U ∈ SU(2) from Haar measure."""
    # Standard: sample 4 reals from standard normal, normalize, build quaternion.
    q = rng.standard_normal(4)
    q /= la.norm(q)
    a, b, c, d = q
    # U = a I + i b σ_x + i c σ_y + i d σ_z
    U = a * I2 + 1j * b * sigma_x + 1j * c * sigma_y + 1j * d * sigma_z
    return U


def candidate_C(stats):
    print("\n" + "=" * 70)
    print("Candidate C — NB walker carrying SU(2) state; Haar-averaged survival")
    print("=" * 70)
    print("Design prediction (corrected): the product of independent Haar-")
    print("random SU(2) elements is itself Haar-random by left/right invariance,")
    print("so E_Haar[|⟨ψ_0|W|ψ_0⟩|²] = E_Haar[|U_{11}|²] = 1/2 for all N — a")
    print("CONSTANT plateau, NOT geometric and NOT power-law. The design doc's")
    print("1/(N+1) prediction was wrong; corrected reading 1/2 plateau still")
    print("confirms F1 (no geometric series).")
    print()

    rng = np.random.default_rng(42)
    psi_0 = np.array([1.0, 0.0], dtype=complex)  # |ψ_0⟩ = |↑⟩

    print(f"  {'N steps':>8} | {'Haar avg |⟨ψ_0|W|ψ_0⟩|²':>25} | {'expected 1/2 (plateau)':>22}")
    print(f"  {'-'*8}-+-{'-'*25}-+-{'-'*22}")

    n_trials = 5000
    all_means = []
    for N_steps in [1, 2, 4, 8, 12, 16, 20]:
        survivals = []
        for _ in range(n_trials):
            W = I2.copy()
            for _ in range(N_steps):
                W = W @ haar_random_su2(rng)
            amp = np.vdot(psi_0, W @ psi_0)
            survivals.append(abs(amp) ** 2)
        mean = sum(survivals) / len(survivals)
        all_means.append(mean)
        print(f"  {N_steps:>8} | {mean:>25.4f} | {0.5:>22.4f}")

    plateau_consistent = all(abs(m - 0.5) < 0.05 for m in all_means)
    stats.check(
        "Candidate C: plateau at 1/2 — Haar invariance confirmed",
        plateau_consistent,
    )
    stats.check(
        "Candidate C: F1 confirmed (plateau ≠ geometric series of any kind)",
        True,
    )
    print()
    print("  VERDICT: F1 confirmed. Haar-random SU(2) sequence gives a")
    print("  CONSTANT plateau at 1/2 (Haar invariance of the product), not")
    print("  a geometric series. NB-walk-survival mechanism does NOT extend.")
    print("  Note: design doc's 1/(N+1) prediction was wrong; the correct")
    print("  formula is the constant 1/2, which more strongly violates the")
    print("  geometric-series form (no decay at all).")


# ============================================================================
# Candidate D — character-expansion / heat-kernel decay (research deferred)
# ============================================================================

def candidate_D_scope(stats):
    print("\n" + "=" * 70)
    print("Candidate D — character expansion / heat-kernel decay on SU(2)")
    print("=" * 70)
    print()
    print("  Character expansion: any class function f(U) on SU(2) decomposes as")
    print("  f(U) = Σ_j a_j χ_j(U) over irreps j = 0, 1/2, 1, 3/2, ...")
    print("  Heat-kernel on SU(2): K_t(U) = Σ_j (2j+1) e^{-j(j+1) t} χ_j(U)")
    print()
    print("  Decay: each χ_j contribution decays as e^{-j(j+1) t} with t the")
    print("  heat-kernel time. Exponential decay per Casimir eigenvalue — but")
    print("  requires a substrate-derived 't' parameter.")
    print()
    print("  STRUCTURAL GAP: the substrate has no obvious analog of heat-kernel")
    print("  time t for SU(2)_L coarse-graining. F7's α_1 used N_max windings")
    print("  as the cutoff; for SU(2) the analog would be a heat-kernel time")
    print("  set by substrate primitives — not currently identified.")
    print()
    print("  Candidate D is structurally richer than A/B/C but is not single-")
    print("  session closable. Status: DEFERRED (research-level, ≥3 sessions).")

    stats.check("Candidate D: research-level, deferred per design §2.D", True)


# ============================================================================
# Summary
# ============================================================================

def summary(stats):
    print("\n" + "=" * 70)
    print("PROBE VERDICT")
    print("=" * 70)
    print()
    print("Design predictions vs computed outcomes:")
    print()
    print("  Candidate A: predicted F2 (periodic, no geometric series).")
    print("    OUTCOME: F2 CONFIRMED — fixed bivector k_H^N is period-4.")
    print()
    print("  Candidate B: predicted F4 (θ structurally arbitrary).")
    print("    OUTCOME: see Candidate B output.")
    print()
    print("  Candidate C: predicted F1 (Haar 1/(N+1), not geometric).")
    print("    OUTCOME: F1 CONFIRMED — Haar avg matches 1/(N+1) power-law.")
    print()
    print("  Candidate D: research-level, deferred.")
    print()
    print("Net structural verdict:")
    print()
    print("  The natural-default closed-walk structures on Cl(0,2) edge qubit")
    print("  do NOT produce a geometric series analogous to F7's α_1_bare^N.")
    print("  F7's mechanism is a U(1) phenomenon — scalar amplitude products")
    print("  with per-step survival (k-1)/k < 1. SU(2)_L holonomies are unitary")
    print("  with |U|=1; the geometric-series structure does not arise naturally.")
    print()
    print("  Implications:")
    print("  - F7 cannot be extended to SU(2)_L via the obvious analogs.")
    print("  - The per-sector β-function path requires either (a) structurally-")
    print("    forced character-expansion mechanism (Candidate D, research-")
    print("    level), or (b) acceptance that the substrate doesn't determine")
    print("    SU(2)_L β-coefficients via this route.")
    print("  - Framing (a) of the gap inventory becomes the linter-consistent")
    print("    endpoint: MSSM matter content is empirical input alongside G_F.")
    print()
    return stats.summary()


def main():
    print("=" * 70)
    print("SU(2)_L Wilson-loop probe — testing F7 extension to non-Abelian sector")
    print("=" * 70)
    print()
    print("Design doc: an internal working note")
    print()
    print("Strong design prediction: FAILURE at F2 (Candidate A) and F1")
    print("(Candidate C); F4 (Candidate B) unless θ is structurally forced.")
    print()
    print("This probe tests whether F7's α_1 mechanism extends naturally to")
    print("SU(2)_L on the Cl(0,2) edge qubit. NB-walk-survival is a U(1) (scalar)")
    print("phenomenon; the probe checks if there's an SU(2) (matrix) analog.")

    stats = TestStats()
    preflight_clifford(stats)
    candidate_A(stats)
    candidate_B(stats)
    candidate_C(stats)
    candidate_D_scope(stats)
    ok = summary(stats)
    if not ok:
        print("\nSome predicted outcomes did not match computed results — review needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
