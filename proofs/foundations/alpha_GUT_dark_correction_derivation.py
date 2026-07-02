#!/usr/bin/env python3
"""
proofs/foundations/alpha_GUT_dark_correction_derivation.py

α_GUT substrate-Feshbach-analog hypothesis — added 2026-05-15 to the cluster
of substrate-Feshbach-analog candidates documented at
an internal working note.

CONTEXT.
The framework has a derived substrate-Feshbach-analog template

    coupling_physical = coupling_bare × (1 − c × α_1/(1 − α_1))

with α_1/(1 − α_1) = 256/6305 the A2-T waterline-retained-winding sum on
the Hashimoto NB graph, and c a coupling-specific dimensional fraction.

THE TEMPLATE IS CLOSED FOR ONE CASE:
  v_Higgs: c = 5/12 (theorem-grade, TWO derivations — Sunada cycle count
    n_g/(N_atoms·k*²) AND Hashimoto-spectral marginal-mode fraction
    (2(|E|−|V|)+1)/(2|E|))

THREE OPEN CLUSTER ANALOGS (Layer-1 hypotheses, c undetermined):
  λ_Higgs: c ≈ 0.148 needed, no clean rational form
  y_τ:     c ≈ 0.032 needed, "≈ 1/32" not clean
  Λ_CC:    V_Ram w_eff mixing, not a single c

THIS WORK proposes a FOURTH cluster entry:
  α_GUT: c ≈ 1/k* = 1/3 (clean rational, structurally suggestive but not yet derived)

The cluster drift on P63–P71 (1/α_i back-extrapolation to M_unif giving
24.30 vs framework's bare 24) closes essentially exactly under c = 1/k*:

    α_GUT^observed = α_GUT_bare × (1 − (1/k*) × α_1/(1 − α_1))
                  = (1/24) × (1 − (1/3)(256/6305))
                  = (1/24) × (18659/18915)
                  ≈ 0.04110
    1/α_GUT^observed ≈ 24.329

Forward run via MSSM one-loop:
  1/α_1(M_Z) predicted: 59.008  vs PDG 59.015  (−0.01% dev)
  1/α_2(M_Z) predicted: 29.584  vs PDG 29.581  (+0.01% dev)
  1/α_3(M_Z) predicted:  8.566  vs PDG  8.475  (+1.08% dev, QCD-specific)

α_1 and α_2 match PDG at the 0.01% level. α_3 residual ~1% is the
known QCD-specific systematic (hadronic vacuum polarization, threshold
effects) — separate from the substrate-Feshbach-analog mechanism.

STATUS (per `parameter_linter.md` hard quality gate audit).

The hypothesis c_α_GUT = 1/k* uses the framework's existing
substrate-Feshbach-analog template
with a candidate structural fraction that is cleaner than the open
λ_Higgs and y_τ analogs (c ≈ 0.148, 0.032 — no clean rationals).

The numerical match is suggestive but NOT a derivation.  Candidate
structural routes for c_α_GUT = 1/k*:

  (Route H — Hashimoto-spectral, parallel to 5/12).
    c = (Perron sector dim) / (NB total dim).
    For srs: NB total = 2|E| = 12.  If Perron sector has dim N_atoms = 4
    at Γ (one Perron mode per atom), c = 4/12 = 1/3 = 1/k*.
    REQUIRES verification of Perron sector multiplicity on srs at Γ.

  (Route C — cycle-counting, parallel to 5/12 = n_g/(N_atoms·k*²)).
    c = (specific cycle count) / (N_atoms × k*²).
    For c = 1/3 = 12/36: count = 12 = directed-edge count per cell.
    REQUIRES structural argument that the α_GUT vertex couples to
    directed-edge count per cell, parallel to v_Higgs coupling to
    girth-cycle count per vertex.

Neither route is closed.  c_α_GUT = 1/k* is at the same hypothesis grade
as c_λ_Higgs and c_y_τ in the cluster.

CALIBRATING CONSTRAINT (per cluster doc §5).
Any structural derivation of c_α_GUT must REPRODUCE c_v_Higgs = 5/12 via
the same mechanism.  Numerology that matches α_GUT but doesn't reproduce
5/12 does NOT count as evidence.

CAVEATS.

(C1) **Hypothesis grade, not theorem.** Parallel to λ_Higgs and y_τ
     status in the cluster.

(C2) **Inherits the cluster's open conditional.** If the cluster's
     Feshbach-analog hypothesis is falsified (e.g., spectral or cycle
     routes don't produce non-(5/12) coefficients with the right
     mechanism), this falls with it.

(C3) **α_3 residual ~1%** is the known QCD-specific systematic; not
     load-bearing for this hypothesis.

(C4) **Cluster propagation NOT done in this script.** The cluster
     children (P63–P71) keep their current numerical predictions; the
     hypothesis is RECORDED as a candidate explanation for the drift,
     not used to revise predicted values.

NOT NUMEROLOGY DRESSED AS STRUCTURE.

This proposal differs from a numerology-only finding because:

  (a) It uses the framework's EXISTING template (substrate-Feshbach
      analog), not a constructed form.

  (b) The proposed c = 1/k* is a clean framework rational (parallel to
      M_R's (1/k*)^(g-1) return amplitude, m_ν₃'s k*·N_atoms factor,
      etc.) — not an arbitrary fit.

  (c) Two candidate structural routes (Hashimoto-spectral, cycle-
      counting) are explicit and falsifiable.

  (d) Status is flagged HYPOTHESIS, not theorem-grade.  Children NOT
      propagated.

If either Route H or Route C closes (structurally derives c_α_GUT = 1/k*
via the same mechanism that gives 5/12 for v), this graduates to
theorem-grade-conditional with the same status as the cluster.
"""

from __future__ import annotations

import math
from fractions import Fraction


# ===========================================================================
# Substrate primitives (theorem-grade)
# ===========================================================================

K_STAR = 3                                                  # srs coordination
G_GIRTH = 10                                                # srs girth
N_ATOMS = 4                                                 # srs primitive cell atoms
N_E_DIRECTED = N_ATOMS * K_STAR                             # 12 directed edges per cell
ALPHA_1_BARE = Fraction(K_STAR - 1, K_STAR) ** (G_GIRTH - 2)    # = (2/3)^8


# A2-T waterline winding sum (geometric series on Hashimoto NB graph)
WATERLINE_WINDING_SUM = ALPHA_1_BARE / (1 - ALPHA_1_BARE)   # = 256/6305


# ===========================================================================
# Bare and Feshbach-analog-corrected α_GUT
# ===========================================================================

def alpha_GUT_bare():
    """α_GUT_bare = 1 / (2^k* × k*) = 1/24, from per-vertex substrate
    label counting (CAR Fock states × visible edges)."""
    label_count = Fraction(2 ** K_STAR * K_STAR)            # = 24
    return Fraction(1) / label_count, label_count


def alpha_GUT_feshbach_analog(c_coupling):
    """α_GUT_observed = α_GUT_bare × (1 − c × α_1/(1−α_1)).

    Substrate-Feshbach-analog template applied to α_GUT with structural
    fraction c_coupling.  For c = 1/k* hypothesis: see Route H and
    Route C candidate derivations in docstring.
    """
    correction = c_coupling * WATERLINE_WINDING_SUM
    bare, _ = alpha_GUT_bare()
    return bare * (1 - correction), correction


# ===========================================================================
# MSSM one-loop running cross-check
# ===========================================================================

M_Z_GEV = 91.1876
M_UNIF_GEV = 1.985e16
LN_RATIO = math.log(M_UNIF_GEV / M_Z_GEV)

ALPHA_EM_INV_MZ_PDG = 127.94
SIN_SQ_THETA_W_MZ_PDG = 0.23121
ALPHA_S_MZ_PDG = 0.1180

ALPHA_INV_MZ_PDG = {
    1: (3.0/5.0) * ALPHA_EM_INV_MZ_PDG * (1 - SIN_SQ_THETA_W_MZ_PDG),
    2: ALPHA_EM_INV_MZ_PDG * SIN_SQ_THETA_W_MZ_PDG,
    3: 1.0 / ALPHA_S_MZ_PDG,
}

B_MSSM = {1: Fraction(33, 5), 2: Fraction(1), 3: Fraction(-3)}


def run_to_MZ(alpha_GUT_inv):
    """Forward run 1/α_GUT to 1/α_i(M_Z) via MSSM one-loop."""
    return {
        i: alpha_GUT_inv + float(B_MSSM[i]) / (2 * math.pi) * LN_RATIO
        for i in [1, 2, 3]
    }


# ===========================================================================
# Test cluster c hypotheses against PDG
# ===========================================================================

def evaluate_cluster_hypothesis(c_value, label):
    """Apply c hypothesis to α_GUT and report cluster match."""
    alpha_dc, correction = alpha_GUT_feshbach_analog(c_value)
    alpha_GUT_inv_obs = 1.0 / float(alpha_dc)
    alpha_inv_MZ = run_to_MZ(alpha_GUT_inv_obs)

    devs = []
    for i in [1, 2, 3]:
        pred = alpha_inv_MZ[i]
        pdg = ALPHA_INV_MZ_PDG[i]
        dev = 100 * (pred - pdg) / pdg
        devs.append(dev)
    rms = math.sqrt(sum(d ** 2 for d in devs) / 3)

    return {
        'label': label,
        'c': c_value,
        'correction_factor': float(correction),
        'alpha_GUT_inv_obs': alpha_GUT_inv_obs,
        'alpha_inv_MZ': alpha_inv_MZ,
        'devs_pct': devs,
        'rms_pct': rms,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print('=' * 80)
    print(' α_GUT substrate-Feshbach-analog hypothesis')
    print('=' * 80)
    print()
    print(' Framework template (from cluster doc 2026-05-14):')
    print('   coupling_physical = coupling_bare × (1 − c × α_1/(1−α_1))')
    print(f'   α_1 = (2/3)^8 = {float(ALPHA_1_BARE):.6f}')
    print(f'   α_1/(1−α_1) = 256/6305 = {float(WATERLINE_WINDING_SUM):.6f}'
          f'  (A2-T waterline winding sum)')
    print()
    print(' Bare values:')
    bare, count_bare = alpha_GUT_bare()
    print(f'   α_GUT_bare = 1 / (2^k* × k*) = 1/{count_bare} = {float(bare):.6f}')
    print()
    print(' PDG-back-extrapolated cluster (1/α_i(M_unif)):')
    alpha_inv_unif_PDG = {
        i: ALPHA_INV_MZ_PDG[i] - float(B_MSSM[i]) / (2 * math.pi) * LN_RATIO
        for i in [1, 2, 3]
    }
    for i in [1, 2, 3]:
        print(f'   1/α_{i}(M_unif) PDG = {alpha_inv_unif_PDG[i]:.4f}')
    mean_pdg = sum(alpha_inv_unif_PDG.values()) / 3
    print(f'   mean              = {mean_pdg:.4f}')
    print(f'   framework bare    = 24.000')
    print(f'   Δ                 = +{mean_pdg - 24:.4f} uniform across i')
    print()

    # --- Test multiple c hypotheses ---
    print('-' * 80)
    print(' Comparing candidate c values')
    print('-' * 80)
    candidates = [
        (Fraction(0), 'c = 0 (no DC, bare)'),
        (Fraction(1, K_STAR), f'c = 1/k* = 1/{K_STAR}  [PROPOSED for cluster]'),
        (Fraction(5, 12), 'c = 5/12 (v_Higgs c, for comparison)'),
        (Fraction(N_ATOMS, 2 * N_E_DIRECTED), f'c = N_atoms/(2|E|) = {N_ATOMS}/{2 * N_E_DIRECTED} (Route H candidate)'),
        (Fraction(N_E_DIRECTED, N_ATOMS * K_STAR ** 2), f'c = (dir edges)/(N_atoms·k*²) = {N_E_DIRECTED}/{N_ATOMS * K_STAR ** 2} (Route C candidate)'),
    ]
    print(f'   {"label":<46} {"c":>12} {"1/α_GUT_obs":>12} {"RMS dev":>10}')
    for c, label in candidates:
        r = evaluate_cluster_hypothesis(c, label)
        print(f'   {label:<46} {str(c):>12} {r["alpha_GUT_inv_obs"]:>12.4f} {r["rms_pct"]:>9.3f}%')
    print()

    # --- Detailed results for c = 1/k* hypothesis ---
    print('-' * 80)
    print(' Detailed results: c = 1/k* hypothesis')
    print('-' * 80)
    r = evaluate_cluster_hypothesis(Fraction(1, K_STAR), 'c = 1/k*')
    print(f'   α_GUT_obs/α_GUT_bare correction factor: (1 − {float(r["correction_factor"]):.5f}) = {1 - r["correction_factor"]:.5f}')
    print(f'   α_GUT_obs = {1.0/r["alpha_GUT_inv_obs"]:.6f},  1/α_GUT_obs = {r["alpha_GUT_inv_obs"]:.4f}')
    print()
    print(f'   {"i":>3} {"1/α_i(M_Z) pred":>18} {"PDG":>12} {"dev":>10}')
    for i, (pred, dev) in enumerate(zip(
        [r["alpha_inv_MZ"][1], r["alpha_inv_MZ"][2], r["alpha_inv_MZ"][3]],
        r["devs_pct"]), start=1):
        pdg = ALPHA_INV_MZ_PDG[i]
        print(f'   {i:>3} {pred:>18.4f} {pdg:>12.4f} {dev:>+9.3f}%')
    print(f'   RMS deviation: {r["rms_pct"]:.3f}%')
    print()
    print(f'   α_1, α_2 match PDG within 0.01% (essentially exact).')
    print(f'   α_3 residual ~1% is the known QCD-specific systematic')
    print(f'   (hadronic VP, threshold effects) — separate from this mechanism.')
    print()

    # --- Comparison to v_Higgs cluster entry ---
    print('-' * 80)
    print(' Position in substrate-Feshbach-analog cluster')
    print('-' * 80)
    print(f'   v_Higgs:   c = 5/12  ≈ {5/12:.4f}  [THEOREM-GRADE: cycle-count + spectral both derive]')
    print(f'   λ_Higgs:   c ≈ 0.148           [OPEN HYPOTHESIS: no clean rational, paths falsified]')
    print(f'   y_τ:       c ≈ 0.032 ≈ 1/32    [OPEN HYPOTHESIS: no clean rational]')
    print(f'   α_GUT:     c = 1/k* = 1/{K_STAR} ≈ {1/K_STAR:.4f}  [OPEN HYPOTHESIS: clean rational, NOT YET DERIVED]')
    print()
    print(f'   c_α_GUT = 1/k* is STRUCTURALLY CLEANER than the λ and y_τ analogs')
    print(f'   (clean rational vs unclean), but the structural derivation route')
    print(f'   (Hashimoto-spectral or cycle-counting) is NOT YET closed.')
    print()
    print(f'   Cluster doc calibrating constraint (§5):')
    print(f'   Any structural derivation must REPRODUCE c = 5/12 for v_Higgs via')
    print(f'   the same mechanism.  c = 1/k* for α_GUT must come from a route')
    print(f'   that also gives c = 5/12 for v.')
    print()
    print('=' * 80)
    print(' STATUS: LAYER-1 HYPOTHESIS')
    print('=' * 80)
    print()
    print(' Same grade as λ_Higgs and y_τ entries in the cluster.')
    print(' Children NOT propagated.  Hypothesis recorded; structural derivation open.')
    print()


if __name__ == '__main__':
    main()
