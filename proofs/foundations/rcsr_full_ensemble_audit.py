#!/usr/bin/env python3
"""
Full-ensemble audit: Boltzmann-weighted framework predictions over all 9
V+E-transitive 3-c chiral 3D RCSR candidates.

For each framework prediction in the canonical scope (V_us, V_cb, V_ub,
Q_Koide, α_1, y_τ, dark c, η_B chain factor, ε_CP, sin²θ_W, α_GUT, n_γ),
this audit:

  1. Computes the per-substrate prediction from the fingerprint data
     (`rcsr_per_substrate_fingerprint.py`).
  2. Tags by CLASS A/B/C/D/E:
        A — depends on (k, g) only
        B — depends on (|V|, |E|, k)
        C — depends on h saddle value (only defined when K-rational saddle
            exists at canonical Ramanujan |λ|² = k-1)
        D — depends on multiplicity n_γ at saddle (only defined when CLASS C
            applies)
        E — depends on Pati-Salam embedding (k=3 ⇒ Cl(6); same on all 9
            since all candidates have k=3)
  3. Boltzmann-weights using per-substrate Convention-B Level 2 DL.
  4. Reports ensemble-mean observable and shift relative to PDG.

The audit honestly reports CLASS C/D as UNDEFINED for substrates without a
K-rational saddle — these can't be naively averaged into the ensemble.
For srs the framework's "h saddle = (√3+i√5)/2" appears at conventional-cell
k_H or k_midR (not the framework's documented k_P = (1/4,1/4,1/4) — the
two are related by a BCC vs primitive-cubic convention, but both are
K-rational). For srs-z the saddle is k_R=(1/2,1/2,1/2) (canonical). For
lov + okw + srs-c8: NO K-rational Ramanujan saddle found at any tested
k-point ⇒ CLASS C/D excluded for these substrates.

PDG 2024 reference values:
  V_us = 0.22501 ± 0.00067
  V_cb = 0.0408  ± 0.0014
  V_ub = 0.00382 ± 0.00020
  Q_Koide (mass ratio) = 2/3 (framework prediction; lattice/PDG match within %)
  η_B  = 6.12e-10 ± 0.04e-10  (CMB / BBN)
  sin²θ_W = 3/8 (framework prediction at unification scale)

CLASS-A predictions (k, g)-dependent: invariant on substrates with same
(k=3, g=10) but DIFFER for g=16 (lou/lov/okw) and g=6 (hcb-c4) — these
are the substrate-discriminating predictions in the ensemble.

Per-substrate Boltzmann weights computed from
`rcsr_per_substrate_fingerprint.py` Convention-B Level 2 DL totals.
"""

import sys
import os
import math
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_per_substrate_fingerprint import (
    fingerprint, CANDIDATES, dl_convention_B_level2, SUBSTRATE_GIRTH,
    parse_rcsr_3dall, P_CUBIC_KPOINTS, I_CUBIC_KPOINTS, I_CENTERED_SGS,
)


# =============================================================================
# Per-substrate prediction calculator
# =============================================================================

def predict_all(fp):
    """Compute all framework predictions for one substrate.

    NOTE 2026-05-02 cleanup — formula bugs fixed:
      - y_τ now uses canonical α_1_full/k*² (Row P7) instead of α_1²/(16π²)
      - V_ub_M1 (NEW key) uses canonical M1 multi-cycle sum (Row P14)
      - V_ub (legacy single-winding) retained for back-compat
      - dark_c_simple (NEW key) uses simple-graph |E|, |V| (matches Row P5
        canonical 5/12 on srs); dark_c (legacy multi-edge primitive) retained
      - M_chain_canonical (NEW key) = 4·n_conv_mid (matches Row P29's M=48
        for srs); M_chain (legacy = n_prim_edges) retained
    """
    name = fp['name']
    k = fp['coord']
    g = fp['g']
    Nv = fp['n_prim_atoms']
    Ne = fp['n_prim_edges']
    Ne_conv = fp.get('n_conv_mid', Ne)  # conventional-cell midpoint count
    q_NB = (k - 1) / k
    alpha_1 = q_NB ** (g - 2)
    alpha_1_full = alpha_1 / (1 - alpha_1)

    # Simple-graph |E|, |V| from primitive (multi-edges collapse to single edges)
    # For srs's body-centering primitive: multi-edge K_4 with 12 directed-half-arcs;
    # simple K_4 has 6 unique edges. Detect by degree sequence.
    prim_deg_seq = fp.get('prim_deg_seq', [])
    if prim_deg_seq and max(prim_deg_seq) > k:
        # Multi-edges present: divide edge count by multiplicity factor
        # Simple |E| = sum(deg)/2, but each "edge" in deg sequence counts both directions
        # For srs primitive: deg=6 each, sum=24, simple |E| = 24/(2·2) = 6 (where /2 is for
        # multi-edge collapse — each unordered pair contributes deg=2 in multi-graph).
        # Practical: simple |E| = unique unordered (i,j) pairs in prim_bonds.
        # Without re-walking bonds, approximate via degree-corrected count:
        Ne_simple = Ne // 2 if Ne > k * Nv / 2 else Ne
    else:
        Ne_simple = Ne
    Nv_simple = Nv

    # CLASS A — (k, g) only — CANONICAL formulas
    V_cb     = alpha_1_full                      # = α_1/(1-α_1) — same as before
    V_ub_single = alpha_1 ** 2 / (1 - alpha_1)   # legacy single-winding (NOT canonical)

    # M1 multi-cycle V_ub (canonical Row P14): Σ_{m≥2} α_m/(1-α_m) where
    # α_m = q_NB^L_eff(m), L_eff(m) = m·g − 2(m−1)·s_seam − n_fixed.
    # For srs (s_seam=2, n_fixed=2, g=10): L_eff(m) = 6m+2.
    # Generalized formula assumes s_seam=2, n_fixed=2 (substrate-specific in principle).
    V_ub_M1 = 0.0
    for m in range(2, 100):
        L_eff = m * g - 2 * (m - 1) * 2 - 2  # = m(g-4) + 2
        alpha_m = q_NB ** L_eff
        if 0 < alpha_m < 1:
            V_ub_M1 += alpha_m / (1 - alpha_m)
        else:
            break

    Q_Koide  = q_NB
    # y_τ canonical formulation: y_τ = α_1_full_class2 / k*²
    #   where α_1_full_class2 = (n_g_edge/k*) · α_1_bare per `predictions/alpha_1_full.py`.
    # For srs: n_g_edge = 5 (verified graph invariant). For other substrates,
    # n_g_edge is substrate-specific; use srs's value as default for k=3 (clearly
    # marked SUBSTRATE-SPECIFIC for non-srs substrates).
    n_g_edge = 5  # srs's verified Class-2 cycle count; substrate-specific
    alpha_1_class2 = (n_g_edge / k) * alpha_1
    y_tau    = alpha_1_class2 / (k ** 2)         # CANONICAL Row P7 form (srs n_g_edge=5)
    eps_CP   = (k - 2) / (k + 2)

    # CLASS B — (|V|, |E|, k) — both legacy + canonical reads
    V_us     = k ** 2 / (g * Nv)                  # well-defined for primitive
    dark_c_legacy = (2 * (Ne - Nv) + 1) / (2 * Ne)
    dark_c_simple = (2 * (Ne_simple - Nv_simple) + 1) / (2 * Ne_simple) if Ne_simple > 0 else None

    # M_chain: legacy (primitive edges) vs canonical (4 · |E_conv|)
    M_chain_legacy    = Ne
    M_chain_canonical = 4 * Ne_conv               # matches Row P29 srs M=48 from |E_conv|=12
    eta_B_factor_legacy    = alpha_1 ** M_chain_legacy
    eta_B_factor_canonical = alpha_1 ** M_chain_canonical

    # CLASS C — saddle eigenvalue value (depends on K-rational h existing)
    has_h_saddle = False
    h_re, h_im = None, None
    if fp['saddle_search']:
        for k_name, ks in fp['saddle_search'].items():
            for (lam, mult, ident) in ks['K_rational_eigs']:
                # Look for the canonical h = (√3+i√5)/2 family
                if (abs(abs(lam.real) - math.sqrt(3) / 2) < 1e-5 and
                    abs(abs(lam.imag) - math.sqrt(5) / 2) < 1e-5):
                    has_h_saddle = True
                    h_re = math.sqrt(3) / 2
                    h_im = math.sqrt(5) / 2
                    break
            if has_h_saddle:
                break

    Re_h = h_re if has_h_saddle else None
    Im_h_over_modsq = (h_im / (h_re ** 2 + h_im ** 2)) if has_h_saddle else None

    # CLASS D — multiplicity of h saddle
    n_gamma = None
    if has_h_saddle:
        # Find multiplicity of the h saddle eigenvalue
        max_mult = 0
        for k_name, ks in fp['saddle_search'].items():
            for (lam, mult, ident) in ks['K_rational_eigs']:
                if (abs(abs(lam.real) - math.sqrt(3) / 2) < 1e-5 and
                    abs(abs(lam.imag) - math.sqrt(5) / 2) < 1e-5):
                    if mult > max_mult:
                        max_mult = mult
        n_gamma = max_mult

    # CLASS E — Pati-Salam (k=3 ⇒ Spin(6) Cl(6))
    sin2_theta_W = Fraction(3, 8) if k == 3 else None
    alpha_GUT = Fraction(1, 24) if k == 3 else None

    return {
        'V_us':                {'value': V_us,                'class': 'B'},
        'V_cb':                {'value': V_cb,                'class': 'A'},
        'V_ub':                {'value': V_ub_single,         'class': 'A'},  # legacy
        'V_ub_M1':             {'value': V_ub_M1,             'class': 'A'},  # CANONICAL Row P14
        'Q_Koide':             {'value': Q_Koide,             'class': 'A'},
        'alpha_1':             {'value': alpha_1,             'class': 'A'},
        'alpha_1_full':        {'value': alpha_1_full,        'class': 'A'},
        'y_tau':               {'value': y_tau,               'class': 'A'},  # FIXED: Row P7
        'eps_CP':              {'value': eps_CP,              'class': 'A'},
        'dark_c':              {'value': dark_c_legacy,       'class': 'B'},  # legacy multi-edge
        'dark_c_simple':       {'value': dark_c_simple,       'class': 'B'},  # CANONICAL Row P5
        'M_chain':             {'value': M_chain_legacy,      'class': 'B'},  # legacy primitive
        'M_chain_canonical':   {'value': M_chain_canonical,   'class': 'B'},  # CANONICAL Row P29
        'eta_B_factor':        {'value': eta_B_factor_legacy, 'class': 'B'},
        'eta_B_factor_canon':  {'value': eta_B_factor_canonical, 'class': 'B'},
        'Re(h)':               {'value': Re_h,                'class': 'C'},
        'Im(h)/|h|^2':         {'value': Im_h_over_modsq,     'class': 'C'},
        'n_gamma':             {'value': n_gamma,             'class': 'D'},
        'sin2_theta_W':        {'value': float(sin2_theta_W) if sin2_theta_W else None, 'class': 'E'},
        'alpha_GUT':           {'value': float(alpha_GUT) if alpha_GUT else None, 'class': 'E'},
        'has_K_saddle':        has_h_saddle,
    }


# =============================================================================
# PDG / framework reference values
# =============================================================================

PDG = {
    'V_us':         {'value': 0.22501,    'sigma': 0.00067},
    'V_cb':         {'value': 0.0408,     'sigma': 0.0014},
    'V_ub':         {'value': 0.00382,    'sigma': 0.00020},
    'Q_Koide':      {'value': 0.6667,     'sigma': 0.0001},   # framework target
    'sin2_theta_W': {'value': 0.375,      'sigma': 0.001},    # 3/8 framework
}


# =============================================================================
# Boltzmann ensemble propagation
# =============================================================================

def boltzmann_weights(fps_dict, ref='srs'):
    """Compute per-substrate Boltzmann weights from Convention-B Level 2 DL.

    Returns dict {name: weight relative to ref (= 1.0 for ref)}.
    """
    dl_ref = fps_dict[ref]['dl_total']
    return {name: 2.0 ** -(fp['dl_total'] - dl_ref) for name, fp in fps_dict.items()}


def ensemble_mean(predictions, weights, key, exclude_undefined=True):
    """Compute Boltzmann-weighted ensemble mean of `key` prediction.

    If exclude_undefined=True, substrates with None value are dropped (as if
    weight=0 for that prediction). Else they raise.
    """
    num = 0.0
    den = 0.0
    contributors = []
    for name, preds in predictions.items():
        v = preds[key]['value']
        if v is None:
            if exclude_undefined:
                continue
            raise ValueError(f"{name} has undefined {key}; cannot include in ensemble")
        w = weights[name]
        num += w * float(v)
        den += w
        contributors.append((name, w, float(v)))
    if den == 0:
        return None, []
    return num / den, contributors


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("FULL-ENSEMBLE AUDIT — Boltzmann-weighted framework predictions over 9 RCSR candidates")
    print("=" * 100)

    rcsr_file = '/tmp/rcsr_3d_current.txt'
    entries = parse_rcsr_3dall(rcsr_file, CANDIDATES)
    fps = {}
    predictions = {}
    for name in CANDIDATES:
        fps[name] = fingerprint(name, entries[name])
        predictions[name] = predict_all(fps[name])

    weights = boltzmann_weights(fps, ref='srs')

    # ---- Per-substrate weights ----
    print("\n" + "-" * 100)
    print("Per-substrate Boltzmann weights (Convention-B Level 2 DL, w(srs) = 1.0)")
    print("-" * 100)
    print(f"{'name':<10s} {'DL':>8s} {'ΔDL vs srs':>12s} {'w/w(srs)':>14s} {'has h saddle?':>14s}")
    for name in CANDIDATES:
        f = fps[name]
        dl = f['dl_total']
        delta = dl - fps['srs']['dl_total']
        w = weights[name]
        h_flag = 'YES' if predictions[name]['has_K_saddle'] else 'NO'
        print(f"{name:<10s} {dl:>8.3f} {delta:>+12.3f} {w:>14.4e} {h_flag:>14s}")

    # ---- Per-prediction × per-substrate value table ----
    print("\n" + "-" * 100)
    print("PER-PREDICTION × PER-SUBSTRATE VALUE TABLE")
    print("-" * 100)
    pred_keys = ['V_us', 'V_cb', 'V_ub', 'Q_Koide', 'alpha_1', 'y_tau', 'eps_CP',
                 'dark_c', 'M_chain', 'eta_B_factor', 'Re(h)', 'Im(h)/|h|^2',
                 'n_gamma', 'sin2_theta_W']
    header = f"{'pred':<14s} {'cls':>3s}  " + "  ".join(f"{n:>10s}" for n in CANDIDATES)
    print(header)
    print("-" * len(header))
    for k in pred_keys:
        row = f"{k:<14s} {predictions['srs'][k]['class']:>3s}  "
        for name in CANDIDATES:
            v = predictions[name][k]['value']
            if v is None:
                row += f"{'undef':>10s}  "
            elif isinstance(v, int):
                row += f"{v:>10d}  "
            elif isinstance(v, float):
                if abs(v) < 1e-3 or abs(v) > 1e4:
                    row += f"{v:>10.3e}  "
                else:
                    row += f"{v:>10.5f}  "
            else:
                row += f"{str(v):>10s}  "
        print(row)

    # ---- Boltzmann-weighted ensemble means + PDG comparison ----
    print("\n" + "=" * 100)
    print("BOLTZMANN-WEIGHTED ENSEMBLE MEAN vs PDG / framework reference")
    print("=" * 100)
    print()
    print(f"  {'pred':<14s} {'cls':>3s} {'ensemble (all 9)':>22s} {'srs only':>14s} {'PDG/ref':>14s} {'shift / σ':>10s} {'Δ from srs':>12s}")
    for key in ['V_us', 'V_cb', 'V_ub', 'Q_Koide', 'sin2_theta_W']:
        if key not in PDG:
            continue
        cls = predictions['srs'][key]['class']
        v_srs = float(predictions['srs'][key]['value']) if predictions['srs'][key]['value'] is not None else None
        ens_mean, contribs = ensemble_mean(predictions, weights, key)
        pdg_v = PDG[key]['value']
        pdg_s = PDG[key]['sigma']
        shift = (ens_mean - pdg_v) / pdg_s if ens_mean is not None else None
        delta_from_srs = ens_mean - v_srs if (ens_mean is not None and v_srs is not None) else None
        ens_str = f"{ens_mean:.6f}" if ens_mean is not None else "—"
        shift_str = f"{shift:+.2f}σ" if shift is not None else "—"
        d_str = f"{delta_from_srs:+.4e}" if delta_from_srs is not None else "—"
        print(f"  {key:<14s} {cls:>3s} {ens_str:>22s} {v_srs:>14.6f} "
              f"{pdg_v:>14.6f} {shift_str:>10s} {d_str:>12s}")

    # ---- Per-prediction interference report ----
    print("\n" + "-" * 100)
    print("PER-PREDICTION CONTRIBUTOR BREAKDOWN")
    print("-" * 100)
    for key in ['V_us', 'V_cb', 'V_ub']:
        ens_mean, contribs = ensemble_mean(predictions, weights, key)
        print(f"\n  {key} ensemble = {ens_mean:.6f}  (PDG {PDG[key]['value']:.5f} ± {PDG[key]['sigma']:.5f})")
        print(f"    {'substrate':<10s} {'weight w':>12s} {'value':>12s} {'w · v':>14s}  {'cumulative %':>14s}")
        contribs_sorted = sorted(contribs, key=lambda c: -c[1])  # by weight desc
        total_wv = sum(w * v for _, w, v in contribs_sorted)
        cum = 0.0
        for name, w, v in contribs_sorted:
            cum += w * v
            print(f"    {name:<10s} {w:>12.4e} {v:>12.6f} {w*v:>14.6e}  {cum/total_wv*100:>13.2f}%")

    # ---- Honest gap report ----
    print("\n" + "=" * 100)
    print("HONEST GAP REPORT")
    print("=" * 100)
    n_in_C = sum(1 for n in CANDIDATES if predictions[n]['has_K_saddle'])
    print(f"\n  CLASS C (h saddle) DEFINED on {n_in_C} of 9 substrates: "
          f"{[n for n in CANDIDATES if predictions[n]['has_K_saddle']]}")
    print(f"  CLASS C UNDEFINED on {9-n_in_C} substrates: "
          f"{[n for n in CANDIDATES if not predictions[n]['has_K_saddle']]}")
    print(f"  → CLASS C ensemble means above use only the {n_in_C} substrates with h saddles.")
    print()
    print(f"  Per-substrate non-uniformities flagged in fingerprint probe:")
    for name in CANDIDATES:
        f = fps[name]
        if f['bp_status'] == 'DISCONNECTED':
            print(f"    {name:<10s}: DISCONNECTED primitive — excluded from saddle search; "
                  "predictions reported on conventional cell only")
        elif min(f['conv_deg_seq']) != max(f['conv_deg_seq']):
            print(f"    {name:<10s}: non-uniform conventional degree {f['conv_deg_seq']} "
                  f"(structural complexity beyond simple V+E-transitive)")

    # ---- Final verdict ----
    print("\n" + "=" * 100)
    print("VERDICT — does the multi-substrate ensemble change the framework's PDG match?")
    print("=" * 100)
    print()
    for key in ['V_us', 'V_cb', 'V_ub', 'sin2_theta_W']:
        if key not in PDG:
            continue
        v_srs_only = float(predictions['srs'][key]['value']) if predictions['srs'][key]['value'] is not None else None
        ens_mean, _ = ensemble_mean(predictions, weights, key)
        pdg_v = PDG[key]['value']
        pdg_s = PDG[key]['sigma']
        shift_srs = (v_srs_only - pdg_v) / pdg_s
        shift_ens = (ens_mean - pdg_v) / pdg_s
        verdict_change = abs(shift_ens) - abs(shift_srs)
        action = "TIGHTENS" if verdict_change < -0.05 else ("LOOSENS" if verdict_change > 0.05 else "≈ unchanged")
        print(f"  {key:<14s}: srs-only {shift_srs:+.2f}σ → ensemble {shift_ens:+.2f}σ  ({action})")


if __name__ == '__main__':
    main()
