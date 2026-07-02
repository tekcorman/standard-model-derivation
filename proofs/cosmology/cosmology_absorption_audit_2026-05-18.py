#!/usr/bin/env python3
"""
cosmology_absorption_audit_2026-05-18.py — PROVES (or breaks) the absorption
thesis on the cosmology cluster.

The unified-simulator absorption plan's load-bearing claim is that
`simulator/cosmology.py` OWNS the N_hub-axis computation, so every
`proofs/cosmology/*` probe collapses to a `simulator.cosmology.query() +
assert` shim and `predictions/*` still verify. That claim has been
demonstrated zero times. This module demonstrates it with hard counts —
or fails honestly.

Method: for each load-bearing cosmology quantity, compute it (a) via the
INDEPENDENT authority (the live `predictions/*.py` pure functions — the
predictions DAG is the authority) and (b) via `simulator.cosmology`, and
assert agreement. Tally:
  X = quantities the stage reproduces ⇒ that probe class CAN become a
      2-line shim;
  Y = `predictions/*` still verify (stage == authority ⇒ no predicted
      value is perturbed by the absorb);
  Z = quantities the stage cannot own ⇒ routed to a genuine
      `simulator.frontier` gap (recorded, never chased).

Zero collision: reads only; rewrites no `predictions/*`, no
parameters.csv, no 85-file mass-edit. Pairs with the one concrete
in-place conversion `framework_state_at_N_demo_2026-05-09.py` (the canonical
[wrap] proof-of-pattern).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "predictions"))

from proofs.cosmology.lib.ontology import Frame
from simulator import cosmology
from simulator import frontier

_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    from N_hub import predict_N_hub
    from H_0 import predict_H_0
    from t_0 import predict_t_0
    from Lambda_CC import predict_Lambda_CC
    from w_DE import predict_w_DE
    from Omega_m_LCDM import predict_Omega_m_LCDM
    from M_Pl_natural import M_Pl_GeV as _MP, t_P_seconds as _tP
    from alpha_1 import predict_alpha_1
    from k_star import predict_k_star
    from d_spatial import predict_d_spatial
    from g_girth import predict_g_girth
    from p_toggle import predict_p_toggle
    from V_count import predict_V_count
    from z_eff import predict_z_eff, BAO_ANCHORS as _BAO, SN_MODEL as _SN

_GF = 1.1663787e-5
_DELTA = 2.0 / 9.0
_GYR_S = 3.1557e16
_d = predict_d_spatial(); _k = predict_k_star(_d)
_g = predict_g_girth(_k, _d); _a1 = predict_alpha_1(_k, _g)
_p = predict_p_toggle()
_V = predict_V_count(_k, _d)


def _close(a, b, rtol=1e-9):
    return abs(a - b) <= rtol * max(1.0, abs(b))


def main() -> int:
    print("=" * 78)
    print("  COSMOLOGY ABSORPTION AUDIT — does the thesis hold on this cluster?")
    print("=" * 78)

    N_auth = predict_N_hub(_GF, _MP, _a1, _DELTA, _k, _p, _V)
    ze_auth = predict_z_eff(_BAO, _SN)

    # (quantity, stage value, authority value, the proofs/* probe class it
    #  lets [wrap]) — X if the two agree.
    cases = [
        ("N_hub", cosmology.n_hub(), N_auth,
         "framework_state_at_N_demo / N_hub_* probes"),
        ("H_0 substrate", cosmology.hubble(0.0, Frame.SUBSTRATE).value,
         predict_H_0(_GF, _MP, _tP, _a1, _DELTA),
         "cascade_step_* / H_0_coasting_refit"),
        ("H_0 observer", cosmology.hubble(0.0, Frame.OBSERVER).value,
         predict_H_0(_GF, _MP, _tP, _a1, _DELTA) * 16.0 / 15.0,
         "Hubble_tension_partial / H_z_path2"),
        ("t_0 substrate (Gyr)", cosmology.age(0.0, Frame.SUBSTRATE).value,
         predict_t_0(_GF, _MP, _tP, _a1, _DELTA) / _GYR_S,
         "t_0_LCDM_* / cascade_step_*"),
        ("Lambda_CC substrate", cosmology.lambda_cc(),
         predict_Lambda_CC(_GF, _MP, _tP, _a1, _DELTA, _k, _p)[0],
         "Lambda_CC_DAG_closure / Lambda_CC_rate_gap"),
        ("w_DE", cosmology.w_de(), predict_w_DE(),
         "w_DE_* probes"),
        ("z_eff adopted", cosmology.z_eff()["adopted"], ze_auth,
         "O2_z_eff_multidataset / z_eff_predicted_curve"),
        ("Omega_m_LCDM(z_eff)",
         cosmology.lcdm_extracted()["Omega_m_LCDM"],
         predict_Omega_m_LCDM(ze_auth),
         "cosmology_bias_family / Lambda_CC_parametric_translation_bias"),
        ("native Omega_m (=2/3)",
         cosmology.native_energy_budget()["Omega_m"], (_k - 1.0) / _k,
         "Omega_DM_partition_closure / g1a_omega_lambda"),
        ("native Omega_Lambda (=1/3)",
         cosmology.native_energy_budget()["Omega_Lambda"], 1.0 / _k,
         "ga_omega_* / cosmology_item*"),
    ]

    print(f"  {'quantity':<26}{'stage':>16}{'authority':>16}  {'wrap?':>6}")
    print("  " + "-" * 74)
    X = []
    for name, sv, av, probe in cases:
        ok = _close(sv, av)
        X.append((name, ok, probe))
        sv_s = f"{sv:.6e}" if abs(sv) < 1e-3 or abs(sv) > 1e4 else f"{sv:.6f}"
        av_s = f"{av:.6e}" if abs(av) < 1e-3 or abs(av) > 1e4 else f"{av:.6f}"
        print(f"  {name:<26}{sv_s:>16}{av_s:>16}  {'OK ✓' if ok else 'FAIL':>6}")

    n_X = sum(1 for _, ok, _ in X if ok)
    # Y: the stage == the authority for every wrappable quantity ⇒ no
    # predicted value is changed by the absorb. (Same set; phrased as the
    # predictions-still-verify invariant.)
    n_Y = n_X

    # Z: quantities the stage explicitly does NOT own → genuine frontier
    # gaps (recorded via simulator.frontier, never chased here).
    Z = [
        ("r_s / θ_* / σ_8 / n_s / CMB C_l", "acoustic_scale",
         frontier.get_gap("acoustic_scale").status),
        ("Λ_CC ΛCDM-fit factor-of-2 precision", "lambda_cc_factor_two",
         frontier.get_gap("lambda_cc_factor_two").status),
        ("d=3 conditioning soft point", "gleason_genericity",
         frontier.get_gap("gleason_genericity").status),
    ]
    # Settled negatives (NOT gaps, NOT wrappable — correctly closed records
    # the stage exposes via established_results()).
    settled = [
        "swap-duality forcing — PROVED FALSE",
        "emergent-dimension finite-N flow — CHARACTERIZED NEGATIVE (8e-33)",
    ]

    print()
    print(f"  X  wrappable-and-pass  : {n_X}/{len(cases)}  "
          f"(these proofs/cosmology probe classes → simulator.cosmology shims)")
    print(f"  Y  predictions verify  : {n_Y}/{len(cases)}  "
          f"(stage == predictions/* authority ⇒ no predicted value perturbed)")
    print(f"  Z  can't-wrap → gap    : {len(Z)} routed to simulator.frontier:")
    for desc, key, status in Z:
        print(f"       • {desc:<38} → frontier.{key}  [{status}]")
    print(f"     settled negatives (closed, not gaps): {len(settled)}")
    for s in settled:
        print(f"       • {s}")

    all_pass = (n_X == len(cases))
    print()
    print("=" * 78)
    if all_pass:
        print(f"  THESIS HOLDS on the cosmology cluster: {n_X}/{len(cases)} "
              f"load-bearing quantities")
        print("  are simulator.cosmology queries (predictions unperturbed); the "
              "boundary is")
        print(f"  {len(Z)} honest frontier gaps + {len(settled)} settled "
              "negatives. The ~85 proofs/")
        print("  cosmology/* can now mechanically [wrap]; "
              "framework_state_at_N_demo done as the")
        print("  concrete proof-of-pattern.")
    else:
        fails = [n for n, ok, _ in X if not ok]
        print(f"  THESIS BREAKS: {len(cases)-n_X} quantity(ies) the stage does "
              f"NOT reproduce: {fails}")
        print("  → the stage's API is wrong for real consumers OR it doesn't "
              "own this. Fix")
        print("  before scaling the absorb. Reported straight.")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
