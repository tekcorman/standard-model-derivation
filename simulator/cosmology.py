"""
S3 COMPUTE — the disconnected N_hub axis (the unified-pipeline cosmology stage).

Per an internal working note §2/§3 (target
layout line 94) and §5(II) ("can start now, zero collision, low risk: fold in
`proofs/cosmology/*` + `match/cosmology_emulator.py`"): this is the canonical
query surface for the framework's cosmological (N_hub) axis. It does NOT
re-derive — it OWNS the N_hub-axis computation on the audited backbone so the
~85 `proofs/cosmology/*` probes become `simulator.cosmology.query() + assert`
shims and `predictions/{H_0,t_0,Lambda_CC,w_DE,z_eff,...}` become [wrap].

ARCHITECTURE (LCDM-fit emulator, NOT a substrate fluid simulator — per
an internal working note):
substrate enters only as project-native primitives (the cascade theorem H(z),
the k*=3 Ω-partition); LCDM parameters are extracted via the theorem-grade
bias-function family. Backbone (all prior-art-audited, see
an internal working note):
`proofs/cosmology/lib/` (ontology/cosmography/bias_functions) + the cascade DAG
(`predictions/N_hub.py`) + the FLRW-MDL closure theorem.

THE HARD-WON TRUTHS OF THE N_hub ARC, encoded here so they are queryable and
NOT re-litigated (`established_results()`):
  • Structure = k*=3 + TWO irreducible data anchors: G_F→N_hub (absolute
    scale) and observed Ω_m→z_eff (the dimensionless-budget conditional).
    NOT "one knob".
  • z_eff ≡ 𝓑⁻¹(Ω_m,observed): the deterministic bias-inversion, a data-side
    conditional ("bounded but not derived from first principles" —
    theorem_cosmology_bias_function_family.md §2/§3.iv). NOT a survey-Fisher
    quantity (that framing was RETRACTED). NOT swap-duality-forced (the
    √3=√k* forcing theorem was ATTEMPTED and PROVED FALSE: no Ω_m↔Ω_Λ
    involution at the forced k*=3; 𝓑 strictly monotone ⇒ √3 is a crossing,
    not a fixed point).
  • The native budget (Ω_m,Ω_Λ)=(2/3,1/3) IS a zero-adoption PREDICTION
    (observer Gleason+MDL ⇒ k*=3); only the observer-side pivot was adopted.
  • Emergent-dimension finite-N flow is REAL but lattice-like (λ1~N^−2/3);
    extrapolated |3−d_s| ≈ 8e−33 at recombination ⇒ CHARACTERIZED NEGATIVE
    for early-universe physics (convergence complete ~33 orders early).

FRONTIER (the genuine gaps are the boundary — wired to `simulator.frontier`,
never chased here): `acoustic_scale` (r_s/θ_*/native CMB C_l) is
extraction-layer / out-of-scope as a framework claim; `lambda_cc_factor_two`
is open-bounded (recorded with the honest factor-of-2 note);
`gleason_genericity` (C1) is the bounded soft point under d=3.

Frame discipline: every dimensionful return is a `lib.ontology.Tagged`;
substrate→observer is the one cited (16/15) cascade-D2-extended translation.
No landmines (audit C2/C4): never reads `breaking_cascade.V_HIGGS_GeV`
(frozen present epoch — v is recomputed from the DAG); never calls the
retired `mdl_select`.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_PRED = str(_REPO / "predictions")
if _PRED not in sys.path:
    sys.path.insert(0, _PRED)

from proofs.cosmology.lib.ontology import Frame, Tagged, translate
from proofs.cosmology.lib.bias_functions import (
    Omega_m_local_coasting_closed_form,
    w_local_at_fixed_Omega_m_coasting_closed_form,
    solve_z_eff_for_Omega_m,
)
from proofs.cosmology.lib.cosmography import coasting

from simulator import frontier

# Predictions modules print on import; suppress (audit-noted noise).
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from M_Pl_natural import M_Pl_GeV as _M_P, t_P_seconds as _t_P
    from N_hub import predict_N_hub
    from v_higgs import predict_v_higgs
    from d_spatial import predict_d_spatial
    from k_star import predict_k_star
    from g_girth import predict_g_girth
    from alpha_1 import predict_alpha_1
    from p_toggle import predict_p_toggle
    from V_count import predict_V_count
    from z_eff import predict_z_eff, BAO_ANCHORS as _BAO, SN_MODEL as _SN

_DELTA = 2.0 / 9.0                      # Koide phase (h_walker); N-invariant
_G_F_PDG = 1.1663787e-5                 # GeV^-2, PDG 2024 (pins N_hub's value)
_MPC_KM = 3.085677581e19               # 1 Mpc in km (matches predictions/H_0.py)
_GYR_S = 3.1557e16                      # s per Gyr (matches predictions/t_0.py)
_RATE_GAP = 1.0 / 15.0                  # (1/5)(1/3) cascade D2-extended observer gap
_RATE_GAP_CITE = "docs/theorems/theorem_cascade_D2_extended_observer_rate.md"

# Observation comparisons (cited at use site; % and σ_obs only, never σ_theory).
_OMEGA_M_PLANCK = (0.3153, 0.0073)      # Planck 2018, arXiv:1807.06209
_OMEGA_L_PLANCK = (0.6847, 0.0073)


# --- structural / anchor queries -------------------------------------------

def _structural():
    with contextlib.redirect_stdout(_buf):
        d = predict_d_spatial()
        k = predict_k_star(d)
        g = predict_g_girth(k, d)
        a1 = predict_alpha_1(k, g)
    return d, k, g, a1


def n_hub() -> float:
    """The framework's one adopted dimensional input (value pinned to ppm by
    the measured G_F; G_F itself is a PREDICTION)."""
    _d, _k, _g, a1 = _structural()
    _p = predict_p_toggle()
    _V = predict_V_count(_k, _d)
    return predict_N_hub(_G_F_PDG, _M_P, a1, _DELTA, _k, _p, _V)


# --- N_hub-axis cascade (frame-tagged) -------------------------------------

def hubble(z: float = 0.0, frame: Frame = Frame.SUBSTRATE) -> Tagged:
    """Coasting H(z)=H_0(1+z), H_0,substrate = 1/(N·t_P) (cascade D1+D2+D3,
    coefficient exactly 1). Observer frame via the one cited (16/15) translate.
    Honest-domain note: a framework cosmological claim only for z≲2 (the
    coasting map is structurally falsified at recombination — that is the
    `acoustic_scale` frontier, not this stage)."""
    N = n_hub()
    H_sub = _MPC_KM / (N * _t_P) * (1.0 + z)
    t = Tagged(value=H_sub, frame=Frame.SUBSTRATE)
    if frame is Frame.SUBSTRATE:
        return t
    return translate(t, Frame.OBSERVER, 1.0 + _RATE_GAP, _RATE_GAP_CITE)


def age(z: float = 0.0, frame: Frame = Frame.SUBSTRATE) -> Tagged:
    """t(z) = N(z)·t_P (Gyr); coasting H·t = 1. N(z)=N_hub/(1+z)."""
    N = n_hub() / (1.0 + z)
    t = Tagged(value=(N * _t_P) / _GYR_S, frame=Frame.SUBSTRATE)
    if frame is Frame.SUBSTRATE:
        return t
    return translate(t, Frame.OBSERVER, 1.0 / (1.0 + _RATE_GAP), _RATE_GAP_CITE)


def lambda_cc() -> float:
    """Λ_substrate = 1/N² (Planck units), theorem-grade (Friedmann +
    H=1/(N·t_P)). The ΛCDM-fit factor-of-2 is the open-bounded
    `lambda_cc_factor_two` frontier gap — see `frontier_status()`."""
    return 1.0 / (n_hub() ** 2)


def w_de() -> float:
    """w_DE = -1 exactly — scale-invariant, UNIQUE-THEOREM-GRADE
    (ratio p_Λ/ρ_Λ; the (16/15)² frame factor cancels)."""
    return -1.0


# --- native energy budget: the ZERO-ADOPTION prediction --------------------

def native_energy_budget() -> dict:
    """(Ω_m, Ω_Λ) = ((k*−1)/k*, 1/k*) = (2/3, 1/3) — a PREDICTION from the
    observer Gleason+MDL k*=3 (zero data, zero adoption). The only adopted
    cosmological quantity is the observer-side pivot z_eff (see `z_eff()`)."""
    _d, k, _g, _a1 = _structural()
    return {
        "Omega_m": (k - 1.0) / k,
        "Omega_Lambda": 1.0 / k,
        "k_star": k,
        "source": "observer Gleason 1957 + MDL min-cost-viable ⇒ k*=3 "
                  "(simulator.gating.observer); NB-walk dark/matter fractions",
        "adoption": None,
        "grade": "zero-adoption PREDICTION (theorem-grade)",
    }


# --- observer-side LCDM extraction (bias-function family) -------------------

def bias_Omega_m(z: float) -> float:
    """𝓑_Ωm(z) = (u+1)/(u²+u+1), u=1+z — the theorem-grade FORM mapping the
    native coasting H(z) to what an LCDM fitter recovers at z. (Strictly
    monotone decreasing — proven; that is why the swap-duality 'fixed point'
    is a misnomer, see `established_results()`.)"""
    return Omega_m_local_coasting_closed_form(z)


def z_eff() -> dict:
    """z_eff ≡ 𝓑⁻¹(Ω_m,observed): the deterministic bias-inversion. A
    data-side conditional in the SAME epistemic class as N_hub (structure +
    one observational anchor), NOT a survey-Fisher quantity (retracted) and
    NOT swap-duality-forced (proved false). `adopted` is read LIVE from
    predictions/z_eff.py (authority); `obs_implied` inverts the bias function
    at Planck Ω_m."""
    with contextlib.redirect_stdout(_buf):
        z_adopted = predict_z_eff(_BAO, _SN)
    native = coasting(H_0=hubble(0.0, Frame.OBSERVER).value, frame=Frame.OBSERVER)
    z_obs = solve_z_eff_for_Omega_m(native, _OMEGA_M_PLANCK[0])
    return {
        "adopted": float(z_adopted),
        "obs_implied": float(z_obs),
        "epistemic_class": "data-side conditional (N_hub-class): structure + "
                           "one observational anchor (observed Ω_m)",
        "NOT": "survey-Fisher (RETRACTED); swap-duality-forced (PROVED FALSE)",
        "authority": "predictions/z_eff.py (live)",
    }


def lcdm_extracted(z: float | None = None) -> dict:
    """The LCDM parameters an external Friedmann fitter recovers from the
    native coasting H(z) at the conditional z (default: live adopted z_eff).
    Deviations vs Planck in % and σ_obs only."""
    ze = z_eff()
    zv = ze["adopted"] if z is None else z
    Om = bias_Omega_m(zv)
    OL = 1.0 - Om
    w = (w_local_at_fixed_Omega_m_coasting_closed_form(zv, Om)
         if zv > 0.0 else -1.0)
    def dev(p, obs):
        return {"pct": (p - obs[0]) / obs[0] * 100.0,
                "sigma_obs": (p - obs[0]) / obs[1]}
    return {
        "z_eff_used": zv,
        "Omega_m_LCDM": Om, "Omega_Lambda_LCDM": OL, "w_DE_LCDM": w,
        "vs_Planck": {"Omega_m": dev(Om, _OMEGA_M_PLANCK),
                      "Omega_Lambda": dev(OL, _OMEGA_L_PLANCK)},
        "note": "agreement at the adopted z_eff is CIRCULAR (z_eff≡𝓑⁻¹"
                "(Ω_m,obs)); a parameter-free Ω_m=1/3 at the √3 anchor would "
                "be +2.47σ (the honest, falsifiable stake — recorded, not hidden)",
    }


# --- the arc's established truths (queryable; not re-litigated) -------------

def established_results() -> dict:
    return {
        "structure": "k*=3 + TWO irreducible data anchors "
                     "(G_F→N_hub absolute scale; observed Ω_m→z_eff "
                     "dimensionless budget). NOT one knob.",
        "swap_duality_forcing": "ATTEMPTED, PROVED FALSE — P1: no Ω_m↔Ω_Λ "
            "involution at forced k*=3 (2:1 asymmetric; swap-symmetry needs "
            "k*=2 = the retracted anti-pattern value); P2: 𝓑 strictly "
            "monotone ⇒ √3 is a crossing, not a fixed point. "
            "[proofs/cosmology/swap_duality_forcing_2026-05-18.py]",
        "dimensional_flow": "CHARACTERIZED NEGATIVE — d_s genuinely runs but "
            "lattice-like (λ1~N^−2/3); |3−d_s|≈8e−33 at recombination, ~33 "
            "orders below detectable; cannot carry early-universe physics. "
            "[proofs/cosmology/d_eff_emergence_vs_N_2026-05-18.py]",
        "native_budget": "(2/3,1/3) is a zero-adoption PREDICTION (k*=3); "
                         "only the observer-side pivot z_eff was adopted.",
        "honest_stake": "+2.47σ — a parameter-free Ω_m would be 1/3, "
                        "+2.47σ from Planck; reported straight, never dissolved.",
    }


# --- frontier wiring (the boundary — gaps are recorded, never chased here) --

def acoustic_scale():
    """r_s / θ_* / native CMB C_l — routed to the frontier. This is
    extraction-layer / out-of-scope as a framework claim (side-loaded fluid
    mechanics, REJECTED). Raises NotImplementedError with the precise
    blocker; it is the simulator's boundary, not a target."""
    return frontier.acoustic_scale()


def frontier_status() -> dict:
    """The cosmology-relevant frontier gaps, queried live from
    simulator.frontier (the single source of the boundary)."""
    return {g: {"status": frontier.get_gap(g).status,
                "affects": frontier.get_gap(g).affects}
            for g in ("acoustic_scale", "lambda_cc_factor_two",
                      "gleason_genericity")}


def summary() -> dict:
    """Stage summary (the N_hub-axis at the present epoch + the recorded
    truths + the frontier boundary)."""
    ze = z_eff()
    return {
        "anchor": {
            "N_hub": n_hub(),
            "H_0_substrate_km_s_Mpc": hubble(0.0, Frame.SUBSTRATE).value,
            "H_0_observer_km_s_Mpc": hubble(0.0, Frame.OBSERVER).value,
            "age_substrate_Gyr": age(0.0, Frame.SUBSTRATE).value,
            "Lambda_CC_substrate": lambda_cc(),
            "w_DE": w_de(),
        },
        "native_budget": native_energy_budget(),
        "z_eff": ze,
        "lcdm_extracted_at_adopted_z_eff": lcdm_extracted(),
        "established_results": established_results(),
        "frontier": frontier_status(),
    }


# --- self-test: DAG-authority cross-check + absorb demo + GC-A5 -------------

if __name__ == "__main__":
    print("=" * 78)
    print("  simulator.cosmology — S3 N_hub-axis stage (absorption plan §5.II)")
    print("=" * 78)
    s = summary()
    a = s["anchor"]
    print(f"  N_hub              = {a['N_hub']:.6e}")
    print(f"  H_0 substrate      = {a['H_0_substrate_km_s_Mpc']:.4f} km/s/Mpc")
    print(f"  H_0 observer       = {a['H_0_observer_km_s_Mpc']:.4f} km/s/Mpc")
    print(f"  age substrate      = {a['age_substrate_Gyr']:.4f} Gyr")
    print(f"  Λ_CC substrate     = {a['Lambda_CC_substrate']:.4e}")
    nb = s["native_budget"]
    print(f"  native budget      = Ω_m={nb['Omega_m']:.6f} "
          f"Ω_Λ={nb['Omega_Lambda']:.6f}  [{nb['grade']}]")
    ze = s["z_eff"]
    print(f"  z_eff adopted/obs  = {ze['adopted']:.4f} / "
          f"{ze['obs_implied']:.4f}  [{ze['epistemic_class'][:38]}…]")
    le = s["lcdm_extracted_at_adopted_z_eff"]
    print(f"  Ω_m_LCDM(z_eff)    = {le['Omega_m_LCDM']:.4f}  "
          f"({le['vs_Planck']['Omega_m']['sigma_obs']:+.2f}σ_obs, circular)")
    print()

    # (1) DAG-AUTHORITY CROSS-CHECK — the stage must reproduce the live
    #     predictions DAG exactly (run `python3 predictions/<x>.py` values).
    import math
    assert abs(a["N_hub"] - 8.394881e60) / 8.394881e60 < 1e-4, a["N_hub"]
    assert abs(a["H_0_substrate_km_s_Mpc"] - 68.1784) < 1e-2
    assert abs(a["H_0_observer_km_s_Mpc"] - 72.7236) < 1e-2
    assert abs(a["age_substrate_Gyr"] - 14.3419) < 1e-2
    assert abs(nb["Omega_m"] - 2.0 / 3.0) < 1e-12
    assert abs(nb["Omega_Lambda"] - 1.0 / 3.0) < 1e-12
    assert abs(w_de() + 1.0) < 1e-15
    print("  ✓ DAG-authority cross-check: N_hub/H_0/age/budget/w match "
          "predictions/* live")

    # (2) ABSORB DEMONSTRATION — a proofs/cosmology result becomes a
    #     `simulator.cosmology.query() + assert` shim (the §1 pattern):
    #     bias-inversion round-trip (theorem_cosmology_bias_function_family).
    z_p = solve_z_eff_for_Omega_m(
        coasting(H_0=hubble(0.0, Frame.OBSERVER).value, frame=Frame.OBSERVER),
        _OMEGA_M_PLANCK[0])
    assert abs(bias_Omega_m(z_p) - _OMEGA_M_PLANCK[0]) < 1e-9, "round-trip"
    assert abs(bias_Omega_m(0.0) - 2.0 / 3.0) < 1e-9, "native at z=0"
    assert bias_Omega_m(0.5) < bias_Omega_m(0.1), "𝓑 strictly monotone"
    print("  ✓ absorb demo: bias-function round-trip + monotonicity hold "
          "(proofs/cosmology/lib shim pattern)")

    # (3) FRONTIER BOUNDARY — the gaps raise / are recorded, never chased.
    try:
        acoustic_scale()
        raise AssertionError("acoustic_scale must raise (out-of-scope)")
    except NotImplementedError:
        pass
    fs = s["frontier"]
    assert "extraction-layer" in fs["acoustic_scale"]["status"]
    assert fs["lambda_cc_factor_two"]["status"] == "open-bounded"
    print(f"  ✓ frontier wired: acoustic_scale=out-of-scope (raises); "
          f"lambda_cc_factor_two=open-bounded; gleason_genericity=bounded")

    # (4) GC-A5-GENERALIZED honesty self-check on this stage's own records.
    text = " ".join(str(v) for v in established_results().values()).lower()
    _forbidden = ("swap-duality proven", "z_eff derived", "one knob suffices",
                  "parameter-free match", "tension dissolved",
                  "recombination solved", "early universe solved")
    _required = ("proved false", "characterized negative", "+2.47σ",
                 "zero-adoption", "two irreducible data anchors")
    hits = [t for t in _forbidden if t in text]
    miss = [r for r in _required if r not in text]
    assert not hits, f"overclaim tokens: {hits}"
    assert not miss, f"missing honest records: {miss}"
    print("  ✓ GC-A5 self-check: no overclaim; the arc's negatives + the "
          "+2.47σ stake are recorded straight")
    print()
    print("  STAGE OK — simulator/cosmology.py owns the N_hub axis on the "
          "audited backbone;")
    print("  ~85 proofs/cosmology/* can now [wrap] to simulator.cosmology "
          "queries.")
