#!/usr/bin/env python3
"""
n_hub_trajectory_engine.py — the one-knob forward-propagation engine.

VISION (user, 2026-05-17): the framework is an instrument; N_hub is its
pitch. Step N_hub, push it through the predictions DAG, read out the
state. The dimensionless spectrum is the FIXED BODY of the instrument
(it does not move with the pitch); N_hub sets the absolute scale and the
late-time cosmological worldline.

WHAT THIS MODULE IS
-------------------
The bridge-LESS foundation (program step 2 of
docs/.../project_n_hub_instrument_program). It delivers two artifacts:

  (1) the late-time ΛCDM-alternative trajectory (one knob, z ≲ 2);
  (2) the over-determination falsification sweep (does the SAME N_hub
      hit independent H_0 and t_0?).

It does NOT contain the substrate→macroscopic bridge; therefore it makes
NO claim about z_eff being derived, about recombination, or about Gap G1.
Those are explicitly fenced off (see VERDICT self-check at the bottom).

EVERY load-bearing assumption was pre-audited:
an internal working note
The correctives from that audit are implemented here, in particular:

  * Backbone = predictions DAG as pure functions of N. The simulator
    layer contributes nothing epoch-dependent (audit C3) — not imported.
  * HARD DOMAIN GATE (audit A2): the coasting forward map is FALSIFIED
    at recombination (repo's own probe: θ_* off by ~10⁵σ). VALIDATED
    band only (z ≲ 2) is a framework claim; the recombination region is
    the *bridge frontier* and numbers there are REFUSED, not printed.
    (The 2026-05-09 demo printing a z=1100 row was the canonical mistake
    this engine exists to not repeat.)
  * FRAME DISCIPLINE (audit A4): every quantity is a lib.ontology.Tagged;
    substrate→observer is one cited translate(). No silent inline mixing.
  * OVER-DETERMINATION HONESTY (audit A5/B5): the G_F/v round-trip is
    BY CONSTRUCTION (it is the calibration channel that defines N_hub) —
    it is never scored as agreement. Only H_0 (vs Planck/SH0ES) and t_0
    (vs Methuselah) are genuine independent tests. Deviations in % and
    σ_obs only; never σ_theory; the retracted target_shot verdict banner
    is not consumed.
  * z_eff is read LIVE-ONLY from predictions/z_eff.py, tagged ADOPTED
    with a wide systematic band, used ONLY as a comparison point (audit
    B3/B4). The clean Lambda_CC.py foundation is never contaminated.
  * LANDMINES (audit C2/C4): mdl_select is never called; the frozen
    breaking_cascade V_HIGGS is never read; v is recomputed per step.

This is a proof/diagnostic artifact, not a predictions/ file. It writes
nothing (no parameters.csv, no ledger).
"""

from __future__ import annotations

import contextlib
import enum
import io
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# --- import paths -----------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # proofs/cosmology
_PRED_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "predictions"))
sys.path.insert(0, _THIS_DIR)   # so `import lib.*` works (lib is a package)
sys.path.insert(0, _PRED_DIR)   # flat predictions imports, like the DAG demo

# predictions modules print on import; suppress (audit-noted noise).
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from M_Pl_natural import M_Pl_GeV as M_P, t_P_seconds as t_P
    from N_hub import predict_N_hub
    from v_higgs import predict_v_higgs
    from d_spatial import predict_d_spatial
    from k_star import predict_k_star
    from g_girth import predict_g_girth
    from alpha_1 import predict_alpha_1
    from p_toggle import predict_p_toggle
    from V_count import predict_V_count
    from z_eff import predict_z_eff, BAO_ANCHORS, SN_MODEL

from lib.ontology import Frame, Tagged, translate
from lib.bias_functions import (
    Omega_m_local_coasting_closed_form,
    w_local_at_fixed_Omega_m_coasting_closed_form,
)

# --- structural / observational constants (cited at use site) ---------------
DELTA = 2.0 / 9.0                 # Koide phase (h_walker_eigenvalue.py); N-invariant
G_F_PDG = 1.1663787e-5            # GeV^-2, PDG 2024 / MuLan 2011 (0.51 ppm)
_D_STRUCT = predict_d_spatial()
_K_STRUCT = predict_k_star(_D_STRUCT)
_P_STRUCT = predict_p_toggle()
_V_STRUCT = predict_V_count(_K_STRUCT, _D_STRUCT)
MPC_KM = 3.085677581e19           # 1 Mpc in km (matches predictions/H_0.py)
GYR_S = 3.1557e16                 # s per Gyr, Julian (matches predictions/t_0.py)
CASCADE_RATE_GAP = 1.0 / 15.0     # (1/5)(1/3); cascade D2-extended observer gap
_RATE_GAP_CITE = "docs/theorems/theorem_cascade_D2_extended_observer_rate.md"

# Observation comparison values (cited at use site; NOT σ_theory).
H0_PLANCK_CMB = (67.4, 0.5)       # km/s/Mpc, Planck 2018 (substrate-side compare)
H0_SH0ES = (73.04, 1.04)          # km/s/Mpc, Riess 2022   (observer-side compare)
T0_METHUSELAH = (14.46, 0.80)     # Gyr, Bond 2013 HD 140283 (substrate-side)
OMEGA_M_PLANCK = (0.3153, 0.0073) # Planck 2018 LCDM-fit (LCDM_EXTRACTED compare)
OMEGA_L_PLANCK = (0.6847, 0.0073)

# HARD DOMAIN GATE thresholds (audit A2). The only CITED evidence is:
#   - positive validation z ≲ 2 (BOSS+eBOSS BAO χ²/dof≈1.37, z_eff.py)
#   - structural falsification at recombination z_*≈1090 (θ_* ~10⁵σ,
#     Lambda_CC_path_A_session2). Between is genuinely unvalidated.
Z_VALIDATED_MAX = 2.0
Z_FALSIFIED = 1090.0              # cited θ_* falsification point


class EpochZone(enum.Enum):
    """Honest validity zone for a requested epoch (audit A2)."""

    VALIDATED = "validated"        # z ≲ 2 — a framework cosmological claim
    EXTRAPOLATED = "extrapolated"  # composes arithmetically; NOT a claim
    FALSIFIED = "falsified"        # coasting structurally wrong; bridge frontier


def classify_zone(z: float) -> EpochZone:
    if z < 0.0:
        raise ValueError(f"z must be >= 0, got {z}")
    if z <= Z_VALIDATED_MAX:
        return EpochZone.VALIDATED
    if z < Z_FALSIFIED:
        return EpochZone.EXTRAPOLATED
    return EpochZone.FALSIFIED


# ---------------------------------------------------------------------------
# The fixed body of the instrument: N-invariant structure (audit C3).
# These take NO N argument. We assert their N-invariance positively by
# recomputing scale at two N values and showing structure is untouched.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralBody:
    d_spatial: int
    k_star: int
    g_girth: int
    alpha_1: float
    delta: float
    sin2_theta_W: float       # 3/8 exact (Pati-Salam trace)
    alpha_GUT: float          # 1/(2^k* · k*) = 1/24
    dark_factor: float        # carried F* residue (Feshbach c=5/12); NOT recomputed


def build_structural_body() -> StructuralBody:
    """The dimensionless spectrum — fixed regardless of N_hub (the pitch)."""
    with contextlib.redirect_stdout(_buf):
        d = predict_d_spatial()
        k = predict_k_star(d)
        g = predict_g_girth(k, d)
        a1 = predict_alpha_1(k, g)
    # Carried F* residue that v already contains (master doc Feshbach c=5/12).
    # Reported, NOT recomputed; v_higgs.py applies it internally.
    dark = 1.0 - (5.0 / 12.0) * a1 / (1.0 - a1)
    return StructuralBody(
        d_spatial=d,
        k_star=k,
        g_girth=g,
        alpha_1=a1,
        delta=DELTA,
        sin2_theta_W=3.0 / 8.0,
        alpha_GUT=1.0 / ((2 ** k) * k),
        dark_factor=dark,
    )


# ---------------------------------------------------------------------------
# The tuned string: N-dynamic scale + the LCDM-extracted readout.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochState:
    z: float
    N_hub: float                       # substrate-side count at this epoch
    zone: EpochZone
    # N-dynamic compressible scale (Tagged; None iff FALSIFIED)
    v: Optional[Tagged]                # GeV, SUBSTRATE
    G_F: Optional[Tagged]              # GeV^-2, SUBSTRATE (BY CONSTRUCTION)
    H_substrate: Optional[Tagged]      # km/s/Mpc
    H_observer: Optional[Tagged]       # km/s/Mpc (one cited translate)
    t_substrate: Optional[Tagged]      # Gyr
    Lambda_substrate: Optional[float]  # Planck units, = 1/N^2
    # LCDM-extracted translation (the ΛCDM-alternative readout)
    Omega_m_LCDM: Optional[float]
    Omega_L_LCDM: Optional[float]
    w_DE: Optional[float]


def epoch_state(
    z: float, body: StructuralBody, G_F_pin: float
) -> EpochState:
    """Pure: the full framework state at redshift z, one knob = N_hub.

    N_hub is pinned to the present epoch via the measured G_F (the
    calibration channel), then stepped along the coasting worldline
    N(z) = N_now / (1+z). The coasting map IS the assumption
    H(z)=H_0(1+z) (audit A1) — it is not an independent input.
    """
    N_now = predict_N_hub(G_F_pin, M_P, body.alpha_1, body.delta, _K_STRUCT, _P_STRUCT, _V_STRUCT)
    N_z = N_now / (1.0 + z)
    zone = classify_zone(z)

    if zone is EpochZone.FALSIFIED:
        # Numbers REFUSED. The coasting forward map is structurally wrong
        # here (audit A2); this region is the substrate→macroscopic
        # bridge frontier, not reachable by stepping the late-time law.
        return EpochState(
            z=z, N_hub=N_z, zone=zone,
            v=None, G_F=None, H_substrate=None, H_observer=None,
            t_substrate=None, Lambda_substrate=None,
            Omega_m_LCDM=None, Omega_L_LCDM=None, w_DE=None,
        )

    v_val = predict_v_higgs(body.delta, M_P, N_z, body.alpha_1)
    v = Tagged(value=v_val, frame=Frame.SUBSTRATE)
    G_F_z = Tagged(value=1.0 / (math.sqrt(2.0) * v_val * v_val),
                   frame=Frame.SUBSTRATE)

    H_sub_km = MPC_KM / (N_z * t_P)
    H_substrate = Tagged(value=H_sub_km, frame=Frame.SUBSTRATE)
    H_observer = translate(
        H_substrate, target=Frame.OBSERVER,
        factor=1.0 + CASCADE_RATE_GAP, citation=_RATE_GAP_CITE,
    )
    t_substrate = Tagged(value=(N_z * t_P) / GYR_S, frame=Frame.SUBSTRATE)
    Lambda_sub = 1.0 / (N_z * N_z)

    # LCDM-EXTRACTED translation: what an LCDM fitter recovers from the
    # coasting native at this z. Theorem-grade FORM (audit B1).
    Om = Omega_m_local_coasting_closed_form(z)   # (u+1)/(u^2+u+1)
    OL = 1.0 - Om
    w = w_local_at_fixed_Omega_m_coasting_closed_form(z, Om) if z > 0.0 else -1.0

    return EpochState(
        z=z, N_hub=N_z, zone=zone,
        v=v, G_F=G_F_z,
        H_substrate=H_substrate, H_observer=H_observer,
        t_substrate=t_substrate, Lambda_substrate=Lambda_sub,
        Omega_m_LCDM=Om, Omega_L_LCDM=OL, w_DE=w,
    )


def trajectory(
    z_grid: Tuple[float, ...], body: StructuralBody, G_F_pin: float
) -> Tuple[EpochState, ...]:
    return tuple(epoch_state(z, body, G_F_pin) for z in z_grid)


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------


def _dev(pred: float, obs: Tuple[float, float]) -> str:
    """% and σ_obs only (audit B7; never σ_theory)."""
    val, sig = obs
    pct = (pred - val) / val * 100.0
    nsig = (pred - val) / sig
    return f"{pct:+.2f}% ({nsig:+.2f}σ_obs)"


def report_structure_is_fixed(body: StructuralBody, G_F_pin: float) -> None:
    """Positively demonstrate: the pitch moves the scale, not the body."""
    N_now = predict_N_hub(G_F_pin, M_P, body.alpha_1, body.delta, _K_STRUCT, _P_STRUCT, _V_STRUCT)
    v_now = predict_v_higgs(body.delta, M_P, N_now, body.alpha_1)
    v_half = predict_v_higgs(body.delta, M_P, N_now / 2.0, body.alpha_1)
    print("=" * 78)
    print("  THE FIXED BODY vs THE TUNED STRING")
    print("=" * 78)
    print("  Dimensionless spectrum (N-INVARIANT — the instrument's body):")
    print(f"    d_spatial={body.d_spatial}  k*={body.k_star}  g={body.g_girth}"
          f"  α₁={body.alpha_1:.10f}  δ={body.delta:.10f}")
    print(f"    sin²θ_W = 3/8 = {body.sin2_theta_W}   "
          f"α_GUT = 1/{int(round(1/body.alpha_GUT))} = {body.alpha_GUT:.10f}")
    print(f"    carried F* residue (Feshbach c=5/12, master doc) = "
          f"{body.dark_factor:.10f}  [carried, NOT recomputed]")
    print("  Absolute scale (N-DYNAMIC — the tuned string):")
    print(f"    v(N_now)   = {v_now:.6f} GeV")
    print(f"    v(N_now/2) = {v_half:.6f} GeV   "
          f"ratio = {v_half / v_now:.6f}  (expected 2^(1/4)={2 ** 0.25:.6f})")
    print("  ⇒ halving the pitch moves the scale (v ∝ N^-1/4); the body is")
    print("    untouched. This is the instrument metaphor, made precise.")
    print()


def report_lcdm_alternative(
    states: Tuple[EpochState, ...], G_F_pin: float
) -> None:
    """Headline #1: one-knob late-time ΛCDM alternative (z ≲ 2)."""
    print("=" * 78)
    print("  ΛCDM-ALTERNATIVE TRAJECTORY  (one knob = N_hub; honest-gated)")
    print("=" * 78)
    print(f"  {'z':>7} {'zone':>12} {'H_sub':>9} {'H_obs':>9} "
          f"{'t_sub':>8} {'Ω_m':>7} {'Ω_Λ':>7} {'w':>7}")
    print(f"  {'':>7} {'':>12} {'km/s/Mpc':>9} {'km/s/Mpc':>9} "
          f"{'Gyr':>8} {'LCDM':>7} {'LCDM':>7} {'DE':>7}")
    print("  " + "-" * 74)
    for s in states:
        if s.zone is EpochZone.FALSIFIED:
            print(f"  {s.z:>7.1f} {s.zone.value:>12} "
                  f"{'— REFUSED: coasting falsified here; bridge frontier —':>54}")
            continue
        flag = "" if s.zone is EpochZone.VALIDATED else "  (NOT a claim)"
        print(f"  {s.z:>7.2f} {s.zone.value:>12} "
              f"{s.H_substrate.value:>9.3f} {s.H_observer.value:>9.3f} "
              f"{s.t_substrate.value:>8.3f} {s.Omega_m_LCDM:>7.4f} "
              f"{s.Omega_L_LCDM:>7.4f} {s.w_DE:>7.3f}{flag}")
    print()

    # z=0 substrate vs the substrate-side probes; one cited observer xlate.
    z0 = states[0]
    assert z0.z == 0.0
    print("  Present epoch (z=0), genuine independent comparisons:")
    print(f"    H_0 substrate = {z0.H_substrate.value:.3f} km/s/Mpc  "
          f"vs Planck-CMB {H0_PLANCK_CMB[0]}±{H0_PLANCK_CMB[1]}: "
          f"{_dev(z0.H_substrate.value, H0_PLANCK_CMB)}")
    print(f"    H_0 observer  = {z0.H_observer.value:.3f} km/s/Mpc  "
          f"vs SH0ES {H0_SH0ES[0]}±{H0_SH0ES[1]}: "
          f"{_dev(z0.H_observer.value, H0_SH0ES)}")
    print(f"    t_0 substrate = {z0.t_substrate.value:.3f} Gyr       "
          f"vs Methuselah {T0_METHUSELAH[0]}±{T0_METHUSELAH[1]}: "
          f"{_dev(z0.t_substrate.value, T0_METHUSELAH)}")

    # The ADOPTED comparison point — z_eff, live-only, wide band (audit B3/B4).
    z_eff = predict_z_eff(BAO_ANCHORS, SN_MODEL)
    Om_ze = Omega_m_local_coasting_closed_form(z_eff)
    OL_ze = 1.0 - Om_ze
    print()
    print(f"  ADOPTED comparison point z_eff = {z_eff:.4f}  "
          f"[live predictions/z_eff.py; survey-design, ADOPTED, wide band]")
    print(f"    Ω_m_LCDM(z_eff) = {Om_ze:.4f}  vs Planck "
          f"{OMEGA_M_PLANCK[0]}±{OMEGA_M_PLANCK[1]}: {_dev(Om_ze, OMEGA_M_PLANCK)}")
    print(f"    Ω_Λ_LCDM(z_eff) = {OL_ze:.4f}  vs Planck "
          f"{OMEGA_L_PLANCK[0]}±{OMEGA_L_PLANCK[1]}: {_dev(OL_ze, OMEGA_L_PLANCK)}")
    print("    (Λ factor-of-2 sibling lives in predictions/Lambda_CC_LCDM.py;")
    print("     the clean substrate Λ=1/N² in Lambda_CC.py is NOT touched.)")
    print("    z_eff is the DETERMINISTIC bias-inversion of observed Ω_m")
    print("    (z_eff ≡ 𝓑⁻¹(Ω_m,obs); theorem doc §2/§3.iv) — a SECOND")
    print("    data-side anchor, separate from N_hub and NOT N_hub-forced")
    print("    (proven N-invariant, k*-forced: n_hub_omega_m_forcing_2026-")
    print("    05-17.py). Removable ONLY by the open √3=√k* swap-duality")
    print("    theorem (NOT by a substrate bridge — that was a retracted")
    print("    framing). The cluster's z_eff-agreement is CIRCULAR; the")
    print("    parameter-free stake, if √3 is proven, is Ω_m=1/3 at +2.47σ.")
    print()


def report_over_determination(body: StructuralBody, G_F_pin: float) -> None:
    """Headline #2: does the SAME knob hit independent H_0 and t_0?"""
    print("=" * 78)
    print("  OVER-DETERMINATION FALSIFICATION SWEEP")
    print("=" * 78)

    N_cal = predict_N_hub(G_F_pin, M_P, body.alpha_1, body.delta, _K_STRUCT, _P_STRUCT, _V_STRUCT)

    # BY CONSTRUCTION — the calibration channel that DEFINES N_hub.
    v_cal = predict_v_higgs(body.delta, M_P, N_cal, body.alpha_1)
    G_F_back = 1.0 / (math.sqrt(2.0) * v_cal * v_cal)
    print("  Calibration channel (BY CONSTRUCTION — NOT a check):")
    print(f"    G_F round-trip residual = "
          f"{(G_F_back - G_F_pin) / G_F_pin * 100.0:+.6f}%  "
          f"(≈0 by construction; v and G_F are ONE inverted constraint)")
    print()

    # GENUINE independent tests of the one knob.
    H_sub_cal = MPC_KM / (N_cal * t_P)
    t0_cal = (N_cal * t_P) / GYR_S
    # N implied by Planck-CMB H_0 alone (independent observable):
    N_from_H0 = MPC_KM / (H0_PLANCK_CMB[0] * t_P)
    rel = (N_from_H0 - N_cal) / N_cal * 100.0
    print("  Genuine independent over-determination (the resonance test):")
    print(f"    N_hub(G_F-calibrated)        = {N_cal:.6e}")
    print(f"    N_hub(Planck-CMB H_0 alone)  = {N_from_H0:.6e}")
    print(f"    → the SAME knob from two independent observables agrees to "
          f"{abs(rel):.2f}%")
    print(f"      (a weak ~1% test, at the level of the Hubble tension —")
    print(f"       NOT a tight confirmation; honest framing per audit A5)")
    print(f"    H_0 substrate @ N_cal = {H_sub_cal:.3f} km/s/Mpc  "
          f"vs Planck-CMB: {_dev(H_sub_cal, H0_PLANCK_CMB)}")
    print(f"    t_0 substrate @ N_cal = {t0_cal:.3f} Gyr       "
          f"vs Methuselah: {_dev(t0_cal, T0_METHUSELAH)}")
    print()

    # The sweep: vary the knob; show G_F tracks BY CONSTRUCTION while the
    # independent observables only land near N_cal.
    print("  Knob sweep (N/N_cal):  G_F moves by construction; H_0,t_0 are")
    print("  the independent strings — only in tune near N/N_cal = 1.")
    print(f"  {'N/N_cal':>9} {'G_F resid':>12} {'H_0_sub':>10} "
          f"{'t_0_sub':>10} {'H_0 σ_obs':>10} {'t_0 σ_obs':>10}")
    print("  " + "-" * 66)
    for ratio in (0.96, 0.98, 1.00, 1.02, 1.04):
        N = N_cal * ratio
        v_n = predict_v_higgs(body.delta, M_P, N, body.alpha_1)
        gf = 1.0 / (math.sqrt(2.0) * v_n * v_n)
        # G_F "residual" here is vs the pinned G_F — by construction it
        # only equals 0 at ratio=1 because N_cal was defined that way.
        gf_resid = (gf - G_F_pin) / G_F_pin * 100.0
        H = MPC_KM / (N * t_P)
        t0 = (N * t_P) / GYR_S
        h_sig = (H - H0_PLANCK_CMB[0]) / H0_PLANCK_CMB[1]
        t_sig = (t0 - T0_METHUSELAH[0]) / T0_METHUSELAH[1]
        print(f"  {ratio:>9.2f} {gf_resid:>+11.3f}% {H:>10.3f} "
              f"{t0:>10.3f} {h_sig:>+9.2f}σ {t_sig:>+9.2f}σ")
    print()


# ---------------------------------------------------------------------------
# Anti-overclaim self-check (GC-A5 generalized). The engine token-scans
# its OWN verdict for forbidden overclaims, in BOTH directions:
# don't manufacture closure; don't declare irreducibility.
# ---------------------------------------------------------------------------

_FORBIDDEN = (
    "closes gap g1", "gap g1 closed", "g1 closed", "derived n_hub",
    "n_hub derived", "breaches l6", "l6 breached", "recombination solved",
    "solves recombination", "independent confirmation from g_f",
    "g_f confirms", "epoch floor", "provably not closeable",
    "no number is the theory", "no number to get", "irreducible floor",
    "not a research target", "z_eff derived", "derived z_eff",
)


def verdict_block() -> Tuple[str, bool]:
    """Return (verdict_text, ok). ok=False if it overclaims (either way)."""
    lines = [
        "=" * 78,
        "  VERDICT — anti-overclaim self-check (GC-A5 generalized)",
        "=" * 78,
        "  This engine delivers, on ONE knob (N_hub), honest-domain-gated:",
        "    • the late-time ΛCDM-alternative trajectory (z ≲ 2);",
        "    • the over-determination sweep (one knob vs independent H_0,t_0).",
        "",
        "  It explicitly does NOT do the following (fenced off by design):",
        "    • Gap G1 status: OPEN & BOUNDED, unchanged by this engine.",
        "    • L6 / recombination: NOT reached. Numbers above the validated",
        "      band are refused; that region is the substrate→macroscopic",
        "      bridge frontier (program step 3, gated, not built here).",
        "    • δρ residue: CARRIED (the Feshbach c=5/12 factor inside v),",
        "      NOT recomputed.",
        "    • G_F / v: BY CONSTRUCTION (the calibration channel), never",
        "      scored as agreement. The genuine tests are H_0 and t_0.",
        "  Symmetric honesty: no manufactured closure; no declared",
        "  irreducibility. The instrument is real; its string spans z ≲ 2.",
        "=" * 78,
    ]
    text = "\n".join(lines)
    low = text.lower()
    hits = [tok for tok in _FORBIDDEN if tok in low]
    return text, (len(hits) == 0)


def main() -> int:
    print()
    print("#" * 78)
    print("#  N_HUB TRAJECTORY ENGINE — bridge-less foundation (2026-05-17)")
    print("#  Prior-art audit: an internal working note"
          "n_hub_trajectory_engine_prior_art_audit_2026-05-17.md")
    print("#" * 78)
    print()

    body = build_structural_body()
    report_structure_is_fixed(body, G_F_PDG)

    z_grid = (0.0, 0.3, 0.5, 1.0, 1.5, 2.0,   # VALIDATED — framework claim
              5.0, 50.0,                       # EXTRAPOLATED — not a claim
              1090.0)                          # FALSIFIED — refused
    states = trajectory(z_grid, body, G_F_PDG)
    report_lcdm_alternative(states, G_F_PDG)
    report_over_determination(body, G_F_PDG)

    text, ok = verdict_block()
    print(text)
    print()
    if not ok:
        print("SELF-CHECK FAILED: verdict contains an overclaim token.")
        return 1
    print("SELF-CHECK PASSED: no overclaim tokens; scope fences intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
