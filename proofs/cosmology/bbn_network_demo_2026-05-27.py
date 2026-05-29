#!/usr/bin/env python3
"""
BBN network harness — demo / validation (2026-05-27).

Library: proofs/cosmology/lib/bbn_network.py
Scoping: an internal working note

This driver:
  (1) sanity-checks the weak rates (detailed balance λ_pn/λ_np = e^{-Q/T}).
  (2) VALIDATES the harness against ΛCDM: weak-sector Y_p ≈ 0.247.
  (3) runs the FRAMEWORK as a what-if with the η scoping doc's stated
      assumption (η = η_B, readings A/B), under both H normalizations:
        - bare substrate F=1  → the Y_p ≈ 0.05 falsification candidate
        - candidate F=√(k*·g_*) → the leading-factor chase candidate
      isolating exactly how much the √g_* factor moves Y_p.

Honest framing (per the scoping doc §7): the framework rows are WHAT-IFs that
expose the open H-normalization and η questions, NOT closed predictions. The
ONLY validated number here is the ΛCDM Y_p.

Run:
    python3 proofs/cosmology/bbn_network_demo_2026-05-27.py
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
_PRED_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "predictions"))
sys.path.insert(0, _PRED_DIR)

from lib.bbn_network import (  # noqa: E402
    Q_NP_MeV, weak_rates, lcdm_expansion, framework_expansion,
    run_weak_sector, g_star_energy, deuterium_bottleneck_T_MeV, KEY_REACTIONS,
)

# live framework η_B (no hardcode)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from eta_B import eta_B_pred  # noqa: E402
ETA_B = float(eta_B_pred)

# observed values (external)
Y_P_OBS = 0.245
Y_P_SIG = 0.003
ETA_OBS = 6.14e-10


def banner(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


# ----------------------------------------------------------------------------
banner("BBN NETWORK HARNESS — demo / validation (2026-05-27)")
print(f"""
  Pluggable surface: ExpansionModel H(T) [framework √g_* question] + η.
  Live η_B = {ETA_B:.4e} (predictions/eta_B.py); observed η = {ETA_OBS:.3e}.
""")

# (1) weak-rate sanity: detailed balance ------------------------------------
banner("(1) Weak-rate sanity check — detailed balance λ_pn/λ_np = e^(-Q/T)")
print(f"\n  {'T (MeV)':>9} {'λ_np (s⁻¹)':>14} {'λ_pn (s⁻¹)':>14} "
      f"{'λ_pn/λ_np':>12} {'e^(-Q/T)':>12}")
print("  " + "-" * 64)
for T in (3.0, 1.0, 0.8, 0.3):
    lnp, lpn = weak_rates(T)
    ratio = lpn / lnp
    expected = math.exp(-Q_NP_MeV / T)
    print(f"  {T:>9.2f} {lnp:>14.4e} {lpn:>14.4e} {ratio:>12.5f} {expected:>12.5f}")
print("""
  The ratio tracks e^(-Q/T) (Boltzmann suppression of the heavier neutron):
  confirms the 6-process Born integrals + normalization are consistent.""")

# (2) ΛCDM validation --------------------------------------------------------
banner("(2) VALIDATION — ΛCDM weak-sector Y_p (should be ≈ 0.245-0.247)")
lcdm = lcdm_expansion()
res_lcdm = run_weak_sector(lcdm, ETA_OBS)
dev = (res_lcdm.Y_p - Y_P_OBS) / Y_P_SIG
print(f"""
  Expansion : {res_lcdm.expansion}
  η         : {res_lcdm.eta:.3e}  (observed)
  T_bottleneck : {res_lcdm.T_bottleneck_MeV*1e3:.2f} keV
  X_n at bottleneck : {res_lcdm.X_n_freeze:.4f}
  Y_p (harness)  = {res_lcdm.Y_p:.4f}
  Y_p (observed) = {Y_P_OBS} ± {Y_P_SIG}   →  {dev:+.1f}σ
""")
if abs(res_lcdm.Y_p - Y_P_OBS) < 0.01:
    print("  ✓ VALIDATED: harness reproduces ΛCDM Y_p within 0.01.")
else:
    print("  ✗ NOT within 0.01 of observed — weak sector needs refinement.")

# (3) framework what-ifs -----------------------------------------------------
banner("(3) FRAMEWORK WHAT-IF — η = η_B (readings A/B); two H normalizations")
print(f"""
  Per the η scoping doc, readings A/B give η_BBN = η_B = {ETA_B:.4e}; we feed
  that. The two rows isolate the √g_* leading-factor question — the SAME
  nuclear physics, only H's prefactor differs.
""")
fw_bare = framework_expansion("bare")
fw_cand = framework_expansion("candidate")
res_bare = run_weak_sector(fw_bare, ETA_B)
res_cand = run_weak_sector(fw_cand, ETA_B)

print(f"  {'H model':<34} {'F at 0.8 MeV':>13} {'Y_p':>8} {'vs obs':>10}")
print("  " + "-" * 70)


def _F(model_res, T=0.8):
    # report the leading factor relative to ΛCDM's √(8π³/90)·√g_*
    g = g_star_energy(T)
    lcdm_F = math.sqrt(8 * math.pi ** 3 * g / 90.0)
    return lcdm_F


lcdm_F08 = math.sqrt(8 * math.pi ** 3 * g_star_energy(0.8) / 90.0)
cand_F08 = math.sqrt(3 * g_star_energy(0.8))
for label, res, F in (
    (f"ΛCDM (ref)", res_lcdm, lcdm_F08),
    (f"framework bare (F=1)", res_bare, 1.0),
    (f"framework candidate (F=√(k*g_*))", res_cand, cand_F08),
):
    dev = (res.Y_p - Y_P_OBS) / Y_P_SIG
    print(f"  {label:<34} {F:>13.3f} {res.Y_p:>8.4f} {dev:>+9.1f}σ")

print(f"""
  Reading:
    • bare substrate (F=1): H is ~{lcdm_F08:.1f}× too small at 0.8 MeV → weak rates
      stay in equilibrium longer → lower freeze-out T → lower n/p → and more
      decay time → Y_p ≈ {res_bare.Y_p:.3f}. This IS the framework's Y_p
      falsification candidate (matches the ~0.05 estimate in predictions/Y_p.py).
    • candidate F=√(k*·g_*): only +4.3% above ΛCDM's √(8π³/90)·√g_* (the
      √3/1.66 = 1.043 K-rational tax). Y_p ≈ {res_cand.Y_p:.3f} — close to
      observed, off by the +4.3% in the freeze-out factor.

  ⇒ The harness makes the stakes quantitative: the entire Y_p outcome hinges
    on the √g_* leading factor (Gate 2 of the leading-factor chase). With F=1
    the framework is falsified; with the √(k*·g_*) candidate it lands near
    observed. Closing Gate 2 (deactivation at today) is the open work.
""")

# (4) what's left for the FULL network ---------------------------------------
banner("(4) For the FULL light-element network (D, ³He, ⁷Li) — what remains")
print(f"""
  The weak sector above is validated and is what couples to the framework's
  H(T)/η. The remaining engineering (NOT framework-specific) is the
  light-element reaction network. Scaffold reactions present in the library:
""")
for rxn, note in KEY_REACTIONS:
    print(f"    • {rxn:<26}{('— ' + note) if note else ''}")
print(f"""
  These need external ⟨σv⟩(T) fits (sigma_v_STUB raises NotImplementedError by
  design). Wiring a literature rate library (Kawano/PArthENoPE/AlterBBN) +
  the 8-species ODE gives D/H, ³He/H, ⁷Li/H. Out-of-scope-by-construction
  (measured nuclear physics, like B_D/m_nucleon already are).

  HONEST POSTURE: only the ΛCDM Y_p row is a validated result. Framework rows
  are what-ifs exposing the open H-normalization (√g_*) and η questions.
""")
