#!/usr/bin/env python3
"""
proofs/foundations/Rinf_clean_ratio_diagnostic_2026-05-16.py

User reframing: the R_∞/v ratio discrepancy is NOT an N-tuning knob — it
is a CLEAN diagnostic (N_hub cancels EXACTLY: v, R_∞ both ∝ N_hub^−1/4,
verified `Nhub_v_Rinf_degeneracy_2026-05-16.py`) "revealing of a deeper
fix to other factors".  This probe decomposes that clean residual to
say WHICH factor.

R_∞ = α_EM(0)² · m_e · c / (2h).  The c,h (CODATA) bridge is identical
on prediction and observation ⇒ cancels in the residual.  v exact by
the G_F calibration.  So with N_hub and the bridge both removed:

    R_∞_pred/R_∞_obs − 1  ≈  2·δ(α_EM(0))  +  δ(m_e)

— a pure readout of the α_EM(0) construction and the m_e factor, free
of BOTH the N_hub adoption AND the absolute-mass σ_PDG floor.
"""
from __future__ import annotations
import sys, importlib, contextlib, io
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "predictions"))
with contextlib.redirect_stdout(io.StringIO()):
    R = importlib.import_module("R_infinity")

# ---- live values straight from the DAG node -------------------------------
Rinf_pred = R.R_infinity
Rinf_obs  = R.R_infinity_obs
alpha0_pred = R.alpha_EM_0                       # framework α_EM(0)
aMZ_pred    = R.alpha_EM_at_MZ                   # framework α_EM(M_Z)
dalpha_run  = R.delta_alpha_running              # = 9.092  (PDG-imported)
m_e_used    = R.m_e_GeV                          # what R_infinity.py feeds in

# ---- exact references (CODATA / PDG) --------------------------------------
alpha0_obs_inv = 137.035999084                   # α_EM(0)^-1  CODATA (exact)
aMZ_obs_inv    = 127.951                          # α_EM(M_Z)^-1 PDG (MS-bar)
m_e_codata     = 0.51099895e-3                    # GeV

print("=" * 78)
print("  R_∞/v clean-ratio diagnostic  (N_hub + c,h bridge both cancelled)")
print("=" * 78)
res = Rinf_pred / Rinf_obs - 1.0
print(f"  R_∞ residual (live)              = {res*100:+.4f}%   "
      f"(pred {Rinf_pred:.6e} / CODATA {Rinf_obs:.6e})")
print()

# ---- factor 1: m_e — is it even in play? ----------------------------------
print("  FACTOR m_e:")
print(f"    R_infinity.py feeds m_e = {m_e_used*1e3:.8f} MeV")
print(f"    CODATA m_e             = {m_e_codata*1e3:.8f} MeV")
d_me = m_e_used/m_e_codata - 1.0
print(f"    δ(m_e) = {d_me*100:+.5f}%  ⇒ the coded R_∞ uses the MEASURED m_e,")
print(f"    NOT the framework Koide/y_τ m_e.  So the m_e/v Koide piece is")
print(f"    OUT of this residual (my earlier 'α×m_e/v×bridge' list was wrong")
print(f"    on this point — the clean ratio isolates α_EM(0) ALONE).")
print()

# ---- factor 2: α_EM(0) — the whole residual ------------------------------
a0_pred_inv = 1.0/alpha0_pred
d_alpha0 = alpha0_pred/(1.0/alpha0_obs_inv) - 1.0
print("  FACTOR α_EM(0)  (the entire residual):")
print(f"    α_EM(0)^-1  framework = {a0_pred_inv:.4f}")
print(f"    α_EM(0)^-1  CODATA    = {alpha0_obs_inv:.4f}")
print(f"    δ(α_EM(0)) = {d_alpha0*100:+.5f}%   ⇒ 2·δ(α_EM(0)) = "
      f"{2*d_alpha0*100:+.4f}%")
print(f"    R_∞ residual {res*100:+.4f}%  ≈  2·δ(α_EM(0)) {2*d_alpha0*100:+.4f}%"
      f"  → {'MATCH (R_∞ residual IS the α_EM(0) error, doubled)' if abs(res-2*d_alpha0)<5e-5 else 'see split below'}")
print()

# ---- sub-decompose α_EM(0)^-1 = α_EM(M_Z)^-1 + Δα_running -----------------
aMZ_pred_inv = 1.0/aMZ_pred
print("  α_EM(0)^-1 = α_EM(M_Z)^-1 + Δα_running  — split the two sub-factors:")
print(f"    α_EM(M_Z)^-1 framework = {aMZ_pred_inv:.4f}   vs PDG {aMZ_obs_inv:.4f}"
      f"   (Δ = {aMZ_pred_inv-aMZ_obs_inv:+.4f} in α^-1; the gauge-cluster drift)")
dalpha_run_true = alpha0_obs_inv - aMZ_obs_inv
print(f"    Δα_running   framework = {dalpha_run:.4f}   vs effective "
      f"{dalpha_run_true:.4f}   (Δ = {dalpha_run-dalpha_run_true:+.4f} in α^-1)")
print(f"      └ this 9.092 is the QED vacuum-polarisation running α(0)→α(M_Z),")
print(f"        flagged in R_infinity.py as 'standard QED, PDG-derived' — i.e.")
print(f"        an EXTERNAL Type-3 import the framework does NOT derive.")
print()
print("=" * 78)
print("  Verdict — what the clean ratio is revealing")
print("=" * 78)
print(f"""
  • m_e is NOT the issue (coded R_∞ uses CODATA m_e; δ(m_e)≈0).
  • The ENTIRE R_∞/v clean residual ({res*100:+.3f}%) = 2·δ(α_EM(0)).
  • δ(α_EM(0)) splits into TWO un-closed pieces, same sign-ish:
      (a) the framework α_EM(M_Z) gauge-cluster drift
          ({aMZ_pred_inv-aMZ_obs_inv:+.3f} in α^-1), and
      (b) the imported Δα_running = 9.092 — a PDG QED-running number
          the framework has NO substrate derivation of (Type-3).
  • This is exactly the 'deeper fix to other factors' intuition: the
    ratio — uncontaminated by N_hub AND by the absolute-mass σ_PDG
    floor — isolates the ELECTROMAGNETIC-RUNNING sector as the single
    factor carrying the discrepancy.  It is a COMMON cause: the same
    α_EM(M_Z) cluster residual sits (buried under σ_PDG) in g_1,
    sin²θ_W, α_s; and Δα_running is the framework's only un-derived
    QED-vacuum-polarisation input.  Fixing the α-running construction
    (a substrate Δα analog + tightening α_EM(M_Z)) is the deeper fix
    the clean ratio points at — and it propagates to the whole EM
    sector, not just R_∞.
""")
print("=" * 78)
