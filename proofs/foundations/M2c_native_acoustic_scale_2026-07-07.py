#!/usr/bin/env python3
"""
proofs/foundations/M2c_native_acoustic_scale_2026-07-07.py

M2(c) — the native coasting acoustic scale -> theta_*. Pre-registered in
internal research notes (committed 95f9f7a BEFORE this
file). M-track M2. Executor: a model Builds on M2(a) c_s = v/sqrt(3) + M2(b) two-component
fluid (cone=radiation + flat-band=matter).

HIGH-RISK / falsification confront. The standard r_s/D_A is DEAD in coasting (r_s log-
diverges). This probe ATTEMPTS the native construction and reports HONESTLY. theta_* is a
genuine falsification exposure; the overclaim guard is BINDING: hold BOTH "the standard
formula is inapplicable" AND "the native prediction is open, could falsify". Do NOT import
LCDM era structure to manufacture 0.0104. theta_* stays OPEN unless it genuinely falls out.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# forced / framework inputs (NO tuning)
c_s_over_c = 1.0 / math.sqrt(3.0)          # M2(a): c_s = v/sqrt(3) => c_s/c = 1/sqrt(3)
N_hub = 8.394881e60                         # framework N_hub (a_start/a_0 ~ 1/N_hub in coasting)
z_rec = 1089.0                              # recombination redshift
theta_planck = 0.0104109                    # rad (Planck observed; the confront target, NOT fitted)

# theta_* = r_s / D_A, both comoving. theta_* = (c_s/c) * Deta(start->rec) / Deta(rec->now),
# Deta = conformal-time interval = int dt/a. The units + c cancel: theta_* is a RATIO.
print(f"    forced inputs: c_s/c = 1/sqrt(3) = {c_s_over_c:.5f} (M2a); N_hub = {N_hub:.3e}; "
      f"z_rec = {z_rec:.0f}; Planck theta_* = {theta_planck:.4e} rad")

# ===========================================================================
banner("M2c-0  re-lock the wall: NATIVE COASTING (a ~ t) => theta_* absurd (formula inapplicable)")
# ===========================================================================
# a ~ t => eta = ln t. Deta(start->rec) = ln(a_rec/a_start) = ln(N_hub/(1+z_rec)); a_start=1/N_hub.
# Deta(rec->now) = ln(a_0/a_rec) = ln(1+z_rec).
Deta_coast_pre = math.log(N_hub / (1 + z_rec))
Deta_coast_post = math.log(1 + z_rec)
theta_coast = c_s_over_c * Deta_coast_pre / Deta_coast_post
print(f"    Deta(start->rec) = ln(N_hub/(1+z_rec)) = {Deta_coast_pre:.2f}  (log-DIVERGENT structure,")
print(f"      cut at the first tick a_start=1/N_hub; the huge value = the pathology)")
print(f"    Deta(rec->now)   = ln(1+z_rec)         = {Deta_coast_post:.2f}")
print(f"    => theta_*(coasting) = {theta_coast:.3f} rad")
check("M2c-0 coasting theta_* is ABSURD (> 2pi rad): the standard r_s/D_A is INAPPLICABLE in coasting",
      theta_coast > 2 * math.pi,
      detail=f"theta_* = {theta_coast:.2f} rad = {theta_coast/theta_planck:.0f}x Planck; r_s >> D_C "
             "(INVERTED hierarchy: observed is r_s << D_C)")
print("    CONFIRMS the prior CHARACTERIZED-NEGATIVE (recombination_theta_star_coasting_2026-05-25):")
print("    coasting inverts the sound-horizon/distance hierarchy. This is NOT a framework prediction --")
print("    it is confirmation the FRW formula does not apply. The native construction is un-built.")

# ===========================================================================
banner("M2c-1  the two-component test: RADIATION ERA (a ~ t^1/2) cures the divergence?")
# ===========================================================================
# M2(b) derived a RADIATION component (the cone). IF a ~ t^1/2 (radiation-dominated), the conformal
# integral CONVERGES at t->0: eta = int dt/t^{1/2} ~ t^{1/2} ~ a (FINITE). Then:
#   r_s   = int_0^{t_rec} c_s dt/a = 2 c_s t_0 a_rec        (finite; the divergence is GONE)
#   D_C   = int_{t_rec}^{t_0} c dt/a = 2 c t_0 (1 - a_rec)
#   theta_* = (c_s/c) a_rec/(1-a_rec)
a_rec = 1.0 / (1 + z_rec)
theta_rad = c_s_over_c * a_rec / (1 - a_rec)
diverges_cured = True   # analytic: int_0 t^{-1/2} dt converges
check("M2c-1 a~t^1/2 (radiation era) CURES the r_s divergence (conformal integral converges at t->0)",
      diverges_cured, detail="eta = int dt/t^{1/2} ~ 2 t^{1/2} ~ a, FINITE (vs ln t divergent for a~t)")
print(f"    => theta_*(radiation-era, single-era estimate) = (1/sqrt3) a_rec/(1-a_rec) = {theta_rad:.3e} rad")
print(f"       vs Planck {theta_planck:.3e} rad  -> ratio {theta_rad/theta_planck:.2f} "
      f"({'too small' if theta_rad < theta_planck else 'too large'}, single-era crude)")
print(f"    FLAG (binding): a~t^1/2 is an ASSUMPTION here. Its NATIVE status is THE open question")
print(f"    (M2c-2). Getting a finite theta_* with an imported era is NOT a derivation -- do NOT bank it.")

# ===========================================================================
banner("M2c-2  the reconciliation (characterize the precise open tension; do NOT resolve by fiat)")
# ===========================================================================
print("""    THE OPEN TENSION (named, not resolved):
      - SUBSTRATE-COUNTING => COASTING a ~ t (theorem-grade: H(z)=H_0(1+z), cascade D1+D2+D3).
        In coasting the comoving sound horizon LOG-DIVERGES and theta_* is absurd (M2c-0).
      - M2(b) TWO-COMPONENT FLUID => a RADIATION component (the cone, w=1/3) that, via ANY
        Friedmann-like sourcing, would give a RADIATION ERA a ~ t^1/2 => finite r_s (M2c-1).
      WHICH governs the pre-recombination expansion? This is a genuine, unresolved structural
      question. The B2 scoping flagged the era structure (a~t^1/2/t^2/3) as 'imported not native';
      M2(b) now supplies the radiation component that COULD make it native -- but the reconciliation
      (coasting substrate-counting vs fluid-sourced expansion) is NOT built. theta_* stays OPEN.""")

# ===========================================================================
banner("M2c-3  the bias-function frame (report-only; the likely-correct confront)")
# ===========================================================================
z_eff = 1.916
print(f"""    The bias-function theorem already closes Omega_m/Omega_Lambda/w_DE as LCDM-FITTER
    extractions of the ONE coasting H(z) at z_eff = {z_eff}. theta_* is very likely ALSO a
    fitter extraction, NOT a native coasting quantity: a LCDM observer fitting the substrate's
    coasting universe reads off a theta_* that maps to the native acoustic feature via the bias
    function at z_eff. The correct confront is thus theta_*(native feature) --bias(z_eff)--> observed,
    a MULTI-SESSION build (the same machinery that closed the other LCDM parameters). Reported as
    the likely-correct direction, NOT executed. No number claimed.""")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
print(f"""    M2(c) OUTCOME = DIAGNOSTIC (theta_* stays OPEN; NOT solved, NOT falsified). Using the new
          M2a/M2b inputs, the native acoustic scale was ATTEMPTED and the failure modes quantified:
            - NATIVE COASTING (a~t): theta_* = {theta_coast:.1f} rad -- ABSURD (>2pi), the standard
              r_s/D_A formula is INAPPLICABLE (r_s >> D_C, inverted hierarchy). Confirms prior.
            - RADIATION ERA (a~t^1/2): CURES the divergence (r_s finite); single-era theta_* =
              {theta_rad:.2e} rad ({theta_rad/theta_planck:.1f}x Planck) -- but a~t^1/2 is an
              ASSUMPTION whose native status is unresolved (M2c-2). NOT banked.
    THE PRECISE OPEN PIECES (named): (i) the native pre-recombination expansion -- does M2(b)'s derived
          radiation component source a radiation era (a~t^1/2, finite r_s), reconciling with the
          theorem-grade coasting a~t? (ii) the bias-function/z_eff extraction -- theta_* is likely a
          LCDM-fitter quantity, needing the bias map (multi-session).
    DISCIPLINE HELD: theta_* is a genuine falsification exposure; the standard formula is inapplicable
          AND the native prediction is open. No LCDM era imported to manufacture 0.0104. theta_* stays
          ❌ OPEN. No scoreboard value moved.""")
print("RESULT:", "ALL CHECKS PASS -- M2(c) DIAGNOSTIC (theta_* OPEN; failure modes quantified, "
      "pieces named)" if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
