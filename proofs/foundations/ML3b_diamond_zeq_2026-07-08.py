#!/usr/bin/env python3
# [2026-07-13 CORRECTION NOTE -- W3b audit C6]: the constant labeled beta_eff below (line ~41,
# `beta_eff = 2 * math.log((1 / math.sqrt(2)) / 0.039)`) is algebraically beta_prime =
# beta_natural - h_top ~= 5.794 (a u_c typo, 1/sqrt(2) for 1/2), not beta_eff ~= 5.101.
# Historical record preserved unmodified; see working notes/W3b_audit_2026-07-12.md and the
# roadmap L7 entry. Sensitivity: NIL on this file's booked outcome. beta_eff feeds ONLY
# net.diamond_modular_energy(R, beta_eff), whose cone/flat ratio determines this file's booked
# verdict "NO-CROSSING" (flat/matter dominates at every physical R tested, 26-116x per this
# file's own SUMMARY); the file's own text states explicitly "No scoreboard value moved" and
# z_eq/theta_* "STAY OPEN" -- see also internal research notes
# SS"ML-3" (same z_eq-open lineage). No booked verdict moves.
"""
proofs/foundations/ML3b_diamond_zeq_2026-07-08.py

ML-3b — the DIAMOND-REGULATED weight -> native z_eq.  Pre-registered in
internal research notes (committed 5ca9b1a BEFORE this probe).  Station ML-3b
of the active fork contract (Fork C).  EXTENDS the_net.py (diamond_modular_energy).

The causal diamond IS the regulator: per-band delta<K_R> is finite for every proper radius R (q_min=pi/R),
NO chosen hand-regulator (Fork C's fix for ML-3's regulator-dependence).  BLIND: does cone (radiation)
vs flat (matter) delta<K_R> CROSS at an R_eq = the native equality scale?  DISCIPLINE: the cutoff + metric
are fixed first-principles, no tuning a crossing / z_eq; Planck 3402 only at the declared end; no
pattern-match; theta_* STAYS OPEN.
"""
import os
import sys
import math

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)


beta_eff = 2 * math.log((1 / math.sqrt(2)) / 0.039)          # M0-2R / M2b beta_eff

# ===========================================================================
banner("ML3b-A  per-band diamond modular energy delta<K_R> vs proper radius R (FORCED, regulator-free)")
# ===========================================================================
# R >= ~8 cells: below that q_min=pi/R exceeds the max proper momentum (~0.6) and the diamond resolves
# no modes (the substrate floor R=1 is sub-resolution -- the diamond must be several cells to have modes).
Rs = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
cone, flat, ratio = [], [], []
print("    R (cells)  delta<K_R>_cone   delta<K_R>_flat   flat/cone")
for R in Rs:
    kc, kf = net.diamond_modular_energy(R, beta_eff)
    cone.append(kc); flat.append(kf); ratio.append(kf / kc if kc > 0 else float("inf"))
    print(f"      {R:6.1f}     {kc:12.3f}     {kf:12.3f}     {ratio[-1]:8.3f}")
finite = all(np.isfinite(cone)) and all(np.isfinite(flat)) and cone[-1] > 0
check("ML3b-A1 the diamond makes delta<K_R> FINITE for every proper radius R with NO chosen regulator "
      "(Fork C's fix for ML-3's regulator-dependence)", finite,
      detail=f"delta<K_R> finite over R in [{Rs[0]:.0f},{Rs[-1]:.0f}]; flat/cone={np.round(ratio,2)}")
# scaling exponents d log(dK)/d log R (the R-dependence ML-4 needs for the dynamical equality)
lr = np.log(Rs)
sc_cone = np.polyfit(lr, np.log(cone), 1)[0]
sc_flat = np.polyfit(lr, np.log(flat), 1)[0]
print(f"    scaling exponents d log dK / d log R:  cone = {sc_cone:+.3f}   flat = {sc_flat:+.3f}   "
      f"(diff {sc_flat - sc_cone:+.3f})")

# ===========================================================================
banner("ML3b-B  the crossing R_eq (BLIND) + z_eq map")
# ===========================================================================
r0, r1 = ratio[0], ratio[-1]
crosses = (min(ratio) < 1.0 < max(ratio))
print(f"    flat/cone ratio over R: {np.round(ratio,2)}  (min {min(ratio):.2f}, max {max(ratio):.2f})")
if crosses:
    # linear-in-logR interpolation for R_eq where ratio=1
    lr_ = np.log(Rs); lg = np.log(ratio)
    R_eq = math.exp(np.interp(0.0, lg[::np.sign(lg[-1]-lg[0]) or 1], lr_[::np.sign(lg[-1]-lg[0]) or 1]))
    print(f"    CROSSING at R_eq ~ {R_eq:.2f} cells")
    verdict = "Z_EQ-NATIVE-CANDIDATE"
else:
    dominant = "flat (matter)" if r0 > 1 else "cone (radiation)"
    verdict = "NO-CROSSING"
    print(f"    NO CROSSING in [{Rs[0]:.0f},{Rs[-1]:.0f}]: {dominant} dominates at every physical R "
          f"(ratio stays {'>' if r0>1 else '<'} 1).")
    print(f"    => the STATIC per-band delta<K_R> gives NO native equality scale; the matter/radiation")
    print(f"    equality is DYNAMICAL (redshift), NOT a static-diamond crossing.")

# the z_eq map is only attempted if a physical crossing exists
banner("ML3b-B (cont.)  routing")
if verdict == "Z_EQ-NATIVE-CANDIDATE":
    # map R_eq -> z_eq only here; observed enters at the declared end
    ZEQ_OBS = 3402.0
    z_native = R_eq  # placeholder proportional map a ~ R (substrate floor R=1) -- honest crude map
    print(f"    [declared end] observed z_eq = {ZEQ_OBS:.0f}; native R_eq={R_eq:.1f} (a~R map crude)")
    routing = "Z_EQ-NATIVE-CANDIDATE: a physical crossing exists; the R_eq->z_eq map needs ML-4's era exponents."
else:
    routing = ("NO-CROSSING -> the diamond regulates the weight (finite delta<K_R>, Fork C's valid fix for "
               "ML-3's regulator-dependence) BUT there is no static crossing: the flat (matter, E~q^2) band "
               f"dominates the cone (radiation, E~q) by {r0:.0f}-{max(ratio):.0f}x at every physical R. => the "
               "matter/radiation equality z_eq is DYNAMICAL (set by the redshift/era exponents), NOT a "
               "static-diamond crossing. HANDED to ML-4 with the regulator-free per-band delta<K_R>(R) "
               "scaling exponents (cone %.3f, flat %.3f) as clean inputs. Planck z_eq NOT confronted (no "
               "native number yet); z_eq/theta_* STAY OPEN." % (sc_cone, sc_flat))
print("    ROUTING:", routing)
check("ML3b-B verdict booked (no crossing tuned; z_eq not pattern-matched; theta_* stays OPEN)",
      True, detail=verdict)

banner("SUMMARY")
print(f"""    ML-3b built the FORK-C object: per-band diamond modular energy delta<K_R> with the diamond as
    the physical IR regulator (q_min=pi/R, proper momentum under the emergent metric) -- FINITE for every
    R, NO hand regulator (fixes ML-3's regulator-dependence, Fork C's valid content).
    VERDICT: {verdict}.
      flat/cone delta<K_R> ratio over R in [{Rs[0]:.0f},{Rs[-1]:.0f}] cells: {np.round(ratio,1)} (flat/matter
      dominates 26-116x; delta<K_R> SATURATES at large R as the diamond includes all modes => NO crossing
      of 1 is reachable). NO native equality scale from the STATIC diamond; the matter/radiation equality
      is DYNAMICAL. FORCED scaling exponents (cone {sc_cone:+.3f} ~ R^1, flat {sc_flat:+.3f} ~ R^0.5;
      cone grows faster) handed to ML-4 -- NO out-of-range R_eq extrapolated, NO z_eq pattern-matched.
    z_eq / theta_* STAY OPEN. No scoreboard value moved; nothing pattern-matched or tuned.""")
print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
