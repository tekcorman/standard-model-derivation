#!/usr/bin/env python3
"""
proofs/cosmology/cS_extent_vs_flux_2026-05-28.py

ATTEMPT to decide the surviving horizon-entropy fork {c_S = 1 vs 2}:
  c_S = 1  net-node EXTENT  (create MINUS destroy; the worldline tick count N)
  c_S = 2  irreversible-operation FLUX (create PLUS destroy crossing the horizon)

This is the single decidable gate left after cS_horizon_entropy_blind_2026-05-28.py
NARROWED the trilemma to this dilemma (the Shannon-surprise c_S~2.585 horn was
eliminated: it is the epistemic E_obs, not the thermodynamic deltaQ=TdS entropy).

WHY IT IS THE WHOLE BALLGAME
----------------------------
The combined holographic + Cai-Kim treatment gives H = c_S * M_Pl / (2N), so the
cascade relation N = 1/(H t_P) is reproduced iff c_S = 2, and with the framework's
own entropy c_S = 1 the Cai-Kim flux coupling gives G_eff = 2G. So:
  c_S = 2  -> kappa = M_Pl/2, G_eff = G, cascade consistent       (flagship)
  c_S = 1  -> G_eff = 2G  (Newton's constant wrong by 2)          (form-level only)

BLIND DISCIPLINE
----------------
Decide c_S from the framework's bit-accounting on the causal horizon, WITHOUT
reference to which value lands G_eff = G. The closure needs c_S = 2 to be FORCED
(not merely consistent). Pre-registered outcomes:
  FORCED-2     a structural relation forces 2 irreversible units per horizon-tick
               -> flagship gravity closes.
  FORCED-1     the accounting forces the net-extent reading (1 unit/tick)
               -> G_eff = 2G; flagship does NOT close; gravity form-level; AND the
                  perceptual-surface "1 bit/t_P" entropy is in tension with G.
  NOT-FORCED   neither is forced by present structure -> R2 stands, form-level.

The test is whether ANY framework quantity supplies a forced factor 2 between the
horizon-entropy increment per tick and the net worldline tick. We enumerate every
candidate the substrate offers and measure it.

sympy + numeric. References (tracked):
  predictions/N_hub_derivation.md      (N = age/t_P = 1 node per Planck time; worldline)
  predictions/lambda_toggle_rate.py    (per-edge create rate = destroy rate = 1/5)
  predictions/k_star.py                (coordination k* = 3)
  docs/theorems/theorem_observer_energy_functional.md  (1 Bayesian update / observation;
       erasure >= 1 bit; T uncalibrated)
  Landauer 1961; Bennett 1973; Cai-Kim 2005 (JHEP 0502:050)
"""
from __future__ import annotations
import math
import sys

import sympy as sp

FAIL = []


def abort(tag, msg):
    print(f"\n  X ABORT [{tag}] — HONEST NEGATIVE\n    {msg}")
    FAIL.append(tag)


def head(s):
    print("\n" + "=" * 78 + f"\n  {s}\n" + "=" * 78)


print(__doc__)

# ----------------------------------------------------------------------
# E-A1 — the over-determination: both schemes agree iff c_S = 2 (sympy)
# ----------------------------------------------------------------------
head("E-A1 — closure needs c_S=2; framework entropy c_S=1 gives G_eff=2G")

kappa, cS, N, H, M = sp.symbols("kappa c_S N H M_Pl", positive=True)
# Cai-Kim flux coupling with S = c_S R_A M_Pl (S' = c_S M_Pl), T = kappa:
G_eff = 1 / (kappa * cS * M)                       # G_eff(kappa, c_S)
# Volume scheme fixes kappa from standard Friedmann + cascade: kappa = M_Pl/2.
kappa_volume = M / 2
G_eff_with_framework_entropy = G_eff.subs({kappa: kappa_volume, cS: 1})
G_eff_if_cS2 = G_eff.subs({kappa: kappa_volume, cS: 2})
G_newton = 1 / M**2
print(f"  Cai-Kim flux coupling: G_eff = 1/(kappa*c_S*M_Pl)")
print(f"  volume scheme fixes kappa = M_Pl/2 (standard Friedmann + cascade clock)")
print(f"  with framework entropy c_S=1:  G_eff = {sp.simplify(G_eff_with_framework_entropy)}"
      f"  = {sp.simplify(G_eff_with_framework_entropy/G_newton)} * G   <-- 2G")
print(f"  with c_S=2:                    G_eff = {sp.simplify(G_eff_if_cS2)}"
      f"  = {sp.simplify(G_eff_if_cS2/G_newton)} * G   <-- G")
ea1 = (sp.simplify(G_eff_with_framework_entropy / G_newton) == 2) and \
      (sp.simplify(G_eff_if_cS2 / G_newton) == 1)
if not ea1:
    abort("E-A1", "over-determination algebra wrong.")
else:
    print("  -> closure REQUIRES a forced reason the horizon entropy is 2/tick, not 1.  OK")

# ----------------------------------------------------------------------
# E-A2 — the framework's worldline accounting: how many entropy units per tick?
#         (three independent angles; all must be measured, not assumed)
# ----------------------------------------------------------------------
head("E-A2 — entropy units per horizon-tick from the framework's accounting")

# Angle (a): observations per tick. The observer reads ONE toggle outcome per
# Planck step (N = 1 node per t_P; perceptual-surface painting = 1 bit/t_P).
obs_per_tick = 1
print(f"  (a) observations per tick (N = 1 node/t_P, worldline painting)   = {obs_per_tick}")

# Angle (b): irreversible Landauer operations per Bayesian update. A posterior
# Beta(a,b) has predecessors {(a-1,b),(a,b-1)} that are valid when a-1>=1, b-1>=1.
# Landauer erasure cost = log2(#predecessors merged). Measure across the model.
def n_predecessors(a, b):
    n = 0
    if a - 1 >= 1:
        n += 1
    if b - 1 >= 1:
        n += 1
    return max(n, 1)
samples = [(a, b) for a in range(1, 8) for b in range(1, 8)]
erase_bits = [math.log2(n_predecessors(a, b)) for a, b in samples]
# generic (interior) updates merge 2 states -> 1 bit; boundary updates -> 0 bits.
interior = [math.log2(n_predecessors(a, b)) for a, b in samples if a > 1 and b > 1]
landauer_per_update_interior = interior[0] if interior else None
print(f"  (b) Landauer bits erased per Bayesian update (interior states)    = "
      f"{landauer_per_update_interior:.3f}   (= log2(2 predecessors) = 1 bit)")
print(f"      max over all sampled states = {max(erase_bits):.3f} bit  (never reaches 2)")

# Angle (c): per-tick toggle EVENTS witnessed. The 2/cycle of lambda_toggle_rate
# is create+destroy per EDGE per CYCLE (T_cycle=5 steps), and is net-zero in steady
# state (create-rate = destroy-rate = 1/5). The observer witnesses 1 outcome/tick.
rate_create = sp.Rational(2, 5) * sp.Rational(1, 2)   # pi_off * p_create = 1/5
rate_destroy = sp.Rational(3, 5) * sp.Rational(1, 3)  # pi_on  * p_destroy = 1/5
net_toggle_rate = rate_create - rate_destroy          # 0  (steady, net-zero)
events_per_tick_witnessed = 1                          # one outcome read per tick
print(f"  (c) per-edge create-rate={rate_create}, destroy-rate={rate_destroy},"
      f" NET={net_toggle_rate} (net-zero); witnessed/tick = {events_per_tick_witnessed}")

units_per_tick = max(obs_per_tick, round(landauer_per_update_interior),
                     events_per_tick_witnessed)
print(f"\n  -> every independent angle gives 1 entropy unit per horizon-tick, NOT 2.")
print(f"     the create+destroy '2' is a per-edge-per-cycle quantity that is")
print(f"     NET-ZERO and does not accumulate as horizon entropy at 2/tick.")
ea2 = (abs(landauer_per_update_interior - 1.0) < 1e-9) and (max(erase_bits) < 2.0) \
      and (net_toggle_rate == 0)
if not ea2:
    abort("E-A2", "worldline accounting did not come out 1 unit/tick.")

# ----------------------------------------------------------------------
# E-A3 — does ANY substrate quantity supply a FORCED factor 2 over the tick?
# ----------------------------------------------------------------------
head("E-A3 — enumerate every candidate forced factor-2; measure each")

k_star = 3   # coordination number (predictions/k_star.py)
candidates = {
    "net-node extent (clock)":        1,                  # the c_S=1 reading
    "edges per node (undirected, k*/2)": sp.Rational(k_star, 2),  # 1.5
    "edges per node (directed, k*)":  k_star,             # 3
    "create+destroy per edge-cycle":  2,                  # net-zero per E-A2; not per-tick
    "Landauer ops per update":        1,                  # E-A2(b)
    "self-inverse toggle apps/cycle":  2,                 # tau^2=id, but 1 app/tick
}
print("  candidate factor (entropy units per net worldline tick):")
forced_two = []
for name, val in candidates.items():
    is_two = (sp.nsimplify(val) == 2)
    per_tick = name in ("net-node extent (clock)", "Landauer ops per update",
                        "edges per node (undirected, k*/2)", "edges per node (directed, k*)")
    tag = "= 2" if is_two else f"= {val}"
    note = "" if per_tick else "  (NOT a per-tick horizon increment: cycle/aggregate)"
    print(f"    {name:34s} {tag}{note}")
    if is_two and per_tick:
        forced_two.append(name)

print(f"""
  Reading off the table: the only candidates equal to 2 are 'create+destroy per
  edge-cycle' and 'self-inverse apps/cycle' — BOTH per-CYCLE aggregates, not
  per-tick horizon increments (E-A2: steady toggling is net-zero; 1 toggle
  application per tick). Every genuine per-tick increment the substrate offers is
  1 (extent / Landauer op) or tracks k*=3 (edge counting: 1.5 or 3) — none is 2.""")
ea3_forced = len(forced_two) > 0
print(f"  candidates that force 2 as a PER-TICK increment: {forced_two if forced_two else 'NONE'}")

# ----------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------
head("VERDICT")
if FAIL:
    print(f"  HONEST NEGATIVE — verifiable-claim aborts tripped: {FAIL}")
    sys.exit(1)

if ea3_forced:
    disposition = "FORCED-2 (flagship closes)"
else:
    disposition = "FORCED-1 / NOT-FORCED (flagship does NOT close)"

print(f"""  DISPOSITION: {disposition}

  The attempt to FORCE c_S = 2 via an extent-vs-flux relation FAILS. Every
  independent angle of the framework's own bit-accounting (E-A2) gives ONE
  entropy unit per horizon-tick:
    (a) one observation per tick (N = 1 node/t_P, the worldline painting);
    (b) one Landauer bit erased per Bayesian update (2 predecessors merged);
    (c) the create+destroy '2' is per-edge-per-CYCLE and NET-ZERO in steady state
        -- it does not accumulate as horizon entropy at 2 per tick.
  And no substrate quantity (E-A3) supplies a forced factor 2 as a per-tick
  horizon increment: the genuine per-tick counts are 1 (extent/Landauer) or track
  k*=3 (edge counting) -- never 2.

  CONSEQUENCE: the framework's own horizon entropy is c_S = 1 (the worldline
  EXTENT, '1 bit per t_P'), the SAME quantity used by the perceptual-surface
  holographic identification. With c_S = 1 the Cai-Kim flux coupling gives
  G_eff = 2G. So the factor of 2 is NOT closed -- it is a genuine internal tension
  between (i) the holographic Friedmann mechanism + the linear worldline entropy
  and (ii) Newton's measured G:

    - the gravitating entropy that the framework actually supplies is c_S = 1;
    - c_S = 2 (needed for G_eff = G and the cascade clock) is NOT forced by any
      framework structure -- it would have to be ADOPTED, which is the goal-seeking
      the blind protocol forbids;
    - therefore kappa / Newton's G does NOT close from the thermal route.

  This SHARPENS R2 (2026-05-17): not merely 'c_S ambiguous' but 'the framework's
  accounting actively favors c_S = 1, which gives G_eff = 2G'. It also undercuts
  the perceptual-surface 'mechanism COMPLETE' claim at the coefficient level: the
  very '1 bit/t_P' entropy that probe relies on is the c_S = 1 that yields 2G.

  DISPOSITION (unchanged from the restored R2 verdict):
    - kappa NOT promotable to predictions/.
    - predictions/G_N.py stays at its asymptotic-safety-conditional grade.
    - Promotable gravity = the FORM (emergent Lorentzian metric; emergent standard
      Friedmann + coasting from the native information-Clausius relation; the
      entropy-temperature compensation, which is c_S-independent and robust). The
      coupling normalization is calibration-fixed, not derived parameter-free.

  What would still reopen it (named, not pursued): a derivation showing the
  GRAVITATING horizon entropy is a distinct object from the observer's worldline
  count -- e.g. the full 2-sphere boundary entropy is twice the single-worldline
  painting -- with the factor 2 FORCED by the boundary geometry. Absent that, the
  worldline accounting gives 1 and gravity stays form-level.
""")
print("=" * 78)
print(f"  EXIT 0 — {disposition}; c_S=1 favored by accounting; gravity form-level")
print("=" * 78)
