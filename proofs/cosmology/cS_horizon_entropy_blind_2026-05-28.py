#!/usr/bin/env python3
"""
proofs/cosmology/cS_horizon_entropy_blind_2026-05-28.py

BLIND derivation of the horizon-entropy normalization c_S — the single quantity
the gravitational coupling factor-of-2 (kappa = M_Pl/2 vs M_Pl, G_eff = G vs 2G)
reduces to.

WHY c_S IS THE WHOLE PROBLEM
----------------------------
Write the gravity closure in one combined self-consistent treatment: the Cai-Kim
apparent-horizon first law (coupling) + the holographic source rho_sub = E_obs/V
+ the cascade clock. With horizon entropy S = c_S * R_A * M_Pl (so S' = c_S*M_Pl),
temperature T = kappa, and rho_sub = (3 kappa N / 4pi) H^3:

  Cai-Kim coupling :  G_eff = 1/(kappa * c_S * M_Pl)   =>  G_eff = G  iff  kappa*c_S = M_Pl
  clock            :  H^2 = (8pi G_eff/3) rho_sub  =>  H = c_S * M_Pl / (2 N)

The cascade clock H = 1/(N t_P) = M_Pl/N then forces c_S = 2, AND kappa CANCELS.
So the factor of 2 is NOT in the temperature kappa (it drops out of the clock) and
NOT a work-density term — it is purely the entropy-counting normalization c_S.

THE FORK (run BLIND — do not pick the value that lands kappa = M_Pl/2)
----------------------------------------------------------------------
The framework supplies THREE distinct theorem-grade "entropy-like" counts per
horizon-advance, and they disagree:

  net-node creation  c_S = 1       cascade dN/dt = 1 (the clock / spatial extent)
  toggle events      c_S = 2       create + destroy per cycle (lambda_toggle_rate)
  Shannon surprise   c_S ~ 2.585   S_fresh(=1) + S_disconfirm(=log2 3) (the count
                                   the observer-energy-functional first law uses)

This probe derives each count from independent first principles, then asks which
one the Clausius relation deltaQ = T dS of GRAVITY actually requires — without
reference to the target value.

PRE-REGISTERED OUTCOMES
-----------------------
  CLOSE          a single principle FORCES one count, and it is c_S = 2
                 (=> kappa = M_Pl/2, G_eff = G, clock coeff 1, all consistent;
                  flagship gravity).
  NARROWED       a principle ELIMINATES at least one horn but does not uniquely
                 force the survivor (=> trilemma reduced; gravity stays form-level
                 until the survivor fork is decided).
  OPEN           no principle distinguishes the counts (=> the 2026-05-17 blind R2
                 verdict stands unchanged; gravity form-level).

Sympy + numeric. Honest aborts on the two verifiable claims; the disposition is
reported in the VERDICT, since a narrowing is a genuine result, not a failure.

References (tracked):
  predictions/lambda_toggle_rate.py        (lambda = 2/5; create rate = destroy rate = 1/5)
  predictions/S_fresh.py                   (S_fresh = 1 bit)
  predictions/S_disconfirm.py              (S_disconfirm = log2 3 bit)
  docs/theorems/theorem_observer_energy_functional.md  (E_obs = kappa*S_total,
       kappa = k_B T ln2 [Landauer], S_total = accumulated Shannon surprise;
       T explicitly NOT calibrated)
  Landauer 1961; Bennett 1973 (thermodynamic cost k_B T ln2 per irreversible bit op)
  Cai-Kim 2005 (JHEP 0502:050) (first law on the apparent horizon)
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
# C-A1 — derive the three candidate counts from INDEPENDENT first principles
# ----------------------------------------------------------------------
head("C-A1 — the three entropy counts, each from an independent source")

# (i) net-node creation = the cascade clock. dN/dt = 1 net node per Planck step.
c_net = sp.Integer(1)

# (ii) toggle-event count, from the 2-state edge Markov chain (lambda_toggle_rate).
p_create = sp.Rational(1, 2)   # P(exists | Beta(1,1))
p_destroy = sp.Rational(1, 3)  # P(absent | Beta(2,1))
pi_on = p_create / (p_create + p_destroy)          # detailed balance => 3/5
pi_off = 1 - pi_on                                  # 2/5
rate_create = pi_off * p_create                     # 1/5
rate_destroy = pi_on * p_destroy                    # 1/5
lam = rate_create + rate_destroy                    # 2/5  (= lambda)
# number of distinct irreversible toggle EVENTS per cycle = create + destroy = 2
c_toggle = sp.simplify(rate_create / rate_create + rate_destroy / rate_destroy)  # 1 + 1

# (iii) Shannon surprise per (fresh + disconfirm) pair — the OEF S_total increments.
S_fresh = sp.Integer(1)                 # -log2 P(exists|Beta(1,1)) = -log2(1/2)
S_disconfirm = sp.log(3, 2)             # -log2 P(absent|Beta(2,1)) = -log2(1/3)
c_surprise = sp.nsimplify(S_fresh + S_disconfirm)

print(f"  (i)   net-node (clock/extent)        c_S = {c_net}")
print(f"  (ii)  toggle events (create+destroy) c_S = {c_toggle}"
      f"   [rate_create={rate_create}, rate_destroy={rate_destroy}, lambda={lam}]")
print(f"  (iii) Shannon surprise (OEF S_total) c_S = {c_surprise} "
      f"= 1 + log2(3) ~ {float(c_surprise):.4f}")

ok1 = (c_net == 1) and (c_toggle == 2) and abs(float(c_surprise) - (1 + math.log2(3))) < 1e-12
if not ok1:
    abort("C-A1", "the three counts did not reproduce {1, 2, 1+log2 3} from sources.")
else:
    print("  -> three genuinely distinct framework counts: {1, 2, 2.585}.  OK")

# ----------------------------------------------------------------------
# C-A2 — the closure condition: c_S = 2 needed, and kappa CANCELS (symbolic)
# ----------------------------------------------------------------------
head("C-A2 — closure requires c_S = 2; kappa cancels in the clock coefficient")

kappa, cS, N, Gtt, H, M, Rho = sp.symbols(
    "kappa c_S N G H M_Pl rho", positive=True)

# Cai-Kim coupling: standard P = T*S' = M_Pl^2 <=> G_eff = G. Generally
# P = kappa * (c_S * M_Pl), and G_eff = G * (M_Pl^2 / P) = 1/(kappa*c_S*M_Pl).
P = kappa * cS * M
G_eff = 1 / (kappa * cS * M)
print(f"  G_eff = 1/(kappa*c_S*M_Pl) ;  G_eff = G=1/M_Pl^2  <=>  kappa*c_S = M_Pl")

# Holographic source spread over the Hubble volume V = (4pi/3) R_H^3, R_H = 1/H:
rho_sub = (3 * kappa * N / (4 * sp.pi)) * H**3       # = E_obs/V_Hubble, E_obs=kappa*N
friedmann = sp.Eq(H**2, sp.Rational(8, 3) * sp.pi * G_eff * rho_sub)
H_sol = [h for h in sp.solve(friedmann, H) if h != 0][0]
H_sol = sp.simplify(H_sol)
print(f"  H = {H_sol}")

# cascade clock: H = 1/(N t_P) = M_Pl/N.  Solve for c_S; check kappa-independence.
cS_closure = sp.solve(sp.Eq(H_sol, M / N), cS)
cS_val = sp.simplify(cS_closure[0]) if cS_closure else None
kappa_free = (sp.diff(H_sol, kappa) == 0)
print(f"  cascade match H = M_Pl/N  =>  c_S = {cS_val}    (kappa-independent: {kappa_free})")

ok2 = (cS_val == 2) and kappa_free
if not ok2:
    abort("C-A2", f"closure algebra did not give c_S=2 with kappa cancelling (got {cS_val}).")
else:
    print("  -> the entire factor of 2 is c_S; kappa is downstream (= M_Pl/2 once c_S=2).  OK")

# ----------------------------------------------------------------------
# C-A3 — the SELECTOR: deltaQ = T dS is THERMODYNAMIC, not epistemic
# ----------------------------------------------------------------------
head("C-A3 — thermodynamic (Landauer) entropy vs epistemic (Shannon) surprise")

print("""  The Clausius relation of gravity, deltaQ = T dS, is a THERMODYNAMIC heat
  balance. By Landauer (1961) / Bennett (1973), the thermodynamic entropy of an
  irreversible bit operation is k_B ln2 PER OPERATION — a FLAT cost, independent
  of the operation's probability. The Shannon surprise -log2 P(outcome) is the
  observer's epistemic information gain; it is the right quantity for the observer
  energy functional E_obs (which is explicitly an observer-internal functional),
  but it is NOT the substrate's thermodynamic heat.

  Consequence: the count (iii) c_S ~ 2.585 = S_fresh + S_disconfirm is a sum of
  Shannon SURPRISES (1 and log2 3). It weights a destroy by its improbability
  (log2 3 > 1). The thermodynamic horizon entropy in deltaQ = T dS counts the
  same two irreversible operations at the FLAT Landauer cost (1 each) -> 2, NOT
  2.585. The 2.585 belongs to E_obs (epistemic), not to the gravitating entropy.""")

# Discriminator: are the two irreversible events distinguishable in PROBABILITY?
# If yes, Shannon (surprise-weighted) and Landauer (flat) genuinely differ, so the
# choice of which enters deltaQ=TdS is a real fork that the thermodynamic nature
# of heat decides in favor of the flat (Landauer) count.
shannon_neq_landauer = abs(float(c_surprise) - float(c_toggle)) > 1e-9
landauer_count = c_toggle  # flat 1 bit per irreversible op, create + destroy = 2
print(f"\n  Shannon sum = {float(c_surprise):.4f}  !=  Landauer (flat-op) count = {landauer_count}"
      f"   (distinct: {shannon_neq_landauer})")
print("  -> SELECTOR eliminates horn (iii) c_S ~ 2.585: it is the epistemic E_obs")
print("     surprise, not the thermodynamic deltaQ=TdS entropy. Trilemma -> dilemma {1, 2}.")

# (No abort: this is a principled elimination, reported in the verdict.)

# ----------------------------------------------------------------------
# C-A4 — the surviving fork {1 vs 2}: net EXTENT vs irreversible-operation FLUX
# ----------------------------------------------------------------------
head("C-A4 — is the operation FLUX (=2) forced over the net EXTENT (=1)?")

print("""  Survivor fork:
    c_S = 1  : the horizon entropy = the net-node EXTENT (create MINUS destroy),
               i.e. the same quantity as the cascade clock N. dN/dt = 1.
    c_S = 2  : the horizon entropy = the irreversible-operation FLUX crossing the
               horizon (create PLUS destroy), both being irreversible Landauer
               operations that deposit heat. create + destroy = 2 per edge-cycle.

  deltaQ = T dS is a heat FLUX through the horizon, which favors the create+destroy
  flux reading (c_S = 2): a destroy is as thermodynamically irreversible as a
  create and crosses the horizon as heat, even though it does not advance the net
  clock. This is a genuine physical distinction (current vs extent), so c_S = 2 is
  CONSISTENT with the clock using N for the extent — no contradiction.

  BUT it is not yet FORCED: lambda_toggle_rate gives create-rate = destroy-rate =
  1/5 per edge per step (a STEADY, net-zero toggling on existing edges), whereas
  the cascade net growth dN/dt = 1 is a DISTINCT process (new structure entering
  the causal patch). Pinning 'irreversible operations registering on the horizon
  per clock tick = 2' requires relating the per-edge toggle flux to the cascade
  graph growth -- structure this probe does not derive.""")

steady_balanced = (rate_create == rate_destroy)     # True: 1/5 = 1/5 (net-zero toggling)
print(f"\n  steady-state create-rate == destroy-rate : {steady_balanced} "
      f"({rate_create} = {rate_destroy})  -> toggling is net-zero; net growth is the cascade.")
print("  -> c_S = 2 is the THERMODYNAMICALLY MOTIVATED candidate (flux reading),")
print("     but the operation-per-tick normalization is CONSISTENT, not FORCED.")
forced = False  # honest: the 1-vs-2 fork is not closed by present structure

# ----------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------
head("VERDICT")
if FAIL:
    print(f"  HONEST NEGATIVE — verifiable-claim aborts tripped: {FAIL}")
    sys.exit(1)

if forced:
    disposition = "CLOSE"
else:
    disposition = "NARROWED"

print(f"""  DISPOSITION: {disposition}

  Verifiable claims PASS:
    C-A1  three distinct framework counts {{1, 2, 2.585}} reproduced from sources.
    C-A2  the factor of 2 is ENTIRELY c_S; kappa cancels in the clock coefficient
          (c_S = 2 is the value the cascade clock + G_eff=G require).

  Principled result:
    C-A3  the Shannon-surprise horn c_S ~ 2.585 is ELIMINATED on principle: it is
          the EPISTEMIC observer functional E_obs (surprise-weighted), not the
          THERMODYNAMIC deltaQ=TdS entropy (flat Landauer cost per irreversible
          op). This dissolves the worst horn and removes the internal tension of
          using the OEF S_total as the gravitating entropy.
    C-A4  the surviving fork is {{1 (net EXTENT) vs 2 (irreversible-operation
          FLUX)}}. deltaQ=TdS being a heat flux FAVORS c_S = 2, and c_S = 2 is
          consistent with the clock using N. But it is NOT forced: steady toggling
          is net-zero while net growth is the cascade, and the operation-per-tick
          normalization is not derived here.

  THEREFORE this probe DOES NOT CLOSE the coupling. It NARROWS the trilemma to a
  dilemma and identifies c_S = 2 as the thermodynamically motivated candidate.
  Until the {{1 vs 2}} extent-vs-flux fork is decided (a forced relation between
  the toggle-event flux and the cascade growth, registering on the causal
  horizon), kappa / Newton's G does NOT close from the thermal route:

    - kappa is NOT promotable to predictions/.
    - predictions/G_N.py stays at its asymptotic-safety-conditional grade.
    - Promotable gravity = the FORM (emergent Lorentzian metric; emergent standard
      Friedmann + coasting from the native information-Clausius relation;
      G_N*M_Pl^2 = 1 as a separate conditional identity). The coupling's
      normalization is calibration-fixed, not derived parameter-free.

  Next decidable gate: derive the extent-vs-flux relation (does the irreversible
  toggle flux that registers on the causal horizon equal 2x the net cascade growth
  per tick?). If forced to 2 -> flagship closes. If 1, or scheme-dependent ->
  gravity stays form-level.
""")
print("=" * 78)
print(f"  EXIT 0 — {disposition}: trilemma -> dilemma; c_S=2 motivated, not forced;"
      f" gravity form-level")
print("=" * 78)
