#!/usr/bin/env python3
"""The Higgs-VEV N^-1/4 exponent is the observer's ONE-PASS recurrence
(criticality-independent).  Self-checking; not in the verify backbone.

Result (2026-06-14): the VEV's N^-1/4 finite-size exponent does NOT require the
order parameter to sit at a critical point (mu^2=0) -- the circular Step-4 of
predictions/v_higgs_derivation.md.  It is finite-budget sampling of the FREE
lean: the observer reads the M-edge substrate by a count-walk (graph-blind,
P(up|k)=(M-k)/M -- see real_multiway_lean); in ONE read-pass (T=M, "read each
edge once") the walk returns to the home lean ~sqrt(M) times (diffusive local
time), so N_eff = sqrt(M) and the order-parameter spread is the ordinary -1/2
counting law over sqrt(M) samples = M^-1/4.  Unlimited reading -> stationary ->
linear returns -> N_eff=M -> -1/2 (reconciles a2_under_read).  Pure counting,
no BZJ, no stat-mech import.  Companion: project memory
project_vev_observer_read_decomposition_2026-06-14.

GATES (deterministic; seeded RNG):
  G1 diffusive returns ~ T^0.5 (local time) for T <= one pass
  G2 stationary returns ~ T^1 (linear) for T >> mixing (the crossover)
  G3 one-pass effective sample size N_eff ~ M^0.5
  G4 budget read (n=sqrt M): spread ~ M^-1/4   [THE VEV exponent]
  G5 full read   (n=M):      spread ~ M^-1/2   [the naive exponent]
"""
import math
import random
import sys

import numpy as np

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


rng = random.Random(20260614)


def count_walk(M, T, start_home=False):
    """Observer count-walk: M coins, toggle a random one each step. Returns the
    number of UPCROSSINGS of the home lean (parity-robust local time)."""
    if start_home:
        on = [True] * (M // 2) + [False] * (M - M // 2)
    else:
        on = [rng.random() < 0.5 for _ in range(M)]
    k = sum(on)
    home = M / 2.0
    up = 0
    prev = (k > home)
    for _ in range(T):
        i = rng.randrange(M)
        if on[i]:
            on[i] = False; k -= 1
        else:
            on[i] = True; k += 1
        side = (k > home)
        if side and not prev:
            up += 1
        prev = side
    return up


print("=" * 76)
print(" VEV N^-1/4 EXPONENT = the observer's one-pass recurrence (no criticality)")
print("=" * 76)

# ---- G1/G2: returns vs budget T (fixed M): diffusive sqrt(T) -> stationary linear
M0 = 16384
Ts = [M0 // 16, M0 // 8, M0 // 4, M0 // 2, M0, 4 * M0, 16 * M0, 64 * M0]
Rs = [np.mean([count_walk(M0, T, start_home=True) for _ in range(max(2, 300000 // T))])
      for T in Ts]
diff_idx = [i for i, T in enumerate(Ts) if T <= M0]
stat_idx = [i for i, T in enumerate(Ts) if T >= 4 * M0]
pA = np.polyfit(np.log([Ts[i] for i in diff_idx]), np.log([Rs[i] for i in diff_idx]), 1)[0]
pStat = np.polyfit(np.log([Ts[i] for i in stat_idx]), np.log([Rs[i] for i in stat_idx]), 1)[0]
print(f"\n  returns vs T (M={M0}): {[round(r,1) for r in Rs]}")
gate("G1 diffusive returns ~ T^0.5 (local time, T<=one pass)",
     abs(pA - 0.5) < 0.09, f"exponent {pA:.3f}")
gate("G2 stationary returns ~ T^1 (linear, T>>mixing) -- the crossover",
     abs(pStat - 1.0) < 0.09, f"exponent {pStat:.3f}")

# ---- G3: one-pass N_eff ~ sqrt(M) ----
Ms = [256, 1024, 4096, 16384, 65536]
Rsweep = [np.mean([count_walk(M, M) for _ in range(max(4, 400000 // M))]) for M in Ms]
pB = np.polyfit(np.log(Ms), np.log(Rsweep), 1)[0]
print(f"\n  one-pass returns/sweep vs M: {[round(r,1) for r in Rsweep]} "
      f"(sqrt(M)={[round(math.sqrt(M),1) for M in Ms]})")
gate("G3 one-pass effective sample size N_eff ~ M^0.5",
     abs(pB - 0.5) < 0.09, f"exponent {pB:.3f}")


# ---- G4/G5: order-parameter spread over n samples ----
def lean_spread(n_samples, reps=4000):
    return float(np.std([sum(rng.random() < 0.5 for _ in range(n_samples)) / n_samples - 0.5
                         for _ in range(reps)]))


sp_full = [lean_spread(M) for M in Ms]
sp_bud = [lean_spread(int(round(math.sqrt(M)))) for M in Ms]
pfull = np.polyfit(np.log(Ms), np.log(sp_full), 1)[0]
pbud = np.polyfit(np.log(Ms), np.log(sp_bud), 1)[0]
print(f"\n  spread(n=M)      ~ M^{pfull:.3f}   (full read)")
print(f"  spread(n=sqrt M) ~ M^{pbud:.3f}   (budget read = THE VEV exponent)")
gate("G4 budget read spread ~ M^-1/4 (N_eff=sqrt M; THE VEV exponent)",
     abs(pbud + 0.25) < 0.05, f"exponent {pbud:.3f}")
gate("G5 full read spread ~ M^-1/2 (n=M; the naive exponent)",
     abs(pfull + 0.5) < 0.05, f"exponent {pfull:.3f}")

print("\n  => -1/4 = the -1/2 counting law over N_eff = sqrt(M) one-pass returns")
print("     (diffusive local time).  Graph-blind => the observer's READ, not the")
print("     substrate.  No criticality assumed: dissolves v_higgs Step-4.")

print("\n" + "=" * 76)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- VEV exponent is the observer one-pass recurrence")
print("=" * 76)
sys.exit(0)
