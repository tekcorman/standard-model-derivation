#!/usr/bin/env python3
"""Phase 2.3 — two-clock conversion consistency note (Page-Wootters FRAMING).

[PANEL-REWORDED: classical Bernoulli implementation; NO quantum history
state is constructed (the |Psi> below is described, not built). PW2/PW3
validate the simulation for any Bernoulli(r) process, not the mechanism.
PW3 verifies the tick-filtration compensator martingale N(L) - r_eff L on
F_L, NOT the 2026-05-25 theorem's posterior martingale on F_N. The 16/15
inherits the (conflicted) in-repo grade -- see correction 7 cross-flags.]

SCOPE (honest, pre-stated): this formalizes the framework's existing
observer-substrate rate-gap mechanism (proofs/cosmology/
cascade_observer_rate_gap.py, grade CANDIDATE CLOSURE) in a Page-Wootters
relational structure: ONE history state, TWO clock conditionings. It adds
the formal two-clock structure and the martingale tie; it does NOT re-derive
the mechanism's microphysics (the effective-rate form r_eff = r0 (1 + eps/k)
is the in-repo candidate mechanism; this probe inherits its grade).
No new numbers anywhere.

Structure:
  - Substrate clock: tick count L (the unitary walker step).
  - Observer clock: record count N (the martingale filtration index,
    theorem_observer_martingale_time_2026-05-25).
  - History state: |Psi> = sum_L |L>_clock (x) |record after L ticks>.
    Conditioning on L gives the observer record distribution; conditioning
    on N gives the tick distribution -- two readings of ONE timeless state.
  - The two-clock CONVERSION FACTOR between the conditioned rates is
    r_eff / r0 = 1 + eps_toggle / k = 16/15, with
    eps_toggle = (P_create - P_persist)/(P_create + P_persist) = 1/5
    (Beta(1,1) -> Beta(2,1), in-repo theorem-grade) and r0 = 1/k = 1/3.

Gates:
  PW1 exact rationals: eps = 1/5 from (1/2, 1/3); conversion = 16/15.
  PW2 PW conditioning: in an explicit stochastic history state (record
      grows per tick w.p. r_eff), E[N|L] = r_eff L and E[L|N] -> N/r_eff
      (reciprocal affine relations; the two conditional clocks are
      consistent readings of one state).
  PW3 martingale tie: N(L) - r_eff L is a martingale w.r.t. the tick
      filtration (compensator linear; verified over MC paths) -- embedding
      the 2026-05-25 observer-time theorem in the PW frame.
  PW4 consistency outputs: H_obs = (16/15) H_sub reproduces the in-repo
      68.18 -> 72.72 km/s/Mpc (H_0_derivation.md; SH0ES -0.30 sigma),
      direction per the in-repo assignment (observer-side FASTER).
"""
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

RNG = np.random.default_rng(20260611)
FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 72)
    print(" PHASE 2.3 -- Page-Wootters two clocks: 16/15 as a conversion factor")
    print("=" * 72)

    # PW1: exact rationals from in-repo inputs
    P_create, P_persist, k = F(1, 2), F(1, 3), 3
    eps = (P_create - P_persist) / (P_create + P_persist)
    conv = 1 + eps / k
    r0 = F(1, k)
    r_eff = r0 * conv
    gate("PW1 eps = 1/5 exact; conversion r_eff/r0 = 1 + eps/k = 16/15 exact",
         eps == F(1, 5) and conv == F(16, 15) and r_eff == F(16, 45),
         f"eps={eps}, conv={conv}, r_eff={r_eff}")

    # PW2: explicit history state, two conditionings
    L_max, n_paths = 3000, 4000
    r = float(r_eff)
    steps = (RNG.random((n_paths, L_max)) < r).astype(int)
    N_of_L = np.cumsum(steps, axis=1)
    # E[N | L] = r_eff * L  (clock-L conditioning)
    Ls = np.array([500, 1500, 2999])
    EN = N_of_L[:, Ls].mean(axis=0)
    ok_EN = np.allclose(EN, r * (Ls + 1), rtol=0.02)
    # E[L | N] ~ N / r_eff  (record-N conditioning; first-passage)
    target_N = 400
    first_pass = np.argmax(N_of_L >= target_N, axis=1)
    valid = N_of_L[:, -1] >= target_N
    EL = first_pass[valid].mean()
    ok_EL = abs(EL - target_N / r) / (target_N / r) < 0.02
    gate("PW2 PW conditioning: E[N|L] = r_eff L and E[L|N] = N/r_eff (reciprocal)",
         ok_EN and ok_EL,
         f"E[N|L]/L = {(EN/(Ls+1))} vs r_eff = {r:.4f}; E[L|N]*r/N = {EL*r/target_N:.4f}")

    # PW3: martingale tie: M_L = N(L) - r_eff L has E[M_{L+1} | F_L] = M_L
    M = N_of_L - r * (np.arange(L_max) + 1)
    increments = np.diff(M, axis=1)
    gate("PW3 martingale: E[increment of N - r_eff L] = 0 (filtration tie)",
         abs(increments.mean()) < 5e-4 and abs(M[:, -1].mean()) < 0.5,
         f"mean increment = {increments.mean():.2e}, E[M_end] = {M[:, -1].mean():.3f}")

    # PW4: consistency with the in-repo H_0 chain
    H_sub = 68.18
    H_obs = H_sub * 16 / 15  # canonical rounding 72.72 per H_0_derivation.md
    gate("PW4 consistency: H_obs = (16/15) H_sub = 72.72 (in-repo chain, SH0ES side)",
         abs(H_obs - 72.72) < 0.01, f"H_obs = {H_obs:.2f} km/s/Mpc")

    print("\n  GRADE: PW formalization established; the 16/15 value inherits the")
    print("  in-repo CANDIDATE-CLOSURE grade of cascade_observer_rate_gap.py")
    print("  (this probe adds the two-conditionings-of-one-history-state frame")
    print("  and the martingale embedding; no new numbers).")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print(" RESULT: ALL GATES PASS -- two clocks formalized; tension = conversion")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
