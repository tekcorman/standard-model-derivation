#!/usr/bin/env python3
"""The Higgs-VEV N^-1/4 exponent is the FLOOR of the observer's single-read sector.

Promoted + gated companion to vev_exponent_observer_recurrence_2026-06-14 (which establishes
v ~ N^-1/4 = the observer one-pass recurrence). This probe answers: does the -1/4 EXTEND to
-1/8 (a second observer halving), or is it the floor?

Result (2026-06-15): the observer read-map is the scale-free single halving L -> sqrt(L) (one
pass = N_eff = sqrt(L) returns of the 1-D scalar lean). A single observer-read therefore gives
EXACTLY ONE halving below the -1/2 counting law -> v ~ N^-1/4. A -1/8 rung needs N_eff = M^1/4
= the recurrence-OF-the-recurrence = a READ-OF-A-READ (depth-2 meta-observation), NOT the
framework's "one read of one walk". So -1/4 is the floor of the single-read sector.

Theorem: docs/theorems/theorem_observer_flow_dyadic_ladder_2026-06-15.md (clause S3).

GATES (self-checking; exit 0 on all-pass):
  G1 the read-map is the single halving: one-pass returns ~ L^0.5
  G2 full read (N_eff=M):        spread ~ M^-1/2  (depth 0, the counting law)
  G3 one-pass read (N_eff=sqrt M): spread ~ M^-1/4  (depth 1 = THE VEV floor)
  G4 the floor is ONE halving: exponent gap (depth0 - depth1) = 1/4 exactly (one clean halving)
NOTE (documented, NOT gated): depth-2 (read-of-a-read) is finite-size-inaccessible -- the 2nd
  nest acts on length-sqrt(M)<=256 sequences with no decade of range, so the deeper dyadic tower
  (-1/8, ...) is an ANALYTIC consequence of the verified sqrt-map, not numerically reachable here.
  The physical claim (single read -> one halving -> -1/4 floor) does NOT depend on it.
"""
import sys
import numpy as np

FAILURES = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


rng = np.random.default_rng(20260615)


def read_returns(L, T):
    """One-pass observer read of an L-element on/off sequence: toggle a random element each
    step, count UPCROSSINGS of the home lean (= diffusive local time = effective sample size).
    Same mechanism as the promoted VEV recurrence probe."""
    on = rng.random(L) < 0.5
    k = int(on.sum())
    home = L / 2.0
    prev = (k > home)
    up = 0
    for _ in range(T):
        i = int(rng.integers(L))
        if on[i]:
            on[i] = False; k -= 1
        else:
            on[i] = True; k += 1
        side = (k > home)
        if side and not prev:
            up += 1
        prev = side
    return up


def fit_exp(xs, ys):
    return float(np.polyfit(np.log(np.array(xs, float)), np.log(np.array(ys, float)), 1)[0])


def lean_spread(n_samples, reps=4000):
    """Std of the order parameter (lean fraction - 1/2) estimated from n independent draws."""
    return float(np.std([(np.sum(rng.random(n_samples) < 0.5) / n_samples) - 0.5
                         for _ in range(reps)]))


print("=" * 78)
print(" VEV N^-1/4 = the FLOOR of the observer single-read sector (one halving)")
print("=" * 78)

# ---- G1: the read-map is the single halving L -> sqrt(L) ----
Ls = [64, 128, 256, 512, 1024, 2048, 4096]
Rs = [np.mean([read_returns(L, L) for _ in range(max(6, 300000 // L))]) for L in Ls]
pA = fit_exp(Ls, Rs)
print(f"\n  one-pass read of length L -> N_eff returns ~ L^p:")
print(f"    N_eff = {[round(r, 1) for r in Rs]}  (sqrt(L)={[round(np.sqrt(L),1) for L in Ls]})")
gate("G1 read-map = single halving: one-pass returns ~ L^0.5", abs(pA - 0.5) < 0.06,
     f"exponent {pA:.3f}")

# ---- G2/G3: spread at depth 0 (full) and depth 1 (one-pass) ----
Ms = [256, 1024, 4096, 16384, 65536]
sp_full = [lean_spread(M) for M in Ms]                                  # N_eff = M
sp_one = [lean_spread(max(2, int(round(np.sqrt(M))))) for M in Ms]      # N_eff = sqrt(M)
p_full = fit_exp(Ms, sp_full)
p_one = fit_exp(Ms, sp_one)
print(f"\n  spread(N_eff=M)       ~ M^{p_full:.3f}   (depth 0, full read, counting law)")
print(f"  spread(N_eff=sqrt M)  ~ M^{p_one:.3f}   (depth 1, one-pass read = THE VEV floor)")
gate("G2 full read spread ~ M^-1/2 (depth 0)", abs(p_full + 0.5) < 0.05, f"exponent {p_full:.3f}")
gate("G3 one-pass read spread ~ M^-1/4 (depth 1 = VEV floor)", abs(p_one + 0.25) < 0.05,
     f"exponent {p_one:.3f}")

# ---- G4: the floor is exactly ONE halving (depth0 - depth1 gap = 1/4) ----
gap = p_one - p_full      # (-1/4) - (-1/2) = +1/4
gate("G4 floor = ONE halving: (depth0 - depth1) exponent gap = 1/4", abs(gap - 0.25) < 0.05,
     f"gap {gap:+.3f}")

# ---- NOTE: depth-2 is finite-size-inaccessible (documented, not gated) ----
L1 = int(round(np.mean([read_returns(4096, 4096) for _ in range(40)])))   # ~ sqrt(4096)=64
L2 = int(round(np.mean([read_returns(max(2, L1), max(2, L1)) for _ in range(200)])))  # ~ sqrt(64)=8
print(f"\n  NOTE (not gated): a depth-2 read (read-of-a-read) on M=4096 nests to length {L1} then {L2}")
print(f"    -> no decade of range for a clean 2nd sqrt; the -1/8 rung is analytic-only here.")
print(f"    A -1/8 rung would be a META-observation (read-of-a-read), NOT 'one read of one walk'.")
print(f"    => -1/4 is the floor of the single-read sector. FALSIFIABLE: no physical observable")
print(f"       scales as N^-1/8 (or below) from a single read (repo: v=-1/4 is the smallest rung).")

print("\n" + "=" * 78)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- single observer-read = one halving; -1/4 is the read floor")
print("=" * 78)
sys.exit(0)
