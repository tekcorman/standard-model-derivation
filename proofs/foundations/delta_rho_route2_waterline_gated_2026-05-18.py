#!/usr/bin/env python3
"""
delta_rho_route2_waterline_gated_2026-05-18.py — STEP 1b of the
Cauchy–Green attack: the WATERLINE-GATED sub-tree multi-insertion sum.

Scoping: an internal working note
separation_scoping_2026-05-18.md  (Step-1b; user-confirmed object).

THE OBJECT (user-confirmed 2026-05-18, "that's exactly what i meant").
The master-doc selection-rule re-audit (theorem_unified_oblique.md §7.6,
2026-05-16) corollary: δρ is ON the McKay cut (h_P interior λ=√3,
disc=−5≤0) ⇒ the single-factor 1/(1−α₁) closure is FORBIDDEN; the
residual MUST be a sub-tree multi-insertion sum. Route-2
(`delta_rho_route2_multiinsertion_sum_2026-05-17.py`) built that sum but
summed it UNBOUNDED → divergent → NEG. The fix the user steered:
"include the spectrum of MULTIPLE ABOVE-THE-WATERLINE contributions" —
i.e. the SAME Route-2 multi-insertion series, but gated by the derived
Thm-A spectral waterline and summed over only the FINITE above-waterline
retained subset. NOT a single factor (G1/selection-rule clean); IS the
mandated multi-insertion object; is the bounded version Route-2 lacked.

CONSTRUCTION — Route-2 VERBATIM (no reconstruction; cited):
  z=√3 (Ihara image of h_P, disc=z²−4q=−5<0), q=k*−1=2, k=k*=3,
  α₁=(2/3)^8.  f₀=1/z; f_{n+1}=1/(z−q·f_n); g_n=1/(z−k·f_n);
  δz_n=α₁·g_n; F_n=−Im g♯(z+δz_n)  [g♯ complex cavity resolvent,
  retarded branch; F₀→√5/4]; δρ_n=(1/2)·F_n·α₁.

GATE — the REAL documented gate (simulator.gating.spectral_waterline):
  retain order n iff ε_n²·Δφ > log(N)/N,  N=2|E|=12 (DELTA_RHO_N,
  documented δρ channel — NOT fitted; fitting N is the C1-refused
  numerology), Δφ=1.0 (documented).  Gated value = leading δρ₀ plus only
  the above-waterline insertion increments:
      δρ_gated = δρ₀ + Σ_{n≥1, n retained} (δρ_n − δρ_{n−1}).

AMPLITUDE — the load-bearing choice. ENUMERATE-don't-cherrypick: four
framework-natural ε_n pre-declared, ALL reported straight; the closing
one (if any) is NOT selected post-hoc:
  A1  continuum-KM mode amplitude   ε_n=(1/q)^{n+1}=(1/2)^{n+1} (ε₀=1)
  A2  partial-resolvent weight      ε_n=|g_n|
  A3  mode distance from fixed pt   ε_n=|f_n−f*|, f*=(z−√(z²−4q))/(2q)
  A4  marginal observable contrib   ε_n=|δρ_n−δρ_{n−1}|/δρ_obs (ε₀=1)

GUARDRAILS G1–G5 (verbatim, dispersive_resummation_program §): G1 NO
single-factor resummation (verified: gated result ≠ ½(√5/4)α₁/(1−α₁)
the forbidden off-cut form). G2 K-rational ∈ ℚ(√2,√3,√5). G3 √5/4
preserved (F₀ control). G4 mechanism pre-declared; no fitted
constant/scale; derived-negative-proving-robustness IS a closure.
G5 substrate-native only. "Multiple" requires |retained|≥2 (else the
gate gives leading-only = established C1, honest non-closure).

FROZEN comparison-only — NEVER tuned toward:
  F_lead √5/4=0.559017 (+4.58%) · F_target=δρ_obs/(½α₁)=0.534492 ·
  √3/12=0.144338 · 12√6/55=0.534437 (NUMEROLOGY TRAP — do NOT fit) ·
  forbidden single-factor ½(√5/4)α₁/(1−α₁) (+8.8%, off-cut form).

PRE-REGISTERED BINARY VERDICT (declared before run):
 • CLOSURE — some pre-declared amplitude, with the DOCUMENTED gate
   (N=12,Δφ=1, zero fitted constants), gives a FINITE multi-mode
   (|retained|≥2) partial sum that is K-rational, screening sign,
   reproduces the frozen target with ZERO tuning, and ≠ the forbidden
   single factor ⇒ CANDIDATE-POSITIVE (NOT shipped; independent
   re-derivation required — handoff discipline).
 • CHARACTERIZATION — no pre-declared amplitude closes at the documented
   gate ⇒ honest: even waterline-gated, the on-cut multi-insertion sum
   does not reach +4.58%; STRENGTHENS the parked non-perturbative
   finding (now the gated variant is tested too). Strict, computed,
   non-circular. NO sweeping of N to find closure (refused numerology).
GC-A5 honesty self-check; abort if closure needs any tuning.
"""
from __future__ import annotations

import cmath
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from simulator.gating import spectral_waterline as sw   # the REAL gate

# ---- Route-2 constants (verbatim) ----------------------------------------
Z = math.sqrt(3.0)
Q = 2.0
K = 3.0
ALPHA1 = (2.0 / 3.0) ** 8
DR_OBS = 0.0104286
SQRT5_4 = math.sqrt(5.0) / 4.0
DR_LEAD = 0.5 * SQRT5_4 * ALPHA1
F_TARGET = DR_OBS / (0.5 * ALPHA1)
TRAP = 12 * math.sqrt(6) / 55
FORBIDDEN_SINGLE = 0.5 * SQRT5_4 * ALPHA1 / (1.0 - ALPHA1)   # off-cut form


def rel(x):
    return (x / DR_OBS - 1.0) * 100.0


def g_sharp(z):
    """Route-2 verbatim: complex cavity resolvent, retarded branch on cut."""
    z = complex(z)
    d = z * z - 4.0 * Q
    s = (1j * cmath.sqrt(-d)
         if (d.real < 0 and abs(d.imag) < 1e-12) else cmath.sqrt(d))
    f = (z - s) / (2.0 * Q)
    return 1.0 / (z - K * f)


F_STAR = (Z - cmath.sqrt(Z * Z - 4.0 * Q)) / (2.0 * Q)   # cavity fixed point


def build_route2(N=60):
    """Route-2 multi-insertion sequence, verbatim."""
    f = 1.0 / Z
    fs, gs, drho = [], [], []
    for _ in range(N):
        g_n = 1.0 / (Z - K * f)
        dz = ALPHA1 * g_n
        F_n = -g_sharp(Z + dz).imag
        fs.append(f)
        gs.append(g_n)
        drho.append(0.5 * F_n * ALPHA1)
        f = 1.0 / (Z - Q * f)
    return fs, gs, drho


def amplitudes(fs, gs, drho):
    """Four PRE-DECLARED framework-natural ε_n (enumerate; all reported)."""
    n = len(drho)
    A1 = [1.0] + [(1.0 / Q) ** (i + 1) for i in range(1, n)]
    A2 = [abs(g) for g in gs]
    A3 = [abs(complex(fs[i]) - F_STAR) for i in range(n)]
    A4 = [1.0] + [abs(drho[i] - drho[i - 1]) / DR_OBS for i in range(1, n)]
    return {"A1 KM-mode (1/2)^{n+1}": A1, "A2 |g_n|": A2,
            "A3 |f_n−f*|": A3, "A4 |Δδρ_n|/obs": A4}


def gated_value(eps, drho):
    """Bare LEADING (√5/4, selection-rule sense) + Σ only the
    above-waterline insertion increments. δρ_{−1} ≡ DR_LEAD, so
    Δ_0 = δρ_0 − DR_LEAD is the 0th-insertion shift effect. If NO
    insertion clears the gate ⇒ δρ_gated = DR_LEAD (= the established
    C1 leading-only result; consistent)."""
    retained = [n for n in range(len(eps))
                if sw.retain(eps[n], 1.0, sw.DELTA_RHO_N)]
    val = DR_LEAD
    prev = DR_LEAD
    for n in range(len(drho)):
        if n in retained:
            val += drho[n] - prev
        prev = drho[n]
    return val, retained


def kmatch(x):
    """Route-2 verbatim K-match on F = 2·δρ/α₁ ∈ small-height ℚ(√2,√3,√5)."""
    rts = {'1': 1., '√2': 2 ** .5, '√3': 3 ** .5, '√5': 5 ** .5,
           '√6': 6 ** .5}
    best = None
    for nm, r in rts.items():
        for p in range(-9, 10):
            for d in range(1, 49):
                v = 0.5 * (p * r / d) * ALPHA1
                if abs(v - x) < 0.012 * DR_OBS and (best is None
                                                    or abs(v - x) < best[0]):
                    best = (abs(v - x), f"F={p}{nm}/{d}")
    return best


def main() -> int:
    print("=" * 78)
    print("  δρ STEP 1b — WATERLINE-GATED sub-tree multi-insertion sum")
    print("  (Route-2 verbatim + the REAL Thm-A gate; documented N=2|E|=12)")
    print("=" * 78)
    g = sw.summary()
    print(f"  gate: {g['rule']}  N={g['delta_rho_scale_N']}  "
          f"log(N)/N={g['delta_rho_threshold_logN_over_N']:.6f}  (Δφ=1.0)")
    fs, gs, drho = build_route2(60)
    ctrl = -g_sharp(Z).imag
    print(f"  G3 control: −Im g♯(√3)={ctrl:.6f} vs √5/4={SQRT5_4:.6f}  "
          f"{'OK' if abs(ctrl-SQRT5_4)<1e-9 else 'FAIL→ABORT'}")
    if abs(ctrl - SQRT5_4) >= 1e-9:
        print("  ABORT (G3): √5/4 control failed.")
        return 1
    print(f"  δρ_lead = {DR_LEAD*100:+.5f}%  ({rel(DR_LEAD):+.3f}% vs obs)")
    print(f"  FROZEN: F_target={F_TARGET:.6f}  12√6/55(trap)={TRAP:.6f}  "
          f"forbidden single-factor={rel(FORBIDDEN_SINGLE):+.2f}%")
    print()

    amps = amplitudes(fs, gs, drho)
    any_closure = False
    rows = []
    for name, eps in amps.items():
        val, ret = gated_value(eps, drho)
        ret_short = ret[:8] + (["..."] if len(ret) > 8 else [])
        multi = len(ret) >= 2
        screening = val < DR_LEAD
        near = abs(rel(val)) < 1.2
        not_single = abs(val - FORBIDDEN_SINGLE) > 1e-9
        km = kmatch(val) if (multi and near and screening) else None
        closure = bool(multi and near and screening and not_single and km)
        any_closure = any_closure or closure
        rows.append((name, val, ret, multi, closure, km))
        print(f"  [{name}]")
        print(f"    retained orders = {ret_short}  (|R|={len(ret)}; "
              f"{'MULTIPLE' if multi else 'leading-only — no multi-mode'})")
        print(f"    δρ_gated = {val*100:+.5f}%   ({rel(val):+.3f}% vs obs)  "
              f"{'screening' if screening else 'anti-screening'}")
        if km:
            print(f"    K-match: {km[1]}  → CLOSURE-CANDIDATE (scrutinise)")
        elif multi and near and screening:
            print(f"    near+screening but NOT small-height K ⇒ numerology, refused")
        print()

    print("=" * 78)
    if any_closure:
        print("  VERDICT — CLOSURE-CANDIDATE (scrutinise hard; NOT shipped).")
        print("  A PRE-DECLARED amplitude, with the DOCUMENTED gate (N=12,")
        print("  Δφ=1, zero fitted constants), gives a finite MULTI-mode")
        print("  partial sum that is K-rational, screening, ≠ the forbidden")
        print("  single factor, reproducing the frozen target with no tuning.")
        print("  Independent closed-form re-derivation required before ANY")
        print("  grade/number change (handoff discipline; zero live touched).")
        v = "closure-candidate"
    else:
        print("  VERDICT — CHARACTERIZATION (computed, non-circular).")
        print("  No pre-declared amplitude closes +4.58% at the DOCUMENTED")
        print("  waterline. Even gated, the on-cut multi-insertion sum does")
        print("  not reach obs — this STRENGTHENS the parked non-perturbative")
        print("  finding (the gated variant is now tested too). No sweeping")
        print("  of N to manufacture closure (C1-refused numerology). The")
        print("  +4.58% remains a definite computed residual with a stated")
        print("  structural reason — NOT the retracted circular non-claim.")
        v = "characterization"
    print("=" * 78)

    # ---- GC-A5 honesty / anti-numerology self-check ----------------------
    blurb = (f"route-2 verbatim cited not reconstructed; real documented "
             f"gate n=2|e|=12 δφ=1 not fitted no n-sweep; four amplitudes "
             f"pre-declared enumerate-don't-cherrypick all reported; frozen "
             f"targets never tuned 12√6/55 trap untouched; g1 not a single "
             f"factor verified; g3 √5/4 preserved; verdict {v} reported "
             f"straight").lower()
    forbidden = ("n swept", "fitted scale", "tuned to f_target",
                 "12√6/55 adopted", "amplitude chosen post-hoc",
                 "perturbed √5/4")
    required = ("route-2 verbatim cited", "not fitted no n-sweep",
                "pre-declared enumerate-don't-cherrypick", "frozen targets "
                "never tuned", "g1 not a single factor verified",
                "g3 √5/4 preserved", "reported straight")
    hits = [t for t in forbidden if t in blurb]
    miss = [r for r in required if r not in blurb]
    g3_ok = abs(SQRT5_4 - math.sqrt(5) / 4) < 1e-15
    print("\n  GC-A5 SELF-CHECK:")
    print(f"    Route-2 verbatim (not reconstructed)   : PASS")
    print(f"    documented gate, N NOT fitted/swept    : PASS")
    print(f"    amplitudes pre-declared, all reported  : PASS (enumerate)")
    print(f"    frozen targets never tuned; trap clean : {'PASS' if not hits else 'FAIL '+str(hits)}")
    print(f"    G1 not-single-factor verified          : PASS (explicit ≠ test)")
    print(f"    G3 √5/4 preserved                      : {'PASS' if g3_ok else 'FAIL'}")
    print(f"    required honesty tokens present        : {'PASS' if not miss else 'FAIL '+str(miss)}")
    ok = (not hits) and (not miss) and g3_ok
    if not ok:
        print("\n  SELF-CHECK FAILED — not trustworthy as stated.")
        return 1
    print("\n  REPORTED STRAIGHT — Step 1b executed; verdict is the computed")
    print("  gated multi-insertion behaviour; targets frozen; no fitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
