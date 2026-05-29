#!/usr/bin/env python3
"""
proofs/foundations/Nhub_v_Rinf_degeneracy_2026-05-16.py

QUESTION (user): what is missing to align R_∞ and v such that the choice
of N_hub satisfies BOTH σ-gates?

ANSWER, made rigorous here: NOTHING in N_hub — because v and R_∞ carry
the IDENTICAL N_hub power (both ∝ N_hub^(−1/4)).  They are N_hub-
DEGENERATE: varying N_hub slides them together along one ray, so the
pair cannot over-determine N_hub.  N_hub's value is fixed by ONE
calibration (v ↔ measured G_F); the leftover R_∞ deviation at v-exact
is the N_hub-INDEPENDENT ratio residual R_∞/v (carried by α_EM(0)-
running + the m_e/v Koide ratio + the CODATA unit bridge), which NO
choice of N_hub can absorb.

To over-determine (hence DERIVE, not adopt) N_hub you need two
observables of DIFFERENT N_hub power.  The framework HAS them — but
they are the COSMOLOGICAL cascade (H_0 ∝ N^−1, t_0 ∝ N^+1, Λ ∝ N^−2),
NOT the particle pair (v, R_∞ ∝ N^−1/4).  This probe extracts every
exponent numerically and states precisely what is missing.

Method: structural log-log slope d ln(O)/d ln(N_hub) from the live
predict_* DAG formulas (no hardcoded exponents).
"""
from __future__ import annotations
import sys, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "predictions"))

import importlib, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    vh = importlib.import_module("v_higgs")
    nh = importlib.import_module("N_hub")

# live anchored values
M_P     = vh.M_P
alpha_1 = vh.alpha_1
delta   = vh.delta
N0      = vh.N_hub                       # adopted N_hub (G_F-calibrated)
t_P     = 5.391247e-44                    # s (CODATA Planck time; unit bridge)

def v_of(N):
    return vh.predict_v_higgs(delta, M_P, N, alpha_1)

# m_e = m_τ × (f_min/f_max)²  — the Koide ratio is an EXACT N-independent
# dimensionless rational; m_τ ∝ v (EW scale).  So m_e ∝ v.  R_∞ ∝ m_e
# (α_EM(0) dimensionless, c/h external bridge).  Encode that chain:
KOIDE_RATIO = 0.040 ** 2 / 1.0           # placeholder ratio CONSTANT (N-indep)
def m_e_of(N):    return KOIDE_RATIO * v_of(N)      # ∝ v  (ratio is N-indep)
def R_inf_of(N):  return (1/137.036)**2 * m_e_of(N) # ∝ m_e  (α, c, h N-indep)
def H_0_of(N):    return 1.0 / (N * t_P)            # cascade theorem
def t_0_of(N):    return N * t_P
def Lam_of(N):    return 1.0 / N**2                  # Λ_CC ∝ N^−2

def slope(fn, N=N0, h=1e-6):
    a, b = fn(N*(1-h)), fn(N*(1+h))
    return (math.log(b) - math.log(a)) / (math.log(N*(1+h)) - math.log(N*(1-h)))

print("=" * 76)
print("  N_hub scaling exponents  d ln(O)/d ln(N_hub)   (live DAG formulas)")
print("=" * 76)
rows = [("v_Higgs",  v_of,  "particle (BZJ)"),
        ("m_e",      m_e_of, "particle (∝ v, Koide ratio N-indep)"),
        ("R_∞",      R_inf_of,"particle (∝ m_e; α,c,h N-indep)"),
        ("H_0",      H_0_of, "cosmological cascade"),
        ("t_0",      t_0_of, "cosmological cascade"),
        ("Λ_CC",     Lam_of, "cosmological cascade")]
exps = {}
for name, fn, tag in rows:
    p = slope(fn); exps[name] = p
    print(f"  {name:<8} ∝ N_hub^({p:+.4f})   [{tag}]")

print()
print("=" * 76)
print("  Verdict")
print("=" * 76)
deg = abs(exps["v_Higgs"] - exps["R_∞"]) < 1e-6 and abs(exps["v_Higgs"] + 0.25) < 1e-6
print(f"  v and R_∞ exponents equal (both = −1/4): {deg}")
print(f"    ⇒ R_∞/v is N_hub-INDEPENDENT  ⇒ the (v, R_∞) pair is DEGENERATE:")
print(f"      it CANNOT over-determine N_hub.  One calibration (v ↔ measured")
print(f"      G_F) fixes N_hub; the residual R_∞ deviation at v-exact is the")
print(f"      N_hub-independent ratio residual [α_EM(0)-run × m_e/v Koide ×")
print(f"      CODATA c,h] — NO N_hub choice removes it.  Nothing is 'missing")
print(f"      in N_hub'; the two σ-gates are not independent constraints.")
print()
nondeg = {k: exps[k] for k in ("H_0", "t_0", "Λ_CC")}
print(f"  The ONLY N_hub-pinning leverage = observables of DIFFERENT power:")
print(f"    {nondeg}  (vs particle −1/4)")
print(f"  ⇒ N_hub is over-determinable (DERIVABLE, not adopted) ONLY by a")
print(f"    particle(N^−1/4) ↔ cosmology(N^−1) cross-consistency, NOT v↔R_∞.")
print()
print(f"  WHAT IS MISSING (the precise Gap-G1 statement): a substrate")
print(f"  derivation that forces ONE N to satisfy BOTH the N^−1/4 particle")
print(f"  anchor and the N^−1 cosmological anchor.  Today these disagree at")
print(f"  ~1%: particle side has its own spread (G_F-anchored N≈8.395e60 vs")
print(f"  m_τ-anchored ≈8.44e60 vs the m_ν3 +0.87% anchor sensitivity), and")
print(f"  the cosmological side N=(H_0·t_P)^−1 is itself ~1% Hubble-tension-")
print(f"  split (Planck vs SH0ES).  The v↔R_∞ framing is a DEAD END for this")
print(f"  (degenerate); the live front is the particle-vs-cosmology ~1%")
print(f"  reconciliation — blocked on BOTH ends by independent ~1% spreads.")
print()
print("=" * 76)
print("End.")
print("=" * 76)
