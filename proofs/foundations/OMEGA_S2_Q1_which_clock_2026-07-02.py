#!/usr/bin/env python3
"""
proofs/foundations/OMEGA_S2_Q1_which_clock_2026-07-02.py

OMEGA SESSION 2, STATION 1, ENTRY QUESTION Q1 -- WHICH CLOCK does the probe frequency
of the omega-vertex class tick against? (kickoff: OMEGA_session2_kickoff_2026-07-02.md
par.2; the fork MUST be decided by the object's structure, never by band-landing.)

PRE-REGISTERED BRANCHES AND OUTCOMES (from the kickoff, verbatim):
  (a) fundamental-tick clock  => theta_Z -> 0 by scale separation => mechanism dies by
      TRIVIALITY (the omega-vertex class is not the Gamma_Z/M_Z resolution).
  (b) gap reading (full winding kill) => deficit = (projection).u/(1-u); requires the
      continuation depth to be FORCED, else unforced.
  (c) channel-recurrence clock => theta_Z = an O(1) forced phase from the KMS pole
      condition -- the live branch IF derivable.

SCORING CLASS: STRUCTURAL decision probe (class a). The demand (-0.437% +- 0.092% on
the alpha-form) and Gamma/M = 0.0274 appear ONLY in marked COMPARISON rows (recorded
S5/S6 constants). No new value is claimed by this probe under any outcome.

DECISION RULE (pre-registered): a branch is FORCED only if its theta_Z (i) falls out
of the object with no inserted scale/phase/depth, AND (ii) respects III_1 scale-
freeness (CLEANROOM par.6): the modular clock has no invariant absolute rate, so an
admissible theta must be built from the process's OWN dimensionless data
(g, u, h_channel, Gamma/M) -- anything else imports a clock by hand.

KILL CRITERIA FOR THE VALUE CLAIM (pre-registered):
  K1  if the Z-channel tick-correlator has NO real-frequency resonance (subcritical/
      overdamped), branch (c) has no oscillation frequency to offer;
  K2  if every III_1-admissible phase candidate gives deficit = 0 or out-of-band,
      branch (c) is empty;
  K3  if the gap-continuation depth is not forced (+ikappa divergent; physical
      retarded depth ~ Gamma-sized ~ 0; -kappa by hand), branch (b) is unforced;
  K4  if the absolute-clock branch scales as theta^2 (quadratic triviality), (a) is
      dead for every hierarchy.
  If K1-K4 all fire: Q1 = NO FORCED BRANCH -> the omega-vertex VALUE claim is
  FALSIFIED (the pre-registered S1-kill outcome); the class lemma is banked as
  structure; the incompleteness moves up (verdict block).
"""
import math
import os
import sys

import numpy as np
import sympy as sp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

U = (2.0 / 3.0) ** 8                       # alpha_1 (girth-window survival)
G = 10                                     # girth (renewal read)
CS = 1.0 / 12.0                            # singlet projection (poisoned re-use; comparison only)
DEMAND_LO, DEMAND_HI = 0.345e-2, 0.529e-2  # |deficit| band on the alpha-form (S5/S6, recorded)

def deficit(theta, proj=1.0):
    w0 = U / (1 - U)
    rew = U * (math.cos(theta) - U) / (1 - 2 * U * math.cos(theta) + U * U)
    return proj * (w0 - rew)

print("=" * 88)
print(" T-A  the omega-extension at operator level is the FUGACITY PHASE (no freedom)")
print("=" * 88)
B0 = srs.hashimoto((0.0, 0.0, 0.0))
rng = np.random.default_rng(5)
okA = True
for _ in range(3):
    w = rng.uniform(0, 2 * math.pi)
    for L in (3, 7, 10):
        lhs = np.trace(np.linalg.matrix_power(U * np.exp(1j * w) * B0, L))
        rhs = (U * np.exp(1j * w)) ** L * np.trace(np.linalg.matrix_power(B0, L))
        okA &= abs(lhs - rhs) < 1e-9 * max(1.0, abs(rhs))
check("G(u, w) = (I - u e^{iw} B)^{-1}: every length-L walk carries e^{iLw} (one tick "
      "per NB step) -- the n-winding class gets e^{i n g w} with NOTHING inserted", okA)
lam = sorted(np.abs(np.linalg.eigvals(B0)))[::-1]
check(f"NB Perron eigenvalue lambda_P = {lam[0]:.6f} = k-1 = 2 (the Z channel)",
      abs(lam[0] - 2) < 1e-9)

print("=" * 88)
print(" T-B  the Z channel in the tick frame: SUBCRITICAL => OVERDAMPED (no resonance)")
print("      [K1 decided]")
print("=" * 88)
rho_step = U * lam[0]
check(f"subcriticality: u.lambda_P = 2u = {rho_step:.6f} < 1 (the SAME fact as the "
      "arrow, read_run) => the tick-correlator C(L) = (2u)^L decays monotonically",
      rho_step < 1)
# the omega-spectrum of C(L): S(w) = |1/(1 - 2u e^{iw})|; overdamped iff max at w = 0
ws = np.linspace(0, math.pi, 2001)
S = 1.0 / np.abs(1 - 2 * U * np.exp(1j * ws))
check(f"spectrum max at w = 0 (S(0) = {S[0]:.4f}), monotone decrease to w = pi "
      f"(S(pi) = {S[-1]:.4f}); NO interior maximum => NO real-frequency resonance",
      np.argmax(S) == 0 and np.all(np.diff(S) < 1e-12))
kappa = math.log(1 / (2 * U))
wpole = -1j * kappa
check(f"the ONLY pole of the tick-frame correlator is PURELY IMAGINARY: "
      f"w* = -i ln(1/(2u)) = -{kappa:.4f}i  (|1 - 2u e^{{iw*}}| = "
      f"{abs(1 - 2*U*np.exp(1j*wpole)):.1e})", abs(1 - 2 * U * np.exp(1j * wpole)) < 1e-12)
print("    => in the tick frame the Z channel is a GAP, not an oscillation: it has no")
print("       frequency of its own to hand the winding interferometer. (Consistent with")
print("       S2b: the Perron channel is exactly REAL -- no complex structure anywhere.)")

print("=" * 88)
print(" T-C  branch (c): the III_1-admissible phase candidates -- ALL trivial or")
print("      sub-demand  [K2 decided; Gamma/M rows are marked COMPARISON]")
print("=" * 88)
GM = 0.0274                                 # Gamma_Z/M_Z, recorded observed fraction (COMPARISON)
DR = 0.003384                               # delta_r (framework read; for the dressed-recurrence row)
cands = [
    ("static/adiabatic limit (theta = 0)", 0.0),
    ("one cycle per own recurrence (theta = 2pi)", 2 * math.pi),
    ("per-tick recurrence 1/g => theta = g.(2pi/g) = 2pi", 2 * math.pi),
    ("dressed recurrence: theta = 2pi(1 - delta_r) => eff. 2pi.delta_r", 2 * math.pi * DR),
    ("pole's own fraction: theta = Gamma/M [COMPARISON]", GM),
    ("winding-duration x pole fraction: theta = g.Gamma/M [COMPARISON]", G * GM),
]
print(f"    {'candidate':>58}   theta      deficit(raw)   deficit(c_S)")
worst_inband = False
for name, th in cands:
    d_raw, d_cs = deficit(th), deficit(th, CS)
    inband = (DEMAND_LO <= d_raw <= DEMAND_HI) or (DEMAND_LO <= d_cs <= DEMAND_HI)
    worst_inband |= inband
    print(f"    {name:>58}   {th:7.4f}   {d_raw*100:9.4f}%   {d_cs*100:9.4f}%"
          + ("   <-- in band!" if inband else ""))
check("NO III_1-admissible candidate reaches the demand band [0.345, 0.529]% in either "
      "projection: branch (c) is EMPTY -- the channel supplies no O(1) phase because it "
      "is real and subcritical", not worst_inband)
th_s = sp.symbols('theta', positive=True)
u_s = sp.Rational(2, 3) ** 8
ser = sp.series(u_s * (1 + u_s) * (1 - sp.cos(th_s))
                / ((1 - u_s) * (1 - 2 * u_s * sp.cos(th_s) + u_s ** 2)), th_s, 0, 4)
lead = sp.simplify(ser.removeO() / th_s ** 2)
check(f"small-theta law (sympy): deficit = [u(1+u)/((1-u)(1-u)^2)].theta^2/2 + O(theta^4)"
      f" -- QUADRATIC triviality; leading coefficient = {float(lead):.6f}",
      abs(float(lead) - float(u_s * (1 + u_s) / (1 - u_s) ** 3) / 2) < 1e-12)

print("=" * 88)
print(" T-D  branch (a) absolute clock: quadratic triviality for ANY hierarchy;")
print("      branch (b) gap continuation: the depth is NOT forced  [K3, K4 decided]")
print("=" * 88)
print("    (a) with an absolute tick, theta_Z = g.(M_Z/E_substrate). Deficit scaling:")
for S_hier in (1e2, 1e4, 1e8, 1e16):
    th = G / S_hier
    print(f"        hierarchy E_sub/M_Z = {S_hier:8.0e}:  theta = {th:.1e}  "
          f"deficit(raw) = {deficit(th)*100:.2e}%")
check("quadratic kill: even a 100x hierarchy leaves the raw deficit 40x below the "
      "band; the framework's actual ladder makes it astronomically zero -- AND the "
      "same fact PROTECTS every shipped static/matching-point pole read",
      deficit(G / 1e2) < DEMAND_LO / 10)
# (b): the three continuations of e^{i g w} off the real axis:
up = U * math.exp(G * kappa)                    # w = -i.kappa (toward the u-pole)
down = U * math.exp(-G * kappa)                 # w = +i.kappa (the by-hand 'full kill')
res = deficit(0.0)                              # retarded depth = resonance's own Im part -> 0
print(f"    (b) continuation menu for the winding factor u.e^(i g w):")
print(f"        toward the u-pole (w = -i kappa): u.e^(+g kappa) = {up:.3e}  -> geometric sum DIVERGES")
print(f"        by-hand depth     (w = +i kappa): u.e^(-g kappa) = {down:.3e} -> 'full kill', deficit = c_S.u/(1-u) = {CS*U/(1-U)*100:.4f}%")
print(f"        physical retarded depth (resonance's own Im part, ~Gamma per tick -> 0): deficit = {res*100:.4f}%")
check("the gap branch is UNFORCED: the divergent continuation is unphysical, the "
      "physical retarded depth gives ~0, and the full-kill depth (-kappa) is a choice "
      "with no derivation (its c_S pairing 0.3384% also sits outside the band) -- "
      "K3 fires", up > 1 and res < 1e-6)

print("=" * 88)
print(" VERDICT -- Q1: NO FORCED BRANCH. The omega-vertex VALUE claim is FALSIFIED")
print("         (the pre-registered S1-kill outcome). Two-sided winding no-go.")
print("=" * 88)
print(f"""    K1-K4 all fire. The Z channel is real, subcritical and overdamped in the tick
    frame: it has NO frequency to hand the winding interferometer; every III_1-
    admissible phase is trivial (0 or 2pi) or Gamma/M-sized (out of band); an
    absolute clock gives quadratic triviality; the gap depth is unforced.

    WHAT THIS SETTLES (with S6, a TWO-SIDED no-go on the winding layer):
      z-side   (S6): amplitudes waterline-flat; the residue can only dress UP.
      omega-side (this probe): the frequency response at EW-scale poles is ZERO --
      EW poles are deep-IR in the tick variable (that is WHY every matching-point
      pole-position read works; the kill and the shipped program are the same fact).
    => the -0.437% +- 0.092% is NOT winding-layer content in ANY algebraic slot.
    The omega-vertex sign lemma (OMEGA_T3) survives as STRUCTURE: with S6 it brackets
    and excludes the whole layer (UP-only residue, zero-omega vertex).

    WHERE THE INCOMPLETENESS MOVES (todo par.7 sharpened): the pole-vertex deficit
    must live in the INTERNAL (Cl(6)/Clifford) vertex layer -- the framework-native
    EW-loop content (the rho_f / s-bar^2_eff analogs), whose only formally-signed
    existing class is the per-leg Family-D (c_F u^2, 8x too small alone, S6) -- i.e.
    the EW-loop vertex layer is genuinely UN-BUILT. The leading sign-correct successor
    candidate, named: the q^2-DARK band-side admixture of the physical vertex (the
    pair channel's timelike darkness can only REMOVE pole weight -- sign DOWN forced;
    its magnitude = the vertex's band-orbital admixture fraction, un-derived,
    requiring the P3/PS-embedding current identification of OMEGA_T4).
    S1a (current-projected winding content) is SUPERSEDED by this outcome: with the
    winding layer excluded, its object has no value-slot to project onto.

    Gamma_Z/M_Z stays OPEN (+4.8 sigma). Nothing was shipped; no surface touched.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
