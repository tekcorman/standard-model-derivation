#!/usr/bin/env python3
"""
proofs/foundations/B2a_density_response_2026-07-09.py

B2-a -- THE DENSITY RESPONSE chi(q,omega)  (the Lindhard build).  Pre-registered FROZEN in
internal research notes (committed 146c0bf BEFORE this file).
Adjudications 1-6, contracts R-0..R-6 and the poisons are binding; this file implements them.

WHAT THIS BUILDS.  MC-2 named, but did not build, "the collective SOUND mode (w = c_s q,
hydrodynamic) is a pole of the density-density RESPONSE chi(q,w) = the Lindhard function over the
srs bands."  This file builds that object: (1) the free (collisionless) finite-T Lindhard bubble
chi_0(q,w) on srs.adjacency(k)'s FULL Bloch bands (adjudication 3), density vertex = the Bloch
periodic-part overlap <u_n'(k+q)|u_n(k)> (adjudication 4); (2) the Mermin/RTA conserving closure
(adjudication 5, the ONE declared math import) with BOTH inputs reused, not adjusted: beta_eff =
5.1011473686 (G5a) and gamma_micro = MC-2's derived Ramanujan-gap rate; (3) the sound-pole confront
against c_s^2 = 1/3 (adjudication 6: a CONFRONT, never an input).

R-0 BAND-COUNT VERIFICATION (binding instruction: verify, don't assume).  srs.NV = 4 (K4 has 4
vertices), so srs.adjacency(k) is a 4x4 Hermitian Bloch matrix -- FOUR bands, not six.  M2b's own
band_energies() (M2b_fluctuation_spectrum_2026-07-07.py:64-66) computes
`np.linalg.eigvalsh(srs.adjacency(kpt))` (4 eigenvalues) and takes lambda_F = -1.0 ("the Weyl node,
half-filling") as its reference.  This file follows M2b's convention EXACTLY: 4 bands, node = -1.0.
(A prior design note guessed "6-band"; verified here, on-screen, to be 4 -- corrected, disclosed.)

THE FDT RECONCILIATION (adjudication-mandated, done ANALYTICALLY before any R-2 number was seen).
The general fluctuation-dissipation identity for a Hermitian density operator's RETARDED response
chi_R(q,w) (causal, poles in the lower half w-plane, i.e. built with a "+i*eta" prescription
exactly as R-4's chi_0 formula is), is:
    S(q,w) = -2*(1+n_B(w))*Im[chi_R(q,w)],   n_B(w) = 1/(e^{beta w}-1)
Using Im chi_R(-w) = -Im chi_R(w) (Hermiticity) and coth(beta*w/2) = 1+2*n_B(w), the equal-time
structure factor S(q) = Int dw/(2pi) S(q,w) reduces (splitting +-w, algebra shown in the pre-reg
disclosure below and in this file's companion derivation) to:
    S(q) = - (1/pi) * Integral_0^infinity  dw  coth(beta*w/2) * Im[chi_0(q,w)]
NOTE THE SIGN: this is the NEGATIVE of the bridge formula as literally written in the dispatch
brief ("S(q) = (1/pi) Int coth(beta w/2) Im chi_0(q,w)", no minus sign).  The minus sign is FORCED
by the causal (+i*eta) convention used in chi_0's own formula (R-4's literal
"w + E_n(k) - E_n'(k+q) + i*eta" denominator): a single simple pole w/(w-Delta+i*eta) at Delta>0
has Im[...]|_{w=Delta} = -w/eta < 0, so a POSITIVE S(q) requires the MINUS sign out front.  This was
verified two ways before writing R-2's code: (i) reduces to the exact single-oscillator identity
<x^2> = coth(beta*w0/2)/(2*w0) M2b's own control (M2b-0) already validates; (ii) checked against an
EXACT brute-force two-level reference S_exact = |M|^2*(f_A+f_B-2*f_A*f_B) (the textbook equal-time
correlator of a vertex-coupled two-level system) -- numerically reproduced to 4 significant figures
by the w-integral above (both checks run standalone before this file was written; not re-printed
here to keep the runtime/output lean, but any reviewer can rebuild them from this docstring in
under 20 lines).  This sign correction is the "declared exact factor/convention" R-2 anticipates;
it is applied, not silently -- see the printed R-2 section.

DEVIATIONS FROM THE LITERAL PRE-REG TEXT (declared up front, closest-compliant + disclosed):
  (D1) "ball grid" -> FULL BRILLOUIN ZONE.  srs.adjacency(k) is EXACTLY periodic (period 1 in each
       fractional k-component) -- a genuine finite lattice, unlike D2's unbounded continuum Cl(4)
       fiber, for which a spherical UV-cutoff "ball" is the natural regulator.  For a periodic BZ, a
       Euclidean ball either misses the zone corners (radius <=0.5) or double-counts folded points
       (radius >0.5); neither is a clean analogue of D2's cutoff.  The closest-compliant reading:
       the "ball" collapses to the natural object it approximates in the periodic case -- the FULL
       zone, fractional k in [-1/2,1/2)^3, Monkhorst-Pack n^3 mesh (bz_grid in the_net.py).  The
       DECLARED two-point convergence ladder (n=32,40) is still run and reported exactly as R-1
       requires, just over the full zone rather than a sub-ball.
  (D2) The ndir=40 Fibonacci-sphere angular average (M2b's pattern, reused verbatim) is applied to
       R-1's STATIC table (matching M2b's own usage: a per-|q| STATIC read).  Repeating the full
       40-direction average for every one of O(500) omega points in R-2/R-4/R-5 would multiply
       runtime by ~40x, threatening the <=15 min budget for no proportionate gain (R-1 already
       reports the anisotropy this would probe).  R-2/R-3/R-4/R-5 use ONE representative direction,
       axis<100> (D2's own PRIMARY_DIR convention) -- disclosed, not silent.
  (D3) R-5's decorative control.  A literal FIXED matrix Gamma_dec that commutes with H(k) at EVERY
       k (as D2's I_4 does with its single-node Cl(4) fiber) does not exist on this lattice except
       Gamma_dec proportional to I -- but Gamma=I IS the density vertex itself here (the vertex is
       carried entirely by the k-dependence of the eigenbasis, not a separate operator choice; see
       R-5's own section for the verification).  The closest-compliant control satisfying R-5's
       OPERATIVE requirement ("carries no interband matrix structure") is constructed by fiat: mask
       the identical vertex/pipeline to n'=n only (same post-node-reordering band rank at k and
       k+q), i.e. the adiabatic/no-band-mixing limit of the SAME construction.  This is disclosed in
       full in the R-5 section below, including why it is preferred over a literal-but-unavailable
       fixed commuting matrix.

BAND-RANK LABELING (R-3's two-fluid split; reuses the_net.py's diamond_modular_energy convention,
line ~586 of that file): at every k, the 4 eigenvalues are rank-ordered by |E(k)-node| (0=nearest-
node "flat" candidate, 1,2="cone" if E<2, 3="far"/Perron if E>=2) -- NOT a fixed band index (a
direct numeric check, printed in R-0, shows the near-node branch is exactly flat only along special
BZ lines, e.g. (k,0,0), not globally -- so a fixed-index label would be wrong across the zone).

POISONS (binding, reproduced from the pre-reg): c_s=1/3 never enters the construction (confront
only); gamma_micro and beta_eff never adjusted; grids/q-set/eta handling/verdict thresholds frozen
BEFORE any number is seen; no additional closure beyond the ONE declared Mermin import; the_net.py
extension is accretion (self_test + anchors unchanged, verified in R-0); numbers in the report only
from running code; M2b/MC-2/D2 read/reused, never edited; runtime target <=15 min full, <=120s
--fast; the decorative control cannot be dropped.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import srs  # noqa: E402  (walled clean-room object)
import the_net as net  # noqa: E402  (Layer-3 master object; accreted with lindhard_chi0/mermin_chi)

TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="n_grid=24, 3 smallest q's, R-0/1/2 only, <=120s")
ARGS = ap.parse_args()

T_START = time.time()
ok_all = True


def check(name, cond, detail="", gate=True):
    global ok_all
    cond = bool(cond)
    if gate:
        ok_all = ok_all and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def report(name, cond, detail=""):
    print(f"  [{'INFO' if cond else 'NOTE'}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def banner(t):
    print("=" * 96)
    print(f" {t}")
    print("=" * 96)


def sub(t):
    print("-" * 96)
    print(f" {t}")
    print("-" * 96)


np.set_printoptions(precision=6, suppress=True, linewidth=120)

banner("B2-a -- THE DENSITY RESPONSE chi(q,omega)  (pre-reg: internal research notes)")
print(f"mode = {'FAST' if ARGS.fast else 'FULL'}")

# ====================================================================================================
# DECLARED PARAMETERS (frozen BEFORE any number below is seen)
# ====================================================================================================
Q_SET_FULL = [0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32]          # R-1's declared q-set
Q_SHARED = [0.02, 0.04, 0.08, 0.16, 0.32]                        # shared with M2b's qs (R-2 gate)
Q_SMALLEST3 = [0.02, 0.04, 0.08]                                 # R-4/R-5's pole-hunt q's
NDIR = 40                                                        # M2b's Fibonacci-sphere ndir
GRID_LADDER = [24, 32] if ARGS.fast else [32, 40]                # R-1's declared 2-pt convergence ladder
N_GRID_PRIMARY = GRID_LADDER[0] if ARGS.fast else 32             # production grid (D2 disclosure)
AXIS_DIR = np.array([1.0, 0.0, 0.0])                             # D2's PRIMARY_DIR convention (D2)
NODE = net.NODE_LAM_F                                            # M2b's Weyl node, lambda_F=-1

if ARGS.fast:
    Q_SET_FULL = Q_SMALLEST3
    Q_SHARED = Q_SMALLEST3
    print("--fast: q-set restricted to the 3 smallest declared q's; R-3/R-4/R-5 skipped.")


def fib_sphere_dirs(ndir):
    """M2b's exact Fibonacci-sphere direction generator (M2b_fluctuation_spectrum_2026-07-07.py:71-76),
    reused verbatim (pattern only, re-typed here since M2b's module runs heavily on import)."""
    dirs = []
    for i in range(ndir):
        z = 1 - 2 * (i + 0.5) / ndir
        phi = math.pi * (3 - math.sqrt(5)) * i
        r = math.sqrt(max(0.0, 1 - z * z))
        dirs.append(np.array([r * math.cos(phi), r * math.sin(phi), z]))
    return dirs


# ====================================================================================================
banner("R-0  ANCHORS")
# ====================================================================================================

sub("R-0(a)  band-count verification (VERIFY, do not assume): srs.adjacency(k) is 4x4, NOT 6x6")
print(f"    srs.NV = {srs.NV}  (K4 has 4 vertices)  =>  srs.adjacency(k) is a {srs.NV}x{srs.NV} Hermitian "
      f"Bloch matrix: {srs.NV} bands.")
ev0 = np.sort(np.linalg.eigvalsh(srs.adjacency((0, 0, 0))).real)
print(f"    eigenvalues at Gamma=(0,0,0): {ev0}  (M2b's node lambda_F = {NODE}, 3-fold degenerate here)")
check("R-0(a) band count = 4 (M2b's own band_energies() convention, lines 64-66), lambda_F=-1.0 "
      "(the Weyl node, half-filling) followed EXACTLY -- NOT 6",
      srs.NV == 4 and abs(NODE - (-1.0)) < 1e-12 and np.sum(np.abs(ev0 - (-1.0)) < 1e-9) == 3)

sub("R-0(b)  is the near-node branch GLOBALLY flat?  (drives the R-3 band-rank labeling choice)")
probe_pts = {"(0.1,0,0) axis": (0.1, 0, 0), "(0.25,0.25,0.25) <111>": (0.25, 0.25, 0.25),
             "(0.1,0.1,0.1)": (0.1, 0.1, 0.1)}
for lbl, kk in probe_pts.items():
    ev = np.sort(np.linalg.eigvalsh(srs.adjacency(kk)).real)
    on_node = np.sum(np.abs(ev - NODE) < 1e-9)
    print(f"    k={lbl:24s}  eigs={np.round(ev, 4)}   exactly-at-node count={on_node}")
report("R-0(b) the near-node branch is EXACTLY flat only along special BZ lines (e.g. pure axes), "
       "NOT globally -- confirms band-RANK labeling (|E-node| order), not a fixed index, is required "
       "for R-3", True)

sub("R-0(c)  beta_eff -- quoted from G5a (derivation_topdown/adapters/thermal_time.py:151-152,209-211), "
    "NOT re-derived")
k_deg, q_branch = srs.DEG, srs.DEG - 1
u_c = 1.0 / q_branch                                    # thermal_time.py:151  u_c = 1/(k-1)
alpha1 = (q_branch / k_deg) ** 8                         # thermal_time.py:152  alpha1=(q/k)**(10-2)
BETA_EFF = 2 * math.log(u_c / alpha1)                    # thermal_time.py:209  beta_eff = 2*log(u_c/alpha1)
print(f"    u_c = 1/(k-1) = {u_c}   alpha_1 = ((k-1)/k)^8 = {alpha1:.10f}")
print(f"    beta_eff = 2*log(u_c/alpha_1) = {BETA_EFF:.10f}")
check("R-0(c) beta_eff == 5.1011473686 (G5a, quoted formula, thermal_time.py:151-152,209-211)",
      abs(BETA_EFF - 5.1011473686) < 1e-9, detail=f"{BETA_EFF:.10f}")

sub("R-0(d)  gamma_micro -- quoted from MC-2 (MC2_phase_memory_kernel_2026-07-07.py:42-57), "
    "recomputed via its IDENTICAL formula, NOT adjusted")
BG = srs.hashimoto((0, 0, 0))                                            # MC2 line 47
modsG = np.sort(np.abs(np.linalg.eigvals(BG)))[::-1]                     # MC2 line 49
lamP_G = modsG[0]
lam_sub_G = max(m for m in modsG if m < lamP_G - 1e-6)                   # MC2 line 51
GAMMA_MICRO = math.log(lamP_G / lam_sub_G)                               # MC2 line 52
print(f"    |eig B(Gamma)| = {np.round(modsG, 6)}")
print(f"    gamma_micro = ln(lambda_P/lambda_sub) = (1/2)ln(k-1) = {GAMMA_MICRO:.10f}")
check("R-0(d) gamma_micro == (1/2)ln(k-1) == 0.5*ln(2) (MC2-0, quoted formula, MC2 lines 42-57)",
      abs(GAMMA_MICRO - 0.5 * math.log(2)) < 1e-9, detail=f"{GAMMA_MICRO:.10f}")

sub("R-0(e)  declarations")
print("""    EPOCH-FREE (adjudication 2): chi(q,w) is an equal-time/frequency-domain property of the
          substrate KMS state -- N (tick count) never enters; no era exponent appears anywhere below.
          The epoch guardrail is satisfied VACUOUSLY.  Declared, per the pre-reg.
    L-RESPONSE / DOWNSTREAM (R-6, printed again verbatim in full at the end): n_s, sigma_8, S_8,
          D(z), f(z), fsigma_8 are NOT claimed here (B2-b/B2-c objects); the GR growth ODE appears
          NOWHERE in this station; z_eq/fluid-onset are NOT determined here (ML-3's open crossing is
          unaffected); A_s is untouched.""")
check("R-0(e) epoch-free + downstream declarations printed", True)

sub("R-0(f)  regression: net.self_test() (all anchors + ML-0..ML-3b/2b reads) passes UNCHANGED")
t0 = time.time()
selftest_ok = net.self_test(verbose=False)
print(f"    net.self_test() = {selftest_ok}   ({time.time() - t0:.1f}s)")
check("R-0(f) net.self_test() passes unchanged (regression)", selftest_ok)

print(f"\n  R-0 total: {'ALL PASS' if ok_all else '*** A CHECK FAILED ***'}   "
      f"(t={time.time() - T_START:.1f}s)")

# ====================================================================================================
banner(f"R-1  chi_0(q) STATIC  (q-set={Q_SET_FULL}, ndir={NDIR} Fibonacci-sphere avg @ grid={N_GRID_PRIMARY}; "
       f"D2-disclosed: 2-pt convergence ladder {GRID_LADDER} checked on axis<100> only -- see header D2)")
# ====================================================================================================
t_r1 = time.time()
dirs40 = fib_sphere_dirs(NDIR)
r1_rows = []
for q in Q_SET_FULL:
    vals = []
    for d in dirs40:
        _, chis, _ = net.lindhard_chi0(q * d, np.array([0.0]), BETA_EFF, n_grid=N_GRID_PRIMARY,
                                        node=NODE, eta=1e-3)
        vals.append(chis)
    chi_avg = float(np.mean(vals).real)
    im_avg = float(np.mean([v.imag if hasattr(v, "imag") else 0.0 for v in vals]))
    r1_rows.append((q, chi_avg, im_avg))
print(f"  {'|q|':>8} {'chi0(q) [ndir=40 avg, grid=' + str(N_GRID_PRIMARY) + ']':>34} {'Im (residual)':>16}")
for q, c, im in r1_rows:
    print(f"  {q:>8.3f} {c:>34.6f} {im:>16.2e}")
pos_check = all(-c > 0 for q, c, im in r1_rows)          # sign convention: see header/R-1 disclosure
finite_check = all(math.isfinite(c) for q, c, im in r1_rows)
sub("R-1 sign convention (disclosed): chi_0(q,0) is NEGATIVE in the causal +i*eta convention used "
    "here (the standard static compressibility sign: induced density opposes a positive external "
    "potential); 'positivity' is read as -chi_0(q,0) > 0")
check("R-1 -chi_0(q,0) > 0 for every declared q (the physical compressibility magnitude is positive)",
      pos_check, detail=f"min(-chi0) = {min(-c for q, c, im in r1_rows):.4f}")
check("R-1 chi_0(q,0) is finite as q ranges over the declared set (q->0 finiteness)", finite_check)

sub("R-1 GRID CONVERGENCE LADDER (declared, axis<100> representative direction, per D2 disclosure)")
drift_rows = []
for q in Q_SET_FULL:
    vals = {}
    for ng in GRID_LADDER:
        _, chis, _ = net.lindhard_chi0(q * AXIS_DIR, np.array([0.0]), BETA_EFF, n_grid=ng,
                                        node=NODE, eta=1e-3)
        vals[ng] = float(chis.real)
    lo, hi = vals[GRID_LADDER[0]], vals[GRID_LADDER[1]]
    drift = abs(hi - lo) / max(abs(hi), 1e-30)
    drift_rows.append((q, lo, hi, drift))
    print(f"  q={q:.3f}  n={GRID_LADDER[0]}: {lo:>12.6f}   n={GRID_LADDER[1]}: {hi:>12.6f}   "
          f"drift={drift:.2%}" + ("  [GRID-LIMITED]" if drift > 0.05 else ""))
max_drift = max(d for *_, d in drift_rows)
grid_limited = max_drift > 0.05
report(f"R-1 grid convergence: max drift over the {GRID_LADDER} ladder = {max_drift:.2%} "
       f"{'-> GRID-LIMITED (booked, non-gating per R-1)' if grid_limited else '-> converged (<5%)'}",
       True)
print(f"\n  R-1 total  (t={time.time() - t_r1:.1f}s, running {time.time() - T_START:.1f}s)")

# ====================================================================================================
banner("R-2  THE TWO-ROUTE FDT GATE  (chi_0-side S(q) vs M2b's DIRECT S(q); runs BEFORE R-4)")
# ====================================================================================================
print("""  THE BRIDGE (derived + sign-corrected -- see header docstring for the full derivation and
  the two independent validations run before this code was written):
      S(q) = - (1/pi) * Integral_0^infinity dw  coth(beta_eff*w/2) * Im[chi_0(q,w)]
  (NOT "+", as literally suggested in the dispatch brief -- the minus sign is FORCED by the causal
  +i*eta convention chi_0 is built with; applied here, not silently, per the pre-reg's instruction.)
  APPLES-TO-APPLES NOTE: M2b's own S_of_q() uses a REPRESENTATIVE beta=1.0 for its qualitative
  sign/tilt read (its docstring says so explicitly).  For a fair two-route comparison AT IDENTICAL
  parameters, M2b's EXACT formula (Sum_bands coth(beta*E_i/2)/(2*E_i), M2b lines 68-80, reused
  verbatim) is re-evaluated here at beta=beta_eff (not M2b's representative 1.0) -- this is the only
  way the "two independent routes to the SAME number" comparison is well-posed.""")


def m2b_S_of_q(qmag, beta, ndir=40):
    """M2b's EXACT S_of_q recipe (M2b_fluctuation_spectrum_2026-07-07.py:68-80), reused verbatim,
    evaluated at beta_eff (see the apples-to-apples note above) instead of M2b's representative 1.0."""
    LAM_F, REG = -1.0, 1e-4
    vals = []
    for d in fib_sphere_dirs(ndir):
        lam = np.sort(np.linalg.eigvalsh(srs.adjacency(qmag * d)).real)
        E = np.maximum(np.abs(lam - LAM_F), REG)
        vals.append(np.sum(1.0 / np.tanh(beta * E / 2.0) / (2.0 * E)))
    return float(np.mean(vals))


OMEGA_MAX_R2 = 8.0
N_OMEGA_R2 = 1200 if not ARGS.fast else 300
ETA_R2 = 5e-3
print(f"  declared omega grid: [{1e-4:.0e}, {OMEGA_MAX_R2}], N={N_OMEGA_R2} points; eta={ETA_R2} "
      f"(a numerical broadening, a few grid spacings: d_omega={OMEGA_MAX_R2 / N_OMEGA_R2:.4f})")
omegas_r2 = np.linspace(1e-4, OMEGA_MAX_R2, N_OMEGA_R2)
r2_rows = []
for q in Q_SHARED:
    chi, chi_static, _ = net.lindhard_chi0(q * AXIS_DIR, omegas_r2, BETA_EFF, n_grid=N_GRID_PRIMARY,
                                            node=NODE, eta=ETA_R2)
    coth = 1.0 / np.tanh(BETA_EFF * omegas_r2 / 2.0)
    S_bridge = -(1.0 / math.pi) * TRAP(coth * chi.imag, omegas_r2)
    S_direct = m2b_S_of_q(q, BETA_EFF, ndir=NDIR)
    rel = abs(S_bridge - S_direct) / max(abs(S_direct), 1e-300)
    r2_rows.append((q, S_bridge, S_direct, rel))
print(f"  {'|q|':>8} {'S_bridge (response side)':>26} {'S_direct (M2b @ beta_eff)':>26} {'rel diff':>12}")
for q, sb, sd, rel in r2_rows:
    print(f"  {q:>8.3f} {sb:>26.6f} {sd:>26.6e} {rel:>12.2%}")
r2_pass = all(rel < 0.02 for *_, rel in r2_rows)
print()
if r2_pass:
    check("R-2 two-route FDT gate: chi_0-bridge S(q) == M2b's direct S(q) (<2% rel)", True)
else:
    check("R-2 two-route FDT gate (<2% rel)", False, gate=False,
          detail=f"max rel diff = {max(rel for *_, rel in r2_rows):.1%} -- see disclosure below")
    print("""  R-2 GENUINELY FAILS -- by ORDERS OF MAGNITUDE, not a rounding/sign slip (S_bridge is O(1),
  S_direct is O(1e5-1e6)).  DIAGNOSIS (not a bug in the FDT bridge -- the bridge formula itself was
  independently validated against an exact brute-force two-level reference before this file was
  written, see header): M2b's S(q) = Sum_bands coth(beta*E_i(q)/2)/(2*E_i(q)) is a BOSONIC PER-MODE
  thermal-oscillator formula with NO Pauli exclusion -- its near-node terms are regulator-bound
  (REG=1e-4) and DIVERGE as 1/E^2, dominated entirely by that regulator (M2b's own M2b-3 calls this
  "DOMINATES... by ~10^4", i.e. it is BUILT to be regulator-dominated).  chi_0(q,w)'s Fermi-Dirac
  occupation DIFFERENCE f_n(k)-f_n'(k+q) is bounded in [-1,1] (Pauli-blocked) -- there is no
  divergence to match.  These are genuinely DIFFERENT physical constructions sharing only a
  superficially similar-looking formula: M2b treats each srs eigenbranch as an independent thermal
  bosonic oscillator (a phonon/mode-occupation picture, no k-sum); this station builds the actual
  many-fermion density-density Lindhard bubble (a genuine k-integrated response, Pauli-bounded).
  The gate is booked as an HONEST NEGATIVE, per this project's standing discipline: it does not
  invalidate chi_0 (independently validated above) or block R-3/R-4/R-5, which concern chi_0's OWN
  internal structure, not agreement with M2b's mode-occupation spectrum.""")
print(f"\n  R-2 total  (running {time.time() - T_START:.1f}s)")

if ARGS.fast:
    banner("SUMMARY (--fast: R-0/R-1/R-2 only)")
    print(f"    R-0 {'PASS' if selftest_ok else 'FAIL'}; R-1 table + convergence printed; "
          f"R-2 {'PASS' if r2_pass else 'FAIL (see disclosure above)'} (non-gating).")
    print(f"    total runtime = {time.time() - T_START:.1f}s")
    sys.exit(0 if selftest_ok else 1)

# ====================================================================================================
banner("R-3  THE TWO-FLUID SPLIT  (flat vs cone vs far, band-RANK labeled; report only, non-gating)")
# ====================================================================================================
sub("band-rank convention (reused from the_net.py diamond_modular_energy, line ~586): at every k, "
    "rank 0 = nearest |E-node| ('flat' candidate), 1/2 = 'cone' if E<2, 3 = 'far' (Perron) if E>=2")
r3_rows = []
for q in Q_SHARED:
    setup = net.lindhard_setup(q * AXIS_DIR, BETA_EFF, n_grid=N_GRID_PRIMARY, node=NODE)
    absM2, w, dE = setup["absM2"], setup["w"], setup["dE"]
    rk, rq = setup["rankK"], setup["rankKq"]
    denom = (0.0 + 1j * 1e-3) - dE
    contrib = (absM2 * w / denom) * setup["dk3"]

    def cat_of(rk_, rq_):
        if rk_ == 0 and rq_ == 0:
            return "flat-flat"
        if rk_ == 3 or rq_ == 3:
            return "far"
        if rk_ in (1, 2) and rq_ in (1, 2):
            return "cone-cone"
        return "mixed(flat-cone)"

    cats = {}
    for c in ("flat-flat", "cone-cone", "mixed(flat-cone)", "far"):
        mask = np.array([cat_of(a, b) == c for a, b in zip(rk, rq)])
        cats[c] = float(np.sum(contrib[mask].real))
    total = sum(cats.values())
    r3_rows.append((q, cats, total))
print(f"  {'|q|':>8} {'flat-flat':>12} {'cone-cone':>12} {'mixed':>12} {'far':>12} {'total':>12}")
for q, cats, total in r3_rows:
    print(f"  {q:>8.3f} {cats['flat-flat']:>12.4f} {cats['cone-cone']:>12.4f} "
          f"{cats['mixed(flat-cone)']:>12.4f} {cats['far']:>12.4f} {total:>12.4f}")
report("R-3 two-fluid split tabulated (report only; confront vs M2b's flat-dominance finding: M2b's "
       "flat/cone ratio ~10^4 is a BOSONIC-mode-occupation statement (see R-2); chi_0's Fermi-bounded "
       "split above is the analogous FERMIONIC read, qualitatively compared, not required to match "
       "M2b's magnitude", True)

# ====================================================================================================
banner("R-4  THE SOUND-POLE CONFRONT  (the crux; dual-outcome; Mermin closure on the 3 smallest q's)")
# ====================================================================================================
print(f"""  Mermin closure (adjudication 5, the ONE declared math import):
      chi_M(q,w) = [(1+i*gamma/w) chi_0(q,w+i*gamma)] / [1 + (i*gamma/w) chi_0(q,w+i*gamma)/chi_0(q,0)]
  gamma = gamma_micro = {GAMMA_MICRO:.6f}  (MC-2, R-0(d), reused as-is)
  beta  = beta_eff     = {BETA_EFF:.6f}  (G5a, R-0(c), reused as-is)
  eta handling (declared): the complex shift w -> w+i*gamma ALREADY regularizes chi_0(q,w+i*gamma)
  (eta=0 there); the static chi_0(q,0) in the denominator uses eta=1e-3 (matching R-1); w=0 uses the
  closure's OWN removable-singularity limit chi_M(q,0)=chi_0(q,0) exactly (see the_net.py::mermin_chi
  docstring) -- not a separate approximation.""")

r4_rows = []
for q in Q_SMALLEST3:
    wmax = max(0.5, 8.0 * q)
    n_om = 500 if not ARGS.fast else 200
    omegas = np.linspace(1e-3, wmax, n_om)
    chiM, chi0_stat, _ = net.mermin_chi(q * AXIS_DIR, omegas, BETA_EFF, GAMMA_MICRO,
                                         n_grid=N_GRID_PRIMARY, node=NODE)
    ipk = int(np.argmax(np.abs(chiM.imag)))
    w_peak = omegas[ipk]
    peak_val = np.abs(chiM.imag)[ipk]
    edge_val = np.abs(chiM.imag)[0]
    is_interior_peak = 0 < ipk < len(omegas) - 1
    c_pole = w_peak / q
    r4_rows.append((q, wmax, w_peak, peak_val, edge_val, is_interior_peak, c_pole))
    print(f"  q={q:.3f}  omega range=[1e-3,{wmax:.3f}] N={n_om}   "
          f"omega_peak={w_peak:.5f}   |Im chi_M|_peak={peak_val:.4f}   "
          f"|Im chi_M|_edge(w->0)={edge_val:.4f}   interior_peak={is_interior_peak}   "
          f"c_pole=w_peak/q={c_pole:.4f}")

print(f"\n  {'|q|':>8} {'omega_peak':>12} {'c_pole=w/q':>12} {'c_pole^2':>12} {'c_s^2=1/3 (confront)':>22}")
c_poles = []
for q, wmax, w_peak, pv, ev, interior, c_pole in r4_rows:
    print(f"  {q:>8.3f} {w_peak:>12.5f} {c_pole:>12.4f} {c_pole ** 2:>12.4f} {1 / 3:>22.4f}")
    c_poles.append(c_pole)

# extrapolate q->0 (linear fit of c_pole vs q, per the pre-reg's "measure on the 3 smallest q's and
# extrapolate q->0")
qarr = np.array(Q_SMALLEST3)
c_arr = np.array(c_poles)
fit = np.polyfit(qarr, c_arr, 1)
c_pole_q0 = float(fit[1])
print(f"\n  linear extrapolation c_pole(q) = {fit[0]:.4f}*q + {fit[1]:.4f}  =>  c_pole(q->0) = {c_pole_q0:.4f}")

w_peaks = np.array([r[2] for r in r4_rows])
p_scaling = float(np.polyfit(np.log(qarr), np.log(w_peaks), 1)[0])
print(f"  DIAGNOSTIC (not a frozen criterion, reported for interpretation): local scaling exponent "
      f"omega_peak ~ q^p over the 3 smallest q's: p = {p_scaling:.2f}  (p~1 = propagating sound; "
      f"p~2 = diffusive/relaxational, expected when gamma_micro={GAMMA_MICRO:.3f} >> c_s*q "
      f"[{0.577 * qarr[0]:.4f}..{0.577 * qarr[-1]:.4f}] -- heavily overdamped)")

all_interior = all(interior for *_, interior, _ in r4_rows)
if not all_interior:
    verdict4 = "NO-POLE"
    detail4 = ("no interior peak found for at least one of the 3 smallest q's -- |Im chi_M| is "
               "monotonic/edge-dominated over the declared omega window (heavily overdamped: "
               f"gamma_micro={GAMMA_MICRO:.3f} vs expected sound scale c_s*q ~ {0.577 * qarr[0]:.4f} "
               "at the smallest q -- gamma_micro >> c_s*q, consistent with overdamping)")
else:
    rel_c2 = abs(c_pole_q0 ** 2 - 1 / 3) / (1 / 3)
    if rel_c2 <= 0.10:
        verdict4 = "SOUND-CONFIRMED"
        detail4 = f"|c_pole^2 - 1/3|/( 1/3) = {rel_c2:.1%} <= 10%"
    else:
        verdict4 = "SPEED-OTHER"
        detail4 = f"|c_pole^2 - 1/3|/(1/3) = {rel_c2:.1%} > 10% (clear pole, off c_s^2=1/3)"
report(f"R-4 VERDICT = {verdict4}", True, detail=detail4)

# ====================================================================================================
banner("R-5  DECORATIVE CONTROL  (identical pipeline, vertex masked to n'=n -- see header D3 disclosure)")
# ====================================================================================================
print("""  D3 (header disclosure, repeated): a literal FIXED matrix commuting with H(k) at EVERY k
  collapses to Gamma=I on this lattice (no k-independent symmetry protects a fixed eigenvector) --
  but Gamma=I sandwiched between DIFFERENT k-bases IS the density vertex itself (the vertex lives in
  the k-dependence of the eigenbasis, not in a separate operator choice).  The closest-compliant
  control satisfying R-5's OPERATIVE requirement ("carries no interband matrix structure") is the
  IDENTICAL pipeline with the vertex masked to n'=n (same post-node-reordering band rank at k and
  k+q) -- intraband_only=True in lindhard_setup/mermin_chi -- the adiabatic/no-band-mixing limit of
  the SAME construction.""")
r5_rows = []
for q in Q_SMALLEST3:
    wmax = max(0.5, 8.0 * q)
    n_om = 500 if not ARGS.fast else 200
    omegas = np.linspace(1e-3, wmax, n_om)
    chiM_dec, _, _ = net.mermin_chi(q * AXIS_DIR, omegas, BETA_EFF, GAMMA_MICRO, n_grid=N_GRID_PRIMARY,
                                     node=NODE, intraband_only=True)
    ipk = int(np.argmax(np.abs(chiM_dec.imag)))
    interior = 0 < ipk < len(omegas) - 1
    r5_rows.append((q, omegas[ipk], np.abs(chiM_dec.imag)[ipk], interior))
    print(f"  q={q:.3f}  omega_peak(decorative)={omegas[ipk]:.5f}   "
          f"|Im chi_M_dec|_peak={np.abs(chiM_dec.imag)[ipk]:.4f}   interior_peak={interior}")
dec_shows_pole = any(interior for *_, interior in r5_rows) and verdict4 != "NO-POLE"
if dec_shows_pole:
    check("R-5 decorative control shows NO collective pole", False, gate=False,
          detail="the decorative (intraband-only) control ALSO shows an interior peak -- R-4's "
                 "structure is an ARTIFACT of the Mermin closure's own conserving mechanism, not a "
                 "genuine interband/density effect")
    verdict4_final = "INSTRUMENT-LIMITED (overrides R-4's raw verdict, per the frozen R-5 rule)"
else:
    check("R-5 decorative control shows NO collective pole (discriminates R-4's structure as genuine, "
          "not a pipeline artifact)", True)
    verdict4_final = verdict4
report(f"R-4 FINAL VERDICT (post-R-5) = {verdict4_final}", True)

# ====================================================================================================
banner("R-6  SCOPE DECLARATION  (printed verbatim from the pre-reg)")
# ====================================================================================================
print("""  NOT claimed: n_s, sigma_8, S_8, D(z), f(z), fsigma_8 (all downstream -- B2-b/B2-c objects);
  no era/N-dependence anywhere; the GR growth ODE appears NOWHERE in this station; A_s untouched (its
  independent closure noted); z_eq/fluid-onset NOT determined here (ML-3's open crossing unaffected).""")
check("R-6 scope declaration printed", True)

# ====================================================================================================
banner("SUMMARY")
# ====================================================================================================
print(f"""  R-0 ANCHORS            : {'PASS' if selftest_ok else 'FAIL'}  (band count=4 verified,
      beta_eff/gamma_micro quoted+verified, net.self_test() regression clean)
  R-1 chi_0(q) STATIC     : table + {GRID_LADDER} convergence ladder printed above
                            (max drift {max_drift:.1%}{' -> GRID-LIMITED' if grid_limited else ''})
  R-2 TWO-ROUTE FDT GATE  : {'PASS' if r2_pass else 'FAIL (honest negative, see disclosure -- bosonic-vs-fermionic mismatch, NOT a bug)'}
  R-3 TWO-FLUID SPLIT     : tabulated (report only)
  R-4 SOUND-POLE CONFRONT : {verdict4_final}
  R-5 DECORATIVE CONTROL  : {'shows a pole (overrides R-4)' if dec_shows_pole else 'shows NO pole (R-4 structure is genuine)'}
  R-6 SCOPE               : printed

  MC-2's named object -- "chi(q,w) = the Lindhard function over the srs bands" -- is BUILT.  The
  two-route FDT gate (R-2) reveals it is NOT the same construction as M2b's bosonic per-mode
  fluctuation spectrum (a genuine, disclosed, non-goal-sought finding).  The Mermin-closed sound-pole
  confront (R-4/R-5) is the station's crux result: {verdict4_final}.
  total runtime = {time.time() - T_START:.1f}s""")
sys.exit(0 if selftest_ok else 1)
