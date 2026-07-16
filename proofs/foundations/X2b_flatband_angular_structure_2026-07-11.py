#!/usr/bin/env python3
"""
proofs/foundations/X2b_flatband_angular_structure_2026-07-11.py

X.2-b -- THE FLAT-BAND ANGULAR STRUCTURE (booked in
internal research notes, Station X.2-b, lines 304-321).
Read the return FIRST: this is the "prerequisite check" the return orders BEFORE Station X.2-a
(the flat/cone rho(beta) crossing).  EXTENDS the_net.py's band_quantum_metric (NOT edited);
reuses M2b_fluctuation_spectrum_2026-07-07.py's own Fibonacci-sphere direction generator
(lines 70-76 there) verbatim.  No cosmological number enters anywhere in this file.

QUESTION (frozen by the return, verbatim): the four-direction check in the return's Sec.2
finding #1 found the m=0 flat band's dispersion coefficient A(n_hat) = E(k)/|k|^2 (fixed small
|k|, E measured from the node via band_quantum_metric's own E channel) is EXACTLY ZERO along the
axis [100] and face [110] directions, and O(1) (~3.29, ~1.50) along body [111] and a generic
direction.  Is the "exactly flat" direction set a MEASURE-ZERO set on the sphere of directions
(isolated high-symmetry points/lines -- Station X.2-a's solid-angle-averaged curvature premise is
sound) or does it extend over a FINITE SOLID ANGLE (a nodal surface -- X.2-a needs a
non-3D-quadratic redesign)?

DUAL-OUTCOME VERDICT (frozen by the return, quoted in SUMMARY below):
  (a) measure-zero flat lines -> proceed with X.2-a's solid-angle-averaged curvature as planned.
  (b) finite-solid-angle flat surface -> X.2-a must be redesigned around the true (non-3D-
      quadratic) density of states.

Goal-seek risk: LOW (the return's own words) -- a direct diagonalization sweep, nothing to tune.
No convention is invented here beyond what the return already froze: the direction generator
(M2b's), the object (band_quantum_metric's E channel), and the two-way verdict split.
"""
import os
import sys
import math

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import the_net as net  # noqa: E402
import srs  # noqa: E402

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


# M2b's EXACT Fibonacci-sphere direction generator, reused verbatim
# (proofs/foundations/M2b_fluctuation_spectrum_2026-07-07.py:70-76).
def fib_dir(i, ndir):
    z = 1 - 2 * (i + 0.5) / ndir
    phi = math.pi * (3 - math.sqrt(5)) * i
    r = math.sqrt(max(0.0, 1 - z * z))
    return np.array([r * math.cos(phi), r * math.sin(phi), z])


def A_of(nhat, r):
    """A(n_hat) = E(k)/|k|^2 at k = nhat*r (nhat already a unit vector), via band_quantum_metric's
    own E channel (the_net.py:509-531, 3rd return value = E_rel_node)."""
    _, _, E = net.band_quantum_metric(nhat * r)
    return E / r ** 2


# ===========================================================================
banner("X2b-0  SANITY: reproduce the return's own 4-direction check (Sec.2 finding #1)")
# ===========================================================================
named = [("axis [100]", [1, 0, 0]), ("face [110]", [1, 1, 0]),
         ("body [111]", [1, 1, 1]), ("generic", [2, 1, 0.5])]
R_SWEEP = 1e-2   # the return's own smallest-radius setting for the clean values
print(f"    A(n_hat) = E/r^2 at r={R_SWEEP}, normalized directions (return quotes 0, 0, 3.29, 1.50):")
sanity = {}
for nm, nhat in named:
    n = np.array(nhat, float); n /= np.linalg.norm(n)
    A = A_of(n, R_SWEEP)
    sanity[nm] = A
    print(f"      {nm:11s}: A = {A:.6f}")
check("X2b-0 reproduces the return's own 4-point check (axis/face ~0 exactly; body ~3.29, generic ~1.50)",
      abs(sanity["axis [100]"]) < 1e-8 and abs(sanity["face [110]"]) < 1e-8
      and abs(sanity["body [111]"] - 3.29) < 0.05 and abs(sanity["generic"] - 1.50) < 0.05,
      detail=str({k: round(v, 4) for k, v in sanity.items()}))

# r^2-scaling sanity (confirms r=1e-2 is deep in the quadratic-in-k regime, ML3-A's own convergence
# check applied to E/|k|^2 rather than tr_g*r^2).
print(f"    r^2-scaling check (A should be ~r-independent) for body/generic across r=(1e-2,3e-2,1e-1):")
r_scale_ok = True
for nm in ("body [111]", "generic"):
    n = np.array(dict(named)[nm], float); n /= np.linalg.norm(n)
    vals = [A_of(n, r) for r in (1e-2, 3e-2, 1e-1)]
    spread = max(vals) / min(vals)
    print(f"      {nm:11s}: A(r) = {[round(v,3) for v in vals]}  (spread {spread:.2f}x)")
    if spread > 1.3:
        r_scale_ok = False
check("X2b-0b A(n_hat) is r-independent to <30% out to r=0.1 (confirms the quadratic regime)", r_scale_ok)

# ===========================================================================
banner("X2b-1  DENSE FIBONACCI-SPHERE SWEEP of A(n_hat), increasing sample density")
# ===========================================================================
NDIRS = [2000, 8000, 32000]
sweeps = {}
for ndir in NDIRS:
    A = np.array([A_of(fib_dir(i, ndir), R_SWEEP) for i in range(ndir)])
    sweeps[ndir] = A
    frac_exact = np.mean(np.abs(A) < 1e-6)
    print(f"    ndir={ndir:6d}: A range [{A.min():.4f}, {A.max():.4f}], mean={A.mean():.4f}, "
          f"median={np.median(A):.4f}, frac(|A|<1e-6)={frac_exact:.6f} "
          f"({int(round(frac_exact*ndir))} of {ndir} points)")

check("X2b-1 essentially NO quasi-random sample point lands within 1e-6 of exactly flat "
      "(consistent with a lower-dimensional -- not area-filling -- zero set)",
      all(np.mean(np.abs(sweeps[n]) < 1e-6) < 5.0 / n for n in NDIRS),
      detail="a genuine 2-D nodal patch would instead show a COUNT of near-zero hits growing "
             "proportional to ndir (a stable area fraction), not staying at ~0 hits")

# ===========================================================================
banner("X2b-2  MEASURE-ZERO vs FINITE-SOLID-ANGLE: epsilon-scaling of the near-flat fraction")
# ===========================================================================
# The discriminating test: fraction(eps) = P(|A(n_hat)| < eps) over the sphere.
#   - finite-solid-angle nodal PATCH (2-D, positive area)  => fraction(eps) -> a POSITIVE CONSTANT
#     as eps shrinks (the patch's own area fraction), independent of eps once eps is below the
#     typical A-scale elsewhere.
#   - measure-zero set (isolated points, 0-D, or lines/great-circles, 1-D)  => fraction(eps) -> 0
#     as eps shrinks (an eps-neighborhood of a lower-dimensional set has vanishing area); the
#     LOG-LOG SLOPE of fraction(eps) vs eps distinguishes points (slope~2) from lines (slope~1).
A_dense = sweeps[max(NDIRS)]
N_DENSE = max(NDIRS)
epsilons = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
fracs = [float(np.mean(np.abs(A_dense) < eps)) for eps in epsilons]
for eps, f in zip(epsilons, fracs):
    print(f"    eps={eps:<10.5g}: fraction(|A|<eps) = {f:.6f}  ({int(round(f*N_DENSE))} of {N_DENSE} pts)")

valid = [(e, f) for e, f in zip(epsilons, fracs) if f > 0]
if len(valid) >= 3:
    le = np.log([e for e, _ in valid])
    lf = np.log([f for _, f in valid])
    slope, intercept = np.polyfit(le, lf, 1)
else:
    slope, intercept = float("nan"), float("nan")
print(f"    log-log slope of fraction(eps) vs eps over the informative range = {slope:.3f} "
      f"(interpretation: ~2 => isolated points, ~1 => lines/great-circles, ~0/plateau => finite "
      f"solid angle / nodal surface)")
plateaued = fracs[-1] > 0 and (fracs[-1] / max(fracs[0], 1e-30)) > 0.3   # stays same order across ~5 decades
vanishes = fracs[-1] == 0.0 or slope > 0.4
check("X2b-2 fraction(eps) VANISHES (does not plateau) as eps shrinks over 5 decades "
      "=> the flat set is MEASURE-ZERO, not a finite-solid-angle patch",
      vanishes and not plateaued,
      detail=f"slope={slope:.2f}, frac(eps=1)={fracs[0]:.4f}, frac(eps=1e-5)={fracs[-1]:.6f}")

# ===========================================================================
banner("X2b-3  TARGETED SCAN: is z=0 an entire flat GREAT CIRCLE, and are x=0/y=0 its images?")
# ===========================================================================
# The return's own two flat hits (axis[100]=(1,0,0), face[110]=(1,1,0)) BOTH have zero
# z-component. Test directly whether the WHOLE z=0 equator is flat (a genuine 1-D line -- still
# measure-zero on S^2), vs only the two special high-symmetry points on it, and whether cubic
# symmetry extends this to the x=0 and y=0 great circles too.
angles = np.linspace(0, 2 * math.pi, 73, endpoint=False)  # every 5 degrees
plane_results = {}
for label, mk in [("z=0", lambda p: np.array([math.cos(p), math.sin(p), 0.0])),
                   ("x=0", lambda p: np.array([0.0, math.cos(p), math.sin(p)])),
                   ("y=0", lambda p: np.array([math.cos(p), 0.0, math.sin(p)]))]:
    vals = np.array([A_of(mk(p), R_SWEEP) for p in angles])
    plane_results[label] = vals
    print(f"    {label} great circle (73 pts, 5 deg apart): A range [{vals.min():.2e}, {vals.max():.2e}], "
          f"max|A| = {np.max(np.abs(vals)):.2e}")
check("X2b-3 ALL THREE coordinate great circles (z=0, x=0, y=0) are flat EVERYWHERE on the "
      "circle, not just at the two special points found in the return's finding #1",
      all(np.max(np.abs(v)) < 1e-8 for v in plane_results.values()))

# cross-check: the body-diagonal plane x=y (which also passes through face[110]'s companion point
# (1,1,1)/sqrt(3)... no: check a plane that shares the [110] point but is NOT one of the 3
# coordinate planes, to confirm flatness does NOT extend to it (i.e. it's specifically the
# coordinate planes, not "any plane through a flat point").
angles2 = np.linspace(0, 2 * math.pi, 73, endpoint=False)
xy_plane_vals = []
for p in angles2:
    # plane spanned by (1,1,0)/sqrt(2) and (0,0,1): n = cos(p)*(1,1,0)/sqrt(2) + sin(p)*(0,0,1)
    n = math.cos(p) * np.array([1, 1, 0.0]) / math.sqrt(2) + math.sin(p) * np.array([0, 0, 1.0])
    xy_plane_vals.append(A_of(n, R_SWEEP))
xy_plane_vals = np.array(xy_plane_vals)
print(f"    control: the (110)/(001) plane through the SAME flat point [110]: A range "
      f"[{xy_plane_vals.min():.4f}, {xy_plane_vals.max():.4f}] (should NOT be uniformly flat)")
check("X2b-3b CONTROL: a plane merely PASSING THROUGH a flat point is NOT itself uniformly flat "
      "(flatness is specific to the 3 coordinate great circles, not generic)",
      np.max(np.abs(xy_plane_vals)) > 0.5)

# ===========================================================================
banner("SUMMARY / VERDICT")
# ===========================================================================
verdict = "MEASURE-ZERO" if (vanishes and not plateaued) else "FINITE-SOLID-ANGLE"
print(f"""    OUTCOME: the m=0 flat band's "exactly flat" direction set (A(n_hat)=E/|k|^2 == 0) is
    {verdict}.
    Evidence: (i) a dense (up to {max(NDIRS)}-point) quasi-uniform Fibonacci-sphere sample of A(n_hat)
      never lands within 1e-6 of exact flatness (X2b-1) -- a positive-area patch would instead show
      a hit-count growing proportional to sample size.
    (ii) fraction(|A|<eps) VANISHES as eps shrinks over 5 decades (X2b-2, log-log slope {slope:.2f}),
      not plateauing to a constant -- the signature of a lower-dimensional (points/lines), not
      area-filling, zero set.
    (iii) direct scan (X2b-3) shows the ENTIRE z=0 (and by cubic symmetry x=0, y=0) great circle is
      EXACTLY flat at every one of 73 sampled angles (not just the 2 special points found in the
      return's finding #1) -- so the zero set is (at least) the union of the 3 coordinate great
      circles: a 1-DIMENSIONAL set (lines), still MEASURE-ZERO on the 2-sphere of directions.
    (iv) CONTROL (X2b-3b): a generic plane merely passing through one flat point ([110]) is NOT
      itself flat -- confirms flatness is a property of the specific coordinate great circles, not
      a generic feature near any flat point (rules out "the patch is bigger than it looks").
    VERDICT PER THE RETURN'S OWN FROZEN SPLIT: (a) measure-zero flat lines -- X.2-a's premise
      (solid-angle-averaged curvature after excluding a measure-zero set) is SOUND; proceed with
      X.2-a as designed. (No z, no era exponent, no Planck number entered anywhere above.)
    NAMED INCOMPLETENESS (out of X.2-b's scope, flagged for X.2-a): the flat set is not merely
      "isolated" -- it is 3 great circles (mutually intersecting at the 6 axis points), a specific,
      non-generic symmetry structure (plausibly the 3 mirror planes of the point group fixing the
      Gamma-point triple degeneracy). X.2-a's solid-angle average must integrate OVER these lines
      (measure zero, so a naive Monte-Carlo/Fibonacci-sphere average is unaffected in the eps->0
      limit) but should be aware the integrand A(n_hat) is not smooth in the naive sense near them
      (it vanishes on a positive-codimension set, i.e. a genuine but thin locus, not a generic
      isolated accident) -- worth a support-shrinking check (does the solid-angle average converge
      as sample density increases, or does the divergent BEHAVIOR NEAR the lines pull the average
      away from its infinite-density limit?) before X.2-a reports a final number.""")
print("RESULT:", "ALL CHECKS PASS" if ok_all else "A CHECK FAILED -- inspect above")
sys.exit(0 if ok_all else 1)
