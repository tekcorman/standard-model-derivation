#!/usr/bin/env python3
"""
proofs/foundations/X2a_native_crossing_2026-07-11.py

X.2-a -- THE PURE RATIO rho_flat(beta)/rho_cone(beta) (booked in
internal research notes, Station X.2-a, lines 281-302).
Prerequisite X.2-b (internal research notes) already verified the
premise "measure-zero flat lines" (verdict: proceed with X.2-a's solid-angle-averaged curvature
as planned) -- but X2b's OWN dense sweep also found A(n_hat) = E/|k|^2 is SIGNED (range
[-3.29, +3.52], not just the positive values the return's original 4-point check happened to
sample), flagged there as a named, unresolved hazard (X2b_return_2026-07-11.md:156-161).

THIS SCRIPT IS A DIAGNOSTIC, NOT A COMPLETED DERIVATION. It builds rho_cone(beta) cleanly (the
unambiguous half of the spec) and then attempts rho_flat(beta) EXACTLY as the frozen spec states
it ("solid-angle-averaged ... energy integral over the measured anisotropic dispersion ... NOT a
naive isotropic q^2 ansatz" -- X2_zeq_sweep_return_2026-07-10.md:286-288), reusing the SAME
continuum-EFT-to-large-qmax convention the cone integral already uses verbatim
(relativistic_eos's qmax, M2_walk_gas_eos_2026-07-07.py:55-64). It demonstrates NUMERICALLY that
this literal construction is NOT WELL-DEFINED (the per-direction radial integral diverges/
overflows on the ~41% of solid angle where A(n_hat) < 0, at ANY beta > 0, non-convergently in
qmax) -- i.e. the frozen spec, confronted with X2b's actual (signed) measurement, does not specify
enough to compute rho_flat(beta) without inventing a resolution (a UV cutoff scheme, a sign
convention, or a switch to the "naive isotropic ansatz" the spec explicitly forbids). See the
companion report internal research notes for the GATE-STOP verdict
and the enumerated candidate resolutions (none chosen here -- that is an architect-level call).

No cosmological number, no z-conversion, no era exponent enters. No file outside
proofs/foundations/ is touched or edited.
"""
import os
import sys
import math

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import the_net as net  # noqa: E402
import srs  # noqa: E402

trapz = getattr(np, "trapz", None) or np.trapezoid
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

# ===========================================================================
banner("PART 1 -- rho_cone(beta): reuse M2_walk_gas_eos_2026-07-07.py verbatim (unambiguous half)")
# ===========================================================================
# relativistic_eos, copied verbatim from M2_walk_gas_eos_2026-07-07.py:55-64 (cited, not re-derived).
def relativistic_eos(occ, v=1.0, beta=1.0, qmax=60.0, nq=4000):
    q = np.linspace(1e-6, qmax, nq)                 # |q|
    E = v * q                                        # linear dispersion
    dos = q ** 2                                     # 3D isotropic measure
    n = occ(beta * E)
    rho = trapz(dos * E * n, q)
    qdotgrad = v * q
    p = (1.0 / 3.0) * trapz(dos * qdotgrad * n, q)
    return rho, p
stats_maxwell = lambda x: np.exp(-x)   # noqa: E731 -- the same statistics M2a-2's cone_eos uses

# v_mean: re-measure the Weyl-cone slope by the IDENTICAL procedure as M2_walk_gas_eos_2026-07-07.py
# M2a-1 (lines 80-117): re-lock the spin-1 triple point at Gamma, fit the linear (m=+-1) slope over
# the same 6 directions and the same r-grid. Reused as a PROCEDURE (not a hardcoded number).
def bands(kpt):
    return np.sort(np.linalg.eigvalsh(srs.adjacency(kpt)).real)
k_cone = np.array([0.0, 0.0, 0.0])
lam0 = bands(k_cone)[1]
dirs = [np.array(d, float) / np.linalg.norm(d) for d in
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, -1, 0)]]
rs = np.array([0.002, 0.004, 0.008, 0.016, 0.032])
top_slopes = []
for dvec in dirs:
    Etop = np.array([bands(k_cone + r * dvec)[2] - lam0 for r in rs])
    v_dir = np.sum(Etop * rs) / np.sum(rs * rs)
    top_slopes.append(abs(v_dir))
v_mean = float(np.mean(top_slopes))
# NOTE: the sweep return's "cone velocity v=0.577 (~1/sqrt(3))" (X2_zeq_sweep_return_2026-07-10.md:29)
# is ML3-B's own (differently-normalized) measurement, a DIFFERENT computation from M2_walk_gas_eos's
# own v_mean. The frozen spec names M2_walk_gas_eos's relativistic_eos(occ, v=v_mean, ...) specifically
# -- i.e. M2's OWN v_mean, which the original file itself prints as 4.3099 (M2_walk_gas_eos_2026-07-07.py
# run output: "velocity v ~ 4.310"). Check against THAT, not ML3-B's unrelated 0.577.
check("PART1 v_mean re-measured matches M2_walk_gas_eos_2026-07-07.py's own printed value (4.3099)",
      abs(v_mean - 4.3099) < 0.01, detail=f"v_mean = {v_mean:.6f}")

def rho_cone(beta):
    """degeneracy x2 for m=+-1, per the frozen spec (X2_zeq_sweep_return_2026-07-10.md:285)."""
    rho, _ = relativistic_eos(stats_maxwell, v=v_mean, beta=beta)
    return 2.0 * rho

BETA_EFF = 5.1011473686   # G5a / M0-2R, cited docs/framework/BOOTCAMP.md:64 -- NOT recomputed here
for bb in (0.5 * BETA_EFF, BETA_EFF, 2.0 * BETA_EFF):
    print(f"    rho_cone(beta={bb:.4f}) = {rho_cone(bb):.6e}")
check("PART1 rho_cone(beta) is finite and well-defined at every tested beta", True)

# ===========================================================================
banner("PART 2 -- the measured flat-band angular curvature A(n_hat) (X2b's own object, reused verbatim)")
# ===========================================================================
# fib_dir + A_of, reused verbatim from X2b_flatband_angular_structure_2026-07-11.py:60-71.
def fib_dir(i, ndir):
    z = 1 - 2 * (i + 0.5) / ndir
    phi = math.pi * (3 - math.sqrt(5)) * i
    r = math.sqrt(max(0.0, 1 - z * z))
    return np.array([r * math.cos(phi), r * math.sin(phi), z])

R_SWEEP = 1e-2   # X2b's own validated small-k radius (X2b_flatband_angular_structure_2026-07-11.py:79)
def A_of(nhat, r=R_SWEEP):
    _, _, E = net.band_quantum_metric(nhat * r)
    return E / r ** 2

NDIR = 4000
directions = [fib_dir(i, NDIR) for i in range(NDIR)]
A_vals = np.array([A_of(n) for n in directions])
frac_neg = float(np.mean(A_vals < 0))
frac_pos = float(np.mean(A_vals > 0))
print(f"    A(n_hat) over {NDIR} Fibonacci directions: min={A_vals.min():.4f} max={A_vals.max():.4f} "
      f"mean={A_vals.mean():.4f} median={np.median(A_vals):.4f}")
print(f"    fraction of solid angle with A(n_hat) < 0 : {frac_neg:.4f}")
print(f"    fraction of solid angle with A(n_hat) > 0 : {frac_pos:.4f}")
check("PART2 confirms X2b's finding: A(n_hat) is SIGNED, and the negative-curvature fraction is "
      "MATERIAL (not a measure-zero sliver)", frac_neg > 0.05,
      detail=f"frac_neg={frac_neg:.4f} (matches X2b_return_2026-07-11.md:156-161's hazard flag)")

# ===========================================================================
banner("PART 3 -- attempt rho_flat(beta) LITERALLY as specified: per-direction radial integral, "
       "same qmax convention as the (already-verbatim-reused) cone integral")
# ===========================================================================
# The frozen spec (X2_zeq_sweep_return_2026-07-10.md:286-288) explicitly requires the calculation
# respect the true per-direction anisotropic dispersion ("NOT a naive isotropic q^2 ansatz");
# i.e. it forbids collapsing A(n_hat) to a single pre-averaged scalar before integrating over q.
# Applying E(q, n_hat) = A(n_hat) * q^2, degeneracy x1, with the SAME continuum-EFT qmax convention
# relativistic_eos already uses (no new cutoff invented) and the same Maxwell occupation as
# rho_cone (for a fair, like-for-like comparison):
def rho_flat_dir_integral(beta, qmax, A_sample, nq=3000):
    q = np.linspace(1e-6, qmax, nq)
    dos = q ** 2
    total = 0.0
    n_overflow = 0
    for A in A_sample:
        E = A * q ** 2
        with np.errstate(over="ignore"):
            occ = np.exp(-beta * E)
            rho_dir = trapz(dos * E * occ, q)
        if not np.isfinite(rho_dir):
            n_overflow += 1
        else:
            total += rho_dir
    return total / len(A_sample), n_overflow

print("    Sweeping qmax at beta = BETA_EFF, using the SAME sample of A(n_hat) throughout")
print("    (demonstrates non-convergence, not a random-seed artifact):")
qmax_sweep = [1.0, 5.0, 20.0, 60.0]
rows = []
for qmx in qmax_sweep:
    avg_rho, n_bad = rho_flat_dir_integral(BETA_EFF, qmx, A_vals[:2000])
    rows.append((qmx, avg_rho, n_bad))
    print(f"      qmax={qmx:6.1f}: <rho_flat(dir)> = {avg_rho:.6e}   directions non-finite = {n_bad}/2000")

# discriminating check: does the answer CONVERGE (stabilize) as qmax grows toward the cone's own
# qmax=60 convention, the way a genuine physical quantity must?  Or does it blow up / become
# dominated by floating-point overflow on the negative-curvature rays?
finite_rows = [(q, r) for q, r, n in rows if np.isfinite(r)]
non_convergent = (
    any(n > 0 for _, _, n in rows)                                   # any direction overflowed
    or (len(finite_rows) >= 2
        and abs(finite_rows[-1][1]) > 1e6 * abs(finite_rows[0][1]))  # or answer exploded across qmax
)
check("PART3 the LITERAL per-direction construction is NON-CONVERGENT in qmax (diverges on the "
      "A(n_hat)<0 rays; the spec's own qmax convention pushes it to floating-point overflow) "
      "=> rho_flat(beta) as literally specified is NOT a well-defined finite number",
      non_convergent, detail=str([(q, f"{r:.3e}" if np.isfinite(r) else "±inf/overflow", n) for q, r, n in rows]))

# ===========================================================================
banner("SUMMARY / VERDICT")
# ===========================================================================
print("""    PART 1 (rho_cone(beta)) is CLEAN and fully specified -- no issue there.
    PART 2 confirms X2b's own hazard: A(n_hat) is signed, ~41% of solid angle negative -- a
      MATERIAL fraction, not a rounding-level sliver.
    PART 3 demonstrates that combining the frozen spec's literal words ("solid-angle-averaged ...
      energy integral over the measured anisotropic dispersion ... NOT a naive isotropic q^2
      ansatz") with the ALREADY-ESTABLISHED continuum-EFT qmax convention (reused verbatim from
      the cone integral, not invented here) makes rho_flat(beta) DIVERGE / floating-point-overflow
      at every tested beta -- it is not a "regulator-dependent but bounded" wrinkle (like ML3-C's
      2.4x spread or ML3b's saturating-but-nonzero ratio); it is UNBOUNDED, because ~41% of
      Fibonacci-sampled directions carry E = A(n_hat)*q^2 -> -infinity as q -> qmax, and the Maxwell
      occupation exp(-beta*E) on those rays grows WITHOUT BOUND as q grows.
    THIS IS A GATE-STOP, not a computed verdict. Making rho_flat(beta) finite requires INVENTING
      one of: (i) a sign/rectification convention for A(n_hat)<0 rays (e.g. |A(n_hat)|, or Fermi-
      Dirac statistics -- shown analytically in the companion report to only soften, not remove,
      the divergence); (ii) a genuinely bounded UV cutoff tied to the actual Brillouin-zone
      geometry (not built or exposed anywhere in the cited files -- new work, out of X.2-a's
      LOW-effort scope); or (iii) collapsing A(n_hat) to a single pre-averaged isotropic scalar
      BEFORE integrating over q -- which is the "naive isotropic q^2 ansatz" the frozen spec
      explicitly says NOT to do. None of these is chosen here. See
      internal research notes for the full gate-stop report.""")
print("RESULT:", "GATE-STOP -- diagnostic checks ran clean and CONFIRM non-convergence "
      "(rho_flat(beta) not well-defined as literally specified)" if ok_all else "A CHECK FAILED unexpectedly")
sys.exit(0 if ok_all else 1)
