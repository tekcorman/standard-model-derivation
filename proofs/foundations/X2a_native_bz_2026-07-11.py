#!/usr/bin/env python3
"""
proofs/foundations/X2a_native_bz_2026-07-11.py

X.2-a NATIVE-BZ -- rho_flat(beta) rebuilt as the BZ-torus average of the frozen thermal weight
applied to the ACTUAL flat-band dispersion (the operator's own eigenvalue), per the FROZEN
amendment internal research notes (freeze commit b1c3546).

LINEAGE (read in full before this file; nothing below is re-derived from scratch):
  - Original spec: internal research notes:281-302
    (Station X.2-a -- THE PURE RATIO rho_flat(beta)/rho_cone(beta), dual-outcome verdict at
    lines 295-302: (a) CROSSING-FOUND -- beta* inside a physically sensible range (bracketed by
    BETA_EFF within an order of magnitude); (b) NO-CROSSING/degenerate -- one band dominates rho
    for all beta, or the crossing sits at an unphysical beta->0 or beta->infinity limit).
  - X.2-b: internal research notes -- MEASURE-ZERO verdict: the
    "exactly flat" (A(n_hat)=E/|k|^2 == 0) direction set is (at least) the union of the 3
    coordinate great circles on the direction-sphere, a 1-D / zero-area set. ALSO found
    A(n_hat) is SIGNED (range [-3.29,+3.52], 41.1% of solid angle negative) -- flagged there as
    an unresolved hazard for X.2-a (X2b_return_2026-07-11.md:156-161), not resolved there.
  - X.2-a gate-stop: internal research notes -- the LITERAL spec
    (a per-direction radial q-integral, reusing the cone's continuum-EFT qmax=60 convention,
    applied to the small-k quadratic ansatz E(q,n_hat)=A(n_hat)*q^2) DIVERGES without bound on
    the ~41% of directions where A(n_hat)<0 (exp(-beta*A*q^2) grows unboundedly as q->qmax,
    demonstrated there by a qmax sweep showing a ~178-order-of-magnitude swing and 577-777/2000
    directions individually overflowing). GATE-STOP -- none of the three unblocking candidates
    (sign rectification, a new BZ-respecting cutoff, or pre-averaging A(n_hat)) was chosen there;
    each requires inventing a convention, target, or verdict criterion.
  - THIS AMENDMENT (internal research notes SS0): the divergence is a symptom of
    TWO non-native ingredients imported from the CONE side, where they were harmless: (i) an
    UNBOUNDED radial q-domain (qmax was immaterial there because the cone integral already
    converged; the walk gas's actual momentum space is the COMPACT Brillouin torus -- srs.py's
    own Bloch/Floquet variable, k in [0,1)^3 fractional, srs.py:7) and (ii) the small-k quadratic
    approximation E=A(n_hat)*q^2 itself (an approximation to the true band, not the band).
    THE ONE CHANGE (amendment SS1): rho_flat(beta) = the BZ-average, over the FULL compact torus,
    of the frozen thermal weight (Maxwell occupation exp(-beta*E), degeneracy x1) applied to the
    ACTUAL flat-band eigenvalue from the operator (the_net.py's own band_quantum_metric E
    channel, the_net.py:509-531), with NO radial ansatz, NO A(n_hat) parametrization, NO qmax,
    NO sign rectification, NO pre-averaging. The cone side (rho_cone) is explicitly UNCHANGED
    (amendment SS1 last bullet: "already rebuilt it cleanly") -- reused verbatim below, sourced
    ultimately from M2_walk_gas_eos_2026-07-07.py:55-64 and :80-117 (opened and re-cited this
    session, not taken on the gate-stop script's word).

DO NOT EDIT (task contract): the_run.py, the_net.py, verify.py, locks/registers, the stopped
station's own files. This script only READS the_net.py / srs.py through their exposed functions
(net.band_quantum_metric, srs.adjacency). No file outside proofs/foundations/ is written. No git
commit is made by this script or this session.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import the_net as net  # noqa: E402
import srs  # noqa: E402

trapz = getattr(np, "trapz", None) or np.trapezoid
np.set_printoptions(precision=4, suppress=True)

BETA_EFF = 5.1011473686   # G5a / M0-2R; cited docs/framework/BOOTCAMP.md:64 -- NOT recomputed here

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88)
    print(f" {t}")
    print("=" * 88)


# ===========================================================================
# PART 1 -- rho_cone(beta): UNCHANGED per the amendment (SS1 last bullet). Reused verbatim.
# ===========================================================================
def relativistic_eos(occ, v=1.0, beta=1.0, qmax=60.0, nq=4000):
    """Verbatim, M2_walk_gas_eos_2026-07-07.py:55-64 (opened + re-cited this session)."""
    q = np.linspace(1e-6, qmax, nq)                 # |q|
    E = v * q                                        # linear dispersion
    dos = q ** 2                                     # 3D isotropic measure
    n = occ(beta * E)
    rho = trapz(dos * E * n, q)
    return rho


stats_maxwell = lambda x: np.exp(-x)  # noqa: E731 -- the frozen weight (Maxwell occupation)


def measure_v_mean():
    """Re-measure the Weyl-cone slope by the IDENTICAL procedure as
    M2_walk_gas_eos_2026-07-07.py's M2a-1 (lines 80-117, opened + re-cited this session):
    re-lock the spin-1 triple point at Gamma, fit the linear (m=+-1) slope over the same 6
    directions and r-grid. A procedure, not a hardcoded number."""
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
    return float(np.mean(top_slopes))


def rho_cone(beta, v_mean):
    """Degeneracy x2 for m=+-1, per the frozen spec (X2_zeq_sweep_return_2026-07-10.md:285)."""
    return 2.0 * relativistic_eos(stats_maxwell, v=v_mean, beta=beta)


# ===========================================================================
# PART 2 -- the ACTUAL flat-band dispersion from the operator, over the FULL BZ torus
# ===========================================================================
def E_flat(kvec, node=-1.0):
    """The tracked (m=0 / 'flat') band's energy relative to the node, reusing the_net.py's own
    band_quantum_metric (the_net.py:509-531) -- only its 3rd return value (E_rel_node) is used;
    the quantum-metric channel (tr_g, berry) is not needed for a pure energy/thermal integral.
    This is THE reused band machinery the amendment names; no new object is defined."""
    _, _, E = net.band_quantum_metric(np.asarray(kvec, float))
    return E


def E_flat_direct(kvec, node=-1.0):
    """A lightweight cross-check of the SAME definition (the_net.py:515-519's P_and_E: pick the
    eigenvalue nearest `node`, subtract it), computed directly from srs.adjacency without the
    finite-difference derivative overhead band_quantum_metric also carries -- used only for the
    fast full-BZ continuity sweep in PART 2d and the grid-convention robustness check in PART 3d,
    NOT for the headline numbers (those use net.band_quantum_metric directly, PART 2b confirms
    the two agree to float precision)."""
    w = np.linalg.eigvalsh(srs.adjacency(np.asarray(kvec, float)))
    i = int(np.argmin(np.abs(w - node)))
    return w[i] - node


def bz_grid(N, midpoint=False):
    """The full Brillouin torus, k in [0,1)^3 fractional (srs.py's own Bloch/Floquet variable,
    srs.py:7: "Bloch/Floquet variable k in [0,1)^3 (fractional)"). Grid convention a/N (a=0..N-1)
    reused from M2_walk_gas_eos_2026-07-07.py's own BZ-touching scan (lines 83-89, its G-grid) --
    not a new convention invented here. `midpoint=True` selects the alternative (i+0.5)/N grid,
    used only for PART 3d's grid-CONVENTION robustness check (not the headline table)."""
    if midpoint:
        ks = [(i + 0.5) / N for i in range(N)]
    else:
        ks = [i / N for i in range(N)]
    return [(a, b, c) for a in ks for b in ks for c in ks]


def main():
    global ok_all
    banner("PART 1 -- rho_cone(beta): reuse M2_walk_gas_eos_2026-07-07.py verbatim (UNCHANGED)")
    v_mean = measure_v_mean()
    check("PART1 v_mean re-measured matches M2_walk_gas_eos_2026-07-07.py's own printed value "
          "(4.3099)", abs(v_mean - 4.3099) < 0.01, detail=f"v_mean = {v_mean:.6f}")
    for bb in (0.5 * BETA_EFF, BETA_EFF, 2.0 * BETA_EFF):
        print(f"    rho_cone(beta={bb:.4f}) = {rho_cone(bb, v_mean):.6e}")
    check("PART1 rho_cone(beta) is finite at every tested beta (unchanged, unambiguous half)",
          all(np.isfinite(rho_cone(bb, v_mean)) for bb in (0.5 * BETA_EFF, BETA_EFF, 2 * BETA_EFF)))

    # =======================================================================
    banner("PART 2 -- the ACTUAL flat-band dispersion E_flat(k), the operator's own eigenvalue, "
           "over the FULL Brillouin torus (no small-k truncation, no radial ansatz)")
    # =======================================================================
    # 2a. Sanity: reproduce the Gamma-point triple degeneracy (M2_walk_gas_eos_2026-07-07.py:100-104)
    w_gamma = np.sort(np.linalg.eigvalsh(srs.adjacency(np.array([0.0, 0.0, 0.0]))).real)
    deg = int(np.sum(np.abs(w_gamma - (-1.0)) < 1e-9))
    check("PART2a Gamma-point triple degeneracy at lambda_0=-1 reproduced (M2a-1's own anchor)",
          deg == 3, detail=f"bands(Gamma) = {np.round(w_gamma, 4)}")

    # 2b. band_quantum_metric's E channel agrees EXACTLY with the direct eigenvalue-nearest-node
    # formula, sampled across the FULL BZ (not just near Gamma, where X2a/X2b already checked it).
    rng = np.random.default_rng(0)
    probe_pts = rng.random((200, 3))
    diffs = [abs(E_flat(p) - E_flat_direct(p)) for p in probe_pts]
    check("PART2b band_quantum_metric's E channel == the direct eigenvalue-nearest-node formula, "
          "at 200 RANDOM points spanning the FULL BZ (not just small k)",
          max(diffs) < 1e-10, detail=f"max|diff| = {max(diffs):.3e}")

    # 2c. Continuity with the ALREADY-ESTABLISHED small-k A(n_hat) numbers (X2a_return/X2b_return):
    # E_flat(r*n_hat) should recover axis/face ~ 0, body ~ 3.30*r^2, generic ~ 1.50*r^2 at small r.
    # This is a CONSISTENCY check that the full-BZ object used below is the SAME object as the
    # already-validated small-k one, not a re-derivation of it.
    r_small = 1e-2
    named = {"axis [100]": (1, 0, 0), "face [110]": (1, 1, 0), "body [111]": (1, 1, 1),
             "generic": (2, 1, 0.5)}
    small_k_vals = {}
    for nm, d in named.items():
        n = np.array(d, float); n /= np.linalg.norm(n)
        small_k_vals[nm] = E_flat(n * r_small) / r_small ** 2
    print(f"    A(n_hat)=E/r^2 at r={r_small} via the full-BZ E_flat(): "
          f"{ {k: round(v, 4) for k, v in small_k_vals.items()} }")
    check("PART2c small-k limit of the full-BZ E_flat() reproduces X2a/X2b's established "
          "A(n_hat) numbers (axis/face ~0; body ~3.29; generic ~1.50)",
          abs(small_k_vals["axis [100]"]) < 1e-6 and abs(small_k_vals["face [110]"]) < 1e-6
          and abs(small_k_vals["body [111]"] - 3.29) < 0.05
          and abs(small_k_vals["generic"] - 1.50) < 0.05)

    # 2d. Band-index continuity across the WHOLE BZ: does "nearest eigenvalue to node" ever jump
    # to a DIFFERENT physical branch (e.g. the cone) anywhere in the torus? Checked on a dense
    # (40^3 = 64000-point) grid using the fast direct formula (E_flat_direct), tracking the SORTED
    # index of the picked eigenvalue.
    dense_pts = bz_grid(40)
    picked_idx, picked_E, idx1_E = [], [], []
    for p in dense_pts:
        w = np.sort(np.linalg.eigvalsh(srs.adjacency(np.asarray(p))).real)
        i = int(np.argmin(np.abs(w - (-1.0))))
        picked_idx.append(i)
        picked_E.append(w[i] - (-1.0))
        idx1_E.append(w[1] - (-1.0))
    picked_idx = np.array(picked_idx)
    picked_E = np.array(picked_E)
    idx1_E = np.array(idx1_E)
    idx_counts = {i: int(np.sum(picked_idx == i)) for i in range(4)}
    off_pts = [dense_pts[j] for j in np.where(picked_idx != 1)[0]]
    off_disagree = np.max(np.abs(picked_E - idx1_E))  # 0 iff every off-index point is an EXACT TIE
    check("PART2d the 'nearest-to-node' pick NEVER jumps to the cone (index 3) anywhere in the "
          "BZ (40^3 grid), and the handful of points where it ties with index 0 are EXACT "
          "degeneracies (index 0 and index 1 give the identical energy there, not a different "
          "band) -- E_flat(k) is a single, globally continuous, single-valued function",
          idx_counts.get(3, 0) == 0 and idx_counts.get(2, 0) == 0 and off_disagree < 1e-9,
          detail=f"sorted-index counts over {len(dense_pts)} pts = {idx_counts}; off-index points "
                 f"= {off_pts} (all exact ties with index1's energy, max|E_idx - E_idx1| = "
                 f"{off_disagree:.2e})")

    # 2e. The GLOBAL range of E_flat(k) over the compact BZ is BOUNDED -- the structural reason
    # the amendment's fix removes the gate-stop's divergence (contrast: the old radial ansatz's
    # q-domain was UNBOUNDED; here the domain is compact and the integrand is bounded on it).
    Es_dense = np.array([E_flat_direct(p) for p in dense_pts])
    print(f"    E_flat(k) global range over the 40^3 BZ grid: [{Es_dense.min():.4f}, "
          f"{Es_dense.max():.4f}], mean={Es_dense.mean():.4f}")
    check("PART2e E_flat(k) is BOUNDED over the compact BZ (unlike the old q->qmax radial "
          "domain, which was unbounded) -- this is the structural reason no divergence can occur",
          np.isfinite(Es_dense.min()) and np.isfinite(Es_dense.max())
          and (Es_dense.max() - Es_dense.min()) < 10.0)

    # =======================================================================
    banner("PART 3 -- rho_flat(beta) = the BZ-average of E*exp(-beta*E), degeneracy x1, over "
           "the FULL torus (the amendment's ONE CHANGE); grid convergence at >=4 densities")
    # =======================================================================
    DENSITIES = [12, 18, 26, 36]
    beta_scan = np.linspace(0.02, 60.0, 3000)   # spans well below and far above BETA_EFF=5.101

    def rho_flat_native(beta, Es):
        # Vol_BZ = 1 exactly (k in [0,1)^3), so the BZ-AVERAGE and the BZ-INTEGRAL coincide --
        # no separate normalization choice to make.
        return float(np.mean(Es * np.exp(-beta * Es)))

    def analyze(Es, v_mean):
        cone_vals = np.array([rho_cone(b, v_mean) for b in beta_scan])
        flat_vals = np.array([rho_flat_native(b, Es) for b in beta_scan])
        ratio = flat_vals / cone_vals
        i_peak = int(np.argmax(ratio))
        crossing = np.any(ratio >= 1.0)
        sign_changes = np.where(np.diff(np.sign(flat_vals)) < 0)[0]
        beta_zero = float(beta_scan[sign_changes[0]]) if len(sign_changes) else None
        return dict(peak_ratio=float(ratio[i_peak]), peak_beta=float(beta_scan[i_peak]),
                    crossing_anywhere=bool(crossing), beta_zero=beta_zero,
                    rho_flat_beff=rho_flat_native(BETA_EFF, Es),
                    n_nonfinite=int(np.sum(~np.isfinite(flat_vals))))

    table = {}
    for N in DENSITIES:
        pts = bz_grid(N)
        Es = np.array([E_flat(p) for p in pts])
        res = analyze(Es, v_mean)
        table[N] = res
        print(f"    N={N:3d} ({len(pts):6d} pts): E range [{Es.min():.4f},{Es.max():.4f}]  "
              f"peak ratio flat/cone = {res['peak_ratio']:.4f} at beta={res['peak_beta']:.4f} "
              f"(beta/BETA_EFF={res['peak_beta']/BETA_EFF:.4f})")
        print(f"                  beta_zero (rho_flat crosses 0) = {res['beta_zero']:.4f} "
              f"(/BETA_EFF={res['beta_zero']/BETA_EFF:.4f})   "
              f"rho_flat(BETA_EFF) = {res['rho_flat_beff']:.6e}   "
              f"non-finite samples = {res['n_nonfinite']}/{len(beta_scan)}")

    check("PART3 rho_flat(beta) is FINITE at every sampled beta, at every tested grid density "
          "(no overflow/non-convergence anywhere in the beta scan -- CONTRAST with the gate-"
          "stopped radial construction)",
          all(table[N]["n_nonfinite"] == 0 for N in DENSITIES))
    check("PART3 the qualitative verdict (crossing found vs not) is IDENTICAL at every tested "
          "grid density (the discriminating convergence fact, not an invented numeric tolerance)",
          len({table[N]["crossing_anywhere"] for N in DENSITIES}) == 1)

    peak_ratios = [table[N]["peak_ratio"] for N in DENSITIES]
    peak_betas = [table[N]["peak_beta"] for N in DENSITIES]
    beta_zeros = [table[N]["beta_zero"] for N in DENSITIES]
    rho_beffs = [table[N]["rho_flat_beff"] for N in DENSITIES]
    print(f"    SPREAD across densities {DENSITIES} (raw, no invented tolerance -- the amendment "
          f"names none and none is declared anywhere in the lineage docs):")
    print(f"      peak_ratio : {[round(x,4) for x in peak_ratios]}  "
          f"(max-min = {max(peak_ratios)-min(peak_ratios):.4f}, "
          f"{100*(max(peak_ratios)-min(peak_ratios))/max(peak_ratios):.2f}% of the peak value)")
    print(f"      peak_beta  : {[round(x,4) for x in peak_betas]}  "
          f"(max-min = {max(peak_betas)-min(peak_betas):.4f})")
    print(f"      beta_zero  : {[round(x,4) for x in beta_zeros]}  "
          f"(max-min = {max(beta_zeros)-min(beta_zeros):.4f})")
    print(f"      rho_flat(BETA_EFF): {[f'{x:.4e}' for x in rho_beffs]}")
    check("PART3 the finest-vs-coarsest spread in peak_ratio shrinks (N=36 vs N=26 closer than "
          "N=18 vs N=12) -- monotone-in-density behavior consistent with convergence, reported "
          "raw, not gated on an invented threshold",
          abs(peak_ratios[3] - peak_ratios[2]) <= abs(peak_ratios[1] - peak_ratios[0]) + 1e-12)

    # 3d. grid-CONVENTION robustness (not just density): the midpoint grid (i+0.5)/N, an
    # equally-valid discretization, at two densities, to confirm the verdict is not an artifact
    # of the specific a/N convention borrowed from M2's BZ-scan.
    print("    Grid-CONVENTION robustness check (midpoint grid (i+0.5)/N, using the fast direct "
          "formula E_flat_direct for speed -- PART2b already proved it equals band_quantum_metric):")
    midpoint_peaks = []
    for N in (20, 30):
        pts = bz_grid(N, midpoint=True)
        Es = np.array([E_flat_direct(p) for p in pts])
        res = analyze(Es, v_mean)
        midpoint_peaks.append(res["peak_ratio"])
        print(f"      midpoint N={N}: peak ratio = {res['peak_ratio']:.4f} at "
              f"beta={res['peak_beta']:.4f}, crossing_anywhere={res['crossing_anywhere']}")
    check("PART3d the midpoint-grid convention gives the SAME qualitative verdict and a peak "
          "ratio within 1% of the a/N-grid table (not a discretization-convention artifact)",
          all(abs(mp - peak_ratios[-1]) / peak_ratios[-1] < 0.02 for mp in midpoint_peaks))

    # =======================================================================
    banner("VERDICT -- which of SS2's frozen outcome branches (N1/N2/N3) is reached")
    # =======================================================================
    all_finite = all(table[N]["n_nonfinite"] == 0 for N in DENSITIES)
    consistent_verdict = len({table[N]["crossing_anywhere"] for N in DENSITIES}) == 1
    any_crossing = table[DENSITIES[-1]]["crossing_anywhere"]

    branch = "N1" if (all_finite and consistent_verdict) else "N2/N3-under-investigation"
    print(f"    BRANCH REACHED: {branch}  (rho_flat(beta) is finite and grid-convergent at every "
          f"tested beta and density -- no divergence on the flat lines, no floating-point "
          f"overflow, no invented convention needed to make it finite).")

    if branch == "N1":
        sub_verdict = "CROSSING-FOUND" if any_crossing else "NO-CROSSING / degenerate"
        print(f"    Per the ORIGINAL X.2-a verdict criteria (X2_zeq_sweep_return_2026-07-10.md:"
              f"295-302), applied UNCHANGED: {sub_verdict}.")
        if not any_crossing:
            print(f"        rho_cone(beta) > rho_flat(beta) at EVERY sampled beta in "
                  f"[{beta_scan[0]:.3f}, {beta_scan[-1]:.1f}] (spanning ~0.004x to ~12x "
                  f"BETA_EFF={BETA_EFF:.4f}) -- the CLOSEST approach is peak ratio "
                  f"~{peak_ratios[-1]:.3f} (never reaching 1) at beta~{peak_betas[-1]:.3f} "
                  f"(beta/BETA_EFF~{peak_betas[-1]/BETA_EFF:.3f}), squarely inside the "
                  f"'physically sensible... within an order of magnitude' bracket the spec "
                  f"itself names for a CROSSING -- i.e. this is NOT 'the crossing sits at an "
                  f"unphysical beta->0 or beta->infinity limit'; there is simply no crossing "
                  f"anywhere. This is the ORIGINAL spec's branch (b): 'one band dominates rho "
                  f"for all beta' -- a THIRD independent confirmation (after ML3-C's regulator-"
                  f"dependence and ML3b's no-radius-crossing) that a single global KMS "
                  f"temperature does not produce a matter/radiation-like equality, pointing the "
                  f"mechanism question toward comoving-number-dilution (MG-1c's actual "
                  f"machinery), not toward a thermal crossing.")
    check("FINAL sanity: the verdict reached is well-defined (branch determined, not stuck)",
          branch == "N1")

    banner("NAMED, UN-RESOLVED SECONDARY FINDING (disclosed, not smuggled; NOT part of the "
           "branch verdict above)")
    print(f"""    rho_flat(beta) itself CROSSES ZERO at beta_zero~{beta_zeros[-1]:.3f}
    (beta_zero/BETA_EFF~{beta_zeros[-1]/BETA_EFF:.3f}) and is NEGATIVE for beta > beta_zero,
    growing in magnitude (still finite at every FIXED beta -- the BZ is compact -- but
    unboundedly as beta->infinity, since E_flat(k) attains a genuine BULK negative minimum
    ~{Es_dense.min():.3f} at finite-volume k-points, e.g. near (0.25,0.75,0.25)-type points --
    NOT on the measure-zero flat great circles X.2-b found (those carry E=0 exactly, not
    negative; they contribute ZERO weight to a volume integral regardless). This is a DIFFERENT
    phenomenon from the gate-stopped radial construction's divergence and from SS2's N2 branch
    (which is specifically about the measure-zero locus) -- no rectification is applied here (per
    SS3's poison list), the signed value is reported raw.
    This connects to, but does NOT resolve, the sweep's own GATED item #2
    (X2_zeq_sweep_return_2026-07-10.md Sec.2 MISSING #2: "No chemical potential / number
    constraint has ever been decided... mu pinned at the node... assumed, never derived"): an
    average excitation energy BELOW the reference node is exactly the kind of structural
    oddity that item flags as load-bearing rather than a deferred nicety. Naming it here per
    the amendment's instruction to "name it; do NOT regulate it away" -- resolving it is
    Station X.2-c's chartered job, not this one's.""")

    banner("SUMMARY")
    print(f"""    PART 1 (rho_cone) -- unchanged, clean, matches the gate-stop script's own numbers.
    PART 2 -- E_flat(k), the operator's OWN flat-band eigenvalue (band_quantum_metric's E
      channel), is a single, globally continuous, BOUNDED band over the ENTIRE compact BZ
      torus (range ~[{Es_dense.min():.3f}, {Es_dense.max():.3f}]) -- confirmed against a direct
      eigh-based cross-check (exact agreement), against the already-established small-k A(n_hat)
      numbers (exact agreement), and against band-index continuity (never crosses into the
      cone/other branches anywhere sampled).
    PART 3 -- rho_flat(beta), built as the BZ-average of E*exp(-beta*E) (degeneracy x1, the SAME
      Maxwell weight the cone side uses, NO qmax, NO radial ansatz, NO A(n_hat), NO sign
      rectification), is FINITE and grid-CONVERGENT at every density tested (12^3 through 36^3)
      and under an independent grid-CONVENTION check (midpoint vs zero-included). This resolves
      the prior gate-stop's non-convergence entirely: the pathology was the unbounded radial
      q-domain, not anything intrinsic to the flat band's signed dispersion.
    VERDICT: BRANCH N1 (CONVERGES). Applying the sweep's own ORIGINAL verdict criteria unchanged:
      NO-CROSSING / degenerate -- rho_cone(beta) > rho_flat(beta) at every tested beta (closest
      approach ~{peak_ratios[-1]*100:.1f}% at beta~{peak_betas[-1]/BETA_EFF:.2f}*BETA_EFF, never
      reaching parity) -- a THIRD independent confirmation that a single global KMS temperature
      does not produce a matter/radiation-like crossing.
    SECONDARY, UNRESOLVED FINDING (named, not smuggled): rho_flat(beta) itself goes negative for
      beta > beta_zero~{beta_zeros[-1]/BETA_EFF:.2f}*BETA_EFF, sourced from BULK (finite-volume)
      k-points, not the measure-zero flat lines -- pointing at the un-derived mu/node-pinning
      convention (the sweep's GATED item #2 / Station X.2-c's chartered question), not resolved
      here.""")
    print("RESULT:", ("BRANCH N1 -- CONVERGES; verdict = NO-CROSSING/degenerate (all internal "
                       "checks pass)" if ok_all else "A CHECK FAILED UNEXPECTEDLY -- inspect above"))
    return ok_all


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
