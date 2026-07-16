#!/usr/bin/env python3
"""
proofs/foundations/D1b_controlled_bw_2026-07-09.py

D1b -- THE CONTROLLED BISOGNANO-WICHMANN READ (Newton's G 2pi).  Pre-registered FROZEN in
internal research notes (commit 2411540, BEFORE this file was written).
Build Ops Protocol (charter 0b7dd6d); the exploitation-wave station named by the D1 post-mortem /
completeness review (internal research notes, item 12/3).

THE DESIGN PRINCIPLE (frozen, from the pre-reg): D1/ML-1''' read a RATIO -- beta(first bond)/x(first
bond) -- directly sensitive to the absolute distance convention (Finding 2's ~2x ambiguity).  D1b reads
a SLOPE: the linear coefficient of the multi-bond near-horizon profile beta(x) fit over MANY bonds at
DERIVED Albanese/Kotani-Sunada Cartesian positions -- invariant under any constant shift of the position
assignment, and only weakly sensitive to per-bond convention residue (tested directly, V-3).

ARCHITECT ADJUDICATIONS (frozen; see the pre-reg for the full text):
  1. Positions are pos(i,x) = Xv[i] + L@x (explore_12_harmonic_geometry, the Kotani-Sunada standard
     realization) -- the isotropization-weld frame.  Distances = Euclidean / v_iso.  No cell counting,
     no hardcoded gaps.
  2. ML-1's graph-hop geodesic metric is NOT claimed as the emergent-metric distance; V-5 reports the
     comparison without gating on it.
  3. Dense eigh ladder M in {8,10,12,14,16} (M=6 excluded -- it drove the retired D1/ML-1''' fit); the
     BFS-heavy Patch.__init__ default is skipped via skip_pair_bfs=True in the main ladder (this
     instrument never needs vertex-graph BFS distances; V-5 alone rebuilds a BFS-enabled Patch).
  4. Convention residue (bond-midpoint vs bond-endpoint; window x<=50% vs x<=33% of proper region
     depth) is MEASURED (V-3), not assumed.

DISCLOSED DEVIATIONS / DESIGN CHOICES (binding to record, not to hide):
  (a) the_net.bond_profile_slope's second argument is realized as a per-bond RECORD list
      (i, j, pos_i, pos_j) rather than a flat all-vertex position array, because selecting which h_A
      entries constitute "the bond coupling" requires physical-adjacency pairing (see the_net.py's
      docstring on bond_profile_slope for the full statement). Functionally this delivers exactly what
      the pre-reg specifies: beta_b=|h_A[i,j]| at physical bonds vs x_b=|cut_normal . declared_point|,
      weighted (by multiplicity, one data point per bond) linear fit over the window.
  (b) the_net.vertex_position takes ONLY the vertex tuple (positions are patch/M-independent; Patch is
      unnecessary plumbing for this bridge).
  (c) the region-cut convention (threshold = midpoint of the achieved cut_normal-projection range over
      the WHOLE patch box) is applied UNIFORMLY to all three directions (axis/<111>/<110>) -- this
      generalizes D1-3's diagonal-cut convention (previously used only for <111>) to the axis cut too,
      rather than reusing D1-1's simpler integer "x0<M/2" rule.  This is a deliberate unification (one
      recipe, applied identically "across directions and conventions" per the pre-reg's own requirement)
      and is DISCLOSED here, not silently substituted.
  (d) "physical bonds crossing/near the cut" = every adjacency edge (of the full non-backtracking-walk
      cover, not just one branch-pair/displacement type) with BOTH endpoints inside region A -- the
      multi-bond, multi-layer generalization of D1's single-edge-type first-bond read.
  (e) V-3's "the fit error's 2x" tolerance uses the PRIMARY (window=50%, midpoint) fit's fit_err as the
      reference scale (disclosed choice; the pre-reg does not disambiguate which of the two compared
      fits' errors is meant).
  (f) V-4's cross-form / convention spread combination uses quadrature addition (⊕), the standard
      reading of the pre-reg's "fit ⊕ cross-form ⊕ convention spread" notation.
  (g) the optional M=18 addendum named in adjudication 3 is NOT run (explicitly optional, non-primary,
      and omitted here to bound total runtime; disclosed, not silently dropped).

HARD RULES / POISONS (binding, from the pre-reg): windows/ladder/fit-forms/tolerances frozen BEFORE any
number is seen; no goal-seek toward 2pi; hbar is derived ONLY under BW-2pi-CONFIRMED; the retired
"+7%/1.068x2pi" is never cited as a prior or a check; M=6 excluded from all fits; no scoreboard value
moves in this station regardless of verdict; the_net.py extension is accretion (existing behavior
default-identical, verified at V-2); prior-art proof files (D1, ML-1''', ML-1') are read, never edited.
"""
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import the_net as net  # noqa: E402
import srs  # noqa: E402
import explore_12_harmonic_geometry as ex12  # noqa: E402  (re-runs its own diagnostic on FIRST import;
                                              # identical pattern to D1/adapters/sunada_geometry.py)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
TWO_PI = 2 * math.pi
T_WALL_START = time.time()
FAST = "--fast" in sys.argv
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    cond = bool(cond)
    ok_all = ok_all and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 96)
    print(f" {t}")
    print("=" * 96)


def elapsed():
    return time.time() - T_WALL_START


banner("D1b -- THE CONTROLLED BISOGNANO-WICHMANN READ  (SLOPE not RATIO)" + ("  [--fast]" if FAST else ""))
print("Pre-reg (FROZEN): internal research notes (commit 2411540)")
print("Prior art (read, NOT reused/edited): proofs/foundations/D1_bw_canonical_2026-07-08.py")

# ================================================================================================
banner("V-0  BENCHMARK  --  benchmark_bw_2pi(800) calibrates the pipeline's units")
# ================================================================================================
sb, rb = net.benchmark_bw_2pi(800)
print(f"    critical-chain near-horizon slope = {sb:.6f}  =  {rb:.6f} x 2pi")
v0_pass = check("V-0 benchmark reproduces ~1.0 x 2pi (pipeline units trusted)", abs(rb - 1.0) < 0.02,
                detail=f"{rb:.6f} x 2pi")
if not v0_pass:
    print("\n*** V-0 FAILED -- the BW-slope pipeline itself is not reproducing its known calibration. "
          "STOPPING (a calibration failure makes every downstream number meaningless). ***")
    sys.exit(1)

# ================================================================================================
banner("THE CANONICAL FRAME  --  derived on-screen from L (explore_12) and g_frac (the_net)")
# ================================================================================================
L = ex12.L
Yv = ex12.Yv
Xv = ex12.Xv
Linv = np.linalg.inv(L)
print(f"L (explore_12's Kotani-Sunada Albanese realization matrix, used AS-IS) =\n{L}")

g_frac = net.emergent_metric()
g_cart = L @ g_frac @ L.T
ev_cart = np.linalg.eigvalsh(g_cart)
v_iso = math.sqrt(float(np.mean(ev_cart)))
print(f"g_cart = L @ emergent_metric() @ L^T; eig = {np.round(ev_cart, 8)}")
print(f"v_iso = sqrt(mean eig(g_cart)) = {v_iso:.8f}  (the proper-distance rescaling; SR-4 isotropized)")


def cut_normal_for(e_frac):
    """The Cartesian unit normal to the fractional-plane family {y: e_frac.y=c}: n = L^{-T} e_frac
    (the reciprocal-lattice functional), normalized.  Identity used throughout (verified numerically
    below): for any full vertex position X(i,x)=Xv[i]+L@x,  n_unnormalized . X(i,x) = e_frac.(x+Yv[i])
    EXACTLY (since Xv=L@Yv and L^{-T}e_frac . L@x = e_frac.x) -- so projecting the CARTESIAN position
    onto this normal reproduces the fractional cell-coordinate functional used by D1-3's diag_bond_mean,
    with no separate fractional-space computation needed."""
    n = Linv.T @ np.asarray(e_frac, dtype=float)
    return n / np.linalg.norm(n)


DIRECTIONS = {"axis": (1, 0, 0), "<111>": (1, 1, 1), "<110>": (1, 1, 0)}
NORMALS = {name: cut_normal_for(e) for name, e in DIRECTIONS.items()}
for name, e in DIRECTIONS.items():
    print(f"  direction {name:5s}  e_frac={e}  cut_normal (Cartesian unit) = {np.round(NORMALS[name], 6)}")

# identity check (printed, not gated -- a sanity cross-check of the projection shortcut above)
_i, _x = 2, np.array([1, -2, 3])
_lhs = float((Linv.T @ np.array([1, 1, 1], float)) @ (Xv[_i] + L @ _x.astype(float)))
_rhs = float(np.array([1, 1, 1], float) @ (_x.astype(float) + Yv[_i]))
print(f"  identity check (<111>, vertex (2,[1,-2,3])): n.X = {_lhs:.10f}  vs  e.(x+Yv) = {_rhs:.10f}  "
      f"|diff|={abs(_lhs - _rhs):.2e}")

# ================================================================================================
banner("V-1  POSITION BRIDGE ROUND-TRIP")
# ================================================================================================
v1_pass = True
for i in range(4):
    p = net.vertex_position((i, (0, 0, 0)))
    v1_pass &= check(f"vertex_position(({i},(0,0,0))) == ex12.Xv[{i}] exactly",
                      np.allclose(p, Xv[i], atol=1e-12), detail=f"|diff|={np.max(np.abs(p - Xv[i])):.2e}")

test_cell = np.array([2, -1, 3])
for i in range(4):
    p = net.vertex_position((i, tuple(int(c) for c in test_cell)))
    expect = Xv[i] + L @ test_cell.astype(float)
    v1_pass &= check(f"vertex_position(({i}, cell={tuple(test_cell)})) == Xv[{i}] + L@cell",
                      np.allclose(p, expect, atol=1e-12), detail=f"|diff|={np.max(np.abs(p - expect)):.2e}")

# vectorized bulk-position helper used below for performance at large N; verified against the
# per-vertex net.vertex_position on a sample (the ONLY reason it exists is speed at N~16000+).
NVc = srs.NV


def positions_array(patch):
    """Vectorized equivalent of [net.vertex_position(v) for v in vertex_adjacency()'s verts list]
    (IDENTICAL formula; vectorized purely for performance -- verts iterates 'for x in box: for i in
    range(NV)', matched here by repeating each box row NV times)."""
    cells = np.repeat(np.asarray(patch.box, dtype=float), NVc, axis=0)
    Xv_tile = np.tile(Xv, (len(patch.box), 1))
    return Xv_tile + cells @ L.T


_patch_v1 = net.Patch(M=3, skip_pair_bfs=True)
_H_v1, _verts_v1 = _patch_v1.vertex_adjacency()
_bulk = positions_array(_patch_v1)
_sample = [0, 5, 17, 40, len(_verts_v1) - 1]
_bulk_err = max(np.max(np.abs(_bulk[k] - net.vertex_position(_verts_v1[k]))) for k in _sample)
v1_pass &= check("vectorized positions_array() == net.vertex_position() on a vertex sample (M=3)",
                  _bulk_err < 1e-12, detail=f"max|diff|={_bulk_err:.2e}")

if not v1_pass:
    print("\n*** V-1 FAILED -- the position bridge does not round-trip. STOPPING. ***")
    sys.exit(1)

# ================================================================================================
banner("V-2  REGRESSION  --  the net's own anchors/reads + Patch default bit-identity")
# ================================================================================================
net_ok = net.self_test(verbose=True)
v2a = check("net.self_test() (all regression anchors + ML-0/ML-2/ML-3/ML-2b reads) PASS", net_ok)

p_default = net.Patch(M=6)
p_skip = net.Patch(M=6, skip_pair_bfs=True)
H1, verts1 = p_default.vertex_adjacency()
H2, verts2 = p_skip.vertex_adjacency()
v2b = check("Patch(M=6) default vs skip_pair_bfs=True: verts + vertex_adjacency() bit-identical",
            verts1 == verts2 and np.array_equal(H1, H2))
v2c = check("Patch(M=6) default vs skip_pair_bfs=True: RD (darts) and B (Hashimoto walk) bit-identical",
            p_default.RD == p_skip.RD and np.array_equal(p_default.B, p_skip.B))


def _raises_runtime(fn):
    try:
        fn()
        return False
    except RuntimeError:
        return True


v2d = check("skip_pair_bfs=True: _dV is None (BFS skipped) and vdist/geodesic_dist_to_vertices raise "
            "a clear RuntimeError",
            p_skip._dV is None and _raises_runtime(lambda: p_skip.vdist(0, 0))
            and _raises_runtime(lambda: p_skip.geodesic_dist_to_vertices([0])))
v2_pass = v2a and v2b and v2c and v2d
if not v2_pass:
    print("\n*** V-2 FAILED -- the_net.py extension broke a regression anchor or the skip_pair_bfs "
          "accretion is not behavior-preserving. STOPPING. ***")
    sys.exit(1)

# ================================================================================================
banner("THE MEASUREMENT  --  dense ladder, all 3 cut directions, shared eigh per M")
# ================================================================================================
FULL_MS = [8, 10, 12, 14, 16]
Ms = [8, 10, 12] if FAST else FULL_MS
dir_names_full = ["axis", "<111>", "<110>"]
dir_names = ["axis", "<111>"] if FAST else dir_names_full
print(f"M-ladder: {Ms}{'  (--fast: M<=12)' if FAST else ''}")
print(f"directions: {dir_names}{'  (--fast: axis + <111> only)' if FAST else ''}")
print("(one eigh/covariance per M, shared across all directions and both point conventions)")


def physical_bonds(patch):
    """Every undirected physical adjacency edge as a (global_i, global_j) index pair (i<j), read
    directly from patch.RD (the real-space darts) via patch.vidx -- NOT by scanning the dense H
    matrix (which would cost O(N^2) memory/time at N~16000).  Deduplicates the two darts per edge."""
    seen = set()
    bonds = []
    for (ta, ha) in patch.RD:
        gi, gj = patch.vidx[ta], patch.vidx[ha]
        key = (gi, gj) if gi < gj else (gj, gi)
        if key not in seen:
            seen.add(key)
            bonds.append(key)
    return bonds


# results[direction][window_point_key] = list of (M, slope, fit_err, n_bonds)
WPK = ["w50_mid", "w33_mid", "w50_end"]
results = {name: {k: [] for k in WPK} for name in dir_names_full}
region_depths = {name: {} for name in dir_names_full}

for M in Ms:
    t0 = time.time()
    patch = net.Patch(M=M, skip_pair_bfs=True)
    H, verts = patch.vertex_adjacency()
    t_build = time.time()
    E, V = np.linalg.eigh(H)
    t_eig = time.time()
    cols = V[:, E < -1.0 - 1e-9]                      # Dirac-sea vacuum fill (cone sector)
    C = cols @ cols.conj().T
    t_cov = time.time()
    X_proper = positions_array(patch) / v_iso          # (N,3) Cartesian, v_iso-scaled proper units
    all_bonds = physical_bonds(patch)
    fill = cols.shape[1]
    print(f"  M={M:2d}  N={len(verts):6d}  fill={fill:6d} ({100.0*fill/len(verts):.1f}%)  "
          f"[build {t_build-t0:5.2f}s  eigh {t_eig-t_build:7.2f}s  cov {t_cov-t_eig:5.2f}s]  "
          f"n_bonds(total)={len(all_bonds)}")

    for name in dir_names:
        cut_normal = NORMALS[name]
        proj = X_proper @ cut_normal
        threshold = (proj.min() + proj.max()) / 2.0
        inA = proj < threshold
        A_idx = np.where(inA)[0]
        region_depth = threshold - proj.min()
        region_depths[name][M] = region_depth
        posA = {int(g): a for a, g in enumerate(A_idx)}
        C_A = C[np.ix_(A_idx, A_idx)]
        h_A = net.entanglement_hamiltonian(C_A)
        shift = threshold * cut_normal
        recs = []
        for (gi, gj) in all_bonds:
            if inA[gi] and inA[gj]:
                recs.append((posA[gi], posA[gj], X_proper[gi] - shift, X_proper[gj] - shift))
        w50, w33 = 0.5 * region_depth, region_depth / 3.0
        s50m, e50m, n50m, _ = net.bond_profile_slope(h_A, recs, cut_normal, w50, point="midpoint")
        s33m, e33m, n33m, _ = net.bond_profile_slope(h_A, recs, cut_normal, w33, point="midpoint")
        s50e, e50e, n50e, _ = net.bond_profile_slope(h_A, recs, cut_normal, w50, point="endpoint")
        results[name]["w50_mid"].append((M, s50m, e50m, n50m))
        results[name]["w33_mid"].append((M, s33m, e33m, n33m))
        results[name]["w50_end"].append((M, s50e, e50e, n50e))
        t_dir = time.time()
        print(f"      dir={name:5s} |A|={len(A_idx):6d} region_depth={region_depth:.4f}  "
              f"n_bonds(A)={len(recs):6d}  "
              f"slope[w50,mid]={s50m:.6f}({s50m/TWO_PI:.4f}x2pi,n={n50m},fiterr={e50m:.4f})  "
              f"slope[w33,mid]={s33m:.6f}({s33m/TWO_PI:.4f}x2pi,n={n33m})  "
              f"slope[w50,end]={s50e:.6f}({s50e/TWO_PI:.4f}x2pi,n={n50e})  "
              f"[{t_dir-t_cov:.2f}s]")
    print(f"    [M={M} total {time.time()-t0:.2f}s, wall so far {elapsed():.1f}s]")

# ================================================================================================
banner("FULL SLOPE TABLE  (direction x M x window x convention, in units of 2pi)")
# ================================================================================================
for name in dir_names:
    print(f"  direction {name}:")
    for k, label in [("w50_mid", "window=50%,midpoint(PRIMARY)"), ("w33_mid", "window=33%,midpoint(ALT window)"),
                      ("w50_end", "window=50%,endpoint(ALT point)")]:
        row = results[name][k]
        print(f"    {label:32s}: " + "  ".join(f"M={M}:{s/TWO_PI:.4f}" for (M, s, e, n) in row))

# ================================================================================================
banner("EXTRAPOLATIONS  M->inf  (per direction, PRIMARY window=50%/midpoint ladder)")
# ================================================================================================


def extrap_linear(Ms_, vals):
    x = np.array([1.0 / m for m in Ms_])
    y = np.array(vals)
    a, b = np.polyfit(x, y, 1)
    resid = float(np.sqrt(np.mean((y - (a * x + b)) ** 2)))
    return float(b), resid


def extrap_plateau(Ms_, vals, Mmin=12):
    sel = [v for m, v in zip(Ms_, vals) if m >= Mmin]
    if not sel:
        return float("nan"), float("nan")
    return float(np.mean(sel)), float(np.std(sel))


def extrap_quadratic(Ms_, vals):
    if len(Ms_) < 3:
        return float("nan"), float("nan")
    x = np.array([1.0 / m for m in Ms_])
    y = np.array(vals)
    coefs = np.polyfit(x, y, 2)
    resid = float(np.sqrt(np.mean((y - np.polyval(coefs, x)) ** 2)))
    return float(coefs[-1]), resid


declared = {}   # name -> dict(linear=(val,err), plateau=(...), quadratic=(...))
for name in dir_names:
    Ms_ = [r[0] for r in results[name]["w50_mid"]]
    vals = [r[1] for r in results[name]["w50_mid"]]
    lin, lin_e = extrap_linear(Ms_, vals)
    plat, plat_e = extrap_plateau(Ms_, vals, Mmin=12)
    quad, quad_e = extrap_quadratic(Ms_, vals)
    declared[name] = dict(linear=(lin, lin_e), plateau=(plat, plat_e), quadratic=(quad, quad_e))
    print(f"  {name:5s}: linear-1/M (DECLARED)  = {lin:12.6f} (+/-{lin_e:.6f})  = {lin/TWO_PI:.6f} x 2pi")
    print(f"          plateau-mean(M>=12)     = {plat:12.6f} (+/-{plat_e:.6f})  = "
          f"{(plat/TWO_PI if plat==plat else float('nan')):.6f} x 2pi")
    print(f"          quadratic-1/M           = {quad:12.6f} (+/-{quad_e:.6f})  = "
          f"{(quad/TWO_PI if quad==quad else float('nan')):.6f} x 2pi")
    spread = max(abs(lin - plat), abs(lin - quad), abs(plat - quad)) if plat == plat else abs(lin - quad)
    print(f"          cross-form spread (max pairwise |diff|) = {spread:.6f}  ({spread/TWO_PI:.6f} x 2pi)")

# ================================================================================================
banner("V-3  CONVENTION INSENSITIVITY  (at M=12, per direction)")
# ================================================================================================
v3_pass = True
v3_detail = {}
for name in dir_names:
    row50m = next(r for r in results[name]["w50_mid"] if r[0] == 12)
    row33m = next(r for r in results[name]["w33_mid"] if r[0] == 12)
    row50e = next(r for r in results[name]["w50_end"] if r[0] == 12)
    _, s50m, e50m, _ = row50m
    _, s33m, e33m, _ = row33m
    _, s50e, e50e, _ = row50e
    point_rel = abs(s50m - s50e) / abs(s50m) if s50m == s50m and s50m != 0 else float("nan")
    window_spread = abs(s50m - s33m)
    window_tol = 2.0 * e50m          # disclosed choice (e): reference = PRIMARY fit's fit_err
    pass_point = point_rel == point_rel and point_rel < 0.05
    pass_window = window_spread <= window_tol
    v3_detail[name] = dict(point_rel=point_rel, window_spread=window_spread, window_tol=window_tol,
                            pass_point=pass_point, pass_window=pass_window)
    v3_pass &= check(f"V-3 [{name}] |slope_mid-slope_end|/slope_mid < 0.05",
                      pass_point, detail=f"{point_rel:.4f}  (mid={s50m:.4f}, end={s50e:.4f})")
    v3_pass &= check(f"V-3 [{name}] |slope_w50-slope_w33| < 2*fiterr(primary)",
                      pass_window, detail=f"spread={window_spread:.4f} vs tol={window_tol:.4f}")

# ================================================================================================
banner("V-4  THE VERDICT  (Lorentz gate, then CONFIRMED / MISS-QUANTIFIED / INSTRUMENT-LIMITED)")
# ================================================================================================
r_d = {name: declared[name]["linear"][0] / TWO_PI for name in dir_names}
r_bar = float(np.mean(list(r_d.values())))
# disclosed reading of the pre-reg's "|r_d-r_bar|/r_bar": abs() is applied to the DENOMINATOR too
# (a relative-deviation gate must use a magnitude scale; r_bar can be near-zero/negative for an
# unstable small-M extrapolation, which would otherwise flip the sign of a positive numerator).
lorentz_devs = {name: abs(r - r_bar) / abs(r_bar) if r_bar != 0 else float("inf") for name, r in r_d.items()}
lorentz_gate = max(lorentz_devs.values()) < 0.05
print(f"  r_d (declared linear-1/M, /2pi): " + ", ".join(f"{n}={v:.6f}" for n, v in r_d.items()))
print(f"  r_bar = {r_bar:.6f}")
print(f"  Lorentz-gate deviations |r_d-r_bar|/r_bar: " +
      ", ".join(f"{n}={v:.4f}" for n, v in lorentz_devs.items()))
check("V-4 LORENTZ GATE: max_d |r_d-r_bar|/r_bar < 0.05 (direction-independence)", lorentz_gate,
      detail=f"max={max(lorentz_devs.values()):.4f}"
      + ("  [FAST MODE: only 2 of 3 directions measured -- NOT the full 3-direction gate]" if FAST else ""))
check("V-4 V-3 gate carried forward (convention insensitivity)", v3_pass)

if (not lorentz_gate) or (not v3_pass):
    verdict = "INSTRUMENT-LIMITED"
    reason = ("the Lorentz gate or V-3 convention-insensitivity FAILED -- booked RAW with the named "
              "residual defect; no verdict on 2pi is claimed.")
    magnitude_str = f"r_bar = {r_bar:.6f} (NOT gated -- instrument-limited)"
else:
    fit_component = float(np.mean([declared[n]["linear"][1] for n in dir_names])) / TWO_PI
    cross_form_component = float(np.mean([
        max(abs(declared[n]["linear"][0] - declared[n]["plateau"][0]),
            abs(declared[n]["linear"][0] - declared[n]["quadratic"][0]),
            abs(declared[n]["plateau"][0] - declared[n]["quadratic"][0])) if declared[n]["plateau"][0] == declared[n]["plateau"][0]
        else abs(declared[n]["linear"][0] - declared[n]["quadratic"][0])
        for n in dir_names])) / TWO_PI
    convention_component = float(np.mean([
        max(v3_detail[n]["point_rel"] * r_d[n], v3_detail[n]["window_spread"] / TWO_PI) for n in dir_names]))
    combined_err = float(math.sqrt(fit_component ** 2 + cross_form_component ** 2 + convention_component ** 2))
    print(f"  uncertainty budget (disclosure f: combined in quadrature):")
    print(f"    fit component (mean linear-fit fit_err/2pi)        = {fit_component:.6f}")
    print(f"    cross-form spread component (mean max pairwise/2pi)= {cross_form_component:.6f}")
    print(f"    convention-residue component (V-3, /2pi)           = {convention_component:.6f}")
    print(f"    combined (quadrature) = {combined_err:.6f}")
    if abs(r_bar - 1.0) <= 0.03:
        verdict = "BW-2PI-CONFIRMED"
        reason = ("|r_bar-1|<=0.03 with the Lorentz gate holding: the local near-horizon slope carries "
                   "EXACTLY the BW 2pi in the derived Cartesian frame => MG-1d's incomplete equation (the "
                   "emergent local Unruh/BW temperature) COMPLETES: kappa_local = hbar/t_P, G_eff=G closes, "
                   "hbar=h/2pi is DERIVED as the boost-side action quantum (per the pre-reg's frozen "
                   "sentence; hbar was NEVER selected -- it is the consequence of this measured verdict).")
        magnitude_str = f"r_bar = {r_bar:.6f} +/- {combined_err:.6f} (CONFIRMED at 2pi)"
    else:
        verdict = "BW-MISS-QUANTIFIED"
        reason = (f"the Lorentz gate holds (direction-independent) but |r_bar-1|={abs(r_bar-1):.4f} > 0.03: "
                   f"the 2pi residual is FINALLY QUANTIFIED at r_bar={r_bar:.6f} +/- {combined_err:.6f} "
                   f"(fit(+)cross-form(+)convention spread, quadrature). Newton's G stays OPEN with this "
                   f"measured magnitude booked (the honest number D1/ML-1''' did not deliver).")
        magnitude_str = f"r_bar = {r_bar:.6f} +/- {combined_err:.6f}  (MISS, quantified)"

print(f"\n>>> V-4 VERDICT: {verdict} <<<")
print(f"    {magnitude_str}")
print(f"BOOKED SENTENCE: {reason}")

# ================================================================================================
banner("V-5  THE FOURTH-CONVENTION CONFRONT  (report only)  --  graph-hop geodesic vs Cartesian, M=12/axis")
# ================================================================================================
if FAST:
    print("  SKIPPED under --fast (needs a fresh BFS-enabled Patch(M=12); omitted to hit the fast-mode "
        "runtime target -- run the full station for V-5).")
else:
    t0 = time.time()
    M12 = 12
    patch_geo = net.Patch(M=M12)                       # skip_pair_bfs=False (default): BFS needed here
    H_geo, verts_geo = patch_geo.vertex_adjacency()
    E_geo, V_geo = np.linalg.eigh(H_geo)
    cols_geo = V_geo[:, E_geo < -1.0 - 1e-9]
    C_geo = cols_geo @ cols_geo.conj().T
    A_idx_geo = np.array([n for n, (i, x) in enumerate(verts_geo) if x[0] < M12 // 2])
    plane_vidx = [n for n, (i, x) in enumerate(verts_geo) if x[0] == M12 // 2]
    dgeo = patch_geo.geodesic_dist_to_vertices(plane_vidx)
    posA_geo = {int(g): a for a, g in enumerate(A_idx_geo)}
    C_A_geo = C_geo[np.ix_(A_idx_geo, A_idx_geo)]
    h_A_geo = net.entanglement_hamiltonian(C_A_geo)
    A_set_geo = set(int(g) for g in A_idx_geo)
    all_bonds_geo = physical_bonds(patch_geo)
    xs_geo, betas_geo = [], []
    for (gi, gj) in all_bonds_geo:
        if gi in A_set_geo and gj in A_set_geo:
            x = min(dgeo[gi], dgeo[gj]) + 0.5           # ML-1' bond-center convention (ONE hop apart)
            xs_geo.append(x)
            betas_geo.append(abs(h_A_geo[posA_geo[gi], posA_geo[gj]]))
    xs_geo = np.array(xs_geo)
    betas_geo = np.array(betas_geo)
    geo_window = 0.5 * xs_geo.max()
    sel = xs_geo <= geo_window
    a_geo, b_geo = np.polyfit(xs_geo[sel], betas_geo[sel], 1)
    slope_geo = float(a_geo)
    r_geo = slope_geo / TWO_PI
    cart_row12 = next(r for r in results["axis"]["w50_mid"] if r[0] == 12)
    r_cart12 = cart_row12[1] / TWO_PI
    print(f"  geodesic-hop metric: n_bonds(A, all depths)={len(xs_geo)}, window(50% of max hop-depth "
          f"{xs_geo.max():.1f})={geo_window:.2f}, n_bonds(window)={int(sel.sum())}")
    print(f"  slope_geo (graph-hop x-axis) = {slope_geo:.6f} = {r_geo:.6f} x 2pi")
    print(f"  slope_cart (M=12, axis, w50/midpoint) = {cart_row12[1]:.6f} = {r_cart12:.6f} x 2pi")
    print(f"  |r_geo - r_cart| = {abs(r_geo - r_cart12):.4f}  ({time.time()-t0:.2f}s)")
    check("V-5 recorded (report only; NOT gating)", True,
          detail=f"geodesic {r_geo:.4f}x2pi vs Cartesian {r_cart12:.4f}x2pi -- a disagreement is EXPECTED "
                  "(adjudication 2: the graph-hop metric is combinatorial, not the emergent-metric "
                  "distance; no reconciliation required)")

# ================================================================================================
banner("SUMMARY")
# ================================================================================================
t_total = elapsed()
print(f"""    V-0 benchmark ................................ {rb:.6f} x 2pi  ({'PASS' if v0_pass else 'FAIL'})
    V-1 position bridge round-trip ............... ({'PASS' if v1_pass else 'FAIL'})
    V-2 regression + Patch bit-identity ........... ({'PASS' if v2_pass else 'FAIL'})
    V-3 convention insensitivity .................. ({'PASS' if v3_pass else 'FAIL'})
    V-4 VERDICT: {verdict}
        {magnitude_str}
    V-5 fourth-convention confront: {'SKIPPED (--fast)' if FAST else f'geodesic {r_geo:.4f}x2pi vs Cartesian {r_cart12:.4f}x2pi'}
    total station wall time: {t_total:.1f}s ({t_total/60.0:.2f} min)
""")

core_pass = v0_pass and v1_pass and v2_pass
print("RESULT:", "CORE CONTRACTS (V-0/V-1/V-2) PASS -- a definite V-4 verdict was reached"
      if core_pass else "A CORE CONTRACT (V-0/V-1/V-2) FAILED -- inspect above")
print(f"(the V-4 verdict {verdict} is a scientific finding, not a script failure; exit code reflects "
      f"only whether V-0/V-1/V-2 passed and the stations completed without a crash)")
sys.exit(0 if core_pass else 1)
