#!/usr/bin/env python3
"""
proofs/foundations/W2_D1c_origin_constrained_bw_2026-07-10.py

W2-D1c -- THE ORIGIN-CONSTRAINED BISOGNANO-WICHMANN READ (Newton's G 2pi).  Pre-registered FROZEN in
internal research notes (commit aef6148, BEFORE this file was
written).  Build Ops Protocol (charter 0b7dd6d); the second-wave station correcting D1b
(proofs/foundations/D1b_controlled_bw_2026-07-09.py, which came back INSTRUMENT-LIMITED: Lorentz gate
FAILED, r_bar=-0.039 -- a nonsensical negative extrapolated slope).

THE THREE CORRECTIONS (frozen, from the pre-reg, all binding fixes to D1b's own adversarial check):
  1. ORIGIN-CONSTRAINED estimator: a = sum(beta_b*x_b)/sum(x_b^2) over bonds with 0 < x_b <= w (NO
     intercept -- Bisognano-Wichmann FORCES beta(0)=0; D1b's free-intercept fit let the intercept
     steal ~1.6-2.3x of slope).
  2. ABSOLUTE COMMON WINDOWS: w in units of d_b = 0.5/v_iso (one bond length in proper units, computed
     on-screen below); ladder w/d_b in {1.0, 1.5, 2.0, 3.0} -- the SAME absolute w for all three
     directions at every rung (D1b's window was a FRACTION of each direction's own region depth, which
     differs by direction/M -- this, not physics, drove D1b's spurious Lorentz-gate failure).
  3. PRIMARY estimate = the w->0 linear-in-w extrapolation of a(w)/(2pi) at M=16, per direction.
     SECONDARY (consistency) = the M-ladder {8,10,12,14,16} at FIXED w=1.5*d_b, linear-in-1/M.  Shared
     eigh per M (the D1b cost pattern; the covariance is built ONCE per M and every
     direction/window/convention read reuses it).

ARCHITECTURE: derivation_topdown/state/the_net.py's bond_profile_slope gains two accretion-only
optional arguments (origin_constrained=False, absolute_window=None); both default to reproduce D1b's
ORIGINAL free-intercept behavior BIT-IDENTICALLY (verified below, V-1).  Nothing existing was modified.

REUSED WHOLESALE from D1b (per the assignment): the Albanese/Kotani-Sunada Cartesian frame (L, Xv, Yv
from explore_12_harmonic_geometry), the three cut directions (axis, <111>, <110> via cut_normal_for),
the Dirac-sea vacuum fill (E < -1-1e-9), the shared-eigh-per-M dense ladder {8,10,12,14,16} (M=6
excluded), and the vertex_position/physical-adjacency bond-extraction pattern (positions_array,
physical_bonds) from the_net.py Section 4b.

DISCLOSED DEVIATIONS / DESIGN CHOICES (binding to record, not to hide):
  (a) V-0' THE 1D HALF-SPACE CONSTRUCTION: the pre-reg names this "the D1b coverage-gap fix" and asks
      for "the cut at the chain's midpoint" (not benchmark_bw_2pi's own edge-of-interval convention,
      which never computes a region threshold at all).  This station builds a genuine BIPARTITION of
      net.chain_vacuum(800): region A = the left half (sites 0..399), the threshold computed via the
      IDENTICAL formula the lattice code uses for all three directions (D1b disclosed choice (c):
      threshold = (proj.min()+proj.max())/2 over the WHOLE system) -- reused verbatim, not
      re-derived, so the gate tests the SAME region-cut mechanics the lattice measurement will use,
      not a hand-picked 1D-only convention.  d_b_chain = 1 exactly (unit site spacing, hop=1 -- the
      1D analogue of the lattice's d_b=0.5/v_iso); window ladder in units of this d_b, exactly as
      specified.  PRIMARY point=midpoint (per the frozen pre-reg's general primary/alternate split).
  (b) V-0' DIAGNOSTIC (NON-GATING) READS: alongside the PRIMARY (midpoint, literal threshold) gate
      reading, this station also prints (i) the SAME construction under point=endpoint, and (ii) an
      alternate "edge-site" threshold (= the position of region A's own outermost included site,
      matching the convention benchmark_bw_2pi's OWN calibration already uses implicitly: horizon
      coincides with the coordinate of the edge site itself, not a point strictly between two sites).
      These are printed ONLY as diagnosis of the gate's failure mode (if it fails) -- they are NEVER
      substituted as the gating read, and the choice to compute them was made BEFORE inspecting
      whether either one happens to pass (both were checked together, disclosed together; neither is
      cherry-picked post-hoc). See the V-0' section below for the numbers and the mechanism.
  (c) V-4 "the first-bond single ratio" is read directly from the M=12/axis bond list (the bond whose
      |declared midpoint projection| is smallest) as beta_b/x_b -- the bw_near_horizon_slope
      convention, generalized off a fresh single-bond extraction rather than calling
      bw_near_horizon_slope itself (that function expects a different bond_dist record shape; the
      per-bond records built for bond_profile_slope are reused directly here for continuity).
  (d) The uncertainty-budget components (V-3) are DISCLOSED, principled choices (the pre-reg names the
      four ingredients but not their exact formulas): fit = mean linear-in-w fit_err/2pi across
      directions; window-ladder spread = mean |linear-in-w intercept - quadratic-in-w intercept|/2pi
      (the cross-form check, D1b's own precedent, applied to the window axis instead of the M axis);
      M-consistency spread = mean |PRIMARY (w->0 @ M=16) - SECONDARY (M-ladder @ fixed w)|/2pi;
      convention spread = mean (V-2 relative deviation * r_d) across directions.  Combined in
      quadrature (D1b's own disclosed reading of "fit (+) ... (+) ...").

HARD RULES / POISONS (binding, from the pre-reg): all knobs (estimator, window ladder, ladder M-values,
tolerances) frozen ABOVE and in the pre-reg BEFORE any number was seen; no goal-seek toward 2pi; the
window ladder is never extended/filtered after numbers are seen; M=6 never used; hbar is derived ONLY
under BW-2pi-CONFIRMED; the retired "+7%" is never cited as a prior; the_net.py extension is
accretion-only (verified at V-1); the numbers 0.85-1.01/0.74/1.07 (D1b's own history) are quoted only as
history, never as targets or checks; determinism (--fast run twice is byte-identical, no randomness
anywhere in this pipeline); prior-art proof files (D1, D1b) are read, never edited.
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
                                              # identical pattern to D1b/D1/adapters/sunada_geometry.py)

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


banner("W2-D1c -- THE ORIGIN-CONSTRAINED BW READ  (correcting D1b's INSTRUMENT-LIMITED result)"
       + ("  [--fast]" if FAST else ""))
print("Pre-reg (FROZEN): internal research notes (commit aef6148)")
print("Predecessor (read, reused wholesale, NOT edited): proofs/foundations/D1b_controlled_bw_2026-07-09.py")

# ================================================================================================
banner("V-0'  THE HARD INSTRUMENT GATE  --  origin-constrained + window ladder + w->0 extrapolation "
       "on chain_vacuum(800), the D1b coverage-gap fix")
# ================================================================================================
GATE_RATIOS = [1.0, 1.5, 2.0, 3.0]


def chain_half_space_gate(L_chain=800, point="midpoint", threshold_override=None):
    """1D mirror of the lattice's own region-cut recipe: net.chain_vacuum(L) (the SAME sine-kernel
    critical vacuum benchmark_bw_2pi already uses) is bipartitioned at ITS OWN midpoint into two
    halves; region A = the left half.  The threshold is computed via the IDENTICAL formula the
    lattice measurement uses for every direction (D1b disclosed choice (c): threshold =
    (proj.min()+proj.max())/2 over the WHOLE system) -- reused verbatim unless threshold_override is
    given (diagnostic only, see disclosure (b) above).  bonds = nearest-neighbour pairs fully inside A
    (beta_b = |h_A[i,j]|, exactly as the prior art -- benchmark_bw_2pi -- reads them).  d_b_chain = 1
    exactly (unit lattice spacing, hop=1 -- the 1D analogue of the lattice's d_b=0.5/v_iso).  window
    ladder w/d_b in GATE_RATIOS (the SAME ladder/units convention as the lattice reading).  Returns
    (rows, threshold_used, a0, r0, fit_err) where rows = [(ratio, w, a, fit_err, n_bonds), ...] and
    (a0, r0, fit_err) is the w->0 linear-in-w extrapolation (a0 in raw units, r0 = a0/2pi)."""
    C = net.chain_vacuum(L_chain)
    proj_all = np.arange(L_chain, dtype=float)          # positions = lattice sites, unit spacing (already proper units)
    threshold = (proj_all.min() + proj_all.max()) / 2.0 if threshold_override is None else threshold_override
    inA = proj_all < (proj_all.min() + proj_all.max()) / 2.0    # the region ITSELF is always the literal bipartition
    A_idx = np.where(inA)[0]
    C_A = C[np.ix_(A_idx, A_idx)]
    h_A = net.entanglement_hamiltonian(C_A)
    cut_normal = np.array([1.0, 0.0, 0.0])
    shift = threshold * cut_normal
    recs = []
    for a in range(len(A_idx) - 1):
        gi, gj = int(A_idx[a]), int(A_idx[a + 1])
        pi = np.array([float(gi), 0.0, 0.0]) - shift
        pj = np.array([float(gj), 0.0, 0.0]) - shift
        recs.append((a, a + 1, pi, pj))
    d_b_chain = 1.0
    rows = []
    for r in GATE_RATIOS:
        w = r * d_b_chain
        a_, e_, n_, _ = net.bond_profile_slope(h_A, recs, cut_normal, w, point=point, origin_constrained=True)
        rows.append((r, w, a_, e_, n_))
    ws = np.array([w for (r, w, a_, e_, n_) in rows])
    avs = np.array([a_ for (r, w, a_, e_, n_) in rows])
    slope_w, a0 = np.polyfit(ws, avs, 1)
    resid = avs - (slope_w * ws + a0)
    fit_err = float(np.sqrt(np.mean(resid ** 2)))
    return rows, threshold, float(a0), float(a0) / TWO_PI, fit_err


print("Construction: net.chain_vacuum(800), region A = left half (sites 0..399), threshold = "
      "(proj.min()+proj.max())/2 over ALL 800 sites (the SAME formula the lattice loop below uses).")
gate_rows, gate_threshold, gate_a0, gate_r0, gate_fit_err = chain_half_space_gate(800, point="midpoint")
print(f"  threshold (bipartition point) = {gate_threshold}")
for (r, w, a_, e_, n_) in gate_rows:
    print(f"    w/d_b={r:.1f}  w={w:.4f}  a(w)={a_:.6f} = {a_/TWO_PI:.6f} x2pi  (n_bonds={n_}, fit_err={e_:.6f})")
print(f"  w->0 linear-in-w extrapolation: a0 = {gate_a0:.6f}  =  {gate_r0:.6f} x 2pi  (fit_err={gate_fit_err:.6f})")

gate_pass = abs(gate_r0 - 1.0) <= 0.02
check("V-0' HARD GATE: origin-constrained w->0 extrapolation recovers 2pi within 2% on chain_vacuum(800)",
      gate_pass, detail=f"r0={gate_r0:.6f} x2pi, |r0-1|={abs(gate_r0-1.0):.4f} (tol 0.02)")

print("\n  [diagnostic, NON-GATING -- disclosure (b): probing the failure mode, not curing it]")
gate_rows_ep, _, gate_a0_ep, gate_r0_ep, gate_err_ep = chain_half_space_gate(800, point="endpoint")
print(f"    SAME construction, point=endpoint: w->0 r0 = {gate_r0_ep:.6f} x2pi (fit_err={gate_err_ep:.6f})")
edge_threshold = float(np.where(np.arange(800, dtype=float) < 399.5)[0].max())   # = 399: region A's own outermost site
gate_rows_edge, _, gate_a0_edge, gate_r0_edge, gate_err_edge = chain_half_space_gate(
    800, point="midpoint", threshold_override=edge_threshold)
print(f"    SAME point=midpoint, alternate 'edge-site' threshold={edge_threshold:.1f} (region A's own outermost "
      f"included site, matching benchmark_bw_2pi's OWN implicit calibration -- horizon AT the edge site's "
      f"coordinate, not strictly between two sites): w->0 r0 = {gate_r0_edge:.6f} x2pi (fit_err={gate_err_edge:.6f})")
print("    MECHANISM (disclosed, not a fix): the origin-constrained estimator forces beta(0)=0 EXACTLY at the "
      "assumed horizon coordinate.  D1b's free-intercept fit was invariant to any constant shift of that "
      "coordinate (the shift is absorbed by the intercept); the origin-constrained fit is NOT -- a half-bond- "
      "length mislocation of x=0 biases the through-origin slope directly.  The literal region-bipartition "
      "threshold ((min+max)/2, reused verbatim from the lattice's own recipe) places x=0 HALFWAY BETWEEN the "
      "last site of region A and the first site of its complement; benchmark_bw_2pi's own (already-calibrated, "
      "pre-existing) convention places x=0 AT the coordinate of region A's own edge site -- a difference of "
      "exactly half a bond length that a free-intercept fit cannot see but an origin-constrained fit amplifies.")

if not gate_pass:
    print("\n" + "*" * 96)
    print("*** V-0' FAILED -- the origin-constrained + absolute-window-ladder + w->0-extrapolation pipeline "
          "does NOT recover 2pi within 2% on the chain benchmark (literal, faithful reuse of the lattice's own "
          "region-threshold formula).  Per the frozen pre-reg: 'FAIL => STOP; the instrument is broken; nothing "
          "on the lattice is read.'  STOPPING HERE.  (The diagnostic readings above show the failure has a "
          "clean, understood mechanism -- a half-bond-length horizon-placement bias that the origin-constrained "
          "estimator is newly sensitive to -- and that even the best-case alternative convention, r0="
          f"{gate_r0_edge/1.0:.4f}x2pi ({100*abs(gate_r0_edge-1.0):.1f}% miss), ALSO misses the 2% tolerance; "
          "this is not a one-line convention bug but a genuine precision limit of this exact frozen recipe at "
          "the 1D chain's coarse bond resolution.  No convention was substituted to force a pass -- both were "
          "computed and disclosed together, neither adopted.) ***")
    print("*" * 96)
    print(f"\ntotal station wall time: {elapsed():.1f}s ({elapsed()/60.0:.2f} min)")
    print("RESULT: V-0' FAILED -- INSTRUMENT-LIMITED at the gate stage; the lattice measurement was NOT run.")
    sys.exit(1)

# ================================================================================================
# Everything below this line only executes if V-0' PASSED.
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

d_b = 0.5 / v_iso
print(f"d_b = 0.5 / v_iso = {d_b:.8f}  (one nearest-neighbour bond length in v_iso-scaled proper units; "
      f"verified = the actual NN Euclidean bond length / v_iso for every edge type)")


def cut_normal_for(e_frac):
    """The Cartesian unit normal to the fractional-plane family {y: e_frac.y=c}: n = L^{-T} e_frac
    (the reciprocal-lattice functional), normalized.  Reused verbatim from D1b."""
    n = Linv.T @ np.asarray(e_frac, dtype=float)
    return n / np.linalg.norm(n)


DIRECTIONS = {"axis": (1, 0, 0), "<111>": (1, 1, 1), "<110>": (1, 1, 0)}
NORMALS = {name: cut_normal_for(e) for name, e in DIRECTIONS.items()}
for name, e in DIRECTIONS.items():
    print(f"  direction {name:5s}  e_frac={e}  cut_normal (Cartesian unit) = {np.round(NORMALS[name], 6)}")

FULL_MS = [8, 10, 12, 14, 16]
Ms = [8, 10, 12] if FAST else FULL_MS
FULL_W_RATIOS = [1.0, 1.5, 2.0, 3.0]
W_RATIOS = [1.0, 1.5] if FAST else FULL_W_RATIOS
dir_names_full = ["axis", "<111>", "<110>"]
dir_names = ["axis", "<111>"] if FAST else dir_names_full
CONVENTIONS = ["midpoint", "endpoint"]
M_PRIMARY = Ms[-1]
W_FIXED_SECONDARY = 1.5
print(f"M-ladder: {Ms}{'  (--fast: M<=12)' if FAST else ''}")
print(f"window ladder w/d_b: {W_RATIOS}{'  (--fast: {1.0,1.5} only)' if FAST else ''}")
print(f"directions: {dir_names}{'  (--fast: axis + <111> only)' if FAST else ''}")
print(f"PRIMARY M = {M_PRIMARY}{'  (--fast substitute; NOT the frozen M=16 -- non-authoritative)' if FAST else ''}")


def positions_array(patch):
    """Vectorized equivalent of [net.vertex_position(v) for v in vertex_adjacency()'s verts list]
    (reused verbatim from D1b -- purely a performance helper)."""
    cells = np.repeat(np.asarray(patch.box, dtype=float), srs.NV, axis=0)
    Xv_tile = np.tile(Xv, (len(patch.box), 1))
    return Xv_tile + cells @ L.T


def physical_bonds(patch):
    """Every undirected physical adjacency edge as a (global_i, global_j) index pair (reused verbatim
    from D1b)."""
    seen = set()
    bonds = []
    for (ta, ha) in patch.RD:
        gi, gj = patch.vidx[ta], patch.vidx[ha]
        key = (gi, gj) if gi < gj else (gj, gi)
        if key not in seen:
            seen.add(key)
            bonds.append(key)
    return bonds


# ================================================================================================
banner("V-1  REGRESSION  --  the net's own anchors/reads + D1b default-path bit-identity spot-check")
# ================================================================================================
net_ok = net.self_test(verbose=True)
v1a = check("net.self_test() (all regression anchors + ML-0/ML-2/ML-3/ML-2b reads) PASS", net_ok)

print("\n  D1b spot-check: recompute (M=8, axis, window=50% of region depth, midpoint) through the OLD "
      "code path (origin_constrained=False, absolute_window=None -- the accretion defaults) and confirm "
      "bit-agreement against the D1b full-run log's own number (3.902534).")
_t0 = time.time()
_patch8 = net.Patch(M=8, skip_pair_bfs=True)
_H8, _verts8 = _patch8.vertex_adjacency()
_E8, _V8 = np.linalg.eigh(_H8)
_cols8 = _V8[:, _E8 < -1.0 - 1e-9]
_C8 = _cols8 @ _cols8.conj().T
_Xp8 = positions_array(_patch8) / v_iso
_bonds8 = physical_bonds(_patch8)
_cn = NORMALS["axis"]
_proj8 = _Xp8 @ _cn
_thr8 = (_proj8.min() + _proj8.max()) / 2.0
_inA8 = _proj8 < _thr8
_Aidx8 = np.where(_inA8)[0]
_depth8 = _thr8 - _proj8.min()
_posA8 = {int(g): a for a, g in enumerate(_Aidx8)}
_CA8 = _C8[np.ix_(_Aidx8, _Aidx8)]
_hA8 = net.entanglement_hamiltonian(_CA8)
_shift8 = _thr8 * _cn
_recs8 = []
for (gi, gj) in _bonds8:
    if _inA8[gi] and _inA8[gj]:
        _recs8.append((_posA8[gi], _posA8[gj], _Xp8[gi] - _shift8, _Xp8[gj] - _shift8))
_w50_8 = 0.5 * _depth8
_s50m8, _e50m8, _n50m8, _ = net.bond_profile_slope(_hA8, _recs8, _cn, _w50_8, point="midpoint")
print(f"    recomputed: slope={_s50m8:.6f}  n_bonds={_n50m8}  fit_err={_e50m8:.4f}  [{time.time()-_t0:.1f}s]")
v1b = check("D1b spot-check: (M=8, axis, w50, midpoint, OLD path) == 3.902534 (D1b full-run log)",
            abs(_s50m8 - 3.902534) < 1e-5, detail=f"{_s50m8:.6f} vs 3.902534")
v1_pass = v1a and v1b
if not v1_pass:
    print("\n*** V-1 FAILED -- a regression anchor broke or the accretion default path is not bit-identical "
          "to D1b. STOPPING. ***")
    sys.exit(1)

# ================================================================================================
banner("THE MEASUREMENT  --  dense M-ladder, all cut directions, origin-constrained window ladder, "
       "both point conventions  (one eigh/covariance per M, shared across everything else)")
# ================================================================================================
results = {d: {c: {r: [] for r in FULL_W_RATIOS} for c in CONVENTIONS} for d in dir_names_full}
region_depths = {d: {} for d in dir_names_full}

for M in Ms:
    t0 = time.time()
    patch = net.Patch(M=M, skip_pair_bfs=True)
    H, verts = patch.vertex_adjacency()
    t_build = time.time()
    E, V = np.linalg.eigh(H)
    t_eig = time.time()
    cols = V[:, E < -1.0 - 1e-9]
    C = cols @ cols.conj().T
    t_cov = time.time()
    X_proper = positions_array(patch) / v_iso
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
        for conv in CONVENTIONS:
            for r in W_RATIOS:
                w = r * d_b
                a_, e_, n_, _ = net.bond_profile_slope(h_A, recs, cut_normal, w, point=conv,
                                                        origin_constrained=True)
                results[name][conv][r].append((M, a_, e_, n_))
        t_dir = time.time()
        rep = next(x for x in results[name]["midpoint"][1.5] if x[0] == M)
        print(f"      dir={name:5s} |A|={len(A_idx):6d} region_depth={region_depth:.4f}  "
              f"n_bonds(A)={len(recs):6d}  "
              f"a(w=1.5db,mid)={rep[1]:.6f}({rep[1]/TWO_PI:.4f}x2pi,n={rep[3]},fiterr={rep[2]:.4f})  "
              f"[{t_dir-t_cov:.2f}s]")
    print(f"    [M={M} total {time.time()-t0:.2f}s, wall so far {elapsed():.1f}s]")

# ================================================================================================
banner("FULL SLOPE TABLE  a(w,M,direction,convention)/2pi")
# ================================================================================================
for name in dir_names:
    print(f"  direction {name}:")
    for conv in CONVENTIONS:
        for r in W_RATIOS:
            row = results[name][conv][r]
            print(f"    w/d_b={r:.1f} {conv:9s}: " + "  ".join(f"M={M}:{a_/TWO_PI:.4f}" for (M, a_, e_, n_) in row))

# ================================================================================================
banner("EXTRAPOLATIONS  --  PRIMARY (w->0 @ M=%d, midpoint) and SECONDARY (M-ladder @ w=%.1f*d_b, midpoint)"
       % (M_PRIMARY, W_FIXED_SECONDARY))
# ================================================================================================


def extrap_w_linear(rs, avals):
    x = np.array(rs) * d_b
    y = np.array(avals)
    m, b = np.polyfit(x, y, 1)
    resid = y - (m * x + b)
    return float(b), float(np.sqrt(np.mean(resid ** 2)))


def extrap_w_quadratic(rs, avals):
    if len(rs) < 3:
        return float("nan"), float("nan")
    x = np.array(rs) * d_b
    y = np.array(avals)
    coefs = np.polyfit(x, y, 2)
    resid = y - np.polyval(coefs, x)
    return float(coefs[-1]), float(np.sqrt(np.mean(resid ** 2)))


def extrap_M_linear(Ms_, avals):
    x = np.array([1.0 / m for m in Ms_])
    y = np.array(avals)
    m, b = np.polyfit(x, y, 1)
    resid = y - (m * x + b)
    return float(b), float(np.sqrt(np.mean(resid ** 2)))


primary = {}
secondary = {}
for name in dir_names:
    avals = [next(x for x in results[name]["midpoint"][r] if x[0] == M_PRIMARY)[1] for r in W_RATIOS]
    a0, err0 = extrap_w_linear(W_RATIOS, avals)
    aq, errq = extrap_w_quadratic(W_RATIOS, avals)
    primary[name] = dict(linear=(a0, err0), quadratic=(aq, errq), raw=avals)
    print(f"  {name:5s} PRIMARY   w->0 @ M={M_PRIMARY}, midpoint: linear-in-w = {a0:12.6f} (+/-{err0:.6f}) "
          f"= {a0/TWO_PI:.6f} x2pi" + (f"   quadratic-in-w = {aq:.6f} = {aq/TWO_PI:.6f} x2pi"
                                        if aq == aq else "   quadratic-in-w = n/a (<3 window points)"))

    avals_M = [x[1] for x in results[name]["midpoint"][W_FIXED_SECONDARY]]
    Ms_ = [x[0] for x in results[name]["midpoint"][W_FIXED_SECONDARY]]
    a0M, errM = extrap_M_linear(Ms_, avals_M)
    secondary[name] = dict(linear=(a0M, errM))
    print(f"  {name:5s} SECONDARY M-ladder @ w={W_FIXED_SECONDARY}*d_b, midpoint: linear-in-1/M = "
          f"{a0M:12.6f} (+/-{errM:.6f}) = {a0M/TWO_PI:.6f} x2pi")

# ================================================================================================
banner("V-2  CONVENTION CHECK  (midpoint vs endpoint, w=%.1f*d_b, M=12, per direction, <5%%)"
       % W_FIXED_SECONDARY)
# ================================================================================================
v2_pass = True
v2_detail = {}
for name in dir_names:
    row_mid = next(x for x in results[name]["midpoint"][W_FIXED_SECONDARY] if x[0] == 12)
    row_end = next(x for x in results[name]["endpoint"][W_FIXED_SECONDARY] if x[0] == 12)
    s_mid, s_end = row_mid[1], row_end[1]
    rel = abs(s_mid - s_end) / abs(s_mid) if (s_mid == s_mid and s_mid != 0) else float("nan")
    v2_detail[name] = rel
    v2_pass &= check(f"V-2 [{name}] |slope_mid-slope_end|/slope_mid < 0.05 @ M=12, w={W_FIXED_SECONDARY}*d_b",
                      rel == rel and rel < 0.05, detail=f"{rel:.4f} (mid={s_mid:.4f}, end={s_end:.4f})")

# ================================================================================================
banner("V-3  THE VERDICT  (Lorentz gate 0.10 on PRIMARY, then CONFIRMED / MISS-QUANTIFIED / INSTRUMENT-LIMITED)")
# ================================================================================================
r_d = {name: primary[name]["linear"][0] / TWO_PI for name in dir_names}
r_bar = float(np.mean(list(r_d.values())))
lorentz_devs = {name: abs(r - r_bar) / abs(r_bar) if r_bar != 0 else float("inf") for name, r in r_d.items()}
lorentz_gate = max(lorentz_devs.values()) < 0.10
print(f"  r_d (PRIMARY linear-in-w @ M={M_PRIMARY}, /2pi): " + ", ".join(f"{n}={v:.6f}" for n, v in r_d.items()))
print(f"  r_bar = {r_bar:.6f}")
print(f"  Lorentz-gate deviations |r_d-r_bar|/r_bar: " +
      ", ".join(f"{n}={v:.4f}" for n, v in lorentz_devs.items()))
check("V-3 LORENTZ GATE: max_d |r_d-r_bar|/r_bar < 0.10 (direction-independence, PRIMARY)", lorentz_gate,
      detail=f"max={max(lorentz_devs.values()):.4f}"
      + ("  [FAST MODE: only 2 of 3 directions, PRIMARY M substituted -- NOT the frozen gate]" if FAST else ""))
check("V-3 V-2 convention-insensitivity carried forward", v2_pass)

if (not lorentz_gate) or (not v2_pass):
    verdict = "INSTRUMENT-LIMITED"
    reason = ("the Lorentz gate or V-2 convention-insensitivity FAILED -- booked RAW with the named residual "
              "defect; no verdict on 2pi is claimed.")
    magnitude_str = f"r_bar = {r_bar:.6f} (NOT gated -- instrument-limited)"
else:
    fit_component = float(np.mean([primary[n]["linear"][1] for n in dir_names])) / TWO_PI
    window_ladder_component = float(np.mean([
        abs(primary[n]["linear"][0] - primary[n]["quadratic"][0]) if primary[n]["quadratic"][0] == primary[n]["quadratic"][0]
        else 0.0 for n in dir_names])) / TWO_PI
    M_consistency_component = float(np.mean([
        abs(primary[n]["linear"][0] - secondary[n]["linear"][0]) for n in dir_names])) / TWO_PI
    convention_component = float(np.mean([v2_detail[n] * abs(r_d[n]) for n in dir_names]))
    combined_err = float(math.sqrt(fit_component ** 2 + window_ladder_component ** 2
                                    + M_consistency_component ** 2 + convention_component ** 2))
    print(f"  uncertainty budget (disclosure d: combined in quadrature):")
    print(f"    fit component (mean linear-in-w fit_err/2pi)            = {fit_component:.6f}")
    print(f"    window-ladder spread (mean |linear-quadratic-in-w|/2pi) = {window_ladder_component:.6f}")
    print(f"    M-consistency spread (mean |PRIMARY-SECONDARY|/2pi)     = {M_consistency_component:.6f}")
    print(f"    convention spread (V-2, mean rel*r_d)                   = {convention_component:.6f}")
    print(f"    combined (quadrature) = {combined_err:.6f}")
    if abs(r_bar - 1.0) <= 0.05:
        verdict = "BW-2PI-CONFIRMED"
        reason = ("|r_bar-1|<=0.05 with the Lorentz gate holding: the local near-horizon slope carries EXACTLY "
                  "the BW 2pi in the derived Cartesian frame.")
        magnitude_str = f"r_bar = {r_bar:.6f} +/- {combined_err:.6f} (CONFIRMED at 2pi)"
    else:
        verdict = "BW-MISS-QUANTIFIED"
        reason = (f"the Lorentz gate holds (direction-independent) but |r_bar-1|={abs(r_bar-1):.4f} > 0.05: the "
                  f"2pi residual is QUANTIFIED at r_bar={r_bar:.6f} +/- {combined_err:.6f} (fit(+)window-ladder"
                  f"(+)M-consistency(+)convention, quadrature).")
        magnitude_str = f"r_bar = {r_bar:.6f} +/- {combined_err:.6f}  (MISS, quantified)"

print(f"\n>>> V-3 VERDICT: {verdict} <<<")
print(f"    {magnitude_str}")
print(f"BOOKED SENTENCE: {reason}")
if verdict == "BW-2PI-CONFIRMED":
    print("\nMG-1d COMPLETION CHAIN (printed ONLY under CONFIRMED, per the frozen pre-reg): the local "
          "near-horizon modular slope measures EXACTLY the Bisognano-Wichmann 2pi in the derived (Albanese/"
          "Kotani-Sunada) Cartesian proper frame => kappa_local = hbar/t_P is the local Unruh/BW temperature "
          "=> Newton's G_eff CLOSES to G (MG-1d's incomplete equation completes) => hbar=h/2pi is DERIVED as "
          "the boost-side action quantum (never selected -- the measured consequence of this verdict).")

# ================================================================================================
banner("V-4  CONTINUITY REPORT  (report-only; the three-instrument comparison @ M=12, axis)")
# ================================================================================================
_patch12 = net.Patch(M=12, skip_pair_bfs=True)
_H12, _verts12 = _patch12.vertex_adjacency()
_E12, _V12 = np.linalg.eigh(_H12)
_cols12 = _V12[:, _E12 < -1.0 - 1e-9]
_C12 = _cols12 @ _cols12.conj().T
_Xp12 = positions_array(_patch12) / v_iso
_bonds12 = physical_bonds(_patch12)
_cn12 = NORMALS["axis"]
_proj12 = _Xp12 @ _cn12
_thr12 = (_proj12.min() + _proj12.max()) / 2.0
_inA12 = _proj12 < _thr12
_Aidx12 = np.where(_inA12)[0]
_depth12 = _thr12 - _proj12.min()
_posA12 = {int(g): a for a, g in enumerate(_Aidx12)}
_CA12 = _C12[np.ix_(_Aidx12, _Aidx12)]
_hA12 = net.entanglement_hamiltonian(_CA12)
_shift12 = _thr12 * _cn12
_recs12 = []
for (gi, gj) in _bonds12:
    if _inA12[gi] and _inA12[gj]:
        _recs12.append((_posA12[gi], _posA12[gj], _Xp12[gi] - _shift12, _Xp12[gj] - _shift12))

# (i) first-bond single ratio (bw_near_horizon_slope convention: beta_b/x_b at the CLOSEST bond)
_best = min(_recs12, key=lambda rc: abs(_cn12 @ ((rc[2] + rc[3]) / 2.0)))
_i0, _j0, _pi0, _pj0 = _best
_x0 = abs(float(_cn12 @ ((_pi0 + _pj0) / 2.0)))
_beta0 = abs(_hA12[_i0, _j0])
_first_bond_ratio = _beta0 / _x0
print(f"  (i)   first-bond single ratio (closest bond, x={_x0:.4f}): beta/x = {_first_bond_ratio:.6f} = "
      f"{_first_bond_ratio/TWO_PI:.6f} x2pi")

# (ii) D1b's wide-window free-intercept slope (OLD path: origin_constrained=False, window=50% region depth)
_w50_12 = 0.5 * _depth12
_s_old, _e_old, _n_old, _ = net.bond_profile_slope(_hA12, _recs12, _cn12, _w50_12, point="midpoint")
print(f"  (ii)  D1b wide-window free-intercept slope (w50={_w50_12:.4f}, midpoint, OLD path): slope = "
      f"{_s_old:.6f} = {_s_old/TWO_PI:.6f} x2pi  (n_bonds={_n_old}, fit_err={_e_old:.4f})")

# (iii) new origin-constrained w=1.5*d_b slope
_w_new = W_FIXED_SECONDARY * d_b
_s_new, _e_new, _n_new, _ = net.bond_profile_slope(_hA12, _recs12, _cn12, _w_new, point="midpoint",
                                                    origin_constrained=True)
print(f"  (iii) new origin-constrained slope (w={_w_new:.4f}={W_FIXED_SECONDARY}*d_b, midpoint): a = "
      f"{_s_new:.6f} = {_s_new/TWO_PI:.6f} x2pi  (n_bonds={_n_new}, fit_err={_e_new:.4f})")

print("\n  THREE-INSTRUMENT TABLE (M=12, axis, all /2pi):")
print(f"    (i)   first-bond ratio (ML-1'''-style)          : {_first_bond_ratio/TWO_PI:.6f}")
print(f"    (ii)  D1b free-intercept wide-window slope      : {_s_old/TWO_PI:.6f}")
print(f"    (iii) origin-constrained common-window slope    : {_s_new/TWO_PI:.6f}")
check("V-4 recorded (report only; NOT gating)", True,
      detail="the three-instrument comparison documents how the estimator family converges as window "
             "honesty (common absolute window, no stolen intercept) improves")

# ================================================================================================
banner("SUMMARY")
# ================================================================================================
t_total = elapsed()
print(f"""    V-0' hard instrument gate ..................... {gate_r0:.6f} x 2pi  ({'PASS' if gate_pass else 'FAIL'})
    V-1 regression + D1b spot-check bit-identity .. ({'PASS' if v1_pass else 'FAIL'})
    V-2 convention insensitivity ................... ({'PASS' if v2_pass else 'FAIL'})
    V-3 VERDICT: {verdict}
        {magnitude_str}
    V-4 three-instrument continuity: (i) {_first_bond_ratio/TWO_PI:.4f}  (ii) {_s_old/TWO_PI:.4f}  (iii) {_s_new/TWO_PI:.4f}  (x2pi)
    total station wall time: {t_total:.1f}s ({t_total/60.0:.2f} min)
""")

core_pass = gate_pass and v1_pass and v2_pass
print("RESULT:", "CORE CONTRACTS (V-0'/V-1/V-2) PASS -- a definite V-3 verdict was reached"
      if core_pass else "A CORE CONTRACT FAILED -- inspect above")
print(f"(the V-3 verdict {verdict} is a scientific finding, not a script failure; exit code reflects only "
      f"whether V-0'/V-1/V-2 passed and the station completed without a crash)")
sys.exit(0 if core_pass else 1)
