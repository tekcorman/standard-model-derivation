#!/usr/bin/env python3
"""
proofs/foundations/ML1d_derived_horizon_2026-07-12.py

ML-1d-b -- THE DERIVED HORIZON + THE OPERATOR-LEVEL BW READ, CORRECTED INSTRUMENT (Push-3 W1).
Pre-registered FROZEN in internal research notes (commit d41e286)
as AMENDED by internal research notes (commit 27b0916, BEFORE this
amended code; the original R2 hard gate FAILED as frozen -- see working notes/
ML1d_return_2026-07-12.md and git ed61816 for the original station verbatim -- and NOTHING on the
lattice was read, so no lattice number contaminated the amendment).  Targets MG-1d's incomplete
equation: the emergent local Unruh/BW temperature (G_eff = G/(2pi) is the open miss).

THE AMENDMENT'S EXACT DELTA (nothing else changed; the amendment checker diffs against ed61816):
  1. THE NEAR-SURFACE SECTOR (Leg B pair selection AND Leg C projection) is now the
     ENTANGLEMENT-CARRYING sector: occupation lambda in [1e-8, 1-1e-8]
     (net.entanglement_carrying_selection, Section 9b).  State-determined dimension, reported
     per region/benchmark.  [Replaces near_surface_selection's frac=0.5, which the ML-1d return
     diagnosed as ~90% round-off-ordered saturated bulk modes.]
  2. THE BOOST TEMPLATE is now the finite-region conformal parabola: K_template =
     sum_b w(x_b).T00_b, w(x) = x*(ell-x)/ell, ell = the region's proper depth along the cut
     normal (net.k_boost_parabolic, Section 9b; ell fixed by geometry, never fitted; per-placement
     DIRECT recomputation since the quadratic w breaks the linear template's shift identity).
     The BW statement under test is c* = 2pi.  [Replaces the linear/infinite-Rindler x*T00.]
  3. The R2 HARD GATE (|c*/2pi - 1| <= 0.02, edge placement) now binds on the GENUINE finite
     open-chain ground-state projector at L=800; chain_vacuum(800) is REPORTED alongside for
     continuity, non-gating.  SECOND gate fail => INSTRUMENT-CLOSED, no third redesign.

THE STATION (three legs; structure verbatim from d41e286):
  Leg A -- INSTRUMENT GATES: R2 (hard) as amended above.  R1 (report, not gate) the mirror-mode
    placement extractor on the 1D chain must reproduce D1c's own finding (underdetermined at the
    half-bond scale) or report what it fixes.
  Leg B (R1) -- THE MODULAR PLACEMENT OF THE ENTANGLING SURFACE: for the pure patch vacuum cut
    A u A', the free-fermion Schmidt/mirror-mode pairing (the_net.mirror_mode_pairing) gives, per
    entanglement-carrying mode pair, x0* = the midpoint of the two modes' position expectations.
    Placement is DERIVED iff the spread of x0* (across pairs and across the three cut
    directions) < 0.15*d_b.
  Leg C (R2) -- THE OPERATOR-LEVEL BOOST TEST: c* = argmin_c ||h_A - c.K_template||_F/||h_A||_F
    on the entanglement-carrying sector.  THREE placements (derived/midway/edge) x TWO states
    (vacuum primary, KMS-at-beta_natural control, NEVER averaged) x THREE directions (axis,
    <111>, <110>) x THREE M in {10,12,14} dense (M=6 never used).
  PLUS the L0b rank/profile read (mandatory, cheap; now in the amendment's explicit W3a form:
  the dimension of the POSITIVE modular spectrum within the entanglement-carrying sector, the
  forced odd-region zero mode excluded exactly -- docs/theorems/CA_half_lemma_2026-07-12.md;
  the original distinct-magnitude count is still printed for continuity) and the diamond
  diagnostic (report-only, no verdict weight).

ARCHITECTURE: derivation_topdown/state/the_net.py Section 9 (mirror_mode_pairing,
mode_position_expectation, project_matrix, boost_c_star, distinct_spectrum_count,
diamond_vertex_region) + Section 9b (the amendment's two corrected definitions:
entanglement_carrying_selection, k_boost_parabolic; plus positive_spectrum_dimension for the
W3a-explicit L0b read) supply every generic reader; this file is pure ORCHESTRATION (Patch/H/eigh
construction, region slicing, the M-ladder/direction/state/placement loop), matching the
D1b/W2-D1c precedent.  REUSED WHOLESALE from D1b/W2-D1c: the Albanese/Kotani-Sunada Cartesian
frame (L, Xv, Yv via explore_12_harmonic_geometry), the three cut directions (axis, <111>, <110>
via cut_normal_for), d_b = 0.5/v_iso, the Dirac-sea vacuum fill (E < -1-1e-9), the
shared-eigh-per-M dense ladder pattern, positions_array/physical_bonds.

DISCLOSED DEVIATIONS / DESIGN CHOICES (binding to record, not to hide -- the corpus's own standing
practice, see D1b SS"DISCLOSED DEVIATIONS"):
  (a) PLACEMENT-SHIFT CONVENTION: a placement p is characterized by delta_p = (p - threshold_mid),
      the SIGNED offset (along +cut_normal) of the placement from the geometric midway threshold
      (proj.min()+proj.max())/2 (D1b/D1c's own convention, reused verbatim for REGION MEMBERSHIP --
      region A never changes with placement, only the coordinate origin used to weight K_template
      does).  delta_midway=0; delta_edge=-0.5*d_b (the historical "edge-site" convention: D1c found
      the OLD benchmark_bw_2pi's 0.9988x2pi corresponds to a threshold exactly 0.5*d_b INSIDE the
      midway threshold); delta_derived = the mean, across entanglement-carrying Leg-B mode pairs,
      of x0* in the SAME (proj-threshold_mid) sign convention -- i.e. delta_derived is directly
      Leg B's own found offset, no sign flip.  Positions at placement p: x_p = x_mid + delta_p
      (x_mid = threshold_mid - proj, positive inside A).  K_template(p) is recomputed DIRECTLY per
      placement (k_boost_parabolic on x_p; the parabola is quadratic, so ed61816's linear shift
      identity K(p) = K0 + delta*H no longer applies and is no longer used).  ell (the parabola's
      depth parameter) = the REGION's geometric proper depth threshold_mid - proj.min() (the D1b
      region_depth, a placement-INDEPENDENT property of the region, per the amendment's "FIXED by
      the region's geometry ... never fitted").
  (b) LEG B RUNS AT ONE M (the PRIMARY/largest rung, M=14) -- the pre-reg names no M-ladder for Leg
      B (only "the three cut directions"); M=14 is the best-resolution rung available, disclosed as
      the natural choice (D1c's own PRIMARY-M precedent).  The found delta_derived per direction is
      then REUSED across the whole M-ladder's "derived" placement column (a bond-scale offset is
      not expected to depend on region size) -- disclosed, not silently assumed.
  (c) LEG A's R1 CONTROL uses a GENUINE finite open tight-binding chain (H=tridiagonal, hop=1,
      L=800, Dirac-sea fill E<0 -- an EXACT rank-L/2 projector by construction), NOT
      net.chain_vacuum(800) directly (chain_vacuum is an IDEALIZED infinite-chain correlator
      restricted to a window, not an exact projector -- unsuitable for the mirror-mode machinery's
      purity-exact Schmidt identity).  Under the AMENDMENT the R2 gate ALSO binds on this genuine
      open-chain projector (amendment item 3); chain_vacuum(800) is reported alongside, non-gating.
  (d) THE SECTOR is net.entanglement_carrying_selection(lam, delta=1e-8) everywhere (Leg B pair
      selection, Leg C projection, the L0b count, both benchmarks) -- the amendment's corrected
      definition 1, verbatim.  near_surface_selection (the superseded frac=0.5 definition) is no
      longer called anywhere in this station; it remains in Section 9 untouched (accretion law).
  (e) THE L0b READ: PRIMARY = net.positive_spectrum_dimension on the VACUUM entanglement-carrying
      sector's eps (the amendment's explicit "positive-spectrum counts" wording, W3a forced-zero
      exclusion built in, zero_tol=1e-6 declared here before any lattice number is seen);
      SECONDARY (continuity with the original freeze's operationalization) =
      net.distinct_spectrum_count on the same sector eps, rel_tol=1e-3.  The TICK-REDUCES
      precondition reads the PRIMARY count.
  (f) BRANCH 3's "three-placement c-spread" (the OR-disjunct under placement-underdetermined) is
      read as: pool c*/2pi over {midway, edge, derived-candidate} x all 3 directions at M_PRIMARY,
      vacuum state; check whether [min,max] of that pooled set sits strictly inside [0.44,1.56].
  (g) beta_natural is READ from net.history_side_flow_generator(1)['beta_natural'] (I2b's own
      anchored 6.4874417297), never re-derived or hardcoded; the KMS occupation uses the SAME
      chemical-potential convention (node=-1.0) the corpus's own vacuum/KMS split already uses
      (ML-1'''/cone_velocity's own default node).

HARD RULES / POISONS (binding, from the pre-reg + the amendment): no 2pi/pi/4pi inserted anywhere;
no placement/window/tolerance chosen after numbers are seen; no other sector fraction/threshold or
template may be substituted after numbers are seen; the retired "+7%" and the 0.85-1.01x2pi
first-bond history are NEVER priors or checks; the ML-1d return SS7 QA-bypass numbers are NOT
priors, NOT checks, and are never cited here; vacuum/KMS NEVER averaged; M=6 NEVER used;
D1/D1b/D1c are NOT re-run; the_net.py extension is accretion-only (verified at V-REGRESSION);
quarantined _field_algebra_a4_rep/spin_lift are NOT touched; species/M_Z/ppm appear nowhere; no
pattern-match of a non-2pi number to any other named constant; no scoreboard value moves
regardless of verdict; SECOND gate fail => INSTRUMENT-CLOSED, no third redesign inside Push 3.
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
import explore_12_harmonic_geometry as ex12  # noqa: E402  (re-runs its own diagnostic on FIRST
                                              # import; identical pattern to D1b/W2-D1c/D1)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
TWO_PI = 2 * math.pi
T_WALL_START = time.time()
FAST = "--fast" in sys.argv
ok_all = True
NUMBERS = {}   # every gate/tolerance/measured value, accumulated for the return doc


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


banner("ML-1d-b -- THE DERIVED HORIZON + THE OPERATOR-LEVEL BW READ (CORRECTED INSTRUMENT)"
       + ("  [--fast]" if FAST else ""))
print("Pre-reg (FROZEN): internal research notes (commit d41e286)")
print("Amendment (FROZEN): internal research notes (commit 27b0916)")
print("Prior art (read, NOT re-run): D1/D1b/D1c, ML-1/ML-1'/ML-1''/ML-1''' (proofs/foundations/)")

# ================================================================================================
banner("THE CANONICAL FRAME  --  derived on-screen from L (explore_12) and g_frac (the_net)")
# ================================================================================================
L = ex12.L
Xv = ex12.Xv
Linv = np.linalg.inv(L)
g_frac = net.emergent_metric()
g_cart = L @ g_frac @ L.T
ev_cart = np.linalg.eigvalsh(g_cart)
v_iso = math.sqrt(float(np.mean(ev_cart)))
d_b = 0.5 / v_iso
print(f"v_iso = {v_iso:.8f}   d_b = 0.5/v_iso = {d_b:.8f}  (one NN bond length, proper units)")
NUMBERS["v_iso"] = v_iso
NUMBERS["d_b"] = d_b
BETA_NATURAL = net.history_side_flow_generator(1)["beta_natural"]
print(f"beta_natural (I2b's own anchored constant, read not re-derived) = {BETA_NATURAL:.10f}")
NUMBERS["beta_natural"] = BETA_NATURAL


def cut_normal_for(e_frac):
    n = Linv.T @ np.asarray(e_frac, dtype=float)
    return n / np.linalg.norm(n)


DIRECTIONS = {"axis": (1, 0, 0), "<111>": (1, 1, 1), "<110>": (1, 1, 0)}
NORMALS = {name: cut_normal_for(e) for name, e in DIRECTIONS.items()}
for name, e in DIRECTIONS.items():
    print(f"  direction {name:5s}  e_frac={e}  cut_normal = {np.round(NORMALS[name], 6)}")


def positions_array(patch):
    cells = np.repeat(np.asarray(patch.box, dtype=float), srs.NV, axis=0)
    Xv_tile = np.tile(Xv, (len(patch.box), 1))
    return Xv_tile + cells @ L.T


def physical_bonds(patch):
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
banner("V-REGRESSION  --  the net's own anchors/reads + Section 9 + Section 9b self-tests")
# ================================================================================================
net_ok = net.self_test(verbose=False)
sec9b_ok = net.ml1db_selftest_2026_07_13(verbose=False)   # chains ml1d_selftest (Section 9) itself
v_reg = check("net.self_test() + ml1db_selftest_2026_07_13() (accretion-only, Sections 1-9b) PASS",
              net_ok and sec9b_ok)
if not v_reg:
    print("\n*** V-REGRESSION FAILED -- STOPPING. ***")
    sys.exit(1)

# ================================================================================================
banner("V-DISCLOSE (a')  --  k_boost_parabolic spot-checked against an independent elementwise "
       "computation (the linear shift identity of ed61816 is NO LONGER USED: w is quadratic)")
# ================================================================================================
_rng0 = np.random.default_rng(1)
_HAtest = _rng0.normal(size=(30, 30))
_HAtest = (_HAtest + _HAtest.T) / 2.0
_xtest = np.sort(np.abs(_rng0.normal(size=30))) * 3.0
_ell_test = float(_xtest.max())
_Kpar = net.k_boost_parabolic(_HAtest, _xtest, _ell_test)
_Kind = np.zeros_like(_HAtest)
for _i in range(30):
    for _j in range(30):
        _m = (_xtest[_i] + _xtest[_j]) / 2.0
        _Kind[_i, _j] = _HAtest[_i, _j] * (_m * (_ell_test - _m) / _ell_test)
v_disc_a = check("k_boost_parabolic == independent elementwise w((x_i+x_j)/2) computation",
                  np.allclose(_Kpar, _Kind, atol=1e-12),
                  detail=f"max|diff|={np.max(np.abs(_Kpar-_Kind)):.2e}")

# ================================================================================================
banner("LEG A -- INSTRUMENT GATES  (amendment item 3: the R2 gate binds on the GENUINE finite "
       "open-chain projector; chain_vacuum(800) reported alongside, non-gating)")
# ================================================================================================


def entanglement_sector_data(C_A):
    """Shared amended core: eigh(C_A) -> the ENTANGLEMENT-CARRYING sector (amendment definition 1)
    -> (h_proj (diagonal, sector eps), Vn (sector eigenbasis columns), eps_sector, lam (full
    occupation spectrum), sector_idx)."""
    w, V = np.linalg.eigh(C_A)
    lam = w.real
    sector_idx = net.entanglement_carrying_selection(lam, delta=1e-8)
    zeta = np.clip(lam[sector_idx], 1e-14, 1 - 1e-14)
    eps_sector = np.log((1 - zeta) / zeta)
    Vn = V[:, sector_idx]
    h_proj = np.diag(eps_sector)
    return h_proj, Vn, eps_sector, lam, sector_idx


def c_star_at_placement(HA, x_p, ell, Vn, h_proj):
    """Amended Leg-C read at ONE placement: K_template = k_boost_parabolic (amendment definition
    2), recomputed DIRECTLY from the placement-frame positions x_p, projected onto the
    entanglement-carrying sector basis Vn; returns (c_star, residual)."""
    K = net.k_boost_parabolic(HA, x_p, ell)
    K_proj = net.project_matrix(K, Vn)
    return net.boost_c_star(h_proj, K_proj)


print("R2 GATE (hard, AMENDED)  --  GENUINE finite open-chain ground-state projector, L=800, EDGE "
      "placement, amended Leg-C operator test must recover c/2pi within 2%")
Lc = 800
Hchain_full = np.diag(np.ones(Lc - 1), 1) + np.diag(np.ones(Lc - 1), -1)
Ec1, Vc1 = np.linalg.eigh(Hchain_full)
cols_c1 = Vc1[:, Ec1 < 0.0]                                       # exact half-filling (L even: no E=0)
Cfull_c1 = cols_c1 @ cols_c1.conj().T                             # EXACT projector by construction
A_chain = np.arange(0, Lc // 2)                                   # sites 0..399 (D1c convention)
Ap_chain = np.arange(Lc // 2, Lc)
threshold_mid_chain = (0.0 + (Lc - 1)) / 2.0                      # 399.5
threshold_edge_chain = threshold_mid_chain - 0.5 * 1.0            # d_b_chain = 1 exactly; = 399.0
C_A_open = Cfull_c1[np.ix_(A_chain, A_chain)]
HA_chain = Hchain_full[np.ix_(A_chain, A_chain)]
x_mid_chain = threshold_mid_chain - A_chain.astype(float)         # positive-inside-A convention
x_edge_chain = threshold_edge_chain - A_chain.astype(float)       # = x_mid_chain - 0.5
ell_chain = float(threshold_mid_chain - 0.0)                      # region proper depth = 399.5
h_proj_g, Vn_g, eps_g, lam_g, sector_g = entanglement_sector_data(C_A_open)
print(f"  entanglement-carrying sector dim = {len(sector_g)}/{len(A_chain)}  "
      f"(state-determined; O(log L) as expected on a 1D-critical benchmark)")
c_star_gate, resid_gate = c_star_at_placement(HA_chain, x_edge_chain, ell_chain, Vn_g, h_proj_g)
r2pi_gate = c_star_gate / TWO_PI
c_star_gate_mid, resid_gate_mid = c_star_at_placement(HA_chain, x_mid_chain, ell_chain, Vn_g, h_proj_g)
print(f"  EDGE placement (THE GATE): c* = {c_star_gate:.6f} = {r2pi_gate:.6f} x 2pi  "
      f"(operator residual = {resid_gate:.4f})")
print(f"  midway placement (report): c*/2pi = {c_star_gate_mid/TWO_PI:.6f}  "
      f"(residual = {resid_gate_mid:.4f})")
NUMBERS["legA_R2_gate_c_over_2pi"] = r2pi_gate
NUMBERS["legA_R2_gate_residual"] = resid_gate
NUMBERS["legA_R2_gate_sector_dim"] = int(len(sector_g))
NUMBERS["legA_R2_gate_midway_c_over_2pi"] = c_star_gate_mid / TWO_PI
gate_pass = check("LEG A R2 HARD GATE (amended): |c*/2pi - 1| <= 0.02 on the genuine open-chain "
                   "projector (L=800), edge placement",
                   abs(r2pi_gate - 1.0) <= 0.02, detail=f"{r2pi_gate:.6f}, |dev|={abs(r2pi_gate-1.0):.4f}")

print("\n  chain_vacuum(800) CONTINUITY READ (amendment item 3: reported alongside, NON-GATING):")
C_chain_cv = net.chain_vacuum(Lc)
C_A_cv = C_chain_cv[np.ix_(A_chain, A_chain)]
h_proj_cv, Vn_cv, eps_cv, lam_cv, sector_cv = entanglement_sector_data(C_A_cv)
c_cv_edge, r_cv_edge = c_star_at_placement(HA_chain, x_edge_chain, ell_chain, Vn_cv, h_proj_cv)
c_cv_mid, r_cv_mid = c_star_at_placement(HA_chain, x_mid_chain, ell_chain, Vn_cv, h_proj_cv)
print(f"  chain_vacuum sector dim = {len(sector_cv)}/{len(A_chain)}; "
      f"edge c*/2pi = {c_cv_edge/TWO_PI:.6f} (resid={r_cv_edge:.4f}), "
      f"midway c*/2pi = {c_cv_mid/TWO_PI:.6f} (resid={r_cv_mid:.4f})")
print("  (chain_vacuum is an infinite-chain window, NOT an exact projector -- its interval "
      "carries entangling cuts at BOTH window ends, a structurally different region than the "
      "boundary-touching genuine-open-chain half; reported raw, no interpretation forced.)")
NUMBERS["legA_R2_chain_vacuum_edge_c_over_2pi"] = c_cv_edge / TWO_PI
NUMBERS["legA_R2_chain_vacuum_midway_c_over_2pi"] = c_cv_mid / TWO_PI
NUMBERS["legA_R2_chain_vacuum_sector_dim"] = int(len(sector_cv))

print("\n  MECHANISTIC DISCLOSURE (printed regardless of pass/fail; computed AFTER the gate line "
      "above, never substituted for it): the per-bond entanglement-hopping profile "
      "beta_b = |h_A[i,i+1]| vs the parabola template 2pi*w(x_b), genuine open chain, edge frame.")
h_A_open_full = net.entanglement_hamiltonian(C_A_open)
profile_rows = []
for _i in [399, 398, 397, 396, 394, 390, 380, 300, 200, 100]:
    _xb = (x_edge_chain[_i] + x_edge_chain[_i - 1]) / 2.0
    _beta = abs(h_A_open_full[_i, _i - 1])
    _w = _xb * (ell_chain - _xb) / ell_chain
    profile_rows.append((_i, _xb, _beta, TWO_PI * _w))
    print(f"    bond({_i-1},{_i})  x_b={_xb:8.2f}  beta_b={_beta:10.5f}  2pi*w(x_b)={TWO_PI*_w:10.5f}"
          f"  ratio={_beta/(TWO_PI*_w):8.4f}")
NUMBERS["legA_R2_first_bond_ratio_beta_over_2piw"] = profile_rows[0][2] / profile_rows[0][3]
print("  (the x->0 first-bond ratio is the historical edge-convention calibration point; the "
      "deeper bonds show how the EXACT lattice h_A profile compares against the conformal "
      "parabola at this region depth -- raw disclosure, not a fit input.)")

print("\n  R1 CONTROL (report, not gate)  --  the mirror-mode placement extractor on the SAME "
      "genuine finite open chain, pair selection = the AMENDED entanglement-carrying sector")
C_ApA_c1 = Cfull_c1[np.ix_(Ap_chain, A_chain)]
mp_c1 = net.mirror_mode_pairing(C_A_open, C_ApA_c1)
print(f"  mirror_mode_pairing residuals: mirror_norm={mp_c1['mirror_norm_residual']:.2e}, "
      f"purity={mp_c1['purity_residual']:.2e}  (a GENUINE finite projector: both near machine eps)")
sector_r1 = net.entanglement_carrying_selection(mp_c1["zeta_A"], delta=1e-8)
sd_A_c1 = A_chain.astype(float) - threshold_mid_chain
sd_Ap_c1 = Ap_chain.astype(float) - threshold_mid_chain
posA_c1 = net.mode_position_expectation(mp_c1["U_A"][:, sector_r1], sd_A_c1)
posAp_c1 = net.mode_position_expectation(mp_c1["mirror_unnorm"][:, sector_r1], sd_Ap_c1)
x0star_c1 = (posA_c1 + posAp_c1) / 2.0
spread_c1 = float(np.max(x0star_c1) - np.min(x0star_c1))
mean_c1 = float(np.mean(x0star_c1))
print(f"  {len(sector_r1)} entanglement-carrying pairs; x0* range=[{x0star_c1.min():.2e},"
      f"{x0star_c1.max():.2e}] (sd units, d_b_chain=1); spread={spread_c1:.2e}  "
      f"(landing tol 0.15*d_b=0.15); mean x0* = {mean_c1:.2e}")
print(f"  reference points in this frame: midway=0.0, edge={-0.5*1.0:.4f}")
r1_landed = spread_c1 < 0.15 * 1.0
NUMBERS["legA_R1_spread_over_db"] = spread_c1
NUMBERS["legA_R1_mean_x0star"] = mean_c1
NUMBERS["legA_R1_n_pairs"] = int(len(sector_r1))
check("LEG A R1 CONTROL (report only): "
      + ("the 1D placement now LANDS -- reports what it fixes (D1c's 'underdetermined' was the "
         "polluted-sector artifact)" if r1_landed
         else "reproduces D1c's own finding (UNDERDETERMINED)"),
      True, detail=f"spread={spread_c1:.2e} vs tol 0.15  "
      + (f"(LANDED at x0*={mean_c1:.2e} = the MIDWAY convention; on THIS benchmark the landing "
         f"point is reflection-symmetry-forced -- the open chain is mirror-symmetric about the "
         f"cut, so a clean pairing MUST midpoint at the symmetry plane; the informative part is "
         f"that the pairing is now CLEAN, not where it lands)" if r1_landed else "(not landed)"))

if not (gate_pass and v_disc_a):
    print("\n" + "*" * 96)
    print("*** LEG A R2 GATE FAILED AGAIN (second frozen instrument definition) -- per the "
          "amendment: 'FAIL => STOP again => the operator-instrument line is booked "
          "INSTRUMENT-CLOSED (two frozen definitions failed the same gate; no third redesign "
          "inside Push 3 -- the W2/W4 normalization route becomes the named remaining supplier), "
          "and nothing on the lattice is read.'  STOPPING HERE. ***")
    print("*" * 96)
    print(f"\ntotal station wall time: {elapsed():.1f}s ({elapsed()/60.0:.2f} min)")
    print("RESULT: LEG A R2 GATE FAILED (AMENDED INSTRUMENT) -- INSTRUMENT-CLOSED; the lattice "
          "measurement was NOT run.")
    sys.exit(1)

# ================================================================================================
banner("THE MEASUREMENT  --  dense M-ladder {10,12,14}, 3 directions, shared eigh per M "
       "(Leg B at M=14 only; Leg C across the full ladder)")
# ================================================================================================
FULL_MS = [14, 12, 10]           # largest FIRST so Leg B's derived offsets are known before reuse
Ms = [12, 10] if FAST else FULL_MS
M_PRIMARY = 14
dir_names_full = ["axis", "<111>", "<110>"]
dir_names = ["axis", "<111>"] if FAST else dir_names_full
states = ["vacuum"] if FAST else ["vacuum", "kms"]
print(f"M-ladder: {sorted(Ms)}{'  (--fast: M<=12)' if FAST else ''}")
print(f"directions: {dir_names}{'  (--fast: axis + <111> only)' if FAST else ''}")
print(f"states: {states}{'  (--fast: vacuum only)' if FAST else ''}")

PLACEMENT_NAMES = ["midway", "edge", "derived"]
leg_c_results = {}      # (M,direction,state,placement) -> (c_star, residual)
leg_b_results = {}      # direction -> dict(x0star, spread, landed, delta_derived)
l0b_results = {}        # (M,direction) -> dict(region_size, surface_bonds, distinct_count)
region_depths = {}      # (M,direction) -> depth

for M in Ms:
    t0 = time.time()
    patch = net.Patch(M=M, skip_pair_bfs=True)
    H, verts = patch.vertex_adjacency()
    t_build = time.time()
    E, V = np.linalg.eigh(H)
    t_eig = time.time()
    X_proper = positions_array(patch) / v_iso
    all_bonds = physical_bonds(patch)
    cols_vac = V[:, E < -1.0 - 1e-9]
    C_vac = cols_vac @ cols_vac.conj().T
    t_cvac = time.time()
    C_kms = None
    if "kms" in states:
        occ = 1.0 / (np.exp(BETA_NATURAL * (E - (-1.0))) + 1.0)
        C_kms = (V * occ) @ V.conj().T
    t_ckms = time.time()
    print(f"  M={M:2d}  N={len(verts):6d}  fill={cols_vac.shape[1]:6d} "
          f"({100.0*cols_vac.shape[1]/len(verts):.1f}%)  "
          f"[build {t_build-t0:.2f}s eigh {t_eig-t_build:.2f}s Cvac {t_cvac-t_eig:.2f}s "
          f"Ckms {t_ckms-t_cvac:.2f}s]")

    for name in dir_names:
        cut_normal = NORMALS[name]
        proj = X_proper @ cut_normal
        threshold_mid = (proj.min() + proj.max()) / 2.0
        inA = proj < threshold_mid
        A_idx = np.where(inA)[0]
        Ap_idx = np.where(~inA)[0]
        region_depth = threshold_mid - proj.min()
        region_depths[(M, name)] = region_depth
        x_mid = threshold_mid - proj[A_idx]
        HA = H[np.ix_(A_idx, A_idx)]
        surface_bonds = 0
        for (gi, gj) in all_bonds:
            if inA[gi] != inA[gj]:
                surface_bonds += 1

        ell_region = float(region_depth)                      # the parabola's FIXED depth parameter
        state_data = {}
        for state in states:
            C_state = C_vac if state == "vacuum" else C_kms
            C_A = C_state[np.ix_(A_idx, A_idx)]
            h_proj, Vn_s, eps_sector, lam_s, sector_idx = entanglement_sector_data(C_A)
            state_data[state] = dict(h_proj=h_proj, Vn=Vn_s, eps_sector=eps_sector,
                                      sector_dim=int(len(sector_idx)))
            if state == "vacuum":
                psd = net.positive_spectrum_dimension(eps_sector, region_dim=len(A_idx))
                dcount = net.distinct_spectrum_count(eps_sector, rel_tol=1e-3)
                l0b_results[(M, name)] = dict(region_size=len(A_idx), surface_bonds=surface_bonds,
                                               sector_dim=int(len(sector_idx)),
                                               n_positive=psd["n_positive"],
                                               has_zero_mode=psd["has_zero_mode"],
                                               zero_forced=psd["zero_forced_by_odd_dim"],
                                               distinct_count=dcount)

        # LEG B -- mirror-mode placement, M_PRIMARY only, vacuum only (amended pair selection:
        # the entanglement-carrying sector)
        if M == M_PRIMARY:
            C_ApA = C_vac[np.ix_(Ap_idx, A_idx)]
            C_A_vac = C_vac[np.ix_(A_idx, A_idx)]
            mp = net.mirror_mode_pairing(C_A_vac, C_ApA)
            sec_b = net.entanglement_carrying_selection(mp["zeta_A"], delta=1e-8)
            UA_near = mp["U_A"][:, sec_b]
            mirror_near = mp["mirror_unnorm"][:, sec_b]
            sd_A = proj[A_idx] - threshold_mid
            sd_Ap = proj[Ap_idx] - threshold_mid
            posA = net.mode_position_expectation(UA_near, sd_A)
            posAp = net.mode_position_expectation(mirror_near, sd_Ap)
            x0star = (posA + posAp) / 2.0
            spread_dir = float(np.max(x0star) - np.min(x0star))
            leg_b_results[name] = dict(x0star=x0star, spread=spread_dir,
                                        delta_derived=float(np.mean(x0star)),
                                        mirror_norm_residual=mp["mirror_norm_residual"],
                                        purity_residual=mp["purity_residual"], n_pairs=len(sec_b))
            print(f"      [LEG B, M={M}] dir={name:5s} {len(sec_b)} entanglement-carrying pairs "
                  f"x0*=[{x0star.min():.4f},{x0star.max():.4f}] spread={spread_dir:.4f} "
                  f"(tol 0.15*d_b={0.15*d_b:.4f})  mirror/purity resid="
                  f"{mp['mirror_norm_residual']:.1e}/{mp['purity_residual']:.1e}")

        # LEG C -- the 3 placements, for every state already built above (amended template:
        # k_boost_parabolic, recomputed DIRECTLY per placement; ell fixed by region geometry)
        delta_derived_dir = leg_b_results.get(name, {}).get("delta_derived", None)
        deltas = {"midway": 0.0, "edge": -0.5 * d_b}
        if delta_derived_dir is not None:
            deltas["derived"] = delta_derived_dir
        for pname, delta_p in deltas.items():
            K_p_full = net.k_boost_parabolic(HA, x_mid + delta_p, ell_region)
            for state in states:
                sd = state_data[state]
                K_proj = net.project_matrix(K_p_full, sd["Vn"])
                c_star, resid = net.boost_c_star(sd["h_proj"], K_proj)
                leg_c_results[(M, name, state, pname)] = (c_star, resid)
            del K_p_full
        c_mid, r_mid = leg_c_results[(M, name, "vacuum", "midway")]
        print(f"      [LEG C] dir={name:5s} |A|={len(A_idx):6d} surf_bonds={surface_bonds:5d} "
              f"sector(vac)={state_data['vacuum']['sector_dim']:5d}"
              + (f" sector(kms)={state_data['kms']['sector_dim']:5d}" if "kms" in states else "")
              + f"  n_pos(vac)={l0b_results[(M,name)]['n_positive']:4d}  "
              f"c*(vac,midway)/2pi={c_mid/TWO_PI:.4f}(resid={r_mid:.3f})")
    print(f"    [M={M} total {time.time()-t0:.2f}s, wall so far {elapsed():.1f}s]")

# ================================================================================================
banner("LEG B SUMMARY -- placement DERIVED?  (spread across near-surface pairs AND across the "
       "three cut directions, < 0.15*d_b)")
# ================================================================================================
if FAST:
    print("  SKIPPED under --fast (Leg B needs all 3 directions at M=14; run the full station.)")
    placement_derived = False
    pooled_spread = float("nan")
else:
    all_x0 = np.concatenate([leg_b_results[d]["x0star"] for d in dir_names_full])
    pooled_spread = float(np.max(all_x0) - np.min(all_x0))
    for d in dir_names_full:
        r = leg_b_results[d]
        print(f"  {d:5s}: {r['n_pairs']} pairs, per-direction spread={r['spread']:.4f}, "
              f"mean x0*={r['delta_derived']:.4f}")
    print(f"  POOLED spread across all near-surface pairs and all 3 directions = {pooled_spread:.4f} "
          f"(tol 0.15*d_b = {0.15*d_b:.4f})")
    placement_derived = pooled_spread < 0.15 * d_b
    NUMBERS["legB_pooled_spread"] = pooled_spread
    NUMBERS["legB_pooled_spread_tol"] = 0.15 * d_b
    check(f"LEG B: placement {'DERIVED' if placement_derived else 'UNDERDETERMINED'} "
          f"(pooled spread {pooled_spread:.4f} vs tol {0.15*d_b:.4f})", True)

# ================================================================================================
banner("L0b RANK/PROFILE READ  --  the dimension of the (vacuum) POSITIVE modular spectrum within "
       "the entanglement-carrying sector (W3a forced-zero excluded): ENRICHES with region growth "
       "(boost-like) or RANK-STARVED (tick-like)?")
# ================================================================================================
for name in dir_names:
    row = [(M, l0b_results[(M, name)]) for M in sorted(Ms)]
    print(f"  {name:5s}: " + "  ".join(
        f"M={M}:|A|={r['region_size']},surf={r['surface_bonds']},sector={r['sector_dim']},"
        f"n_pos={r['n_positive']}{'(zero excl.)' if r['has_zero_mode'] else ''},"
        f"distinct={r['distinct_count']}"
        for M, r in row))
axis_counts = [l0b_results[(M, "axis")]["n_positive"] for M in sorted(Ms)]
axis_distinct = [l0b_results[(M, "axis")]["distinct_count"] for M in sorted(Ms)]
rank_enriches = len(axis_counts) >= 2 and axis_counts[-1] > axis_counts[0]
NUMBERS["l0b_axis_n_positive_by_M"] = dict(zip(sorted(Ms), axis_counts))
NUMBERS["l0b_axis_distinct_counts_by_M"] = dict(zip(sorted(Ms), axis_distinct))
check(f"L0b (PRIMARY, positive-spectrum dim): axis n_positive vs M "
      f"{dict(zip(sorted(Ms), axis_counts))} -- "
      f"{'ENRICHES (grows with M, boost-like)' if rank_enriches else 'STAYS FLAT/SHRINKS (rank-starved, tick-like)'}"
      f"; SECONDARY (continuity, distinct magnitudes): {dict(zip(sorted(Ms), axis_distinct))}",
      True)

# ================================================================================================
banner("DIAMOND DIAGNOSTIC  (report-only, NO verdict weight)  --  exact causal diamonds at two "
       "radii vs the half-space boost's near-surface structure")
# ================================================================================================
M_diamond = min(Ms) if Ms else 10
patch_d = net.Patch(M=M_diamond, skip_pair_bfs=True)
H_d, verts_d = patch_d.vertex_adjacency()
E_d, V_d = np.linalg.eigh(H_d)
cols_d = V_d[:, E_d < -1.0 - 1e-9]
C_d = cols_d @ cols_d.conj().T
base_d = patch_d.central_dart()
diamond_summ = {}
for depth in (2, 4):
    region = net.diamond_vertex_region(patch_d, base_d, depth)
    if len(region) < 4:
        print(f"  depth={depth}: region too small ({len(region)} vertices) -- skipped")
        continue
    C_A_diam = C_d[np.ix_(region, region)]
    w_diam, V_diam = np.linalg.eigh(C_A_diam)
    lam_diam = w_diam.real
    sec_diam = net.entanglement_carrying_selection(lam_diam, delta=1e-8)
    zeta_diam = np.clip(lam_diam[sec_diam], 1e-14, 1 - 1e-14)
    eps_diam = np.log((1 - zeta_diam) / zeta_diam)
    psd_diam = net.positive_spectrum_dimension(eps_diam, region_dim=len(region))
    dcount_diam = net.distinct_spectrum_count(eps_diam, rel_tol=1e-3)
    diamond_summ[depth] = dict(region_size=len(region), sector_dim=int(len(sec_diam)),
                                n_positive=psd_diam["n_positive"], distinct_count=dcount_diam)
    print(f"  depth={depth}: |region|={len(region):4d}  sector={len(sec_diam):4d}  "
          f"n_pos={psd_diam['n_positive']:4d}  distinct(eps)={dcount_diam}")
axis_ref = l0b_results.get((M_diamond, "axis"))
if axis_ref is not None and diamond_summ:
    print(f"  half-space reference @ M={M_diamond}, axis: |A|={axis_ref['region_size']}, "
          f"sector={axis_ref['sector_dim']}, n_pos={axis_ref['n_positive']}, "
          f"distinct(eps)={axis_ref['distinct_count']}")
    print("  (report-only comparison: diamond regions are far smaller than the half-space cut; "
          "a qualitatively similar or smaller distinct-magnitude count is expected either way -- "
          "no verdict weight is assigned to this diagnostic, per the frozen pre-reg.)")
check("DIAMOND DIAGNOSTIC recorded (report only; NOT gating)", True)
NUMBERS["diamond_diagnostic"] = diamond_summ

# ================================================================================================
banner("LEG C FULL TABLE  --  c*/2pi (residual) per (M, direction, state, placement)")
# ================================================================================================
for M in sorted(Ms):
    for name in dir_names:
        for state in states:
            row = []
            for pname in PLACEMENT_NAMES:
                key = (M, name, state, pname)
                if key in leg_c_results:
                    c_star, resid = leg_c_results[key]
                    row.append(f"{pname}:{c_star/TWO_PI:.4f}(r={resid:.3f})")
            print(f"  M={M:2d} {name:5s} {state:6s}: " + "  ".join(row))

# ================================================================================================
banner("V-VERDICT  --  the frozen four-branch tree, M_PRIMARY=%d, vacuum unless stated" % M_PRIMARY)
# ================================================================================================
BRACKET = (0.44, 1.56)


def cbar_and_lorentz(state, pname):
    """mean/direction spread of c*/2pi at M_PRIMARY for a given (state, placement); None if any
    direction is missing (e.g. --fast or 'derived' not available)."""
    vals = {}
    for name in dir_names_full:
        key = (M_PRIMARY, name, state, pname)
        if key not in leg_c_results:
            return None
        vals[name] = leg_c_results[key][0] / TWO_PI
    cbar = float(np.mean(list(vals.values())))
    if cbar == 0.0:
        lorentz = float("inf")
    else:
        lorentz = max(abs(v - cbar) / abs(cbar) for v in vals.values())
    return dict(vals=vals, cbar=cbar, lorentz=lorentz)


if FAST:
    print("  SKIPPED under --fast (the verdict needs the full M-ladder x 3-direction grid; run the "
          "full station for a definite verdict).")
    verdict = "NOT-EVALUATED-FAST-MODE"
else:
    primary_pname = "derived" if placement_derived else "midway"
    cl_vac = cbar_and_lorentz("vacuum", primary_pname)
    print(f"  primary placement for the verdict = '{primary_pname}' "
          f"(placement_derived={placement_derived})")
    if cl_vac is not None:
        print(f"  c_d (vacuum, {primary_pname}, M={M_PRIMARY}): " +
              ", ".join(f"{n}={v:.6f}" for n, v in cl_vac["vals"].items()))
        print(f"  c_bar = {cl_vac['cbar']:.6f}   Lorentz max_d|c_d-cbar|/cbar = {cl_vac['lorentz']:.4f} "
              f"(gate < 0.05)")
        NUMBERS["cbar_vacuum_primary_placement"] = cl_vac["cbar"]
        NUMBERS["lorentz_dev_vacuum_primary_placement"] = cl_vac["lorentz"]

    # rank/profile + "tracks the global tick" diagnostic (V-DISCLOSE: operationalized via the SAME
    # rank/profile read, per the pre-reg's own pairing of the two clauses under one arrow "=>").
    tick_reduces = not rank_enriches
    check(f"TICK-REDUCES precondition (rank/profile read): "
          f"{'the modular spectrum does NOT enrich' if tick_reduces else 'the modular spectrum ENRICHES with M'}",
          True)

    branch1 = False
    branch2 = tick_reduces
    branch3a = False
    branch3b = False
    branch4 = False
    between_branches = False

    if tick_reduces:
        verdict = "TICK-REDUCES"
    elif placement_derived and cl_vac is not None and cl_vac["lorentz"] < 0.05:
        if abs(cl_vac["cbar"] - 1.0) <= 0.03:
            branch1 = True
            verdict = "PLACEMENT-DERIVED + BW-2PI-FORCED"
        else:
            branch3a = True
            verdict = "MISS-QUANTIFIED (placement derived, Lorentz gate holds, |cbar/2pi-1|>0.03)"
    elif not placement_derived:
        # pooled three-placement c-spread @ M_PRIMARY, vacuum, all directions x {midway,edge,derived-candidate}
        pooled_vals = []
        for name in dir_names_full:
            for pname in PLACEMENT_NAMES:
                key = (M_PRIMARY, name, "vacuum", pname)
                if key in leg_c_results:
                    pooled_vals.append(leg_c_results[key][0] / TWO_PI)
        pooled_lo, pooled_hi = (min(pooled_vals), max(pooled_vals)) if pooled_vals else (float("nan"),) * 2
        print(f"  three-placement pooled c*/2pi range @ M={M_PRIMARY}, vacuum, all directions: "
              f"[{pooled_lo:.4f}, {pooled_hi:.4f}]  (bracket {BRACKET})")
        NUMBERS["pooled_placement_c_range"] = [pooled_lo, pooled_hi]
        if pooled_vals and BRACKET[0] < pooled_lo and pooled_hi < BRACKET[1]:
            branch3b = True
            verdict = "BRACKET-NARROWED (placement underdetermined, 3-placement spread inside [0.44,1.56])"
        else:
            branch4 = True
            verdict = "PLACEMENT-UNDERDETERMINED (BLOCKED)"
    else:
        # placement derived but the Lorentz gate FAILS: NOT a literal branch of the frozen tree.
        between_branches = True
        verdict = "BETWEEN-BRANCHES (placement derived, but the Lorentz direction-independence gate " \
                  "FAILS -- not literally covered by the frozen four-branch tree; booked raw, no " \
                  "branch invented)"

    print(f"\n>>> V-VERDICT: {verdict} <<<")
    if between_branches:
        print("  NOTE: the frozen tree does not name this combination (placement DERIVED, Lorentz "
              "gate FAILS). Per instruction, this is reported literally rather than stretched into "
              "an existing branch or a new fifth branch. The closest branches in spirit are 3 "
              "(a quantified miss) and 4 (blocked); neither is claimed.")
    NUMBERS["verdict"] = verdict

# ================================================================================================
banner("SUMMARY")
# ================================================================================================
t_total = elapsed()
print(f"""    Leg A R2 GATE (hard, amended: open-chain projector) ... {r2pi_gate:.6f} x2pi  (PASS)
    Leg A chain_vacuum continuity (non-gating) ............. edge {c_cv_edge/TWO_PI:.4f} / midway {c_cv_mid/TWO_PI:.4f} x2pi
    Leg A R1 CONTROL (report) .............................. spread={spread_c1:.2e} vs 0.15  ({'LANDED at midway' if r1_landed else 'underdetermined, as D1c found'})
    Leg B placement ........................................ {'SKIPPED (--fast)' if FAST else ('DERIVED' if placement_derived else 'UNDERDETERMINED')}{'' if FAST else f' (pooled spread={pooled_spread:.4f} vs {0.15*d_b:.4f})'}
    L0b rank/profile (positive-spectrum dim, W3a) .......... axis n_pos vs M = {dict(zip(sorted(Ms), axis_counts))}
    V-VERDICT: {verdict}
    total station wall time: {t_total:.1f}s ({t_total/60.0:.2f} min)
""")

core_pass = gate_pass and v_reg and v_disc_a
print("RESULT:", "CORE CONTRACTS (V-REGRESSION / LEG-A R2 GATE / V-DISCLOSE) PASS -- a definite "
      "verdict was reached" if core_pass else "A CORE CONTRACT FAILED -- inspect above")
print(f"(the V-VERDICT {verdict if not FAST else '(fast mode, not evaluated)'} is a scientific "
      f"finding, not a script failure; exit code reflects only whether the core contracts passed)")
sys.exit(0 if core_pass else 1)
