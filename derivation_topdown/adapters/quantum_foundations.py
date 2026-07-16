#!/usr/bin/env python3
"""
derivation_topdown/adapters/quantum_foundations.py

G7 ADAPTER -- the QUANTUM-FOUNDATIONS GRAFT: Born rule, CHSH/Tsirelson, decoherence/pointer.
Pre-registered in internal research notes (frozen BEFORE this
file). Companion charter: internal research notes; protocol:
internal research notes (this file = pipeline step 3, IMPLEMENTATION).
The Symphony gate's LAST station.

WHAT THIS FILE IS: an ADAPTER, not a new derivation, EXCEPT for QF-2 which is explicitly a NEW
COMPUTATION (the first Bell/CHSH read on the object). Three contract families, each with its
honest classification stated: QF-1 Born = THEOREM-CHECK (conditional on A3, the framework's
adopted purification axiom -- conditionality printed, never hidden); QF-2 CHSH = NEW COMPUTATION
(dual-outcome + a crisis branch, frozen thresholds); QF-3 decoherence/pointer = THEOREM
RE-EXPRESSION (phase3_1/phase3_2 results re-expressed as contracts). Falsification probes per the
G5a standard throughout (a probe that FAILS to fail is itself a contract failure).

THE CONTRACTS (frozen; internal research notes verbatim):
  QF-0 ANCHORS               -- tick-2pi (net.anchor_tick_2pi()); the exact-light-cone anchor
                                 (Patch.anticommutator_below_cone == 0.0), both reused; the A3
                                 STATUS LINE printed (never dropped).
  QF-1 THE BORN RULE         -- (a) the run marginal's per-tick ratio == (alpha_1/u_c)^2 at
                                 <1e-12 (M0-2R recipe, reuse verbatim); (b) THE MECHANISM: the
                                 subleading (Ramanujan) modes of B(Gamma) are ORTHOGONAL to the
                                 Perron vector (overlaps <1e-12) and the modular slope ratio
                                 [-dlog p_n/dn]/log(u_c/alpha_1) == 2 exactly (<1e-9); (c)
                                 FALSIFICATION PROBE: deform the measure to ||amp||^(2+delta),
                                 delta=0.1 -- the affine-fit / Gibbs identification against the
                                 FIXED, independently-derived beta_eff must FAIL by a quantified
                                 margin (a passing deformation = contract FAIL).
  QF-2 CHSH / TSIRELSON      -- (a) two disjoint vertex-cluster regions O_A, O_B on Patch(M=5),
                                 separation declared (a far pair + a declared adjacent-region
                                 robustness point); (b) 4 Majorana operators from 2 complex
                                 fermion (site) modes per region; dichotomic A(theta) = cos(theta)
                                 (i*g1*g2) + sin(theta)(i*g1*g3), A^2=I verified <1e-12, grid >=64
                                 angles; (c) correlations via Wick/Pfaffian on the Majorana
                                 covariance, MINI TWO-ROUTE CHECK (Wick == dense many-body,
                                 <1e-10) BEFORE the sweep; commutation [A_i,B_j]=0 re-verified on
                                 the actual operators (<1e-12) on the mini instance, generalized to
                                 the big patch via the disjoint-Majorana-support/twisted-locality
                                 argument (stated, not re-proved on a huge matrix); (d) THE SWEEP +
                                 FROZEN VERDICTS: S maximized over the declared family via the
                                 Horodecki 2x2 correlation-tensor reduction (closed form) AND a
                                 declared angle-grid+refine cross-check.
                                 VIOLATION-FOUND:      2.05 <= S_max <= 2sqrt2+1e-9
                                 NO-VIOLATION-IN-FAMILY: S_max < 2.05  (honest negative, NOT
                                                          evidence of classicality)
                                 TSIRELSON-BREACH:      S_max > 2sqrt2+1e-9  -> STOP, exit 1 loudly
  QF-3 DECOHERENCE/POINTER   -- (a) the derived-GKLS gate (phase3_1's Lindblad identity as
                                 contract); (b) the pointer basis = the record superselection of
                                 the step isometry (phase3_2 S1 identities re-verified); (c)
                                 thermalization to KMS at rate u_c (M0-2R T1's decay fit
                                 re-expressed). Reuse verbatim; one contract each.
  QF-4 SCOPE DECLARATION     -- printed, NOT computed; never gates PASS/FAIL. NOT claimed: a
                                 solution of the measurement problem; a Gleason derivation
                                 independent of A3; spacelike-separated measurement EVENTS (QF-2
                                 is a fixed-time vacuum-correlation read -- no loophole language);
                                 decoherence beyond the tick/record sector; any interpretation
                                 commitment.

REUSE MAP (zero physics added beyond QF-2's declared new computation; every physics-bearing
symbol below is copied/re-expressed from the named prior-art file, not re-derived):
  proofs/foundations/M0_2R_T1_run_kms_tick_2026-07-07.py
      lines ~48-61 (S0): k=srs.DEG, q=k-1, u_c=1/q, alpha_1=(q/k)^(10-2), B0=srs.hashimoto(Gamma),
      lam_P -- copied verbatim (QF-1 S0 block).
      lines ~67-80: shell_norms/marginal -- copied verbatim, generalized with an `exp` parameter
      (default 2.0, exactly reproducing the original at exp=2.0) for the QF-1c deformation probe.
      lines ~86-98: the geometric-ratio / Born-2 check -- QF-1a.
      lines ~106-124: the affine-fit / beta_eff derivation -- QF-1b slope-ratio and QF-1c's fixed
      prediction to falsify against.
      lines ~157-176: the localized-seed thermalization decay-rate fit (rate ~ (sqrt(q)/q)^2=u_c)
      -- copied verbatim as QF-3c.
  derivation_topdown/state/the_net.py
      anchor_tick_2pi(), Patch(M).anticommutator_below_cone(T) -- QF-0 anchors (reused,
      unmodified); Patch(M).vertex_adjacency() -- QF-2's real-space single-particle Hamiltonian;
      twisted_locality_holds() -- cited as the general disjoint-region-commutation argument
      backing QF-2's big-patch commutation claim.
  proofs/foundations/ML1ppp_computed_2pi_2026-07-08.py
      lines ~66-76: Patch.vertex_adjacency() -> eigh -> Dirac-sea fill (E < -1-1e-9) ->
      C = V_filled @ V_filled^dagger -- QF-2's vacuum construction, reused verbatim (same filling
      threshold, same convention <a_i^dagger a_j> = C_ij = (V_filled V_filled^dagger)_ij, checked
      against a dense many-body Slater determinant in the mini two-route check below).
  proofs/foundations/phase3_1_davies_gkls_compression_2026-06-11.py
      lines ~77-95 (build_ops): the coherent U(P) / edge-projector P_e construction on the 12-dart
      P-fiber (proofs.common.find_bonds); lines ~105-116 (choi_of_super/is_cptp): the Choi-PSD
      CPTP test; lines ~162-189 (D2b): the Davies-scaled generator's ccp (conditional complete
      positivity) test and its integration back to a CPTP semigroup -- QF-3a, re-expressed as a
      contract, same construction and same tolerances.
  proofs/foundations/phase3_2_s1_step_isometry_2026-06-11.py
      lines ~33-60 (the V isometry): the Stinespring dilation V|psi> = sqrt(q)(U|psi>)|0>_rec +
      sqrt(1-q) sum_e (P_e|psi>)|e>_rec; lines ~63-89 (S1a/S1b/S1c): isometry / visible-marginal /
      record-marginal identities -- QF-3b, re-expressed as a contract (NOT imported as a module:
      that file executes its own gates and a bare sys.exit at import time; the identities are
      replicated here instead, same construction, same tolerances, lines cited).

POISONS (binding, per pre-reg): the A3 conditionality is printed, never dropped; the QF-1c
deformation probe must actually fail (a tolerant probe = contract FAIL); QF-2's observable family,
regions, grids, and verdict thresholds are FROZEN BEFORE the sweep runs (no family enlargement
after seeing S_max -- a small family's honest negative beats a shopped positive); the crisis
branch (TSIRELSON-BREACH) cannot be softened; engine/proofs untouched; ONE new file; no git
commits.
"""
import math
import os
import sys
import time

import numpy as np

_T0 = time.time()

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402

from proofs.common import find_bonds  # noqa: E402  (QF-3a/b reuse, side-effect-free import)

np.set_printoptions(precision=6, suppress=True)
ok_all = True          # gates QF-0,1,2(machinery+verdict-reached),3 -- NOT softened for TSIRELSON
tsirelson_breach = False


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


print("=" * 88)
print(" G7 ADAPTER -- quantum foundations: Born rule, CHSH/Tsirelson, decoherence/pointer")
print(" (Chiribella-D'Ariano-Perinotti 2011; Bell 1964, Clauser-Horne-Shimony-Holt 1969,")
print("  Cirel'son 1980; Davies-GKLS, Zurek pointer-basis/einselection)")
print("=" * 88)

# ===========================================================================
banner("QF-0  ANCHORS  (regression, reused unmodified) + THE A3 STATUS LINE")
# ===========================================================================
a_tick = net.anchor_tick_2pi()
check("QF-0a net.anchor_tick_2pi(): N-hat integer spectrum, minimal period exactly 2pi",
      a_tick, detail=f"anchor_tick_2pi() = {a_tick}")

_patch0 = net.Patch(M=4)
_cone_worst = _patch0.anticommutator_below_cone(T=3)
check("QF-0b the exact-light-cone anchor: {alpha_a(t),a_c^dagger}=(B^t)_ca IDENTICALLY 0.0 "
      "strictly below the geometric horizon (Patch(M=4), T=3)",
      _cone_worst == 0.0, detail=f"max|below cone| = {_cone_worst:.1e}")

print("""
  A3 STATUS LINE (printed verbatim, per pre-reg -- never dropped, never hidden):
  "A3 (MDL-canonicalization = purification; adopted axiom 2026-04-18) supplies the
   complex-Hilbert structure (G.1/G.5 via CDP-2011); the Born contract below is a
   THEOREM-CHECK CONDITIONAL on A3 -- the exponent-2 MECHANICS is checked unconditionally."
""")

# ===========================================================================
banner("QF-1(a)  THE BORN RULE: per-tick ratio == (alpha_1/u_c)^2  (M0-2R recipe, verbatim)")
# ===========================================================================
# S0 (verbatim from M0_2R_T1_run_kms_tick_2026-07-07.py lines ~48-61 / thermal_time.py KMS-1):
k = srs.DEG                                   # coordination number (READ) = 3
q = k - 1                                     # NB branching = continuations per dart
u_c = 1.0 / q                                 # path-gas critical fugacity (Perron combinatorics)
alpha1 = (q / k) ** (10 - 2)                  # the run's operating fugacity ((2/3)^8), the_run.py
B0 = srs.hashimoto((0, 0, 0)).real            # 12x12 NB step at Gamma (real, integer)
ND = B0.shape[0]
lam_P = max(abs(np.linalg.eigvals(B0)))       # Perron eigenvalue of B(Gamma)
print(f"    k={k} q=k-1={q}  u_c=1/(k-1)={u_c}  alpha_1=(2/3)^8={alpha1:.10f}  "
      f"lam_P(B(Gamma))={lam_P:.10f}")
check("QF-1 pre-req: Perron eigenvalue of B(Gamma) = k-1 (=> u_c = 1/(k-1) is the correct read)",
      abs(lam_P - q) < 1e-9, detail=f"lam_P={lam_P:.10f}, q={q}")

PERRON = np.ones(ND) / math.sqrt(ND)          # the stationary (Perron/equilibrium) run seed


def shell_norms(u, Bmat, lamP, N, seed, exp=2.0):
    """||shell n||^exp = (u*lamP)^(exp*n) * ||Bhat^n seed||^exp, Perron-normalised (verbatim M0-2R
    recipe at exp=2.0; generalized with an `exp` parameter for the QF-1c deformation probe --
    the physical amplitude at tick n is amp_n = (u*lamP)^n * Bhat^n seed, and this returns
    ||amp_n||^exp, the generalized Born measure)."""
    Bh = Bmat / lamP
    v = seed.astype(complex).copy()
    out = []
    for n in range(N + 1):
        norm2 = float(np.vdot(v, v).real)
        out.append(((u * lamP) ** (exp * n)) * (norm2 ** (exp / 2.0)))
        v = Bh @ v
    return np.array(out)


def marginal(u, Bmat, lamP, N, seed, exp=2.0):
    w = shell_norms(u, Bmat, lamP, N, seed, exp)
    return w / w.sum()


N_max = 40
p = marginal(alpha1, B0, lam_P, N_max, PERRON)
ratios = p[1:] / p[:-1]
r_mean = float(np.mean(ratios))
r_rel_std = float(np.std(ratios) / r_mean)
born2 = (alpha1 / u_c) ** 2
print(f"    N_max={N_max}; per-tick ratio p_(n+1)/p_n: mean={r_mean:.12f}, "
      f"rel_std={r_rel_std:.3e}")
print(f"      (alpha_1/u_c)^2 = {born2:.12f}   |ratio - Born2| = {abs(r_mean - born2):.3e}")
check("QF-1a the run marginal is EXACTLY GEOMETRIC (rel std < 1e-12)", r_rel_std < 1e-12)
check("QF-1a THEOREM-CHECK (conditional on A3): per-tick ratio == (alpha_1/u_c)^2 at < 1e-12",
      abs(r_mean - born2) < 1e-12, detail=f"|dev| = {abs(r_mean - born2):.3e}")

# ===========================================================================
banner("QF-1(b)  THE MECHANISM: Perron/Ramanujan orthogonality + the modular slope == 2")
# ===========================================================================
w_eig, V_eig = np.linalg.eig(B0)
order = np.argsort(-np.abs(w_eig))
w_eig = w_eig[order]; V_eig = V_eig[:, order]
perron_vec = V_eig[:, 0] / np.linalg.norm(V_eig[:, 0])
raman_idx = [i for i in range(1, ND) if abs(abs(w_eig[i]) - math.sqrt(q)) < 1e-6]
print(f"    B(Gamma) spectrum (sorted by |lambda|): Perron={w_eig[0]:.6f}; "
      f"{len(raman_idx)} Ramanujan modes at |lambda|=sqrt(q)={math.sqrt(q):.6f}: "
      f"{np.round(w_eig[raman_idx], 4)}")
overlaps = [abs(np.vdot(perron_vec, V_eig[:, i] / np.linalg.norm(V_eig[:, i]))) for i in raman_idx]
worst_overlap = max(overlaps) if overlaps else float("nan")
print(f"    Perron-Ramanujan overlaps |<Perron|Ramanujan_i>|: max = {worst_overlap:.3e}")
check("QF-1b MECHANISM: the Ramanujan modes are ORTHOGONAL to the Perron vector "
      "(the cross-term vanishing that makes the square mechanical, overlaps < 1e-12)",
      len(raman_idx) == 6 and worst_overlap < 1e-12,
      detail=f"{len(raman_idx)} Ramanujan modes, worst overlap = {worst_overlap:.3e}")

nn = np.arange(N_max + 1, dtype=float)
Mgen = -np.log(p)
Afit = np.vstack([nn, np.ones(N_max + 1)]).T
slope, intercept = np.linalg.lstsq(Afit, Mgen, rcond=None)[0]
beta_eff_pred = 2 * math.log(u_c / alpha1)
log_ratio_denom = math.log(u_c / alpha1)
slope_ratio = slope / log_ratio_denom
print(f"    affine fit slope = {slope:.10f};  log(u_c/alpha_1) = {log_ratio_denom:.10f}")
print(f"    MEASURED exponent [-dlog p_n/dn]/log(u_c/alpha_1) = {slope_ratio:.12f}")
check("QF-1b the modular slope ratio MEASURES the exponent == 2 exactly (< 1e-9)",
      abs(slope_ratio - 2.0) < 1e-9, detail=f"|ratio - 2| = {abs(slope_ratio - 2.0):.3e}")

# ===========================================================================
banner("QF-1(c)  FALSIFICATION PROBE: deform ||amp||^(2+delta), delta=0.1 -- MUST FAIL")
# ===========================================================================
delta = 0.1
p_def = marginal(alpha1, B0, lam_P, N_max, PERRON, exp=2.0 + delta)
Mgen_def = -np.log(p_def)


def detrend_spread(Mvec, beta, nvec):
    """max-min of (-log p_n - beta*n): ~0 iff -log p_n is EXACTLY affine at the FIXED slope beta
    (the frozen, independently-derived beta_eff_pred -- NOT a re-fit slope)."""
    r = Mvec - beta * nvec
    return float(np.max(r) - np.min(r))


spread_pristine = detrend_spread(Mgen, beta_eff_pred, nn)
spread_deformed = detrend_spread(Mgen_def, beta_eff_pred, nn)
blowup_affine = spread_deformed / max(spread_pristine, 1e-300)
print(f"    affine-fit-vs-FIXED-beta_eff spread: pristine = {spread_pristine:.3e}, "
      f"delta=0.1 deformed = {spread_deformed:.3e}  (blow-up factor = {blowup_affine:.3e})")

gibbs_unnorm = np.exp(-beta_eff_pred * nn)
gibbs = gibbs_unnorm / gibbs_unnorm.sum()
mat_dist_pristine = float(np.max(np.abs(p - gibbs)))
mat_dist_deformed = float(np.max(np.abs(p_def - gibbs)))
blowup_gibbs = mat_dist_deformed / max(mat_dist_pristine, 1e-300)
print(f"    Gibbs-form (KMS-3 style) max-abs distance to e^(-beta_eff_pred*N)/Z: "
      f"pristine = {mat_dist_pristine:.3e}, deformed = {mat_dist_deformed:.3e}  "
      f"(blow-up factor = {blowup_gibbs:.3e})")

slope_def, _ = np.linalg.lstsq(Afit, Mgen_def, rcond=None)[0]
print(f"    (diagnostic) the deformed data's OWN best-fit slope = {slope_def:.8f} = "
      f"(2+delta)*log(u_c/alpha_1)/1 -- i.e. still perfectly geometric at the WRONG rate; only "
      f"anchoring to the fixed, independently-derived beta_eff_pred exposes the deformation.")

falsify_ok = (blowup_affine > 1e6 and spread_deformed > 1e-3
              and blowup_gibbs > 1e6 and mat_dist_deformed > 1e-3)
check("QF-1c FALSIFICATION PROBE genuinely FAILS by a quantified margin (a passing/tolerant "
      "deformation would itself be a contract FAIL): both the affine-fit and Gibbs-form checks "
      "blow up by >1e6x under the delta=0.1 deformation",
      falsify_ok, detail=f"affine blow-up={blowup_affine:.2e}, Gibbs blow-up={blowup_gibbs:.2e}")

# ===========================================================================
banner("QF-2  CHSH / TSIRELSON -- NEW COMPUTATION (dual-outcome + crisis branch)")
# ===========================================================================
NGRID_QF2 = 64                                # frozen: >=64 angles per observable (pre-reg (b))


def majorana_from_C(C):
    """Majorana covariance Gamma_ab = -i(<gamma_a gamma_b> - delta_ab), real antisymmetric, from
    the complex covariance C_ij = <a_i^dagger a_j> (number-conserving Gaussian state, <a_i a_j>=0
    -- true for any Dirac-sea/Slater-determinant vacuum). Convention: gamma_{2i}=a_i+a_i^dagger,
    gamma_{2i+1}=-i(a_i-a_i^dagger). Validated against a dense many-body cross-check below
    (TECHNICAL CARE point (i))."""
    n = C.shape[0]
    X = C.real; Y = C.imag
    Gam = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            Gam[2 * i, 2 * j] = 2 * Y[i, j]
            Gam[2 * i + 1, 2 * j + 1] = 2 * Y[i, j]
            d = 1.0 if i == j else 0.0
            Gam[2 * i, 2 * j + 1] = d - 2 * X[i, j]
            Gam[2 * i + 1, 2 * j] = 2 * X[i, j] - d
    return Gam


def _perm_sign(order):
    p = list(order); parity = 1
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            parity *= -1
    return parity


def wick4(Gam, idx):
    """<gamma_i1 gamma_i2 gamma_i3 gamma_i4> for 4 DISTINCT Majorana indices (any order), via the
    Pfaffian formula for a Gaussian state: for a<b<c<d, <gaga gb gc gd> = -Gam[a,b]*Gam[c,d] +
    Gam[a,c]*Gam[b,d] - Gam[a,d]*Gam[b,c] (== i^2 * Pf of the 4x4 sub-block); general index order
    handled via the sorting permutation's sign (fermionic reordering)."""
    ordr = sorted(range(4), key=lambda t: idx[t])
    sgn = _perm_sign(ordr)
    a, b, c, d = [idx[o] for o in ordr]
    val = -Gam[a, b] * Gam[c, d] + Gam[a, c] * Gam[b, d] - Gam[a, d] * Gam[b, c]
    return sgn * val


def jw_ops(n):
    """Explicit Jordan-Wigner Fock representation of n abstract fermion modes (2^n dim). Used
    ONLY for the small-instance dense cross-check, never for the big-patch sweep."""
    I2 = np.eye(2); Z2 = np.diag([1.0, -1.0]); a1 = np.array([[0.0, 1.0], [0.0, 0.0]])

    def kron_list(ms):
        out = np.array([[1.0]])
        for m in ms:
            out = np.kron(out, m)
        return out

    a = [kron_list([Z2] * p + [a1] + [I2] * (n - 1 - p)) for p in range(n)]
    return a, [op.conj().T for op in a]


def majorana_ops_dense(n, a, adag):
    gam = []
    for i in range(n):
        gam.append(a[i] + adag[i])
        gam.append(-1j * (a[i] - adag[i]))
    return gam


def dense_rho_from_C(n, C):
    """The dense many-body Gaussian density matrix with single-particle covariance C, built by
    diagonalizing C = V.diag(occ).V^dagger and rotating to the modes b_k = sum_i V[i,k]*a_i (NO
    conjugate -- fixed so <a_i^dagger a_j> = C_ij matches the C=V.occ.V^dagger convention used
    throughout; verified below against the Wick/Pfaffian route)."""
    w, V = np.linalg.eigh(C)
    dim = 2 ** n
    a, adag = jw_ops(n)
    b = [sum(V[i, kk] * a[i] for i in range(n)) for kk in range(n)]
    bdag = [op.conj().T for op in b]
    rho = np.eye(dim, dtype=complex)
    for kk in range(n):
        nk = bdag[kk] @ b[kk]
        wk = float(np.clip(w[kk].real, 0.0, 1.0))
        rho = rho @ (wk * nk + (1 - wk) * (np.eye(dim) - nk))
    return rho, a, adag


def T_tensor(Gam, gA, gB):
    """The 2x2 correlation tensor T_ab = <(i*gA[0]*gA[1+a])(i*gB[0]*gB[1+b])> for the declared
    2-plane basis {i*g0*g1, i*g0*g2} per region (pre-reg (b): A(theta) = cos(theta)(i g1 g2) +
    sin(theta)(i g1 g3) -- gA=[g0,g1,g2,g3] region-A Majorana labels, gB likewise for region B."""
    g0, g1, g2, _ = gA
    h0, h1, h2, _ = gB
    basisA = [(g0, g1), (g0, g2)]
    basisB = [(h0, h1), (h0, h2)]
    T = np.zeros((2, 2))
    for ia, (p, qidx) in enumerate(basisA):
        for ib, (r, s) in enumerate(basisB):
            T[ia, ib] = -wick4(Gam, (p, qidx, r, s))
    return T


def S_horodecki(T):
    """Closed-form CHSH maximum for a 2x2 correlation tensor (Horodecki 1995 criterion restricted
    to a 2-dim subspace per party): S_max = 2*sqrt(sigma1^2+sigma2^2), sigma = singular values of
    T (both singular values of a 2x2 matrix enter -- 'the two largest' IS all of them here)."""
    sv = np.linalg.svd(T, compute_uv=False)
    return 2 * math.sqrt(sv[0] ** 2 + sv[1] ** 2), sv


def chsh_grid_refine(T, ngrid=NGRID_QF2):
    """Declared-grid + local-refinement cross-check of S_horodecki, WITHOUT a brute 4-angle grid
    (pre-reg technical care: '32^4 is too big'). Method: for Gaussian states <A(th1)B(ph)> is
    bilinear in (cos th, sin th) x (cos ph, sin ph) via T; S(th1,th2) = ||T^T(u1+u2)|| +
    ||T^T(u1-u2)|| is the EXACT max over the two B-angles at fixed A-angles (u1,u2 unit vectors) --
    so a declared grid of >=64 A-angles (both th1,th2) plus a local refinement finds the max over
    the WHOLE declared family; B's angles are recovered analytically (arctan of T^T(u1+-u2)), not
    grid-searched, but ALSO cross-checked against a literal 64-point B-grid at the best point."""
    thetas = np.linspace(0, 2 * math.pi, ngrid, endpoint=False)

    def Sfun(t1, t2):
        u1 = np.array([math.cos(t1), math.sin(t1)])
        u2 = np.array([math.cos(t2), math.sin(t2)])
        return np.linalg.norm(T.T @ (u1 + u2)) + np.linalg.norm(T.T @ (u1 - u2))

    best = -1.0; best_th = (0.0, 0.0)
    for t1 in thetas:
        for t2 in thetas:
            v = Sfun(t1, t2)
            if v > best:
                best = v; best_th = (t1, t2)
    t1, t2 = best_th
    step = 2 * math.pi / ngrid
    for _ in range(40):
        improved = False
        for dt1 in (-step / 2, 0.0, step / 2):
            for dt2 in (-step / 2, 0.0, step / 2):
                v = Sfun(t1 + dt1, t2 + dt2)
                if v > best:
                    best = v; t1, t2 = t1 + dt1, t2 + dt2; improved = True
        step /= 2
        if not improved and step < 1e-10:
            break
    # literal 64-point cross-check on phi at the refined (t1,t2)
    def Efun(th, ph):
        a = np.array([math.cos(th), math.sin(th)]); bb = np.array([math.cos(ph), math.sin(ph)])
        return a @ T @ bb

    grid_best = -1.0
    for p1 in thetas:
        for p2 in thetas:
            v = Efun(t1, p1) + Efun(t1, p2) + Efun(t2, p1) - Efun(t2, p2)
            grid_best = max(grid_best, v)
    return best, (t1, t2), grid_best


def build_gaussian_vacuum(Msize):
    patch = net.Patch(M=Msize)
    H, verts = patch.vertex_adjacency()
    vpos = {v: nidx for nidx, v in enumerate(verts)}
    E, V = np.linalg.eigh(H)
    cols = V[:, E < -1.0 - 1e-9]                # cone Dirac sea (ML1ppp convention, verbatim)
    C_full = cols @ cols.conj().T
    return patch, verts, vpos, C_full


def two_route_check(C_sub, thetaA, thetaB):
    """MANDATORY pre-sweep check: on the small (4-mode, 16-dim) instance, the Wick/Pfaffian
    correlator MUST equal the dense many-body Slater-determinant expectation. ALSO verifies A^2=I
    and [A,B]=0 on the ACTUAL dense operators (not assumed)."""
    Gam = majorana_from_C(C_sub)
    antisym = float(np.max(np.abs(Gam + Gam.T)))
    T = T_tensor(Gam, [0, 1, 2, 3], [4, 5, 6, 7])
    rho, a_ops, adag_ops = dense_rho_from_C(4, C_sub)
    gam_dense = majorana_ops_dense(4, a_ops, adag_ops)

    def A_dense(theta, g0, g1, g2):
        P1 = 1j * gam_dense[g0] @ gam_dense[g1]
        P2 = 1j * gam_dense[g0] @ gam_dense[g2]
        return math.cos(theta) * P1 + math.sin(theta) * P2

    Aop = A_dense(thetaA, 0, 1, 2)
    Bop = A_dense(thetaB, 4, 5, 6)
    A2resid = float(np.max(np.abs(Aop @ Aop - np.eye(16))))
    B2resid = float(np.max(np.abs(Bop @ Bop - np.eye(16))))
    comm_resid = float(np.max(np.abs(Aop @ Bop - Bop @ Aop)))
    dense_corr = float(np.real(np.trace(rho @ Aop @ Bop)))
    wick_corr = (math.cos(thetaA) * math.cos(thetaB) * T[0, 0]
                 + math.cos(thetaA) * math.sin(thetaB) * T[0, 1]
                 + math.sin(thetaA) * math.cos(thetaB) * T[1, 0]
                 + math.sin(thetaA) * math.sin(thetaB) * T[1, 1])
    return {
        "antisym": antisym, "T": T, "A2resid": A2resid, "B2resid": B2resid,
        "comm_resid": comm_resid, "dense_corr": dense_corr, "wick_corr": wick_corr,
        "diff": abs(dense_corr - wick_corr),
    }


def run_qf2_instance(label, C_full, patch, verts, vpos, candA, candB):
    banner(f"QF-2 instance: {label}")
    pA = [vpos[v] for v in candA]
    pB = [vpos[v] for v in candB]
    sep = min(patch.vdist(patch.vidx[verts[i]], patch.vidx[verts[j]]) for i in pA for j in pB)
    print(f"    region O_A vertices: {candA}")
    print(f"    region O_B vertices: {candB}")
    print(f"    declared separation (vertex-graph hops) = {sep}")
    idxU = pA + pB
    C_sub = C_full[np.ix_(idxU, idxU)]
    herm = float(np.max(np.abs(C_sub - C_sub.conj().T)))
    check(f"[{label}] region covariance C_sub Hermitian", herm < 1e-10, detail=f"{herm:.2e}")

    tr = two_route_check(C_sub, 0.37, 0.91)
    check(f"[{label}] Majorana covariance Gamma antisymmetric (<1e-10)",
          tr["antisym"] < 1e-10, detail=f"{tr['antisym']:.2e}")
    check(f"[{label}] MINI TWO-ROUTE CHECK: Wick/Pfaffian == dense many-body ⟨A B⟩ (<1e-10, "
          "BEFORE the sweep)", tr["diff"] < 1e-10,
          detail=f"dense={tr['dense_corr']:.10f} wick={tr['wick_corr']:.10f} diff={tr['diff']:.2e}")
    check(f"[{label}] A(theta)^2 = I on the actual dense operator (<1e-12)",
          tr["A2resid"] < 1e-12, detail=f"{tr['A2resid']:.2e}")
    check(f"[{label}] B(phi)^2 = I on the actual dense operator (<1e-12)",
          tr["B2resid"] < 1e-12, detail=f"{tr['B2resid']:.2e}")
    check(f"[{label}] [A(theta),B(phi)] = 0 on the ACTUAL dense operators (<1e-12, re-verified "
          "not assumed)", tr["comm_resid"] < 1e-12, detail=f"{tr['comm_resid']:.2e}")
    print("    BIG-PATCH GENERALIZATION ARGUMENT (stated, not re-proved on a huge matrix): "
          "region A's Majorana operators have support ONLY on modes {pA}; region B's ONLY on "
          "{pB}; pA and pB are disjoint by construction. Even (bilinear, parity-preserving) "
          "operators built from disjoint fermion-mode supports commute identically -- a "
          "basis-independent CAR-algebra fact, the SAME fact the_net.py's "
          "twisted_locality_holds() certifies generally for this object's even sub-algebras. "
          "The mini-check above verifies this concretely on the actual operators; it "
          "generalizes to any disjoint region pair on the big patch.")

    T = tr["T"]
    S_an, sv = S_horodecki(T)
    S_grid, best_th, S_grid_phi_check = chsh_grid_refine(T)
    print(f"    T tensor:\n{T}")
    print(f"    singular values of T: {sv}")
    print(f"    METHOD: S_max via the closed-form Horodecki 2x2 reduction "
          f"(S_max = 2*sqrt(sigma1^2+sigma2^2)) = {S_an:.8f}; cross-checked by a declared "
          f"{NGRID_QF2}-angle grid + local refinement over (theta1,theta2) with the B-angles "
          f"recovered analytically at each point (S = 2*sqrt(sigma1^2+sigma2^2) is convex in the "
          f"B-angles at fixed A-angles) = {S_grid:.8f}; a literal {NGRID_QF2}-point grid over "
          f"(phi1,phi2) at the refined optimum gives {S_grid_phi_check:.8f} (should agree).")
    agree = abs(S_an - S_grid) < 1e-6 and abs(S_an - S_grid_phi_check) < 1e-3
    check(f"[{label}] closed-form vs grid+refine vs literal-phi-grid AGREE", agree,
          detail=f"analytic={S_an:.8f} grid={S_grid:.8f} phi-grid={S_grid_phi_check:.8f}")

    S_max = max(S_an, S_grid, S_grid_phi_check)
    two_sqrt2 = 2 * math.sqrt(2)
    if S_max > two_sqrt2 + 1e-9:
        verdict = "TSIRELSON-BREACH"
    elif S_max >= 2.05:
        verdict = "VIOLATION-FOUND"
    else:
        verdict = "NO-VIOLATION-IN-FAMILY"
    print(f"    S_max = {S_max:.8f}   (classical bound 2, this family's frozen threshold 2.05, "
          f"Tsirelson 2sqrt2 = {two_sqrt2:.8f})")
    print(f"    >>> VERDICT [{label}]: {verdict} <<<")
    return {"label": label, "sep": sep, "T": T, "S_max": S_max, "verdict": verdict,
            "two_route": tr}


Msize = 5
patch5, verts5, vpos5, C_full5 = build_gaussian_vacuum(Msize)
print(f"    Patch(M={Msize}): {len(verts5)} vertices; Dirac-sea filled modes = "
      f"{int(round(np.trace(C_full5).real))}; C_full idempotent check = "
      f"{float(np.max(np.abs(C_full5 @ C_full5 - C_full5))):.2e}")

# declared FAR separation: opposite corners of the patch (branches 0,1 each)
candA_far = [(0, (0, 0, 0)), (1, (0, 0, 0))]
candB_far = [(0, (Msize - 1, Msize - 1, Msize - 1)), (1, (Msize - 1, Msize - 1, Msize - 1))]
res_far = run_qf2_instance("FAR (declared regions)", C_full5, patch5, verts5, vpos5,
                            candA_far, candB_far)

# declared robustness point: ADJACENT regions (same cell, disjoint branches, separation=1)
c0 = Msize // 2
candA_near = [(0, (c0, c0, c0)), (1, (c0, c0, c0))]
candB_near = [(2, (c0, c0, c0)), (3, (c0, c0, c0))]
res_near = run_qf2_instance("NEAR/ADJACENT (declared robustness point)", C_full5, patch5, verts5,
                             vpos5, candA_near, candB_near)

if res_far["verdict"] == "TSIRELSON-BREACH" or res_near["verdict"] == "TSIRELSON-BREACH":
    tsirelson_breach = True

qf2_verdict_definite = res_far["verdict"] in ("VIOLATION-FOUND", "NO-VIOLATION-IN-FAMILY",
                                               "TSIRELSON-BREACH")
check("QF-2 a DEFINITE verdict was reached for the declared (FAR) regions (one of the three "
      "frozen branches)", qf2_verdict_definite, detail=f"verdict = {res_far['verdict']}")

banner("QF-2  SUMMARY TABLE")
print(f"  {'label':38s} {'sep':>4s} {'S_max':>12s}  verdict")
for r in (res_far, res_near):
    print(f"  {r['label']:38s} {r['sep']:>4d} {r['S_max']:>12.6f}  {r['verdict']}")

# ===========================================================================
banner("QF-3(a)  THE DERIVED-GKLS GATE  (phase3_1_davies_gkls_compression_2026-06-11.py, reused)")
# ===========================================================================
NF = 12
Q_NB = 2.0 / 3.0


def qf3a_build_ops():
    """Verbatim from phase3_1 lines ~77-95 (build_ops): the coherent U(P) and the flip S on the
    12-dart P-fiber, at the pre-declared Bloch point P=(1/4,1/4,1/4)."""
    bonds = find_bonds()
    edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
    rev = {}
    for a, (i, j, c) in enumerate(edges):
        t = (j, i, tuple(-x for x in c))
        for b, e2 in enumerate(edges):
            if e2 == t:
                rev[a] = b
    P = np.array([0.25, 0.25, 0.25])
    Cop = np.zeros((NF, NF), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        for b, (i2, j2, c2) in enumerate(edges):
            if i2 == i:
                Cop[b, a] = 2.0 / 3.0 - (1.0 if b == a else 0.0)
    Sop = np.zeros((NF, NF), dtype=complex)
    for a, (i, j, c) in enumerate(edges):
        Sop[rev[a], a] = np.exp(2j * np.pi * np.dot(P, np.asarray(c, float)))
    return Sop @ Cop, Sop


def qf3a_kraus_to_super(Ks):
    L = np.zeros((NF * NF, NF * NF), dtype=complex)
    for K in Ks:
        L += np.kron(K, K.conj())
    return L


def qf3a_choi_of_super(Sup):
    T = Sup.reshape(NF, NF, NF, NF)
    return T.transpose(0, 2, 1, 3).reshape(NF * NF, NF * NF)


def qf3a_is_cptp(Ks):
    tp = np.linalg.norm(sum(K.conj().T @ K for K in Ks) - np.eye(NF)) < 1e-10
    Cchoi = qf3a_choi_of_super(qf3a_kraus_to_super(Ks))
    ev = np.linalg.eigvalsh((Cchoi + Cchoi.conj().T) / 2)
    return tp, float(ev.min())


Uop, Sop = qf3a_build_ops()
Pe = [np.zeros((NF, NF), dtype=complex) for _ in range(NF)]
for e in range(NF):
    Pe[e][e, e] = 1.0
sq, sp_ = math.sqrt(Q_NB), math.sqrt(1 - Q_NB)
M3_dephasing = [sq * Uop] + [sp_ * P for P in Pe]        # phase3_1's adopted-class channel
tp_ok, cmin = qf3a_is_cptp(M3_dephasing)
check("QF-3a D1 (phase3_1 lines ~112-116/141-142): the adopted M3 dephasing compression channel "
      "is exactly CPTP (Choi PSD)", tp_ok and cmin > -1e-10, detail=f"min Choi eigval={cmin:.2e}")

omega_v = np.eye(NF).reshape(-1) / math.sqrt(NF)
Pperp = np.eye(NF * NF) - np.outer(omega_v, omega_v.conj())


def qf3a_ccp_min(Lsup):
    CL = qf3a_choi_of_super(Lsup)
    CL = (CL + CL.conj().T) / 2
    return float(np.linalg.eigvalsh(Pperp @ CL @ Pperp).min())


evU, VU = np.linalg.eig(Uop)
Hham = VU @ np.diag(-np.angle(evU)) @ np.linalg.inv(VU)
Hham = (Hham + Hham.conj().T) / 2
Idf = np.eye(NF)
L_ham = -1j * (np.kron(Hham, Idf) - np.kron(Idf, Hham.T))
gamma_rate = 1.0 - Q_NB
Diss = qf3a_kraus_to_super([P for P in Pe]) - np.eye(NF * NF)
L_gen = L_ham + gamma_rate * Diss
ccp_val = qf3a_ccp_min(L_gen)
from scipy.linalg import expm as _expm
cpt_ok = True
for sgrid in (0.1, 1.0, 5.0):
    Cs = qf3a_choi_of_super(_expm(sgrid * L_gen))
    cpt_ok &= float(np.linalg.eigvalsh((Cs + Cs.conj().T) / 2).min()) > -1e-8
check("QF-3a D2b (phase3_1 lines ~162-189): the Davies-scaled generator is GKLS (ccp test) and "
      "integrates to a CPTP semigroup -- LINDBLAD FORM DERIVED in the continuum scaling",
      ccp_val > -1e-8 and cpt_ok, detail=f"ccp={ccp_val:.2e}, integrate-back CPTP={cpt_ok}")

# ===========================================================================
banner("QF-3(b)  POINTER BASIS = RECORD SUPERSELECTION  (phase3_2_s1_step_isometry, replicated)")
# ===========================================================================
Rrec = NF + 1
Viso = np.zeros((NF * Rrec, NF), dtype=complex)
for col in range(NF):
    psi = np.zeros(NF, dtype=complex); psi[col] = 1
    out = np.zeros((NF, Rrec), dtype=complex)
    out[:, 0] = math.sqrt(Q_NB) * (Uop @ psi)
    for e in range(NF):
        out[:, 1 + e] = math.sqrt(1 - Q_NB) * (Pe[e] @ psi)
    Viso[:, col] = out.reshape(-1)

s1a_resid = float(np.linalg.norm(Viso.conj().T @ Viso - np.eye(NF)))
check("QF-3b S1a (phase3_2 lines ~52-64): V is an exact isometry (V^dagger V = I)",
      s1a_resid < 1e-12, detail=f"||V+V-I||={s1a_resid:.2e}")


def qf3b_vis_marginal(rho):
    big = Viso @ rho @ Viso.conj().T
    Tb = big.reshape(NF, Rrec, NF, Rrec)
    return np.einsum('irjr->ij', Tb)


def qf3b_rec_marginal(rho):
    big = Viso @ rho @ Viso.conj().T
    Tb = big.reshape(NF, Rrec, NF, Rrec)
    return np.einsum('iris->rs', Tb)


def qf3b_Phi3(rho):
    return Q_NB * Uop @ rho @ Uop.conj().T + (1 - Q_NB) * sum(P @ rho @ P for P in Pe)


rng = np.random.default_rng(7)
worst_s1b = 0.0
for _ in range(6):
    Arand = rng.standard_normal((NF, NF)) + 1j * rng.standard_normal((NF, NF))
    rho_t = Arand @ Arand.conj().T
    rho_t /= np.trace(rho_t)
    worst_s1b = max(worst_s1b, float(np.linalg.norm(qf3b_vis_marginal(rho_t) - qf3b_Phi3(rho_t))))
check("QF-3b S1b (phase3_2 lines ~66-83): Tr_rec(V rho V^dagger) = Phi_M3(rho) -- the pointer/"
      "record-traced dynamics IS the visible-sector compression channel",
      worst_s1b < 1e-12, detail=f"worst={worst_s1b:.2e}")

rr = qf3b_rec_marginal(np.eye(NF) / NF)
s1c_ok = (abs(np.trace(rr) - 1) < 1e-12 and abs(rr[0, 0].real - Q_NB) < 1e-12
          and float(np.linalg.eigvalsh((rr + rr.conj().T) / 2).min()) > -1e-12)
check("QF-3b S1c (phase3_2 lines ~85-89): the record marginal (the pointer/superselection basis) "
      "is a valid state; coherent-record weight = q = 2/3 on the maximally mixed input",
      s1c_ok, detail=f"rec[0,0]={rr[0, 0].real:.6f}")

# ===========================================================================
banner("QF-3(c)  THERMALIZATION TO KMS AT RATE u_c  (M0-2R T1 decay fit, verbatim)")
# ===========================================================================
seed_loc = np.zeros(ND); seed_loc[0] = 1.0
p_loc = marginal(alpha1, B0, lam_P, N_max, seed_loc)
dev = np.abs((p_loc[1:] / p_loc[:-1]) - born2)
wn = np.arange(6, 26)
good = dev[wn] > 1e-14
rate = math.exp(np.polyfit(wn[good], np.log(dev[wn][good]), 1)[0])
rate_pred = (math.sqrt(q) / q) ** 2
print(f"    localized-seed transient thermalizes to the KMS marginal at rate ~{rate:.4f}/tick")
print(f"    prediction (Ramanujan gap)^2 = (sqrt(k-1)/(k-1))^2 = 1/(k-1) = u_c = {rate_pred:.4f}")
check("QF-3c thermalization rate == the Ramanujan-gap-squared prediction = u_c (< 0.02)",
      abs(rate - rate_pred) < 0.02, detail=f"|rate-u_c|={abs(rate - rate_pred):.4f}")

# ===========================================================================
banner("QF-4  SCOPE DECLARATION  (printed, NOT computed; never gates PASS/FAIL)")
# ===========================================================================
print("""  This suite does NOT claim, and none of QF-0..QF-3 establishes:
    (i)   A solution of the measurement problem (QF-3 shows decoherence/pointer-selection
          machinery on the tick/record sector; it does not address outcome definiteness).
    (ii)  A Gleason derivation of the Born rule INDEPENDENT of A3 -- the QF-1 theorem-check is
          explicitly CONDITIONAL on the A3 purification axiom (printed above, every run).
    (iii) Spacelike-separated measurement EVENTS. QF-2 is a FIXED-TIME vacuum-correlation read on
          the net's own Gaussian state -- no detector clicks, no signalling/no-loophole language,
          no Bell-experiment causal structure is claimed.
    (iv)  Decoherence beyond the tick/record sector (QF-3's GKLS/pointer results are scoped to the
          compression-channel construction of phase3_1/phase3_2, not a general open-system claim).
    (v)   Any interpretation commitment (Everettian, Bohmian, Copenhagen, ...): the contracts are
          instantiation checks on the object, not a choice of interpretation.
  These remain OPEN and are carried into adapters/README.md as declared, unclaimed scope.""")

# ===========================================================================
banner("QF-2b  THE SMEARED BELL READ  (pre-reg internal research notes,"
       " commit 13bcb03; S3/QF-2's named follow-up -- widens ONLY the observable family)")
# ===========================================================================
print("""  ARCHITECT ADJUDICATIONS (frozen, printed verbatim per the pre-reg):
  (1) THE COMPLEX-FLUX BRANCH IS DEAD -- killed by the framework's own theorems: any phase added
      to the patch hopping is either cell-periodic => PURE GAUGE (the COVER GAUGE-TRIVIALITY
      theorem, I6/holonomy program) or a chosen finite-k sector => decoration in a vacuum read (the
      HOLONOMY-TRIVIALITY theorem, 192/192 cover-closed cycles => +I exactly, locates phase physics
      at finite-k/tick). The real flux-free hopping IS the derived object (machine-checked in F-0
      below). Two complex doors remain NAMED, BOOKED, OUT OF SCOPE (not implemented here): (i) a
      DERIVED non-product patch-level quasi-free state omega with Im C != 0 restricting to the
      cell's C=(I+iJ6)/2 (a new derivation station); (ii) the finite-k/tick sector.
  (2) THE FAMILY RULE IS INSTRUMENT, NOT SHOPPING -- smeared modes = top-r SINGULAR VECTORS of
      X_AB = Re C_full[A,B] (deterministic, knob-free; r in {1,2}; declared BEFORE any covariance
      is computed). Verdict thresholds UNCHANGED: VIOLATION-FOUND >= 2.05; TSIRELSON-BREACH >
      2sqrt2+1e-9 (crisis, exits loudly); else NO-VIOLATION-IN-FAMILY.
  (3) CONTROL-FIRST -- chain_vacuum(400) runs BEFORE the patch, no lore gate.
  (4) TERMINALITY PRE-ADJUDICATED -- a double-null (instrument validates + both families return
      NO-VIOLATION at all ladder rungs) is an OBJECT-LEVEL statement, NOT classicality, NOT
      terminal (the two named doors of (1) survive).
""")

qf2b_all_definite = True   # gates QF-2b's own additive contribution to the final exit condition


def qf2b_verdict(S_max):
    two_sqrt2_ = 2 * math.sqrt(2)
    if S_max > two_sqrt2_ + 1e-9:
        return "TSIRELSON-BREACH"
    elif S_max >= 2.05:
        return "VIOLATION-FOUND"
    else:
        return "NO-VIOLATION-IN-FAMILY"


# ===========================================================================
banner("QF-2b F-0  THE FLUX ADJUDICATION  (machine-checked)")
# ===========================================================================
_im_max = float(np.max(np.abs(C_full5.imag)))
check("[F-0] the shipped vacuum C_full (Patch(M=5)) is EXACTLY real: max|Im C_full| == 0.0 "
      "(the derived, flux-free object)", _im_max == 0.0, detail=f"max|Im|={_im_max:.3e}")
print("""    ADJUDICATION 1 (durable contract): the real (flux-free) hopping IS the derived object;
    the complex-flux branch is dead by the cover gauge-triviality theorem (cell-periodic phase =
    pure gauge) and the holonomy-triviality theorem (192/192 cover-closed cycles' Cl(6) holonomy
    = +I exactly; phase physics lives at finite-k/tick, not the vacuum-Gaussian-state level).
    Two OUT-OF-SCOPE doors (named, booked, NOT implemented): (i) a derived non-product patch-level
    state omega with Im C != 0 restricting to M0's C=(I+iJ6)/2; (ii) the finite-k/tick sector.""")

# ===========================================================================
banner("QF-2b HELPERS  (the smearing rule + the r=1 dense analog; F-2 family rule, adjudication 2)")
# ===========================================================================


def svd_smear(C_full, idxA, idxB, r):
    """F-2 FAMILY RULE (frozen): X_AB = Re C_full[A,B] (rows=region-A site indices, cols=region-B);
    SVD X_AB = U diag(S) V^T; region-A's r smeared complex modes = top-r columns of U (isometry
    WA = U[:,:r].T over region-A's OWN site indices, orthonormal by construction); region-B's =
    top-r columns of V (WB = Vt[:r,:]). C_eff = W C_full_sub W^dagger (W block-diagonal across the
    two regions -- each region's smeared modes are built ONLY from that region's own sites) is the
    smeared covariance in the reduced r+r mode basis, reusable by majorana_from_C() UNMODIFIED.
    Deterministic, knob-free (SVD has no continuous weight to tune); r_eff = min(r,|A|,|B|) (F-2's
    r in {1,2} sub-selects the top r_eff <= r columns when a region is smaller than r)."""
    idxA = list(idxA); idxB = list(idxB)
    X = C_full[np.ix_(idxA, idxB)].real
    U, Svals, Vt = np.linalg.svd(X, full_matrices=False)
    r_eff = min(r, len(idxA), len(idxB))
    WA = U[:, :r_eff].T
    WB = Vt[:r_eff, :]
    iso_resid = max(float(np.max(np.abs(WA @ WA.conj().T - np.eye(r_eff)))),
                     float(np.max(np.abs(WB @ WB.conj().T - np.eye(r_eff)))))
    nA, nB = len(idxA), len(idxB)
    W = np.zeros((2 * r_eff, nA + nB), dtype=complex)
    W[:r_eff, :nA] = WA
    W[r_eff:, nA:] = WB
    idxU = idxA + idxB
    Csub = C_full[np.ix_(idxU, idxU)]
    C_eff = W @ Csub @ W.conj().T
    return C_eff, r_eff, Svals[:r_eff], iso_resid


def two_route_check_r1(C_eff):
    """The r=1 ANALOG of two_route_check: only 1 complex smeared mode per region => 2 Majoranas
    each (n=2 total dense modes, dim 4). Cl(2)'s even part is 1-DIMENSIONAL (spanned by i*g0*g1
    alone) -- there is NO continuous angle freedom (a structural fact, not a computational limit);
    this verifies the SINGLE fixed dichotomic pair A=i*g0*g1 (region A), B=i*g2*g3 (region B):
    A^2=I, B^2=I, [A,B]=0, and Wick(-wick4) == dense, on the actual dense operators."""
    Gam = majorana_from_C(C_eff)
    antisym = float(np.max(np.abs(Gam + Gam.T)))
    rho, a_ops, adag_ops = dense_rho_from_C(2, C_eff)
    gam_dense = majorana_ops_dense(2, a_ops, adag_ops)
    Aop = 1j * gam_dense[0] @ gam_dense[1]
    Bop = 1j * gam_dense[2] @ gam_dense[3]
    A2resid = float(np.max(np.abs(Aop @ Aop - np.eye(4))))
    B2resid = float(np.max(np.abs(Bop @ Bop - np.eye(4))))
    comm_resid = float(np.max(np.abs(Aop @ Bop - Bop @ Aop)))
    dense_corr = float(np.real(np.trace(rho @ Aop @ Bop)))
    wick_corr = -wick4(Gam, (0, 1, 2, 3))
    return {"antisym": antisym, "A2resid": A2resid, "B2resid": B2resid, "comm_resid": comm_resid,
            "dense_corr": dense_corr, "wick_corr": wick_corr,
            "diff": abs(dense_corr - wick_corr)}


def run_qf2b_instance(label, C_full, idxA, idxB, r, sep):
    """The F-2 sweep engine: smear (svd_smear), MANDATORY dense mini two-route check BEFORE any
    S_max read, then S_max via the SAME Horodecki/grid-refine machinery as QF-2 (r_eff=2, BYTE-
    IDENTICAL functions T_tensor/S_horodecki/chsh_grid_refine/two_route_check) or the r=1 analog."""
    global qf2b_all_definite, tsirelson_breach
    C_eff, r_eff, svals, iso_resid = svd_smear(C_full, idxA, idxB, r)
    print(f"    [{label}] r={r} (r_eff={r_eff})  |A|={len(idxA)} |B|={len(idxB)}  "
          f"declared sep={sep}   top singular values={np.round(svals, 6)}  "
          f"isometry residual={iso_resid:.2e}")
    if r_eff == 2:
        tr = two_route_check(C_eff, 0.37, 0.91)
    elif r_eff == 1:
        tr = two_route_check_r1(C_eff)
    else:
        raise ValueError(f"r_eff={r_eff} outside the frozen family {{1,2}}")
    check(f"[{label} r={r_eff}] smeared Gamma antisymmetric (<1e-10)", tr["antisym"] < 1e-10,
          detail=f"{tr['antisym']:.2e}")
    check(f"[{label} r={r_eff}] MANDATORY MINI TWO-ROUTE: Wick==dense (<1e-10, BEFORE the S_max "
          "read)", tr["diff"] < 1e-10, detail=f"diff={tr['diff']:.2e}")
    check(f"[{label} r={r_eff}] A^2=I on the actual dense operator (<1e-12)",
          tr["A2resid"] < 1e-12, detail=f"{tr['A2resid']:.2e}")
    check(f"[{label} r={r_eff}] B^2=I on the actual dense operator (<1e-12)",
          tr["B2resid"] < 1e-12, detail=f"{tr['B2resid']:.2e}")
    check(f"[{label} r={r_eff}] [A,B]=0 on the actual dense operators (<1e-12)",
          tr["comm_resid"] < 1e-12, detail=f"{tr['comm_resid']:.2e}")
    if r_eff == 2:
        T = tr["T"]
        S_an, sv = S_horodecki(T)
        S_grid, best_th, S_grid_phi_check = chsh_grid_refine(T)
        agree = abs(S_an - S_grid) < 1e-6 and abs(S_an - S_grid_phi_check) < 1e-3
        check(f"[{label} r=2] closed-form vs grid+refine vs literal-phi-grid AGREE", agree,
              detail=f"analytic={S_an:.8f} grid={S_grid:.8f} phi-grid={S_grid_phi_check:.8f}")
        S_max = max(S_an, S_grid, S_grid_phi_check)
    else:
        S_max = 2.0 * abs(tr["dense_corr"])
        print(f"    [{label} r=1] NO ANGLE FREEDOM (Cl(2)'s even part is 1-dimensional: only ONE "
              f"independent even bilinear i*g0*g1 exists per 2-Majorana region -- a structural "
              f"fact, disclosed, not a search limit): single fixed correlator <A B> = "
              f"{tr['dense_corr']:.8f}; S_analog := 2|<A B>| is NOT a CHSH sweep (no second "
              f"setting exists to combine) and is bounded <=2 BY CONSTRUCTION.")
    verdict = qf2b_verdict(S_max)
    if verdict == "TSIRELSON-BREACH":
        tsirelson_breach = True
    definite = verdict in ("VIOLATION-FOUND", "NO-VIOLATION-IN-FAMILY", "TSIRELSON-BREACH")
    qf2b_all_definite = qf2b_all_definite and definite
    check(f"[{label} r={r_eff}] a DEFINITE verdict was reached", definite, detail=verdict)
    print(f"    >>> [{label} r={r_eff}] S_max = {S_max:.8f}  (thresholds: classical 2, "
          f"VIOLATION>=2.05, Tsirelson 2sqrt2={2 * math.sqrt(2):.8f})   VERDICT: {verdict} <<<")
    return {"label": label, "r": r_eff, "sep": sep, "S_max": S_max, "verdict": verdict}


def patch_region_idx_sep(patch, verts, vpos, candA, candB):
    """Factor of run_qf2_instance's own sep computation (byte-identical logic), reused here so
    QF-2b's regions are measured by the EXACT same rule QF-2 already used."""
    idxA = [vpos[v] for v in candA]
    idxB = [vpos[v] for v in candB]
    sep = min(patch.vdist(patch.vidx[verts[i]], patch.vidx[verts[j]]) for i in idxA for j in idxB)
    return idxA, idxB, sep


qf2b_results = []

# ===========================================================================
banner("QF-2b F-1  THE CONTROL  (chain_vacuum(400), critical-chain Dirac vacuum; NO lore gate)")
# ===========================================================================
L_chain = 400
C_chain = net.chain_vacuum(L_chain)
print(f"    chain_vacuum({L_chain}) returns the {C_chain.shape} critical half-filled free-fermion "
      f"chain covariance (real, symmetric by inspection); center={L_chain // 2}, "
      f"interval size=8 sites each side, symmetric separations declared = {{2, 20}}")
_chain_im_check = float(np.max(np.abs(C_chain.imag))) if np.iscomplexobj(C_chain) else 0.0
check("[F-1] chain_vacuum(400) covariance is real (context: same zero-block structure as the "
      "patch)", _chain_im_check == 0.0, detail=f"max|Im|={_chain_im_check:.3e}")
_ctr = L_chain // 2
_size_ctrl = 8
for sep_c in (2, 20):
    idxA_c = list(range(_ctr - sep_c - _size_ctrl, _ctr - sep_c))
    idxB_c = list(range(_ctr + sep_c, _ctr + sep_c + _size_ctrl))
    for r_ctrl in (1, 2):
        res = run_qf2b_instance(f"F-1 CONTROL chain sep={sep_c}", C_chain, idxA_c, idxB_c,
                                 r_ctrl, sep_c)
        qf2b_results.append(res)

# ===========================================================================
banner("QF-2b F-2  C1 -- SINGULAR-MODE SMEARED BILINEARS  (the primary; QF-2's own FAR/NEAR + "
       "BFS-radius-2 balls)")
# ===========================================================================
idxA_far2, idxB_far2, sep_far2 = patch_region_idx_sep(patch5, verts5, vpos5, candA_far, candB_far)
idxA_near2, idxB_near2, sep_near2 = patch_region_idx_sep(patch5, verts5, vpos5, candA_near,
                                                          candB_near)

_centerA_v = (0, (1, 1, 1))
_centerB_v = (0, (3, 3, 3))
_cAidx = patch5.vidx[_centerA_v]; _cBidx = patch5.vidx[_centerB_v]
candA_bfs = [v for v in verts5 if patch5.vdist(_cAidx, patch5.vidx[v]) <= 2]
candB_bfs = [v for v in verts5 if patch5.vdist(_cBidx, patch5.vidx[v]) <= 2]
print(f"    BFS balls (radius 2): center A={_centerA_v} ({len(candA_bfs)} vertices), "
      f"center B={_centerB_v} ({len(candB_bfs)} vertices); center-to-center vdist = "
      f"{patch5.vdist(_cAidx, _cBidx)}")
idxA_bfs2, idxB_bfs2, sep_bfs2 = patch_region_idx_sep(patch5, verts5, vpos5, candA_bfs, candB_bfs)
check("[F-2] BFS-radius-2 balls are disjoint (no shared vertex)",
      len(set(idxA_bfs2) & set(idxB_bfs2)) == 0)

_f2_regions = [("F-2 FAR", idxA_far2, idxB_far2, sep_far2),
               ("F-2 NEAR", idxA_near2, idxB_near2, sep_near2),
               ("F-2 BFS-ball(r=2)", idxA_bfs2, idxB_bfs2, sep_bfs2)]
for name, idxA_, idxB_, sep_ in _f2_regions:
    for r_ in (1, 2):
        res = run_qf2b_instance(name, C_full5, idxA_, idxB_, r_, sep_)
        qf2b_results.append(res)

banner("QF-2b F-2  SUMMARY TABLE  (system x region-pair x family r x separation)")
print(f"  {'label':28s} {'r':>2s} {'sep':>4s} {'S_max':>12s}  verdict")
for r_ in qf2b_results:
    print(f"  {r_['label']:28s} {r_['r']:>2d} {r_['sep']:>4d} {r_['S_max']:>12.6f}  {r_['verdict']}")

# ===========================================================================
banner("QF-2b F-3  C3 -- THE QUARTIC PAIR  (the quadratic-obstruction test, on the FROZEN F-2 "
       "r=2 modes)")
# ===========================================================================


def wick_general(Gam, idx_ordered):
    """General fermionic Wick/Pfaffian contraction for an EVEN number of DISTINCT Majorana indices
    IN THE GIVEN OPERATOR ORDER (the recursion below handles fermionic reordering signs internally
    -- no external sort/permutation-sign bookkeeping needed, unlike wick4's approach).
    C2[a,b] := <gamma_a gamma_b> = delta_ab + i*Gam[a,b] (majorana_from_C's OWN convention, run
    backwards). Standard recursive Wick identity for fermionic operators:
      <O_1 O_2 ... O_2m> = sum_{k=2}^{2m} (-1)^k <O_1 O_k> <O_2...O_(k-1) O_(k+1)...O_2m>.
    VALIDATED below (BEFORE use in the sweep): at m=2 this is proven algebraically identical to
    wick4's own formula (both expand to -Gam_ab*Gam_cd+Gam_ac*Gam_bd-Gam_ad*Gam_bc for sorted
    a<b<c<d), cross-checked numerically; at m=4 (8-point) it is validated against a dense
    many-body Slater-determinant expectation on the mini-instance (<1e-10)."""
    n = len(idx_ordered)
    if n == 0:
        return 1.0 + 0j
    if n == 2:
        a, b = idx_ordered
        d = 1.0 if a == b else 0.0
        return d + 1j * Gam[a, b]
    o1 = idx_ordered[0]
    rest = idx_ordered[1:]
    total = 0.0 + 0j
    for k in range(len(rest)):
        d = 1.0 if o1 == rest[k] else 0.0
        c2 = d + 1j * Gam[o1, rest[k]]
        sub = rest[:k] + rest[k + 1:]
        total += ((-1) ** k) * c2 * wick_general(Gam, sub)
    return total


# validation (i): m=2 against wick4, on a throwaway random antisymmetric Gamma
_rng_val = np.random.default_rng(3)
_Gtest = _rng_val.standard_normal((4, 4))
_Gtest = _Gtest - _Gtest.T
_wg2 = wick_general(_Gtest, [0, 1, 2, 3])
_w4v = wick4(_Gtest, (0, 1, 2, 3))
check("[F-3] wick_general VALIDATION (m=2): reproduces wick4 EXACTLY on a random antisymmetric "
      "Gamma (<1e-10, BEFORE use at m=4)", abs(_wg2.real - _w4v) < 1e-10 and abs(_wg2.imag) < 1e-10,
      detail=f"wick_general={_wg2:.10f} wick4={_w4v:.10f}")

# the FROZEN F-2 r=2 modes (re-derived deterministically -- same svd_smear rule, same regions)
Ceff_far3, _, _, _ = svd_smear(C_full5, idxA_far2, idxB_far2, 2)
Gam_far3 = majorana_from_C(Ceff_far3)
n_modes_far3 = 4  # 2 region-A smeared modes + 2 region-B smeared modes
rho_far3, a3, adag3 = dense_rho_from_C(n_modes_far3, Ceff_far3)
gam3 = majorana_ops_dense(n_modes_far3, a3, adag3)
m = gam3  # m[0..3] region A's own 4 Majoranas, m[4..7] region B's own 4 Majoranas
I16 = np.eye(16)


def _resid_eq(X, Y):
    return float(np.max(np.abs(X - Y)))


print("    F-3 CONSTRUCTION 1 (pre-reg literal): Q1=(i*m1*m2)(i*m3*m4), Q2=(i*m1*m3)(i*m2*m4) "
      "on region A's own 4 Majoranas {m0,m1,m2,m3} (0-indexed):")
B01 = 1j * m[0] @ m[1]; B02 = 1j * m[0] @ m[2]; B03 = 1j * m[0] @ m[3]
B12 = 1j * m[1] @ m[2]; B13 = 1j * m[1] @ m[3]; B23 = 1j * m[2] @ m[3]
Q1 = B01 @ B23
Q2 = B02 @ B13
Q1_sq = _resid_eq(Q1 @ Q1, I16); Q2_sq = _resid_eq(Q2 @ Q2, I16)
Q1Q2_acomm = _resid_eq(Q1 @ Q2 + Q2 @ Q1, np.zeros((16, 16)))
Q1Q2_comm = _resid_eq(Q1 @ Q2 - Q2 @ Q1, np.zeros((16, 16)))
check("[F-3 construction 1] Q1^2=I, Q2^2=I (dichotomy, dense, <1e-10)",
      Q1_sq < 1e-10 and Q2_sq < 1e-10, detail=f"Q1^2-I={Q1_sq:.2e} Q2^2-I={Q2_sq:.2e}")
print(f"    Q1,Q2 commutator residual={Q1Q2_comm:.2e}  anticommutator residual={Q1Q2_acomm:.2e}")
constr1_anticommutes = Q1Q2_acomm < 1e-10
constr1_confirms_degeneracy = Q1Q2_comm < 1e-10 and Q1Q2_acomm > 1.99
check("[F-3 construction 1] CONFIRMS the algebraic-degeneracy PREDICTION (G5a falsification-probe "
      "style: the pre-reg's literal pairing is PREDICTED to fail anticommutation because all 3 "
      "pairings of 4 Majoranas into 2 bilinear products give the SAME operator up to an overall "
      "sign -- the region's UNIQUE quartic = its fermion-parity operator; a probe that instead DID "
      "anticommute would be the contract failure here)",
      constr1_confirms_degeneracy and not constr1_anticommutes,
      detail=f"comm={Q1Q2_comm:.2e} acomm={Q1Q2_acomm:.2e} (Q1,Q2 commute/proportional, as "
      "predicted -- NOT a genuine anticommuting pair)")

Q3 = B03 @ B12
print(f"    cross-check (third pairing): Q3=(i*m1*m4)(i*m2*m3); Q1 vs Q3 commutator="
      f"{_resid_eq(Q1@Q3-Q3@Q1, np.zeros((16,16))):.2e}  "
      f"Q2 vs Q3 commutator={_resid_eq(Q2@Q3-Q3@Q2, np.zeros((16,16))):.2e}  "
      f"(all three pairings mutually commute => confirms they are proportional, not independent)")

print("\n    F-3 CONSTRUCTION 2 (declared alternative per pre-reg instruction: mixed "
      "bilinear-vs-quartic pair) Q1'=i*m0*m1 (bilinear), Q2'=m0*m1*m2*m3 (the region's unique "
      "quartic = fermion parity):")
Gamma_A = m[0] @ m[1] @ m[2] @ m[3]
GA_herm = _resid_eq(Gamma_A, Gamma_A.conj().T)
GA_sq = _resid_eq(Gamma_A @ Gamma_A, I16)
check("[F-3 construction 2] Gamma_A Hermitian + Gamma_A^2=I (dichotomy, dense, <1e-10)",
      GA_herm < 1e-10 and GA_sq < 1e-10, detail=f"herm={GA_herm:.2e} sq-I={GA_sq:.2e}")
mixed_acomm = _resid_eq(B01 @ Gamma_A + Gamma_A @ B01, np.zeros((16, 16)))
mixed_comm = _resid_eq(B01 @ Gamma_A - Gamma_A @ B01, np.zeros((16, 16)))
print(f"    {{i*m0*m1, Gamma_A}} anticommutator residual={mixed_acomm:.2e}   "
      f"commutator residual={mixed_comm:.2e}")
constr2_anticommutes = mixed_acomm < 1e-10
constr2_confirms_centrality = mixed_comm < 1e-10 and mixed_acomm > 1.99
check("[F-3 construction 2] CONFIRMS the CENTRALITY prediction (declared alternative per pre-reg "
      "instruction, also PREDICTED to fail anticommutation: verified dense across ALL 6 region-A "
      "bilinears vs Gamma_A below -- the parity operator is CENTRAL in the even sub-algebra, an "
      "algebraic fact; a probe that instead anticommuted would be the contract failure here)",
      constr2_confirms_centrality and not constr2_anticommutes,
      detail=f"comm={mixed_comm:.2e} acomm={mixed_acomm:.2e} (bilinear and quartic commute, as "
      "predicted -- NOT a genuine anticommuting pair)")

_all_bilinears = {"01": B01, "02": B02, "03": B03, "12": B12, "13": B13, "23": B23}
_central_worst = 0.0
for _nm, _B in _all_bilinears.items():
    _c = _resid_eq(_B @ Gamma_A - Gamma_A @ _B, np.zeros((16, 16)))
    _central_worst = max(_central_worst, _c)
print(f"    EXHAUSTIVE CHECK: max over all 6 bilinear-vs-Gamma_A COMMUTATOR residuals = "
      f"{_central_worst:.2e} (all ~0 => Gamma_A commutes with the ENTIRE even sub-algebra: it is "
      f"CENTRAL, hence no bilinear can ever anticommute with it)")
check("[F-3] Gamma_A is CENTRAL in region A's own even sub-algebra (all 6 bilinear commutators "
      "<1e-10, dense, exhaustive)", _central_worst < 1e-10, detail=f"worst={_central_worst:.2e}")

print("""
    F-3 DISCLOSED FINDING (deviation from the pre-reg's literal ask, prominently flagged): with
    EXACTLY 4 Majoranas per region (the frozen r=2 family), the even sub-algebra's only degree-4
    element is the region's OWN fermion-parity operator, UNIQUE up to an overall sign (all 3
    distinct pairings of 2 bilinears coincide, verified dense) and PROVABLY CENTRAL (commutes with
    all 6 bilinears, verified dense, exhaustive). Therefore NO pair of genuinely independent,
    mutually anticommuting quartic-degree observables exists within a single r=2 (4-Majorana)
    region -- this is an ALGEBRAIC IMPOSSIBILITY, not a failed search; both the pre-reg's literal
    construction AND its own declared alternative were implemented and BOTH dense-verified to fail
    anticommutation, for this identified structural reason. The CLOSEST COMPLIANT, well-defined
    substitute implemented below: report the single FIXED (no-angle-freedom, same degeneracy class
    as the r=1 bilinear case above) region-A-parity/region-B-parity correlator <Gamma_A Gamma_B>,
    an honest 8-point object, via the newly-validated general Wick contraction -- informational,
    explicitly NOT a CHSH sweep (no second setting exists to combine).""")

Gamma_B_dense = gam3[4] @ gam3[5] @ gam3[6] @ gam3[7]
_dense_corr_quartic = float(np.real(np.trace(rho_far3 @ Gamma_A @ Gamma_B_dense)))
_wick_corr_quartic = wick_general(Gam_far3, [0, 1, 2, 3, 4, 5, 6, 7])
_diff_quartic = abs(_dense_corr_quartic - _wick_corr_quartic.real)
check("[F-3] wick_general VALIDATION (m=4, 8-point <Gamma_A Gamma_B>): Wick == dense many-body "
      "(<1e-10, BEFORE the cross-region sweep)", _diff_quartic < 1e-10 and abs(_wick_corr_quartic.imag) < 1e-8,
      detail=f"dense={_dense_corr_quartic:.10f} wick={_wick_corr_quartic.real:.10f} "
      f"diff={_diff_quartic:.2e} Im(wick)={_wick_corr_quartic.imag:.2e}")

print(f"\n    F-3 INFORMATIONAL SWEEP: <Gamma_A Gamma_B> across the F-2 region-pairs (r=2 frozen "
      f"modes), via the validated wick_general (big-patch, no dense feasible there):")
qf2b_quartic_results = []
for name, idxA_, idxB_, sep_ in _f2_regions:
    Ceff_, _, _, _ = svd_smear(C_full5, idxA_, idxB_, 2)
    Gam_ = majorana_from_C(Ceff_)
    val_ = wick_general(Gam_, [0, 1, 2, 3, 4, 5, 6, 7])
    S_quartic = 2.0 * abs(val_.real)
    verd_quartic = qf2b_verdict(S_quartic)
    if verd_quartic == "TSIRELSON-BREACH":
        tsirelson_breach = True
    print(f"    [{name} r=2 quartic] sep={sep_:<3d} <Gamma_A Gamma_B>={val_.real:.8f} "
          f"(Im={val_.imag:.2e})  S_analog=2|.|={S_quartic:.8f}  verdict={verd_quartic}")
    qf2b_quartic_results.append({"label": name, "sep": sep_, "S_max": S_quartic,
                                  "verdict": verd_quartic})
qf2b_all_definite = qf2b_all_definite and all(
    r_["verdict"] in ("VIOLATION-FOUND", "NO-VIOLATION-IN-FAMILY", "TSIRELSON-BREACH")
    for r_ in qf2b_quartic_results)

# ===========================================================================
banner("QF-2b F-4  THE SEPARATION LADDER  (sep in {1,3,9,27}, F-2 r=2 family, NO re-optimization)")
# ===========================================================================


def find_cellB_for_sep(patch, M, cellA, brsA, brsB, target):
    """Deterministic geometric search (canonical box order x-slowest/z-fastest, matching Patch's
    own `itertools.product(range(M),repeat=3)` box construction) for the FIRST cellB != cellA whose
    region-pair minimum vertex-graph separation (branches brsA at cellA vs brsB at cellB) equals
    `target` EXACTLY. Purely geometric -- searched BEFORE any covariance/CHSH computation, so it
    cannot be a re-optimization on S_max."""
    for x in range(M):
        for y in range(M):
            for z in range(M):
                cellB = (x, y, z)
                if cellB == cellA:
                    continue
                seps = [patch.vdist(patch.vidx[(a, cellA)], patch.vidx[(b, cellB)])
                        for a in brsA for b in brsB]
                if min(seps) == target:
                    return cellB
    return None


_cellA_ladder = (0, 0, 0)
_ladder_defs = []
# sep=1 requires the differing-branch SAME-cell construction (proven: no disjoint-cell same-branch
# choice realizes sep=1 -- the minimum achievable same-branch separation at any disjoint cell is 2,
# verified by exhaustive search over Patch(M=5)'s box); declared explicitly, not searched.
_ladder_defs.append((1, (0, 1), (2, 3), _cellA_ladder))
for _target in (3, 9, 27):
    _cellB = find_cellB_for_sep(patch5, Msize, _cellA_ladder, (0, 1), (0, 1), _target)
    _ladder_defs.append((_target, (0, 1), (0, 1), _cellB))

print("    ladder region construction (region A fixed = branches{0,1}@(0,0,0)):")
qf2b_ladder = []
for _target, _brsA, _brsB, _cellB in _ladder_defs:
    candA_L = [(b, _cellA_ladder) for b in _brsA]
    candB_L = [(b, _cellB) for b in _brsB]
    idxA_L, idxB_L, sep_L = patch_region_idx_sep(patch5, verts5, vpos5, candA_L, candB_L)
    print(f"      target sep={_target:<3d} region B: branches{_brsB}@{_cellB}  "
          f"(achieved sep={sep_L})")
    check(f"[F-4] ladder rung sep={_target} achieves the EXACT target separation",
          sep_L == _target, detail=f"achieved={sep_L}")
    res = run_qf2b_instance(f"F-4 LADDER sep={_target}", C_full5, idxA_L, idxB_L, 2, sep_L)
    qf2b_ladder.append(res)

banner("QF-2b F-4  THE CURVE  (raw, whatever it shows)")
print(f"  {'sep':>4s} {'S_max':>12s}  verdict")
for res in qf2b_ladder:
    print(f"  {res['sep']:>4d} {res['S_max']:>12.6f}  {res['verdict']}")

# ===========================================================================
banner("QF-2b F-5  SCOPE + THE BOOKED SENTENCE")
# ===========================================================================
print("""  QF-4's scope declaration (re-printed, unchanged): this suite does NOT claim a solution of
  the measurement problem, a Gleason derivation of Born independent of A3, spacelike-separated
  measurement EVENTS (QF-2/QF-2b are fixed-time vacuum-correlation reads), decoherence beyond the
  tick/record sector, or any interpretation commitment. QF-2b adds nothing to this list; it only
  widens the observable family on the SAME object and state.""")

_instrument_validates = ok_all  # every dense/algebra check above passed (or did not: see per-check)
_all_qf2b_novint = all(r_["verdict"] == "NO-VIOLATION-IN-FAMILY" for r_ in qf2b_results) and \
    all(r_["verdict"] == "NO-VIOLATION-IN-FAMILY" for r_ in qf2b_ladder)
if tsirelson_breach:
    _booked = ("A TSIRELSON-BREACH was detected -- see the crisis report below; this is NOT the "
               "double-null branch.")
elif _all_qf2b_novint and _instrument_validates:
    _booked = ("ADJUDICATION 4, DOUBLE-NULL BRANCH: the instrument validates (all dense/algebra "
               "checks above passed) and BOTH families (C1 smeared bilinears at r in {1,2} across "
               "FAR/NEAR/BFS-ball/ladder, and the C3 quartic-pair route -- which turned out to be "
               "algebraically degenerate at r=2, see F-3) return NO-VIOLATION at every rung. This "
               "is an OBJECT-LEVEL statement: this Dirac-sea vacuum's accessible even-algebra "
               "correlations do not violate CHSH at these separations and families. It is still "
               "NOT classicality (QF-4 scope) and still NOT terminal for vacuum-Bell on the "
               "framework -- the two named doors of adjudication 1 (a derived Im C != 0 state; "
               "the finite-k/tick sector) survive as open, out-of-scope routes.")
else:
    _booked = ("NOT a clean double-null: either a VIOLATION-FOUND occurred in some family/rung, or "
               "at least one instrument/algebra check failed -- see the per-contract detail above "
               "(a finding, not a bug); the adjudication-4 double-null sentence does not apply "
               "as stated.")
print(f"\n  >>> BOOKED SENTENCE: {_booked}")

# ===========================================================================
banner("QF-2c  THE BELL COMPLETION -- two legs  (pre-reg internal research notes"
       "prereg_2026-07-10.md, commit aef6148; QF-2b's own checker-validated follow-up)")
# ===========================================================================
print("""  FROZEN PER THE PRE-REG (verbatim): the prior-art sweep IS QF-2b's own adversarial check
  (booked c661340) -- it PROVED the shared-pivot family's rank-1 lemma (that family can never
  violate), CONFIRMED the natural per-mode family genuinely reaches rank 2 (diagnostic S_max ~
  1.62 in a 60-trial sample -- a family-space finding, quoted as HISTORY, never as a target/floor),
  and VERIFIED independent anticommuting quartics exist at r=3 (6 Majoranas, necessarily
  overlapping support). QF-2c runs exactly those two checker-validated legs on the SAME object/
  state, SAME thresholds (VIOLATION-FOUND >= 2.05; TSIRELSON-BREACH > 2sqrt2+1e-9, crisis, exits
  loudly), SAME crisis branch (the shared `tsirelson_breach` flag). Both legs reuse QF-2b's
  machinery BYTE-IDENTICALLY: svd_smear, majorana_from_C, wick4, wick_general, dense_rho_from_C,
  majorana_ops_dense, S_horodecki, chsh_grid_refine, qf2b_verdict, patch_region_idx_sep,
  find_cellB_for_sep/_ladder_defs, and every QF-2b region construction (candA_far/candB_far,
  candA_near/candB_near, candA_bfs/candB_bfs, the chain_vacuum(400) control). QF-0..QF-4 and QF-2b
  above are byte-untouched; this section is a pure append.
""")

qf2c_all_definite = True   # gates QF-2c's own additive contribution to the final exit condition


def qf2c_verdict(S_max):
    """Thin, self-documenting alias: QF-2c's thresholds are BYTE-IDENTICAL to qf2b_verdict's
    (2.05 / 2sqrt2+1e-9) -- delegates rather than re-implements."""
    return qf2b_verdict(S_max)


# ===========================================================================
banner("QF-2c LEG A  THE NATURAL PER-MODE FAMILY  (frozen r=2, basis [(g0,g1),(g2,g3)] per region "
       "-- the QF-2b rank-2-capable rule, unchanged; ONLY the observable 2-plane differs from "
       "T_tensor's shared-pivot basis)")
# ===========================================================================


def T_tensor_natural(Gam, gA, gB):
    """QF-2c LEG A (frozen): the NATURAL per-mode 2-plane basis -- each region's OWN smeared-mode
    bilinear pair [(g0,g1), (g2,g3)] (mode-1's own bilinear, mode-2's own bilinear) instead of
    T_tensor's (line ~381 above) shared-pivot basis [(g0,g1),(g0,g2)]. A SIBLING function:
    T_tensor itself is NOT modified and NOT called any differently anywhere else in this file.
    Same wick4 Pfaffian formula, same convention, same (2,2) return shape -- reusable BYTE-
    IDENTICALLY by S_horodecki/chsh_grid_refine (both are basis-agnostic: they only see the raw
    T matrix)."""
    g0, g1, g2, g3 = gA
    h0, h1, h2, h3 = gB
    basisA = [(g0, g1), (g2, g3)]
    basisB = [(h0, h1), (h2, h3)]
    T = np.zeros((2, 2))
    for ia, (p, qidx) in enumerate(basisA):
        for ib, (r, s) in enumerate(basisB):
            T[ia, ib] = -wick4(Gam, (p, qidx, r, s))
    return T


def two_route_check_natural(C_sub, thetaA, thetaB):
    """The LEG-A analog of two_route_check (line ~460 above): IDENTICAL construction
    (dense_rho_from_C, majorana_ops_dense, the mini 4-mode/16-dim instance) except A(theta)/B(phi)
    are built from the NATURAL basis (mode-1's own bilinear P1=i*g0*g1, mode-2's own bilinear
    P2=i*g2*g3) instead of the shared-pivot basis. two_route_check itself is untouched; this is a
    sibling. ALSO reports the (anti)commutation structure of P1,P2 THEMSELVES (region-local),
    discovered live: for the shared-pivot basis P1,P2 SHARE a Majorana (g0) and anticommute
    (proven earlier in this file: T_tensor's own A(theta)^2=I passes at <1e-12 in QF-2/QF-2b);
    P1,P2 here act on DISJOINT Majorana pairs (mode-1's {g0,g1} vs mode-2's {g2,g3}) -- the SAME
    even-even-disjoint-commute CAR fact used throughout this file (e.g. twisted_locality_holds)
    predicts they COMMUTE instead. Checked live, not assumed."""
    Gam = majorana_from_C(C_sub)
    antisym = float(np.max(np.abs(Gam + Gam.T)))
    T = T_tensor_natural(Gam, [0, 1, 2, 3], [4, 5, 6, 7])
    rho, a_ops, adag_ops = dense_rho_from_C(4, C_sub)
    gam_dense = majorana_ops_dense(4, a_ops, adag_ops)
    I16 = np.eye(16)
    P1_A = 1j * gam_dense[0] @ gam_dense[1]
    P2_A = 1j * gam_dense[2] @ gam_dense[3]
    P1_B = 1j * gam_dense[4] @ gam_dense[5]
    P2_B = 1j * gam_dense[6] @ gam_dense[7]
    P1A_sq = float(np.max(np.abs(P1_A @ P1_A - I16)))
    P2A_sq = float(np.max(np.abs(P2_A @ P2_A - I16)))
    P1B_sq = float(np.max(np.abs(P1_B @ P1_B - I16)))
    P2B_sq = float(np.max(np.abs(P2_B @ P2_B - I16)))
    commA = float(np.max(np.abs(P1_A @ P2_A - P2_A @ P1_A)))
    acommA = float(np.max(np.abs(P1_A @ P2_A + P2_A @ P1_A)))
    commB = float(np.max(np.abs(P1_B @ P2_B - P2_B @ P1_B)))
    acommB = float(np.max(np.abs(P1_B @ P2_B + P2_B @ P1_B)))

    def A_dense_natural(theta, g0, g1, g2, g3):
        P1 = 1j * gam_dense[g0] @ gam_dense[g1]
        P2 = 1j * gam_dense[g2] @ gam_dense[g3]
        return math.cos(theta) * P1 + math.sin(theta) * P2

    Aop = A_dense_natural(thetaA, 0, 1, 2, 3)
    Bop = A_dense_natural(thetaB, 4, 5, 6, 7)
    A2resid = float(np.max(np.abs(Aop @ Aop - I16)))
    B2resid = float(np.max(np.abs(Bop @ Bop - I16)))
    comm_resid = float(np.max(np.abs(Aop @ Bop - Bop @ Aop)))
    dense_corr = float(np.real(np.trace(rho @ Aop @ Bop)))
    wick_corr = (math.cos(thetaA) * math.cos(thetaB) * T[0, 0]
                 + math.cos(thetaA) * math.sin(thetaB) * T[0, 1]
                 + math.sin(thetaA) * math.cos(thetaB) * T[1, 0]
                 + math.sin(thetaA) * math.sin(thetaB) * T[1, 1])
    return {
        "antisym": antisym, "T": T, "A2resid": A2resid, "B2resid": B2resid,
        "comm_resid": comm_resid, "dense_corr": dense_corr, "wick_corr": wick_corr,
        "diff": abs(dense_corr - wick_corr),
        "P1A_sq": P1A_sq, "P2A_sq": P2A_sq, "P1B_sq": P1B_sq, "P2B_sq": P2B_sq,
        "commA": commA, "acommA": acommA, "commB": commB, "acommB": acommB,
    }


def run_qf2c_lega_instance(label, C_full, idxA, idxB, sep):
    """LEG A instance runner: svd_smear at the FROZEN r=2 (QF-2b's own rule, unchanged), the
    MANDATORY dense mini two-route check (natural basis) BEFORE the S_max read (the QF-2b
    per-instance pattern).

    DISCLOSED DEVIATION (found live by the mandatory check, exactly the safeguard it exists for):
    the pre-reg's ask was to read S_max via the SAME S_horodecki/chsh_grid_refine machinery as
    QF-2/QF-2b. That machinery's rotation family A(theta)=cos(theta)P1+sin(theta)P2 is dichotomic
    (A(theta)^2=I) ONLY IF {P1,P2}=0 -- true for the shared-pivot basis (P1,P2 share a Majorana),
    re-confirmed there. For the NATURAL basis P1=i*g0*g1, P2=i*g2*g3 act on DISJOINT Majorana
    pairs, so they PROVABLY COMMUTE instead (an algebraic, state-independent CAR fact, verified
    below on the actual dense operators) -- A(theta)^2 != I at generic theta, so the continuous
    rotation family is NOT a valid dichotomic family here, and S_horodecki's SVD-optimization
    (which assumes exactly that) is NOT LICENSED. This is checked and CONFIRMED, not assumed.

    The CLOSEST COMPLIANT, well-defined substitute (same discipline as QF-2b F-3): the classic
    FIXED 4-correlator Bell-CHSH statistic S_fixed = max over the 4 equivalent sign conventions of
    |+-T00+-T01+-T10+-T11| (an odd number of minus signs), built directly from T's own four
    entries <P_a^A P_b^B> -- valid for ANY four individually-dichotomic operators (each P_a^A/P_b^B
    verified P^2=I on its own, [A,B]=0 cross-region, both checked below), with NO requirement that
    {P1,P2}=0 (that requirement is needed ONLY for the rotation-optimization shortcut, not for the
    Bell-CHSH inequality itself). This IS reported as the instance's real, licensed S_max/verdict;
    the informal (unlicensed) S_horodecki number is ALSO printed, clearly flagged, for
    transparency/comparison only -- never used for the verdict."""
    global qf2c_all_definite, tsirelson_breach
    C_eff, r_eff, svals, iso_resid = svd_smear(C_full, idxA, idxB, 2)
    print(f"    [{label}] r=2 (r_eff={r_eff})  |A|={len(idxA)} |B|={len(idxB)}  declared sep={sep}  "
          f"top singular values={np.round(svals, 6)}  isometry residual={iso_resid:.2e}")
    check(f"[LEG-A {label}] r_eff == 2 (the frozen rank-2-capable family)", r_eff == 2,
          detail=f"r_eff={r_eff}")
    tr = two_route_check_natural(C_eff, 0.37, 0.91)
    check(f"[LEG-A {label}] smeared Gamma antisymmetric (<1e-10)", tr["antisym"] < 1e-10,
          detail=f"{tr['antisym']:.2e}")
    check(f"[LEG-A {label}] MANDATORY MINI TWO-ROUTE (natural basis): Wick==dense (<1e-10, BEFORE "
          "the S_max read; this bilinear identity holds regardless of dichotomic-ness)",
          tr["diff"] < 1e-10, detail=f"diff={tr['diff']:.2e}")
    check(f"[LEG-A {label}] each PURE setting is individually dichotomic: P1_A^2=I, P2_A^2=I, "
          "P1_B^2=I, P2_B^2=I (<1e-12, dense)",
          tr["P1A_sq"] < 1e-12 and tr["P2A_sq"] < 1e-12 and tr["P1B_sq"] < 1e-12
          and tr["P2B_sq"] < 1e-12,
          detail=f"P1A={tr['P1A_sq']:.1e} P2A={tr['P2A_sq']:.1e} P1B={tr['P1B_sq']:.1e} "
          f"P2B={tr['P2B_sq']:.1e}")
    check(f"[LEG-A {label}] [A,B]=0 on the actual dense operators (<1e-12, cross-region, "
          "independent of the intra-region (anti)commutation structure)", tr["comm_resid"] < 1e-12,
          detail=f"{tr['comm_resid']:.2e}")
    disjoint_commutes = (tr["commA"] < 1e-10 and tr["acommA"] > 1.99
                          and tr["commB"] < 1e-10 and tr["acommB"] > 1.99)
    check(f"[LEG-A {label}] CONFIRMS the disjoint-support COMMUTING prediction (P1,P2 act on "
          "DISJOINT Majorana pairs -> [P1,P2]=0, NOT {P1,P2}=0, unlike the shared-pivot basis; "
          "the mandatory-check safeguard catching exactly this)", disjoint_commutes,
          detail=f"commA={tr['commA']:.2e} acommA={tr['acommA']:.2e} commB={tr['commB']:.2e} "
          f"acommB={tr['acommB']:.2e}")
    non_dichotomic_confirmed = tr["A2resid"] > 1e-3 and tr["B2resid"] > 1e-3
    check(f"[LEG-A {label}] CONSEQUENTLY CONFIRMS A(theta)^2 != I at a generic angle (theta=0.37, "
          "0.91 rad) -- the rotation family is NOT dichotomic here (PREDICTED by the commuting "
          "structure just confirmed, NOT a bug)", non_dichotomic_confirmed,
          detail=f"A2resid={tr['A2resid']:.2e} B2resid={tr['B2resid']:.2e}")

    T = tr["T"]
    sv = np.linalg.svd(T, compute_uv=False)
    S_horo_informal, _ = S_horodecki(T)
    print(f"    [{label}] T (natural basis) =\n{T}")
    print(f"    [{label}] singular values of T = {sv}  (BOTH > 0 => the rank-2 escape confirmed "
          f"LIVE on this instance, unlike the shared-pivot family's rank-1 lemma)")
    print(f"    [{label}] S_horodecki (INFORMAL, NOT LICENSED -- P1,P2 commute, so the rotation "
          f"optimization it assumes is invalid here) = {S_horo_informal:.8f}  [printed for "
          f"transparency ONLY, never used for the verdict]")
    S_options = [T[0, 0] + T[0, 1] + T[1, 0] - T[1, 1],
                 T[0, 0] + T[0, 1] - T[1, 0] + T[1, 1],
                 T[0, 0] - T[0, 1] + T[1, 0] + T[1, 1],
                 -T[0, 0] + T[0, 1] + T[1, 0] + T[1, 1]]
    S_fixed = max(abs(s) for s in S_options)
    print(f"    [{label}] S_fixed (LICENSED: the classic FIXED 4-correlator Bell-CHSH statistic, "
          f"max over the 4 equivalent sign conventions of the actual T entries -- requires only "
          f"[A,B]=0 and each setting individually dichotomic, both verified above) = "
          f"{S_fixed:.8f}")
    S_max = S_fixed
    verdict = qf2c_verdict(S_max)
    if verdict == "TSIRELSON-BREACH":
        tsirelson_breach = True
    definite = verdict in ("VIOLATION-FOUND", "NO-VIOLATION-IN-FAMILY", "TSIRELSON-BREACH")
    qf2c_all_definite = qf2c_all_definite and definite
    check(f"[LEG-A {label}] a DEFINITE verdict was reached (on the LICENSED S_fixed)", definite,
          detail=verdict)
    print(f"    >>> [LEG-A {label}] S_max (=S_fixed) = {S_max:.8f}  (classical 2, VIOLATION>=2.05, "
          f"Tsirelson 2sqrt2={2 * math.sqrt(2):.8f})   VERDICT: {verdict} <<<")
    return {"label": label, "sep": sep, "S_max": S_max, "sv": sv, "verdict": verdict}


qf2c_lega_results = []
_lega_regions = [("FAR", idxA_far2, idxB_far2, sep_far2),
                  ("NEAR", idxA_near2, idxB_near2, sep_near2),
                  ("BFS-ball(r=2)", idxA_bfs2, idxB_bfs2, sep_bfs2)]
for name, idxA_, idxB_, sep_ in _lega_regions:
    qf2c_lega_results.append(run_qf2c_lega_instance(name, C_full5, idxA_, idxB_, sep_))

print("\n    LEG A -- the separation ladder (sep in {1,3,9,27}, the SAME ladder region "
      "constructions as QF-2b F-4, reused verbatim, NOT re-searched):")
for _target, _brsA, _brsB, _cellB in _ladder_defs:
    candA_L = [(b, _cellA_ladder) for b in _brsA]
    candB_L = [(b, _cellB) for b in _brsB]
    idxA_L, idxB_L, sep_L = patch_region_idx_sep(patch5, verts5, vpos5, candA_L, candB_L)
    qf2c_lega_results.append(run_qf2c_lega_instance(f"LADDER sep={_target}", C_full5, idxA_L,
                                                      idxB_L, sep_L))

banner("QF-2c LEG A  SUMMARY TABLE  (region-pair x S_max x T singular values x verdict)")
print(f"  {'label':24s} {'sep':>4s} {'S_max':>12s}  singular values of T             verdict")
for r_ in qf2c_lega_results:
    print(f"  {r_['label']:24s} {r_['sep']:>4d} {r_['S_max']:>12.6f}  "
          f"{np.array2string(r_['sv'], precision=6):24s}  {r_['verdict']}")

# ===========================================================================
banner("QF-2c LEG B  THE r=3 QUARTIC PAIR  (6 Majoranas m1..m6 per region; Q1=(i m1 m2)(i m3 m4), "
       "Q2=(i m1 m3)(i m2 m5) -- IDENTICAL construction on BOTH sides)")
# ===========================================================================


def quartic_ops_dense(gam_dense, base):
    """Build the FROZEN quartic pair Q1=(i*m1*m2)(i*m3*m4), Q2=(i*m1*m3)(i*m2*m5) (the pre-reg's
    1-indexed m1..m6 -> 0-indexed m0..m5 here) on ONE region's own 6 Majoranas, starting at
    gam_dense[base:base+6]. The IDENTICAL formula is used on both region A (base=0) and region B
    (base=6) -- 'on BOTH sides identically' per the pre-reg."""
    m = gam_dense[base:base + 6]
    Q1 = (1j * m[0] @ m[1]) @ (1j * m[2] @ m[3])
    Q2 = (1j * m[0] @ m[2]) @ (1j * m[1] @ m[4])
    return Q1, Q2


def quartic_wick_corr(Gam, idxA6, idxB6):
    """The Wick/Pfaffian route to <Q_a^A Q_b^B> for a,b in {1,2}: Q1=(i m_p m_q)(i m_r m_s) =
    i^2*(m_p m_q m_r m_s) = -(ordered product), for ANY quartic of this shape -- so Q_a^A Q_b^B =
    (-1)*(-1)*(ordered 8-Majorana product) = +(ordered product), and <Q_a^A Q_b^B> =
    wick_general(Gam, ordered-8-index-list) with NO extra sign. idxA6/idxB6 are the region's 6
    GLOBAL Majorana indices (local m0..m5 -> idxA6[0..5] / idxB6[0..5])."""
    a0, a1, a2, a3, a4, _a5 = idxA6
    b0, b1, b2, b3, b4, _b5 = idxB6
    order1A, order2A = (a0, a1, a2, a3), (a0, a2, a1, a4)
    order1B, order2B = (b0, b1, b2, b3), (b0, b2, b1, b4)
    T = np.zeros((2, 2), dtype=complex)
    for ia, oA in enumerate((order1A, order2A)):
        for ib, oB in enumerate((order1B, order2B)):
            T[ia, ib] = wick_general(Gam, list(oA) + list(oB))
    return T


def run_qf2c_legb_instance(label, C_full, idxA, idxB, sep):
    """LEG B instance runner: svd_smear at r=3 (the frozen, knob-free rule) -- reports r_eff
    HONESTLY: r_eff = min(3,|A|,|B|) structurally clamps below 3 whenever a region has fewer than
    3 sites (a DISCLOSED structural fact, discovered live below, not assumed or engineered
    around -- the region constructions are FROZEN/reused verbatim, never enlarged). Only
    instances that genuinely achieve r_eff=3 (6 Majoranas/side) carry the quartic-pair CHSH read;
    for those, the FULL dense algebra validation (Q^2=I, {Q1,Q2}=0, [A,B]=0, wick_general==dense)
    is re-verified on EVERY such instance (a superset of the pre-reg's 'one mandatory mini-
    instance before the sweep' -- since svd_smear always reduces to a <=12x12 Gamma regardless of
    the original region size, dense verification is cheap and tractable on every instance, not
    just one)."""
    global qf2c_all_definite, tsirelson_breach
    C_eff, r_eff, svals, iso_resid = svd_smear(C_full, idxA, idxB, 3)
    print(f"    [{label}] r=3 requested (r_eff={r_eff})  |A|={len(idxA)} |B|={len(idxB)}  "
          f"declared sep={sep}  top singular values={np.round(svals, 6)}  "
          f"isometry residual={iso_resid:.2e}")
    if r_eff < 3:
        check(f"[LEG-B {label}] region too small for r=3 (STRUCTURAL, disclosed, not a search "
              "failure)", True, detail=f"r_eff={r_eff}<3, |A|={len(idxA)} |B|={len(idxB)}")
        print(f"    [{label}] STRUCTURALLY EXCLUDED from the quartic-pair sweep: r_eff={r_eff}<3 "
              "(the 6-Majorana-per-side construction needs >=3 sites/side; this region's size is "
              "a FACT of the frozen, reused-verbatim construction, not a limit of the search).")
        return {"label": label, "sep": sep, "r_eff": r_eff, "S_max": float("nan"),
                "sv": np.array([float("nan"), float("nan")]),
                "verdict": "STRUCTURALLY-EXCLUDED (r_eff<3)"}
    Gam = majorana_from_C(C_eff)
    antisym = float(np.max(np.abs(Gam + Gam.T)))
    check(f"[LEG-B {label}] smeared Gamma antisymmetric (<1e-10)", antisym < 1e-10,
          detail=f"{antisym:.2e}")
    rho, a_ops, adag_ops = dense_rho_from_C(6, C_eff)
    gam_dense = majorana_ops_dense(6, a_ops, adag_ops)
    Q1_A, Q2_A = quartic_ops_dense(gam_dense, 0)
    Q1_B, Q2_B = quartic_ops_dense(gam_dense, 6)
    I64 = np.eye(64)
    Q1A_sq = float(np.max(np.abs(Q1_A @ Q1_A - I64)))
    Q2A_sq = float(np.max(np.abs(Q2_A @ Q2_A - I64)))
    Q1B_sq = float(np.max(np.abs(Q1_B @ Q1_B - I64)))
    Q2B_sq = float(np.max(np.abs(Q2_B @ Q2_B - I64)))
    acommA = float(np.max(np.abs(Q1_A @ Q2_A + Q2_A @ Q1_A)))
    acommB = float(np.max(np.abs(Q1_B @ Q2_B + Q2_B @ Q1_B)))
    check(f"[LEG-B {label}] Q1_A^2=I, Q2_A^2=I (dichotomy, dense, <1e-10)",
          Q1A_sq < 1e-10 and Q2A_sq < 1e-10, detail=f"Q1A^2-I={Q1A_sq:.2e} Q2A^2-I={Q2A_sq:.2e}")
    check(f"[LEG-B {label}] Q1_B^2=I, Q2_B^2=I (dichotomy, dense, <1e-10)",
          Q1B_sq < 1e-10 and Q2B_sq < 1e-10, detail=f"Q1B^2-I={Q1B_sq:.2e} Q2B^2-I={Q2B_sq:.2e}")
    check(f"[LEG-B {label}] {{Q1_A,Q2_A}}=0 (anticommute, dense, <1e-10)", acommA < 1e-10,
          detail=f"{acommA:.2e}")
    check(f"[LEG-B {label}] {{Q1_B,Q2_B}}=0 (anticommute, dense, <1e-10)", acommB < 1e-10,
          detail=f"{acommB:.2e}")
    comm_worst = max(
        float(np.max(np.abs(Q1_A @ Q1_B - Q1_B @ Q1_A))),
        float(np.max(np.abs(Q1_A @ Q2_B - Q2_B @ Q1_A))),
        float(np.max(np.abs(Q2_A @ Q1_B - Q1_B @ Q2_A))),
        float(np.max(np.abs(Q2_A @ Q2_B - Q2_B @ Q2_A))),
    )
    check(f"[LEG-B {label}] [Q_a^A,Q_b^B]=0 for all a,b in {{1,2}} (<1e-12, actual dense operators)",
          comm_worst < 1e-12, detail=f"worst={comm_worst:.2e}")

    idxA6, idxB6 = list(range(0, 6)), list(range(6, 12))
    T_wick = quartic_wick_corr(Gam, idxA6, idxB6)
    T_dense = np.array([[float(np.real(np.trace(rho @ Qa @ Qb))) for Qb in (Q1_B, Q2_B)]
                         for Qa in (Q1_A, Q2_A)])
    diff_wick = float(np.max(np.abs(T_wick.real - T_dense)))
    im_wick = float(np.max(np.abs(T_wick.imag)))
    check(f"[LEG-B {label}] wick_general == dense for the FOUR 8-point correlators <Qa^A Qb^B> "
          "(<1e-10, BEFORE the S_max read)", diff_wick < 1e-10 and im_wick < 1e-8,
          detail=f"max|wick-dense|={diff_wick:.2e} max|Im wick|={im_wick:.2e}")
    T = T_dense
    print(f"    [{label}] T (quartic pair) =\n{T}")
    plane_ok_A = Q1A_sq < 1e-10 and Q2A_sq < 1e-10 and acommA < 1e-10
    plane_ok_B = Q1B_sq < 1e-10 and Q2B_sq < 1e-10 and acommB < 1e-10
    check(f"[LEG-B {label}] the 2-plane requirements (A^2=I, {{A1,A2}}=0) hold for BOTH sides' "
          "quartics -- S_horodecki is licensed", plane_ok_A and plane_ok_B)
    S_an, sv = S_horodecki(T)
    S_grid, best_th, S_grid_phi_check = chsh_grid_refine(T)
    print(f"    [{label}] singular values of T = {sv}")
    agree = abs(S_an - S_grid) < 1e-6 and abs(S_an - S_grid_phi_check) < 1e-3
    check(f"[LEG-B {label}] closed-form vs grid+refine vs literal-phi-grid AGREE", agree,
          detail=f"analytic={S_an:.8f} grid={S_grid:.8f} phi-grid={S_grid_phi_check:.8f}")
    S_max = max(S_an, S_grid, S_grid_phi_check)
    verdict = qf2c_verdict(S_max)
    if verdict == "TSIRELSON-BREACH":
        tsirelson_breach = True
    definite = verdict in ("VIOLATION-FOUND", "NO-VIOLATION-IN-FAMILY", "TSIRELSON-BREACH")
    qf2c_all_definite = qf2c_all_definite and definite
    check(f"[LEG-B {label}] a DEFINITE verdict was reached", definite, detail=verdict)
    print(f"    >>> [LEG-B {label}] S_max = {S_max:.8f}  (classical 2, VIOLATION>=2.05, "
          f"Tsirelson 2sqrt2={2 * math.sqrt(2):.8f})   VERDICT: {verdict} <<<")
    return {"label": label, "sep": sep, "r_eff": r_eff, "S_max": S_max, "sv": sv, "verdict": verdict}


print("""  LEG B MANDATORY PRE-SWEEP VALIDATION (per pre-reg): a mini-instance with EXACTLY 3 SVD
  modes/side (6 Majoranas/side, 12 total, dense dim=2^6=64 -- tractable regardless of the
  original region's vertex count, since svd_smear ALWAYS reduces to r_eff total complex modes).
  The QF-2b F-1 CONTROL construction (chain_vacuum(400), sep=2, interval size 8/side -- reused
  VERBATIM, |A|=|B|=8>=3) serves as this mini-instance: Q1^2=Q2^2=I, {Q1,Q2}=0, [A,B]=0, and
  wick_general==dense for all four 8-point <Qa^A Qb^B> correlators are verified below, BEFORE any
  S_max is read anywhere in Leg B.""")

qf2c_legb_results = []
_ctr_b = L_chain // 2
_size_ctrl_b = 8
for sep_c in (2, 20):
    idxA_c = list(range(_ctr_b - sep_c - _size_ctrl_b, _ctr_b - sep_c))
    idxB_c = list(range(_ctr_b + sep_c, _ctr_b + sep_c + _size_ctrl_b))
    qf2c_legb_results.append(run_qf2c_legb_instance(
        f"CONTROL chain sep={sep_c} (reused QF-2b F-1)", C_chain, idxA_c, idxB_c, sep_c))

print("\n    LEG B MAIN SWEEP -- the pre-reg's named region set FAR+NEAR+BFS-ball (attempted "
      "VERBATIM; region size vs r=3 feasibility checked LIVE below, not assumed):")
_legb_regions = [("FAR", idxA_far2, idxB_far2, sep_far2),
                  ("NEAR", idxA_near2, idxB_near2, sep_near2),
                  ("BFS-ball(r=3)", idxA_bfs2, idxB_bfs2, sep_bfs2)]
for name, idxA_, idxB_, sep_ in _legb_regions:
    qf2c_legb_results.append(run_qf2c_legb_instance(name, C_full5, idxA_, idxB_, sep_))

print("\n    LEG B LADDER CHECK (sep in {1,3,9,27}, the SAME QF-2b F-4 constructions, reused "
      "verbatim) -- feasibility checked live; 'declare either way' per the pre-reg:")
for _target, _brsA, _brsB, _cellB in _ladder_defs:
    candA_L = [(b, _cellA_ladder) for b in _brsA]
    candB_L = [(b, _cellB) for b in _brsB]
    idxA_L, idxB_L, sep_L = patch_region_idx_sep(patch5, verts5, vpos5, candA_L, candB_L)
    qf2c_legb_results.append(run_qf2c_legb_instance(f"LADDER sep={_target}", C_full5, idxA_L,
                                                      idxB_L, sep_L))

banner("QF-2c LEG B  SUMMARY TABLE  (region-pair x r_eff x S_max x T singular values x verdict)")
print(f"  {'label':34s} {'r_eff':>5s} {'sep':>4s} {'S_max':>12s}  singular values of T       verdict")
for r_ in qf2c_legb_results:
    smax_str = f"{r_['S_max']:.6f}" if not math.isnan(r_['S_max']) else "n/a"
    print(f"  {r_['label']:34s} {r_['r_eff']:>5d} {r_['sep']:>4d} {smax_str:>12s}  "
          f"{np.array2string(r_['sv'], precision=6):24s}  {r_['verdict']}")

print("""
    LEG B DISCLOSED STRUCTURAL FINDING (machine-verified above, not assumed): the FAR/NEAR/ladder
    region constructions (verbatim from QF-2/QF-2b) use exactly 2 branches/side; r=3 needs >=3
    sites/side, so those regions structurally clamp at r_eff=2 and carry NO quartic-pair CHSH read
    (disclosed per-instance above). Only the chain CONTROL (interval size 8/side) and the
    BFS-ball (10 vertices/side) regions genuinely realize the 6-Majorana/side family.""")

# ===========================================================================
banner("QF-2c  TERMINALITY  (the pre-reg's pre-adjudicated ladder; whichever outcome obtains)")
# ===========================================================================
_lega_all_novint = all(r_["verdict"] == "NO-VIOLATION-IN-FAMILY" for r_ in qf2c_lega_results)
_legb_run = [r_ for r_ in qf2c_legb_results if r_["verdict"] != "STRUCTURALLY-EXCLUDED (r_eff<3)"]
_legb_all_novint = (len(_legb_run) > 0
                    and all(r_["verdict"] == "NO-VIOLATION-IN-FAMILY" for r_ in _legb_run))
_qf2c_run_novint = _lega_all_novint and _legb_all_novint
_viol = [r_["label"] for r_ in (qf2c_lega_results + qf2c_legb_results)
         if r_["verdict"] == "VIOLATION-FOUND"]

if tsirelson_breach:
    _qf2c_booked = ("A TSIRELSON-BREACH was detected in QF-2c -- see the crisis report below; "
                     "this is NOT the double-null branch.")
    _qf2c_terminality_label = "TSIRELSON-BREACH"
elif _qf2c_run_novint and qf2c_all_definite and ok_all:
    # Wording tightened at integration (2026-07-10, adversarial-check mandate; the tightened text
    # claims strictly LESS than the pre-reg's frozen version -- coverage-accurate scoping):
    _qf2c_booked = (
        "DOUBLE-NULL AGAIN (both legs, dense checks passing) => the object-level negative "
        "STRENGTHENS to: \"this Dirac-sea vacuum's accessible even-algebra correlations do not "
        "violate CHSH for rank-2-capable smeared bilinear families at any tested separation "
        "(1,3,9,10,27) -- a STRUCTURAL null: the four settings mutually commute, so this family "
        "cannot violate CHSH for ANY state -- nor for quartic families at r <= 3 at the one native "
        "graph geometry reached (BFS-ball, sep=10) plus two reused critical-chain control "
        "separations (sep=2,20); FAR/NEAR/ladder are structurally excluded at r=3 (2 sites/side).\" "
        "STILL not classicality; the remaining named doors are the two PHYSICS doors only (the "
        "derived J6-compatible patch omega; the finite-k/tick sector) -- instrument-side iteration "
        "beyond this point is DECLARED CLOSED (no QF-2d without a new physics object; this clause "
        "is binding).")
    _qf2c_terminality_label = "DOUBLE-NULL AGAIN (instrument-side iteration DECLARED CLOSED)"
elif _viol:
    _qf2c_booked = (f"ANY violation respecting Tsirelson => the graft's headline; booked with the "
                     f"family/region stated: VIOLATION-FOUND in {_viol}.")
    _qf2c_terminality_label = f"VIOLATION-FOUND {_viol}"
else:
    _qf2c_booked = ("NOT a clean double-null: at least one instrument/algebra check failed, or a "
                     "structural-exclusion left too few instances to call -- see the per-contract "
                     "detail above (a finding, not a bug); the double-null terminality sentence "
                     "does not apply as stated.")
    _qf2c_terminality_label = "NOT a clean double-null"
print(f"\n  >>> QF-2c TERMINALITY: {_qf2c_booked}")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
elapsed = time.time() - _T0
print(f"""  QF-0  anchors (tick-2pi, exact light cone) + A3 status line printed : {'PASS' if a_tick and _cone_worst == 0.0 else 'FAIL'}
  QF-1a Born-2 theorem-check (ratio=(a1/uc)^2, <1e-12)      : {'PASS' if abs(r_mean-born2)<1e-12 else 'FAIL'}
  QF-1b mechanism (Ramanujan-orth <1e-12; slope==2 <1e-9)   : {'PASS' if worst_overlap<1e-12 and abs(slope_ratio-2.0)<1e-9 else 'FAIL'}
  QF-1c falsification probe genuinely FAILS (blow-up >1e6x) : {'PASS' if falsify_ok else 'FAIL'}
  QF-2  FAR  regions  sep={res_far['sep']:<3d} S_max={res_far['S_max']:.6f}  verdict={res_far['verdict']}
  QF-2  NEAR regions  sep={res_near['sep']:<3d} S_max={res_near['S_max']:.6f}  verdict={res_near['verdict']}
  QF-3a GKLS gate (D1 CPTP; D2b Davies-scaled ccp+integrate): {'PASS' if (tp_ok and cmin>-1e-10 and ccp_val>-1e-8 and cpt_ok) else 'FAIL'}
  QF-3b pointer/record superselection (S1a/b/c)             : {'PASS' if (s1a_resid<1e-12 and worst_s1b<1e-12 and s1c_ok) else 'FAIL'}
  QF-3c thermalization to KMS at rate u_c (<0.02)           : {'PASS' if abs(rate-rate_pred)<0.02 else 'FAIL'}
  QF-4  scope declaration                                    : printed above (declaration only, not a gate)
  QF-2b F-0 flux adjudication (C_full exactly real)         : {'PASS' if _im_max == 0.0 else 'FAIL'}
  QF-2b F-1 control (chain_vacuum(400), sep={{2,20}}, r in {{1,2}}): {len(qf2b_results[:4])} instances, all definite = {all(r_['verdict'] in ('VIOLATION-FOUND','NO-VIOLATION-IN-FAMILY','TSIRELSON-BREACH') for r_ in qf2b_results[:4])}
  QF-2b F-2 FAR/NEAR/BFS-ball (r in {{1,2}})                  : {len(qf2b_results[4:])} instances, all definite = {all(r_['verdict'] in ('VIOLATION-FOUND','NO-VIOLATION-IN-FAMILY','TSIRELSON-BREACH') for r_ in qf2b_results[4:])}
  QF-2b F-3 quartic pair (algebraic-degeneracy CONFIRMED; informational <Gamma_A Gamma_B> sweep): {'PASS' if (constr1_confirms_degeneracy and constr2_confirms_centrality and _central_worst < 1e-10) else 'FAIL'}
  QF-2b F-4 separation ladder sep=[1,3,9,27] (r=2)          : {[r_['verdict'] for r_ in qf2b_ladder]}
  QF-2b F-5 booked sentence                                  : {'DOUBLE-NULL (object-level)' if (_all_qf2b_novint and _instrument_validates and not tsirelson_breach) else ('TSIRELSON-BREACH' if tsirelson_breach else 'NOT a clean double-null')}
  QF-2c LEG A natural-basis family (r=2; FAR/NEAR/BFS-ball/ladder): {len(qf2c_lega_results)} instances, all definite = {all(r_['verdict'] in ('VIOLATION-FOUND','NO-VIOLATION-IN-FAMILY','TSIRELSON-BREACH') for r_ in qf2c_lega_results)}
  QF-2c LEG B r=3 quartic pair (control+FAR/NEAR/BFS-ball/ladder) : {len(qf2c_legb_results)} instances attempted, {len(_legb_run)} genuinely r_eff=3 (rest structurally excluded, disclosed)
  QF-2c TERMINALITY                                          : {_qf2c_terminality_label}
  wall time: {elapsed:.1f}s""")

if tsirelson_breach:
    print("\n  *** TSIRELSON-BREACH DETECTED ***")
    print("  S_max exceeded 2*sqrt(2)+1e-9 in at least one declared region family. This would be")
    print("  a structural crisis for the net (non-quantum correlations) -- reported LOUDLY, NOT")
    print("  rationalized. STOPPING with a non-zero exit code per the frozen contract.")
    print("RESULT: TSIRELSON-BREACH -- see above")
    sys.exit(1)

# EXIT-CONDITION DEFINITION, EXTENDED ADDITIVELY (disclosed): exit 0 now requires -- (a) the
# original QF-0/1/3 contracts pass and QF-2 reached a definite verdict (unchanged, `ok_all` already
# accumulates this); AND (b) QF-2b's F-0 flux check passes, every F-1/F-2/F-4 instance reached a
# definite verdict (`qf2b_all_definite`), and F-3's algebraic-degeneracy predictions were correctly
# CONFIRMED (also folded into `ok_all` via the same check() calls above -- no separate gate
# variable needed). TSIRELSON-BREACH anywhere (original QF-2 OR any QF-2b instance/rung/quartic
# read) still triggers the SAME crisis branch above, verbatim, via the SAME `tsirelson_breach` flag.
qf2b_gate_ok = qf2b_all_definite and ok_all
# FURTHER EXTENDED, ADDITIVELY, for QF-2c (disclosed, same pattern): every LEG-A instance reached a
# definite verdict (`qf2c_all_definite`; STRUCTURALLY-EXCLUDED LEG-B instances are an honest
# not-applicable, not a failure, and do not gate this) -- also folded into `ok_all` via the same
# check() calls used throughout. TSIRELSON-BREACH anywhere in QF-2c triggers the SAME crisis branch
# above (already executed before this point via the shared `tsirelson_breach` flag).
qf2c_gate_ok = qf2c_all_definite and qf2b_gate_ok
print("RESULT:", "ALL QF-0,1,3 CONTRACTS PASS, QF-2 REACHED A DEFINITE VERDICT, QF-2b's F-0.."
      "F-4 AND QF-2c's LEG-A/LEG-B CONTRACTS ALL REACHED A DEFINITE VERDICT / CONFIRMED THEIR "
      "PREDICTIONS (NO-VIOLATION-IN-FAMILY and/or VIOLATION-FOUND throughout, never "
      "TSIRELSON-BREACH)"
      if (ok_all and qf2c_gate_ok) else "AT LEAST ONE CONTRACT FAILED -- see per-contract detail "
      "above (a finding, not a bug)")
sys.exit(0 if ok_all else 1)
